import type { JsonRow, ScanSpec, ScanCluster } from "../types/json";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a SpatialScan model. */
export interface SpatialScanParams {
  /**
   * Monte-Carlo replications for the p-value (SaTScan recommends 999).
   * Default: 199 (fast). Use 999 for publication-grade results.
   */
  replications?: number;
  /** Significance level: clusters with p ≤ value are reported. Default: 0.05. */
  significance?: number;
  /** Max zones per window, as a fraction of all zones. Default: 0.5. */
  maxWindowFraction?: number;
  /** Seed for reproducible Monte-Carlo draws. */
  randomState?: number;
  /** Field name filled by fill_predict. Default: 'cluster'. */
  clusterField?: string;
}

/** A learned zone (geometry + population + baseline cases). */
interface Zone {
  id: string;
  x: number;
  y: number;
  population: number;
  baselineCases: number;
}

/**
 * SpatialScan: Kulldorff's spatial scan statistic (SaTScan). Detects
 * statistically significant spatial clusters — e.g. a localized disease
 * outbreak or a geographic concentration of unusual purchases.
 *
 * A circular window of every size is slid around every zone; each window is
 * scored with the Poisson log-likelihood ratio (cases vs population-expected),
 * and the most likely cluster is assessed with a Monte-Carlo p-value: random
 * datasets with the same total cases distributed by population are simulated,
 * and the p-value is the proportion of random max-LLRs reaching the observed
 * one.
 *
 * Workflow: `fit` on a baseline (learns zone geometry, populations and the
 * global case rate), then `cluster` / `predict` / `fill_predict` on the
 * current case counts.
 */
class SpatialScan {
  private zones: Zone[] = [];
  /** Zone index per id (for lookup at predict time). */
  private zoneIndex = new Map<string, number>();
  /** Zone order per center, sorted by distance (precomputed at fit). */
  private orders: number[][] = [];
  /** Global baseline case rate per unit population. */
  private globalRate = 0;
  private totalPopulation = 0;
  private spec: ScanSpec | null = null;
  private replications = 199;
  private significance = 0.05;
  private maxWindowFraction = 0.5;
  private randomState?: number;
  private clusterField = "cluster";
  private fitted = false;

  /**
   * Creates a model.
   * @param params - Model configuration (replications, significance, seed, ...).
   */
  constructor(params?: SpatialScanParams) {
    if (params?.replications !== undefined) {
      this.replications = params.replications;
    }
    if (params?.significance !== undefined) {
      this.significance = params.significance;
    }
    if (params?.maxWindowFraction !== undefined) {
      this.maxWindowFraction = params.maxWindowFraction;
    }
    if (params?.randomState !== undefined) {
      this.randomState = params.randomState;
    }
    if (params?.clusterField !== undefined) {
      this.clusterField = params.clusterField;
    }
  }

  /**
   * Learns the zone geometry, populations and the global case rate.
   * @param data - One row per zone: id, coordinates, population, baseline cases.
   * @param spec - Specification of the zone/coordinate/population/cases fields.
   */
  fit(data: JsonRow[], spec: ScanSpec): void {
    const zones: Zone[] = [];
    const seen = new Set<string>();
    for (const row of data) {
      const id = String(row[spec.zone]);
      if (seen.has(id)) {
        throw new Error(`Duplicate zone id "${id}" at fit time.`);
      }
      seen.add(id);
      const x = Number(row[spec.coordinates[0]]);
      const y = Number(row[spec.coordinates[1]]);
      const population = Number(row[spec.population]);
      const baselineCases = Number(row[spec.cases]);
      if (
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        !Number.isFinite(population) ||
        !Number.isFinite(baselineCases)
      ) {
        throw new Error(`Invalid zone row: ${JSON.stringify(row)}`);
      }
      zones.push({ id, x, y, population, baselineCases });
    }
    if (zones.length === 0) {
      throw new Error("SpatialScan.fit requires at least one zone.");
    }
    const totalPopulation = zones.reduce((a, z) => a + z.population, 0);
    if (totalPopulation <= 0) {
      throw new Error("Total population must be > 0.");
    }
    const totalCases = zones.reduce((a, z) => a + z.baselineCases, 0);

    this.zones = zones;
    this.zoneIndex = new Map(zones.map((z, i) => [z.id, i]));
    this.globalRate = totalCases / totalPopulation;
    this.totalPopulation = totalPopulation;
    this.rebuildOrders();
    this.spec = spec;
    this.fitted = true;
  }

  /**
   * Runs the scan on the current case counts and returns the most likely
   * cluster when it is significant (p ≤ significance), otherwise null.
   * @param data - One row per zone with the current case counts.
   * @returns The detected cluster, or null when none is significant.
   */
  cluster(data: JsonRow[]): ScanCluster | null {
    this.requireFitted();
    const cases = this.extractCases(data);
    const C = cases.reduce((a, b) => a + b, 0);
    if (C <= 0) return null;

    const expected = this.zones.map((z) => z.population * this.globalRate);
    const obs = this.mostLikelyWindow(cases, expected);
    if (obs.llr <= 0) return null;

    // Monte-Carlo p-value: simulate random datasets with the same total
    // cases distributed proportionally to population.
    const probs = this.zones.map((z) => z.population / this.totalPopulation);
    const rng = mulberry32(
      this.randomState ?? Math.floor(Math.random() * 2 ** 32),
    );
    let count = 0;
    for (let r = 0; r < this.replications; r++) {
      const sim = drawMultinomial(C, probs, rng);
      const m = this.mostLikelyWindow(sim, expected);
      if (m.llr >= obs.llr) count++;
    }
    const pValue = (count + 1) / (this.replications + 1);
    if (pValue > this.significance) return null;

    return {
      zones: obs.zones.map((zi) => this.zones[zi]!.id),
      cases: obs.cases,
      expected: obs.expected,
      llr: obs.llr,
      pValue,
    };
  }

  /**
   * Predicts cluster membership per row (1 = inside the detected cluster).
   * @param data - One row per zone with the current case counts.
   * @returns 1 per row whose zone is inside the significant cluster, else 0.
   */
  predict(data: JsonRow[]): number[] {
    const inCluster = new Set(this.cluster(data)?.zones ?? []);
    return data.map((row) =>
      inCluster.has(String(row[this.spec!.zone])) ? 1 : 0,
    );
  }

  /**
   * Predicts and returns the input rows with the cluster field filled
   * (new objects, the input is not mutated).
   * @param data - One row per zone with the current case counts.
   * @returns The input rows + `cluster: boolean`.
   */
  fill_predict(data: JsonRow[]): JsonRow[] {
    const inCluster = new Set(this.cluster(data)?.zones ?? []);
    return data.map((row) => ({
      ...row,
      [this.clusterField]: inCluster.has(String(row[this.spec!.zone])),
    }));
  }

  /** Learned zones (id, coordinates, population). */
  get zonesList(): Array<{ id: string; x: number; y: number; population: number }> {
    return this.zones.map((z) => ({
      id: z.id,
      x: z.x,
      y: z.y,
      population: z.population,
    }));
  }

  /** Global baseline case rate per unit population (from fit). */
  get expectedRate(): number {
    return this.globalRate;
  }

  /**
   * Exports the fitted model (zones + rate + spec + params) to a `<name>.json`
   * file (async).
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    this.requireFitted();
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      spec: this.spec,
      replications: this.replications,
      significance: this.significance,
      maxWindowFraction: this.maxWindowFraction,
      randomState: this.randomState ?? null,
      clusterField: this.clusterField,
      globalRate: this.globalRate,
      totalPopulation: this.totalPopulation,
      zones: this.zones,
    };
    await Bun.file(filePath).write(JSON.stringify(payload));
  }

  /**
   * Loads a model previously exported with `export()`.
   * @param name - File name (with or without .json extension).
   */
  async load(name: string): Promise<void> {
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = JSON.parse(await Bun.file(filePath).text()) as {
      version?: number;
      spec: ScanSpec;
      replications: number;
      significance: number;
      maxWindowFraction: number;
      randomState?: number | null;
      clusterField: string;
      globalRate: number;
      totalPopulation: number;
      zones: Zone[];
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.spec = payload.spec;
    this.replications = payload.replications;
    this.significance = payload.significance;
    this.maxWindowFraction = payload.maxWindowFraction;
    this.randomState = payload.randomState ?? undefined;
    this.clusterField = payload.clusterField;
    this.globalRate = payload.globalRate;
    this.totalPopulation = payload.totalPopulation;
    this.zones = payload.zones;
    this.zoneIndex = new Map(payload.zones.map((z, i) => [z.id, i]));
    this.rebuildOrders();
    this.fitted = true;
  }

  /** Reads the current case count per learned zone from `data`. */
  private extractCases(data: JsonRow[]): number[] {
    const byZone = new Map<string, number>();
    for (const row of data) {
      const id = String(row[this.spec!.zone]);
      if (!this.zoneIndex.has(id)) {
        throw new Error(`Zone "${id}" was not seen at fit time.`);
      }
      const v = Number(row[this.spec!.cases]);
      const val = Number.isFinite(v) ? v : 0;
      byZone.set(id, (byZone.get(id) ?? 0) + val);
    }
    return this.zones.map((z) => byZone.get(z.id) ?? 0);
  }

  /** Precomputes the per-center distance-sorted zone orders. */
  private rebuildOrders(): void {
    this.orders = this.zones.map((center, ci) => {
      const withDist = this.zones.map((z, zi) => ({
        zi,
        d: (z.x - center.x) ** 2 + (z.y - center.y) ** 2,
      }));
      withDist.sort((a, b) => a.d - b.d || a.zi - b.zi);
      return withDist.map((e) => e.zi);
    });
  }

  /** Finds the window with the maximum log-likelihood ratio. */
  private mostLikelyWindow(
    cases: number[],
    expected: number[],
  ): { llr: number; cases: number; expected: number; zones: number[] } {
    const C = cases.reduce((a, b) => a + b, 0);
    const K = Math.max(1, Math.ceil(this.zones.length * this.maxWindowFraction));
    let best: { llr: number; cases: number; expected: number; zones: number[] } =
      { llr: 0, cases: 0, expected: 0, zones: [] };

    for (const order of this.orders) {
      let cumCases = 0;
      let cumExp = 0;
      const limit = Math.min(K, order.length);
      for (let k = 0; k < limit; k++) {
        const zi = order[k]!;
        cumCases += cases[zi]!;
        cumExp += expected[zi]!;
        if (cumCases > cumExp && cumExp > 0) {
          const llr = this.llr(C, cumCases, cumExp);
          if (llr > best.llr) {
            best = {
              llr,
              cases: cumCases,
              expected: cumExp,
              zones: order.slice(0, k + 1),
            };
          }
        }
      }
    }
    return best;
  }

  /** Poisson log-likelihood ratio of a window (0·ln(0) = 0 convention). */
  private llr(C: number, c: number, e: number): number {
    const term1 = c > 0 && e > 0 ? c * Math.log(c / e) : 0;
    const b = C - c;
    const d = C - e;
    const term2 = b > 0 && d > 0 ? b * Math.log(b / d) : 0;
    return term1 + term2;
  }

  private requireFitted(): void {
    if (!this.fitted) {
      throw new Error("Call fit before cluster.");
    }
  }
}

/** Draws C cases into zones with the given probabilities (multinomial). */
function drawMultinomial(C: number, probs: number[], rng: () => number): number[] {
  const out = new Array<number>(probs.length).fill(0);
  const cum: number[] = [];
  let acc = 0;
  for (const p of probs) {
    acc += p;
    cum.push(acc);
  }
  for (let k = 0; k < C; k++) {
    const u = rng() * acc;
    let lo = 0;
    let hi = cum.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid]! >= u) hi = mid;
      else lo = mid + 1;
    }
    out[lo] = (out[lo] ?? 0) + 1;
  }
  return out;
}

/** mulberry32 — tiny seeded PRNG returning uniform values in [0, 1). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export { SpatialScan };
export type { ScanSpec, ScanCluster, JsonRow } from "../types/json";
