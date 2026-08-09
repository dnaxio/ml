import type { JsonRow, HotspotSpec } from "../@types/json";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a GetisOrd (Gi*) hotspot model. */
export interface GetisOrdParams {
  /**
   * Neighborhood radius (same unit as the coordinates). Every zone within
   * `distance` of zone i is a neighbor, including zone i itself (Gi*).
   * Default: the mean nearest-neighbor distance across zones (data-driven,
   * deterministic) — each zone then has ~1-2 neighbors besides itself.
   */
  distance?: number;
  /**
   * Significance level (two-sided). A zone is a hotspot when z > 0 and
   * p ≤ significance, a cold spot when z < 0 and p ≤ significance.
   * Default: 0.05.
   */
  significance?: number;
  /** Field name filled by fill_predict. Default: 'hot'. */
  hotField?: string;
}

/** A learned zone (geometry only — Gi* reads the case counts per analysis). */
interface Zone {
  id: string;
  x: number;
  y: number;
}

/** Per-zone result of a Gi* analysis. */
export interface HotspotResult {
  /** Zone id. */
  zone: string;
  /** Gi* z-score: positive = above-average neighborhood, negative = below. */
  zScore: number;
  /** Two-sided normal p-value (≤ significance = statistically significant). */
  pValue: number;
  /** True when zScore > 0 and pValue ≤ significance. */
  hot: boolean;
  /** True when zScore < 0 and pValue ≤ significance. */
  cold: boolean;
}

/**
 * Getis-Ord Gi*: local spatial autocorrelation statistic that flags each
 * zone as a significant hotspot or cold spot — no Monte-Carlo, O(n·k) per
 * zone. It answers "which zones are unusually hot?" (one result per zone),
 * complementing SpatialScan's "where is the single most likely cluster?".
 *
 * For zone i with neighbors j (within `distance`, including i itself):
 *   Gi* = [Σ w_ij·x_j − X̄·Σ w_ij] / [S·sqrt((n·Σ w_ij² − (Σ w_ij)²)/(n−1))]
 * with binary weights w_ij ∈ {0,1}, X̄ the global case mean and S the case
 * standard deviation. A positive z-score with p ≤ significance marks a
 * hotspot (e.g. a localized sales peak); a negative one marks a cold spot.
 *
 * Workflow: `fit` on the zones (learns geometry + default distance), then
 * `hotspots` / `predict` / `fill_predict` on the current case counts.
 */
class GetisOrd {
  private zones: Zone[] = [];
  /** Zone index per id (for lookup at analysis time). */
  private zoneIndex = new Map<string, number>();
  private spec: HotspotSpec | null = null;
  /** Resolved neighborhood radius (user-provided or mean NN distance). */
  private resolvedDistance = 0;
  private significanceLevel = 0.05;
  private hotField = "hot";
  private fitted = false;

  /**
   * Creates a model.
   * @param params - Model configuration (distance, significance, hotField).
   */
  constructor(params?: GetisOrdParams) {
    if (params?.distance !== undefined) this.resolvedDistance = params.distance;
    if (params?.significance !== undefined) {
      this.significanceLevel = params.significance;
    }
    if (params?.hotField !== undefined) this.hotField = params.hotField;
  }

  /**
   * Learns the zone geometry. When `distance` was not provided, it is
   * resolved as the mean nearest-neighbor distance across zones.
   * @param data - One row per zone: id + coordinates.
   * @param spec - Specification of the zone/coordinate fields.
   */
  fit(data: JsonRow[], spec: HotspotSpec): void {
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
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw new Error(`Invalid zone row: ${JSON.stringify(row)}`);
      }
      zones.push({ id, x, y });
    }
    if (zones.length < 2) {
      throw new Error("GetisOrd.fit requires at least two zones.");
    }
    this.zones = zones;
    this.zoneIndex = new Map(zones.map((z, i) => [z.id, i]));
    if (this.resolvedDistance <= 0) {
      this.resolvedDistance = meanNearestNeighborDistance(zones);
    }
    this.spec = spec;
    this.fitted = true;
  }

  /**
   * Computes the Gi* statistic for every zone on the current case counts.
   * @param data - One row per zone (same zones as fit) with the case counts.
   * @returns One result per row, in the input order.
   */
  hotspots(data: JsonRow[]): HotspotResult[] {
    this.requireFitted();
    const n = this.zones.length;
    const cases = this.extractCases(data);
    if (cases.length !== n) {
      throw new Error(
        `GetisOrd expects ${n} zones (one row per fit zone), got ${cases.length}.`,
      );
    }

    const mean = cases.reduce((a, b) => a + b, 0) / n;
    const s = Math.sqrt(
      cases.reduce((a, x) => a + x * x, 0) / n - mean * mean,
    );

    const results: HotspotResult[] = [];
    for (let i = 0; i < n; i++) {
      const nb = this.neighborsOf(i);
      const k = nb.length;
      const sumX = nb.reduce((a, j) => a + (cases[j] ?? 0), 0);
      let zScore = 0;
      let pValue = 1;
      if (s > 0 && k > 0) {
        // Binary weights: Σ w = k and Σ w² = k.
        const denom = s * Math.sqrt((n * k - k * k) / (n - 1));
        zScore = (sumX - mean * k) / denom;
        pValue = clamp01(2 * (1 - normCdf(Math.abs(zScore))));
      }
      results.push({
        zone: this.zones[i]!.id,
        zScore,
        pValue,
        hot: zScore > 0 && pValue <= this.significanceLevel,
        cold: zScore < 0 && pValue <= this.significanceLevel,
      });
    }
    return results;
  }

  /**
   * Predicts significant hotspots per row.
   * @param data - One row per zone with the current case counts.
   * @returns 1 = significant hotspot (z > 0, p ≤ significance), else 0.
   */
  predict(data: JsonRow[]): number[] {
    return this.hotspots(data).map((r) => (r.hot ? 1 : 0));
  }

  /**
   * Predicts and returns the input rows with the hotspot field filled
   * (new objects, the input is not mutated).
   * @param data - One row per zone with the current case counts.
   * @returns The input rows + `hot: boolean`.
   */
  fill_predict(data: JsonRow[]): JsonRow[] {
    const hot = this.hotspots(data).map((r) => r.hot);
    return data.map((row, i) => ({ ...row, [this.hotField]: hot[i] }));
  }

  /** Learned zone ids (same order as fit). */
  get zonesList(): string[] {
    return this.zones.map((z) => z.id);
  }

  /** Resolved neighborhood radius (user-provided or mean NN distance). */
  get distance(): number {
    return this.resolvedDistance;
  }

  /** Significance level used for hot/cold classification. */
  get significance(): number {
    return this.significanceLevel;
  }

  /**
   * Exports the fitted model (geometry + resolved distance + params) to a
   * `<name>.json` file (async).
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    this.requireFitted();
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      spec: this.spec,
      distance: this.resolvedDistance,
      significance: this.significanceLevel,
      hotField: this.hotField,
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
      spec: HotspotSpec;
      distance: number;
      significance: number;
      hotField: string;
      zones: Zone[];
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.zones = payload.zones;
    this.zoneIndex = new Map(this.zones.map((z, i) => [z.id, i]));
    this.spec = payload.spec;
    this.resolvedDistance = payload.distance;
    this.significanceLevel = payload.significance;
    this.hotField = payload.hotField;
    this.fitted = true;
  }

  /** Indices of the zones within `distance` of zone i, including i itself. */
  private neighborsOf(i: number): number[] {
    const out: number[] = [];
    const zi = this.zones[i]!;
    for (let j = 0; j < this.zones.length; j++) {
      const zj = this.zones[j]!;
      if (Math.hypot(zi.x - zj.x, zi.y - zj.y) <= this.resolvedDistance) {
        out.push(j);
      }
    }
    return out;
  }

  /** Case counts from the data, aligned with the fit zone order. */
  private extractCases(data: JsonRow[]): number[] {
    return data.map((row, idx) => {
      const zi = this.zoneIndex.get(String(row[this.spec!.zone]));
      if (zi === undefined) {
        throw new Error(
          `Unknown zone "${String(row[this.spec!.zone])}" at row ${idx} — must be one of the fit zones.`,
        );
      }
      const v = Number(row[this.spec!.cases]);
      if (!Number.isFinite(v)) {
        throw new Error(
          `Field "${this.spec!.cases}" must hold numbers (row ${idx}: ${JSON.stringify(row)}).`,
        );
      }
      return v;
    });
  }

  private requireFitted(): void {
    if (!this.fitted || !this.spec) {
      throw new Error("Call fit before hotspots.");
    }
  }
}

/** Mean distance from each zone to its nearest neighbor (deterministic). */
function meanNearestNeighborDistance(zones: Zone[]): number {
  let total = 0;
  for (let i = 0; i < zones.length; i++) {
    const zi = zones[i]!;
    let nearest = Infinity;
    for (let j = 0; j < zones.length; j++) {
      if (j === i) continue;
      const d = Math.hypot(zi.x - zones[j]!.x, zi.y - zones[j]!.y);
      if (d < nearest) nearest = d;
    }
    total += nearest;
  }
  return total / zones.length;
}

/** Standard normal CDF via the Abramowitz-Stegun erf approximation. */
function normCdf(z: number): number {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

/** Error function (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7). */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
      0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-ax * ax));
  return sign * y;
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

export { GetisOrd };
export type { HotspotSpec, JsonRow } from "../@types/json";
