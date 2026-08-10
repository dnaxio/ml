import type { JsonRow, MonitorSpec } from "../types/json";
import { extractSeries, mean, std, median, mad } from "./series";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a CUSUM model. */
export interface CUSUMParams {
  /**
   * Reference mean μ0. Estimated from the training data when omitted.
   * The CUSUM accumulates deviations from this target.
   */
  target?: number;
  /** Standard deviation σ of the series. Estimated from the data when omitted. */
  std?: number;
  /**
   * Allowable slack k (default: 0.5·σ). Shifts smaller than k per step are
   * treated as background noise and are not accumulated.
   */
  k?: number;
  /**
   * Alert threshold h (default: 5·σ). An alert fires when the cumulative
   * statistic S_n exceeds h. Lower h = earlier but noisier alarms.
   */
  h?: number;
  /** Direction of the shift to detect. Default: 'increase'. */
  direction?: "increase" | "decrease" | "both";
  /** Field name filled by fill_predict. Default: 'alert'. */
  alertField?: string;
  /**
   * Estimate μ0/σ with robust statistics (median + MAD) instead of
   * mean + std, so outliers in the baseline barely affect the thresholds.
   * Default: false.
   */
  robust?: boolean;
}

/**
 * CUSUM (Cumulative SUM): statistical process control for detecting small,
 * sustained shifts in a series — e.g. the onset of a sales peak.
 *
 * Upper CUSUM (detect an increase): S_n = max(0, S_{n-1} + (x_n − μ0 − k)),
 * with S_0 = 0. An alert fires as soon as S_n > h. Unlike a threshold on raw
 * values, the accumulation reacts to the *start* of a drift, not its peak.
 */
class CUSUM {
  private mu0 = 0;
  private sigma = 1;
  private k?: number;
  private h?: number;
  private shiftDirection: "increase" | "decrease" | "both" = "increase";
  private alertField = "alert";
  private robust = false;
  private targetProvided?: number;
  private stdProvided?: number;
  private spec: MonitorSpec | null = null;
  private fitted = false;
  /** Online (update) state — accumulated since fit or reset. */
  private onlineState = 0;
  private onlineStateDown = 0;

  /**
   * Creates a model. Optional params override the statistics otherwise
   * estimated from the training data in `fit`.
   * @param params - Model configuration (target, std, k, h, ...).
   */
  constructor(params?: CUSUMParams) {
    if (params?.target !== undefined) {
      this.targetProvided = params.target;
      this.mu0 = params.target;
    }
    if (params?.std !== undefined) {
      this.stdProvided = params.std;
      this.sigma = params.std;
    }
    if (params?.k !== undefined) this.k = params.k;
    if (params?.h !== undefined) this.h = params.h;
    if (params?.direction !== undefined) this.shiftDirection = params.direction;
    if (params?.alertField !== undefined) this.alertField = params.alertField;
    if (params?.robust !== undefined) this.robust = params.robust;
    // Stats complete → the model can predict without fit (used by ParallelMonitor.load).
    if (this.targetProvided !== undefined && this.stdProvided !== undefined) {
      this.fitted = true;
    }
  }

  /**
   * Learns the reference statistics (μ0, σ) from the training rows, then
   * resolves k = 0.5·σ and h = 5·σ when they were not provided.
   * @param data - Row objects of the baseline period, e.g. [{ time, value }].
   * @param spec - Specification of the field to monitor.
   */
  fit(data: JsonRow[], spec: MonitorSpec): void {
    const values = extractSeries(data, spec);
    if (values.length === 0) {
      throw new Error("CUSUM.fit requires at least one row.");
    }
    this.spec = spec;
    this.mu0 =
      this.targetProvided ?? (this.robust ? median(values) : mean(values));
    this.sigma = this.stdProvided ?? (this.robust ? mad(values) : std(values));
    this.k = this.k ?? 0.5 * this.sigma;
    this.h = this.h ?? 5 * this.sigma;
    this.onlineState = 0;
    this.onlineStateDown = 0;
    this.fitted = true;
  }

  /**
   * Computes the cumulative statistic S_n per point. With
   * `direction: "both"`, the maximum of the increase and decrease stats.
   * @param data - Row objects with the same monitored field as at fit time.
   * @returns S_n per row (monotone while a shift persists, resets via max(0,·)).
   */
  scores(data: JsonRow[]): number[] {
    this.requireFitted();
    const values = extractSeries(data, this.spec!);
    const out: number[] = [];
    const k = this.k ?? 0.5 * this.sigma;
    let S = 0;
    let Sd = 0;
    for (const x of values) {
      if (this.shiftDirection === "both") {
        S = Math.max(0, S + (x - this.mu0 - k));
        Sd = Math.max(0, Sd + (this.mu0 - k - x));
        out.push(Math.max(S, Sd));
      } else {
        const inc =
          this.shiftDirection === "increase"
            ? x - this.mu0 - k
            : this.mu0 - k - x;
        S = Math.max(0, S + inc);
        out.push(S);
      }
    }
    return out;
  }

  /**
   * Predicts alerts on JSON rows.
   * @param data - Row objects with the same monitored field as at fit time.
   * @returns 1 = alert (S_n > h), 0 = normal, one per row.
   */
  predict(data: JsonRow[]): number[] {
    const h = this.h ?? 5 * this.sigma;
    return this.scores(data).map((S) => (S > h ? 1 : 0));
  }

  /**
   * Online update: consumes one row and advances the cumulative statistic
   * (S_n from S_{n-1}), so monitoring can run point-by-point in real time
   * without re-processing the history. The state is shared with `predict`:
   * starting from a fresh fit, calling `update` on a series produces the
   * same scores as `scores(data)` on that series.
   * @param row - One row with the monitored field.
   * @returns The current score (S_n) and whether it exceeds the threshold.
   */
  update(row: JsonRow): { alert: boolean; score: number } {
    this.requireFitted();
    const x = extractSeries([row], this.spec!)[0]!;
    const k = this.k ?? 0.5 * this.sigma;
    const h = this.h ?? 5 * this.sigma;
    if (this.shiftDirection === "both") {
      this.onlineState = Math.max(0, this.onlineState + (x - this.mu0 - k));
      this.onlineStateDown = Math.max(
        0,
        this.onlineStateDown + (this.mu0 - k - x),
      );
      const score = Math.max(this.onlineState, this.onlineStateDown);
      return { alert: score > h, score };
    }
    const inc =
      this.shiftDirection === "increase"
        ? x - this.mu0 - k
        : this.mu0 - k - x;
    this.onlineState = Math.max(0, this.onlineState + inc);
    return { alert: this.onlineState > h, score: this.onlineState };
  }

  /** Resets the online (update) state — useful to restart monitoring. */
  reset(): this {
    this.onlineState = 0;
    this.onlineStateDown = 0;
    return this;
  }

  /**
   * Predicts and returns the input rows with the alert field filled
   * (new objects, the input is not mutated).
   * @param data - Row objects with the same monitored field as at fit time.
   * @returns The input rows with `alert: true | false` added.
   */
  fill_predict(data: JsonRow[]): JsonRow[] {
    const alerts = this.predict(data);
    return data.map((row, i) => ({
      ...row,
      [this.alertField]: alerts[i] === 1,
    }));
  }

  /** Reference mean μ0 (user-provided or estimated at fit). */
  get target(): number {
    return this.mu0;
  }

  /** Standard deviation σ of the series. */
  get std(): number {
    return this.sigma;
  }

  /** Resolved allowable slack k. */
  get slack(): number {
    return this.k ?? 0.5 * this.sigma;
  }

  /** Resolved alert threshold h. */
  get threshold(): number {
    return this.h ?? 5 * this.sigma;
  }

  /**
   * Estimated onset of the drift per point (index in `data`), or -1 where no
   * drift is accumulating (S = 0). For a one-sided CUSUM this is the point
   * right after the statistic last reset to 0 — the beginning of the current
   * excursion, i.e. the likely start of the shift.
   * @param data - Row objects with the same monitored field as at fit time.
   * @returns Estimated change-point index per row (-1 = no drift in progress).
   */
  changePoint(data: JsonRow[]): number[] {
    this.requireFitted();
    const values = extractSeries(data, this.spec!);
    const out: number[] = [];
    const k = this.k ?? 0.5 * this.sigma;
    let S = 0;
    let Sd = 0;
    let reset = 0;
    let dReset = 0;
    for (let n = 0; n < values.length; n++) {
      const x = values[n]!;
      const sInc = Math.max(0, S + (x - this.mu0 - k));
      const sDec = Math.max(0, Sd + (this.mu0 - k - x));
      if (this.shiftDirection === "both") {
        if (sInc >= sDec) {
          if (sInc === 0) reset = n + 1;
          out.push(sInc === 0 ? -1 : reset);
        } else {
          if (sDec === 0) dReset = n + 1;
          out.push(sDec === 0 ? -1 : dReset);
        }
        S = sInc;
        Sd = sDec;
      } else {
        const inc =
          this.shiftDirection === "increase"
            ? x - this.mu0 - k
            : this.mu0 - k - x;
        S = Math.max(0, S + inc);
        if (S === 0) reset = n + 1;
        out.push(S === 0 ? -1 : reset);
      }
    }
    return out;
  }

  /** Direction of the shift to detect ('increase' | 'decrease' | 'both'). */
  get direction(): "increase" | "decrease" | "both" {
    return this.shiftDirection;
  }

  /**
   * Attaches the monitored field spec without re-fitting (used internally by
   * ParallelMonitor.load). Normal usage: `fit` sets it automatically.
   */
  setSpec(spec: MonitorSpec): this {
    this.spec = spec;
    return this;
  }

  /**
   * Exports the fitted model (statistics + spec) to a `<name>.json` file
   * (async).
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    this.requireFitted();
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      mu0: this.mu0,
      sigma: this.sigma,
      k: this.k,
      h: this.h,
      direction: this.direction,
      robust: this.robust,
      alertField: this.alertField,
      spec: this.spec,
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
      mu0: number;
      sigma: number;
      k?: number;
      h?: number;
      direction: "increase" | "decrease" | "both";
      robust?: boolean;
      alertField: string;
      spec: MonitorSpec;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.mu0 = payload.mu0;
    this.sigma = payload.sigma;
    this.k = payload.k;
    this.h = payload.h;
    this.shiftDirection = payload.direction;
    this.robust = payload.robust ?? false;
    this.alertField = payload.alertField;
    this.spec = payload.spec;
    this.fitted = true;
  }

  private requireFitted(): void {
    if (!this.fitted || !this.spec) {
      throw new Error("Call fit before predict.");
    }
  }
}

export { CUSUM };
export type { MonitorSpec, JsonRow } from "../types/json";
