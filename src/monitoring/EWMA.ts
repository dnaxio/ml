import type { JsonRow, MonitorSpec } from "../@types/json";
import { extractSeries, mean, std, median, mad } from "./series";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of an EWMA model. */
export interface EWMAParams {
  /**
   * Smoothing factor λ ∈ (0, 1]. Default: 0.25. Small λ = smooth series,
   * sensitive to small sustained shifts; λ = 1 reduces to raw values.
   */
  lambda?: number;
  /**
   * Control-limit width L (in σ units). Default: 3. An alert fires when the
   * smoothed value z_n leaves μ0 ± L·σ_z(n). Lower L = earlier but noisier.
   */
  limit?: number;
  /**
   * Reference mean μ0. Estimated from the training data when omitted.
   */
  target?: number;
  /** Standard deviation σ of the series. Estimated from the data when omitted. */
  std?: number;
  /** Field name filled by fill_predict. Default: 'alert'. */
  alertField?: string;
  /**
   * Estimate μ0/σ with robust statistics (median + MAD) instead of
   * mean + std, so outliers in the baseline barely affect the limits.
   * Default: false.
   */
  robust?: boolean;
}

/**
 * EWMA (Exponentially Weighted Moving Average): statistical process control
 * that smooths a series with a decaying memory and raises an alert when the
 * smoothed value drifts beyond time-varying control limits.
 *
 * z_n = λ·x_n + (1 − λ)·z_{n−1},  with z_0 = μ0, and
 * limits = μ0 ± L·σ·√(λ/(2−λ)·(1−(1−λ)^(2n))).
 *
 * Good at detecting emerging trends while staying insensitive to isolated
 * spikes (they are smoothed out).
 */
class EWMA {
  private mu0 = 0;
  private sigma = 1;
  private lambdaValue = 0.25;
  private limitValue = 3;
  private alertField = "alert";
  private robust = false;
  private targetProvided?: number;
  private stdProvided?: number;
  private spec: MonitorSpec | null = null;
  private fitted = false;
  /** Online (update) state — smoothed value and point counter since fit/reset. */
  private onlineZ: number | undefined;
  private onlineN = 0;

  /**
   * Creates a model. Optional params override the statistics otherwise
   * estimated from the training data in `fit`.
   * @param params - Model configuration (lambda, limit, target, std, ...).
   */
  constructor(params?: EWMAParams) {
    if (params?.lambda !== undefined) this.lambdaValue = params.lambda;
    if (params?.limit !== undefined) this.limitValue = params.limit;
    if (params?.target !== undefined) {
      this.targetProvided = params.target;
      this.mu0 = params.target;
    }
    if (params?.std !== undefined) {
      this.stdProvided = params.std;
      this.sigma = params.std;
    }
    if (params?.alertField !== undefined) this.alertField = params.alertField;
    if (params?.robust !== undefined) this.robust = params.robust;
    // Stats complete → the model can predict without fit (used by ParallelMonitor.load).
    if (this.targetProvided !== undefined && this.stdProvided !== undefined) {
      this.fitted = true;
    }
  }

  /**
   * Learns the reference statistics (μ0, σ) from the training rows.
   * @param data - Row objects of the baseline period, e.g. [{ time, value }].
   * @param spec - Specification of the field to monitor.
   */
  fit(data: JsonRow[], spec: MonitorSpec): void {
    const values = extractSeries(data, spec);
    if (values.length === 0) {
      throw new Error("EWMA.fit requires at least one row.");
    }
    this.spec = spec;
    this.mu0 =
      this.targetProvided ?? (this.robust ? median(values) : mean(values));
    this.sigma = this.stdProvided ?? (this.robust ? mad(values) : std(values));
    this.onlineZ = undefined;
    this.onlineN = 0;
    this.fitted = true;
  }

  /**
   * Computes the smoothed value z_n per point.
   * @param data - Row objects with the same monitored field as at fit time.
   * @returns z_n per row.
   */
  scores(data: JsonRow[]): number[] {
    this.requireFitted();
    const values = extractSeries(data, this.spec!);
    const out: number[] = [];
    let z = this.mu0;
    for (const x of values) {
      z = this.lambdaValue * x + (1 - this.lambdaValue) * z;
      out.push(z);
    }
    return out;
  }

  /** Standard error of the smoothed statistic at point n (0-based). */
  private se(n: number): number {
    // t = n + 1 in the textbook formula (limits start finite).
    return (
      this.sigma *
      Math.sqrt(
        (this.lambdaValue / (2 - this.lambdaValue)) *
          (1 - Math.pow(1 - this.lambdaValue, 2 * (n + 1))),
      )
    );
  }

  /**
   * Predicts alerts on JSON rows: an alert fires when the smoothed value
   * leaves the time-varying control band μ0 ± L·σ_z(n).
   * @param data - Row objects with the same monitored field as at fit time.
   * @returns 1 = alert, 0 = normal, one per row.
   */
  predict(data: JsonRow[]): number[] {
    this.requireFitted();
    const values = extractSeries(data, this.spec!);
    const out: number[] = [];
    let z = this.mu0;
    for (let n = 0; n < values.length; n++) {
      const x = values[n]!;
      z = this.lambdaValue * x + (1 - this.lambdaValue) * z;
      const ucl = this.mu0 + this.limitValue * this.se(n);
      const lcl = this.mu0 - this.limitValue * this.se(n);
      out.push(z > ucl || z < lcl ? 1 : 0);
    }
    return out;
  }

  /**
   * Time-varying control limits per point (μ0 ± L·σ_z(n)). Useful for
   * plotting the chart or building a heatmap of the monitored band.
   * @param data - Row objects with the same monitored field as at fit time.
   * @returns Upper and lower control limits, one per row.
   */
  limits(data: JsonRow[]): { ucl: number[]; lcl: number[] } {
    this.requireFitted();
    const values = extractSeries(data, this.spec!);
    const ucl: number[] = [];
    const lcl: number[] = [];
    for (let n = 0; n < values.length; n++) {
      const w = this.limitValue * this.se(n);
      ucl.push(this.mu0 + w);
      lcl.push(this.mu0 - w);
    }
    return { ucl, lcl };
  }

  /**
   * Online update: consumes one row and advances the smoothed value
   * (z_n from z_{n-1}), so monitoring can run point-by-point in real time.
   * Starting from a fresh fit, calling `update` on a series produces the
   * same alerts as `predict(data)` on that series.
   * @param row - One row with the monitored field.
   * @returns The smoothed value and whether it leaves the control band.
   */
  update(row: JsonRow): { alert: boolean; score: number } {
    this.requireFitted();
    const x = extractSeries([row], this.spec!)[0]!;
    this.onlineZ =
      this.onlineZ === undefined
        ? this.lambdaValue * x + (1 - this.lambdaValue) * this.mu0
        : this.lambdaValue * x + (1 - this.lambdaValue) * this.onlineZ;
    const w = this.limitValue * this.se(this.onlineN);
    this.onlineN++;
    const ucl = this.mu0 + w;
    const lcl = this.mu0 - w;
    return {
      alert: this.onlineZ > ucl || this.onlineZ < lcl,
      score: this.onlineZ,
    };
  }

  /** Resets the online (update) state — useful to restart monitoring. */
  reset(): this {
    this.onlineZ = undefined;
    this.onlineN = 0;
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

  /** Smoothing factor λ. */
  get lambda(): number {
    return this.lambdaValue;
  }

  /** Control-limit width L (in σ units). */
  get limit(): number {
    return this.limitValue;
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
      lambda: this.lambda,
      limit: this.limit,
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
      lambda: number;
      limit: number;
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
    this.lambdaValue = payload.lambda;
    this.limitValue = payload.limit;
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

export { EWMA };
export type { MonitorSpec, JsonRow } from "../@types/json";
