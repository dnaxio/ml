import type { JsonRow, SeasonalSpec } from "../types/json";
import { CUSUM } from "./CUSUM";
import type { CUSUMParams } from "./CUSUM";
import { EWMA } from "./EWMA";
import type { EWMAParams } from "./EWMA";
import { extractSeries, dayOfWeek } from "./series";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Internal field name carrying the deseasonalized residual. */
const RESIDUAL_FIELD = "__seasonal_residual";

/** Parameters of a SeasonalMonitor model. */
export interface SeasonalMonitorParams {
  /** Deseasonalized chart: 'cusum' (default) or 'ewma'. */
  model?: "cusum" | "ewma";
  /** Field name filled by fill_predict. Default: 'alert'. */
  alertField?: string;
  /** CUSUM/EWMA params applied to the residuals (k, h, lambda, limit, ...). */
  chart?: CUSUMParams | EWMAParams;
}

/**
 * SeasonalMonitor: weekly deseasonalized monitoring. Fits a reference mean
 * **per day of week** (μ0 for Monday, Tuesday, ...) from the baseline, then
 * monitors the deseasonalized residuals `x_n − μ0_{dayOfWeek(n)}` with a
 * CUSUM (default) or EWMA chart.
 *
 * This removes the weekly cycle (weekend dips, Monday peaks) from the noise,
 * so a constant-mean chart no longer inflates σ (and delays alarms) because
 * of it — a genuine epidemic shift is detected earlier.
 *
 * Needs at least one full week of baseline (every day-of-week present).
 */
class SeasonalMonitor {
  private model: "cusum" | "ewma" = "cusum";
  private alertField = "alert";
  private chartParams?: CUSUMParams | EWMAParams;
  private chart: CUSUM | EWMA | null = null;
  private spec: SeasonalSpec | null = null;
  /** Reference mean per day of week (0 = Sunday … 6 = Saturday). */
  private dowMeans = new Array<number>(7).fill(0);
  private fitted = false;

  /**
   * Creates a model.
   * @param params - Model configuration (model, alertField, chart params).
   */
  constructor(params?: SeasonalMonitorParams) {
    if (params?.model !== undefined) this.model = params.model;
    if (params?.alertField !== undefined) this.alertField = params.alertField;
    if (params?.chart !== undefined) this.chartParams = params.chart;
  }

  /**
   * Learns the per-day-of-week means and fits the chart on the residuals.
   * @param data - Row objects of the baseline (at least one full week), with a date field.
   * @param spec - Specification of the value + date fields.
   */
  fit(data: JsonRow[], spec: SeasonalSpec): void {
    if (data.length === 0) {
      throw new Error("SeasonalMonitor.fit requires at least one row.");
    }
    const values = extractSeries(data, { field: spec.field, missing: spec.missing });

    const sums = new Array<number>(7).fill(0);
    const counts = new Array<number>(7).fill(0);
    for (const [i, row] of data.entries()) {
      const dow = dayOfWeek(row[spec.dateField]);
      sums[dow]! += values[i]!;
      counts[dow]! += 1;
    }
    for (let d = 0; d < 7; d++) {
      if (counts[d] === 0) {
        throw new Error(
          `No baseline rows for day-of-week ${d} — SeasonalMonitor needs at least one full week.`,
        );
      }
      this.dowMeans[d] = sums[d]! / counts[d]!;
    }

    this.chart =
      this.model === "ewma"
        ? new EWMA(this.chartParams as EWMAParams)
        : new CUSUM(this.chartParams as CUSUMParams);
    this.chart.fit(this.residualRows(data, values, spec), {
      field: RESIDUAL_FIELD,
    });

    this.spec = spec;
    this.fitted = true;
  }

  /**
   * Predicts alerts on JSON rows (1 = the deseasonalized chart fires).
   * @param data - Row objects with the same value + date fields as at fit time.
   * @returns 1 = alert, 0 = normal, one per row.
   */
  predict(data: JsonRow[]): number[] {
    this.requireFitted();
    return this.chart!.predict(this.residualRows(data, extractSeries(data, this.spec!), this.spec!));
  }

  /**
   * Raw monitoring statistics per point: the cumulative statistic (CUSUM S_n)
   * or the smoothed value (EWMA z_n) of the deseasonalized residuals.
   * @param data - Row objects with the same value + date fields as at fit time.
   */
  scores(data: JsonRow[]): number[] {
    this.requireFitted();
    return this.chart!.scores(this.residualRows(data, extractSeries(data, this.spec!), this.spec!));
  }

  /**
   * Estimated onset of the drift per point (index in `data`), or -1.
   * CUSUM only (the EWMA chart has no change-point estimator).
   * @param data - Row objects with the same value + date fields as at fit time.
   */
  changePoint(data: JsonRow[]): number[] {
    this.requireFitted();
    if (this.model !== "cusum") {
      throw new Error("changePoint is only available with model: 'cusum'.");
    }
    return (this.chart as CUSUM).changePoint(
      this.residualRows(data, extractSeries(data, this.spec!), this.spec!),
    );
  }

  /**
   * Time-varying control limits per point. EWMA only (CUSUM has a fixed
   * threshold). Useful for plotting the deseasonalized chart.
   * @param data - Row objects with the same value + date fields as at fit time.
   */
  limits(data: JsonRow[]): { ucl: number[]; lcl: number[] } {
    this.requireFitted();
    if (this.model !== "ewma") {
      throw new Error("limits is only available with model: 'ewma'.");
    }
    return (this.chart as EWMA).limits(
      this.residualRows(data, extractSeries(data, this.spec!), this.spec!),
    );
  }

  /**
   * Predicts and returns the input rows with the alert field filled
   * (new objects, the input is not mutated).
   * @param data - Row objects with the same value + date fields as at fit time.
   * @returns The input rows + `alert: boolean`.
   */
  fill_predict(data: JsonRow[]): JsonRow[] {
    this.requireFitted();
    const alerts = this.predict(data);
    return data.map((row, i) => ({
      ...row,
      [this.alertField]: alerts[i] === 1,
    }));
  }

  /**
   * Reference mean per day of week (the seasonal profile learned at fit).
   * Day 0 = Sunday … day 6 = Saturday. Useful as a diagnostic: a max/min
   * ratio above ~1.3 confirms a weekly cycle worth deseasonalizing.
   */
  get dayProfile(): Array<{ day: string; mean: number }> {
    const names = [
      "Sunday",
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
    ];
    return this.dowMeans.map((m, d) => ({ day: names[d]!, mean: m }));
  }

  /**
   * Exports the fitted model (seasonal profile + chart state + spec) to a
   * `<name>.json` file (async).
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    this.requireFitted();
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      model: this.model,
      alertField: this.alertField,
      chartParams: this.chartParams ?? null,
      spec: this.spec,
      dowMeans: this.dowMeans,
      chart:
        this.model === "ewma"
          ? {
              target: (this.chart as EWMA).target,
              std: (this.chart as EWMA).std,
              lambda: (this.chart as EWMA).lambda,
              limit: (this.chart as EWMA).limit,
            }
          : {
              target: (this.chart as CUSUM).target,
              std: (this.chart as CUSUM).std,
              k: (this.chart as CUSUM).slack,
              h: (this.chart as CUSUM).threshold,
              direction: (this.chart as CUSUM).direction,
            },
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
      model: "cusum" | "ewma";
      alertField: string;
      chartParams?: CUSUMParams | EWMAParams | null;
      spec: SeasonalSpec;
      dowMeans: number[];
      chart: {
        target: number;
        std: number;
        k?: number;
        h?: number;
        direction?: "increase" | "decrease" | "both";
        lambda?: number;
        limit?: number;
      };
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = payload.model;
    this.alertField = payload.alertField;
    this.chartParams = payload.chartParams ?? undefined;
    this.spec = payload.spec;
    this.dowMeans = payload.dowMeans;
    this.chart =
      this.model === "ewma"
        ? new EWMA({
            target: payload.chart.target,
            std: payload.chart.std,
            lambda: payload.chart.lambda,
            limit: payload.chart.limit,
          }).setSpec({ field: RESIDUAL_FIELD })
        : new CUSUM({
            target: payload.chart.target,
            std: payload.chart.std,
            k: payload.chart.k,
            h: payload.chart.h,
            direction: payload.chart.direction,
          }).setSpec({ field: RESIDUAL_FIELD });
    this.fitted = true;
  }

  /** Builds deseasonalized residual rows for the chart. */
  private residualRows(
    data: JsonRow[],
    values: number[],
    spec: SeasonalSpec,
  ): JsonRow[] {
    return data.map((row, i) => ({
      [RESIDUAL_FIELD]:
        values[i]! - this.dowMeans[dayOfWeek(row[spec.dateField])]!,
    }));
  }

  /**
   * Online update: consumes one row, computes the deseasonalized residual
   * (value − day-of-week mean) and advances the internal chart point by
   * point.
   * @param row - One row with the value + date fields.
   * @returns The chart score and whether it alerts.
   */
  update(row: JsonRow): { alert: boolean; score: number } {
    this.requireFitted();
    const value = extractSeries([row], {
      field: this.spec!.field,
      missing: this.spec!.missing,
    })[0]!;
    const residual =
      value - this.dowMeans[dayOfWeek(row[this.spec!.dateField])]!;
    return this.chart!.update({ [RESIDUAL_FIELD]: residual });
  }

  private requireFitted(): void {
    if (!this.fitted) {
      throw new Error("Call fit before predict.");
    }
  }
}

export { SeasonalMonitor };
export type { SeasonalSpec, JsonRow } from "../types/json";
