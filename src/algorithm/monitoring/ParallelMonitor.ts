import type { JsonRow, ParallelSpec } from "../../@types/json";
import { CUSUM } from "./CUSUM";
import type { CUSUMParams } from "./CUSUM";
import { EWMA } from "./EWMA";
import type { EWMAParams } from "./EWMA";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a ParallelMonitor model. */
export interface ParallelMonitorParams {
  /** Univariate chart per series: 'cusum' (default) or 'ewma'. */
  model?: "cusum" | "ewma";
  /**
   * Family-wise false-alarm rate (Bonferroni across fields): each chart is
   * given the two-sided level `familyError / N`. When omitted, each chart
   * keeps its own conservative defaults (h = 5·σ, L = 3).
   */
  familyError?: number;
  /** Field name filled by fill_predict. Default: 'alert'. */
  alertField?: string;
  /** Field name listing the triggering fields. Default: 'alertFields'. */
  alertFieldsField?: string;
}

/** A fitted univariate chart bound to a field. */
interface Chart {
  field: string;
  monitor: CUSUM | EWMA;
}

/**
 * ParallelMonitor: monitors several series at once — one univariate chart
 * (CUSUM or EWMA) per field. An alert fires when at least one chart fires,
 * and `alertFields` reports *which* fields triggered (interpretable alarms).
 * Optionally applies a Bonferroni correction (`familyError`) so the
 * family-wise false-alarm rate is controlled instead of inflating with the
 * number of monitored fields (multiple-testing problem).
 */
class ParallelMonitor {
  private charts: Chart[] = [];
  private model: "cusum" | "ewma" = "cusum";
  private familyError?: number;
  private alertField = "alert";
  private alertFieldsField = "alertFields";
  private spec: ParallelSpec | null = null;
  private fitted = false;

  /**
   * Creates a model.
   * @param params - Model configuration (model, familyError, field names).
   */
  constructor(params?: ParallelMonitorParams) {
    if (params?.model !== undefined) this.model = params.model;
    if (params?.familyError !== undefined) {
      this.familyError = params.familyError;
    }
    if (params?.alertField !== undefined) this.alertField = params.alertField;
    if (params?.alertFieldsField !== undefined) {
      this.alertFieldsField = params.alertFieldsField;
    }
  }

  /**
   * Fits one chart per monitored field on the baseline period.
   * @param data - Row objects of the baseline period (all fields present).
   * @param spec - Specification of the fields to monitor (+ optional per-field params).
   */
  fit(data: JsonRow[], spec: ParallelSpec): void {
    const fields = spec.fields;
    if (fields.length === 0) {
      throw new Error("ParallelSpec.fields must contain at least one field.");
    }
    const N = fields.length;
    // Bonferroni: per-chart two-sided level = familyError / N.
    const level =
      this.familyError !== undefined ? this.familyError / N : undefined;
    const LAdj = level !== undefined ? normInv(1 - level / 2) : undefined;

    this.spec = spec;
    this.charts = fields.map((field) => {
      const p = spec.params?.[field];
      if (this.model === "ewma") {
        const base: EWMAParams = { ...(p as EWMAParams) };
        if (base.limit === undefined && LAdj !== undefined) {
          base.limit = LAdj;
        }
        const monitor = new EWMA(base);
        monitor.fit(data, { field, missing: spec.missing });
        return { field, monitor };
      }
      const base: CUSUMParams = { ...(p as CUSUMParams) };
      // h and the EWMA limit L are both in σ units: scale h like L.
      if (base.h === undefined && LAdj !== undefined) {
        base.h = 5 * (LAdj / 3);
      }
      const monitor = new CUSUM(base);
      monitor.fit(data, { field, missing: spec.missing });
      return { field, monitor };
    });
    this.fitted = true;
  }

  /**
   * Predicts alerts on JSON rows: 1 when at least one chart fires.
   * @param data - Row objects with the same monitored fields as at fit time.
   * @returns 1 = alert, 0 = normal, one per row.
   */
  predict(data: JsonRow[]): number[] {
    this.requireFitted();
    const n = data.length;
    const out = new Array<number>(n).fill(0);
    for (const { monitor } of this.charts) {
      const alerts = monitor.predict(data);
      for (let i = 0; i < n; i++) {
        if (alerts[i] === 1) out[i] = 1;
      }
    }
    return out;
  }

  /**
   * Which fields triggered, per row (empty array = no alert).
   * @returns One `string[]` per row, listing the fields in alert.
   */
  alertFields(data: JsonRow[]): string[][] {
    this.requireFitted();
    const n = data.length;
    const out: string[][] = Array.from({ length: n }, () => []);
    for (const { field, monitor } of this.charts) {
      const alerts = monitor.predict(data);
      for (let i = 0; i < n; i++) {
        if (alerts[i] === 1) out[i]!.push(field);
      }
    }
    return out;
  }

  /**
   * Raw monitoring statistics per field (S_n for CUSUM, z_n for EWMA).
   * @returns One array per monitored field.
   */
  scores(data: JsonRow[]): Record<string, number[]> {
    this.requireFitted();
    const out: Record<string, number[]> = {};
    for (const { field, monitor } of this.charts) {
      out[field] = monitor.scores(data);
    }
    return out;
  }

  /**
   * Predicts and returns the input rows with the alert fields filled
   * (new objects, the input is not mutated).
   * @param data - Row objects with the same monitored fields as at fit time.
   * @returns Rows + `alert: boolean` + `alertFields: string[]`.
   */
  fill_predict(data: JsonRow[]): JsonRow[] {
    this.requireFitted();
    const alerts = this.predict(data);
    const fields = this.alertFields(data);
    return data.map((row, i) => ({
      ...row,
      [this.alertField]: alerts[i] === 1,
      [this.alertFieldsField]: fields[i],
    }));
  }

  /** The monitored field names, in spec order. */
  get monitoredFields(): string[] {
    return this.charts.map((c) => c.field);
  }

  /** Per-field reference means μ0 (estimated at fit, or provided). */
  get targets(): Record<string, number> {
    const out: Record<string, number> = {};
    for (const { field, monitor } of this.charts) {
      out[field] = monitor.target;
    }
    return out;
  }

  /** Resolved per-field params (useful to inspect the Bonferroni limits). */
  get params(): Record<string, CUSUMParams | EWMAParams> {
    const out: Record<string, CUSUMParams | EWMAParams> = {};
    for (const { field, monitor } of this.charts) {
      if (monitor instanceof EWMA) {
        out[field] = {
          lambda: monitor.lambda,
          limit: monitor.limit,
          target: monitor.target,
          std: monitor.std,
        };
      } else {
        out[field] = {
          target: monitor.target,
          std: monitor.std,
          k: monitor.slack,
          h: monitor.threshold,
        };
      }
    }
    return out;
  }

  /**
   * Exports the fitted model (per-chart stats + spec) to a `<name>.json`
   * file (async).
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    this.requireFitted();
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      model: this.model,
      familyError: this.familyError ?? null,
      alertField: this.alertField,
      alertFieldsField: this.alertFieldsField,
      spec: this.spec,
      charts: this.charts.map(({ field, monitor }) =>
        monitor instanceof EWMA
          ? {
              field,
              target: monitor.target,
              std: monitor.std,
              lambda: monitor.lambda,
              limit: monitor.limit,
            }
          : {
              field,
              target: monitor.target,
              std: monitor.std,
              k: monitor.slack,
              h: monitor.threshold,
              direction: monitor.direction,
            },
      ),
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
      familyError?: number | null;
      alertField: string;
      alertFieldsField: string;
      spec: ParallelSpec;
      charts: Array<{
        field: string;
        target: number;
        std: number;
        k?: number;
        h?: number;
        lambda?: number;
        limit?: number;
        direction?: "increase" | "decrease" | "both";
      }>;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = payload.model;
    this.familyError = payload.familyError ?? undefined;
    this.alertField = payload.alertField;
    this.alertFieldsField = payload.alertFieldsField;
    this.spec = payload.spec;
    this.charts = payload.charts.map((c) => {
      if (this.model === "ewma") {
        const monitor = new EWMA({
          target: c.target,
          std: c.std,
          lambda: c.lambda,
          limit: c.limit,
        });
        monitor.setSpec({ field: c.field, missing: this.spec?.missing });
        return { field: c.field, monitor };
      }
      const monitor = new CUSUM({
        target: c.target,
        std: c.std,
        k: c.k,
        h: c.h,
        direction: c.direction,
      });
      monitor.setSpec({ field: c.field, missing: this.spec?.missing });
      return { field: c.field, monitor };
    });
    this.fitted = true;
  }

  /**
   * Online update: consumes one row and advances every chart (each chart
   * keeps its own state since fit/reset).
   * @param row - One row with all the monitored fields.
   * @returns Global alert, the triggering fields, and each field's score.
   */
  update(row: JsonRow): {
    alert: boolean;
    alertFields: string[];
    scores: Record<string, number>;
  } {
    this.requireFitted();
    const alertFields: string[] = [];
    const scores: Record<string, number> = {};
    let alert = false;
    for (const { field, monitor } of this.charts) {
      const res = monitor.update(row);
      scores[field] = res.score;
      if (res.alert) {
        alert = true;
        alertFields.push(field);
      }
    }
    return { alert, alertFields, scores };
  }

  private requireFitted(): void {
    if (!this.fitted) {
      throw new Error("Call fit before predict.");
    }
  }
}

/** Inverse standard normal CDF (Acklam's rational approximation). */
function normInv(p: number): number {
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q +
        c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
    );
  }
  if (p <= pHigh) {
    const q = p - 0.5;
    const r = q * q;
    return (
      ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r +
        a[5]!) *
        q) /
      (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
    );
  }
  const q = Math.sqrt(-2 * Math.log(1 - p));
  return (
    -(
      ((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q +
      c[5]!
    ) /
    ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
  );
}

export { ParallelMonitor };
export type { ParallelSpec, JsonRow } from "../../@types/json";
