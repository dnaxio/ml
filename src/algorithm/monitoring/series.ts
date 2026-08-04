import type { JsonRow, MonitorSpec } from "../../@types/json";

/**
 * Shared helpers for the monitoring (time-series) models: extract a numeric
 * series from JSON rows and estimate the reference statistics (mean / std).
 */

/** Coerces a JSON value to a number (boolean → 1/0, numeric string → Number). */
export function toNumber(v: unknown): number | undefined {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) {
    return Number(v);
  }
  return undefined;
}

/**
 * Extracts the numeric series from the field selected by the spec.
 * Non-numeric values throw by default; with `missing: "fill"` they are
 * replaced by the last known value (carry-forward / LOCF), keeping the
 * output aligned with the input rows.
 * @param data - Row objects, e.g. [{ time: 1, value: 120 }].
 * @param spec - Specification of the field to monitor.
 * @returns The numeric series, one value per row.
 */
export function extractSeries(data: JsonRow[], spec: MonitorSpec): number[] {
  const out: number[] = [];
  let last: number | undefined;
  for (const row of data) {
    const v = toNumber(row[spec.field]);
    if (v === undefined) {
      if (spec.missing !== "fill") {
        throw new Error(
          `Field "${spec.field}" must hold numbers (row: ${JSON.stringify(row)}).`,
        );
      }
      if (last === undefined) {
        throw new Error(
          `Field "${spec.field}" is missing at the start of the series (no value to carry forward).`,
        );
      }
      out.push(last);
      continue;
    }
    last = v;
    out.push(v);
  }
  return out;
}

/** Arithmetic mean of a series. */
export function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Population standard deviation (ddof = 0, consistent with StandardScaler). */
export function std(values: number[]): number {
  const m = mean(values);
  return Math.sqrt(
    values.reduce((a, v) => a + (v - m) ** 2, 0) / values.length,
  );
}

/** Median of a series. */
export function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? (sorted[mid] ?? 0)
    : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
}

/**
 * Median absolute deviation, scaled to be a consistent estimator of σ
 * (1.4826 · MAD). Robust to outliers in the baseline: a few anomalous days
 * barely move it, unlike the standard deviation.
 */
export function mad(values: number[]): number {
  const m = median(values);
  return 1.4826 * median(values.map((v) => Math.abs(v - m)));
}

/**
 * Day of week (0 = Sunday … 6 = Saturday) from a Date, an ISO date string
 * ("2026-08-03" is parsed as local midnight) or a timestamp.
 */
export function dayOfWeek(v: unknown): number {
  if (v instanceof Date) return v.getDay();
  if (typeof v === "number") return new Date(v).getDay();
  if (typeof v === "string") {
    if (/^\d{4}-\d{2}-\d{2}$/.test(v)) {
      const [y, m, d] = v.split("-").map(Number);
      return new Date(y ?? 0, (m ?? 1) - 1, d ?? 1).getDay();
    }
    const parsed = new Date(v);
    if (!Number.isNaN(parsed.getTime())) return parsed.getDay();
  }
  throw new Error(
    `dateField must hold a Date, ISO string or timestamp (got: ${JSON.stringify(v)}).`,
  );
}
