import { kml } from "../core";
import type { JsonFitSpec, JsonRow } from "../types/json";

/** Options for mutual-information scoring. */
export interface MutualInfoOptions {
  /** Per-feature (or global) discreteness flag. Default: false. */
  discreteFeatures?: boolean | boolean[];
  /** Number of neighbors for the MI estimator. Default: 5. */
  nNeighbors?: number;
  /** Seed for reproducible neighbor sampling. */
  randomState?: number;
}

/** One feature's univariate score against the target. */
export interface UnivariateResult {
  /** Feature (column) name. */
  feature: string;
  /** Importance score (higher = more informative). */
  score: number;
  /** Two-sided p-value (chi2 / fClassif only). */
  pValue?: number;
}

/**
 * Univariate feature scoring on JSON rows: each feature is scored
 * independently against the target, one result per `spec.features` entry
 * (results sorted by score, most informative first). Features must be
 * numeric or boolean (categorical strings are not supported — one-hot is not
 * applied: the score must stay per feature). `options.missing` is honored
 * ('throw' default | 'fill0' | 'drop' whole row). Features are sorted
 * alphabetically, matching the transformer layout.
 */

/** Chi-square score (classification, non-negative features). */
export function chi2(data: JsonRow[], spec: JsonFitSpec): UnivariateResult[] {
  return score(spec, data, (X, y) => kml.FeatureSelection.chi2(X, y));
}

/** ANOVA F score (classification). */
export function fClassif(data: JsonRow[], spec: JsonFitSpec): UnivariateResult[] {
  return score(spec, data, (X, y) => kml.FeatureSelection.fClassif(X, y));
}

/** Mutual information (classification target) — captures non-linear links. */
export function mutualInfoClassif(
  data: JsonRow[],
  spec: JsonFitSpec,
  options?: MutualInfoOptions,
): UnivariateResult[] {
  return score(
    spec,
    data,
    (X, y) => {
      const s = kml.FeatureSelection.mutualInfoClassif(X, y, options);
      return [s, s.map(() => NaN)];
    },
    false,
  );
}

/** Mutual information (regression target) — captures non-linear links. */
export function mutualInfoRegression(
  data: JsonRow[],
  spec: JsonFitSpec,
  options?: MutualInfoOptions,
): UnivariateResult[] {
  return score(
    spec,
    data,
    (X, y) => {
      const s = kml.FeatureSelection.mutualInfoRegression(X, y, options);
      return [s, s.map(() => NaN)];
    },
    false,
  );
}

type ScoreFn = (X: number[][], y: number[]) => [number[], number[]];

function score(
  spec: JsonFitSpec,
  data: JsonRow[],
  fn: ScoreFn,
  hasPValues = true,
): UnivariateResult[] {
  const { X, Y } = buildMatrix(data, spec);
  const [scores, pValues] = fn(X, Y);
  const names = [...spec.features].sort(); // alphabetical, like the transformer
  return names
    .map((feature, i) => ({
      feature,
      score: scores[i] ?? 0,
      ...(hasPValues ? { pValue: pValues[i] ?? 1 } : {}),
    }))
    .sort((a, b) => b.score - a.score);
}

/** Builds a numeric X + target y, honoring `options.missing`. */
function buildMatrix(
  data: JsonRow[],
  spec: JsonFitSpec,
): { X: number[][]; Y: number[] } {
  const features = [...spec.features].sort();
  const missing = spec.options?.missing ?? "throw";
  const X: number[][] = [];
  const Y: number[] = [];
  for (const row of data) {
    const xRow: number[] = [];
    let dropped = false;
    for (const f of features) {
      const v = toNumber(row[f]);
      if (v === undefined) {
        if (missing === "throw") {
          throw new Error(
            `Field "${f}" must hold numbers (row: ${JSON.stringify(row)}).`,
          );
        }
        if (missing === "drop") {
          dropped = true;
          break;
        }
        xRow.push(0); // 'fill0'
      } else {
        xRow.push(v);
      }
    }
    if (dropped) continue;
    const yv = toNumber(row[spec.target]);
    if (yv === undefined) {
      if (missing === "throw") {
        throw new Error(
          `Field "${spec.target}" must hold numbers (row: ${JSON.stringify(row)}).`,
        );
      }
      if (missing === "drop") continue;
      Y.push(0);
    } else {
      Y.push(yv);
    }
    X.push(xRow);
  }
  if (X.length === 0) {
    throw new Error("Feature selection requires at least one valid row.");
  }
  return { X, Y };
}

/** number | boolean | numeric string → number (categorical strings → undefined). */
function toNumber(v: unknown): number | undefined {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) {
    return Number(v);
  }
  return undefined;
}
