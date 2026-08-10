import type { JsonFitSpec, JsonRow } from "../types/json";

/** A @dnax/ml model usable as the importance estimator (structural type). */
export interface SelectableEstimator {
  fit(data: JsonRow[], spec: JsonFitSpec): void;
  /** Trees / ensembles (per-column importances, throws before fit). */
  featureImportances?: number[];
  /** Linear models (per-feature weights, throws before fit). */
  coef?: number[] | number[][];
}

/** Parameters of a SelectFromModel selector. */
export interface SelectFromModelParams {
  /**
   * Any @dnax/ml model exposing `featureImportances` (trees/ensemble) or
   * `coef` (linear models).
   */
  estimator: SelectableEstimator;
  /**
   * Importance cut-off. Default: 'mean'. A feature is kept when its
   * importance ≥ threshold.
   */
  threshold?: number | "mean" | "median";
  /** Hard cap on the number of selected features (top-k by importance). */
  maxFeatures?: number;
}

/**
 * SelectFromModel: model-based feature selection. Fits an estimator (a
 * linear model or a tree/ensemble), reads its per-feature importance
 * (`|coef|` or `featureImportances`), and keeps the features whose
 * importance is ≥ a threshold ('mean' / 'median' / explicit number).
 *
 * Per-feature semantics: one-hot is not applied — `options.oneHot` must stay
 * off so each importance value maps to exactly one `spec.features` entry.
 * Features are sorted alphabetically (transformer layout). The selected
 * feature names are ready to feed a fresh `fit` spec.
 */
class SelectFromModel {
  private estimator: SelectableEstimator;
  private threshold: number | "mean" | "median";
  private maxFeatures?: number;
  private featureNames: string[] = [];
  private scores: number[] = [];
  private supportMask: boolean[] = [];
  private selected: string[] = [];
  private resolvedThreshold = 0;

  constructor(params: SelectFromModelParams) {
    this.estimator = params.estimator;
    this.threshold = params.threshold ?? "mean";
    this.maxFeatures = params.maxFeatures;
  }

  /**
   * Fits the estimator on JSON rows and computes the feature support mask.
   * @param data - Row objects, e.g. [{ age, solde, inutile, achete }].
   * @param spec - Specification of the numeric features and the target.
   */
  fit(data: JsonRow[], spec: JsonFitSpec): void {
    if (spec.options?.oneHot) {
      throw new Error(
        "SelectFromModel requires per-feature importances: disable options.oneHot (categorical features are not expanded).",
      );
    }
    this.estimator.fit(data, spec);
    this.featureNames = [...spec.features].sort();
    this.scores = extractImportance(this.estimator, this.featureNames.length);
    this.resolvedThreshold = resolveThreshold(
      this.threshold,
      this.scores,
    );
    this.supportMask = this.scores.map((s) => s >= this.resolvedThreshold);
    this.applyMaxFeatures();
    this.selected = this.featureNames.filter((_, i) => this.supportMask[i]);
  }

  /** Boolean mask aligned with the sorted `spec.features` (true = keep). */
  get support(): boolean[] {
    return this.supportMask;
  }

  /** Names of the selected features — ready for a fresh `fit` spec. */
  get selectedFeatures(): string[] {
    return this.selected;
  }

  /** Per-feature importances (|coef| mean or featureImportances). */
  get featureScores(): number[] {
    return this.scores;
  }

  /** The threshold that was actually used (number / mean / median). */
  get thresholdUsed(): number {
    return this.resolvedThreshold;
  }

  /** The fitted estimator (trained on the full feature set). */
  get fittedEstimator(): SelectableEstimator {
    return this.estimator;
  }

  /** Caps the selection to the top-k features by importance. */
  private applyMaxFeatures(): void {
    if (this.maxFeatures === undefined || this.maxFeatures < 0) return;
    const ranked = this.featureNames
      .map((f, i) => ({ f, s: this.scores[i] ?? 0, keep: this.supportMask[i] }))
      .filter((r) => r.keep)
      .sort((a, b) => b.s - a.s)
      .slice(0, this.maxFeatures);
    const keep = new Set(ranked.map((r) => r.f));
    this.supportMask = this.featureNames.map((f) => keep.has(f));
  }
}

/** Per-feature importance from the estimator (|coef| or featureImportances). */
function extractImportance(
  estimator: SelectableEstimator,
  n: number,
): number[] {
  const m = estimator as unknown as {
    featureImportances?: number[];
    coef?: number[] | number[][];
  };
  if (m.featureImportances !== undefined) {
    if (m.featureImportances.length !== n) {
      throw new Error(
        `featureImportances length (${m.featureImportances.length}) does not match the feature count (${n}).`,
      );
    }
    return m.featureImportances;
  }
  const coef = m.coef;
  if (coef !== undefined) {
    const rows = Array.isArray(coef[0]) ? (coef as number[][]) : [coef as number[]];
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (const row of rows) acc += Math.abs(row[i] ?? 0);
      out.push(acc / rows.length);
    }
    return out;
  }
  throw new Error(
    "estimator must expose featureImportances (trees/ensemble) or coef (linear).",
  );
}

function resolveThreshold(
  threshold: number | "mean" | "median",
  scores: number[],
): number {
  if (typeof threshold === "number") return threshold;
  if (threshold === "median") return median(scores);
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? (sorted[mid] ?? 0)
    : ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2;
}

export { SelectFromModel };
export type { JsonFitSpec, JsonRow } from "../types/json";
