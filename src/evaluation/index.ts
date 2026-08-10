import { kml } from "../core";
import type { JsonRow, JsonFitSpec } from "../types/json";

/**
 * Shared helpers for model evaluation (`score`, `mse`, `classificationReport`,
 * `rocAucScore`). Ground truth is read from the `target` field of the rows,
 * aligned with the rows kept by the last transform ('drop' strategy).
 */

/** Coerces a target value to a number for scoring (boolean → 1/0, numeric string → Number). */
export function toScoreNumber(v: unknown): number | undefined {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) {
    return Number(v);
  }
  return undefined;
}

/**
 * Mean absolute error between predictions and ground truth. Expressed in the
 * target's units — easier to interpret than MSE ("on average the model is off
 * by X"), but it does not penalize large errors as strongly.
 * @param preds - Model predictions.
 * @param truth - Ground-truth values (same length as `preds`).
 * @returns MAE (0 = perfect).
 */
export function meanAbsoluteError(preds: number[], truth: number[]): number {
  if (preds.length !== truth.length) {
    throw new Error(
      `Predictions (${preds.length}) and truth (${truth.length}) must have the same length.`,
    );
  }
  if (preds.length === 0) return 0;
  return (
    preds.reduce((sum, p, i) => sum + Math.abs(p - (truth[i] ?? 0)), 0) /
    preds.length
  );
}

/**
 * F-beta score computed from precision and recall. The `beta` parameter
 * weights recall relative to precision (β = 1 → F1, β = 2 → recall twice as
 * important, β = 0.5 → precision twice as important). Used by
 * `classificationReport(data, beta)` on the classifiers.
 * @param precision - Precision (TP / (TP + FP)).
 * @param recall - Recall (TP / (TP + FN)).
 * @param beta - Recall weight. Default: 1 (F1).
 * @returns Fβ in [0, 1] (0 when precision or recall is 0).
 */
export function fbetaFromPrecisionRecall(
  precision: number,
  recall: number,
  beta = 1,
): number {
  const b2 = beta * beta;
  const denom = b2 * precision + recall;
  if (denom === 0) return 0;
  return ((1 + b2) * (precision * recall)) / denom;
}

/**
 * Root mean squared error: sqrt(MSE), in the **same units as the target** —
 * the most reported regression metric in production.
 * @param preds - Model predictions.
 * @param truth - Ground-truth values.
 * @returns RMSE (0 = perfect).
 */
export function rmse(preds: number[], truth: number[]): number {
  checkLengths(preds, truth, "rmse");
  return kml.Metrics.rootMeanSquaredError(preds, truth);
}

/**
 * Mean absolute percentage error: how many percent off on average
 * (sklearn's epsilon-clamped formula: |y − p| / max(|y|, eps)).
 * @param preds - Model predictions.
 * @param truth - Ground-truth values (0-values are clamped to machine eps).
 * @returns MAPE as a fraction (0 = perfect, 0.5 = 50% off on average).
 */
export function mape(preds: number[], truth: number[]): number {
  checkLengths(preds, truth, "mape");
  return kml.Metrics.meanAbsolutePercentageError(preds, truth);
}

/**
 * Median absolute error: the median of |y − p| — robust to a few large
 * outliers that inflate MAE.
 * @param preds - Model predictions.
 * @param truth - Ground-truth values.
 * @returns Median AE (0 = perfect).
 */
export function medianAbsoluteError(preds: number[], truth: number[]): number {
  checkLengths(preds, truth, "medianAbsoluteError");
  return kml.Metrics.medianAbsoluteError(preds, truth);
}

/**
 * Matthews correlation coefficient: the single most robust number for
 * (imbalanced) classification — −1 (inverted), 0 (random), +1 (perfect).
 * Preferred over F1/accuracy on imbalanced data (sklearn's recommendation).
 * @param preds - Predicted labels (0/1).
 * @param truth - Ground-truth labels.
 * @returns MCC in [−1, 1].
 */
export function mcc(preds: number[], truth: number[]): number {
  checkLengths(preds, truth, "mcc");
  return kml.Metrics.matthewsCorrcoef(preds, truth);
}

/** Options for `balancedAccuracy`. */
export interface BalancedAccuracyOptions {
  /** Chance-adjust so random performance scores 0 (sklearn `adjusted`). */
  adjusted?: boolean;
}

/**
 * Balanced accuracy: macro-average of per-class recall — the "honest"
 * accuracy when one class dominates.
 * @param preds - Predicted labels.
 * @param truth - Ground-truth labels.
 * @param options - Chance adjustment.
 * @returns Balanced accuracy in [0, 1].
 */
export function balancedAccuracy(
  preds: number[],
  truth: number[],
  options?: BalancedAccuracyOptions,
): number {
  checkLengths(preds, truth, "balancedAccuracy");
  return kml.Metrics.balancedAccuracyScore(preds, truth, options);
}

/** Options for `logLoss`. */
export interface LogLossOptions {
  /** Probability clipping epsilon ('auto' = machine epsilon). */
  eps?: number | "auto";
  /** Mean per-sample (default) or sum. */
  normalize?: boolean;
  /** Explicit class labels (required when truth has a single class). */
  labels?: number[];
}

/**
 * Log loss (cross-entropy): punishes confident **and wrong** probabilities.
 * Needs a `predict_proba` model. Truth first, probabilities second
 * (kml/sklearn convention for score-based metrics).
 * @param truth - Ground-truth labels.
 * @param proba - Positive-class probabilities (1-D) or class matrix (2-D).
 * @param options - Clipping, normalization, labels.
 * @returns Log loss (0 = perfect calibration).
 */
export function logLoss(
  truth: number[],
  proba: number[] | number[][],
  options?: LogLossOptions,
): number {
  checkLengths(truth, proba, "logLoss");
  return kml.Metrics.logLoss(truth, proba, options);
}

/** One point set of a curve (ROC or PR). */
export interface CurveResult {
  /** False positive rate (ROC) or precision (PR), per threshold. */
  fpr?: number[];
  tpr?: number[];
  precision?: number[];
  recall?: number[];
  /** Decision thresholds (one per point). */
  thresholds: number[];
}

/**
 * ROC curve points (FPR/TPR per threshold) — the full curve, not just the
 * AUC, for plots and threshold selection.
 * @param truth - Ground-truth labels.
 * @param proba - Positive-class probabilities.
 * @returns `{ fpr, tpr, thresholds }` (truth first, kml convention).
 */
export function rocCurve(
  truth: number[],
  proba: number[],
): { fpr: number[]; tpr: number[]; thresholds: number[] } {
  checkLengths(truth, proba, "rocCurve");
  return kml.Metrics.rocCurve(truth, proba, 1);
}

/**
 * Precision-Recall AUC (average precision): the right curve metric for
 * **imbalanced** detection — it drops when false alarms grow, unlike ROC-AUC.
 * sklearn-consistent recall-weighted interpolation.
 * @param truth - Ground-truth labels.
 * @param proba - Positive-class probabilities.
 * @returns Average precision in [0, 1] (1 = perfect ranking).
 */
export function prAucScore(truth: number[], proba: number[]): number {
  checkLengths(truth, proba, "prAucScore");
  const { precision, recall } = kml.Metrics.precisionRecallCurve(
    truth,
    proba,
    1,
  );
  // kml returns the curve in descending-threshold order (recall non-increasing).
  // Walk it backwards, from recall 0 up, like sklearn's average_precision_score:
  // AP = Σ (R_n − R_{n−1}) · P_n with the implicit start (recall 0, precision 1).
  let auc = 0;
  let prevRecall = 0;
  for (let i = precision.length - 1; i >= 0; i--) {
    auc += (recall[i]! - prevRecall) * (precision[i] ?? 0);
    prevRecall = recall[i]!;
  }
  return auc;
}

/**
 * Optimal decision threshold via Youden's J (maximize tpr − fpr on the ROC
 * curve). Use it to turn probabilities into alerts: predict positive when
 * proba ≥ returned threshold.
 * @param truth - Ground-truth labels.
 * @param proba - Positive-class probabilities.
 * @returns The threshold maximizing tpr − fpr.
 */
export function optimalThreshold(truth: number[], proba: number[]): number {
  checkLengths(truth, proba, "optimalThreshold");
  const { fpr, tpr, thresholds } = kml.Metrics.rocCurve(truth, proba, 1);
  let best = 0;
  let bestJ = -Infinity;
  for (let i = 0; i < thresholds.length; i++) {
    const j = (tpr[i] ?? 0) - (fpr[i] ?? 0);
    if (j > bestJ) {
      bestJ = j;
      best = thresholds[i] ?? 0;
    }
  }
  return best;
}

/** Guards the array lengths of the pure metric helpers. */
function checkLengths(a: unknown[], b: unknown[], name: string): void {
  if (a.length !== b.length) {
    throw new Error(
      `${name}: predictions (${a.length}) and truth (${b.length}) must have the same length.`,
    );
  }
}

/**
 * Extracts the ground-truth target values aligned with the prediction output
 * (the same rows kept by the 'drop' missing strategy at predict time).
 * @param data - Row objects including the `target` field.
 * @param keptIndices - Indices of the rows kept by the last transform.
 * @param targetField - The output (target) field name.
 * @returns One truth value per kept row.
 */
export function truthValues(
  data: JsonRow[],
  keptIndices: number[],
  targetField: string,
): number[] {
  const out: number[] = [];
  for (const idx of keptIndices) {
    const v = toScoreNumber(data[idx]?.[targetField]);
    if (v === undefined) {
      throw new Error(
        `Target field "${targetField}" must hold numbers for scoring (row ${idx}: ${JSON.stringify(data[idx])}).`,
      );
    }
    out.push(v);
  }
  return out;
}

/** Options for the streaming helpers. */
export interface StreamOptions {
  /** Rows processed per chunk. Default: 1000. */
  chunkSize?: number;
}

/** A model exposing the batch `predict` interface. */
interface Predictable {
  predict(data: JsonRow[]): number[];
}

/**
 * Streams predictions over a (possibly huge) iterable of rows, processing
 * them in chunks so memory stays bounded. Yields one prediction per row.
 * Works with plain arrays, generators and async generators.
 * @param model - Any fitted model with `predict(data)`.
 * @param rows - Rows as an array, generator or async generator.
 * @param options - Chunk size (default 1000).
 */
export async function* predictStream(
  model: Predictable,
  rows: Iterable<JsonRow> | AsyncIterable<JsonRow>,
  options: StreamOptions = {},
): AsyncGenerator<number> {
  const chunkSize = Math.max(1, options.chunkSize ?? 1000);
  let chunk: JsonRow[] = [];
  for await (const row of rows) {
    chunk.push(row);
    if (chunk.length >= chunkSize) {
      yield* model.predict(chunk);
      chunk = [];
    }
  }
  if (chunk.length > 0) {
    yield* model.predict(chunk);
  }
}

/** A model exposing the batch `fill_predict` interface. */
interface FillPredictable {
  fill_predict(data: JsonRow[]): JsonRow[];
}

/**
 * Streams `fill_predict` over an iterable of rows, invoking `onRow` for each
 * enriched row as it is produced (memory stays bounded).
 * @param model - Any fitted model with `fill_predict(data)`.
 * @param rows - Rows as an array, generator or async generator.
 * @param onRow - Callback (sync or async) per enriched row.
 * @param options - Chunk size (default 1000).
 */
export async function fillPredictStream(
  model: FillPredictable,
  rows: Iterable<JsonRow> | AsyncIterable<JsonRow>,
  onRow: (row: JsonRow) => void | Promise<void>,
  options: StreamOptions = {},
): Promise<void> {
  const chunkSize = Math.max(1, options.chunkSize ?? 1000);
  let chunk: JsonRow[] = [];
  for await (const row of rows) {
    chunk.push(row);
    if (chunk.length >= chunkSize) {
      for (const filled of model.fill_predict(chunk)) {
        await onRow(filled);
      }
      chunk = [];
    }
  }
  if (chunk.length > 0) {
    for (const filled of model.fill_predict(chunk)) {
      await onRow(filled);
    }
  }
}

/**
 * Positive-class probabilities from a `predict_proba` output (binary case):
 * column 1 of each row (kml binary probas are [p0, p1]).
 */
export function positiveProbabilities(probas: number[][]): number[] {
  return probas.map((row) => row[1] ?? 0);
}

/** Options for `trainTestSplit`. */
export interface TrainTestSplitOptions {
  /** Fraction (0, 1) or absolute count for the test fold. Default: 0.25. */
  testSize?: number;
  /** Fraction (0, 1) or absolute count for the train fold (complement of testSize). */
  trainSize?: number;
  /** Whether to shuffle before splitting. Default: true. */
  shuffle?: boolean;
  /** Seed for reproducible splits. */
  randomState?: number;
  /** Field to stratify on: preserves the class proportions in both folds. */
  stratify?: string;
}

/**
 * Splits JSON rows into train/test sets (JSON-first wrapper of the kml
 * sampling utility). The rows keep all their fields — ready for `fit` and
 * `score`.
 * @param data - Row objects to split.
 * @param options - Split configuration (sizes, shuffle, seed, stratification).
 * @returns `{ train, test }` — the two row arrays.
 */
export function trainTestSplit(
  data: JsonRow[],
  options: TrainTestSplitOptions = {},
): { train: JsonRow[]; test: JsonRow[] } {
  const stratifyValues = options.stratify
    ? data.map((row) => row[options.stratify!])
    : undefined;
  const res = kml.utils.Sampling.trainTestSplit(data, stratifyValues, {
    testSize: options.testSize,
    trainSize: options.trainSize,
    shuffle: options.shuffle ?? true,
    randomState: options.randomState,
    stratify: stratifyValues,
  });
  return { train: res.XTrain, test: res.XTest };
}

/** A supervised model usable with `crossValScore` (structural type). */
interface EvaluableModel {
  fit(data: JsonRow[], spec: JsonFitSpec): void;
  score(data: JsonRow[]): number;
  mse?(data: JsonRow[]): number;
  mae?(data: JsonRow[]): number;
  rocAucScore?(data: JsonRow[]): number;
}

/** Options for `crossValScore`. */
export interface CrossValOptions {
  /** Number of folds. Default: 5. */
  cv?: number;
  /**
   * Evaluation method called on each held-out fold. Default: 'score'
   * (R² on regressors, accuracy on classifiers).
   */
  scoring?: "score" | "mse" | "mae" | "rocAucScore";
  /**
   * Stratify folds by the target field (requires ≥ cv samples per class).
   * Default: auto — stratified when the target is discrete with enough
   * samples per class, plain k-fold otherwise.
   */
  stratify?: boolean;
  /** Seed for reproducible folds. */
  randomState?: number;
}

/**
 * K-fold cross-validation on @dnax/ml wrappers. Fits a fresh model (via the
 * factory) on k−1 folds and scores the held-out fold, returning one score
 * per fold — a robust model comparison without a single arbitrary split.
 * @param createModel - Factory returning a fresh model, e.g. `() => new LinearRegression()`.
 * @param data - Row objects including the `target` field.
 * @param spec - The fit specification (features + target + options).
 * @param options - Fold count, scoring method, stratification, seed.
 * @returns One score per fold.
 */
export function crossValScore(
  createModel: () => EvaluableModel,
  data: JsonRow[],
  spec: JsonFitSpec,
  options: CrossValOptions = {},
): number[] {
  const cv = options.cv ?? 5;
  if (cv < 2) {
    throw new Error("crossValScore requires cv >= 2.");
  }
  const scoring = options.scoring ?? "score";
  const target = spec.target;

  const y = target
    ? data.map((row) => toScoreNumber(row[target]) ?? 0)
    : undefined;

  // Auto-stratification: only when the target is discrete with enough
  // samples per class (kml StratifiedKFold throws otherwise).
  let stratify = options.stratify;
  if (stratify === undefined) {
    if (y) {
      const counts = new Map<number, number>();
      for (const v of y) counts.set(v, (counts.get(v) ?? 0) + 1);
      stratify = counts.size > 1 && [...counts.values()].every((c) => c >= cv);
    } else {
      stratify = false;
    }
  }

  const splitter = stratify
    ? new kml.utils.ModelSelection.StratifiedKFold({
        nSplits: cv,
        shuffle: true,
        randomState: options.randomState,
      })
    : new kml.utils.ModelSelection.KFold({
        nSplits: cv,
        shuffle: true,
        randomState: options.randomState,
      });
  const folds = splitter.split(data, y);

  const scores: number[] = [];
  for (const { trainIndices, testIndices } of folds) {
    const trainData = trainIndices
      .map((i) => data[i]!)
      .filter((r) => r !== undefined);
    const testData = testIndices
      .map((i) => data[i]!)
      .filter((r) => r !== undefined);
    const model = createModel();
    model.fit(trainData, spec);
    const scorer = (model as unknown as Record<string, unknown>)[scoring];
    if (typeof scorer !== "function") {
      throw new Error(
        `Scoring method "${scoring}" is not available on this model (choose 'score', 'mse', 'mae' or 'rocAucScore').`,
      );
    }
    scores.push((scorer as (d: JsonRow[]) => number).call(model, testData));
  }
  return scores;
}

/**
 * Detects the ML task from the target field: `'classification'` when the
 * target is boolean or has few unique values (≤ 10), `'regression'`
 * otherwise. Useful to pick a model family / default scoring for a case.
 * @param data - Row objects including the `target` field.
 * @param spec - The fit specification (features + target + options).
 * @returns 'classification' | 'regression'.
 */
export function detectTask(
  data: JsonRow[],
  spec: JsonFitSpec,
): "classification" | "regression" {
  const n = data.length;
  const seen = new Set<string>();
  let allBoolean = true;
  for (const row of data) {
    const v = row[spec.target];
    if (typeof v !== "boolean") allBoolean = false;
    seen.add(typeof v === "boolean" ? (v ? "1" : "0") : String(v));
  }
  // Boolean targets are always classification.
  if (allBoolean && n > 0) return "classification";
  // Discrete (≤ 10 unique values and a minority of the rows) → classification.
  if (seen.size <= 10 && seen.size <= Math.max(2, Math.floor(n / 2))) {
    return "classification";
  }
  return "regression";
}

/** Options for `compareModels`. */
export interface CompareModelsOptions {
  /** Number of folds per model. Default: 5. */
  cv?: number;
  /**
   * Evaluation method per fold. Default: 'score' — each model uses its own
   * `score()`: R² for regressors, accuracy for classifiers (the ML case is
   * detected per model).
   */
  scoring?: "score" | "mse" | "mae" | "rocAucScore";
  /** Stratify folds by the target field. Default: auto (see crossValScore). */
  stratify?: boolean;
  /** Seed for reproducible folds. */
  randomState?: number;
}

/** One model's result in the leaderboard. */
export interface ModelBenchmark {
  /** Model name (key of the models map). */
  name: string;
  /** Mean score across folds. Higher is better for 'score'/'rocAucScore', lower for 'mse'/'mae'. */
  mean: number;
  /** Standard deviation across folds (lower = more stable). */
  std: number;
  /** One score per fold. */
  scores: number[];
}

/**
 * Trains, evaluates and **ranks** several models on the same case (same
 * data + spec + folds). Each model gets a fresh fit per fold via
 * `crossValScore`, the same spec (so `options.scale` etc. are applied
 * equally to every model), and the results are sorted into a leaderboard.
 *
 * Use models of the same family — all regressors (R² comparable) or all
 * classifiers (accuracy comparable).
 * @param models - Named factories, e.g. `{ ridge: () => new RidgeRegression({ alpha: 1 }) }`.
 * @param data - Row objects including the `target` field.
 * @param spec - The fit specification (features + target + options).
 * @param options - Folds, scoring method, stratification, seed.
 * @returns One benchmark per model, sorted best first (ascending for 'mse'/'mae').
 */
export function compareModels(
  models: Record<string, () => EvaluableModel>,
  data: JsonRow[],
  spec: JsonFitSpec,
  options: CompareModelsOptions = {},
): ModelBenchmark[] {
  const entries = Object.entries(models);
  if (entries.length === 0) {
    throw new Error("compareModels requires at least one model.");
  }
  const scoring = options.scoring ?? "score";
  const higherIsBetter = scoring !== "mse" && scoring !== "mae";
  const ranked = entries.map(([name, create]) => {
    const scores = crossValScore(create, data, spec, {
      cv: options.cv,
      scoring,
      stratify: options.stratify,
      randomState: options.randomState,
    });
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance =
      scores.reduce((a, s) => a + (s - mean) ** 2, 0) / scores.length;
    return { name, mean, std: Math.sqrt(variance), scores };
  });
  return ranked.sort((a, b) =>
    higherIsBetter ? b.mean - a.mean : a.mean - b.mean,
  );
}
