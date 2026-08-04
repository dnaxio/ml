import { kml } from "../core";
import type { JsonRow, JsonFitSpec } from "../../@types/json";

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
  scoring?: "score" | "mse" | "rocAucScore";
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
        `Scoring method "${scoring}" is not available on this model (choose 'score', 'mse' or 'rocAucScore').`,
      );
    }
    scores.push((scorer as (d: JsonRow[]) => number).call(model, testData));
  }
  return scores;
}
