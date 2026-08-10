import { kml, coefOf, fittedOf, setCoef, setFitted, classesOf } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { JsonFitSpec, JsonRow } from "../types/json";
import { truthValues, positiveProbabilities, fbetaFromPrecisionRecall } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters that can be injected into a LogisticRegression model. */
export interface LogisticParams {
  /** Coefficients (one weight per feature, or a matrix for multiclass). */
  coef?: number[] | number[][];
  /** Intercept (bias term, binary case). */
  intercept?: number;
}

/**
 * Logistic regression classifier: predicts a category (e.g. boolean target
 * where `false` → 0 and `true` → 1). Same JSON-first API as LinearRegression,
 * plus `predict_proba` for per-class probabilities.
 */
class LogisticRegression {
  private model: InstanceType<typeof kml.Linear.LogisticRegression>;
  private transformer: JsonTransformer | null = null;

  /**
   * Creates a model. Optionally restores previously learned parameters
   * (e.g. from `new LogisticRegression({ coef, intercept })`).
   * @param params - Optional model parameters (coef + intercept).
   */
  constructor(params?: LogisticParams) {
    this.model = new kml.Linear.LogisticRegression();
    if (params) {
      this.setParams(params);
    }
  }

  /**
   * Fits the model on JSON rows.
   * @param data - Array of row objects, e.g. [{ name, note, admis }].
   * @param spec - Specification of the features and target fields + options.
   */
  fit(data: JsonRow[], spec: JsonFitSpec): void {
    this.transformer = new JsonTransformer(spec);
    const { X, Y } = this.transformer.fitTransform(data);
    this.model.fit(X, Y);
  }

  /**
   * Predicts class labels on JSON rows reusing the transformation learned
   * during fit. A boolean target is encoded `false` → 0, `true` → 1.
   * @param data - Row objects with the same features fields as at training time.
   * @returns Predicted labels of length data.length.
   */
  predict(data: JsonRow[]): number[] {
    if (!this.transformer) {
      throw new Error("Call fit before predict.");
    }
    const { X } = this.transformer.transform(data);
    return this.model.predict(X);
  }

  /**
   * Predicts and returns the input rows with the `target` field filled with
   * the predicted label (new objects, the input is not mutated).
   * @param data - Row objects with the same features fields as at training time.
   * @returns The input rows with the target field set to the predicted label
   * (rows dropped by 'drop' are excluded).
   */
  fill_predict(data: JsonRow[]): JsonRow[] {
    if (!this.transformer) {
      throw new Error("Call fit before fill_predict.");
    }
    const { X } = this.transformer.transform(data);
    const preds = this.model.predict(X);
    const kept = this.transformer.keptIndices;
    const target = this.transformer.targetField;
    const asBoolean = this.transformer.targetBoolean;
    return kept.map((idx, i) => {
      const pred = preds[i] ?? 0;
      return { ...data[idx], [target]: asBoolean ? pred === 1 : pred };
    });
  }

  /**
   * Predicts per-class probabilities on JSON rows.
   * @param data - Row objects with the same features fields as at training time.
   * @returns Probabilities of shape [nSamples, nClasses], each row sums to 1.
   */
  predict_proba(data: JsonRow[]): number[][] {
    if (!this.transformer) {
      throw new Error("Call fit before predict_proba.");
    }
    const { X } = this.transformer.transform(data);
    return this.model.predictProba(X);
  }

  /**
   * Predicts and returns the input rows with the `target` field filled with
   * the probability of the positive class, always a number in [0, 1].
   * - Binary: probability of class `classes[1]` (e.g. `true` for a boolean
   *   target, `1` for a 0/1 target).
   * - Multiclass: the highest class probability (confidence).
   * The input is not mutated; rows dropped by 'drop' are excluded.
   * @param data - Row objects with the same features fields as at training time.
   */
  fill_predict_proba(data: JsonRow[]): JsonRow[] {
    if (!this.transformer) {
      throw new Error("Call fit before fill_predict_proba.");
    }
    const { X } = this.transformer.transform(data);
    const probas = this.model.predictProba(X);
    const classes = this.model.getClasses();
    const binary = classes.length === 2;
    const kept = this.transformer.keptIndices;
    const target = this.transformer.targetField;
    return kept.map((idx, i) => {
      const row = probas[i] ?? [];
      const p = binary ? (row[1] ?? 0) : Math.max(...row, 0);
      return { ...data[idx], [target]: p };
    });
  }

  /** Coefficients (one weight per feature, or a matrix for multiclass). */
  get coef(): number[] | number[][] {
    if (!fittedOf(this.model)) {
      throw new Error("Call fit before accessing coef.");
    }
    return coefOf(this.model);
  }

  /** Intercept (bias term, binary case). */
  get intercept(): number {
    if (!fittedOf(this.model)) {
      throw new Error("Call fit before accessing intercept.");
    }
    return (this.model as unknown as { bias?: number }).bias ?? 0;
  }

  /**
   * Injects model parameters (coef / intercept), restoring a trained model.
   * Unknown keys throw. Returns `this` for chaining (sklearn-style).
   * @param params - Parameters to set.
   */
  setParams(params: LogisticParams): this {
    const allowed: (keyof LogisticParams)[] = ["coef", "intercept"];
    for (const key of Object.keys(params) as (keyof LogisticParams)[]) {
      if (!allowed.includes(key)) {
        throw new Error(`Unknown parameter: ${String(key)}`);
      }
    }
    const m = this.model as unknown as { bias?: number; classes?: number[] };
    if (params.coef !== undefined) setCoef(this.model, params.coef as number[]);
    if (params.intercept !== undefined) m.bias = params.intercept;
    if (params.coef !== undefined || params.intercept !== undefined) {
      m.classes = [0, 1];
    }
    setFitted(this.model);
    return this;
  }

  /**
   * Returns the current model parameters (coef + intercept).
   * The counterpart of `setParams()` (sklearn-style `get_params()`).
   */
  getParams(): Required<LogisticParams> {
    return { coef: this.coef, intercept: this.intercept };
  }

  /** X column names (useful to interpret the coefficients). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last fit/predict. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /**
   * Exports the trained model (parameters + learned transformation) to a
   * `<name>.json` file (async).
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    if (!this.transformer) {
      throw new Error("Call fit before export.");
    }
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      coef: this.coef,
      intercept: this.intercept,
      transformer: this.transformer.toJSON(),
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
      coef: number[] | number[][];
      intercept: number;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.setParams({ coef: payload.coef, intercept: payload.intercept });
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }

  /**
   * Accuracy score on rows with the target field. sklearn-style `score()`.
   * @param data - Row objects including the `target` field (ground truth).
   */
  score(data: JsonRow[]): number {
    if (!this.transformer) {
      throw new Error("Call fit before score.");
    }
    const preds = this.predict(data);
    const truth = truthValues(
      data,
      this.transformer.keptIndices,
      this.transformer.targetField,
    );
    return kml.Metrics.accuracyScore(preds, truth);
  }

  /**
   * Precision / recall / Fβ / support + confusion matrix on rows with the
   * target field (binary, positive class = 1).
   * @param beta - Recall weight (default 1 = F1). Higher β prioritizes recall.
   * @param data - Row objects including the `target` field (ground truth).
   */
  classificationReport(data: JsonRow[], beta?: number): {
    accuracy: number;
    precision: number;
    recall: number;
    fScore: number;
    support: number[];
    confusionMatrix: number[][];
  } {
    if (!this.transformer) {
      throw new Error("Call fit before classificationReport.");
    }
    const preds = this.predict(data);
    const truth = truthValues(
      data,
      this.transformer.keptIndices,
      this.transformer.targetField,
    );
    const prf = kml.Metrics.precisionRecallFscoreSupport(preds, truth, {
      average: "binary",
      positiveLabel: 1,
    });
    return {
      accuracy: kml.Metrics.accuracyScore(preds, truth),
      precision: prf.precision,
      recall: prf.recall,
      fScore: fbetaFromPrecisionRecall(prf.precision, prf.recall, beta ?? 1),
      support: prf.support,
      confusionMatrix: kml.Metrics.confusionMatrix(preds, truth),
    };
  }

  /**
   * Area under the ROC curve (binary only) using the positive-class
   * probabilities. 1 = perfect, 0.5 = random.
   * @param data - Row objects including the `target` field (ground truth).
   */
  rocAucScore(data: JsonRow[]): number {
    if (!this.transformer) {
      throw new Error("Call fit before rocAucScore.");
    }
    const classes = classesOf(this.model);
    if (classes.length !== 2) {
      throw new Error("rocAucScore supports binary classification only.");
    }
    const truth = truthValues(
      data,
      this.transformer.keptIndices,
      this.transformer.targetField,
    );
    const probas = this.predict_proba(data);
    return kml.Metrics.rocAucScore(
      truth,
      positiveProbabilities(probas),
      1, // positiveLabel (binary overload takes it positionally)
    );
  }
}

export { LogisticRegression };
export type { JsonFitSpec, JsonRow } from "../types/json";
