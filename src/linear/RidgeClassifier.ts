import {
  kml,
  classesOf,
  interceptOf,
  setCoef,
  setIntercept,
  toJSONOf,
  loadModel,
  asyncMode,
} from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { JsonFitSpec, JsonRow } from "../@types/json";
import { truthValues } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a RidgeClassifier model (constructor config + restore). */
export interface RidgeClassifierParams {
  /** L2 regularization strength. Default: 1. */
  alpha?: number;
  /** Fit the intercept/bias term. Default: true. */
  fitIntercept?: boolean;
  /** Coefficients matrix (one row per class) — restore a trained model. */
  coef?: number[][];
  /** Intercepts (one per class) — restore a trained model. */
  intercept?: number[];
  /** Sorted class labels — restore a trained model. */
  classes?: number[];
}

/**
 * RidgeClassifier: L2-regularized linear classifier (one-vs-rest ridge
 * models). Fast and interpretable on numeric tabular data. Binary and
 * multiclass supported. Same JSON-first API as LogisticRegression, but
 * WITHOUT `predict_proba` (no probabilities — only labels).
 */
/** Serialized shape of the kml RidgeClassifier (official serializer). */
type SerializedRidgeClassifier = ReturnType<
  InstanceType<typeof kml.Linear.RidgeClassifier>["toJSON"]
>;

class RidgeClassifier {
  private model: InstanceType<typeof kml.Linear.RidgeClassifier>;
  private transformer: JsonTransformer | null = null;
  private fitted = false;

  /**
   * Creates a model. `alpha` / `fitIntercept` configure the regularization;
   * `coef` / `intercept` optionally restore a previously trained model.
   * @param params - Model parameters.
   */
  constructor(params?: RidgeClassifierParams) {
    const { coef, intercept, alpha, fitIntercept } = params ?? {};
    this.model = new kml.Linear.RidgeClassifier({ alpha, fitIntercept });
    if (coef !== undefined || intercept !== undefined) {
      this.setParams({ coef, intercept });
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
    this.fitted = true;
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

  /** X column names (useful to interpret the coefficients). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last fit/predict. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /** Coefficients matrix (one row of weights per class). */
  get coef(): number[][] {
    if (!this.fitted) {
      throw new Error("Call fit before accessing coef.");
    }
    return this.model.coef;
  }

  /** Intercepts (one per class). */
  get intercept(): number[] {
    if (!this.fitted) {
      throw new Error("Call fit before accessing intercept.");
    }
    const m = this.model as unknown as {
      models?: InstanceType<typeof kml.Linear.RidgeRegression>[];
    };
    return (m.models ?? []).map((r) => interceptOf(r));
  }

  /** Sorted class labels. */
  get classes(): number[] {
    return classesOf(this.model);
  }

  /**
   * Injects model parameters (coef / intercept / classes), restoring a
   * trained model. Unknown keys throw. Returns `this` for chaining
   * (sklearn-style).
   * @param params - Parameters to set.
   */
  setParams(
    params: Pick<RidgeClassifierParams, "coef" | "intercept" | "classes">,
  ): this {
    const allowed: (keyof Pick<
      RidgeClassifierParams,
      "coef" | "intercept" | "classes"
    >)[] = ["coef", "intercept", "classes"];
    for (const key of Object.keys(params) as (keyof Pick<
      RidgeClassifierParams,
      "coef" | "intercept" | "classes"
    >)[]) {
      if (!allowed.includes(key)) {
        throw new Error(`Unknown parameter: ${String(key)}`);
      }
    }
    const m = this.model as unknown as {
      models?: InstanceType<typeof kml.Linear.RidgeRegression>[];
      classes?: number[];
    };
    if (params.coef) {
      const intercepts = params.intercept ?? [];
      m.models = params.coef.map((c, i) => {
        const r = new kml.Linear.RidgeRegression();
        setCoef(r, c);
        setIntercept(r, intercepts[i] ?? 0);
        return r;
      });
    }
    if (params.classes) m.classes = params.classes;
    this.fitted = true;
    return this;
  }

  /**
   * Returns the current model parameters (coef matrix + intercepts + classes).
   * The counterpart of `setParams()` (sklearn-style `get_params()`).
   */
  getParams(): Pick<RidgeClassifierParams, "coef" | "intercept" | "classes"> {
    return {
      coef: this.coef,
      intercept: this.intercept,
      classes: this.classes,
    };
  }

  /**
   * Exports the trained model (parameters + learned transformation) to a
   * `<name>.json` file (async). Uses the official kml serializer.
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    if (!this.transformer) {
      throw new Error("Call fit before export.");
    }
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      model: toJSONOf(this.model),
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
      model: SerializedRidgeClassifier;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<InstanceType<typeof kml.Linear.RidgeClassifier>>(
      payload.model,
    );
    this.fitted = true;
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }

  /**
   * Predicts asynchronously: the JSON → matrix transformation runs on the
   * main thread, then the per-class score computation (argmax) is offloaded
   * to a worker (via kml `utils.asyncMode`).
   * @param data - Row objects with the same features fields as at training time.
   * @returns Predictions of length data.length.
   */
  predictAsync(data: JsonRow[]): Promise<number[]> {
    if (!this.transformer) {
      throw new Error("Call fit before predictAsync.");
    }
    const { X } = this.transformer.transform(data);
    const coef = this.coef;
    const intercept = this.intercept;
    const classes = this.classes;
    // Self-contained function: no closure over `this` (worker-safe).
    const run = asyncMode(
      (rows: number[][], W: number[][], B: number[], cls: number[]): number[] =>
        rows.map((row) => {
          let bestCls = cls[0] ?? 0;
          let bestScore = -Infinity;
          for (let k = 0; k < W.length; k++) {
            const w = W[k] ?? [];
            let s = B[k] ?? 0;
            for (let i = 0; i < row.length; i++) s += row[i]! * (w[i] ?? 0);
            if (s > bestScore) {
              bestScore = s;
              bestCls = cls[k] ?? 0;
            }
          }
          return bestCls;
        }),
    );
    return run(X, coef, intercept, classes);
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
   * Precision / recall / F1 / support + confusion matrix on rows with the
   * target field (binary, positive class = 1).
   * @param data - Row objects including the `target` field (ground truth).
   */
  classificationReport(data: JsonRow[]): {
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
      fScore: prf.fScore,
      support: prf.support,
      confusionMatrix: kml.Metrics.confusionMatrix(preds, truth),
    };
  }
}

export { RidgeClassifier };
export type { JsonFitSpec, JsonRow } from "../@types/json";
