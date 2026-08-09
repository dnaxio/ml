import {
  kml,
  setCoef,
  setIntercept,
  toJSONOf,
  loadModel,
  asyncMode,
} from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { JsonFitSpec, JsonRow } from "../@types/json";
import { truthValues, meanAbsoluteError } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a RANSACRegressor model (constructor config + restore). */
export interface RANSACParams {
  /** Base estimator (default: LinearRegression). */
  estimator?: unknown;
  /** Minimum number of samples per random subset. */
  minSamples?: number;
  /** Maximum residual for a sample to be an inlier. */
  residualThreshold?: number | null;
  /** Maximum number of random trials. Default: 100. */
  maxTrials?: number;
  /** Probability target of finding a valid consensus set. Default: 0.99. */
  stopProbability?: number;
  /** Random seed. */
  randomState?: number;
  /** Coefficients (from the fitted inner estimator) — restore a model. */
  coef?: number[];
  /** Intercept (from the fitted inner estimator) — restore a model. */
  intercept?: number;
}

/** Serialized shape of the kml RANSACRegressor (official serializer). */
type SerializedRANSAC = ReturnType<
  InstanceType<typeof kml.Linear.RANSACRegressor>["toJSON"]
>;

/**
 * RANSACRegressor: linear regression robust to gross outliers. Fits many
 * random subsets, keeps the consensus set of inliers, and predicts from the
 * best fit. Same JSON-first API as LinearRegression.
 */
class RANSACRegressor {
  private model: InstanceType<typeof kml.Linear.RANSACRegressor>;
  private transformer: JsonTransformer | null = null;
  private fitted = false;

  /**
   * Creates a model. Config props configure the RANSAC search; `coef` /
   * `intercept` optionally restore a previously trained model (the inner
   * estimator is assumed linear).
   * @param params - Model parameters.
   */
  constructor(params?: RANSACParams) {
    const { coef, intercept, ...config } = params ?? {};
    this.model = new kml.Linear.RANSACRegressor(
      config as ConstructorParameters<typeof kml.Linear.RANSACRegressor>[0],
    );
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
   * Predicts on JSON rows reusing the transformation learned during fit.
   * @param data - Row objects with the same features fields as at training time.
   * @returns Predictions of length data.length (rows dropped by 'drop' are excluded).
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
   * the predicted value (new objects, the input is not mutated).
   * @param data - Row objects with the same features fields as at training time.
   * @returns The input rows with the target field set to the prediction
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

  /** Coefficients (from the fitted inner estimator). */
  get coef(): number[] {
    if (!this.fitted) {
      throw new Error("Call fit before accessing coef.");
    }
    const inner = this.model.estimatorFitted as unknown as {
      coef?: number[];
    };
    return inner.coef ?? [];
  }

  /** Intercept (from the fitted inner estimator). */
  get intercept(): number {
    if (!this.fitted) {
      throw new Error("Call fit before accessing intercept.");
    }
    const inner = this.model.estimatorFitted as unknown as {
      intercept?: number;
    };
    return inner.intercept ?? 0;
  }

  /** Which training samples were inliers (true) or outliers (false). */
  get inlierMask(): boolean[] {
    return this.model.inlierMask;
  }

  /** Number of random trials performed during fit. */
  get nTrials(): number {
    return this.model.nTrials;
  }

  /**
   * Injects model parameters (coef / intercept), restoring a trained model.
   * Unknown keys throw. The inner estimator is rebuilt as a linear model.
   * Returns `this` for chaining (sklearn-style).
   * @param params - Parameters to set.
   */
  setParams(params: Pick<RANSACParams, "coef" | "intercept">): this {
    const allowed: (keyof Pick<RANSACParams, "coef" | "intercept">)[] = [
      "coef",
      "intercept",
    ];
    for (const key of Object.keys(params) as (keyof Pick<
      RANSACParams,
      "coef" | "intercept"
    >)[]) {
      if (!allowed.includes(key)) {
        throw new Error(`Unknown parameter: ${String(key)}`);
      }
    }
    const m = this.model as unknown as { estimatorState?: unknown };
    const inner = new kml.Linear.LinearRegression();
    if (params.coef !== undefined) setCoef(inner, params.coef);
    if (params.intercept !== undefined) setIntercept(inner, params.intercept);
    m.estimatorState = inner;
    this.fitted = true;
    return this;
  }

  /**
   * Returns the current model parameters (coef + intercept).
   * The counterpart of `setParams()` (sklearn-style `get_params()`).
   */
  getParams(): Pick<RANSACParams, "coef" | "intercept"> {
    return { coef: this.coef, intercept: this.intercept };
  }

  /**
   * Exports the trained model (full RANSAC state + learned transformation) to
   * a `<name>.json` file (async). Uses the official kml serializer so the
   * inlier mask, trials count and config are preserved.
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
      model: SerializedRANSAC;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<InstanceType<typeof kml.Linear.RANSACRegressor>>(
      payload.model,
    );
    this.fitted = true;
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }

  /**
   * Predicts asynchronously: the JSON → matrix transformation runs on the
   * main thread, then the matrix-vector multiplication is offloaded to a
   * worker (via kml `utils.asyncMode`), keeping the event loop responsive.
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
    // Self-contained function: no closure over `this` (worker-safe).
    const run = asyncMode(
      (rows: number[][], w: number[], b: number): number[] =>
        rows.map((row) => row.reduce((sum, x, i) => sum + x * (w[i] ?? 0), b)),
    );
    return run(X, coef, intercept);
  }

  /**
   * R² score (coefficient of determination) on rows with the target field.
   * Higher is better; 1 = perfect fit. sklearn-style `score()`.
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
    return kml.Metrics.r2Score(preds, truth);
  }

  /**
   * Mean squared error on rows with the target field (lower is better).
   * @param data - Row objects including the `target` field (ground truth).
   */
  mse(data: JsonRow[]): number {
    if (!this.transformer) {
      throw new Error("Call fit before mse.");
    }
    const preds = this.predict(data);
    const truth = truthValues(
      data,
      this.transformer.keptIndices,
      this.transformer.targetField,
    );
    return kml.Metrics.meanSquaredError(preds, truth);
  }

  /**
   * Mean absolute error on rows with the target field (lower is better).
   * @param data - Row objects including the `target` field (ground truth).
   */
  mae(data: JsonRow[]): number {
    if (!this.transformer) {
      throw new Error("Call fit before mae.");
    }
    const preds = this.predict(data);
    const truth = truthValues(
      data,
      this.transformer.keptIndices,
      this.transformer.targetField,
    );
    return meanAbsoluteError(preds, truth);
  }
}

export { RANSACRegressor };
export type { JsonFitSpec, JsonRow } from "../@types/json";
