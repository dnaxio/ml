import { kml, coefOf, setCoef, configOf, asyncMode } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { JsonFitSpec, JsonRow } from "../@types/json";
import { truthValues, meanAbsoluteError } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a PoissonRegressor model (constructor config + restore). */
export interface PoissonParams {
  /** L2 regularization strength. Default: 1. */
  alpha?: number;
  /** Fit the intercept/bias term. Default: true. */
  fitIntercept?: boolean;
  /** Maximum number of iterations. Default: 100. */
  maxIter?: number;
  /** Convergence tolerance. Default: 1e-4. */
  tol?: number;
  /** Coefficients (one weight per feature) — restore a trained model. */
  coef?: number[];
  /** Intercept (bias term) — restore a trained model. */
  intercept?: number;
}

/** Parameters accepted by `setParams` / returned by `getParams`. */
type PoissonLearnedParams = Pick<PoissonParams, "coef" | "intercept">;

/**
 * PoissonRegressor: generalized linear model for count targets (non-negative
 * integers). Uses a log link, so predictions are always >= 0. Same
 * JSON-first API as LinearRegression.
 */
class PoissonRegressor {
  private model: InstanceType<typeof kml.Linear.PoissonRegressor>;
  private transformer: JsonTransformer | null = null;
  private fitted = false;

  /**
   * Creates a model. `alpha` / `fitIntercept` / `maxIter` / `tol` configure
   * the model; `coef` / `intercept` optionally restore a trained model.
   * @param params - Model parameters.
   */
  constructor(params?: PoissonParams) {
    const { coef, intercept, alpha, fitIntercept, maxIter, tol } = params ?? {};
    this.model = new kml.Linear.PoissonRegressor({
      alpha,
      fitIntercept,
      maxIter,
      tol,
    });
    if (coef !== undefined || intercept !== undefined) {
      this.setParams({ coef, intercept });
    }
  }

  /**
   * Fits the model on JSON rows.
   * @param data - Array of row objects, e.g. [{ jour, pub, ventes }].
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
   * Predictions are counts-like values and are always >= 0 (log link).
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

  /** Coefficients (one weight per feature). */
  get coef(): number[] {
    if (!this.fitted) {
      throw new Error("Call fit before accessing coef.");
    }
    return coefOf(this.model);
  }

  /** Intercept (bias term). */
  get intercept(): number {
    if (!this.fitted) {
      throw new Error("Call fit before accessing intercept.");
    }
    const m = this.model as unknown as { interceptState?: number };
    return m.interceptState ?? 0;
  }

  /** Number of iterations performed during fit. */
  get nIter(): number {
    const m = this.model as unknown as { nIterState?: number };
    return m.nIterState ?? 0;
  }

  /**
   * Injects model parameters (coef / intercept), restoring a trained model.
   * Unknown keys throw. Returns `this` for chaining (sklearn-style).
   * @param params - Parameters to set.
   */
  setParams(params: PoissonLearnedParams): this {
    const allowed: (keyof PoissonLearnedParams)[] = ["coef", "intercept"];
    for (const key of Object.keys(params) as (keyof PoissonLearnedParams)[]) {
      if (!allowed.includes(key)) {
        throw new Error(`Unknown parameter: ${String(key)}`);
      }
    }
    const m = this.model as unknown as { interceptState?: number };
    if (params.coef !== undefined) setCoef(this.model, params.coef);
    if (params.intercept !== undefined) m.interceptState = params.intercept;
    this.fitted = true;
    return this;
  }

  /**
   * Returns the current model parameters (coef + intercept).
   * The counterpart of `setParams()` (sklearn-style `get_params()`).
   */
  getParams(): Required<PoissonLearnedParams> {
    return { coef: this.coef, intercept: this.intercept };
  }

  /**
   * Exports the trained model (config + parameters + learned transformation)
   * to a `<name>.json` file (async).
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    if (!this.transformer) {
      throw new Error("Call fit before export.");
    }
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const m = configOf(this.model, [
      "regularization",
      "fitIntercept",
      "maxIter",
      "tol",
    ]);
    const payload = {
      version: EXPORT_VERSION,
      alpha: m.regularization,
      fitIntercept: m.fitIntercept,
      maxIter: m.maxIter,
      tol: m.tol,
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
      alpha?: number;
      fitIntercept?: boolean;
      maxIter?: number;
      tol?: number;
      coef: number[];
      intercept: number;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = new kml.Linear.PoissonRegressor({
      alpha: payload.alpha,
      fitIntercept: payload.fitIntercept,
      maxIter: payload.maxIter,
      tol: payload.tol,
    });
    this.setParams({ coef: payload.coef, intercept: payload.intercept });
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }

  /**
   * Predicts asynchronously: the JSON → matrix transformation runs on the
   * main thread, then the log-link computation (exp of the linear predictor)
   * is offloaded to a worker (via kml `utils.asyncMode`).
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
    // Poisson uses a log link: y = exp(X·w + b)
    const run = asyncMode(
      (rows: number[][], w: number[], b: number): number[] =>
        rows.map((row) =>
          Math.exp(row.reduce((sum, x, i) => sum + x * (w[i] ?? 0), b)),
        ),
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

export { PoissonRegressor };
export type { JsonFitSpec, JsonRow } from "../@types/json";
