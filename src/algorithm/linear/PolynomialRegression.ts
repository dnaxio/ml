import { kml, interceptOf, setIntercept, setFitted, asyncMode } from "../core";
import { JsonTransformer } from "../../transformation/json";
import type { JsonTransformerState } from "../../transformation/json";
import type { JsonFitSpec, JsonRow } from "../../@types/json";
import { truthValues } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a PolynomialRegression model (constructor config + restore). */
export interface PolynomialParams {
  /** Polynomial degree (>= 1). Default: 2. */
  degree?: number;
  /** Coefficients (one per expanded feature) — restore a trained model. */
  coef?: number[];
  /** Intercept (bias term) — restore a trained model. */
  intercept?: number;
}

/** Parameters accepted by `setParams` / returned by `getParams`. */
type PolynomialLearnedParams = Pick<PolynomialParams, "coef" | "intercept">;

/**
 * PolynomialRegression: fits a nonlinear curve by expanding each feature to
 * powers 1..degree, then fitting ordinary least squares on the expanded
 * matrix. Same JSON-first API as LinearRegression.
 */
class PolynomialRegression {
  private model: InstanceType<typeof kml.Linear.PolynomialRegression>;
  private transformer: JsonTransformer | null = null;
  private fitted = false;

  /**
   * Creates a model. `degree` configures the polynomial expansion; `coef` /
   * `intercept` optionally restore a previously trained model.
   * @param params - Model parameters.
   */
  constructor(params?: PolynomialParams) {
    const { coef, intercept, degree } = params ?? {};
    this.model = new kml.Linear.PolynomialRegression({ degree });
    if (coef !== undefined || intercept !== undefined) {
      this.setParams({ coef, intercept });
    }
  }

  /**
   * Fits the model on JSON rows.
   * @param data - Array of row objects, e.g. [{ quantite, cout }].
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

  /** Coefficients (one per expanded feature: x, x², ..., x^degree). */
  get coef(): number[] {
    if (!this.fitted) {
      throw new Error("Call fit before accessing coef.");
    }
    const m = this.model as unknown as { polynomialCoef?: number[] };
    return m.polynomialCoef ?? [];
  }

  /** Intercept (bias term). */
  get intercept(): number {
    if (!this.fitted) {
      throw new Error("Call fit before accessing intercept.");
    }
    return interceptOf(this.model);
  }

  /** The polynomial degree. */
  get degree(): number {
    const m = this.model as unknown as { degree?: number };
    return m.degree ?? 2;
  }

  /**
   * Injects model parameters (coef / intercept), restoring a trained model.
   * Unknown keys throw. Returns `this` for chaining (sklearn-style).
   * @param params - Parameters to set.
   */
  setParams(params: PolynomialLearnedParams): this {
    const allowed: (keyof PolynomialLearnedParams)[] = ["coef", "intercept"];
    for (const key of Object.keys(
      params,
    ) as (keyof PolynomialLearnedParams)[]) {
      if (!allowed.includes(key)) {
        throw new Error(`Unknown parameter: ${String(key)}`);
      }
    }
    const m = this.model as unknown as { polynomialCoef?: number[] };
    if (params.coef !== undefined) m.polynomialCoef = params.coef;
    if (params.intercept !== undefined)
      setIntercept(this.model, params.intercept);
    setFitted(this.model); // kml predict() guards on its own `fitted` flag
    this.fitted = true;
    return this;
  }

  /**
   * Returns the current model parameters (coef + intercept).
   * The counterpart of `setParams()` (sklearn-style `get_params()`).
   */
  getParams(): Required<PolynomialLearnedParams> {
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
    const payload = {
      version: EXPORT_VERSION,
      degree: this.degree,
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
      degree?: number;
      coef: number[];
      intercept: number;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = new kml.Linear.PolynomialRegression({
      degree: payload.degree,
    });
    this.setParams({ coef: payload.coef, intercept: payload.intercept });
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }

  /**
   * Predicts asynchronously: the JSON → matrix transformation runs on the
   * main thread, then the polynomial expansion + dot product is offloaded to
   * a worker (via kml `utils.asyncMode`).
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
    const degree = this.degree;
    // Self-contained function: no closure over `this` (worker-safe).
    // Expands each feature to powers 1..degree, then X·w + b.
    const run = asyncMode(
      (rows: number[][], w: number[], b: number, deg: number): number[] =>
        rows.map((row) => {
          const expanded: number[] = [];
          for (const x of row) {
            for (let p = 1; p <= deg; p++) expanded.push(x ** p);
          }
          return expanded.reduce((sum, x, i) => sum + x * (w[i] ?? 0), b);
        }),
    );
    return run(X, coef, intercept, degree);
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
}

export { PolynomialRegression };
export type { JsonFitSpec, JsonRow } from "../../@types/json";
