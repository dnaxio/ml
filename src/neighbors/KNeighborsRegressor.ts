import { kml, toJSONOf, loadModel } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { JsonFitSpec, JsonRow } from "../types/json";
import { truthValues, meanAbsoluteError } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a KNeighborsRegressor model (both kml and sklearn namings). */
export interface KNeighborsRegressorParams {
  /** Number of nearest neighbors averaged for the prediction. Default: 5. */
  nNeighbors?: number;
  /** Alias of `nNeighbors` (kml naming). */
  kNeighbors?: number;
  /** Neighbor weights: 'uniform' (plain average) or 'distance' (inverse-distance). */
  weights?: "uniform" | "distance";
  /** Alias of `weights`. */
  weightType?: string;
  /** Distance metric name. Default: 'euclidean'. */
  metric?: string;
  /** Alias of `metric`. */
  distanceType?: string;
  /** Minkowski p-norm (only for 'minkowski'). Default: 2. */
  p?: number;
  /** Alias of `p`. */
  pNorm?: number;
}

/** Serialized shape of the kml KNeighborsRegressor (official serializer). */
type SerializedKNNR = ReturnType<
  InstanceType<typeof kml.Neighbors.KNeighborsRegressor>["toJSON"]
>;

/**
 * KNeighborsRegressor: instance-based regression. Each prediction is the
 * (weighted) average of the target values of the `nNeighbors` nearest labeled
 * examples. No model is learned — good as a non-parametric baseline and for
 * very local relationships.
 */
class KNeighborsRegressor {
  private model: InstanceType<typeof kml.Neighbors.KNeighborsRegressor>;
  private transformer: JsonTransformer | null = null;

  constructor(params?: KNeighborsRegressorParams) {
    this.model = new kml.Neighbors.KNeighborsRegressor(
      resolveParams(params) as unknown as ConstructorParameters<typeof kml.Neighbors.KNeighborsRegressor>[0],
    );
  }

  /**
   * Fits the model on JSON rows (memorizes the labeled examples).
   * @param data - Row objects, e.g. [{ age, salaire }].
   * @param spec - Specification of the features and target fields + options.
   */
  fit(data: JsonRow[], spec: JsonFitSpec): void {
    this.transformer = new JsonTransformer(spec);
    const { X, Y } = this.transformer.fitTransform(data);
    this.model.fit(X, Y);
  }

  /**
   * Predicts on JSON rows reusing the transformation learned during train.
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
   * @returns The input rows with the target field set to the prediction.
   */
  fill_predict(data: JsonRow[]): JsonRow[] {
    if (!this.transformer) {
      throw new Error("Call fit before fill_predict.");
    }
    const { X } = this.transformer.transform(data);
    const preds = this.model.predict(X);
    const kept = this.transformer.keptIndices;
    const target = this.transformer.targetField;
    return kept.map((idx, i) => ({ ...data[idx], [target]: preds[i] ?? 0 }));
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

  /** X column names (useful to interpret the neighbors). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last transform. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /**
   * Current kml parameters (`nNeighbors` / `weights` / `metric` / `p`).
   * The counterpart of `setParams()` (sklearn-style `get_params()`).
   */
  getParams(): Record<string, unknown> {
    return this.model.getParams();
  }

  /**
   * Injects parameters (sklearn-style `set_params()`). Accepts both namings
   * (`kNeighbors` / `weightType` / `distanceType` / `pNorm` are aliases).
   * Rebuilds the model **unfitted** — call `fit` again before `predict`.
   * Unknown keys throw. Returns `this` for chaining.
   * @param params - Parameters to set.
   */
  setParams(params: KNeighborsRegressorParams): this {
    this.model.setParams(resolveParams(params) as Record<string, unknown>);
    this.transformer = null; // kml rebuilt the model unfitted → force refit
    return this;
  }

  /**
   * Exports the fitted model (training rows + params + transformation) to a
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
      model: SerializedKNNR;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<InstanceType<typeof kml.Neighbors.KNeighborsRegressor>>(
      payload.model,
    );
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }
}

export { KNeighborsRegressor };
export type { JsonFitSpec, JsonRow } from "../types/json";

/** Maps the SDK params (with kml aliases) to the kml prop names. */
function resolveParams(
  p?: KNeighborsRegressorParams,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const n = p?.nNeighbors ?? p?.kNeighbors;
  if (n !== undefined) out.nNeighbors = n;
  const w = p?.weights ?? p?.weightType;
  if (w !== undefined) out.weights = w;
  const m = p?.metric ?? p?.distanceType;
  if (m !== undefined) out.metric = m;
  const pn = p?.p ?? p?.pNorm;
  if (pn !== undefined) out.p = pn;
  return out;
}
