import { kml, loadModel, toJSONOf } from "../core";
import { JsonTransformer } from "../../transformation/json";
import type { JsonTransformerState } from "../../transformation/json";
import type { JsonFitSpec, JsonRow } from "../../@types/json";
import { truthValues } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a RandomForestRegressor model. */
export interface RandomForestRegressorParams {
  /** Number of decision trees. Default: 100. */
  nEstimators?: number;
  /** Bootstrap sampling with replacement. Default: true. */
  bootstrap?: boolean;
  /** Maximum tree depth (limit to avoid overfitting). */
  maxDepth?: number;
  /** Minimum samples required to keep splitting. */
  minSamplesSplit?: number;
  /** Features to consider per split. Default: 'sqrt'. */
  maxFeatures?: number | "sqrt" | "log2";
  /** Seed for reproducible training. */
  randomState?: number;
}

/** Serialized shape of the kml RandomForestRegressor (official serializer). */
type SerializedRF = ReturnType<
  InstanceType<typeof kml.Ensemble.RandomForestRegressor>["toJSON"]
>;

/**
 * RandomForestRegressor: many decision trees combined by averaging their
 * predictions. More robust and accurate than a single tree. Exposes
 * `featureImportances` instead of `coef` / `intercept`.
 */
class RandomForestRegressor {
  private model: InstanceType<typeof kml.Ensemble.RandomForestRegressor>;
  private transformer: JsonTransformer | null = null;
  private fitted = false;

  /**
   * Creates a model.
   * @param params - Model configuration (nEstimators, maxDepth, ...).
   */
  constructor(params?: RandomForestRegressorParams) {
    this.model = new kml.Ensemble.RandomForestRegressor(
      params as ConstructorParameters<
        typeof kml.Ensemble.RandomForestRegressor
      >[0],
    );
  }

  /**
   * Fits the model on JSON rows.
   * @param data - Array of row objects, e.g. [{ note, heures, resultat }].
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
   * @param data - Row objects with the same features fields as at fit time.
   * @returns Predictions of length data.length.
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
   * @param data - Row objects with the same features fields as at fit time.
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
    const asBoolean = this.transformer.targetBoolean;
    return kept.map((idx, i) => {
      const pred = preds[i] ?? 0;
      return { ...data[idx], [target]: asBoolean ? pred === 1 : pred };
    });
  }

  /** X column names (aligned with `featureImportances`). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last fit/predict. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /** How much each feature contributes to the splits (higher = more important). */
  get featureImportances(): number[] {
    if (!this.fitted) {
      throw new Error("Call fit before accessing featureImportances.");
    }
    return this.model.featureImportances;
  }

  /**
   * Exports the trained model (forest + learned transformation) to a
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
      model: SerializedRF;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<
      InstanceType<typeof kml.Ensemble.RandomForestRegressor>
    >(payload.model);
    this.fitted = true;
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
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

export { RandomForestRegressor };
export type { JsonFitSpec, JsonRow } from "../../@types/json";
