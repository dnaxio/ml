import { kml, loadModel, toJSONOf } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { JsonFitSpec, JsonRow } from "../types/json";
import { truthValues, fbetaFromPrecisionRecall } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a RandomForestClassifier model. */
export interface RandomForestClassifierParams {
  /** Number of decision trees. Default: 100. */
  nEstimators?: number;
  /** Bootstrap sampling with replacement. Default: true. */
  bootstrap?: boolean;
  /** Features to consider per split. Default: 'sqrt'. */
  maxFeatures?: number | "sqrt" | "log2";
  /** Maximum tree depth (limit to avoid overfitting). */
  max_depth?: number;
  /** Minimum samples required to keep splitting. */
  min_samples_split?: number;
  /** Impurity criterion. Default: 'entropy'. */
  criterion?: "entropy" | "gini";
  /** Seed for reproducible training. */
  randomState?: number;
}

/** Serialized shape of the kml RandomForestClassifier (official serializer). */
type SerializedRF = ReturnType<
  InstanceType<typeof kml.Ensemble.RandomForestClassifier>["toJSON"]
>;

/**
 * RandomForestClassifier: many decision trees combined by majority vote.
 * More robust and accurate than a single tree. Exposes `featureImportances`
 * instead of `coef` / `intercept`.
 */
class RandomForestClassifier {
  private model: InstanceType<typeof kml.Ensemble.RandomForestClassifier>;
  private transformer: JsonTransformer | null = null;
  private fitted = false;

  /**
   * Creates a model.
   * @param params - Model configuration (nEstimators, max_depth, ...).
   */
  constructor(params?: RandomForestClassifierParams) {
    this.model = new kml.Ensemble.RandomForestClassifier(
      params as ConstructorParameters<
        typeof kml.Ensemble.RandomForestClassifier
      >[0],
    );
  }

  /**
   * Fits the model on JSON rows.
   * @param data - Array of row objects, e.g. [{ age, solde, achete }].
   * @param spec - Specification of the features and target fields + options.
   */
  fit(data: JsonRow[], spec: JsonFitSpec): void {
    this.transformer = new JsonTransformer(spec);
    const { X, Y } = this.transformer.fitTransform(data);
    this.model.fit(X, Y);
    this.fitted = true;
  }

  /**
   * Predicts class labels on JSON rows. A boolean target is encoded
   * `false` → 0, `true` → 1.
   * @param data - Row objects with the same features fields as at fit time.
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
      InstanceType<typeof kml.Ensemble.RandomForestClassifier>
    >(payload.model);
    this.fitted = true;
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
}

export { RandomForestClassifier };
export type { JsonFitSpec, JsonRow } from "../types/json";
