import { kml, loadModel, toJSONOf, classesOf } from "../core";
import { JsonTransformer } from "../../transformation/json";
import type { JsonTransformerState } from "../../transformation/json";
import type { JsonFitSpec, JsonRow } from "../../@types/json";
import { truthValues, positiveProbabilities } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of an ExtraTreeClassifier model. */
export interface ExtraTreeClassifierParams {
  /** Maximum tree depth (limit to avoid overfitting). Default: Infinity. */
  max_depth?: number;
  /** Minimum samples required to keep splitting. Default: 2. */
  min_samples_split?: number;
  /** Impurity criterion. Default: 'entropy'. */
  criterion?: "entropy" | "gini";
  /** Number of features to consider per split. */
  max_features?: number | "sqrt" | "log2";
  /** Seed for reproducible random feature selection. */
  randomState?: number;
}

/** Serialized shape of the kml ExtraTreeClassifier (official serializer). */
type SerializedTree = ReturnType<
  InstanceType<typeof kml.Tree.ExtraTreeClassifier>["toJSON"]
>;

/**
 * ExtraTreeClassifier: randomized decision tree classifier. More random
 * splits than DecisionTreeClassifier, usually a better baseline. Exposes
 * `predict_proba` and `fill_predict_proba` for per-class probabilities.
 */
class ExtraTreeClassifier {
  private model: InstanceType<typeof kml.Tree.ExtraTreeClassifier>;
  private transformer: JsonTransformer | null = null;
  private fitted = false;

  /**
   * Creates a model.
   * @param params - Model configuration (max_depth, min_samples_split, ...).
   */
  constructor(params?: ExtraTreeClassifierParams) {
    this.model = new kml.Tree.ExtraTreeClassifier(
      params as ConstructorParameters<typeof kml.Tree.ExtraTreeClassifier>[0],
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
   * the predicted label (new objects, the input is not mutated).
   * @param data - Row objects with the same features fields as at training time.
   * @returns The input rows with the target field set to the predicted label.
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
   * Predicts and returns the input rows with the `target` field filled with
   * the probability of the positive class, always a number in [0, 1].
   * - Binary: probability of class `classes[1]`.
   * - Multiclass: highest class probability (confidence).
   */
  fill_predict_proba(data: JsonRow[]): JsonRow[] {
    if (!this.transformer) {
      throw new Error("Call fit before fill_predict_proba.");
    }
    const { X } = this.transformer.transform(data);
    const probas = this.model.predictProba(X);
    const classes = this.model.classes;
    const binary = classes.length === 2;
    const kept = this.transformer.keptIndices;
    const target = this.transformer.targetField;
    return kept.map((idx, i) => {
      const row = probas[i] ?? [];
      const p = binary ? (row[1] ?? 0) : Math.max(...row, 0);
      return { ...data[idx], [target]: p };
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
   * Exports the trained model (tree + learned transformation) to a
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
      model: SerializedTree;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<InstanceType<typeof kml.Tree.ExtraTreeClassifier>>(
      payload.model,
    );
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

export { ExtraTreeClassifier };
export type { JsonFitSpec, JsonRow } from "../../@types/json";
