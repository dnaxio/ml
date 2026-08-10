import { kml, toJSONOf, loadModel } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { JsonFitSpec, JsonRow } from "../types/json";
import { truthValues, fbetaFromPrecisionRecall } from "../evaluation";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a KNeighborsClassifier model (both kml and sklearn namings). */
export interface KNeighborsClassifierParams {
  /** Number of nearest neighbors used for the vote. Default: 5. */
  kNeighbors?: number;
  /** Alias of `kNeighbors` (sklearn naming). */
  nNeighbors?: number;
  /** Neighbor weights: 'uniform' (majority vote) or 'distance' (inverse-distance). */
  weightType?: string;
  /** Alias of `weightType`. */
  weights?: "uniform" | "distance";
  /** Distance metric name. Default: 'euclidean'. */
  distanceType?: string;
  /** Alias of `distanceType`. */
  metric?: string;
  /** Minkowski p-norm (only for 'minkowski'). Default: 2. */
  pNorm?: number;
  /** Alias of `pNorm`. */
  p?: number;
}

/** Serialized shape of the kml KNeighborsClassifier (official serializer). */
type SerializedKNN = ReturnType<
  InstanceType<typeof kml.Neighbors.KNeighborsClassifier>["toJSON"]
>;

/**
 * KNeighborsClassifier: instance-based (non-parametric) classification. Each
 * point is classified by majority vote of its `kNeighbors` nearest labeled
 * examples — no model is learned, the training rows are memorized. A simple
 * and strong baseline: if kNN beats your parametric models, the relationship
 * is very local.
 */
class KNeighborsClassifier {
  private model: InstanceType<typeof kml.Neighbors.KNeighborsClassifier>;
  private transformer: JsonTransformer | null = null;

  constructor(params?: KNeighborsClassifierParams) {
    this.model = new kml.Neighbors.KNeighborsClassifier(
      resolveParams(params) as unknown as ConstructorParameters<
        typeof kml.Neighbors.KNeighborsClassifier
      >[0],
    );
  }

  /**
   * Fits the model on JSON rows (memorizes the labeled examples).
   * @param data - Row objects, e.g. [{ age, solde, achete }].
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
   * the predicted label (new objects, the input is not mutated).
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
    const asBoolean = this.transformer.targetBoolean;
    return kept.map((idx, i) => {
      const pred = preds[i] ?? 0;
      return { ...data[idx], [target]: asBoolean ? pred === 1 : pred };
    });
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

  /** X column names (useful to interpret the neighbors). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last transform. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /**
   * Sorted class labels learned at fit time (sklearn-style `classes_`).
   * kml's KNN stores the training targets instead of a `classes` field, so
   * they are derived from the memorized `trainY`.
   */
  get classes(): number[] {
    const trainY = (this.model as unknown as { trainY?: number[] }).trainY;
    if (!trainY || trainY.length === 0) {
      throw new Error("Call fit before accessing classes.");
    }
    return [...new Set(trainY)].sort((a, b) => a - b);
  }

  /**
   * Current kml parameters (`kNeighbors` / `weightType` / `distanceType` /
   * `pNorm`). The counterpart of `setParams()` (sklearn-style `get_params()`).
   */
  getParams(): Record<string, unknown> {
    return this.model.getParams();
  }

  /**
   * Injects parameters (sklearn-style `set_params()`). Accepts both namings
   * (`nNeighbors` / `weights` / `metric` / `p` are aliases). Rebuilds the
   * model **unfitted** — call `fit` again before `predict`. Unknown keys
   * throw. Returns `this` for chaining.
   * @param params - Parameters to set.
   */
  setParams(params: KNeighborsClassifierParams): this {
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
      model: SerializedKNN;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<InstanceType<typeof kml.Neighbors.KNeighborsClassifier>>(
      payload.model,
    );
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }
}

export { KNeighborsClassifier };
export type { JsonFitSpec, JsonRow } from "../types/json";

/** Maps the SDK params (with sklearn aliases) to the kml prop names. */
function resolveParams(
  p?: KNeighborsClassifierParams,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const k = p?.kNeighbors ?? p?.nNeighbors;
  if (k !== undefined) out.kNeighbors = k;
  const w = p?.weightType ?? p?.weights;
  if (w !== undefined) out.weightType = w;
  const d = p?.distanceType ?? p?.metric;
  if (d !== undefined) out.distanceType = d;
  const n = p?.pNorm ?? p?.p;
  if (n !== undefined) out.pNorm = n;
  return out;
}
