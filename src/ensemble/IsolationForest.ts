import { kml, loadModel, toJSONOf } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { ClusterSpec, JsonRow } from "../types/json";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of an IsolationForest model. */
export interface IsolationForestParams {
  /** Number of samples used to build each tree. Default: 256. */
  subsampling_size?: number;
  /** Number of isolation trees in the forest. Default: 100. */
  tree_num?: number;
  /** Expected proportion of outliers ('auto' or 0..1). Default: 'auto'. */
  contamination?: "auto" | number;
  /** Seed for reproducible training. */
  random_state?: number;
}

/** Serialized shape of the kml IsolationForest (official serializer). */
type SerializedIF = ReturnType<
  InstanceType<typeof kml.Ensemble.IsolationForest>["toJSON"]
>;

/**
 * IsolationForest: unsupervised anomaly detection. Builds random partition
 * trees and flags points that are isolated quickly (short path length) as
 * anomalies. No target field — the JSON spec only selects `features`.
 */
class IsolationForest {
  private model: InstanceType<typeof kml.Ensemble.IsolationForest>;
  private transformer: JsonTransformer | null = null;

  constructor(params?: IsolationForestParams) {
    this.model = new kml.Ensemble.IsolationForest(
      params as ConstructorParameters<typeof kml.Ensemble.IsolationForest>[0],
    );
  }

  /**
   * Fits the model on JSON rows (builds the isolation trees). No target —
   * the spec only selects `features`.
   * @param data - Row objects, e.g. [{ montant, heure, distance }].
   * @param spec - Specification of the features (no target).
   */
  fit(data: JsonRow[], spec: ClusterSpec): void {
    this.transformer = new JsonTransformer(spec);
    const { X } = this.transformer.fitTransform(data);
    this.model.fit(X);
  }

  /**
   * Predicts anomaly labels on rows using the forest learned during fit.
   * @param data - Row objects with the same features fields as at training time.
   * @returns 1 = anomaly, 0 = normal.
   */
  predict(data: JsonRow[]): number[] {
    if (!this.transformer) {
      throw new Error("Call fit before predict.");
    }
    const { X } = this.transformer.transform(data);
    return this.model.predict(X);
  }

  /**
   * Returns a continuous anomaly score for a single row (higher = more
   * anomalous). Useful for ranking and custom thresholds.
   */
  anomaly_score(row: JsonRow): number {
    if (!this.transformer) {
      throw new Error("Call fit before anomaly_score.");
    }
    const { X } = this.transformer.transform([row]);
    return this.model.anomalyScore(X[0]!);
  }

  /** X column names (useful to interpret the anomalies). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last transform. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /**
   * Exports the fitted model (forest + learned transformation) to a
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
      model: SerializedIF;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<InstanceType<typeof kml.Ensemble.IsolationForest>>(
      payload.model,
    );
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }
}

export { IsolationForest };
export type { ClusterSpec, JsonRow } from "../types/json";
