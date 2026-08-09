import { kml } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { ClusterSpec, JsonRow } from "../@types/json";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a DBSCAN model. */
export interface DBSCANParams {
  /** Neighborhood radius. Default: 0.5. */
  eps?: number;
  /** Minimum neighborhood size (including the point) for a core point. Default: 5. */
  minSamples?: number;
  /** Distance metric name ('euclidean' | 'manhattan' | 'minkowski' | ...). */
  distanceType?: string;
}

/**
 * DBSCAN: density-based unsupervised clustering. Groups dense regions and
 * marks isolated points as noise (label -1). No target field — the JSON
 * spec only selects `features`.
 */
class DBSCAN {
  private model: InstanceType<typeof kml.Clusters.DBSCAN>;
  private transformer: JsonTransformer | null = null;
  private lastLabels: number[] = [];

  constructor(params?: DBSCANParams) {
    this.model = new kml.Clusters.DBSCAN(
      params as ConstructorParameters<typeof kml.Clusters.DBSCAN>[0],
    );
  }

  /**
   * Fits the model on JSON rows. No target — the spec only selects `features`.
   * Cluster labels become available via `labels_`.
   * @param data - Row objects, e.g. [{ lat, lon }].
   * @param spec - Specification of the features (no target).
   */
  fit(data: JsonRow[], spec: ClusterSpec): void {
    this.transformer = new JsonTransformer(spec);
    const { X } = this.transformer.fitTransform(data);
    this.lastLabels = this.model.fitPredict(X);
  }

  /**
   * Convenience: fits and returns the cluster labels for the training rows.
   * -1 means noise (isolated point).
   * @param data - Row objects, e.g. [{ lat, lon }].
   * @param spec - Specification of the features (no target).
   * @returns Cluster label per row; -1 means noise.
   */
  fit_predict(data: JsonRow[], spec: ClusterSpec): number[] {
    this.fit(data, spec);
    return this.lastLabels;
  }

  /**
   * DBSCAN does NOT support predicting labels for unseen points (it is a
   * density-based method recomputed over the whole point set, exactly as in
   * scikit-learn). Use `fit_predict` on the full dataset, or recluster via
   * `fit` then read `labels_`.
   * @throws always
   */
  predict(_data: JsonRow[]): number[] {
    throw new Error(
      "DBSCAN does not support predict on new points; use fit_predict on the full dataset.",
    );
  }

  /** Cluster labels from the last `fit` / `fit_predict` (one per training row; -1 = noise). */
  get labels_(): number[] {
    return this.lastLabels;
  }

  /** X column names (useful to interpret the clusters). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last fit_predict. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /**
   * Exports the learned transformation + config to a `<name>.json` file
   * (async). DBSCAN is stateless (recomputed at each fit_predict), so the
   * config is preserved to reproduce identical clustering.
   * @param name - File name (with or without .json extension).
   */
  async export(name: string): Promise<void> {
    if (!this.transformer) {
      throw new Error("Call fit before export.");
    }
    const filePath = name.endsWith(".json") ? name : `${name}.json`;
    const payload = {
      version: EXPORT_VERSION,
      params: this.model.getParams(),
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
      params?: Record<string, unknown>;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = new kml.Clusters.DBSCAN(
      payload.params as ConstructorParameters<typeof kml.Clusters.DBSCAN>[0],
    );
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }
}

export { DBSCAN };
export type { ClusterSpec, JsonRow } from "../@types/json";
