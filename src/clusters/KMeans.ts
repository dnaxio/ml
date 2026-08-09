import { kml, toJSONOf, loadModel } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { ClusterSpec, JsonRow } from "../@types/json";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of a KMeans model. */
export interface KMeansParams {
  /** Number of clusters. Default: 2. */
  n_clusters?: number;
  /** Relative tolerance on inertia change (convergence). Default: 0.05. */
  tol?: number;
  /** Optional user-provided initial centers (forces n_init = 1). */
  initCenters?: number[][];
  /** Maximum Lloyd iterations per run. Default: 30. */
  max_iter?: number;
  /** Seed for reproducible k-means++ initialization. */
  random_state?: number;
  /** Number of k-means++ restarts. Default: 10. */
  n_init?: number;
}

/** Serialized shape of the kml KMeans (official serializer). */
type SerializedKMeans = ReturnType<
  InstanceType<typeof kml.Clusters.KMeans>["toJSON"]
>;

/**
 * KMeans: unsupervised clustering that groups rows into `n_clusters`
 * centroids. No target field — the JSON spec only selects `features`.
 */
class KMeans {
  private model: InstanceType<typeof kml.Clusters.KMeans>;
  private transformer: JsonTransformer | null = null;
  private lastLabels: number[] = [];

  constructor(params?: KMeansParams) {
    this.model = new kml.Clusters.KMeans(params);
  }

  /**
   * Fits the model on JSON rows (builds the centroids). No target —
   * the spec only selects `features`.
   * @param data - Row objects, e.g. [{ age, solde }].
   * @param spec - Specification of the features (no target).
   */
  fit(data: JsonRow[], spec: ClusterSpec): void {
    this.transformer = new JsonTransformer(spec);
    const { X } = this.transformer.fitTransform(data);
    this.lastLabels = this.model.fitPredict(X);
  }

  /**
   * Assigns each row to the nearest centroid learned during `fit`.
   * @param data - Row objects with the same features fields as at fit time.
   * @returns Cluster label (0..n_clusters-1) per row.
   */
  predict(data: JsonRow[]): number[] {
    if (!this.transformer) {
      throw new Error("Call fit before predict.");
    }
    const centroids = this.model.getCentroids();
    if (!centroids || centroids.length === 0) {
      throw new Error("Call fit before predict.");
    }
    const { X } = this.transformer.transform(data);
    return X.map((row) => nearestCentroid(row, centroids));
  }

  /**
   * Convenience: fits and returns the cluster labels for the training rows.
   * Equivalent to `fit` then reading `labels_`.
   * @param data - Row objects, e.g. [{ age, solde }].
   * @param spec - Specification of the features (no target).
   * @returns Cluster label (0..n_clusters-1) per row.
   */
  fit_predict(data: JsonRow[], spec: ClusterSpec): number[] {
    this.fit(data, spec);
    return this.lastLabels;
  }

  /** Cluster labels from the last `fit` (one per training row). */
  get labels_(): number[] {
    return this.lastLabels;
  }

  /** X column names (useful to interpret the centroids). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last fit_predict. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /** Cluster centroids (one row per cluster) or null if not fitted. */
  get centroids(): number[][] | null {
    return this.model.getCentroids();
  }

  /** Within-cluster sum of squares (lower = more compact clusters). */
  get inertia(): number {
    return this.model.getInertia();
  }

  /**
   * Exports the fitted model (centroids + learned transformation) to a
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
      model: SerializedKMeans;
      transformer: JsonTransformerState;
    };
    if (payload.version !== EXPORT_VERSION) {
      throw new Error(
        `Unsupported export format version: ${payload.version} (expected ${EXPORT_VERSION}).`,
      );
    }
    this.model = loadModel<InstanceType<typeof kml.Clusters.KMeans>>(
      payload.model,
    );
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }
}

export { KMeans };
export type { ClusterSpec, JsonRow } from "../@types/json";

/** Returns the index of the nearest centroid (euclidean distance). */
function nearestCentroid(row: number[], centroids: number[][]): number {
  let best = 0;
  let bestDist = Infinity;
  for (let k = 0; k < centroids.length; k++) {
    const c = centroids[k] ?? [];
    let dist = 0;
    for (let i = 0; i < row.length; i++) {
      const d = (row[i] ?? 0) - (c[i] ?? 0);
      dist += d * d;
    }
    if (dist < bestDist) {
      bestDist = dist;
      best = k;
    }
  }
  return best;
}
