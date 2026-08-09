import { kml } from "../core";
import { JsonTransformer } from "../transformation/json";
import type { JsonTransformerState } from "../transformation/json";
import type { ClusterSpec, JsonRow } from "../@types/json";

/** Current version of the export file format. */
const EXPORT_VERSION = 1;

/** Parameters of an HDBSCAN model. */
export interface HDBSCANParams {
  /** Smallest group of points considered a cluster (clamped to >= 2). Default: 5. */
  min_cluster_size?: number;
  /** Neighborhood size for core distances; defaults to min_cluster_size. */
  min_samples?: number | null;
  /** Clusters split below this distance are merged back together. */
  cluster_selection_epsilon?: number;
  /** Distance metric name. Default: 'euclidean'. */
  metric?: string;
  /** Allow the root of the tree to be selected as a single cluster. */
  allow_single_cluster?: boolean;
}

/**
 * HDBSCAN: hierarchical density-based clustering. Extends DBSCAN by building
 * a hierarchy of densities and selecting the most stable clusters — it finds
 * clusters of **arbitrary shape and varying density**, and labels isolated
 * points as noise (-1). No target field — the JSON spec only selects
 * `features`.
 *
 * Labels are computed on the training rows only (no out-of-sample `predict`,
 * like DBSCAN). `getProbabilities` exposes the membership strength per point
 * (0 = noise).
 */
class HDBSCAN {
  private model: InstanceType<typeof kml.Clusters.HDBSCAN>;
  private transformer: JsonTransformer | null = null;
  private lastLabels: number[] = [];

  constructor(params?: HDBSCANParams) {
    this.model = new kml.Clusters.HDBSCAN(
      params as ConstructorParameters<typeof kml.Clusters.HDBSCAN>[0],
    );
  }

  /**
   * Fits the model on JSON rows (builds the cluster hierarchy). No target —
   * the spec only selects `features`.
   * @param data - Row objects, e.g. [{ x, y }].
   * @param spec - Specification of the features (no target).
   */
  fit(data: JsonRow[], spec: ClusterSpec): void {
    this.transformer = new JsonTransformer(spec);
    const { X } = this.transformer.fitTransform(data);
    this.lastLabels = this.model.fitPredict(X);
  }

  /**
   * Convenience: fits and returns the cluster labels for the training rows.
   * Equivalent to `fit` then reading `labels_`.
   * @param data - Row objects, e.g. [{ x, y }].
   * @param spec - Specification of the features (no target).
   * @returns Cluster label per row; -1 = noise (isolated point).
   */
  fit_predict(data: JsonRow[], spec: ClusterSpec): number[] {
    this.fit(data, spec);
    return this.lastLabels;
  }

  /** Cluster labels from the last `fit` (one per training row; -1 = noise). */
  get labels_(): number[] {
    return this.lastLabels;
  }

  /**
   * Cluster membership strength of each training point from the last `fit`
   * (0..1; noise points have 0). One value per row of the last fit.
   */
  get probabilities(): number[] {
    return this.model.getProbabilities();
  }

  /** X column names (useful to interpret the clusters). */
  get columnNames(): string[] {
    return this.transformer?.columnNames ?? [];
  }

  /** Number of rows removed by the 'drop' strategy in the last fit. */
  get droppedRows(): number {
    return this.transformer?.droppedRows ?? 0;
  }

  /**
   * Exports the learned transformation + config to a `<name>.json` file
   * (async). HDBSCAN is stateful (hierarchy rebuilt at each fit), so the
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
    this.model = new kml.Clusters.HDBSCAN(
      payload.params as ConstructorParameters<typeof kml.Clusters.HDBSCAN>[0],
    );
    this.transformer = JsonTransformer.fromJSON(payload.transformer);
  }
}

export { HDBSCAN };
export type { ClusterSpec, JsonRow } from "../@types/json";
