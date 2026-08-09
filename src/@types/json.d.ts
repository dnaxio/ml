import type { CUSUMParams } from "../monitoring/CUSUM";
import type { EWMAParams } from "../monitoring/EWMA";

/** A JSON data row (row object). */
export type JsonRow = Record<string, unknown>;

/** Options for transforming JSON values into a matrix. */
export interface JsonTransformOptions {
  /**
   * If true, categorical fields (non-numeric strings) are one-hot encoded
   * (one column per category learned at training time).
   * If false (default), a non-numeric string throws an error.
   */
  oneHot?: boolean;
  /**
   * In one-hot, drop the first category (reference) to avoid collinearity
   * (dummy variable trap). Default: true.
   */
  dropFirst?: boolean;
  /** Handling of missing values (undefined / null). Default: 'throw'. */
  missing?: "throw" | "drop" | "fill0";
  /**
   * If true, standardize the features (StandardScaler: mean 0, unit variance)
   * before training/prediction. Default: false.
   */
  scale?: boolean;
  /**
   * Standard deviation of Gaussian noise added to continuous feature values
   * during training only (data augmentation / jittering). Applied after
   * scaling, so with `scale: true` a value of 0.05 means 5% of a column
   * standard deviation. One-hot columns are never perturbed. Inference
   * (`transform`) stays deterministic. Useful as a light regularizer for
   * linear models; not recommended for trees or IsolationForest.
   * Default: no noise.
   */
  noise?: number;
  /**
   * Seed for the noise generator (mulberry32). Same seed → same noise, so
   * fits are reproducible. Default: random per fit.
   */
  noiseSeed?: number;
}

/** Specification for jsonFit / jsonPredict transformation. */
export interface JsonFitSpec {
  /** Fields to use as features (X). One column per field, in this order. */
  features: string[];
  /** Output field (Y). A single output for regression. */
  target: string;
  /** Transformation options. */
  options?: JsonTransformOptions;
}

/** Specification for unsupervised clustering (no target field). */
export interface ClusterSpec {
  /** Fields to use as features (X). One column per field, in this order. */
  features: string[];
  /** Transformation options. */
  options?: JsonTransformOptions;
}

/** Specification for time-series monitoring (univariate: one value field). */
export interface MonitorSpec {
  /** Field holding the numeric series to monitor. */
  field: string;
  /**
   * Handling of absent / non-numeric values. 'throw' (default) throws;
   * 'fill' carries the last known value forward (LOCF), keeping the series
   * aligned with the input rows.
   */
  missing?: "throw" | "fill";
}

/** Specification for monitoring several series at once (ParallelMonitor). */
export interface ParallelSpec {
  /** Fields to monitor — one univariate chart per field. */
  fields: string[];
  /**
   * Optional per-field params:
   * - cusum: { target?, std?, k?, h?, direction?, robust? }
   * - ewma:  { lambda?, limit?, target?, std?, robust? }
   */
  params?: Record<string, CUSUMParams | EWMAParams>;
  /**
   * Handling of absent / non-numeric values, applied to every chart
   * ('throw' default | 'fill' carry-forward).
   */
  missing?: "throw" | "fill";
}

/** Specification for deseasonalized (weekly) monitoring (SeasonalMonitor). */
export interface SeasonalSpec {
  /** Field holding the numeric series to monitor. */
  field: string;
  /**
   * Field holding a date (Date, ISO string or timestamp) — used to compute
   * the day of week and remove the weekly cycle.
   */
  dateField: string;
  /** Handling of absent / non-numeric values ('throw' | 'fill' carry-forward). */
  missing?: "throw" | "fill";
}

/** Specification for a spatial scan (Kulldorff / SaTScan). */
export interface ScanSpec {
  /** Field identifying the spatial zone (unique id per row). */
  zone: string;
  /** Two fields with the zone coordinates, e.g. ["x", "y"] or ["lon", "lat"]. */
  coordinates: [string, string];
  /** Field with the population at risk per zone. */
  population: string;
  /** Field with the case count per zone (baseline at fit, current at scan). */
  cases: string;
}

/** Specification for a Getis-Ord Gi* hotspot analysis. */
export interface HotspotSpec {
  /** Field identifying the spatial zone (unique id per row). */
  zone: string;
  /** Two fields with the zone coordinates, e.g. ["x", "y"] or ["lon", "lat"]. */
  coordinates: [string, string];
  /** Field with the case count per zone (the variable of interest). */
  cases: string;
}

/** A cluster detected by the spatial scan. */
export interface ScanCluster {
  /** Zone identifiers inside the cluster. */
  zones: string[];
  /** Observed cases in the cluster. */
  cases: number;
  /** Expected cases in the cluster (population × global case rate). */
  expected: number;
  /** Log-likelihood ratio (larger = stronger cluster). */
  llr: number;
  /** Empirical Monte-Carlo p-value (≤ significance = significant). */
  pValue: number;
}

/** Common shape accepted by the transformer (target optional). */
export interface TransformerSpec {
  features: string[];
  target?: string;
  options?: JsonTransformOptions;
}

/** Result of a transformation: feature matrix + output vector. */
export interface JsonTransformResult {
  X: number[][];
  Y: number[];
}
