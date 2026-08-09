---
title: API Reference
description: Every model class, constructor parameter, method and getter — the complete reference for all algorithms.
navigation:
  icon: lucide:list-tree
---

Every model follows the same contract: `fit(data, spec)` → `predict(data)` →
`fill_predict(data)` → `export(name)` / `load(name)`, plus `score(data)` on
supervised models. Getters `columnNames` and `droppedRows` are common to all.

## Linear family

Common surface: `fit`, `predict`, `fill_predict`, `predictAsync`, `score`
(R²), `mse`, `mae`, `getParams()`, `setParams()`, `export`, `load`, `coef`,
`intercept`.

| Class | Constructor params | Notes |
| ----- | ------------------ | ----- |
| `LinearRegression` | — | ordinary least squares |
| `RidgeRegression` | `alpha` (1) | L2 regularization |
| `LassoRegression` | `alpha` (1) | L1, zeroes uninformative features |
| `ElasticNet` | `alpha` (1), `l1Ratio` (0.5) | L1 + L2 |
| `LogisticRegression` | — | + `predict_proba`, `fill_predict_proba`, `rocAucScore`; binary & multiclass |
| `RidgeClassifier` | `alpha` (1) | labels only, no `predict_proba` |
| `RANSACRegressor` | `randomState` | + `inlierMask`, `nTrials` getters |
| `PoissonRegressor` | `alpha` (1) | counts, predictions `>= 0`; + `nIter` |
| `PolynomialRegression` | `degree` (2) | + `degree` getter |

```ts
const reg = new linear.RidgeRegression({ alpha: 1 });
reg.fit(data, { features: ["x"], target: "y" });
reg.score(test);        // R²
reg.mse(test);          // MSE
reg.mae(test);          // MAE
await reg.predictAsync(big); // worker offload
```

## Tree family

Common surface: `fit`, `predict`, `fill_predict`, `score`, `export`, `load`,
`featureImportances`.

| Class | Params | Notes |
| ----- | ------ | ----- |
| `DecisionTreeClassifier` | `max_depth`, `min_samples_split`, `criterion`, `max_features`, `randomState` | no `predict_proba` |
| `DecisionTreeRegressor` | same | + `mse`, `mae` |
| `ExtraTreeClassifier` | same | + `predict_proba`, `fill_predict_proba`, `rocAucScore` |
| `ExtraTreeRegressor` | same | + `mse`, `mae` |

## Ensemble family

Common surface: `fit`, `predict`, `fill_predict`, `score`, `export`, `load`,
`featureImportances`.

| Class | Key params | Notes |
| ----- | ---------- | ----- |
| `IsolationForest` | `subsampling_size`, `tree_num`, `contamination`, `random_state` | unsupervised anomaly; `predict` (1 = anomaly) + `anomaly_score(row)` |
| `RandomForestClassifier` | `nEstimators`, `maxDepth`, `maxFeatures`, `criterion`, `randomState` | no `predict_proba` |
| `RandomForestRegressor` | same | + `mse`, `mae` |
| `AdaBoostClassifier` | `nEstimators`, `learningRate`, `randomState` | + proba + `rocAucScore` |
| `AdaBoostRegressor` | same | + `mse`, `mae` |
| `GradientBoostingClassifier` | `nEstimators`, `learningRate`, `maxDepth`, `minSamplesSplit`, `subsample`, `maxFeatures`, `randomState` | + proba + `rocAucScore` |
| `GradientBoostingRegressor` | same | + `mse`, `mae` |
| `XGBoostClassifier` | `nEstimators`, `learningRate`, `maxDepth`, `lambda`, `gamma`, `minChildWeight`, `subsample`, `colsampleByTree`, `baseScore`, `randomState` | + proba + `rocAucScore` |
| `XGBoostRegressor` | same | + `mse`, `mae` |

## Clustering family

Spec has **no target** — only `features`.

| Class | Params | Methods & getters |
| ----- | ------ | ----------------- |
| `KMeans` | `n_clusters`, `tol`, `max_iter`, `initCenters`, `random_state` | `fit`, `predict` (nearest centroid), `fit_predict`, `export`, `load` · `labels_`, `centroids`, `inertia` |
| `DBSCAN` | `eps`, `minSamples`, `distanceType` | `fit`, `fit_predict`, `export`, `load` · `labels_` (`-1` = noise). No `predict` on new points |
| `HDBSCAN` | `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `metric`, `allow_single_cluster` | `fit`, `fit_predict`, `export`, `load` · `labels_`, `probabilities` (0 = noise). No `predict` on new points |

## Monitoring family

`MonitorSpec` = `{ field, missing? }`. Common surface: `fit`, `predict`,
`scores`, `fill_predict`, `export`, `load`.

| Class | Params | Extra |
| ----- | ------ | ----- |
| `CUSUM` | `target`, `std`, `k` (0.5σ), `h` (5σ), `direction` (`'increase'`\|`'decrease'`\|`'both'`), `robust`, `alertField` | `changePoint(data)`, `update(row)`, `reset()` · getters `target`, `std`, `slack`, `threshold`, `direction` |
| `EWMA` | `lambda` (0.25), `limit` (3), `target`, `std`, `robust`, `alertField` | `limits(data)`, `update(row)`, `reset()` · getters `target`, `std`, `lambda`, `limit` |
| `ParallelMonitor` | `model` (`'cusum'`\|`'ewma'`), `familyError`, `alertField`, `alertFieldsField` | `ParallelSpec` = `{ fields, params?, missing? }` · `alertFields(data)`, `update(row)` · getters `monitoredFields`, `targets`, `params` |
| `SeasonalMonitor` | `model`, `alertField`, `chart` | `SeasonalSpec` = `{ field, dateField, missing? }` · `changePoint`, `limits` (per model), `update(row)` · getter `dayProfile` |

## Spatial scan

`ScanSpec` = `{ zone, coordinates: [xField, yField], population, cases }` ·
`HotspotSpec` = `{ zone, coordinates: [xField, yField], cases }`.

| Class | Params | Methods & getters |
| ----- | ------ | ----------------- |
| `SpatialScan` | `replications` (199), `significance` (0.05), `maxWindowFraction` (0.5), `randomState`, `clusterField` | `fit`, `cluster(data)` → `ScanCluster \| null`, `predict`, `fill_predict`, `export`, `load` · getters `zonesList`, `expectedRate` |
| `GetisOrd` | `distance`, `significance` (0.05), `hotField` | `fit`, `hotspots(data)` → `HotspotResult[]`, `predict`, `fill_predict`, `export`, `load` · getters `zonesList`, `distance`, `significance` |

`ScanCluster` = `{ zones: string[], cases, expected, llr, pValue }` ·
`HotspotResult` = `{ zone, zScore, pValue, hot, cold }`.

## Evaluation

Functions exported from the package root:

```ts
trainTestSplit(data, { testSize?, trainSize?, shuffle?, randomState?, stratify? });
// → { train: JsonRow[], test: JsonRow[] }

crossValScore(() => new LinearRegression(), data, spec, { cv?, scoring?, stratify?, randomState? });
// → number[] — one score per fold

// Streaming inference (memory bounded, works with generators/async iterables)
for await (const pred of predictStream(model, rows, { chunkSize? })) { /* ... */ }
await fillPredictStream(model, rows, (row) => { /* ... */ }, { chunkSize? });
```

Supervised models add:

| Method | Returns | Availability |
| ------ | ------- | ------------ |
| `score(data)` | `number` — R² (regressors) / accuracy (classifiers) | all 21 supervised models |
| `mse(data)` | `number` — mean squared error | 13 regressors |
| `mae(data)` | `number` — mean absolute error | 13 regressors |
| `classificationReport(data)` | `{ accuracy, precision, recall, fScore, support, confusionMatrix }` | 8 classifiers |
| `rocAucScore(data)` | `number` — AUC (binary) | 5 classifiers with `predict_proba` |

## Spec types

```ts
// Supervised
interface JsonFitSpec { features: string[]; target: string; options?: JsonTransformOptions }

// Unsupervised
interface ClusterSpec { features: string[]; options?: JsonTransformOptions }
```

> **Supervised vs unsupervised**: supervised models (`JsonFitSpec`, 21 models)
> have a `target` and expose `score`/`mse`/`mae`/`classificationReport`. Unsupervised
> models take only `features` (`ClusterSpec`) and have no `score` — see
> [Core concepts](/03-core-concepts) for the full classification.

```ts
// Monitoring
interface MonitorSpec { field: string; missing?: "throw" | "fill" }
interface ParallelSpec { fields: string[]; params?: Record<string, unknown>; missing?: "throw" | "fill" }
interface SeasonalSpec { field: string; dateField: string; missing?: "throw" | "fill" }

// Spatial scan
interface ScanSpec {
  zone: string;
  coordinates: [string, string];
  population: string;
  cases: string;
}

// Getis-Ord Gi* hotspot analysis
interface HotspotSpec {
  zone: string;
  coordinates: [string, string];
  cases: string;
}

// Options
interface JsonTransformOptions {
  oneHot?: boolean;      // default false
  dropFirst?: boolean;   // default true
  missing?: "throw" | "fill0" | "drop";  // default 'throw'
  scale?: boolean;       // default false
  noise?: number;        // Gaussian jittering, training only
  noiseSeed?: number;    // reproducibility
}
```
