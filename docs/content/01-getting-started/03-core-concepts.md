---
title: Core Concepts
description: The JSON-first spec, value encoding, missing values, scoped imports and the common API surface.
---

## The idea

Every model in `@dnax/ml` follows the same contract:

```ts
model.fit(data, spec);       // learn from JSON rows
model.predict(data);         // predict on new rows
model.fill_predict(data);    // predict + add the result into each row
```

`data` is an **array of row objects** and `spec` describes which fields to
use. There is never a manual matrix conversion.

## The spec

### Supervised — `JsonFitSpec`

```ts
reg.fit(data, {
  features: ["note", "admis"], // fields used as features (X)
  target: "note",              // single output field (Y)
  options: {
    oneHot: false,   // encode categorical strings as columns (default: false)
    dropFirst: true, // drop first category in one-hot (default: true)
    missing: "throw",// 'throw' | 'fill0' | 'drop' (default: 'throw')
    scale: false,    // standardize features (StandardScaler) (default: false)
    noise: 0.05,     // Gaussian jittering, training only (default: none)
    noiseSeed: 42,   // seed for the noise generator (default: random)
  },
});
```

### Unsupervised — `ClusterSpec`

Clustering has **no target** — the spec only selects features:

```ts
km.fit(data, { features: ["age", "solde"] });
```

### Monitoring — `MonitorSpec`

```ts
cusum.fit(data, { field: "ventes" });               // one series
cusum.fit(data, { field: "ventes", missing: "fill" }); // carry-forward gaps
```

### Parallel monitoring — `ParallelSpec`

```ts
pm.fit(days, { fields: ["paracetamol", "ibuprofen"], missing: "fill" });
// per-field chart overrides: { fields, params: { paracetamol: { k: 1, h: 10 } } }
```

### Seasonal monitoring — `SeasonalSpec`

```ts
sm.fit(days, { field: "ventes", dateField: "date" }); // weekly profile
```

### Spatial scan — `ScanSpec`

```ts
scan.fit(zones, {
  zone: "zone",
  coordinates: ["lon", "lat"],
  population: "population",
  cases: "cases",
});
```

### Getis-Ord hotspots — `HotspotSpec`

Same geometry but **no population** — works on counts or rates:

```ts
gi.fit(zones, { zone: "zone", coordinates: ["lon", "lat"], cases: "cases" });
```

## Supervised vs unsupervised models

| Category | Models | Spec | Goal | Evaluation |
| -------- | ------ | ---- | ---- | ---------- |
| **Supervised** (23) | **14 regressors**: `LinearRegression`, `RidgeRegression`, `LassoRegression`, `ElasticNet`, `RANSACRegressor`, `PoissonRegressor`, `PolynomialRegression`, `DecisionTreeRegressor`, `ExtraTreeRegressor`, `RandomForestRegressor`, `AdaBoostRegressor`, `GradientBoostingRegressor`, `XGBoostRegressor`, `KNeighborsRegressor` · **9 classifiers**: `LogisticRegression`, `RidgeClassifier`, `DecisionTreeClassifier`, `ExtraTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `XGBoostClassifier`, `KNeighborsClassifier` | `{ features, target }` | learn Y from X, predict new Y | `score`, `mse`, `mae`, `classificationReport`, `rocAucScore` ✅ |
| **Unsupervised** (4) | `KMeans`, `DBSCAN`, `HDBSCAN` (clustering) · `IsolationForest` (anomaly detection) | `{ features }` only — no target | find groups / flag anomalies | ❌ no `score` |
| **Monitoring** (4) | `CUSUM`, `EWMA`, `ParallelMonitor`, `SeasonalMonitor` | `{ field }` — one series | detect when a series drifts from its normal | `update(row)` for real-time |
| **Spatial scan** (2) | `SpatialScan` (Kulldorff) · `GetisOrd` (Gi* hotspots) | `{ zone, coordinates, population, cases }` / `{ zone, coordinates, cases }` | detect statistically significant spatial clusters / hotspots | `cluster(data)` → p-value · `hotspots(data)` → z-scores |

The rule of thumb: **supervised** models have a `target` in the spec and can be
scored against ground truth; **unsupervised** models only take `features` and
have no `score`. Monitoring and spatial scan have no target either — they
answer different questions (time drift / spatial clusters).

## Value encoding

| JSON value              | Encoded as                                           |
| ----------------------- | ---------------------------------------------------- |
| `number`                | as-is                                                |
| `boolean`               | `1` / `0`                                            |
| numeric string (`"42"`) | `Number("42")`                                       |
| non-numeric string      | one-hot column(s) if `oneHot: true`, otherwise error |
| `undefined` / `null`    | depends on `missing`                                 |

Fields not selected in `features` / `target` are ignored.

Features are sorted **alphabetically** (and one-hot categories as well), so
permuting the `features` array has no effect on the column layout — the same
data always produces the same matrix, coefficients, and predictions.

## Missing-value strategies

| Option              | Behavior                                         |
| ------------------- | ------------------------------------------------ |
| `'throw'` (default) | throws an explicit error                         |
| `'fill0'`           | fills the column with `0`, the row is kept       |
| `'drop'`            | removes the whole row (tracked in `droppedRows`) |

## Scoped imports

All models are available as **flat** named exports and grouped by **family**
(kml-style namespaces):

```ts
import { linear, clusters, tree, ensemble, monitoring, scan, neighbors, featureSelection, evaluation } from "@dnax/ml";

const reg = new linear.LinearRegression();
const km = new clusters.KMeans({ n_clusters: 3 });
const dt = new tree.DecisionTreeClassifier();
const iso = new ensemble.IsolationForest();
const cusum = new monitoring.CUSUM();
const scanModel = new scan.SpatialScan();
const knn = new neighbors.KNeighborsClassifier();
evaluation.mcc(preds, truth); // evaluation helpers (compareModels, rmse, ...)

// flat imports also work:
import { LinearRegression, KMeans } from "@dnax/ml";
```

## The common API surface

Every model shares the same core loop, and each family layers its own extras
on top.

### Universal (every model)

- `fit(data, spec)` — train
- `predict(data)` → `number[]` — predict
- `fill_predict(data)` → `JsonRow[]` — predict **and** fill the result into
  each row (new objects, the input is not mutated; a boolean target produces
  `true`/`false`, a numeric one stays numeric)
- `export(name)` / `load(name)` — async persistence to `<name>.json`
- getters `columnNames` (X column names — interpret `coef`) and `droppedRows`
  (rows removed by the `'drop'` missing strategy)

### Supervised evaluation

Scored on rows that include the `target` field (ground truth), reusing the
transformation learned at fit time:

| Method | Returns | Availability |
| ------ | ------- | ------------ |
| `score(data)` | R² (regressors) / accuracy (classifiers) | all 23 supervised models |
| `mse(data)` / `mae(data)` | mean squared / absolute error | 14 regressors |
| `classificationReport(data, beta?)` | precision / recall / Fβ / confusion matrix | 9 classifiers |
| `rocAucScore(data)` | AUC (binary) | 5 models with `predict_proba` |

### Linear & neighbors

- `getParams()` / `setParams()` — read / inject learned parameters
  (sklearn-style; `setParams` rebuilds the model unfitted — refit before
  `predict`)
- `predictAsync(data)` — offload the matrix-vector product to a worker
  (8 linear models — all but `LogisticRegression`)

### Probability models

`LogisticRegression`, `ExtraTreeClassifier`, `AdaBoostClassifier`,
`GradientBoostingClassifier` and `XGBoostClassifier` add
`predict_proba(data)` → `number[][]` and `fill_predict_proba(data)`.

### Trees & ensembles

- `featureImportances` — how much each feature drives the splits (all tree
  and ensemble models **except** `IsolationForest`)
- `IsolationForest` adds `anomaly_score(row)` (continuous); its `predict`
  returns 1 = anomaly

### Clustering

- `fit_predict(data)` — labels for the training rows in one call
- getters `labels_`, `centroids`, `inertia` (KMeans), `probabilities`
  (HDBSCAN, 0 ≈ noise)
- `DBSCAN` / `HDBSCAN` have **no `predict`** on new points

### Monitoring

- `predict` → alerts per point · `scores` → the raw statistic
- `changePoint(data)` (CUSUM) · `limits(data)` (EWMA) · `alertFields(data)`
  (ParallelMonitor) · `dayProfile` (SeasonalMonitor)
- `update(row)` / `reset()` — online monitoring, one row at a time

### Spatial scan

- `cluster(data)` → `ScanCluster | null` (SpatialScan)
- `hotspots(data)` → `HotspotResult[]` (GetisOrd)
- `predict` / `fill_predict` flag the significant zones

### Streaming & evaluation helpers

Memory-bounded inference and model comparison, exported from the package
root:

```ts
import { trainTestSplit, crossValScore, compareModels, detectTask, predictStream, fillPredictStream } from "@dnax/ml";

const { train, test } = trainTestSplit(data, { testSize: 0.3, randomState: 42 });
const scores = crossValScore(() => new LinearRegression(), data, spec, { cv: 5 });
const ranking = compareModels({ linear: () => new LinearRegression() }, data, spec, { cv: 5 });
detectTask(data, spec); // "classification" | "regression"

for await (const pred of predictStream(model, rows)) { /* one per row */ }
await fillPredictStream(model, rows, (row) => { /* enriched row */ });
```

Plus a pure metric library (`rmse`, `mape`, `medianAbsoluteError`, `mcc`,
`balancedAccuracy`, `logLoss`, `prAucScore`, `rocCurve`, `optimalThreshold`,
`meanAbsoluteError`, `fbetaFromPrecisionRecall`) and feature selection
(`featureSelection.chi2` / `fClassif` / `mutualInfoClassif` /
`mutualInfoRegression` / `SelectFromModel`).

## Noise (data jittering)

`options.noise` adds **Gaussian noise** `N(0, noise)` to continuous feature
values **during training only** — a light data augmentation / regularizer:

- Applied **after scaling**, so with `scale: true` a value of `0.05` means 5%
  of a column standard deviation.
- **One-hot columns are never perturbed**.
- **Inference stays deterministic**: `predict` never adds noise.
- `noiseSeed` makes the noise reproducible; the config survives `export`/`load`.

```ts
const robust = new LinearRegression();
robust.fit(data, {
  features: ["prix_affiche", "quantite"],
  target: "prix_facture",
  options: { scale: true, noise: 0.05, noiseSeed: 42 },
});
```

**When to use it**: linear models (regularizer against memorization). It is
**neutral for trees/boosting** and **counter-productive for IsolationForest**
(it blurs the anomalies). For KMeans/DBSCAN it can serve as a stability
diagnostic.
