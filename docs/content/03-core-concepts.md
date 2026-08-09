---
title: Core Concepts
description: The JSON-first spec, value encoding, missing values, scoped imports and the common API surface.
navigation:
  icon: lucide:book-open
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

### Spatial scan — `ScanSpec`

```ts
scan.fit(zones, {
  zone: "zone",
  coordinates: ["lon", "lat"],
  population: "population",
  cases: "cases",
});
```

## Supervised vs unsupervised models

| Category | Models | Spec | Goal | Evaluation |
| -------- | ------ | ---- | ---- | ---------- |
| **Supervised** (21) | **13 regressors**: `LinearRegression`, `RidgeRegression`, `LassoRegression`, `ElasticNet`, `RANSACRegressor`, `PoissonRegressor`, `PolynomialRegression`, `DecisionTreeRegressor`, `ExtraTreeRegressor`, `RandomForestRegressor`, `AdaBoostRegressor`, `GradientBoostingRegressor`, `XGBoostRegressor` · **8 classifiers**: `LogisticRegression`, `RidgeClassifier`, `DecisionTreeClassifier`, `ExtraTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `XGBoostClassifier` | `{ features, target }` | learn Y from X, predict new Y | `score`, `mse`, `mae`, `classificationReport`, `rocAucScore` ✅ |
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
import { linear, clusters, tree, ensemble, monitoring, scan } from "@dnax/ml";

const reg = new linear.LinearRegression();
const km = new clusters.KMeans({ n_clusters: 3 });
const dt = new tree.DecisionTreeClassifier();
const iso = new ensemble.IsolationForest();
const cusum = new monitoring.CUSUM();
const scanModel = new scan.SpatialScan();

// flat imports also work:
import { LinearRegression, KMeans } from "@dnax/ml";
```

## The common API surface

Every supervised model exposes:

- `fit(data, spec)` — train
- `predict(data)` → `number[]` — predict
- `fill_predict(data)` → `JsonRow[]` — predict **and** fill the `target`
  field into each row (new objects, the input is not mutated). A boolean
  target produces `true`/`false`, a numeric one stays numeric.
- `export(name)` / `load(name)` — async persistence to `<name>.json`

Getters you can rely on:

- `columnNames` — X column names (interpret `coef`)
- `droppedRows` — rows removed by the `'drop'` missing strategy

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
