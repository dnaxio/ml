# Core Concepts

> The JSON-first spec, value encoding, missing values, scoped imports and the common API surface.

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

<table>
<thead>
  <tr>
    <th>
      Category
    </th>
    
    <th>
      Models
    </th>
    
    <th>
      Spec
    </th>
    
    <th>
      Goal
    </th>
    
    <th>
      Evaluation
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <strong>
        Supervised
      </strong>
      
       (23)
    </td>
    
    <td>
      <strong>
        14 regressors
      </strong>
      
      : <code>
        LinearRegression
      </code>
      
      , <code>
        RidgeRegression
      </code>
      
      , <code>
        LassoRegression
      </code>
      
      , <code>
        ElasticNet
      </code>
      
      , <code>
        RANSACRegressor
      </code>
      
      , <code>
        PoissonRegressor
      </code>
      
      , <code>
        PolynomialRegression
      </code>
      
      , <code>
        DecisionTreeRegressor
      </code>
      
      , <code>
        ExtraTreeRegressor
      </code>
      
      , <code>
        RandomForestRegressor
      </code>
      
      , <code>
        AdaBoostRegressor
      </code>
      
      , <code>
        GradientBoostingRegressor
      </code>
      
      , <code>
        XGBoostRegressor
      </code>
      
      , <code>
        KNeighborsRegressor
      </code>
      
       · <strong>
        9 classifiers
      </strong>
      
      : <code>
        LogisticRegression
      </code>
      
      , <code>
        RidgeClassifier
      </code>
      
      , <code>
        DecisionTreeClassifier
      </code>
      
      , <code>
        ExtraTreeClassifier
      </code>
      
      , <code>
        RandomForestClassifier
      </code>
      
      , <code>
        AdaBoostClassifier
      </code>
      
      , <code>
        GradientBoostingClassifier
      </code>
      
      , <code>
        XGBoostClassifier
      </code>
      
      , <code>
        KNeighborsClassifier
      </code>
    </td>
    
    <td>
      <code>
        { features, target }
      </code>
    </td>
    
    <td>
      learn Y from X, predict new Y
    </td>
    
    <td>
      <code>
        score
      </code>
      
      , <code>
        mse
      </code>
      
      , <code>
        mae
      </code>
      
      , <code>
        classificationReport
      </code>
      
      , <code>
        rocAucScore
      </code>
      
       ✅
    </td>
  </tr>
  
  <tr>
    <td>
      <strong>
        Unsupervised
      </strong>
      
       (4)
    </td>
    
    <td>
      <code>
        KMeans
      </code>
      
      , <code>
        DBSCAN
      </code>
      
      , <code>
        HDBSCAN
      </code>
      
       (clustering) · <code>
        IsolationForest
      </code>
      
       (anomaly detection)
    </td>
    
    <td>
      <code>
        { features }
      </code>
      
       only — no target
    </td>
    
    <td>
      find groups / flag anomalies
    </td>
    
    <td>
      ❌ no <code>
        score
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <strong>
        Monitoring
      </strong>
      
       (4)
    </td>
    
    <td>
      <code>
        CUSUM
      </code>
      
      , <code>
        EWMA
      </code>
      
      , <code>
        ParallelMonitor
      </code>
      
      , <code>
        SeasonalMonitor
      </code>
    </td>
    
    <td>
      <code>
        { field }
      </code>
      
       — one series
    </td>
    
    <td>
      detect when a series drifts from its normal
    </td>
    
    <td>
      <code>
        update(row)
      </code>
      
       for real-time
    </td>
  </tr>
  
  <tr>
    <td>
      <strong>
        Spatial scan
      </strong>
      
       (2)
    </td>
    
    <td>
      <code>
        SpatialScan
      </code>
      
       (Kulldorff) · <code>
        GetisOrd
      </code>
      
       (Gi* hotspots)
    </td>
    
    <td>
      <code>
        { zone, coordinates, population, cases }
      </code>
      
       / <code>
        { zone, coordinates, cases }
      </code>
    </td>
    
    <td>
      detect statistically significant spatial clusters / hotspots
    </td>
    
    <td>
      <code>
        cluster(data)
      </code>
      
       → p-value · <code>
        hotspots(data)
      </code>
      
       → z-scores
    </td>
  </tr>
</tbody>
</table>

The rule of thumb: **supervised** models have a `target` in the spec and can be
scored against ground truth; **unsupervised** models only take `features` and
have no `score`. Monitoring and spatial scan have no target either — they
answer different questions (time drift / spatial clusters).

## Value encoding

<table>
<thead>
  <tr>
    <th>
      JSON value
    </th>
    
    <th>
      Encoded as
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        number
      </code>
    </td>
    
    <td>
      as-is
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        boolean
      </code>
    </td>
    
    <td>
      <code>
        1
      </code>
      
       / <code>
        0
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      numeric string (<code>
        "42"
      </code>
      
      )
    </td>
    
    <td>
      <code>
        Number("42")
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      non-numeric string
    </td>
    
    <td>
      one-hot column(s) if <code>
        oneHot: true
      </code>
      
      , otherwise error
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        undefined
      </code>
      
       / <code>
        null
      </code>
    </td>
    
    <td>
      depends on <code>
        missing
      </code>
    </td>
  </tr>
</tbody>
</table>

Fields not selected in `features` / `target` are ignored.

Features are sorted **alphabetically** (and one-hot categories as well), so
permuting the `features` array has no effect on the column layout — the same
data always produces the same matrix, coefficients, and predictions.

## Missing-value strategies

<table>
<thead>
  <tr>
    <th>
      Option
    </th>
    
    <th>
      Behavior
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        'throw'
      </code>
      
       (default)
    </td>
    
    <td>
      throws an explicit error
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        'fill0'
      </code>
    </td>
    
    <td>
      fills the column with <code>
        0
      </code>
      
      , the row is kept
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        'drop'
      </code>
    </td>
    
    <td>
      removes the whole row (tracked in <code>
        droppedRows
      </code>
      
      )
    </td>
  </tr>
</tbody>
</table>

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

<table>
<thead>
  <tr>
    <th>
      Method
    </th>
    
    <th>
      Returns
    </th>
    
    <th>
      Availability
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        score(data)
      </code>
    </td>
    
    <td>
      R² (regressors) / accuracy (classifiers)
    </td>
    
    <td>
      all 23 supervised models
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        mse(data)
      </code>
      
       / <code>
        mae(data)
      </code>
    </td>
    
    <td>
      mean squared / absolute error
    </td>
    
    <td>
      14 regressors
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        classificationReport(data, beta?)
      </code>
    </td>
    
    <td>
      precision / recall / Fβ / confusion matrix
    </td>
    
    <td>
      9 classifiers
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        rocAucScore(data)
      </code>
    </td>
    
    <td>
      AUC (binary)
    </td>
    
    <td>
      5 models with <code>
        predict_proba
      </code>
    </td>
  </tr>
</tbody>
</table>

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
- `DBSCAN` / `HDBSCAN` have **no predict** on new points

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
