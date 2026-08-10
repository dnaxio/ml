# @dnax/ml

A JSON-first machine learning layer for JavaScript / TypeScript, built on top of [@kanaries/ml](https://ml.kanaries.net/docs). Train and predict directly on **JSON row objects** — no manual matrix conversion needed.

## Features

- **JSON-first API**: fit and predict on plain arrays of objects like `[{ name, note, admis }]`
- **scikit-learn-aligned vocabulary**: `coef`, `intercept`, `getParams()` / `setParams()`, `features` / `target`
- **Automatic transformation**: booleans → 1/0, numeric strings → numbers, categorical fields → one-hot
- **Robust missing-value handling**: `throw` | `fill0` | `drop`
- **Model persistence**: export to / load from a `.json` file

## Scoped imports

All models are available both as **flat** named exports and grouped by
**family** (kml-style namespaces):

```ts
import { linear, clusters, tree, ensemble, monitoring, scan, neighbors, featureSelection, evaluation } from "@dnax/ml";

const reg = new linear.LinearRegression();
const km = new clusters.KMeans({ n_clusters: 3 });
const dt = new tree.DecisionTreeClassifier();
const iso = new ensemble.IsolationForest();
const cusum = new monitoring.CUSUM();
const scanModel = new scan.SpatialScan();
const knn = new neighbors.KNeighborsClassifier();
evaluation.compareModels({ ... }, data, spec); // scoped evaluation helpers

// flat imports also work:
import { LinearRegression, KMeans } from "@dnax/ml";
```

## Evaluation

Every supervised model scores itself on rows that include the `target` field
(ground truth), reusing the transformation learned at fit time (rows dropped
by the `'drop'` strategy are excluded):

```ts
// Regressors → R² + MSE + MAE
reg.score(data); // number — R² (1 = perfect fit)
reg.mse(data); // number — mean squared error (0 = perfect)
reg.mae(data); // number — mean absolute error (0 = perfect)

// Classifiers → accuracy + full report
clf.score(data); // number — accuracy
clf.classificationReport(data, 2); // { ..., fScore: F2 } — beta weights recall
//   (default 1 = F1; 2 = recall twice as important; 0.5 = precision twice as important)

// Classifiers with predict_proba (LogisticRegression, ExtraTreeClassifier,
// AdaBoostClassifier, GradientBoostingClassifier, XGBoostClassifier) → ROC
clf.rocAucScore(data); // number — AUC (1 = perfect, 0.5 = random, binary only)

// Metric library — pure functions on arrays
import { rmse, mape, mcc, balancedAccuracy, logLoss, prAucScore, optimalThreshold } from "@dnax/ml";
rmse(preds, truth);          // √MSE (target units)
mape(preds, truth);          // average % off
mcc(preds, truth);           // Matthews −1..+1 (imbalance)
balancedAccuracy(preds, truth); // macro recall
logLoss(truth, proba);       // cross-entropy (proba = positive class)
prAucScore(truth, proba);    // PR-AUC — the imbalance-friendly curve metric
optimalThreshold(truth, proba); // Youden's J decision threshold
```

Implemented on: 14 regressors (`score`/`mse`/`mae`) and 9 classifiers
(`score`/`classificationReport`, + `rocAucScore` on the 5 with `predict_proba`).
Clustering, IsolationForest, monitoring and scan have no target → no `score`.

The ground truth convention matches kml: predictions first, truth second.

### Splitting & cross-validation

```ts
import { trainTestSplit, crossValScore, compareModels, detectTask } from "@dnax/ml";

// Train/test split (JSON-first, seeded, stratified by a field)
const { train, test } = trainTestSplit(data, {
  testSize: 0.3,
  shuffle: true,
  randomState: 42,
  stratify: "achete", // preserves class proportions in both folds
});
reg.fit(train, { features: ["x"], target: "y" });
const r2 = reg.score(test);

// K-fold cross-validation — compare models without one arbitrary split
const scores = crossValScore(
  () => new LinearRegression(), // fresh model per fold
  data,
  { features: ["x"], target: "y" },
  { cv: 5, scoring: "score", randomState: 42 }, // 'score' | 'mse' | 'mae' | 'rocAucScore'
);
// → [0.91, 0.88, 0.93, ...] — one score per fold

// Model leaderboard — train, evaluate and rank several models on one case
const leaderboard = compareModels(
  {
    linear: () => new LinearRegression(),
    ridge: () => new RidgeRegression({ alpha: 1 }),
    gbr: () => new GradientBoostingRegressor({ randomState: 42 }),
  },
  data,
  { features: ["x"], target: "y" }, // same spec (scale, noise…) for every model
  { cv: 5, scoring: "score", randomState: 42 },
);
// → [{ name, mean, std, scores }] — sorted best first (ascending for mse/mae)
console.log(detectTask(data, { features: ["x"], target: "y" })); // 'regression'
```

`crossValScore` stratifies automatically when the target is discrete with
enough samples per class, and falls back to plain k-fold otherwise.

### Streaming (big-data inference)

`predictStream` and `fillPredictStream` process rows **in chunks** (arrays,
generators or async generators), so inference over millions of rows stays
memory-bounded:

```ts
import { predictStream, fillPredictStream } from "@dnax/ml";

for await (const pred of predictStream(reg, rows, { chunkSize: 1000 })) { /* one per row */ }
await fillPredictStream(reg, rows, (row) => { /* enriched row */ });
```

## Installation

```bash
npm install @dnax/ml
# or
bun add @dnax/ml
```

## Quick start

```ts
import { LinearRegression } from "@dnax/ml";

const students = [
  { name: "koffi", note: 1, admis: true },
  { name: "Jules", note: 2, admis: false },
  { name: "Aya", note: 3, admis: true },
  { name: "Sam", note: 4, admis: false },
  { name: "Léo", note: 5, admis: true },
];

const reg = new LinearRegression();
reg.fit(students, { features: ["note", "admis"], target: "note" });

const preds = reg.predict([{ name: "x", note: 6, admis: true }]);
console.log(preds[0]); // 6
```

## Classification with `LogisticRegression`

```ts
import { LogisticRegression } from "@dnax/ml";

const clients = [
  { age: 20, solde: 100, achete: false },
  { age: 35, solde: 5000, achete: true },
  // ...
];

const clf = new LogisticRegression();
clf.fit(clients, { features: ["age", "solde"], target: "achete" });

const labels = clf.predict([{ age: 38, solde: 7000 }]); // [1]
const probs = clf.predict_proba([{ age: 38, solde: 7000 }]); // [[~0, ~1]]
```

A boolean `target` is encoded automatically: `false` → 0, `true` → 1. It
shares the same JSON-first API as `LinearRegression` (`fit`, `predict`,
`fill_predict`, `coef`, `intercept`, `getParams`, `setParams`, `export`,
`load`) plus `predict_proba` and `fill_predict_proba` for probabilities.

### `RidgeRegression` — regularized regression

Linear regression with **L2 regularization** (`alpha`): stable coefficients
when features are correlated or when ordinary least squares overfits. It
shares the full JSON-first API of `LinearRegression`.

```ts
import { RidgeRegression } from "@dnax/ml";

const reg = new RidgeRegression({ alpha: 1 }); // L2 penalty (default: 1)
reg.fit(data, { features: ["age", "annees_exp"], target: "salaire" });
reg.predict(data);
```

The `alpha` / `fitIntercept` config is preserved through `export` / `load`.

### `LassoRegression` — sparse regression / feature selection

Linear regression with **L1 regularization** (`alpha`): some coefficients are
shrunk to exactly `0`, discarding uninformative features (feature selection).
It shares the full JSON-first API of `LinearRegression`.

```ts
import { LassoRegression } from "@dnax/ml";

const reg = new LassoRegression({ alpha: 1 }); // L1 penalty (default: 1)
reg.fit(data, {
  features: ["age", "annees_exp", "inutile"],
  target: "salaire",
});
reg.coef; // → [..., 0]  (uninformative features are set to 0)
```

The `alpha` / `fitIntercept` / `maxIter` / `tol` config is preserved through
`export` / `load`.

### `ElasticNet` — combined L1 + L2 regularization

Combines Lasso (L1) and Ridge (L2) penalties: stable coefficients on
correlated features **and** feature selection. `l1Ratio` balances the two
(`0` = pure Ridge, `1` = pure Lasso, default `0.5`). Shares the full
JSON-first API of `LinearRegression`.

```ts
import { ElasticNet } from "@dnax/ml";

const reg = new ElasticNet({ alpha: 1, l1Ratio: 0.5 });
reg.fit(data, {
  features: ["age", "annees_exp", "inutile"],
  target: "salaire",
});
reg.coef; // → stable on correlated features, 0 on uninformative ones
```

The `alpha` / `l1Ratio` / `fitIntercept` / `maxIter` / `tol` config is
preserved through `export` / `load`.

### `RidgeClassifier` — fast regularized classifier

L2-regularized linear classifier (one-vs-rest ridge models). Fast and
interpretable on numeric tabular data; binary and multiclass supported.
Same JSON-first API as `LogisticRegression` but **without** `predict_proba`
(no probabilities — only labels).

```ts
import { RidgeClassifier } from "@dnax/ml";

const clf = new RidgeClassifier({ alpha: 1 });
clf.fit(data, { features: ["note", "heures"], target: "admis" });
clf.predict(data); // labels
clf.fill_predict(data); // rows + predicted label
clf.coef; // matrix (one row of weights per class)
```

### `RANSACRegressor` — robust to outliers

Linear regression that ignores **gross outliers**: it fits many random
subsets, keeps the consensus set of inliers, and predicts from the best fit.
It also exposes `inlierMask` to identify which training samples were
considered outliers. Shares the full JSON-first API of `LinearRegression`.

```ts
import { RANSACRegressor } from "@dnax/ml";

const reg = new RANSACRegressor({ randomState: 42 });
reg.fit(data, { features: ["x"], target: "y" }); // y contains outliers
reg.coef; // fitted on the inliers only
reg.inlierMask; // [true, ..., false] — which samples are outliers
```

`export` / `load` use the official kml serializer, preserving the inlier
mask, trial count, and config.

### `PoissonRegressor` — count targets

Generalized linear model for **counts** (non-negative integers like number
of sales, clicks, incidents). Uses a log link, so predictions are always
`>= 0`. Shares the full JSON-first API of `LinearRegression`.

```ts
import { PoissonRegressor } from "@dnax/ml";

const reg = new PoissonRegressor({ alpha: 0.1 });
reg.fit(data, { features: ["pub", "prix"], target: "ventes" });
reg.predict(data); // always >= 0
```

The `alpha` / `fitIntercept` / `maxIter` / `tol` config is preserved through
`export` / `load`.

### `PolynomialRegression` — nonlinear curves

Fits a nonlinear curve by expanding each feature to powers `1..degree`, then
running ordinary least squares on the expanded matrix. Use when a linear
baseline is too simple but the target follows a smooth curve. Shares the
full JSON-first API of `LinearRegression`.

```ts
import { PolynomialRegression } from "@dnax/ml";

const reg = new PolynomialRegression({ degree: 2 }); // default: 2
reg.fit(data, { features: ["quantite"], target: "cout" });
reg.predict(data);
```

The `degree` config is preserved through `export` / `load`. Beware of very
high degrees: they overfit.

## Unsupervised clustering

Clustering groups unlabeled rows into meaningful groups — there is **no
target field**, the spec only selects `features`.

### `IsolationForest` — anomaly detection

Builds random partition trees and flags points isolated quickly (short path
length) as anomalies. **No target** — only `features`. Returns binary
labels and a continuous `anomaly_score` for custom thresholds.

```ts
import { IsolationForest } from "@dnax/ml";

const model = new IsolationForest({ random_state: 42 });
model.fit(data, { features: ["montant", "heure"] }); // build the forest
const labels = model.predict(data); // → [0, 0, 0, 1, 0, 1]  — 1 = anomaly

const score = model.anomaly_score({ montant: 9000, heure: 3 });
// → 0.62 — continuous anomaly score
```

Ideal for **fraud**, **IoT sensor faults**, or **data cleaning** before
supervised training.

### `RandomForestClassifier` & `RandomForestRegressor`

Many decision trees combined by **majority vote** (classifier) or
**averaging** (regressor). More robust and accurate than a single tree.
Shares the tree JSON-first API — exposes `featureImportances` instead of
`coef` / `intercept`. Note: kml's RandomForestClassifier does **not** expose
`predict_proba` (only `predict`).

```ts
import { RandomForestClassifier, RandomForestRegressor } from "@dnax/ml";

const clf = new RandomForestClassifier({
  nEstimators: 100,
  max_depth: 5,
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict(data);
clf.featureImportances;

const reg = new RandomForestRegressor({ nEstimators: 100, randomState: 42 });
reg.fit(data, { features: ["note", "heures"], target: "resultat" });
reg.predict(data);
```

### `AdaBoostClassifier` — sequential boosting

Trains decision stumps sequentially, each one focusing on the samples
misclassified by the previous rounds. Effective when a simple baseline misses
hard cases. Exposes `predict_proba` / `fill_predict_proba` and
`featureImportances`.

```ts
import { AdaBoostClassifier } from "@dnax/ml";

const clf = new AdaBoostClassifier({
  nEstimators: 25,
  learningRate: 0.5,
  randomState: 42,
});
clf.fit(data, { features: ["x1", "x2"], target: "classe" });
clf.predict_proba(data);
```

Beware of overfitting as `nEstimators` grows on small datasets. Note: on a
perfectly symmetric XOR pattern, stumps cannot converge (known AdaBoost
limitation).

### `AdaBoostRegressor` — sequential boosting (regression)

The regression counterpart of AdaBoost (AdaBoost.R2): small trees reweight
the worst-predicted samples each round, and the final prediction is a
**weighted median** (robust to outlier estimators). Exposes
`featureImportances` instead of `coef` / `intercept`.

```ts
import { AdaBoostRegressor } from "@dnax/ml";

const reg = new AdaBoostRegressor({
  nEstimators: 50,
  learningRate: 0.5,
  randomState: 42,
});
reg.fit(data, { features: ["x"], target: "prix" });
reg.predict(data);
```

### `GradientBoostingClassifier` — the strongest tabular default

Additive chain of shallow trees, each fitting the residuals of the previous
(gradient descent on log loss). Produces **calibrated probabilities**
(sigmoid for binary, softmax for multiclass). Exposes `predict_proba` /
`fill_predict_proba` and `featureImportances`.

```ts
import { GradientBoostingClassifier } from "@dnax/ml";

const clf = new GradientBoostingClassifier({
  nEstimators: 100,
  learningRate: 0.1, // key lever: low + many trees = robust
  maxDepth: 3, // keep 2-4
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data); // calibrated per-class probabilities
```

### `GradientBoostingRegressor`, `XGBoostClassifier` & `XGBoostRegressor`

The strongest boosting models: gradient boosting on squared error
(regressor) and regularized exact-greedy boosting close to the `xgboost`
library (classifier + regressor). The classifiers expose calibrated
`predict_proba` / `fill_predict_proba`; all expose `featureImportances`.

```ts
import {
  GradientBoostingRegressor,
  XGBoostClassifier,
  XGBoostRegressor,
} from "@dnax/ml";

const gbr = new GradientBoostingRegressor({
  nEstimators: 100,
  learningRate: 0.1,
  maxDepth: 3,
});
gbr.fit(data, { features: ["x"], target: "prix" });

const xclf = new XGBoostClassifier({ nEstimators: 50, maxDepth: 3 });
xclf.fit(data, { features: ["x1", "x2"], target: "classe" });
xclf.predict_proba(data);

const xreg = new XGBoostRegressor({ nEstimators: 50, maxDepth: 3 });
xreg.fit(data, { features: ["x"], target: "prix" });
```

Note: XGBoost's `baseScore` (default 0.5) and shallow data can underfit;
use clearly separated classes or tune `baseScore` / `learningRate`.

### `KMeans` — centroid clustering

```ts
import { KMeans } from "@dnax/ml";

const km = new KMeans({ n_clusters: 3, random_state: 42 });
km.fit(data, { features: ["age", "solde"] });
km.labels_; // [0, 0, 1, 1, 2, ...] — cluster label per training row
km.predict(data); // assign new rows to the nearest centroid
km.centroids; // cluster centers
km.inertia; // compactness (lower = tighter)
```

### `DBSCAN` — density clustering (handles noise)

```ts
import { DBSCAN } from "@dnax/ml";

const db = new DBSCAN({ eps: 0.5, minSamples: 2 });
db.fit(data, { features: ["x", "y"] });
db.labels_; // [0, 0, 1, 1, -1, ...] — -1 means noise
// Note: DBSCAN does not support predict on new points (like scikit-learn);
// use fit_predict on the full dataset instead.
```

### `HDBSCAN` — hierarchical density clustering

Extends DBSCAN with a density hierarchy: clusters of arbitrary shape and
varying density are selected by stability, and isolated points get a
near-zero membership probability.

```ts
import { HDBSCAN } from "@dnax/ml";

const hdb = new HDBSCAN({ min_cluster_size: 5 });
hdb.fit(ventesGeo, { features: ["x", "y"] });
hdb.labels_;       // [0, 0, 1, 1, ...] — cluster label per row
hdb.probabilities; // [1, 0.9, 1, ...] — membership strength (0 = noise)
```

KMeans groups by *count* (you pick k) · DBSCAN by *density* (you pick eps) ·
HDBSCAN by *density hierarchy* (it picks the shapes) — start with HDBSCAN
when the clusters have uneven densities or shapes.

Clustering specs support the same transformation options (`oneHot`,
`missing`, `scale`). Normalize features (`scale: true`) before clustering so
distances behave consistently.

## Neighbors (instance-based)

No model is learned: predictions come from the **nearest labeled examples**.
A simple and strong non-parametric baseline — if kNN beats your parametric
models, the relationship is very local. **Distance-based**: use
`options: { scale: true }`.

```ts
import { KNeighborsClassifier, KNeighborsRegressor } from "@dnax/ml";

const knn = new KNeighborsClassifier({ kNeighbors: 5 });
knn.fit(data, { features: ["age", "solde"], target: "achete" });
knn.predict(data);           // labels
knn.classificationReport(data); // precision / recall / F1

const knnR = new KNeighborsRegressor({ nNeighbors: 5 });
knnR.fit(data, { features: ["age"], target: "salaire" });
knnR.score(data); // R²
```

## Feature selection

Rank or select the informative columns before training — lower cost, better
interpretability, less overfitting. `featureSelection.*` works on JSON rows
with a `JsonFitSpec` (numeric/boolean features only).

```ts
import { featureSelection } from "@dnax/ml";

// 1) Univariate ranking — most informative first
const ranking = featureSelection.mutualInfoClassif(data, {
  features: ["age", "solde", "nb_visites", "inutile"],
  target: "achete",
});
// → [{ feature: "age", score: 0.42 }, { feature: "inutile", score: 0.01 }, ...]

// 2) Model-based selection — keep what the model finds important
const sel = new featureSelection.SelectFromModel({
  estimator: new LassoRegression({ alpha: 1 }),
  threshold: "mean",
});
sel.fit(data, { features: ["salaire", "bruit"], target: "age" });
sel.selectedFeatures; // ["salaire"]

// 3) Feed the reduced spec to a fresh model
clf.fit(data, { features: sel.selectedFeatures, target: "achete" });
```

- `chi2` / `fClassif` (classification targets) → score + `pValue`
- `mutualInfoClassif` / `mutualInfoRegression` → capture **non-linear** links
- `SelectFromModel({ estimator, threshold, maxFeatures })` → `support`,
  `selectedFeatures`, `featureScores`, `fittedEstimator`

## Tree models

Interpretable if/else decision rules. Unlike the linear family, trees expose
`featureImportances` instead of `coef` / `intercept`.

### `DecisionTreeClassifier`

```ts
import { DecisionTreeClassifier } from "@dnax/ml";

const clf = new DecisionTreeClassifier({ max_depth: 3 });
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict(data); // labels
clf.fill_predict(data); // rows + predicted label
clf.featureImportances; // how much each feature drives the splits
```

### `DecisionTreeRegressor`

```ts
import { DecisionTreeRegressor } from "@dnax/ml";

const reg = new DecisionTreeRegressor({ max_depth: 4 });
reg.fit(data, { features: ["note", "heures"], target: "resultat" });
reg.predict(data);
reg.featureImportances;
```

Trees are restored through `export` / `load` using the official kml
serializer. Control `max_depth` / `min_samples_split` — unconstrained trees
overfit quickly.

### `ExtraTreeClassifier` & `ExtraTreeRegressor`

More randomized tree variants than the standard DecisionTree: the same API,
usually a better baseline. **ExtraTreeClassifier exposes `predict_proba`**
and `fill_predict_proba` (per-class probabilities), unlike `DecisionTreeClassifier`.

```ts
import { ExtraTreeClassifier, ExtraTreeRegressor } from "@dnax/ml";

const clf = new ExtraTreeClassifier({ max_depth: 5 });
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data); // per-class probabilities
clf.fill_predict_proba(data); // rows + probability in the target

const reg = new ExtraTreeRegressor({ max_depth: 5, randomState: 42 });
reg.fit(data, { features: ["note", "heures"], target: "resultat" });
```

### `fill_predict_proba(data)` (LogisticRegression only)

Like `fill_predict`, but always fills the `target` field with the probability
of the positive class, a number in `[0, 1]` (binary case), or the highest
class probability (confidence) in the multiclass case.

```ts
const filled = clf.fill_predict_proba([{ age: 38, solde: 7000 }]);
// → [{ age: 38, solde: 7000, achete: 0.97 }]
```

## API reference

### `fit(data, spec)`

Fits the model on JSON rows.

```ts
reg.fit(data, {
  features: ["note", "admis"], // fields used as features (one column per field, in order)
  target: "note", // single output field
  options: {
    // optional
    oneHot: false, // encode categorical strings as columns (default: false)
    dropFirst: true, // drop first category in one-hot to avoid collinearity (default: true)
    missing: "throw", // 'throw' | 'fill0' | 'drop' (default: 'throw')
    scale: false, // standardize features (StandardScaler) before training (default: false)
    noise: 0.05, // Gaussian jittering std on continuous features, training only (default: none)
    noiseSeed: 42, // seed for the noise generator → reproducible fits (default: random)
  },
});
```

### `predict(data)`

Predicts on JSON rows, reusing the transformation learned at training time. The `target` field is not required.

```ts
const preds = reg.predict([{ note: 6, admis: true }]); // number[]
```

### `fill_predict(data)`

Same input as `predict`, but returns the input rows **with the `target` field
filled with the predicted value** (new objects, the input is not mutated).
Rows dropped by the `'drop'` missing strategy are excluded.

The filled value type follows the target type seen at training time:
a boolean target produces `true`/`false`, a numeric (0/1) target stays numeric.

```ts
const filled = reg.fill_predict([{ note: 6, admis: true }]);
// → [{ note: 6, admis: true, note: 5.8 }]  (target "note" filled with the prediction)
```

### Model parameters

```ts
reg.coef; // number[] — coefficients (one per feature)
reg.intercept; // number   — bias term

reg.getParams(); // { coef: number[], intercept: number }
reg.setParams({ coef, intercept }); // injects parameters, throws on unknown keys, returns this
```

### Persistence

```ts
// Export (async, writes <name>.json with model params + learned transformation)
await reg.export("my-model"); // → my-model.json

// Load (async, restores a previously exported model)
const restored = new LinearRegression();
await restored.load("my-model");
restored.predict(newData); // works with the same columns

// Restore directly from parameters
const fromParams = new LinearRegression({
  coef: reg.coef,
  intercept: reg.intercept,
});
```

The export file is **versioned** (`version` field). `load` rejects files with
an unsupported version to protect against format changes.

### `predictAsync(data)`

Like `predict`, but offloads the matrix-vector multiplication to a worker
thread (via kml `utils.asyncMode`), keeping the event loop responsive on
large prediction sets. The JSON → matrix transformation still runs on the
main thread.

```ts
const preds = await reg.predictAsync(bigData);
```

### Utilities

```ts
reg.columnNames; // string[] — X column names (useful to interpret coefficients)
reg.droppedRows; // number — rows removed by the 'drop' missing strategy
```

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

## Noise (data jittering)

`options.noise` adds **Gaussian noise** `N(0, noise)` to continuous feature
values **during training only** — a light data augmentation / regularizer.

- Applied **after scaling**, so with `scale: true` a value of `0.05` means 5%
  of a column standard deviation (scale-free). Without `scale`, it is in raw
  feature units.
- **One-hot columns are never perturbed** (binary indicators stay valid).
- **Inference stays deterministic**: `predict` / `fill_predict` never add noise.
- `noiseSeed` makes the noise reproducible (same seed → same fit).
- The config is part of `spec.options`, so it **survives export/load**.

```ts
const robust = new LinearRegression();
robust.fit(data, {
  features: ["prix_affiche", "quantite"],
  target: "prix_facture",
  options: { scale: true, noise: 0.05, noiseSeed: 42 },
});
```

**When to use it**: linear models (`LinearRegression`, `RidgeRegression`,
`LassoRegression`, `ElasticNet`, `RANSACRegressor`, `PolynomialRegression`,
`PoissonRegressor`, `LogisticRegression`, `RidgeClassifier`) — it prevents the
fit from memorizing exact training values and improves robustness to small
measurement errors. It is **neutral for trees/boosting** (they have their own
stochasticity) and **counter-productive for `IsolationForest`** (it blurs the
anomalies). For `KMeans`/`DBSCAN` it can be used as a stability diagnostic.

## Monitoring (time-series)

`CUSUM` and `EWMA` are **statistical process control** methods that detect the
**onset of a sustained shift** in a series — e.g. the beginning of a sales
peak. They complement the ML models: IsolationForest flags anomalies _per
transaction_, CUSUM/EWMA flag anomalies _over time_.

Both follow the SDK convention (`fit` → `predict` → `fill_predict`) on JSON
rows, with a `MonitorSpec` selecting the numeric field to monitor:

```ts
import { CUSUM, EWMA } from "@dnax/ml";

const days = [
  { day: 1, sales: 118 },
  { day: 2, sales: 121 },
  // ...
  { day: 18, sales: 200 }, // peak
];

// 1) Learn the reference (μ0, σ) on the normal period
const cusum = new CUSUM();
cusum.fit(days.slice(0, 12), { field: "sales" });

// 2) Monitor the whole series → alert as soon as the rise starts
const alerts = cusum.predict(days); // number[]: 1 = alert, 0 = normal
const scores = cusum.scores(days); // cumulative S_n per point

// 3) JSON rows with the alert field filled (input is not mutated)
const tracked = cusum.fill_predict(days); // { day, sales, alert: true | false }

// Exponential smoothing, same usage:
const ewma = new EWMA({ lambda: 0.25, limit: 3 });
ewma.fit(days.slice(0, 12), { field: "sales" });
const ewmaAlerts = ewma.predict(days);
```

Both models persist with `export` / `load` (the learned statistics + spec are
saved). `export` / `load` are async, like the rest of the SDK.

### `CUSUM` — Cumulative SUM

Accumulates deviations from a reference mean: `S_n = max(0, S_{n-1} + (x_n − μ0 − k))`
and alerts when `S_n > h`. Very sensitive to **small sustained shifts**.

| Param        | Default      | Role                                                           |
| ------------ | ------------ | -------------------------------------------------------------- |
| `target`     | estimated    | reference mean μ0 (deviations are measured against it)         |
| `std`        | estimated    | series standard deviation σ                                    |
| `k`          | `0.5·σ`      | allowable slack — sub-k shifts are ignored as background noise |
| `h`          | `5·σ`        | alert threshold — lower = earlier but noisier alarms           |
| `direction`  | `'increase'` | shift to detect: `'increase'` or `'decrease'`                  |
| `alertField` | `'alert'`    | field name filled by `fill_predict`                            |

### `EWMA` — Exponentially Weighted Moving Average

Smooths the series with a decaying memory (`z_n = λ·x_n + (1−λ)·z_{n−1}`) and
alerts when the smoothed value leaves `μ0 ± L·σ_z(n)`. Recovers quickly after
isolated spikes (they are smoothed out) — good for noisy daily data.

| Param        | Default   | Role                                                             |
| ------------ | --------- | ---------------------------------------------------------------- |
| `lambda`     | `0.25`    | smoothing factor λ ∈ (0,1] — small = smooth, sensitive to trends |
| `limit`      | `3`       | control-limit width L (in σ units)                               |
| `target`     | estimated | reference mean μ0                                                |
| `std`        | estimated | series standard deviation σ                                      |
| `alertField` | `'alert'` | field name filled by `fill_predict`                              |

### `ParallelMonitor` — several series at once

Monitors several series simultaneously with one chart per field. An alert
fires when **at least one** chart fires, and `alertFields` reports **which**
fields triggered (interpretable alarms).

```ts
import { ParallelMonitor } from "@dnax/ml";

const pm = new ParallelMonitor(); // one CUSUM per field (default)
pm.fit(days, { fields: ["paracetamol", "ibuprofen"] });

const alerts = pm.predict(days); // 1 when at least one chart fires
const fields = pm.alertFields(days); // ["paracetamol"] per row — which fields
const tracked = pm.fill_predict(days); // + alert: boolean + alertFields: string[]

// EWMA charts, with a controlled family-wise false-alarm rate (Bonferroni):
const pmE = new ParallelMonitor({ model: "ewma", familyError: 0.05 });
pmE.fit(days, { fields: ["paracetamol", "ibuprofen"] });
```

**Why the Bonferroni correction matters**: with N independent charts, the
chance of at least one false alarm grows with N (5 charts at 5% each → ~23%
family-wise). `familyError` divides the risk across charts: each chart gets
the two-sided level `familyError / N` (normal quantile). When omitted, each
chart keeps its own conservative defaults (`h = 5·σ`, `L = 3`).

| Param              | Default         | Role                                                    |
| ------------------ | --------------- | ------------------------------------------------------- |
| `model`            | `'cusum'`       | chart type per field: `'cusum'` or `'ewma'`             |
| `familyError`      | —               | family-wise false-alarm rate (Bonferroni across fields) |
| `alertField`       | `'alert'`       | boolean field filled by `fill_predict`                  |
| `alertFieldsField` | `'alertFields'` | field listing the triggering fields                     |

Per-field overrides go in the spec: `{ fields, params: { paracetamol: { k, h } } }`
(user values always win over the Bonferroni adjustment). Persistence works
exactly like the other models (`export` / `load`, async).

> **Tip**: estimate `k`/`h` (or `target`/`std`) on a known-normal baseline
> period, then monitor the full series — otherwise the peak itself inflates
> the reference statistics.

### Advanced options (CUSUM / EWMA)

- **`changePoint(data)`** (CUSUM) — estimates where the drift _began_ (the
  point right after the statistic last reset to 0), not just where it was
  detected. Returns one index per row (`-1` when no drift is accumulating).
- **`limits(data)`** (EWMA) — returns the time-varying control band
  `{ ucl, lcl }` per point, useful for plotting or heatmaps.
- **`robust: true`** — estimates μ0/σ with the **median** and the **MAD**
  (1.4826·MAD) instead of the mean/std, so a few anomalous baseline days
  barely move the thresholds (they would inflate the std and delay alarms).
- **`direction: "both"`** (CUSUM) — runs the increase _and_ decrease
  statistics and alerts on the max (detects any sustained drift, not just
  peaks).
- **`missing: "fill"`** — carries the last known value forward (LOCF) when
  the field is absent / non-numeric, keeping the output aligned with the
  input rows (e.g. closed days). Default `'throw'`.

```ts
const cp = new CUSUM({ robust: true, direction: "both" });
cp.fit(baseline, { field: "sales", missing: "fill" });
const onset = cp.changePoint(sales); // estimated drift onset per day

const ew = new EWMA({ lambda: 0.25, limit: 3 });
ew.fit(baseline, { field: "sales" });
const { ucl, lcl } = ew.limits(sales); // control band for plot/heatmap
```

### `SeasonalMonitor` — weekly deseasonalization

Pharmacy sales follow a **weekly cycle** (weekend dips, Monday peaks). On a
constant-mean chart this cycle inflates σ, raising the thresholds and
**delaying** the detection of a real epidemic shift. `SeasonalMonitor` fits a
reference mean **per day of week** from the baseline, then runs a CUSUM (or
EWMA) on the deseasonalized residuals `x_n − μ0_{dayOfWeek(n)}`.

```ts
import { SeasonalMonitor } from "@dnax/ml";

const sm = new SeasonalMonitor();
sm.fit(baseline, { field: "sales", dateField: "date" }); // at least one full week

const alerts = sm.predict(sales); // 1 = the deseasonalized chart fires
const profile = sm.dayProfile; // reference mean per day of week (diagnostic)
const tracked = sm.fill_predict(sales); // + alert: boolean
```

- **Needs ≥ 1 full week of baseline** (every day-of-week present) — throws otherwise.
- `dateField` accepts a `Date`, an ISO date (`"2026-08-03"` parsed as local)
  or a timestamp.
- On a real baseline with a weekly cycle (max/min profile ratio > ~1.3), it
  detects a moderate epidemic **several days earlier** than a plain CUSUM
  (test: day 16 vs day 20).
- `changePoint(data)` (CUSUM) and `limits(data)` (EWMA) are available on the
  deseasonalized chart.

## Spatial scan (Kulldorff / SaTScan)

`SpatialScan` detects **statistically significant spatial clusters** — e.g. a
localized disease outbreak. It slides a circular window of every size around
every zone, scores each window with the Poisson log-likelihood ratio (cases vs
population-expected), and assesses the most likely cluster with a Monte-Carlo
p-value (random datasets with the same total cases distributed by population).

```ts
import { SpatialScan } from "@dnax/ml";

// 1) Baseline: zone geometry + population + usual case counts
const scan = new SpatialScan({ replications: 999, randomState: 42 });
scan.fit(zones, {
  zone: "zone",
  coordinates: ["lon", "lat"],
  population: "population",
  cases: "cases", // baseline (background) case rate
});

// 2) Current period: scan the new case counts
const cluster = scan.cluster(current); // { zones, cases, expected, llr, pValue } | null
const suivi = scan.fill_predict(current); // rows + cluster: boolean
```

### `GetisOrd` — Gi* local hotspots

Complementary statistic: Kulldorff answers *"where is the single most likely
cluster?"*; **Getis-Ord Gi*** answers *"which zones are unusually hot or
cold?"* — one result per zone, no Monte-Carlo, O(n·k).

```ts
import { GetisOrd } from "@dnax/ml";

const gi = new GetisOrd({ distance: 1 });
gi.fit(zones, { zone: "zone", coordinates: ["lon", "lat"], cases: "cases" });

const results = gi.hotspots(current); // { zone, zScore, pValue, hot, cold }[]
const flagged = gi.fill_predict(current); // rows + hot: boolean
```

- `distance` defaults to the mean nearest-neighbor distance (deterministic).
- Works on raw counts **or** rates — pass population-adjusted rates in
  `cases` to compare zones of different sizes fairly.
- Uniform cases (zero variance) → z = 0, p = 1 → no hotspot.

| Param               | Default     | Role                                                 |
| ------------------- | ----------- | ---------------------------------------------------- |
| `replications`      | `199`       | Monte-Carlo draws for the p-value (SaTScan uses 999) |
| `significance`      | `0.05`      | clusters with `p ≤ value` are reported               |
| `maxWindowFraction` | `0.5`       | max zones per window (fraction of all zones)         |
| `randomState`       | —           | seed for reproducible Monte-Carlo draws              |
| `clusterField`      | `'cluster'` | boolean field filled by `fill_predict`               |

Notes:

- `fit` learns the zone map + populations + the global baseline case rate;
  `cluster` / `predict` / `fill_predict` scan the **current** case counts.
- Returns the **most likely cluster** when significant, `null` otherwise.
  SaTScan-style secondary (non-overlapping) clusters are not reported yet.
- On a grid where cold zones are equidistant from hot ones, the circular
  window may include a few extra zones (genuine Kulldorff behavior).
- Each call reruns the Monte-Carlo: call `cluster()` once and reuse the
  result, or use `fill_predict()`.

## Missing-value strategies

| Option              | Behavior                                         |
| ------------------- | ------------------------------------------------ |
| `'throw'` (default) | throws an explicit error                         |
| `'fill0'`           | fills the column with `0`, the row is kept       |
| `'drop'`            | removes the whole row (tracked in `droppedRows`) |

## Architecture

`@dnax/ml` is a **JSON-first facade** over the `@kanaries/ml` engine (pinned
to an exact version — `1.1.0` — to freeze its internal serialization format):

```
linear/ clusters/ tree/ ensemble/       ← model wrappers (never import kml directly)
monitoring/ scan/                       ← pure JS, zero dependency
core/                                   ← the SINGLE import point of @kanaries/ml
│   ├── kml.ts         (engine namespace + factories)
│   └── state.ts       (centralized internal-state access + serialization)
evaluation/                             ← scoring helpers
transformation/                         ← JSON → matrix + scaler via core
```

Swapping the engine (or vendoring it) only requires rewriting `core/`
— the wrappers never touch the package directly.

## License

MIT
