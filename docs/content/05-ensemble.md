---
title: Ensemble Models
description: Random forests, AdaBoost, Gradient Boosting and XGBoost — plus IsolationForest for anomaly detection.
navigation:
  icon: lucide:layers
---

Ensembles combine many weak models into one strong model. They expose
`featureImportances` instead of `coef` / `intercept`, plus the evaluation
methods (`score`, `classificationReport`, `rocAucScore` where applicable).

## `IsolationForest` — anomaly detection

Unsupervised: builds random partition trees and flags points that are
isolated quickly (short path length) as anomalies. **No target** — only
`features`. Returns binary labels and a continuous `anomaly_score`.

```ts
import { IsolationForest } from "@dnax/ml";

const model = new IsolationForest({ random_state: 42 });
model.fit(data, { features: ["montant", "heure"] }); // build the forest
const labels = model.predict(data); // → [0, 0, 0, 1, 0, 1] — 1 = anomaly

const score = model.anomaly_score({ montant: 9000, heure: 3 });
// → 0.62 — continuous anomaly score (custom thresholds)
```

Ideal for **fraud**, **IoT sensor faults**, or **data cleaning** before
supervised training.

## `RandomForestClassifier` & `RandomForestRegressor`

Many decision trees combined by **majority vote** (classifier) or
**averaging** (regressor). More robust and accurate than a single tree.
Note: kml's `RandomForestClassifier` does **not** expose `predict_proba`.

```ts
import { RandomForestClassifier, RandomForestRegressor } from "@dnax/ml";

const clf = new RandomForestClassifier({
  nEstimators: 100,
  max_depth: 10,
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });

const reg = new RandomForestRegressor({ nEstimators: 100, randomState: 42 });
reg.fit(data, { features: ["age", "annees_exp"], target: "salaire" });
```

## `AdaBoostClassifier` — sequential boosting

Fits **stumps** (1-level trees) sequentially, each correcting the previous
mistakes (weighted resampling). Exposes calibrated `predict_proba`.

```ts
const clf = new AdaBoostClassifier({
  nEstimators: 100,
  learningRate: 1.0,
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data); // per-class probabilities
```

> **Pitfall**: AdaBoost stumps cannot converge on a perfectly symmetric XOR
> pattern — probabilities stay near `[0.5, 0.5]`. Use linearly segmentable
> data.

## `AdaBoostRegressor` — sequential boosting (regression)

```ts
const reg = new AdaBoostRegressor({
  nEstimators: 100,
  learningRate: 1.0,
  randomState: 42,
});
reg.fit(data, { features: ["x"], target: "prix" });
```

## `GradientBoostingClassifier` — the strongest tabular default

Sequential trees fitted on the **residuals** of the previous ensemble, with a
small learning rate. Calibrated probabilities.

```ts
const clf = new GradientBoostingClassifier({
  nEstimators: 100,
  learningRate: 0.1, // key lever: low + many trees = robust
  maxDepth: 3,       // keep 2-4
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data); // calibrated per-class probabilities
```

## `GradientBoostingRegressor`, `XGBoostClassifier` & `XGBoostRegressor`

Regularized gradient boosting close to the `xgboost` library (XGBoost adds
L1/L2 regularization and row/feature subsampling).

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

const xclf = new XGBoostClassifier({ nEstimators: 100, maxDepth: 3 });
xclf.fit(data, { features: ["x1", "x2"], target: "classe" });
xclf.predict_proba(data);  // calibrated probabilities
xclf.rocAucScore(data);    // AUC on rows with the target

const xreg = new XGBoostRegressor({ nEstimators: 100, maxDepth: 3 });
xreg.fit(data, { features: ["x"], target: "prix" });
```

> **Pitfall**: XGBoost can underfit on poorly-separated data with the default
> `baseScore = 0.5` (which must stay in `(0,1)` for binary logistic). Use
> clearly separated classes, or lower `baseScore`.

## Common parameters

| Param (XGBoost)      | Default | Role                                  |
| -------------------- | ------- | ------------------------------------- |
| `nEstimators`        | `100`   | boosting rounds                       |
| `learningRate`       | `0.3`   | step size (lower + more trees = robust) |
| `maxDepth`           | `6`     | tree depth                            |
| `lambda`             | `1`     | L2 regularization on weights          |
| `gamma`              | `0`     | min loss reduction for a split        |
| `minChildWeight`     | `1`     | min sum of weights in a child         |
| `subsample`          | `1`     | fraction of rows per round            |
| `colsampleByTree`    | `1`     | fraction of features per tree         |
| `baseScore`          | `0.5`   | initial prediction score              |
| `randomState`        | —       | seed for reproducible fits            |
