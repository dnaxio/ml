---
title: Linear Models
description: Regression and classification on linear relationships — 9 models sharing one JSON-first API.
navigation:
  icon: lucide:function-square
---

> **Scoped vs flat import** — scoped: `new linear.LinearRegression()` ·
> flat: `import { LinearRegression } from "@dnax/ml"`. Both are identical.

All linear models share the same JSON-first API: `fit`, `predict`,
`fill_predict`, `coef`, `intercept`, `getParams()` / `setParams()`,
`predictAsync`, `score`, `mse` and `export` / `load`.

## `LinearRegression`

Ordinary least squares — the baseline every other model is compared to.

```ts
import { LinearRegression } from "@dnax/ml";

const reg = new LinearRegression();
reg.fit(students, { features: ["note", "admis"], target: "note" });
const preds = reg.predict([{ name: "x", note: 6, admis: true }]);
console.log(preds[0]); // 6
```

## `LogisticRegression`

Binary (and multiclass) classification. A boolean `target` is encoded
automatically: `false` → 0, `true` → 1. Adds `predict_proba` and
`fill_predict_proba` for probabilities.

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

## `RidgeRegression` — regularized regression

L2 regularization (`alpha`): stable coefficients when features are correlated
or when ordinary least squares overfits.

```ts
const reg = new RidgeRegression({ alpha: 1 }); // L2 penalty (default: 1)
reg.fit(data, { features: ["age", "annees_exp"], target: "salaire" });
```

The `alpha` / `fitIntercept` config is preserved through `export` / `load`.

## `LassoRegression` — sparse regression / feature selection

L1 regularization (`alpha`): some coefficients shrink to exactly `0`,
discarding uninformative features.

```ts
const reg = new LassoRegression({ alpha: 1 }); // L1 penalty (default: 1)
reg.fit(data, { features: ["age", "annees_exp", "inutile"], target: "salaire" });
reg.coef; // → [..., 0]  (uninformative features are set to 0)
```

The `alpha` / `fitIntercept` / `maxIter` / `tol` config is preserved through
`export` / `load`.

## `ElasticNet` — combined L1 + L2

Lasso + Ridge penalties: stable on correlated features **and** feature
selection. `l1Ratio` balances the two (`0` = pure Ridge, `1` = pure Lasso,
default `0.5`).

```ts
const reg = new ElasticNet({ alpha: 1, l1Ratio: 0.5 });
reg.fit(data, { features: ["age", "annees_exp", "inutile"], target: "salaire" });
reg.coef; // stable on correlated features, 0 on uninformative ones
```

The `alpha` / `l1Ratio` / `fitIntercept` / `maxIter` / `tol` config is
preserved through `export` / `load`.

## `RidgeClassifier` — fast regularized classifier

L2-regularized linear classifier (one-vs-rest ridge models). Fast and
interpretable on numeric tabular data; binary and multiclass. **No
`predict_proba`** — labels only.

```ts
const clf = new RidgeClassifier({ alpha: 1 });
clf.fit(data, { features: ["note", "heures"], target: "admis" });
clf.predict(data);      // labels
clf.fill_predict(data); // rows + predicted label
clf.coef;               // matrix (one row of weights per class)
```

## `RANSACRegressor` — robust to outliers

Fits many random subsets, keeps the consensus set of inliers, and predicts
from the best fit. Exposes `inlierMask` to identify outliers.

```ts
const reg = new RANSACRegressor({ randomState: 42 });
reg.fit(data, { features: ["x"], target: "y" }); // y contains outliers
reg.coef;         // fitted on the inliers only
reg.inlierMask;   // [true, ..., false] — which samples are outliers
```

`export` / `load` use the official kml serializer (inlier mask, trial count,
config preserved).

## `PoissonRegressor` — count targets

Generalized linear model for **counts** (non-negative integers). Uses a log
link, so predictions are always `>= 0`.

```ts
const reg = new PoissonRegressor({ alpha: 0.1 });
reg.fit(data, { features: ["pub", "prix"], target: "ventes" });
reg.predict(data); // always >= 0
```

The `alpha` / `fitIntercept` / `maxIter` / `tol` config is preserved through
`export` / `load`.

## `PolynomialRegression` — nonlinear curves

Expands each feature to powers `1..degree`, then fits ordinary least squares
on the expanded matrix. Use when the target follows a smooth curve.

```ts
const reg = new PolynomialRegression({ degree: 2 }); // default: 2
reg.fit(data, { features: ["quantite"], target: "cout" });
reg.predict(data);
```

The `degree` config is preserved through `export` / `load`. Beware of very
high degrees: they overfit.

## `predictAsync(data)`

Every regressor can offload the prediction to a worker, keeping the event
loop responsive on large sets:

```ts
const preds = await reg.predictAsync(bigData);
```
