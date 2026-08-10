---
title: Linear
description: 9 regression & classification models — constructor params and methods.
---

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
