---
title: Overview
description: Regression and classification on linear relationships — 9 models sharing one JSON-first API.
---

> **Scoped vs flat import** — scoped: `new linear.LinearRegression()` ·
> flat: `import { LinearRegression } from "@dnax/ml"`. Both are identical.

All linear models share the same JSON-first API: `fit`, `predict`,
`fill_predict`, `coef`, `intercept`, `getParams()` / `setParams()`,
`predictAsync`, `score`, `mse`, `mae` and `export` / `load`.

| Model | Purpose |
| ----- | ------- |
| [Linear Regression](/04-linear/01-linear-regression) | ordinary least squares — the baseline |
| [Logistic Regression](/04-linear/02-logistic-regression) | classification with probabilities |
| [Ridge Regression](/04-linear/03-ridge-regression) | L2 — stable with correlated features |
| [Lasso Regression](/04-linear/04-lasso-regression) | L1 — feature selection |
| [Elastic Net](/04-linear/05-elastic-net) | L1 + L2 combined |
| [Ridge Classifier](/04-linear/06-ridge-classifier) | fast label-only classifier |
| [RANSAC Regressor](/04-linear/07-ransac-regressor) | robust to outliers |
| [Poisson Regressor](/04-linear/08-poisson-regressor) | count targets, predictions ≥ 0 |
| [Polynomial Regression](/04-linear/09-polynomial-regression) | nonlinear curves |

`predictAsync(data)` is available on every regressor: it offloads the
prediction to a worker so the event loop stays responsive on large sets.

```ts
const preds = await reg.predictAsync(bigData);
```
