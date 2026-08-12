# XGBoostRegressor

> Regularized gradient boosting for regression — the go-to for big datasets.

Regularized gradient boosting for a **continuous target**, close to the
`xgboost` library (L1/L2 regularization and row/feature subsampling).

**When to use it** — the go-to choice on **big datasets with many features**;
overkill on tiny data.

```ts
import { ensemble } from "@dnax/ml";

const reg = new ensemble.XGBoostRegressor({ nEstimators: 100, maxDepth: 3 });
reg.fit(data, { features: ["x"], target: "prix" });
```

> **Pitfall**: like the classifier, it underfits on poorly-separated data with
> the default `baseScore = 0.5` — tune `baseScore` / `learningRate` or use
> clearly separated classes.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`featureImportances`, `export` / `load`.
