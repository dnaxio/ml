# XGBoostClassifier

> Regularized gradient boosting — state of the art on large tabular data.

Regularized gradient boosting close to the `xgboost` library (L1/L2
regularization and row/feature subsampling). Calibrated probabilities.

**When to use it** — regularized boosting, **state of the art on large tabular
data** (hundreds+ rows). On tiny sets it underfits.

```ts
import { ensemble } from "@dnax/ml";

const clf = new ensemble.XGBoostClassifier({ nEstimators: 100, maxDepth: 3 });
clf.fit(data, { features: ["x1", "x2"], target: "classe" });
clf.predict_proba(data);  // calibrated probabilities
clf.rocAucScore(data);    // AUC on rows with the target
```

> **Pitfall**: XGBoost can underfit on poorly-separated data with the default
> `baseScore = 0.5` (which must stay in `(0,1)` for binary logistic). Use
> clearly separated classes, or lower `baseScore`.

Methods: `predict`, `predict_proba`, `fill_predict`, `fill_predict_proba`,
`score` (accuracy), `classificationReport`, `rocAucScore`,
`featureImportances`, `export` / `load`.
