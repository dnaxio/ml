---
title: Overview
description: Random forests, AdaBoost, Gradient Boosting and XGBoost — plus IsolationForest for anomaly detection.
---

> **Scoped vs flat import** — scoped: `new ensemble.RandomForestClassifier()` ·
> flat: `import { RandomForestClassifier } from "@dnax/ml"`. Both are identical.

Ensembles combine many weak models into one strong model. They expose
`featureImportances` instead of `coef` / `intercept`, plus the evaluation
methods (`score`, `classificationReport`, `rocAucScore` where applicable).

| Model | Purpose |
| ----- | ------- |
| [IsolationForest](/08-ensemble/01-isolation-forest) | unsupervised anomaly detection |
| [RandomForestClassifier](/08-ensemble/02-random-forest-classifier) | bagged trees — robust default, labels only |
| [RandomForestRegressor](/08-ensemble/03-random-forest-regressor) | bagged trees for regression |
| [AdaBoostClassifier](/08-ensemble/04-adaboost-classifier) | sequential boosting of stumps |
| [AdaBoostRegressor](/08-ensemble/05-adaboost-regressor) | sequential boosting for regression |
| [GradientBoostingClassifier](/08-ensemble/06-gradient-boosting-classifier) | the strongest tabular default |
| [GradientBoostingRegressor](/08-ensemble/07-gradient-boosting-regressor) | gradient boosting for regression |
| [XGBoostClassifier](/08-ensemble/08-xgboost-classifier) | regularized boosting — large data |
| [XGBoostRegressor](/08-ensemble/09-xgboost-regressor) | regularized boosting for regression |

## Common parameters (XGBoost)

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
