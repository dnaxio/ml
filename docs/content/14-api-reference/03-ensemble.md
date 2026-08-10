---
title: Ensemble
description: Random forests, AdaBoost, Gradient Boosting, XGBoost and IsolationForest.
---

Common surface: `fit`, `predict`, `fill_predict`, `score`, `export`, `load`,
`featureImportances`.

| Class | Key params | Notes |
| ----- | ---------- | ----- |
| `IsolationForest` | `subsampling_size`, `tree_num`, `contamination`, `random_state` | unsupervised anomaly; `predict` (1 = anomaly) + `anomaly_score(row)` |
| `RandomForestClassifier` | `nEstimators`, `maxDepth`, `maxFeatures`, `criterion`, `randomState` | no `predict_proba` |
| `RandomForestRegressor` | same | + `mse`, `mae` |
| `AdaBoostClassifier` | `nEstimators`, `learningRate`, `randomState` | + proba + `rocAucScore` |
| `AdaBoostRegressor` | same | + `mse`, `mae` |
| `GradientBoostingClassifier` | `nEstimators`, `learningRate`, `maxDepth`, `minSamplesSplit`, `subsample`, `maxFeatures`, `randomState` | + proba + `rocAucScore` |
| `GradientBoostingRegressor` | same | + `mse`, `mae` |
| `XGBoostClassifier` | `nEstimators`, `learningRate`, `maxDepth`, `lambda`, `gamma`, `minChildWeight`, `subsample`, `colsampleByTree`, `baseScore`, `randomState` | + proba + `rocAucScore` |
| `XGBoostRegressor` | same | + `mse`, `mae` |
