---
title: Tree
description: DecisionTree and ExtraTree classifiers and regressors.
---

Common surface: `fit`, `predict`, `fill_predict`, `score`, `export`, `load`,
`featureImportances`.

| Class | Params | Notes |
| ----- | ------ | ----- |
| `DecisionTreeClassifier` | `max_depth`, `min_samples_split`, `criterion`, `max_features`, `randomState` | no `predict_proba` |
| `DecisionTreeRegressor` | same | + `mse`, `mae` |
| `ExtraTreeClassifier` | same | + `predict_proba`, `fill_predict_proba`, `rocAucScore` |
| `ExtraTreeRegressor` | same | + `mse`, `mae` |
