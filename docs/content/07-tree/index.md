---
title: Overview
description: Interpretable decision trees — DecisionTree and ExtraTree classifiers and regressors.
---

> **Scoped vs flat import** — scoped: `new tree.DecisionTreeClassifier()` ·
> flat: `import { DecisionTreeClassifier } from "@dnax/ml"`. Both are identical.

Trees split the feature space into **if/else rules** — the most interpretable
models in the SDK. They share a common JSON-first API and expose
`featureImportances` (how much each feature drives the splits) instead of
`coef` / `intercept`.

| Model | Purpose |
| ----- | ------- |
| [DecisionTreeClassifier](/07-tree/01-decision-tree-classifier) | readable if/then rules — labels only |
| [DecisionTreeRegressor](/07-tree/02-decision-tree-regressor) | stepwise predictions (leaf mean) |
| [ExtraTreeClassifier](/07-tree/03-extra-tree-classifier) | randomized tree with probabilities |
| [ExtraTreeRegressor](/07-tree/04-extra-tree-regressor) | randomized tree for continuous targets |

## Common parameters

| Param               | Default        | Role                                        |
| ------------------- | -------------- | ------------------------------------------- |
| `max_depth`         | `Infinity`     | tree depth (limit to avoid overfitting)     |
| `min_samples_split` | `2`            | min samples required to keep splitting      |
| `criterion`         | `'entropy'`    | impurity criterion (`'gini'` available)     |
| `max_features`      | —              | features per split (`number` | `'sqrt'` | `'log2'`) |
| `randomState`       | —              | seed for reproducible random feature selection |

All tree models also expose the evaluation methods (`score` for accuracy/R²,
`mse`/`mae` for regressors, `classificationReport` for classifiers) and full
persistence (`export` / `load`).
