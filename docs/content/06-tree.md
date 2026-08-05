---
title: Tree Models
description: Interpretable decision trees — DecisionTree and ExtraTree classifiers and regressors.
navigation:
  icon: lucide:git-branch
---

> **Scoped vs flat import** — scoped: `new tree.DecisionTreeClassifier()` ·
> flat: `import { DecisionTreeClassifier } from "@dnax/ml"`. Both are identical.

Trees split the feature space into **if/else rules** — the most interpretable
models in the SDK. They share a common JSON-first API and expose
`featureImportances` (how much each feature drives the splits) instead of
`coef` / `intercept`.

## `DecisionTreeClassifier`

```ts
import { DecisionTreeClassifier } from "@dnax/ml";

const clf = new DecisionTreeClassifier({ max_depth: 3 });
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict(data);           // labels
clf.fill_predict(data);      // rows + predicted label
clf.featureImportances;      // how much each feature drives the splits
```

Note: kml's `DecisionTreeClassifier` has **no** `predict_proba` — labels only.

## `DecisionTreeRegressor`

```ts
import { DecisionTreeRegressor } from "@dnax/ml";

const reg = new DecisionTreeRegressor({ max_depth: 3 });
reg.fit(data, { features: ["age", "annees_exp"], target: "salaire" });
reg.predict(data);
```

## `ExtraTreeClassifier` & `ExtraTreeRegressor`

Extremely randomized trees: split thresholds are chosen **randomly** instead
of optimally. Faster and often more robust against overfitting. The
classifier exposes `predict_proba` / `fill_predict_proba`.

```ts
import { ExtraTreeClassifier, ExtraTreeRegressor } from "@dnax/ml";

const clf = new ExtraTreeClassifier({ max_depth: 5 });
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data);          // per-class probabilities
clf.fill_predict_proba(data);     // rows + probability in the target

const reg = new ExtraTreeRegressor({ max_depth: 5, randomState: 42 });
reg.fit(data, { features: ["note", "heures"], target: "resultat" });
```

## Common parameters

| Param               | Default        | Role                                        |
| ------------------- | -------------- | ------------------------------------------- |
| `max_depth`         | `Infinity`     | tree depth (limit to avoid overfitting)     |
| `min_samples_split` | `2`            | min samples required to keep splitting      |
| `criterion`         | `'entropy'`    | impurity criterion (`'gini'` available)     |
| `max_features`      | —              | features per split (`number` | `'sqrt'` | `'log2'`) |
| `randomState`       | —              | seed for reproducible random feature selection |

All tree models also expose the evaluation methods (`score` for accuracy/R²,
`mse` for regressors, `classificationReport` for classifiers) and full
persistence (`export` / `load`).
