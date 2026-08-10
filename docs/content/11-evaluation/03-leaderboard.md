---
title: Leaderboard
description: compareModels ranks several models on the same case, and detectTask tells you the ML case.
---

`compareModels` trains, evaluates and **ranks** several models on the same
case — the same data + spec + folds, one fresh model per fold (via
`crossValScore`). Use models of the same family: all regressors (R²
comparable) or all classifiers (accuracy comparable).

```ts
import { compareModels, detectTask } from "@dnax/ml";

// 0) What ML case is this?
const task = detectTask(data, { features: ["age", "annees_exp"], target: "salaire" });
// → "regression"  ("classification" when the target is boolean / discrete)

const leaderboard = compareModels(
  {
    linear: () => new LinearRegression(),
    ridge: () => new RidgeRegression({ alpha: 1 }),
    gbr: () => new GradientBoostingRegressor({ randomState: 42 }),
    knn: () => new KNeighborsRegressor(),
  },
  data,
  { features: ["age", "annees_exp"], target: "salaire" }, // scale: true propagé
  { cv: 5, scoring: "score", randomState: 42 },
);
// → [{ name, mean, std, scores: [...] }, ...] — sorted best first
const best = leaderboard[0];
```

- `scoring` default `'score'` uses each model's own `score()`: R² for
  regressors, accuracy for classifiers (the ML case is detected per model).
  For `'mse'`/`'mae'` the ranking is ascending (lower = better).
- `std` is the fold spread — prefer a model with a tight `std` over a
  slightly higher mean from one lucky fold.

## `detectTask` — which ML case is this?

Tells you whether a `{ features, target }` spec is a **classification** or a
**regression** problem: the target is boolean, or has few unique values
(≤ 10 and a minority of the rows) → `'classification'`, otherwise
`'regression'`. Use it to pick a model family or a default scoring.

```ts
import { detectTask } from "@dnax/ml";

detectTask(clients, { features: ["age", "solde"], target: "achete" });
// → "classification"  (boolean target)

detectTask(employees, { features: ["annee", "age"], target: "salaire" });
// → "regression"
```
