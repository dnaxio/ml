---
title: KNeighborsRegressor
description: Predict by averaging the target values of the k nearest labeled examples.
---

Each prediction is the **average** (or inverse-distance weighted average) of
the target values of the `nNeighbors` nearest labeled examples.

**When to use it** — same principle for a **continuous target**: the prediction
is the average of the k nearest known values. Ideal for very local patterns
(price per m² in a neighborhood, delivery time per route). Distance-based →
`scale: true` is required.

```ts
import { neighbors } from "@dnax/ml";

const reg = new neighbors.KNeighborsRegressor({ nNeighbors: 5 });
reg.fit(employees, {
  features: ["age", "annees_exp"],
  target: "salaire",
  options: { scale: true },
});

reg.predict(newEmployees);  // predicted values
reg.score(test);            // R²
reg.mae(test);              // mean absolute error
```

| Param       | Default | Role                                          |
| ----------- | ------- | --------------------------------------------- |
| `nNeighbors`| `5`     | number of nearest neighbors averaged          |
| `weights`   | —       | `'uniform'` (plain average) or `'distance'`   |
| `metric`    | —       | distance metric (`'euclidean'`, `'manhattan'`, ...) |
| `p`         | `2`     | Minkowski p-norm (only for `'minkowski'`)     |

kml-style aliases are accepted: `kNeighbors`, `weightType` and `distanceType`
map to `nNeighbors`, `weights` and `metric`.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`getParams()` / `setParams()`, `export` / `load`.
