---
title: Neighbors
description: KNeighborsClassifier and KNeighborsRegressor — instance-based, distance-based.
---

Instance-based (non-parametric) learning: predictions come from the nearest
labeled examples, no model is learned. **Distance-based** — use
`options: { scale: true }`.

| Class | Params | Notes |
| ----- | ------ | ----- |
| `KNeighborsClassifier` | `kNeighbors` (5), `weightType`, `distanceType`, `pNorm` | + `score`, `classificationReport`, `classes`, `getParams()`/`setParams()`. No `predict_proba`. sklearn aliases accepted (`nNeighbors`/`weights`/`metric`) |
| `KNeighborsRegressor` | `nNeighbors` (5), `weights` (`'uniform'`\|`'distance'`), `metric`, `p` | + `score`, `mse`, `mae`, `getParams()`/`setParams()`. kml aliases accepted (`kNeighbors`/`weightType`/`distanceType`) |

```ts
const knn = new neighbors.KNeighborsClassifier({ kNeighbors: 5 });
knn.fit(data, { features: ["age", "solde"], target: "achete", options: { scale: true } });
knn.predict(data); // labels
```
