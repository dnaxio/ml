---
title: KNeighborsClassifier
description: Classify each point by majority vote of its k nearest labeled examples.
---

Each point is classified by **majority vote** of its `kNeighbors` nearest
labeled examples (with `weightType: "distance"`, nearer neighbors get more
weight).

**When to use it** — **no training phase**: predictions come from the nearest
labeled examples. Use as a non-parametric baseline, or when the decision
boundary is very local (a fraud looks like the frauds near it). Distance-based
→ `scale: true` is required.

```ts
import { neighbors } from "@dnax/ml";

const clf = new neighbors.KNeighborsClassifier({ kNeighbors: 5 });
clf.fit(clients, {
  features: ["age", "solde"],
  target: "achete",
  options: { scale: true },
});

clf.predict(newClients);            // labels
clf.fill_predict(newClients);       // rows + predicted label
clf.classes;                        // [0, 1] — sorted class labels (classes_)
clf.score(test);                    // accuracy
clf.classificationReport(test);     // precision / recall / F1 / confusion matrix
```

Note: kml's KNeighborsClassifier has **no `predict_proba`** — labels only.
`classes` exposes the sorted class labels (sklearn-style `classes_`).

| Param          | Default | Role                                          |
| -------------- | ------- | --------------------------------------------- |
| `kNeighbors`   | `5`     | number of nearest neighbors in the vote       |
| `weightType`   | —       | `'uniform'` (majority) or `'distance'` (inverse-distance) |
| `distanceType` | —       | distance metric (`'euclidean'`, `'manhattan'`, ...) |
| `pNorm`        | `2`     | Minkowski p-norm (only for `'minkowski'`)     |

sklearn-style aliases are accepted: `nNeighbors`, `weights` and `metric`
map to `kNeighbors`, `weightType` and `distanceType`.

Methods: `predict`, `fill_predict`, `score` (accuracy), `classificationReport`,
`classes`, `getParams()` / `setParams()`, `export` / `load`.
