---
title: KMeans
description: Centroid clustering — partitions rows into n_clusters groups.
---

Partitions rows into `n_clusters` groups around centroids. `fit` assigns
labels to the training rows, `predict` assigns new rows to the nearest
centroid.

**When to use it** — you know the **number of groups** you want
(`n_clusters`): customer segments, market zones, inventory categories. Fast
and interpretable via `centroids`. `scale: true` is required (distances).

```ts
import { clusters } from "@dnax/ml";

const km = new clusters.KMeans({ n_clusters: 3, random_state: 42 });
km.fit(data, { features: ["age", "solde"] });
km.labels_;   // [0, 0, 1, 1, 2, ...] — cluster label per training row
km.predict(data); // assign new rows to the nearest centroid
km.centroids; // cluster centers
km.inertia;   // compactness (lower = tighter)
```

| Param          | Default | Role                                       |
| -------------- | ------- | ------------------------------------------ |
| `n_clusters`   | `2`     | number of clusters                         |
| `tol`          | `0.05`  | convergence tolerance on inertia           |
| `max_iter`     | `30`    | max Lloyd iterations per run               |
| `initCenters`  | —       | user-provided initial centers              |
| `random_state` | —       | seed for reproducible k-means++ init       |

Methods: `fit`, `predict` (nearest centroid), `fit_predict`, `export` / `load`
· getters `labels_`, `centroids`, `inertia`.
