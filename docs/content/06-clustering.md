---
title: Clustering
description: Unsupervised grouping — KMeans and density-based DBSCAN.
navigation:
  icon: lucide:boxes
---

Clustering groups unlabeled rows into meaningful groups — there is **no
target field**, the spec only selects `features`.

## `KMeans` — centroid clustering

Partitions rows into `n_clusters` groups around centroids. `fit` assigns
labels to the training rows, `predict` assigns new rows to the nearest
centroid.

```ts
import { KMeans } from "@dnax/ml";

const km = new KMeans({ n_clusters: 3, random_state: 42 });
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

## `DBSCAN` — density clustering

Groups **dense regions** and marks isolated points as **noise** (label `-1`).
You don't choose the number of clusters — you choose density.

```ts
import { DBSCAN } from "@dnax/ml";

const db = new DBSCAN({ eps: 0.5, minSamples: 2 });
db.fit(data, { features: ["x", "y"] });
db.labels_; // [0, 0, 1, 1, -1, ...] — -1 means noise
```

| Param         | Default | Role                                              |
| ------------- | ------- | ------------------------------------------------- |
| `eps`         | `0.5`   | neighborhood radius                               |
| `minSamples`  | `5`     | min neighborhood size for a core point            |
| `distanceType`| —       | `'euclidean'` | `'manhattan'` | `'minkowski'` | ... |

> **Note**: DBSCAN does not support `predict` on new points (like
> scikit-learn) — use `fit_predict` on the full dataset instead.

## Tips

- Normalize features (`options: { scale: true }`) before clustering so
  distances behave consistently.
- `options.noise` can be used as a **stability diagnostic**: if clusters
  change a lot under slight jittering, the signal is weak.
