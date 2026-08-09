---
title: Clustering
description: Unsupervised grouping — KMeans, density-based DBSCAN and hierarchical HDBSCAN.
navigation:
  icon: lucide:boxes
---

> **Scoped vs flat import** — scoped: `new clusters.KMeans()` ·
> flat: `import { KMeans } from "@dnax/ml"`. Both are identical.

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

## `HDBSCAN` — hierarchical density clustering

Extends DBSCAN with a **density hierarchy**: clusters of arbitrary shape and
**varying density** are selected by stability, and isolated points get a
near-zero membership probability. No `predict` on new points — labels are
computed on the training rows.

```ts
import { HDBSCAN } from "@dnax/ml";

const hdb = new HDBSCAN({ min_cluster_size: 5 });
hdb.fit(ventesGeo, { features: ["x", "y"] });
hdb.labels_;       // [0, 0, 1, 1, ...] — cluster label per row
hdb.probabilities; // [1, 0.9, 1, ...] — membership strength (0 = noise)
```

| Param                    | Default | Role                                              |
| ------------------------ | ------- | ------------------------------------------------- |
| `min_cluster_size`       | `5`     | smallest group considered a cluster (≥ 2)         |
| `min_samples`            | `null`  | neighborhood size for core distances (= min size) |
| `cluster_selection_epsilon` | —    | clusters split below this distance are merged     |
| `metric`                 | `'euclidean'` | distance metric name                       |
| `allow_single_cluster`   | `false` | allow the root hierarchy as one cluster           |

## Tips

- Normalize features (`options: { scale: true }`) before clustering so
  distances behave consistently.
- `options.noise` can be used as a **stability diagnostic**: if clusters
  change a lot under slight jittering, the signal is weak.
- **KMeans** groups by *count* (you pick k) · **DBSCAN** by *density*
  (you pick eps) · **HDBSCAN** by *density hierarchy* (it picks the shapes) —
  start with HDBSCAN when the clusters have uneven densities or shapes.
