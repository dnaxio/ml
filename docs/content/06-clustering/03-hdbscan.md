---
title: HDBSCAN
description: Hierarchical density clustering — clusters of arbitrary shape and varying density, with membership probabilities.
---

Extends DBSCAN with a **density hierarchy**: clusters of arbitrary shape and
**varying density** are selected by stability, and isolated points get a
near-zero membership probability. No `predict` on new points — labels are
computed on the training rows.

**When to use it** — densities **vary** and you don't want to pick `k` or
`eps`: it finds the hierarchy by itself, and `probabilities` tells you how
confident each point's membership is (≈ 0 = isolated).

```ts
import { clusters } from "@dnax/ml";

const hdb = new clusters.HDBSCAN({ min_cluster_size: 5 });
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

Methods: `fit`, `fit_predict`, `export` / `load` · getters `labels_`,
`probabilities` (0 = noise). No `predict` on new points.
