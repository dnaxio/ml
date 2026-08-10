---
title: Clustering
description: KMeans, DBSCAN and HDBSCAN — spec has no target.
---

Spec has **no target** — only `features`.

| Class | Params | Methods & getters |
| ----- | ------ | ----------------- |
| `KMeans` | `n_clusters`, `tol`, `max_iter`, `initCenters`, `random_state` | `fit`, `predict` (nearest centroid), `fit_predict`, `export`, `load` · `labels_`, `centroids`, `inertia` |
| `DBSCAN` | `eps`, `minSamples`, `distanceType` | `fit`, `fit_predict`, `export`, `load` · `labels_` (`-1` = noise). No `predict` on new points |
| `HDBSCAN` | `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`, `metric`, `allow_single_cluster` | `fit`, `fit_predict`, `export`, `load` · `labels_`, `probabilities` (0 = noise). No `predict` on new points |
