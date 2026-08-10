---
title: Overview
description: Unsupervised grouping — KMeans, density-based DBSCAN and hierarchical HDBSCAN.
---

> **Scoped vs flat import** — scoped: `new clusters.KMeans()` ·
> flat: `import { KMeans } from "@dnax/ml"`. Both are identical.

Clustering groups unlabeled rows into meaningful groups — there is **no
target field**, the spec only selects `features`.

| Model | Purpose |
| ----- | ------- |
| [KMeans](/06-clustering/01-kmeans) | centroid clustering — you pick `n_clusters` |
| [DBSCAN](/06-clustering/02-dbscan) | density clustering — you pick `eps`, noise = `-1` |
| [HDBSCAN](/06-clustering/03-hdbscan) | density hierarchy — it picks the shapes |

## Tips

- Normalize features (`options: { scale: true }`) before clustering so
  distances behave consistently.
- `options.noise` can be used as a **stability diagnostic**: if clusters
  change a lot under slight jittering, the signal is weak.
- **KMeans** groups by *count* (you pick k) · **DBSCAN** by *density*
  (you pick eps) · **HDBSCAN** by *density hierarchy* (it picks the shapes) —
  start with HDBSCAN when the clusters have uneven densities or shapes.
