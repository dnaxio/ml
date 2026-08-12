# Clustering

> 3 examples — KMeans, density-based DBSCAN and hierarchical HDBSCAN.

## `KMeans` — client segments (young vs wealthy)

**When to use it** — you know the **number of groups** you want (`n_clusters`):
customer segments, market zones, inventory categories. Fast and interpretable
via `centroids`. `scale: true` is required (distances).

```ts
const clients = [
  { age: 25, balance: 100 },
  { age: 28, balance: 150 },
  { age: 55, balance: 9000 },
  { age: 60, balance: 8500 },
  { age: 40, balance: 3000 },
];
const km = new clusters.KMeans({ n_clusters: 2, random_state: 42 });
km.fit(clients, { features: ["age", "balance"], options: { scale: true } });
km.labels_;   // → [0, 0, 1, 1, ?] — young vs wealthy clients
km.centroids; // → centers of the 2 segments
```

## `DBSCAN` — store groups by density (noise = −1)

**When to use it** — groups have **arbitrary shape** and density is what
defines them, plus you want the leftovers **flagged as noise (−1)**. No `k` to
choose — you pick `eps` instead.

```ts
const stores = [
  { x: 0.1, y: 0.1 },
  { x: 0.2, y: 0.2 },
  { x: 0.3, y: 0.1 },
  { x: 10.1, y: 10.2 },
  { x: 10.2, y: 10.1 },
];
const db = new clusters.DBSCAN({ eps: 0.5, minSamples: 2 });
db.fit(stores, { features: ["x", "y"] });
db.labels_; // → [0, 0, 0, 1, 1] — 2 dense groups, no noise
```

## `HDBSCAN` — clusters of varying density (with probabilities)

**When to use it** — densities **vary** and you don't want to pick `k` or
`eps`: it finds the hierarchy by itself, and `probabilities` tells you how
confident each point's membership is (≈ 0 = isolated).

```ts
const points = [
  { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 },
  { x: 10, y: 10 }, { x: 11, y: 10 }, { x: 10, y: 11 },
  { x: 5, y: 5 }, // ← isolated (probability ~0)
];
const hdb = new clusters.HDBSCAN({ min_cluster_size: 3 });
hdb.fit(points, { features: ["x", "y"] });
hdb.labels_;       // → 2 clusters; the isolated point has proba ~0 (not necessarily −1)
hdb.probabilities; // → [1, 1, 1, 1, 1, 1, 1, ~0]
```
