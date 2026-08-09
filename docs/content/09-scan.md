---
title: Spatial Scan
description: Detect statistically significant spatial clusters — Kulldorff's scan (most likely cluster) and Getis-Ord Gi* (per-zone hotspots).
navigation:
  icon: lucide:map-pin
---

> **Scoped vs flat import** — scoped: `new scan.SpatialScan()` ·
> flat: `import { SpatialScan } from "@dnax/ml"`. Both are identical.

`SpatialScan` (Kulldorff / SaTScan) detects **statistically significant
spatial clusters** — e.g. a localized disease outbreak. It slides a circular
window of every size around every zone, scores each window with the Poisson
log-likelihood ratio (cases vs population-expected), and assesses the most
likely cluster with a Monte-Carlo p-value (random datasets with the same
total cases distributed by population).

```ts
import { SpatialScan } from "@dnax/ml";

// 1) Baseline: zone geometry + population + usual case counts
const scan = new SpatialScan({ replications: 999, randomState: 42 });
scan.fit(zones, {
  zone: "zone",
  coordinates: ["lon", "lat"],
  population: "population",
  cases: "cases", // baseline (background) case rate
});

// 2) Current period: scan the new case counts
const cluster = scan.cluster(current); // { zones, cases, expected, llr, pValue } | null
const suivi = scan.fill_predict(current); // rows + cluster: boolean
```

## Parameters

| Param               | Default     | Role                                                 |
| ------------------- | ----------- | ---------------------------------------------------- |
| `replications`      | `199`       | Monte-Carlo draws for the p-value (SaTScan uses 999) |
| `significance`      | `0.05`      | clusters with `p ≤ value` are reported               |
| `maxWindowFraction` | `0.5`       | max zones per window (fraction of all zones)         |
| `randomState`       | —           | seed for reproducible Monte-Carlo draws              |
| `clusterField`      | `'cluster'` | boolean field filled by `fill_predict`               |

## Notes

- `fit` learns the zone map + populations + the global baseline case rate;
  `cluster` / `predict` / `fill_predict` scan the **current** case counts.
- Returns the **most likely cluster** when significant, `null` otherwise.
  SaTScan-style secondary (non-overlapping) clusters are not reported yet.
- On a grid where cold zones are equidistant from hot ones, the circular
  window may include a few extra zones (genuine Kulldorff behavior).
- Each call reruns the Monte-Carlo: call `cluster()` once and reuse the
  result, or use `fill_predict()`.

## Result shape — `ScanCluster`

```ts
interface ScanCluster {
  zones: string[];      // zone ids inside the cluster
  cases: number;        // observed cases in the cluster
  expected: number;     // expected cases (population × global rate)
  llr: number;          // log-likelihood ratio (larger = stronger)
  pValue: number;       // empirical Monte-Carlo p-value
}
```

## `GetisOrd` — Gi* local hotspots

Complementary statistic: Kulldorff answers *"where is the single most likely
cluster?"*; **Getis-Ord Gi*** answers *"which zones are unusually hot or
cold?"* — **one result per zone**, no Monte-Carlo, O(n·k).

```ts
import { GetisOrd } from "@dnax/ml";

const gi = new GetisOrd({ distance: 1, significance: 0.05 });
gi.fit(zones, {
  zone: "zone",
  coordinates: ["lon", "lat"],
  cases: "cases", // the variable of interest (counts or rates)
});

const results = gi.hotspots(current); // per zone
// → { zone, zScore, pValue, hot, cold }[]
const flagged = gi.fill_predict(current); // rows + hot: boolean
```

**How it works**: for each zone i, the Gi* z-score compares the case sum over
the neighborhood (zones within `distance`, i included) to what the global
mean would predict. A **positive z with p ≤ significance = hotspot**, a
**negative z = cold spot**.

| Param         | Default | Role                                              |
| ------------- | ------- | ------------------------------------------------- |
| `distance`    | mean NN | neighborhood radius (coordinates unit)            |
| `significance`| `0.05`  | two-sided level for hot/cold classification       |
| `hotField`    | `'hot'` | boolean field filled by `fill_predict`            |

**Notes**

- `fit` learns geometry only (same zones as `SpatialScan`); the default
  `distance` is the mean nearest-neighbor distance (deterministic).
- `hotspots(data)` returns `{ zone, zScore, pValue, hot, cold }` per row;
  `predict` → 1 = significant hotspot, `fill_predict` → + `hot: boolean`.
- Works on raw counts **or** rates — pass population-adjusted rates in
  `cases` to compare zones of different sizes fairly.
- Uniform cases (zero variance) → z = 0, p = 1 → no hotspot.

**When to use which**

| Question | Model |
| -------- | ----- |
| "Where is the most likely cluster?" (retrospective, with population) | `SpatialScan` |
| "Which zones are statistically hot right now?" (map of hotspots) | `GetisOrd` |
| "Cluster the cases without a population denominator" | `DBSCAN` / `HDBSCAN` |
