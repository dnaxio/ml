---
title: Spatial Scan
description: Detect statistically significant spatial clusters — a localized outbreak, a geographic concentration.
navigation:
  icon: lucide:map-pin
---

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
