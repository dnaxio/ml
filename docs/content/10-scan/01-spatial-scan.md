---
title: SpatialScan
description: Kulldorff's scan statistic — the most likely geographic cluster, with population.
---

**When to use it** — **where is the most likely geographic cluster?** —
epidemic outbreak location, fraud hotspots. Needs a **population
denominator** per zone. Retrospective (scans the current case counts against
the learned baseline).

```ts
import { scan } from "@dnax/ml";

const scanModel = new scan.SpatialScan({ replications: 999, randomState: 42 });
scanModel.fit(zones, {
  zone: "zone",
  coordinates: ["lon", "lat"],
  population: "population",
  cases: "cases", // baseline (background) case rate
});

const cluster = scanModel.cluster(current); // { zones, cases, expected, llr, pValue } | null
const tracked = scanModel.fill_predict(current); // rows + cluster: boolean
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

Methods: `fit`, `cluster(data)` → `ScanCluster | null`, `predict`,
`fill_predict`, `export` / `load` · getters `zonesList`, `expectedRate`.
