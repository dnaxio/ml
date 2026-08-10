---
title: GetisOrd
description: Getis-Ord Gi* — which zones are statistically hot or cold, no population needed.
---

Complementary statistic: Kulldorff answers *"where is the single most likely
cluster?"*; **Getis-Ord Gi*** answers *"which zones are unusually hot or
cold?"* — **one result per zone**, no Monte-Carlo, O(n·k).

**When to use it** — **which zones are significantly hot/cold?** — per-zone
significance without a population. Needs more zones (10+) to reach
significance.

```ts
import { scan } from "@dnax/ml";

const gi = new scan.GetisOrd({ distance: 1, significance: 0.05 });
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

Methods: `fit`, `hotspots(data)` → `HotspotResult[]`, `predict`,
`fill_predict`, `export` / `load` · getters `zonesList`, `distance`,
`significance`.
