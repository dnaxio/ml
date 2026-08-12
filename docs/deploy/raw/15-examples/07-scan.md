# Spatial Scan

> 2 examples — Kulldorff's scan and Getis-Ord Gi* hotspots.

## `SpatialScan` — locate the epidemic cluster (Kulldorff)

**When to use it** — **where is the most likely geographic cluster?** —
epidemic outbreak location, fraud hotspots. Needs a **population
denominator** per zone.

```ts
const zones = [
  { zone: "A", lon: 0, lat: 0, population: 1000, cases: 8 }, // ← hotspot
  { zone: "B", lon: 1, lat: 0, population: 1000, cases: 1 },
  { zone: "C", lon: 0, lat: 1, population: 1000, cases: 1 },
  { zone: "D", lon: 1, lat: 1, population: 1000, cases: 1 },
  { zone: "E", lon: 5, lat: 5, population: 1000, cases: 1 },
];
const model = new scan.SpatialScan({ replications: 199, randomState: 42 });
model.fit(zones, { zone: "zone", coordinates: ["lon", "lat"], population: "population", cases: "cases" });
model.cluster(zones);
// → { zones: ["A"], cases: 8, expected: ~2.4, llr: ~6.1, pValue: 0.005 }
//   A is the most likely cluster (p < 0.05). Returns null when nothing is significant.
```

## `GetisOrd` — abnormally hot zone (Gi* hotspot)

**When to use it** — **which zones are significantly hot/cold?** — per-zone
significance without a population. Needs more zones (10+) than the other
examples on this page.

> ⚠️ Spatial statistics need more zones: Gi* cannot reach significance with
> ≤ 5 rows (the global variance absorbs the hotspot). This example uses a
> 3×3 grid — the only exception to the 5-row rule on this page.

```ts
const zones = [
  { zone: "A", lon: 0, lat: 0, cases: 1 }, { zone: "B", lon: 1, lat: 0, cases: 8 },
  { zone: "C", lon: 2, lat: 0, cases: 1 }, { zone: "D", lon: 0, lat: 1, cases: 8 },
  { zone: "E", lon: 1, lat: 1, cases: 8 }, { zone: "F", lon: 2, lat: 1, cases: 8 },
  { zone: "G", lon: 0, lat: 2, cases: 1 }, { zone: "H", lon: 1, lat: 2, cases: 8 },
  { zone: "I", lon: 2, lat: 2, cases: 1 },
];
const gi = new scan.GetisOrd({ distance: 1.1 });
gi.fit(zones, { zone: "zone", coordinates: ["lon", "lat"], cases: "cases" });
gi.hotspots(zones);
// → [{ zone: "E", zScore: ~2.83, pValue: ~0.005, hot: true, cold: false }, ...]
//   E (center) is the only significant hot spot: its 8-case arms lift its
//   neighborhood above the flat 1-case corners.
```
