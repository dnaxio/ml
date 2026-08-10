---
title: Spatial Scan
description: SpatialScan (Kulldorff) and GetisOrd (Gi*) — params, methods and result shapes.
---

`ScanSpec` = `{ zone, coordinates: [xField, yField], population, cases }` ·
`HotspotSpec` = `{ zone, coordinates: [xField, yField], cases }`.

| Class | Params | Methods & getters |
| ----- | ------ | ----------------- |
| `SpatialScan` | `replications` (199), `significance` (0.05), `maxWindowFraction` (0.5), `randomState`, `clusterField` | `fit`, `cluster(data)` → `ScanCluster \| null`, `predict`, `fill_predict`, `export`, `load` · getters `zonesList`, `expectedRate` |
| `GetisOrd` | `distance`, `significance` (0.05), `hotField` | `fit`, `hotspots(data)` → `HotspotResult[]`, `predict`, `fill_predict`, `export`, `load` · getters `zonesList`, `distance`, `significance` |

`ScanCluster` = `{ zones: string[], cases, expected, llr, pValue }` ·
`HotspotResult` = `{ zone, zScore, pValue, hot, cold }`.
