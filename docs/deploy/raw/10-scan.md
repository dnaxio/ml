# Overview

> Detect statistically significant spatial clusters — Kulldorff's scan (most likely cluster) and Getis-Ord Gi* (per-zone hotspots).

> **Scoped vs flat import** — scoped: `new scan.SpatialScan()` ·
> flat: `import { SpatialScan } from "@dnax/ml"`. Both are identical.

`SpatialScan` (Kulldorff / SaTScan) detects **statistically significant
spatial clusters** — e.g. a localized disease outbreak. It slides a circular
window of every size around every zone, scores each window with the Poisson
log-likelihood ratio (cases vs population-expected), and assesses the most
likely cluster with a Monte-Carlo p-value (random datasets with the same
total cases distributed by population).

```ts
import { scan } from "@dnax/ml";

// 1) Baseline: zone geometry + population + usual case counts
const scanModel = new scan.SpatialScan({ replications: 999, randomState: 42 });
scanModel.fit(zones, {
  zone: "zone",
  coordinates: ["lon", "lat"],
  population: "population",
  cases: "cases", // baseline (background) case rate
});

// 2) Current period: scan the new case counts
const cluster = scanModel.cluster(current); // { zones, cases, expected, llr, pValue } | null
const tracked = scanModel.fill_predict(current); // rows + cluster: boolean
```

<table>
<thead>
  <tr>
    <th>
      Model
    </th>
    
    <th>
      Purpose
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <a href="/10-scan/01-spatial-scan">
        SpatialScan
      </a>
    </td>
    
    <td>
      "where is the most likely cluster?" (with population)
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/10-scan/02-getis-ord">
        GetisOrd
      </a>
    </td>
    
    <td>
      "which zones are statistically hot right now?" (no population)
    </td>
  </tr>
</tbody>
</table>

## When to use which

<table>
<thead>
  <tr>
    <th>
      Question
    </th>
    
    <th>
      Model
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      "Where is the most likely cluster?" (retrospective, with population)
    </td>
    
    <td>
      <code>
        SpatialScan
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      "Which zones are statistically hot right now?" (map of hotspots)
    </td>
    
    <td>
      <code>
        GetisOrd
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      "Cluster the cases without a population denominator"
    </td>
    
    <td>
      <code>
        DBSCAN
      </code>
      
       / <code>
        HDBSCAN
      </code>
    </td>
  </tr>
</tbody>
</table>
