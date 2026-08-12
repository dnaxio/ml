# GetisOrd

> Getis-Ord Gi* — which zones are statistically hot or cold, no population needed.

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

<table>
<thead>
  <tr>
    <th>
      Param
    </th>
    
    <th>
      Default
    </th>
    
    <th>
      Role
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        distance
      </code>
    </td>
    
    <td>
      mean NN
    </td>
    
    <td>
      neighborhood radius (coordinates unit)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        significance
      </code>
    </td>
    
    <td>
      <code>
        0.05
      </code>
    </td>
    
    <td>
      two-sided level for hot/cold classification
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        hotField
      </code>
    </td>
    
    <td>
      <code>
        'hot'
      </code>
    </td>
    
    <td>
      boolean field filled by <code>
        fill_predict
      </code>
    </td>
  </tr>
</tbody>
</table>

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
