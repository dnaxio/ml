# DBSCAN

> Density-based clustering — dense regions, isolated points become noise (-1).

Groups **dense regions** and marks isolated points as **noise** (label `-1`).
You don't choose the number of clusters — you choose density.

**When to use it** — groups have **arbitrary shape** and density is what
defines them, plus you want the leftovers **flagged as noise (−1)**: store
clusters, anomaly zones. No `k` to choose — you pick `eps` instead.

```ts
import { clusters } from "@dnax/ml";

const db = new clusters.DBSCAN({ eps: 0.5, minSamples: 2 });
db.fit(data, { features: ["x", "y"] });
db.labels_; // [0, 0, 1, 1, -1, ...] — -1 means noise
```

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
        eps
      </code>
    </td>
    
    <td>
      <code>
        0.5
      </code>
    </td>
    
    <td>
      neighborhood radius
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        minSamples
      </code>
    </td>
    
    <td>
      <code>
        5
      </code>
    </td>
    
    <td>
      min neighborhood size for a core point
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        distanceType
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      <code>
        'euclidean'
      </code>
    </td>
  </tr>
</tbody>
</table>

> **Note**: DBSCAN does not support `predict` on new points (like
> scikit-learn) — use `fit_predict` on the full dataset instead.

Methods: `fit`, `fit_predict`, `export` / `load` · getter `labels_`
(`-1` = noise). No `predict` on new points.
