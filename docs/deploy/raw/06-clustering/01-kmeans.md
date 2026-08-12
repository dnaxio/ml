# KMeans

> Centroid clustering — partitions rows into n_clusters groups.

Partitions rows into `n_clusters` groups around centroids. `fit` assigns
labels to the training rows, `predict` assigns new rows to the nearest
centroid.

**When to use it** — you know the **number of groups** you want
(`n_clusters`): customer segments, market zones, inventory categories. Fast
and interpretable via `centroids`. `scale: true` is required (distances).

```ts
import { clusters } from "@dnax/ml";

const km = new clusters.KMeans({ n_clusters: 3, random_state: 42 });
km.fit(data, { features: ["age", "solde"] });
km.labels_;   // [0, 0, 1, 1, 2, ...] — cluster label per training row
km.predict(data); // assign new rows to the nearest centroid
km.centroids; // cluster centers
km.inertia;   // compactness (lower = tighter)
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
        n_clusters
      </code>
    </td>
    
    <td>
      <code>
        2
      </code>
    </td>
    
    <td>
      number of clusters
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        tol
      </code>
    </td>
    
    <td>
      <code>
        0.05
      </code>
    </td>
    
    <td>
      convergence tolerance on inertia
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        max_iter
      </code>
    </td>
    
    <td>
      <code>
        30
      </code>
    </td>
    
    <td>
      max Lloyd iterations per run
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        initCenters
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      user-provided initial centers
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        random_state
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      seed for reproducible k-means++ init
    </td>
  </tr>
</tbody>
</table>

Methods: `fit`, `predict` (nearest centroid), `fit_predict`, `export` / `load`
· getters `labels_`, `centroids`, `inertia`.
