# HDBSCAN

> Hierarchical density clustering — clusters of arbitrary shape and varying density, with membership probabilities.

Extends DBSCAN with a **density hierarchy**: clusters of arbitrary shape and
**varying density** are selected by stability, and isolated points get a
near-zero membership probability. No `predict` on new points — labels are
computed on the training rows.

**When to use it** — densities **vary** and you don't want to pick `k` or
`eps`: it finds the hierarchy by itself, and `probabilities` tells you how
confident each point's membership is (≈ 0 = isolated).

```ts
import { clusters } from "@dnax/ml";

const hdb = new clusters.HDBSCAN({ min_cluster_size: 5 });
hdb.fit(ventesGeo, { features: ["x", "y"] });
hdb.labels_;       // [0, 0, 1, 1, ...] — cluster label per row
hdb.probabilities; // [1, 0.9, 1, ...] — membership strength (0 = noise)
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
        min_cluster_size
      </code>
    </td>
    
    <td>
      <code>
        5
      </code>
    </td>
    
    <td>
      smallest group considered a cluster (≥ 2)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        min_samples
      </code>
    </td>
    
    <td>
      <code>
        null
      </code>
    </td>
    
    <td>
      neighborhood size for core distances (= min size)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        cluster_selection_epsilon
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      clusters split below this distance are merged
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        metric
      </code>
    </td>
    
    <td>
      <code>
        'euclidean'
      </code>
    </td>
    
    <td>
      distance metric name
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        allow_single_cluster
      </code>
    </td>
    
    <td>
      <code>
        false
      </code>
    </td>
    
    <td>
      allow the root hierarchy as one cluster
    </td>
  </tr>
</tbody>
</table>

Methods: `fit`, `fit_predict`, `export` / `load` · getters `labels_`,
`probabilities` (0 = noise). No `predict` on new points.
