# Overview

> Unsupervised grouping — KMeans, density-based DBSCAN and hierarchical HDBSCAN.

> **Scoped vs flat import** — scoped: `new clusters.KMeans()` ·
> flat: `import { KMeans } from "@dnax/ml"`. Both are identical.

Clustering groups unlabeled rows into meaningful groups — there is **no
target field**, the spec only selects `features`.

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
      <a href="/06-clustering/01-kmeans">
        KMeans
      </a>
    </td>
    
    <td>
      centroid clustering — you pick <code>
        n_clusters
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/06-clustering/02-dbscan">
        DBSCAN
      </a>
    </td>
    
    <td>
      density clustering — you pick <code>
        eps
      </code>
      
      , noise = <code>
        -1
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/06-clustering/03-hdbscan">
        HDBSCAN
      </a>
    </td>
    
    <td>
      density hierarchy — it picks the shapes
    </td>
  </tr>
</tbody>
</table>

## Tips

- Normalize features (`options: { scale: true }`) before clustering so
distances behave consistently.
- `options.noise` can be used as a **stability diagnostic**: if clusters
change a lot under slight jittering, the signal is weak.
- **KMeans** groups by *count* (you pick k) · **DBSCAN** by *density*
(you pick eps) · **HDBSCAN** by *density hierarchy* (it picks the shapes) —
start with HDBSCAN when the clusters have uneven densities or shapes.
