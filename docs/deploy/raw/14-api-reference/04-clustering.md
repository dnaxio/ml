# Clustering

> KMeans, DBSCAN and HDBSCAN — spec has no target.

Spec has **no target** — only `features`.

<table>
<thead>
  <tr>
    <th>
      Class
    </th>
    
    <th>
      Params
    </th>
    
    <th>
      Methods & getters
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        KMeans
      </code>
    </td>
    
    <td>
      <code>
        n_clusters
      </code>
      
      , <code>
        tol
      </code>
      
      , <code>
        max_iter
      </code>
      
      , <code>
        initCenters
      </code>
      
      , <code>
        random_state
      </code>
    </td>
    
    <td>
      <code>
        fit
      </code>
      
      , <code>
        predict
      </code>
      
       (nearest centroid), <code>
        fit_predict
      </code>
      
      , <code>
        export
      </code>
      
      , <code>
        load
      </code>
      
       · <code>
        labels_
      </code>
      
      , <code>
        centroids
      </code>
      
      , <code>
        inertia
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        DBSCAN
      </code>
    </td>
    
    <td>
      <code>
        eps
      </code>
      
      , <code>
        minSamples
      </code>
      
      , <code>
        distanceType
      </code>
    </td>
    
    <td>
      <code>
        fit
      </code>
      
      , <code>
        fit_predict
      </code>
      
      , <code>
        export
      </code>
      
      , <code>
        load
      </code>
      
       · <code>
        labels_
      </code>
      
       (<code>
        -1
      </code>
      
       = noise). No <code>
        predict
      </code>
      
       on new points
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        HDBSCAN
      </code>
    </td>
    
    <td>
      <code>
        min_cluster_size
      </code>
      
      , <code>
        min_samples
      </code>
      
      , <code>
        cluster_selection_epsilon
      </code>
      
      , <code>
        metric
      </code>
      
      , <code>
        allow_single_cluster
      </code>
    </td>
    
    <td>
      <code>
        fit
      </code>
      
      , <code>
        fit_predict
      </code>
      
      , <code>
        export
      </code>
      
      , <code>
        load
      </code>
      
       · <code>
        labels_
      </code>
      
      , <code>
        probabilities
      </code>
      
       (0 = noise). No <code>
        predict
      </code>
      
       on new points
    </td>
  </tr>
</tbody>
</table>
