# Overview

> Every model class, constructor parameter, method and getter — the complete reference for all algorithms.

> **Scoped vs flat import** — scoped: `new linear.RidgeRegression()` ·
> flat: `import { RidgeRegression } from "@dnax/ml"`. Both are identical.

Every model follows the same contract: `fit(data, spec)` → `predict(data)` →
`fill_predict(data)` → `export(name)` / `load(name)`, plus `score(data)` on
supervised models. Getters `columnNames` and `droppedRows` are common to all.

<table>
<thead>
  <tr>
    <th>
      Page
    </th>
    
    <th>
      Families
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <a href="/14-api-reference/01-linear">
        Linear
      </a>
    </td>
    
    <td>
      9 regression & classification models
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/02-tree">
        Tree
      </a>
    </td>
    
    <td>
      DecisionTree & ExtraTree
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/03-ensemble">
        Ensemble
      </a>
    </td>
    
    <td>
      forests, boosting, IsolationForest
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/04-clustering">
        Clustering
      </a>
    </td>
    
    <td>
      KMeans, DBSCAN, HDBSCAN
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/05-neighbors">
        Neighbors
      </a>
    </td>
    
    <td>
      kNN classifier & regressor
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/06-monitoring">
        Monitoring
      </a>
    </td>
    
    <td>
      CUSUM, EWMA, ParallelMonitor, SeasonalMonitor
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/07-scan">
        Spatial scan
      </a>
    </td>
    
    <td>
      SpatialScan, GetisOrd
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/08-feature-selection">
        Feature selection
      </a>
    </td>
    
    <td>
      univariate + SelectFromModel
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/09-evaluation">
        Evaluation
      </a>
    </td>
    
    <td>
      splits, CV, leaderboard, metrics
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/14-api-reference/10-spec-types">
        Spec types
      </a>
    </td>
    
    <td>
      JsonFitSpec, ClusterSpec, MonitorSpec, ScanSpec…
    </td>
  </tr>
</tbody>
</table>
