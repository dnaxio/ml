# Overview

> Ready-made JSON datasets (≤ 5 rows) for every model and API — concrete cases with expected outputs. Scoped imports everywhere.

> **Scoped vs flat import** — the examples use **scoped** imports
> (`new linear.LinearRegression()`, `evaluation.mcc(...)`). Flat imports
> (`new LinearRegression()`) are identical. Note that `options` (scale,
> oneHot, missing…) always live in the **fit spec**, not the constructor.

Each example is a small, concrete case: a 5-row JSON dataset, the scoped call,
and the **expected output**. Values marked `~` are approximate (real data
never fits perfectly). Every section also says **when to use the model** in
the real world.

<table>
<thead>
  <tr>
    <th>
      Page
    </th>
    
    <th>
      Examples
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <a href="/15-examples/01-linear">
        Linear
      </a>
    </td>
    
    <td>
      9 regressors & classifiers — salary, churn, feature selection, counts, curves
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/02-neighbors">
        Neighbors
      </a>
    </td>
    
    <td>
      kNN classifier & regressor
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/03-tree">
        Tree
      </a>
    </td>
    
    <td>
      DecisionTree & ExtraTree classifiers/regressors
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/04-ensemble">
        Ensemble
      </a>
    </td>
    
    <td>
      forests, boosting, IsolationForest
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/05-clustering">
        Clustering
      </a>
    </td>
    
    <td>
      KMeans, DBSCAN, HDBSCAN
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/06-monitoring">
        Monitoring
      </a>
    </td>
    
    <td>
      CUSUM, EWMA, ParallelMonitor, SeasonalMonitor
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/07-scan">
        Spatial Scan
      </a>
    </td>
    
    <td>
      Kulldorff & Getis-Ord Gi*
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/08-feature-selection">
        Feature Selection
      </a>
    </td>
    
    <td>
      mutual information, SelectFromModel
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/09-evaluation">
        Evaluation
      </a>
    </td>
    
    <td>
      compareModels, detectTask, splits, pure metrics
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/15-examples/10-metrics">
        Metric Cheat Sheet
      </a>
    </td>
    
    <td>
      what score, mse, mae, mcc… measure and when
    </td>
  </tr>
</tbody>
</table>
