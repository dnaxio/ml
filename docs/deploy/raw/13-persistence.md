# Overview

> Export and load models, inject parameters, and offload predictions.

> **Scoped vs flat import** — scoped: `new linear.LinearRegression()` ·
> flat: `import { LinearRegression } from "@dnax/ml"`. Both are identical.

Every model in the SDK supports `export` / `load` — including the monitoring
models (CUSUM, EWMA, ParallelMonitor, SeasonalMonitor), the spatial scan, and
the clusters.

<table>
<thead>
  <tr>
    <th>
      Page
    </th>
    
    <th>
      What it covers
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <a href="/13-persistence/01-persistence">
        Persistence
      </a>
    </td>
    
    <td>
      <code>
        export
      </code>
      
       / <code>
        load
      </code>
      
       / restore from params
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/13-persistence/02-parameters">
        Model parameters
      </a>
    </td>
    
    <td>
      <code>
        coef
      </code>
      
      , <code>
        intercept
      </code>
      
      , <code>
        getParams
      </code>
      
      , <code>
        setParams
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/13-persistence/03-predict-async">
        predictAsync
      </a>
    </td>
    
    <td>
      worker offload
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/13-persistence/04-utilities">
        Utilities
      </a>
    </td>
    
    <td>
      <code>
        columnNames
      </code>
      
      , <code>
        droppedRows
      </code>
    </td>
  </tr>
</tbody>
</table>
