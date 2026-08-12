# Overview

> Regression and classification on linear relationships — 9 models sharing one JSON-first API.

> **Scoped vs flat import** — scoped: `new linear.LinearRegression()` ·
> flat: `import { LinearRegression } from "@dnax/ml"`. Both are identical.

All linear models share the same JSON-first API: `fit`, `predict`,
`fill_predict`, `coef`, `intercept`, `getParams()` / `setParams()`,
`predictAsync`, `score`, `mse`, `mae` and `export` / `load`.

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
      <a href="/04-linear/01-linear-regression">
        Linear Regression
      </a>
    </td>
    
    <td>
      ordinary least squares — the baseline
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/02-logistic-regression">
        Logistic Regression
      </a>
    </td>
    
    <td>
      classification with probabilities
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/03-ridge-regression">
        Ridge Regression
      </a>
    </td>
    
    <td>
      L2 — stable with correlated features
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/04-lasso-regression">
        Lasso Regression
      </a>
    </td>
    
    <td>
      L1 — feature selection
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/05-elastic-net">
        Elastic Net
      </a>
    </td>
    
    <td>
      L1 + L2 combined
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/06-ridge-classifier">
        Ridge Classifier
      </a>
    </td>
    
    <td>
      fast label-only classifier
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/07-ransac-regressor">
        RANSAC Regressor
      </a>
    </td>
    
    <td>
      robust to outliers
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/08-poisson-regressor">
        Poisson Regressor
      </a>
    </td>
    
    <td>
      count targets, predictions ≥ 0
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/04-linear/09-polynomial-regression">
        Polynomial Regression
      </a>
    </td>
    
    <td>
      nonlinear curves
    </td>
  </tr>
</tbody>
</table>

`predictAsync(data)` is available on every regressor: it offloads the
prediction to a worker so the event loop stays responsive on large sets.

```ts
const preds = await reg.predictAsync(bigData);
```
