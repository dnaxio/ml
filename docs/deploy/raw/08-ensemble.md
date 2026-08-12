# Overview

> Random forests, AdaBoost, Gradient Boosting and XGBoost — plus IsolationForest for anomaly detection.

> **Scoped vs flat import** — scoped: `new ensemble.RandomForestClassifier()` ·
> flat: `import { RandomForestClassifier } from "@dnax/ml"`. Both are identical.

Ensembles combine many weak models into one strong model. They expose
`featureImportances` instead of `coef` / `intercept`, plus the evaluation
methods (`score`, `classificationReport`, `rocAucScore` where applicable).

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
      <a href="/08-ensemble/01-isolation-forest">
        IsolationForest
      </a>
    </td>
    
    <td>
      unsupervised anomaly detection
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/02-random-forest-classifier">
        RandomForestClassifier
      </a>
    </td>
    
    <td>
      bagged trees — robust default, labels only
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/03-random-forest-regressor">
        RandomForestRegressor
      </a>
    </td>
    
    <td>
      bagged trees for regression
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/04-adaboost-classifier">
        AdaBoostClassifier
      </a>
    </td>
    
    <td>
      sequential boosting of stumps
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/05-adaboost-regressor">
        AdaBoostRegressor
      </a>
    </td>
    
    <td>
      sequential boosting for regression
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/06-gradient-boosting-classifier">
        GradientBoostingClassifier
      </a>
    </td>
    
    <td>
      the strongest tabular default
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/07-gradient-boosting-regressor">
        GradientBoostingRegressor
      </a>
    </td>
    
    <td>
      gradient boosting for regression
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/08-xgboost-classifier">
        XGBoostClassifier
      </a>
    </td>
    
    <td>
      regularized boosting — large data
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/08-ensemble/09-xgboost-regressor">
        XGBoostRegressor
      </a>
    </td>
    
    <td>
      regularized boosting for regression
    </td>
  </tr>
</tbody>
</table>

## Common parameters (XGBoost)

<table>
<thead>
  <tr>
    <th>
      Param (XGBoost)
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
        nEstimators
      </code>
    </td>
    
    <td>
      <code>
        100
      </code>
    </td>
    
    <td>
      boosting rounds
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        learningRate
      </code>
    </td>
    
    <td>
      <code>
        0.3
      </code>
    </td>
    
    <td>
      step size (lower + more trees = robust)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        maxDepth
      </code>
    </td>
    
    <td>
      <code>
        6
      </code>
    </td>
    
    <td>
      tree depth
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        lambda
      </code>
    </td>
    
    <td>
      <code>
        1
      </code>
    </td>
    
    <td>
      L2 regularization on weights
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        gamma
      </code>
    </td>
    
    <td>
      <code>
        0
      </code>
    </td>
    
    <td>
      min loss reduction for a split
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        minChildWeight
      </code>
    </td>
    
    <td>
      <code>
        1
      </code>
    </td>
    
    <td>
      min sum of weights in a child
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        subsample
      </code>
    </td>
    
    <td>
      <code>
        1
      </code>
    </td>
    
    <td>
      fraction of rows per round
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        colsampleByTree
      </code>
    </td>
    
    <td>
      <code>
        1
      </code>
    </td>
    
    <td>
      fraction of features per tree
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        baseScore
      </code>
    </td>
    
    <td>
      <code>
        0.5
      </code>
    </td>
    
    <td>
      initial prediction score
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        randomState
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      seed for reproducible fits
    </td>
  </tr>
</tbody>
</table>
