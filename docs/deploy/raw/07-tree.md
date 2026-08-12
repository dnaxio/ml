# Overview

> Interpretable decision trees — DecisionTree and ExtraTree classifiers and regressors.

> **Scoped vs flat import** — scoped: `new tree.DecisionTreeClassifier()` ·
> flat: `import { DecisionTreeClassifier } from "@dnax/ml"`. Both are identical.

Trees split the feature space into **if/else rules** — the most interpretable
models in the SDK. They share a common JSON-first API and expose
`featureImportances` (how much each feature drives the splits) instead of
`coef` / `intercept`.

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
      <a href="/07-tree/01-decision-tree-classifier">
        DecisionTreeClassifier
      </a>
    </td>
    
    <td>
      readable if/then rules — labels only
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/07-tree/02-decision-tree-regressor">
        DecisionTreeRegressor
      </a>
    </td>
    
    <td>
      stepwise predictions (leaf mean)
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/07-tree/03-extra-tree-classifier">
        ExtraTreeClassifier
      </a>
    </td>
    
    <td>
      randomized tree with probabilities
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/07-tree/04-extra-tree-regressor">
        ExtraTreeRegressor
      </a>
    </td>
    
    <td>
      randomized tree for continuous targets
    </td>
  </tr>
</tbody>
</table>

## Common parameters

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
        max_depth
      </code>
    </td>
    
    <td>
      <code>
        Infinity
      </code>
    </td>
    
    <td>
      tree depth (limit to avoid overfitting)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        min_samples_split
      </code>
    </td>
    
    <td>
      <code>
        2
      </code>
    </td>
    
    <td>
      min samples required to keep splitting
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        criterion
      </code>
    </td>
    
    <td>
      <code>
        'entropy'
      </code>
    </td>
    
    <td>
      impurity criterion (<code>
        'gini'
      </code>
      
       available)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        max_features
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      features per split (<code>
        number
      </code>
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
      seed for reproducible random feature selection
    </td>
  </tr>
</tbody>
</table>

All tree models also expose the evaluation methods (`score` for accuracy/R²,
`mse`/`mae` for regressors, `classificationReport` for classifiers) and full
persistence (`export` / `load`).
