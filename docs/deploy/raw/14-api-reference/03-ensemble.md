# Ensemble

> Random forests, AdaBoost, Gradient Boosting, XGBoost and IsolationForest.

Common surface: `fit`, `predict`, `fill_predict`, `score`, `export`, `load`,
`featureImportances`.

<table>
<thead>
  <tr>
    <th>
      Class
    </th>
    
    <th>
      Key params
    </th>
    
    <th>
      Notes
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        IsolationForest
      </code>
    </td>
    
    <td>
      <code>
        subsampling_size
      </code>
      
      , <code>
        tree_num
      </code>
      
      , <code>
        contamination
      </code>
      
      , <code>
        random_state
      </code>
    </td>
    
    <td>
      unsupervised anomaly; <code>
        predict
      </code>
      
       (1 = anomaly) + <code>
        anomaly_score(row)
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        RandomForestClassifier
      </code>
    </td>
    
    <td>
      <code>
        nEstimators
      </code>
      
      , <code>
        maxDepth
      </code>
      
      , <code>
        maxFeatures
      </code>
      
      , <code>
        criterion
      </code>
      
      , <code>
        randomState
      </code>
    </td>
    
    <td>
      no <code>
        predict_proba
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        RandomForestRegressor
      </code>
    </td>
    
    <td>
      same
    </td>
    
    <td>
      + <code>
        mse
      </code>
      
      , <code>
        mae
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        AdaBoostClassifier
      </code>
    </td>
    
    <td>
      <code>
        nEstimators
      </code>
      
      , <code>
        learningRate
      </code>
      
      , <code>
        randomState
      </code>
    </td>
    
    <td>
      + proba + <code>
        rocAucScore
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        AdaBoostRegressor
      </code>
    </td>
    
    <td>
      same
    </td>
    
    <td>
      + <code>
        mse
      </code>
      
      , <code>
        mae
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        GradientBoostingClassifier
      </code>
    </td>
    
    <td>
      <code>
        nEstimators
      </code>
      
      , <code>
        learningRate
      </code>
      
      , <code>
        maxDepth
      </code>
      
      , <code>
        minSamplesSplit
      </code>
      
      , <code>
        subsample
      </code>
      
      , <code>
        maxFeatures
      </code>
      
      , <code>
        randomState
      </code>
    </td>
    
    <td>
      + proba + <code>
        rocAucScore
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        GradientBoostingRegressor
      </code>
    </td>
    
    <td>
      same
    </td>
    
    <td>
      + <code>
        mse
      </code>
      
      , <code>
        mae
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        XGBoostClassifier
      </code>
    </td>
    
    <td>
      <code>
        nEstimators
      </code>
      
      , <code>
        learningRate
      </code>
      
      , <code>
        maxDepth
      </code>
      
      , <code>
        lambda
      </code>
      
      , <code>
        gamma
      </code>
      
      , <code>
        minChildWeight
      </code>
      
      , <code>
        subsample
      </code>
      
      , <code>
        colsampleByTree
      </code>
      
      , <code>
        baseScore
      </code>
      
      , <code>
        randomState
      </code>
    </td>
    
    <td>
      + proba + <code>
        rocAucScore
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        XGBoostRegressor
      </code>
    </td>
    
    <td>
      same
    </td>
    
    <td>
      + <code>
        mse
      </code>
      
      , <code>
        mae
      </code>
    </td>
  </tr>
</tbody>
</table>
