# Tree

> DecisionTree and ExtraTree classifiers and regressors.

Common surface: `fit`, `predict`, `fill_predict`, `score`, `export`, `load`,
`featureImportances`.

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
      Notes
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        DecisionTreeClassifier
      </code>
    </td>
    
    <td>
      <code>
        max_depth
      </code>
      
      , <code>
        min_samples_split
      </code>
      
      , <code>
        criterion
      </code>
      
      , <code>
        max_features
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
        DecisionTreeRegressor
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
        ExtraTreeClassifier
      </code>
    </td>
    
    <td>
      same
    </td>
    
    <td>
      + <code>
        predict_proba
      </code>
      
      , <code>
        fill_predict_proba
      </code>
      
      , <code>
        rocAucScore
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        ExtraTreeRegressor
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
