# Linear

> 9 regression & classification models — constructor params and methods.

Common surface: `fit`, `predict`, `fill_predict`, `predictAsync`, `score`
(R²), `mse`, `mae`, `getParams()`, `setParams()`, `export`, `load`, `coef`,
`intercept`.

<table>
<thead>
  <tr>
    <th>
      Class
    </th>
    
    <th>
      Constructor params
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
        LinearRegression
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      ordinary least squares
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        RidgeRegression
      </code>
    </td>
    
    <td>
      <code>
        alpha
      </code>
      
       (1)
    </td>
    
    <td>
      L2 regularization
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        LassoRegression
      </code>
    </td>
    
    <td>
      <code>
        alpha
      </code>
      
       (1)
    </td>
    
    <td>
      L1, zeroes uninformative features
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        ElasticNet
      </code>
    </td>
    
    <td>
      <code>
        alpha
      </code>
      
       (1), <code>
        l1Ratio
      </code>
      
       (0.5)
    </td>
    
    <td>
      L1 + L2
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        LogisticRegression
      </code>
    </td>
    
    <td>
      —
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
      
      ; binary & multiclass
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        RidgeClassifier
      </code>
    </td>
    
    <td>
      <code>
        alpha
      </code>
      
       (1)
    </td>
    
    <td>
      labels only, no <code>
        predict_proba
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        RANSACRegressor
      </code>
    </td>
    
    <td>
      <code>
        randomState
      </code>
    </td>
    
    <td>
      + <code>
        inlierMask
      </code>
      
      , <code>
        nTrials
      </code>
      
       getters
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        PoissonRegressor
      </code>
    </td>
    
    <td>
      <code>
        alpha
      </code>
      
       (1)
    </td>
    
    <td>
      counts, predictions <code>
        >= 0
      </code>
      
      ; + <code>
        nIter
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        PolynomialRegression
      </code>
    </td>
    
    <td>
      <code>
        degree
      </code>
      
       (2)
    </td>
    
    <td>
      + <code>
        degree
      </code>
      
       getter
    </td>
  </tr>
</tbody>
</table>

```ts
const reg = new linear.RidgeRegression({ alpha: 1 });
reg.fit(data, { features: ["x"], target: "y" });
reg.score(test);        // R²
reg.mse(test);          // MSE
reg.mae(test);          // MAE
await reg.predictAsync(big); // worker offload
```
