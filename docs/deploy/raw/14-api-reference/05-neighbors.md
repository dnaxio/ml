# Neighbors

> KNeighborsClassifier and KNeighborsRegressor — instance-based, distance-based.

Instance-based (non-parametric) learning: predictions come from the nearest
labeled examples, no model is learned. **Distance-based** — use
`options: { scale: true }`.

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
        KNeighborsClassifier
      </code>
    </td>
    
    <td>
      <code>
        kNeighbors
      </code>
      
       (5), <code>
        weightType
      </code>
      
      , <code>
        distanceType
      </code>
      
      , <code>
        pNorm
      </code>
    </td>
    
    <td>
      + <code>
        score
      </code>
      
      , <code>
        classificationReport
      </code>
      
      , <code>
        classes
      </code>
      
      , <code>
        getParams()
      </code>
      
      /<code>
        setParams()
      </code>
      
      . No <code>
        predict_proba
      </code>
      
      . sklearn aliases accepted (<code>
        nNeighbors
      </code>
      
      /<code>
        weights
      </code>
      
      /<code>
        metric
      </code>
      
      )
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        KNeighborsRegressor
      </code>
    </td>
    
    <td>
      <code>
        nNeighbors
      </code>
      
       (5), <code>
        weights
      </code>
      
       (<code>
        'uniform'
      </code>
      
      |<code>
        'distance'
      </code>
      
      ), <code>
        metric
      </code>
      
      , <code>
        p
      </code>
    </td>
    
    <td>
      + <code>
        score
      </code>
      
      , <code>
        mse
      </code>
      
      , <code>
        mae
      </code>
      
      , <code>
        getParams()
      </code>
      
      /<code>
        setParams()
      </code>
      
      . kml aliases accepted (<code>
        kNeighbors
      </code>
      
      /<code>
        weightType
      </code>
      
      /<code>
        distanceType
      </code>
      
      )
    </td>
  </tr>
</tbody>
</table>

```ts
const knn = new neighbors.KNeighborsClassifier({ kNeighbors: 5 });
knn.fit(data, { features: ["age", "solde"], target: "achete", options: { scale: true } });
knn.predict(data); // labels
```
