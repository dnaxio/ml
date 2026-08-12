# KNeighborsRegressor

> Predict by averaging the target values of the k nearest labeled examples.

Each prediction is the **average** (or inverse-distance weighted average) of
the target values of the `nNeighbors` nearest labeled examples.

**When to use it** — same principle for a **continuous target**: the prediction
is the average of the k nearest known values. Ideal for very local patterns
(price per m² in a neighborhood, delivery time per route). Distance-based →
`scale: true` is required.

```ts
import { neighbors } from "@dnax/ml";

const reg = new neighbors.KNeighborsRegressor({ nNeighbors: 5 });
reg.fit(employees, {
  features: ["age", "annees_exp"],
  target: "salaire",
  options: { scale: true },
});

reg.predict(newEmployees);  // predicted values
reg.score(test);            // R²
reg.mae(test);              // mean absolute error
```

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
        nNeighbors
      </code>
    </td>
    
    <td>
      <code>
        5
      </code>
    </td>
    
    <td>
      number of nearest neighbors averaged
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        weights
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      <code>
        'uniform'
      </code>
      
       (plain average) or <code>
        'distance'
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        metric
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      distance metric (<code>
        'euclidean'
      </code>
      
      , <code>
        'manhattan'
      </code>
      
      , ...)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        p
      </code>
    </td>
    
    <td>
      <code>
        2
      </code>
    </td>
    
    <td>
      Minkowski p-norm (only for <code>
        'minkowski'
      </code>
      
      )
    </td>
  </tr>
</tbody>
</table>

kml-style aliases are accepted: `kNeighbors`, `weightType` and `distanceType`
map to `nNeighbors`, `weights` and `metric`.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`getParams()` / `setParams()`, `export` / `load`.
