# Univariate Scoring

> Score each feature independently against the target — linear and nonlinear measures.

One result per feature, **sorted by score** (most informative first):

<table>
<thead>
  <tr>
    <th>
      Function
    </th>
    
    <th>
      Target
    </th>
    
    <th>
      Returns
    </th>
    
    <th>
      Captures
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        chi2(data, spec)
      </code>
    </td>
    
    <td>
      classification
    </td>
    
    <td>
      score + <code>
        pValue
      </code>
    </td>
    
    <td>
      linear (requires non-negative features)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        fClassif(data, spec)
      </code>
    </td>
    
    <td>
      classification
    </td>
    
    <td>
      score + <code>
        pValue
      </code>
    </td>
    
    <td>
      linear (ANOVA F)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        mutualInfoClassif(data, spec, opts?)
      </code>
    </td>
    
    <td>
      classification
    </td>
    
    <td>
      score
    </td>
    
    <td>
      <strong>
        non-linear
      </strong>
      
       links
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        mutualInfoRegression(data, spec, opts?)
      </code>
    </td>
    
    <td>
      regression
    </td>
    
    <td>
      score
    </td>
    
    <td>
      <strong>
        non-linear
      </strong>
      
       links
    </td>
  </tr>
</tbody>
</table>

```ts
import { featureSelection } from "@dnax/ml";

const ranking = featureSelection.mutualInfoClassif(data, {
  features: ["age", "solde", "nb_visites", "inutile"],
  target: "achete",
});
// → [
//     { feature: "age", score: 0.42 },
//     { feature: "nb_visites", score: 0.18 },
//     { feature: "solde", score: 0.09 },
//     { feature: "inutile", score: 0.01 },
//   ]  — most informative first

const top3 = ranking.slice(0, 3).map((r) => r.feature);
clf.fit(data, { features: top3, target: "achete" });
```

`mutualInfo*` options: `discreteFeatures` (boolean or per-feature), `nNeighbors`
(default 5), `randomState`.
