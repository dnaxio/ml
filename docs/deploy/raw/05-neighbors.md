# Overview

> Instance-based learning — KNeighborsClassifier and KNeighborsRegressor predict from the nearest labeled examples.

> **Scoped vs flat import** — scoped: `new neighbors.KNeighborsClassifier()` ·
> flat: `import { KNeighborsClassifier } from "@dnax/ml"`. Both are identical.

Nearest-neighbor models are **instance-based**: no model is learned, the
training rows are memorized and each prediction comes from the **nearest
labeled examples**. They make an excellent non-parametric baseline — if kNN
beats your parametric models, the relationship is very local.

**⚠️ Distance-based**: normalize the features (`options: { scale: true }`) or
the columns with the largest magnitude will dominate the distances.

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
      <a href="/05-neighbors/01-kneighbors-classifier">
        KNeighborsClassifier
      </a>
    </td>
    
    <td>
      majority vote of the k nearest labeled examples
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/05-neighbors/02-kneighbors-regressor">
        KNeighborsRegressor
      </a>
    </td>
    
    <td>
      average of the k nearest target values
    </td>
  </tr>
</tbody>
</table>

## Tips

- **kNN is not an interpolator**: a training point predicts the average/vote
of itself + its neighbors, so the training error is never ~0. Use a
hold-out set (see [Evaluation](/11-evaluation)) to judge quality.
- **Choose k by validation**: small `k` = flexible but noisy, large `k` =
smooth but blurry. Compare `score` across a few values with
`crossValScore`.
- **'distance' weights** help when the nearest neighbor is much closer than
the others.
- **getParams() / setParams()** work on both models (sklearn-style).
`setParams` rebuilds the model **unfitted** — call `fit` again before
`predict`; unknown keys throw.
- Both models persist with `export` / `load` (training rows are saved).
