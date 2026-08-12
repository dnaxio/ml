# Overview

> Rank or select the informative columns before training — univariate scores and model-based selection on JSON rows.

> **Scoped vs flat import** — scoped: `featureSelection.chi2(data, spec)` ·
> flat: `import { chi2 } from "@dnax/ml"`. Both are identical.

Feature selection removes **weak or redundant columns before training**. It
reduces cost, improves interpretability, and limits overfitting. Two
families: **univariate scoring** (each feature scored independently against
the target) and **model-based selection** (keep what a trained model finds
important).

**⚠️ Numeric features only**: the scores stay *per feature*, so one-hot is
not applied. Boolean targets are encoded automatically (→ 1/0).
`options.missing` is honored (`'throw'` default | `'fill0'` | `'drop'` whole
row). Features are sorted alphabetically, matching the transformer layout.

<table>
<thead>
  <tr>
    <th>
      Page
    </th>
    
    <th>
      What it covers
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <a href="/12-feature-selection/01-univariate">
        Univariate scoring
      </a>
    </td>
    
    <td>
      <code>
        chi2
      </code>
      
      , <code>
        fClassif
      </code>
      
      , <code>
        mutualInfoClassif
      </code>
      
      , <code>
        mutualInfoRegression
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/12-feature-selection/02-select-from-model">
        SelectFromModel
      </a>
    </td>
    
    <td>
      model-based selection with a threshold
    </td>
  </tr>
</tbody>
</table>

## Tips

- **Rank first, then model**: start with `mutualInfo` (non-linear) to discard
obvious noise, then confirm with `SelectFromModel` before the final fit.
- **Feature importance is not causality** — it ranks association strength,
which is exactly what you want for column pruning.
- Compare models before/after reduction with
[cross-validation](/11-evaluation): fewer columns should not hurt the CV
score, and usually improves it on small datasets.
