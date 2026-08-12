# Feature Selection

> Univariate scoring and SelectFromModel — API reference.

Rank or select the informative columns before training (lower cost, better
interpretability, less overfitting) — full guide on
[Feature selection](/12-feature-selection). Features must be **numeric or
boolean** (one-hot is not applied — the scores stay per feature);
`options.missing` is honored. `spec` is a `JsonFitSpec`.

Univariate scoring (`featureSelection.*`): one result per feature, **sorted
by score** (most informative first). `chi2` / `fClassif` are for
classification targets and include a `pValue`; `mutualInfoClassif` /
`mutualInfoRegression` capture **non-linear** links (no p-value).

```ts
import { featureSelection } from "@dnax/ml";

const ranking = featureSelection.mutualInfoClassif(data, {
  features: ["age", "solde", "nb_visites", "inutile"],
  target: "achete",
});
// → [{ feature: "age", score: 0.42 }, ...] — pick the top k, feed a new spec
```

Model-based selection (`featureSelection.SelectFromModel`): fits an
estimator (`coef` from linear models, or `featureImportances` from
trees/ensemble), keeps the features above a threshold (`'mean'` |
`'median'` | number, optional `maxFeatures` top-k).

```ts
const sel = new featureSelection.SelectFromModel({
  estimator: new LassoRegression({ alpha: 1 }),
  threshold: "mean",
});
sel.fit(data, { features: ["salaire", "bruit"], target: "age" });
sel.selectedFeatures; // ["salaire"] — ready for a fresh fit spec
sel.support;          // [true, false]
sel.featureScores;    // per-feature importances
sel.fittedEstimator;  // the trained estimator
```
