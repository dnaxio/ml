---
title: SelectFromModel
description: Keep the features a trained model finds useful — threshold + optional top-k cap.
---

Fits an estimator (any model exposing `featureImportances` — trees/ensemble —
or `coef` — linear models), reads its per-feature importance, and keeps the
features above a threshold (`'mean'` | `'median'` | an explicit number),
optionally capped to the top-k (`maxFeatures`).

```ts
import { featureSelection, LassoRegression } from "@dnax/ml";

const sel = new featureSelection.SelectFromModel({
  estimator: new LassoRegression({ alpha: 1 }),
  threshold: "mean",
  maxFeatures: 5, // optional hard cap
});
sel.fit(data, { features: ["salaire", "bruit"], target: "age" });

sel.selectedFeatures; // ["salaire"] — ready for a fresh fit spec
sel.support;          // [true, false] — boolean mask aligned with the features
sel.featureScores;    // per-feature importances (|coef| mean, or featureImportances)
sel.fittedEstimator;  // the trained estimator
```

**Notes**

- `options.oneHot` must stay **off** (throws otherwise): each importance
  value must map to exactly one `spec.features` entry.
- For linear classifiers, importance = mean `|coef|` across classes.
- `SelectFromModel` is a selection **utility** (like `trainTestSplit`) — no
  `export`/`load`; persist the base estimator instead, then re-select.
