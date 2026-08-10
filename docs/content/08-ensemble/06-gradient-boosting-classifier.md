---
title: GradientBoostingClassifier
description: Sequential trees fitted on residuals — the strongest tabular default.
---

Sequential trees fitted on the **residuals** of the previous ensemble, with a
small learning rate. Calibrated probabilities.

**When to use it** — the **strongest off-the-shelf tabular classifier** with
calibrated probabilities: use when accuracy matters most and you can tune
`learningRate` / `maxDepth`.

```ts
import { ensemble } from "@dnax/ml";

const clf = new ensemble.GradientBoostingClassifier({
  nEstimators: 100,
  learningRate: 0.1, // key lever: low + many trees = robust
  maxDepth: 3,       // keep 2-4
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data); // calibrated per-class probabilities
```

Methods: `predict`, `predict_proba`, `fill_predict`, `fill_predict_proba`,
`score` (accuracy), `classificationReport`, `rocAucScore`,
`featureImportances`, `export` / `load`.
