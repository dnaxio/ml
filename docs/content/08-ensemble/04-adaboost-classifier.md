---
title: AdaBoostClassifier
description: Sequential boosting of stumps, each correcting the previous mistakes.
---

Fits **stumps** (1-level trees) sequentially, each correcting the previous
mistakes (weighted resampling). Exposes calibrated `predict_proba`.

**When to use it** — sequential boosting of weak learners. Try it when
`RandomForestClassifier` underperforms; it returns **probabilities** and works
well on moderate-size tabular data.

```ts
import { ensemble } from "@dnax/ml";

const clf = new ensemble.AdaBoostClassifier({
  nEstimators: 100,
  learningRate: 1.0,
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data); // per-class probabilities
```

> **Pitfall**: AdaBoost stumps cannot converge on a perfectly symmetric XOR
> pattern — probabilities stay near `[0.5, 0.5]`. Use linearly segmentable
> data.

Methods: `predict`, `predict_proba`, `fill_predict`, `fill_predict_proba`,
`score` (accuracy), `classificationReport`, `rocAucScore`,
`featureImportances`, `export` / `load`.
