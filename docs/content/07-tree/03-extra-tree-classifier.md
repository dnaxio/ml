---
title: ExtraTreeClassifier
description: Extremely randomized tree classifier — with probabilities.
---

Extremely randomized trees: split thresholds are chosen **randomly** instead
of optimally. Faster and often more robust against overfitting. Exposes
`predict_proba` / `fill_predict_proba`.

**When to use it** — same family but **randomized** and it **gives
probabilities** (`predict_proba`, `rocAucScore`): use when you need a
probability output from a cheap tree model.

```ts
import { tree } from "@dnax/ml";

const clf = new tree.ExtraTreeClassifier({ max_depth: 5 });
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict_proba(data);          // per-class probabilities
clf.fill_predict_proba(data);     // rows + probability in the target
```

Methods: `predict`, `predict_proba`, `fill_predict`, `fill_predict_proba`,
`score` (accuracy), `classificationReport`, `rocAucScore`,
`featureImportances`, `export` / `load`.
