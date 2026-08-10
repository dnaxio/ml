---
title: RandomForestClassifier
description: Many trees combined by majority vote — robust tabular default.
---

Many decision trees combined by **majority vote**. More robust and accurate
than a single tree. Note: kml's `RandomForestClassifier` does **not** expose
`predict_proba`.

**When to use it** — the **robust default** for tabular classification:
handles nonlinearity and feature interactions with little tuning. Labels only.

```ts
import { ensemble } from "@dnax/ml";

const clf = new ensemble.RandomForestClassifier({
  nEstimators: 100,
  max_depth: 10,
  randomState: 42,
});
clf.fit(data, { features: ["age", "solde"], target: "achete" });
```

Methods: `predict`, `fill_predict`, `score` (accuracy), `classificationReport`,
`featureImportances`, `export` / `load`.
