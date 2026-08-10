---
title: DecisionTreeClassifier
description: Classify with readable if/then rules — labels only, no probabilities.
---

Splits the feature space into **if/else rules** — the most interpretable
classifier in the SDK. **When to use it** — you need rules you can explain to
a human (regulatory audits, medical triage, loan decisions). Best with a small
set of features.

```ts
import { tree } from "@dnax/ml";

const clf = new tree.DecisionTreeClassifier({ max_depth: 3 });
clf.fit(data, { features: ["age", "solde"], target: "achete" });
clf.predict(data);           // labels
clf.fill_predict(data);      // rows + predicted label
clf.featureImportances;      // how much each feature drives the splits
```

Note: kml's `DecisionTreeClassifier` has **no** `predict_proba` — labels only.

Methods: `predict`, `fill_predict`, `score` (accuracy), `classificationReport`,
`featureImportances`, `export` / `load`.
