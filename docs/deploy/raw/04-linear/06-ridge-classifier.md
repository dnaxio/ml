# Ridge Classifier

> Fast L2-regularized linear classifier — labels only, no probabilities.

L2-regularized linear classifier (one-vs-rest ridge models). Fast and
interpretable on numeric tabular data; binary and multiclass. **No
predict_proba** — labels only.

**When to use it** — a fast **label-only** classifier: quick baseline, or a
yes/no decision on large data where a probability is not needed.

```ts
import { linear } from "@dnax/ml";

const clf = new linear.RidgeClassifier({ alpha: 1 });
clf.fit(data, { features: ["note", "heures"], target: "admis" });
clf.predict(data);      // labels
clf.fill_predict(data); // rows + predicted label
clf.coef;               // matrix (one row of weights per class)
```

Methods: `predict`, `fill_predict`, `score` (accuracy), `classificationReport`,
`classes`, `coef`, `intercept`, `getParams()` / `setParams()`,
`export` / `load`.
