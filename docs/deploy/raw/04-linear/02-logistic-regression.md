# Logistic Regression

> Binary and multiclass classification with probabilities.

Binary (and multiclass) classification. A boolean `target` is encoded
automatically: `false` → 0, `true` → 1. Adds `predict_proba` and
`fill_predict_proba` for probabilities, plus `rocAucScore`.

**When to use it** — the answer is a **yes/no** (or a class) and you want the
**probability**, not just the label: churn, fraud, will-buy, disease risk.

```ts
import { linear } from "@dnax/ml";

const clients = [
  { age: 20, solde: 100, achete: false },
  { age: 35, solde: 5000, achete: true },
  // ...
];

const clf = new linear.LogisticRegression();
clf.fit(clients, { features: ["age", "solde"], target: "achete" });

const labels = clf.predict([{ age: 38, solde: 7000 }]); // [1]
const probs = clf.predict_proba([{ age: 38, solde: 7000 }]); // [[~0, ~1]]
```

Methods: `predict`, `predict_proba`, `fill_predict`, `fill_predict_proba`,
`score` (accuracy), `classificationReport`, `rocAucScore`, `coef`,
`intercept`, `getParams()` / `setParams()`, `export` / `load`.
