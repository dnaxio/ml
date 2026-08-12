# Feature Selection

> 2 examples — mutual information ranking and SelectFromModel.

## `mutualInfoClassif` — signal before noise (nonlinear)

**When to use it** — rank features by their **nonlinear association** with the
target *before* training: find the signal, drop the noise, build a smaller
spec.

```ts
const data = [
  { signal: 1, noise: 0.1, bought: false },
  { signal: 2, noise: 0.8, bought: false },
  { signal: 8, noise: 0.3, bought: true },
  { signal: 9, noise: 0.9, bought: true },
  { signal: 5, noise: 0.4, bought: true },
];
const ranking = featureSelection.mutualInfoClassif(data, {
  features: ["signal", "noise"], target: "bought",
});
// → [
//     { feature: "signal", score: ~0.6 },  ← ranked first
//     { feature: "noise", score: ~0 },
//   ]
```

## `SelectFromModel` — keep what the Lasso finds useful

**When to use it** — keep the features a trained model actually finds useful:
automatically build a **smaller spec** (faster training, less overfitting,
easier explanations).

```ts
const data = [
  { salary: 38000, noise: 7, age: 30 },
  { salary: 46000, noise: 2, age: 35 },
  { salary: 55000, noise: 9, age: 40 },
  { salary: 71000, noise: 6, age: 50 },
  { salary: 95000, noise: 5, age: 65 },
];
const sel = new featureSelection.SelectFromModel({
  estimator: new linear.LassoRegression({ alpha: 1 }),
});
sel.fit(data, { features: ["salary", "noise"], target: "age" });
sel.selectedFeatures; // → ["salary"] — noise dropped
sel.support;          // → [true, false]
```
