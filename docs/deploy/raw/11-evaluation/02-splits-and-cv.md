# Splits & Cross-validation

> trainTestSplit and crossValScore — measure generalization on unseen rows.

```ts
import { trainTestSplit, crossValScore } from "@dnax/ml";

// Train/test split (JSON-first, seeded, stratified by a field)
const { train, test } = trainTestSplit(data, {
  testSize: 0.3,
  shuffle: true,
  randomState: 42,
  stratify: "achete", // preserves class proportions in both folds
});
reg.fit(train, { features: ["x"], target: "y" });
const r2 = reg.score(test);

// K-fold cross-validation — compare models without one arbitrary split
const scores = crossValScore(
  () => new LinearRegression(), // fresh model per fold
  data,
  { features: ["x"], target: "y" },
  { cv: 5, scoring: "score", randomState: 42 }, // 'score' | 'mse' | 'rocAucScore'
);
// → [0.91, 0.88, 0.93, ...] — one score per fold
```

`crossValScore` stratifies automatically when the target is discrete with
enough samples per class, and falls back to plain k-fold otherwise.
