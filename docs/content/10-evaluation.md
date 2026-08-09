---
title: Evaluation
description: Score your models — R², MSE, MAE, accuracy, classification reports, ROC/AUC, train/test splits and cross-validation.
navigation:
  icon: lucide:gauge
---

Every supervised model scores itself on rows that include the `target` field
(ground truth), reusing the transformation learned at fit time (rows dropped
by the `'drop'` strategy are excluded):

```ts
// Regressors → R² + MSE + MAE
reg.score(data); // number — R² (1 = perfect fit)
reg.mse(data); // number — mean squared error (0 = perfect)
reg.mae(data); // number — mean absolute error (0 = perfect)

// Classifiers → accuracy + full report
clf.score(data); // number — accuracy
clf.classificationReport(data); // { accuracy, precision, recall, fScore,
//   support, confusionMatrix }

// Classifiers with predict_proba (LogisticRegression, ExtraTreeClassifier,
// AdaBoostClassifier, GradientBoostingClassifier, XGBoostClassifier) → ROC
clf.rocAucScore(data); // number — AUC (1 = perfect, 0.5 = random, binary only)
```

Implemented on: 13 regressors (`score`/`mse`/`mae`) and 8 classifiers
(`score`/`classificationReport`, + `rocAucScore` on the 5 with
`predict_proba`). Clustering, IsolationForest, monitoring and scan have no
target → no `score`.

The ground truth convention matches kml: **predictions first, truth second**.

## Splitting & cross-validation

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

## Streaming (big-data inference)

`predictStream` and `fillPredictStream` process rows **in chunks**, so you can
run inference over millions of rows (file, database, API) without loading
them all into memory. They accept arrays, generators and async generators:

```ts
import { predictStream, fillPredictStream } from "@dnax/ml";

// read rows from anywhere (NDJSON file, DB cursor, API)
async function* rows() {
  for await (const line of Bun.stdin.stream()) {
    yield JSON.parse(line.toString());
  }
}

// stream predictions, one per row
for await (const pred of predictStream(reg, rows(), { chunkSize: 1000 })) {
  // → one prediction per row, as they arrive
}

// stream enriched rows (like fill_predict, memory bounded)
await fillPredictStream(reg, rows(), (row) => {
  console.log(row); // { ..., target: prediction }
});
```

Works with **every supervised model** (one generic implementation — no
per-model duplication). `chunkSize` controls the memory/compute trade-off.

## Reading the scores

- **R²**: 1 = perfect, 0 = no better than predicting the mean, negative =
  worse than the mean. Compare models with the CV **mean**, and check the
  fold spread (a stable model has tight folds).
- **AUC**: threshold-free ranking quality — 1 = perfect separation,
  0.5 = random.
- On small datasets, single-fold scores are noisy: prefer the mean of a
  repeated k-fold over one arbitrary split.
