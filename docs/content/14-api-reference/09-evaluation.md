---
title: Evaluation
description: Splits, cross-validation, leaderboard, streaming and pure metrics — API reference.
---

Functions are exported flat **and** through the `evaluation` namespace
(`evaluation.compareModels`, `evaluation.mcc`, … — same functions):

```ts
trainTestSplit(data, { testSize?, trainSize?, shuffle?, randomState?, stratify? });
// → { train: JsonRow[], test: JsonRow[] }

crossValScore(() => new LinearRegression(), data, spec, { cv?, scoring?, stratify?, randomState? });
// → number[] — one score per fold

compareModels({ linear: () => new LinearRegression(), ridge: () => new RidgeRegression({ alpha: 1 }) }, data, spec,
  { cv?, scoring?, stratify?, randomState? });
// → ModelBenchmark[] — { name, mean, std, scores }[] sorted best first

detectTask(data, spec); // → 'classification' | 'regression'

// Streaming inference (memory bounded, works with generators/async iterables)
for await (const pred of predictStream(model, rows, { chunkSize? })) { /* ... */ }
await fillPredictStream(model, rows, (row) => { /* ... */ }, { chunkSize? });

meanAbsoluteError(preds, truth);                    // MAE
fbetaFromPrecisionRecall(precision, recall, beta);  // Fβ from raw values (β = 1 → F1)
```

Supervised models add:

| Method | Returns | Availability |
| ------ | ------- | ------------ |
| `score(data)` | `number` — R² (regressors) / accuracy (classifiers) | all 23 supervised models |
| `mse(data)` | `number` — mean squared error | 14 regressors |
| `mae(data)` | `number` — mean absolute error | 14 regressors |
| `classificationReport(data, beta?)` | `{ accuracy, precision, recall, fScore, support, confusionMatrix }` — `fScore` is Fβ (`beta` defaults to 1 = F1, β > 1 favors recall) | 9 classifiers |
| `rocAucScore(data)` | `number` — AUC (binary) | 5 classifiers with `predict_proba` |

Pure metric functions (arrays in, number/curve out):

```ts
// Regression — predictions first, truth second
rmse(preds, truth);              // √MSE, target units
mape(preds, truth);              // mean absolute percentage error
medianAbsoluteError(preds, truth);

// Classification
mcc(preds, truth);               // Matthews correlation coefficient (−1..+1)
balancedAccuracy(preds, truth);  // macro recall

// Probability metrics — truth first (kml style), proba = positive class
logLoss(truth, proba);           // cross-entropy
prAucScore(truth, proba);        // precision-recall AUC (imbalance-friendly)
rocCurve(truth, proba);          // → { fpr, tpr, thresholds }
optimalThreshold(truth, proba);  // Youden's J — best decision threshold
```
