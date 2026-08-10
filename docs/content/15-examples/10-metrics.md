---
title: Metric Cheat Sheet
description: What score, mse, mae, rmse, mape, mcc, prAucScore… measure, and when to prefer each one.
---

Model methods are available on the models themselves (`reg.score(data)`,
`clf.mae(data)`…); the pure functions are exported from the package root and
called `evaluation.xxx(...)`. Convention: **predictions first, truth second**
for label/regression metrics; **truth first** for probability/curve metrics
(kml style).

## Regression — `score`, `mse`, `mae`, `rmse`, `mape`, `medianAbsoluteError`

| Metric | What it measures | When to prefer it |
| ------ | ---------------- | ----------------- |
| `score` (R²) | share of variance explained — 1 = perfect, 0 = no better than the mean | **overall fit quality**, comparing models |
| `mse` | mean squared error — 0 = perfect; large errors weigh a lot | **big mistakes are costly** (a 100-unit error is worse than 10× 10-unit) |
| `rmse` | √MSE — same **units as the target** | you want the typical error in euros/days/sales |
| `mae` | mean absolute error — same units, **outlier-robust** | the typical error you will actually see in practice |
| `mape` | average **% error** (0.05 = 5% off on average) | relative error matters (compare small and large items fairly) |
| `medianAbsoluteError` | median of \|y − ŷ\| | outliers must not move the metric |

## Classification — `score`, `classificationReport`, `mcc`, `balancedAccuracy`

| Metric | What it measures | When to prefer it |
| ------ | ---------------- | ----------------- |
| `score` (accuracy) | share of correct labels | only on **balanced** classes; misleading otherwise |
| precision | of the *predicted* positives, how many are right | **false alarms are costly** (a false fraud alert wastes an investigation) |
| recall | of the *actual* positives, how many were found | **missed positives are costly** (fraud, epidemics — you must not miss cases) |
| `fScore` (F1 / Fβ) | harmonic balance of precision/recall — β > 1 favors recall | a single number when you want **balance** (default F1) |
| `classificationReport` | accuracy + precision + recall + Fβ + support + confusion matrix | full picture of a classifier in one call |
| `mcc` | Matthews correlation −1..+1 — robust to imbalance | **THE number for imbalanced data** (fraud, rare diseases) |
| `balancedAccuracy` | macro recall (mean recall per class) | “honest” accuracy when classes are skewed |

## Probability — `rocAucScore`, `prAucScore`, `logLoss`, `rocCurve`, `optimalThreshold`

| Metric | What it measures | When to prefer it |
| ------ | ---------------- | ----------------- |
| `rocAucScore` | ranking quality — 1 = perfect separation, 0.5 = random (binary) | global ranking quality of a `predict_proba` model |
| `prAucScore` | precision-recall AUC | **rare events** (epidemic, fraud): drops when false alarms grow, unlike ROC |
| `logLoss` | cross-entropy — punishes confident **and wrong** probabilities | probability calibration matters (risk scores, betting) |
| `rocCurve` | `{ fpr, tpr, thresholds }` — the full curve | plotting, or choosing where the curve bends |
| `optimalThreshold` | Youden's J — the threshold that maximizes tpr − fpr | turning probabilities into **alerts/decisions** |

## Workflow helpers

| Function | What it does | When to use it |
| -------- | ------------ | -------------- |
| `trainTestSplit` | splits rows into train/test (seeded, stratifiable) | measure generalization on **unseen** rows |
| `crossValScore` | k-fold scores — one score per fold | robust model comparison (mean **and** fold spread) |
| `compareModels` | trains & ranks several models with CV | **choose the best model family** for your case |
| `detectTask` | `"classification"` \| `"regression"` from a spec | auto-pick a family and a default scoring |
| `predictStream` / `fillPredictStream` | chunked inference (generators OK) | run predictions over **millions of rows** without loading them all |

**Which metric first?** For regression start with **R²** (overall quality),
then **MAE** (typical error in target units); add **MAPE** if you compare
items of very different sizes. For classification on imbalanced data (fraud,
epidemics) never trust accuracy alone — look at **recall**, **prAucScore** and
**mcc**.
