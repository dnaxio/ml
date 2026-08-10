---
title: Overview
description: Score your models — R², MSE, MAE, accuracy, classification reports, ROC/AUC, train/test splits and cross-validation.
---

> **Scoped vs flat import** — scoped: `evaluation.compareModels({...}, data, spec)` ·
> flat: `import { compareModels } from "@dnax/ml"`. Both are identical.

Every supervised model scores itself on rows that include the `target` field
(ground truth), reusing the transformation learned at fit time (rows dropped
by the `'drop'` strategy are excluded).

Implemented on: 14 regressors (`score`/`mse`/`mae`) and 9 classifiers
(`score`/`classificationReport(data, beta?)`, + `rocAucScore` on the 5 with
`predict_proba`). Clustering, IsolationForest, monitoring and scan have no
target → no `score`.

The ground truth convention matches kml: **predictions first, truth second**.

| Page | What it covers |
| ---- | -------------- |
| [Model scoring](/11-evaluation/01-model-scoring) | `score`, `mse`, `mae`, `classificationReport`, `rocAucScore` + reading the scores |
| [Splits & cross-validation](/11-evaluation/02-splits-and-cv) | `trainTestSplit`, `crossValScore` |
| [Leaderboard](/11-evaluation/03-leaderboard) | `compareModels` + `detectTask` |
| [Streaming](/11-evaluation/04-streaming) | `predictStream`, `fillPredictStream` |
| [Metric library](/11-evaluation/05-metric-library) | `rmse`, `mape`, `mcc`, `prAucScore`, `optimalThreshold`… |
