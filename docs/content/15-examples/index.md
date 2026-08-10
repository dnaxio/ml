---
title: Overview
description: Ready-made JSON datasets (≤ 5 rows) for every model and API — concrete cases with expected outputs. Scoped imports everywhere.
---

> **Scoped vs flat import** — the examples use **scoped** imports
> (`new linear.LinearRegression()`, `evaluation.mcc(...)`). Flat imports
> (`new LinearRegression()`) are identical. Note that `options` (scale,
> oneHot, missing…) always live in the **fit spec**, not the constructor.

Each example is a small, concrete case: a 5-row JSON dataset, the scoped call,
and the **expected output**. Values marked `~` are approximate (real data
never fits perfectly). Every section also says **when to use the model** in
the real world.

| Page | Examples |
| ---- | -------- |
| [Linear](/15-examples/01-linear) | 9 regressors & classifiers — salary, churn, feature selection, counts, curves |
| [Neighbors](/15-examples/02-neighbors) | kNN classifier & regressor |
| [Tree](/15-examples/03-tree) | DecisionTree & ExtraTree classifiers/regressors |
| [Ensemble](/15-examples/04-ensemble) | forests, boosting, IsolationForest |
| [Clustering](/15-examples/05-clustering) | KMeans, DBSCAN, HDBSCAN |
| [Monitoring](/15-examples/06-monitoring) | CUSUM, EWMA, ParallelMonitor, SeasonalMonitor |
| [Spatial Scan](/15-examples/07-scan) | Kulldorff & Getis-Ord Gi* |
| [Feature Selection](/15-examples/08-feature-selection) | mutual information, SelectFromModel |
| [Evaluation](/15-examples/09-evaluation) | compareModels, detectTask, splits, pure metrics |
| [Metric Cheat Sheet](/15-examples/10-metrics) | what score, mse, mae, mcc… measure and when |
