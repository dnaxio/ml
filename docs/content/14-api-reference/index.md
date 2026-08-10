---
title: Overview
description: Every model class, constructor parameter, method and getter — the complete reference for all algorithms.
---

> **Scoped vs flat import** — scoped: `new linear.RidgeRegression()` ·
> flat: `import { RidgeRegression } from "@dnax/ml"`. Both are identical.

Every model follows the same contract: `fit(data, spec)` → `predict(data)` →
`fill_predict(data)` → `export(name)` / `load(name)`, plus `score(data)` on
supervised models. Getters `columnNames` and `droppedRows` are common to all.

| Page | Families |
| ---- | -------- |
| [Linear](/14-api-reference/01-linear) | 9 regression & classification models |
| [Tree](/14-api-reference/02-tree) | DecisionTree & ExtraTree |
| [Ensemble](/14-api-reference/03-ensemble) | forests, boosting, IsolationForest |
| [Clustering](/14-api-reference/04-clustering) | KMeans, DBSCAN, HDBSCAN |
| [Neighbors](/14-api-reference/05-neighbors) | kNN classifier & regressor |
| [Monitoring](/14-api-reference/06-monitoring) | CUSUM, EWMA, ParallelMonitor, SeasonalMonitor |
| [Spatial scan](/14-api-reference/07-scan) | SpatialScan, GetisOrd |
| [Feature selection](/14-api-reference/08-feature-selection) | univariate + SelectFromModel |
| [Evaluation](/14-api-reference/09-evaluation) | splits, CV, leaderboard, metrics |
| [Spec types](/14-api-reference/10-spec-types) | JsonFitSpec, ClusterSpec, MonitorSpec, ScanSpec… |
