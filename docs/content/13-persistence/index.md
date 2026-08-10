---
title: Overview
description: Export and load models, inject parameters, and offload predictions.
---

> **Scoped vs flat import** — scoped: `new linear.LinearRegression()` ·
> flat: `import { LinearRegression } from "@dnax/ml"`. Both are identical.

Every model in the SDK supports `export` / `load` — including the monitoring
models (CUSUM, EWMA, ParallelMonitor, SeasonalMonitor), the spatial scan, and
the clusters.

| Page | What it covers |
| ---- | -------------- |
| [Persistence](/13-persistence/01-persistence) | `export` / `load` / restore from params |
| [Model parameters](/13-persistence/02-parameters) | `coef`, `intercept`, `getParams`, `setParams` |
| [predictAsync](/13-persistence/03-predict-async) | worker offload |
| [Utilities](/13-persistence/04-utilities) | `columnNames`, `droppedRows` |
