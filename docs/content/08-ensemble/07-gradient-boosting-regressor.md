---
title: GradientBoostingRegressor
description: Sequential trees fitted on residuals — a strong regression default.
---

Sequential trees fitted on the **residuals** of the previous ensemble, with a
small learning rate.

**When to use it** — often the **best default regressor** on tabular data:
nonlinear trends + feature interactions with well-calibrated errors.

```ts
import { ensemble } from "@dnax/ml";

const reg = new ensemble.GradientBoostingRegressor({
  nEstimators: 100,
  learningRate: 0.1,
  maxDepth: 3,
});
reg.fit(data, { features: ["x"], target: "prix" });
```

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`featureImportances`, `export` / `load`.
