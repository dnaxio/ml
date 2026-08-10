---
title: RANSAC Regressor
description: Robust regression that ignores outliers.
---

Fits many random subsets, keeps the consensus set of inliers, and predicts
from the best fit. Exposes `inlierMask` to identify outliers.

**When to use it** — your data contains **bad measurements** (sensor glitch,
manual entry error, one-off event): it fits the healthy line and ignores the
noise. `inlierMask` flags the bad rows for investigation.

```ts
import { linear } from "@dnax/ml";

const reg = new linear.RANSACRegressor({ randomState: 42 });
reg.fit(data, { features: ["x"], target: "y" }); // y contains outliers
reg.coef;         // fitted on the inliers only
reg.inlierMask;   // [true, ..., false] — which samples are outliers
```

`export` / `load` use the official kml serializer (inlier mask, trial count,
config preserved).

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`, `predictAsync`,
`inlierMask`, `nTrials`, `coef`, `intercept`, `export` / `load`.
