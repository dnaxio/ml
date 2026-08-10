---
title: Polynomial Regression
description: Fits nonlinear curves by expanding features to powers 1..degree.
---

Expands each feature to powers `1..degree`, then fits ordinary least squares
on the expanded matrix. Use when the target follows a smooth curve.

**When to use it** — the data **bends**: quadratic cost curves, price
elasticities, diminishing returns. `degree` 2+ captures curvature a straight
line misses.

```ts
import { linear } from "@dnax/ml";

const reg = new linear.PolynomialRegression({ degree: 2 }); // default: 2
reg.fit(data, { features: ["quantite"], target: "cout" });
reg.predict(data);
```

The `degree` config is preserved through `export` / `load`. Beware of very
high degrees: they overfit.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`, `predictAsync`,
`degree`, `coef`, `intercept`, `getParams()` / `setParams()`, `export` / `load`.
