---
title: Poisson Regressor
description: Generalized linear model for count targets — predictions are always ≥ 0.
---

Generalized linear model for **counts** (non-negative integers). Uses a log
link, so predictions are always `>= 0`.

**When to use it** — the target is a **count**: units sold, incidents,
patients per day, calls per hour. `LinearRegression` could predict negative
numbers here; Poisson cannot.

```ts
import { linear } from "@dnax/ml";

const reg = new linear.PoissonRegressor({ alpha: 0.1 });
reg.fit(data, { features: ["pub", "prix"], target: "ventes" });
reg.predict(data); // always >= 0
```

The `alpha` / `fitIntercept` / `maxIter` / `tol` config is preserved through
`export` / `load`.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`, `predictAsync`,
`nIter`, `coef`, `intercept`, `getParams()` / `setParams()`, `export` / `load`.
