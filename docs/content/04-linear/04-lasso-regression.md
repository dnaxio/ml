---
title: Lasso Regression
description: L1-regularized regression — sparse coefficients, feature selection.
---

L1 regularization (`alpha`): some coefficients shrink to exactly `0`,
discarding uninformative features — a built-in feature selector.

**When to use it** — you have many features and suspect most are noise: L1
**zeroes the useless ones**, giving you a smaller, interpretable model.

```ts
import { linear } from "@dnax/ml";

const reg = new linear.LassoRegression({ alpha: 1 }); // L1 penalty (default: 1)
reg.fit(data, { features: ["age", "annees_exp", "inutile"], target: "salaire" });
reg.coef; // → [..., 0]  (uninformative features are set to 0)
```

The `alpha` / `fitIntercept` / `maxIter` / `tol` config is preserved through
`export` / `load`.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`, `predictAsync`,
`coef`, `intercept`, `getParams()` / `setParams()`, `export` / `load`.
