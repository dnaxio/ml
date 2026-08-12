# Ridge Regression

> L2-regularized regression — stable coefficients on correlated features.

L2 regularization (`alpha`): stable coefficients when features are correlated
or when ordinary least squares overfits.

**When to use it** — your features are **correlated** (multicollinearity): the
L2 penalty shrinks the coefficients instead of letting them explode.

```ts
import { linear } from "@dnax/ml";

const reg = new linear.RidgeRegression({ alpha: 1 }); // L2 penalty (default: 1)
reg.fit(data, { features: ["age", "annees_exp"], target: "salaire" });
```

The `alpha` / `fitIntercept` config is preserved through `export` / `load`.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`, `predictAsync`,
`coef`, `intercept`, `getParams()` / `setParams()`, `export` / `load`.
