# Elastic Net

> Combined L1 + L2 regularization — stable on correlated features with feature selection.

Lasso + Ridge penalties: stable on correlated features **and** feature
selection. `l1Ratio` balances the two (`0` = pure Ridge, `1` = pure Lasso,
default `0.5`).

**When to use it** — both worlds: **correlated features AND useless columns**.
The L1 + L2 compromise is the robust default when you don't know which of the
two problems you have.

```ts
import { linear } from "@dnax/ml";

const reg = new linear.ElasticNet({ alpha: 1, l1Ratio: 0.5 });
reg.fit(data, { features: ["age", "annees_exp", "inutile"], target: "salaire" });
reg.coef; // stable on correlated features, 0 on uninformative ones
```

The `alpha` / `l1Ratio` / `fitIntercept` / `maxIter` / `tol` config is
preserved through `export` / `load`.

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`, `predictAsync`,
`coef`, `intercept`, `getParams()` / `setParams()`, `export` / `load`.
