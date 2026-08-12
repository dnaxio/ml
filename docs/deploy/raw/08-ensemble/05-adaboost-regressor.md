# AdaBoostRegressor

> Sequential boosting for regression — each round corrects the previous residuals.

Sequential boosting for a **continuous target**: each round focuses on the
residuals of the previous ensemble.

**When to use it** — boosting for regression: good for demand, price and load
forecasts when the features are clean.

```ts
import { ensemble } from "@dnax/ml";

const reg = new ensemble.AdaBoostRegressor({
  nEstimators: 100,
  learningRate: 1.0,
  randomState: 42,
});
reg.fit(data, { features: ["x"], target: "prix" });
```

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`featureImportances`, `export` / `load`.
