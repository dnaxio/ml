---
title: DecisionTreeRegressor
description: Stepwise regression — each leaf predicts the mean of its bucket.
---

**When to use it** — the relationship is **stepwise**: salary bands, tax
brackets, quantity discounts. It predicts the mean of the bucket, not a smooth
line — perfect when real life jumps in levels.

```ts
import { tree } from "@dnax/ml";

const reg = new tree.DecisionTreeRegressor({ max_depth: 3 });
reg.fit(data, { features: ["age", "annees_exp"], target: "salaire" });
reg.predict(data);
```

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`featureImportances`, `export` / `load`.
