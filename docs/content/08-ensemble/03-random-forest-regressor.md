---
title: RandomForestRegressor
description: Many trees combined by averaging — robust regression default.
---

Many decision trees combined by **averaging**. More robust and accurate than
a single tree.

**When to use it** — the **robust default** for regression: solid results
without much tuning, harder to overfit than a single tree. Good when features
interact.

```ts
import { ensemble } from "@dnax/ml";

const reg = new ensemble.RandomForestRegressor({ nEstimators: 100, randomState: 42 });
reg.fit(data, { features: ["age", "annees_exp"], target: "salaire" });
```

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`featureImportances`, `export` / `load`.
