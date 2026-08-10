---
title: ExtraTreeRegressor
description: Extremely randomized tree regressor.
---

Extremely randomized tree for a **continuous target**: split thresholds are
chosen randomly instead of optimally. Faster and often more robust against
overfitting than a plain tree.

**When to use it** — a solid cheap base model for forecasts and sensor values
(temperature, demand). `randomState` keeps it reproducible.

```ts
import { tree } from "@dnax/ml";

const reg = new tree.ExtraTreeRegressor({ max_depth: 5, randomState: 42 });
reg.fit(data, { features: ["note", "heures"], target: "resultat" });
reg.predict(data);
```

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`,
`featureImportances`, `export` / `load`.
