---
title: Linear Regression
description: Ordinary least squares — the baseline every other model is compared to.
---

Ordinary least squares: fits a straight line by minimizing the squared error.
The baseline every other model is compared to.

**When to use it** — the relationship looks like a straight line and you need a
**continuous number** out: salary, price, sales volume, delivery time.
`coef` / `intercept` explain the impact of each feature.

```ts
import { linear } from "@dnax/ml";

const reg = new linear.LinearRegression();
reg.fit(students, { features: ["note", "admis"], target: "note" });
const preds = reg.predict([{ name: "x", note: 6, admis: true }]);
console.log(preds[0]); // 6
```

Methods: `predict`, `fill_predict`, `score` (R²), `mse`, `mae`, `predictAsync`,
`coef`, `intercept`, `getParams()` / `setParams()`, `export` / `load`.
