---
title: SeasonalMonitor
description: Weekly deseasonalization — removes the weekday cycle before detecting shifts.
---

Pharmacy-type sales follow a **weekly cycle** (weekend dips, Monday peaks). On
a constant-mean chart this cycle inflates σ and **delays** the detection of a
real shift. `SeasonalMonitor` fits a reference mean **per day of week**, then
runs a CUSUM (or EWMA) on the deseasonalized residuals `x_n − μ0_{dayOfWeek(n)}`.

**When to use it** — your series has a **weekly cycle** (pharmacy sales,
traffic): it removes the weekday pattern so a real rise is not confused with a
normal Monday peak. Needs **≥ 1 full week** of baseline.

```ts
import { monitoring } from "@dnax/ml";

const sm = new monitoring.SeasonalMonitor();
sm.fit(baseline, { field: "sales", dateField: "date" }); // at least one full week

const alerts = sm.predict(sales); // 1 = the deseasonalized chart fires
const profile = sm.dayProfile;    // reference mean per day of week (diagnostic)
const tracked = sm.fill_predict(sales); // + alert: boolean
```

- **Needs ≥ 1 full week of baseline** (every day-of-week present) — throws otherwise.
- `dateField` accepts a `Date`, an ISO date (`"2026-08-03"` parsed as local)
  or a timestamp.
- On a real weekly cycle (max/min profile ratio > ~1.3), it detects a
  moderate epidemic **several days earlier** than a plain CUSUM.
- `changePoint(data)` (CUSUM) and `limits(data)` (EWMA) are available on the
  deseasonalized chart.

Methods: `fit`, `predict`, `scores`, `fill_predict`, `update(row)`,
`changePoint` / `limits` (per model), `export` / `load` · getter `dayProfile`.
