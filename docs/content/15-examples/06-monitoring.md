---
title: Monitoring
description: 4 examples — CUSUM, EWMA, multi-series and weekly deseasonalization.
---

## `CUSUM` — onset of a sales spike

**When to use it** — detect the **onset of a sustained shift** in a series:
early epidemic warnings, sales spikes, rising defect rates. Fit on a
**known-normal baseline** first (see `/09-monitoring`).

```ts
const days = [
  { day: 1, sales: 100 },
  { day: 2, sales: 102 },
  { day: 3, sales: 101 },
  { day: 4, sales: 105 },
  { day: 5, sales: 140 }, // ← spike
];
const cusum = new monitoring.CUSUM();
cusum.fit(days.slice(0, 4), { field: "sales" }); // normal baseline only
cusum.predict(days);      // → [0, 0, 0, 0, 1]
cusum.fill_predict(days); // → + alert: true on day 5
```

## `EWMA` — spike caught by exponential smoothing

**When to use it** — same goal but it **smooths the noise** first: fewer false
alarms on jittery series. `limits()` gives the control band to draw on a
chart.

```ts
const days = [
  { day: 1, sales: 100 },
  { day: 2, sales: 102 },
  { day: 3, sales: 101 },
  { day: 4, sales: 105 },
  { day: 5, sales: 140 },
];
const ewma = new monitoring.EWMA({ lambda: 0.25, limit: 3 });
ewma.fit(days.slice(0, 4), { field: "sales" });
ewma.predict(days); // → [0, 0, 0, 0, 1]
ewma.limits(days);  // → { ucl, lcl } per day (control band)
```

## `ParallelMonitor` — paracetamol drifting, ibuprofen stable

**When to use it** — you watch **several series at once** (many products,
meters, branches): it monitors each field and tells you **which ones**
triggered (`alertFields`).

```ts
const days = [
  { day: 1, paracetamol: 100, ibuprofen: 60 },
  { day: 2, paracetamol: 102, ibuprofen: 62 },
  { day: 3, paracetamol: 101, ibuprofen: 61 },
  { day: 4, paracetamol: 105, ibuprofen: 63 },
  { day: 5, paracetamol: 150, ibuprofen: 62 }, // ← paracetamol spike only
];
const pm = new monitoring.ParallelMonitor();
pm.fit(days.slice(0, 4), { fields: ["paracetamol", "ibuprofen"] });
pm.alertFields(days); // → [[], [], [], [], ["paracetamol"]]
```

## `SeasonalMonitor` — weekly deseasonalization (≥ 1 full week required)

**When to use it** — your series has a **weekly cycle** (pharmacy sales,
traffic): it removes the weekday pattern so a real rise is not confused with a
normal Monday peak. Needs **≥ 1 full week** of baseline.

```ts
const days = [
  { date: "2026-07-27", sales: 130 }, // Monday (weekly peak)
  { date: "2026-07-28", sales: 100 },
  { date: "2026-07-29", sales: 100 },
  { date: "2026-07-30", sales: 100 },
  { date: "2026-08-03", sales: 170 }, // next Monday → anomaly vs profile
];
const sm = new monitoring.SeasonalMonitor();
sm.fit(days, { field: "sales", dateField: "date" });
// ⚠️ Throws: "SeasonalMonitor requires at least one full week of baseline"
// (every weekday must be present at fit time — bring 7+ rows, see /09-monitoring)
// After a proper 7+ day fit: sm.dayProfile → { monday: 130, tuesday: 100, … }
```
