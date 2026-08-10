---
title: Overview
description: Detect when a series drifts from its normal — CUSUM, EWMA, weekly deseasonalization and multi-series monitoring.
---

> **Scoped vs flat import** — scoped: `new monitoring.CUSUM()` ·
> flat: `import { CUSUM } from "@dnax/ml"`. Both are identical.

Statistical process control methods that detect the **onset of a sustained
shift** in a series — e.g. the beginning of a sales peak. They complement the
ML models: IsolationForest flags anomalies *per transaction*, CUSUM/EWMA flag
anomalies *over time*.

Both follow the SDK convention (`fit` → `predict` → `fill_predict`) on JSON
rows, with a `MonitorSpec` selecting the numeric field to monitor:

```ts
import { monitoring } from "@dnax/ml";

const days = [
  { day: 1, sales: 118 },
  { day: 2, sales: 121 },
  // ...
  { day: 18, sales: 200 }, // peak
];

// 1) Learn the reference (μ0, σ) on the normal period
const cusum = new monitoring.CUSUM();
cusum.fit(days.slice(0, 12), { field: "sales" });

// 2) Monitor the whole series → alert as soon as the rise starts
const alerts = cusum.predict(days); // number[]: 1 = alert, 0 = normal
const scores = cusum.scores(days);  // cumulative S_n per point

// 3) JSON rows with the alert field filled (input is not mutated)
const tracked = cusum.fill_predict(days); // { day, sales, alert: true | false }
```

Both models persist with `export` / `load` (statistics + spec saved).

| Model | Purpose |
| ----- | ------- |
| [CUSUM](/09-monitoring/01-cusum) | cumulative sum — sensitive to small sustained shifts |
| [EWMA](/09-monitoring/02-ewma) | exponential smoothing — fewer false alarms on noise |
| [ParallelMonitor](/09-monitoring/03-parallel-monitor) | several series at once, per-field alerts |
| [SeasonalMonitor](/09-monitoring/04-seasonal-monitor) | removes the weekly cycle before detecting |

## Online monitoring (real-time)

CUSUM and EWMA are inherently sequential (`S_n` from `S_{n-1}`, `z_n` from
`z_{n-1}`): `update(row)` consumes **one row** and advances the statistic
point by point — no history is kept, no re-processing. Use it for live
dashboards (one point per second):

```ts
const cusum = new monitoring.CUSUM();
cusum.fit(baseline, { field: "sales" });

cusum.update({ sales: 120 }); // → { alert: false, score: 0.5 }
cusum.update({ sales: 135 }); // → { alert: true,  score: 12.3 }

cusum.reset(); // restart monitoring from zero
```

- Starting from a fresh fit, calling `update` over a series yields the same
  scores/alerts as `scores(data)` / `predict(data)` (verified by tests).
- `ParallelMonitor.update(row)` returns `{ alert, alertFields, scores }` —
  which fields triggered, per field score.
- `SeasonalMonitor.update(row)` computes the day-of-week residual and
  advances the internal chart.
- `reset()` restarts the online state.

## Advanced options (CUSUM / EWMA)

- **`changePoint(data)`** (CUSUM) — estimates where the drift *began* (the
  point right after the statistic last reset to 0), not just where it was
  detected. Returns one index per row (`-1` when no drift is accumulating).
- **`limits(data)`** (EWMA) — returns the time-varying control band
  `{ ucl, lcl }` per point, useful for plotting or heatmaps.
- **`robust: true`** — estimates μ0/σ with the **median** and the **MAD**
  (1.4826·MAD) instead of the mean/std, so a few anomalous baseline days
  barely move the thresholds.
- **`direction: "both"`** (CUSUM) — runs the increase *and* decrease
  statistics and alerts on the max.
- **`missing: "fill"`** — carries the last known value forward (LOCF) when
  the field is absent / non-numeric, keeping the output aligned with the
  input rows (e.g. closed days). Default `'throw'`.

```ts
const cp = new monitoring.CUSUM({ robust: true, direction: "both" });
cp.fit(baseline, { field: "sales", missing: "fill" });
const onset = cp.changePoint(sales); // estimated drift onset per day

const ew = new monitoring.EWMA({ lambda: 0.25, limit: 3 });
ew.fit(baseline, { field: "sales" });
const { ucl, lcl } = ew.limits(sales); // control band for plot/heatmap
```

> **Tip**: estimate `k`/`h` (or `target`/`std`) on a known-normal baseline
> period, then monitor the full series — otherwise the peak itself inflates
> the reference statistics.
