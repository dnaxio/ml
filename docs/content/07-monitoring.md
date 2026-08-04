---
title: Monitoring
description: Detect when a series drifts from its normal — CUSUM, EWMA, weekly deseasonalization and multi-series monitoring.
navigation:
  icon: lucide:activity
---

Statistical process control methods that detect the **onset of a sustained
shift** in a series — e.g. the beginning of a sales peak. They complement the
ML models: IsolationForest flags anomalies *per transaction*, CUSUM/EWMA flag
anomalies *over time*.

Both follow the SDK convention (`fit` → `predict` → `fill_predict`) on JSON
rows, with a `MonitorSpec` selecting the numeric field to monitor:

```ts
import { CUSUM, EWMA } from "@dnax/ml";

const days = [
  { day: 1, sales: 118 },
  { day: 2, sales: 121 },
  // ...
  { day: 18, sales: 200 }, // peak
];

// 1) Learn the reference (μ0, σ) on the normal period
const cusum = new CUSUM();
cusum.fit(days.slice(0, 12), { field: "sales" });

// 2) Monitor the whole series → alert as soon as the rise starts
const alerts = cusum.predict(days); // number[]: 1 = alert, 0 = normal
const scores = cusum.scores(days);  // cumulative S_n per point

// 3) JSON rows with the alert field filled (input is not mutated)
const tracked = cusum.fill_predict(days); // { day, sales, alert: true | false }

// Exponential smoothing, same usage:
const ewma = new EWMA({ lambda: 0.25, limit: 3 });
ewma.fit(days.slice(0, 12), { field: "sales" });
const ewmaAlerts = ewma.predict(days);
```

Both models persist with `export` / `load` (statistics + spec saved).

## `CUSUM` — Cumulative SUM

Accumulates deviations from a reference mean: `S_n = max(0, S_{n-1} + (x_n − μ0 − k))`
and alerts when `S_n > h`. Very sensitive to **small sustained shifts**.

| Param        | Default      | Role                                                           |
| ------------ | ------------ | -------------------------------------------------------------- |
| `target`     | estimated    | reference mean μ0 (deviations are measured against it)         |
| `std`        | estimated    | series standard deviation σ                                    |
| `k`          | `0.5·σ`      | allowable slack — sub-k shifts are ignored as background noise |
| `h`          | `5·σ`        | alert threshold — lower = earlier but noisier alarms           |
| `direction`  | `'increase'` | shift to detect: `'increase'` | `'decrease'` | `'both'`     |
| `alertField` | `'alert'`    | field name filled by `fill_predict`                            |

## `EWMA` — Exponentially Weighted Moving Average

Smooths the series with a decaying memory (`z_n = λ·x_n + (1−λ)·z_{n−1}`) and
alerts when the smoothed value leaves `μ0 ± L·σ_z(n)`. Recovers quickly after
isolated spikes.

| Param        | Default   | Role                                                             |
| ------------ | --------- | ---------------------------------------------------------------- |
| `lambda`     | `0.25`    | smoothing factor λ ∈ (0,1] — small = smooth, sensitive to trends |
| `limit`      | `3`       | control-limit width L (in σ units)                               |
| `target`     | estimated | reference mean μ0                                                |
| `std`        | estimated | series standard deviation σ                                      |
| `alertField` | `'alert'` | field name filled by `fill_predict`                              |

### Online monitoring (real-time)

CUSUM and EWMA are inherently sequential (`S_n` from `S_{n-1}`, `z_n` from
`z_{n-1}`): `update(row)` consumes **one row** and advances the statistic
point by point — no history is kept, no re-processing. Use it for live
dashboards (one point per second):

```ts
const cusum = new CUSUM();
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

## `ParallelMonitor` — several series at once

Monitors several series simultaneously with one chart per field. An alert
fires when **at least one** chart fires, and `alertFields` reports **which**
fields triggered (interpretable alarms).

```ts
import { ParallelMonitor } from "@dnax/ml";

const pm = new ParallelMonitor(); // one CUSUM per field (default)
pm.fit(days, { fields: ["paracetamol", "ibuprofen"] });

const alerts = pm.predict(days); // 1 when at least one chart fires
const fields = pm.alertFields(days); // ["paracetamol"] per row — which fields
const tracked = pm.fill_predict(days); // + alert: boolean + alertFields: string[]

// EWMA charts, with a controlled family-wise false-alarm rate (Bonferroni):
const pmE = new ParallelMonitor({ model: "ewma", familyError: 0.05 });
pmE.fit(days, { fields: ["paracetamol", "ibuprofen"] });
```

**Why the Bonferroni correction matters**: with N independent charts, the
chance of at least one false alarm grows with N (5 charts at 5% each → ~23%
family-wise). `familyError` divides the risk across charts (each chart gets
the two-sided level `familyError / N`). When omitted, each chart keeps its
own conservative defaults (`h = 5·σ`, `L = 3`).

| Param              | Default        | Role                                                    |
| ------------------ | -------------- | ------------------------------------------------------- |
| `model`            | `'cusum'`      | chart type per field: `'cusum'` or `'ewma'`             |
| `familyError`      | —              | family-wise false-alarm rate (Bonferroni across fields) |
| `alertField`       | `'alert'`      | boolean field filled by `fill_predict`                  |
| `alertFieldsField` | `'alertFields'`| field listing the triggering fields                     |

Per-field overrides go in the spec: `{ fields, params: { paracetamol: { k, h } } }`
(user values always win over the Bonferroni adjustment).

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
const cp = new CUSUM({ robust: true, direction: "both" });
cp.fit(baseline, { field: "sales", missing: "fill" });
const onset = cp.changePoint(sales); // estimated drift onset per day

const ew = new EWMA({ lambda: 0.25, limit: 3 });
ew.fit(baseline, { field: "sales" });
const { ucl, lcl } = ew.limits(sales); // control band for plot/heatmap
```

## `SeasonalMonitor` — weekly deseasonalization

Pharmacy-type sales follow a **weekly cycle** (weekend dips, Monday peaks). On
a constant-mean chart this cycle inflates σ and **delays** the detection of a
real shift. `SeasonalMonitor` fits a reference mean **per day of week**, then
runs a CUSUM (or EWMA) on the deseasonalized residuals `x_n − μ0_{dayOfWeek(n)}`.

```ts
import { SeasonalMonitor } from "@dnax/ml";

const sm = new SeasonalMonitor();
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

> **Tip**: estimate `k`/`h` (or `target`/`std`) on a known-normal baseline
> period, then monitor the full series — otherwise the peak itself inflates
> the reference statistics.
