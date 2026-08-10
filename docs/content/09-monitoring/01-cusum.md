---
title: CUSUM
description: Cumulative sum — very sensitive to small sustained shifts.
---

Accumulates deviations from a reference mean: `S_n = max(0, S_{n-1} + (x_n − μ0 − k))`
and alerts when `S_n > h`. Very sensitive to **small sustained shifts**.

**When to use it** — detect the **onset of a sustained shift** in a series:
early epidemic warnings, sales spikes, rising defect rates. Fit on a
**known-normal baseline** first.

| Param        | Default      | Role                                                           |
| ------------ | ------------ | -------------------------------------------------------------- |
| `target`     | estimated    | reference mean μ0 (deviations are measured against it)         |
| `std`        | estimated    | series standard deviation σ                                    |
| `k`          | `0.5·σ`      | allowable slack — sub-k shifts are ignored as background noise |
| `h`          | `5·σ`        | alert threshold — lower = earlier but noisier alarms           |
| `direction`  | `'increase'` | shift to detect: `'increase'` | `'decrease'` | `'both'`     |
| `alertField` | `'alert'`    | field name filled by `fill_predict`                            |

Methods: `fit`, `predict`, `scores`, `fill_predict`, `changePoint(data)`,
`update(row)`, `reset()`, `export` / `load` · getters `target`, `std`,
`slack`, `threshold`, `direction`.
