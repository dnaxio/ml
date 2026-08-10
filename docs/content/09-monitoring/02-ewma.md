---
title: EWMA
description: Exponentially weighted moving average — smooths noise, fewer false alarms.
---

Smooths the series with a decaying memory (`z_n = λ·x_n + (1−λ)·z_{n−1}`) and
alerts when the smoothed value leaves `μ0 ± L·σ_z(n)`. Recovers quickly after
isolated spikes.

**When to use it** — same goal as CUSUM but it **smooths the noise** first:
fewer false alarms on jittery series. `limits()` gives the control band to
draw on a chart.

| Param        | Default   | Role                                                             |
| ------------ | --------- | ---------------------------------------------------------------- |
| `lambda`     | `0.25`    | smoothing factor λ ∈ (0,1] — small = smooth, sensitive to trends |
| `limit`      | `3`       | control-limit width L (in σ units)                               |
| `target`     | estimated | reference mean μ0                                                |
| `std`        | estimated | series standard deviation σ                                      |
| `alertField` | `'alert'` | field name filled by `fill_predict`                              |

Methods: `fit`, `predict`, `scores`, `fill_predict`, `limits(data)`,
`update(row)`, `reset()`, `export` / `load` · getters `target`, `std`,
`lambda`, `limit`.
