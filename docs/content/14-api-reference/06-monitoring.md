---
title: Monitoring
description: CUSUM, EWMA, ParallelMonitor and SeasonalMonitor — params and extras.
---

`MonitorSpec` = `{ field, missing? }`. Common surface: `fit`, `predict`,
`scores`, `fill_predict`, `export`, `load`.

| Class | Params | Extra |
| ----- | ------ | ----- |
| `CUSUM` | `target`, `std`, `k` (0.5σ), `h` (5σ), `direction` (`'increase'`\|`'decrease'`\|`'both'`), `robust`, `alertField` | `changePoint(data)`, `update(row)`, `reset()` · getters `target`, `std`, `slack`, `threshold`, `direction` |
| `EWMA` | `lambda` (0.25), `limit` (3), `target`, `std`, `robust`, `alertField` | `limits(data)`, `update(row)`, `reset()` · getters `target`, `std`, `lambda`, `limit` |
| `ParallelMonitor` | `model` (`'cusum'`\|`'ewma'`), `familyError`, `alertField`, `alertFieldsField` | `ParallelSpec` = `{ fields, params?, missing? }` · `alertFields(data)`, `update(row)` · getters `monitoredFields`, `targets`, `params` |
| `SeasonalMonitor` | `model`, `alertField`, `chart` | `SeasonalSpec` = `{ field, dateField, missing? }` · `changePoint`, `limits` (per model), `update(row)` · getter `dayProfile` |
