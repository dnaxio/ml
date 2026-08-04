---
title: Persistence & Model Parameters
description: Export and load models, inject parameters, and offload predictions.
navigation:
  icon: lucide:database
---

## Persistence

Export (async, writes `<name>.json` with model params + learned
transformation) and load (async, restores a previously exported model):

```ts
// Export
await reg.export("my-model"); // → my-model.json

// Load
const restored = new LinearRegression();
await restored.load("my-model");
restored.predict(newData); // works with the same columns

// Restore directly from parameters
const fromParams = new LinearRegression({
  coef: reg.coef,
  intercept: reg.intercept,
});
```

The export file is **versioned** (`version` field). `load` rejects files with
an unsupported version to protect against format changes.

Every model in the SDK supports `export` / `load` — including the monitoring
models (CUSUM, EWMA, ParallelMonitor, SeasonalMonitor), the spatial scan, and
the clusters.

## Model parameters

```ts
reg.coef; // number[] — coefficients (one per feature)
reg.intercept; // number   — bias term

reg.getParams(); // { coef: number[], intercept: number }
reg.setParams({ coef, intercept }); // injects parameters, throws on unknown keys, returns this
```

`setParams` / `getParams` follow the sklearn vocabulary and let you restore a
model without re-training. Unknown keys throw.

## `predictAsync(data)`

Like `predict`, but offloads the matrix-vector multiplication to a worker
(via kml `asyncMode`), keeping the event loop responsive on large prediction
sets. The JSON → matrix transformation still runs on the main thread.

```ts
const preds = await reg.predictAsync(bigData);
```

## Utilities

```ts
reg.columnNames; // string[] — X column names (useful to interpret coefficients)
reg.droppedRows; // number — rows removed by the 'drop' missing strategy
```
