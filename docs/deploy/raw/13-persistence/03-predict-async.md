# predictAsync

> Offload the matrix-vector product to a worker, keeping the event loop responsive.

Like `predict`, but offloads the matrix-vector multiplication to a worker
(via kml `asyncMode`), keeping the event loop responsive on large prediction
sets. The JSON → matrix transformation still runs on the main thread.

```ts
const preds = await reg.predictAsync(bigData);
```

Available on the linear regressors (all but `LogisticRegression`).
