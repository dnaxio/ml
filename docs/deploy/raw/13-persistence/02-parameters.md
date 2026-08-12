# Model Parameters

> coef, intercept, getParams and setParams — inspect and inject learned parameters.

```ts
reg.coef; // number[] — coefficients (one per feature)
reg.intercept; // number   — bias term

reg.getParams(); // { coef: number[], intercept: number }
reg.setParams({ coef, intercept }); // injects parameters, throws on unknown keys, returns this
```

`setParams` / `getParams` follow the sklearn vocabulary and let you restore a
model without re-training. Unknown keys throw.
