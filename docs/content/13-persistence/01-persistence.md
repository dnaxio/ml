---
title: Persistence
description: Export to and load from a versioned .json file.
---

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
