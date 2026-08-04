---
title: Getting Started
description: Install @dnax/ml and train your first model on JSON rows.
navigation:
  icon: lucide:rocket
---

## Installation

```bash
bun add @dnax/ml
```

`@dnax/ml` runs on Bun, Node.js and in the browser. It has a single
dependency (`@kanaries/ml`, pinned to an exact version), which is itself
dependency-free — the runtime footprint is minimal.

## Quick start

Train a linear regression directly on JSON rows — no matrix conversion:

```ts
import { LinearRegression } from "@dnax/ml";

const students = [
  { name: "koffi", note: 1, admis: true },
  { name: "Jules", note: 2, admis: false },
  { name: "Aya", note: 3, admis: true },
  { name: "Sam", note: 4, admis: false },
  { name: "Léo", note: 5, admis: true },
];

const reg = new LinearRegression();
reg.fit(students, { features: ["note", "admis"], target: "note" });

const preds = reg.predict([{ name: "x", note: 6, admis: true }]);
console.log(preds[0]); // 6
```

What just happened:

1. `fit(students, spec)` — the **spec** says which fields are `features` (X)
   and which is the `target` (Y). The rows are transformed automatically
   (`admis: boolean` → 1/0).
2. `predict(rows)` — same fields, no target needed, returns predictions.
3. The model remembers the transformation: `coef`, `columnNames` and
   `export`/`load` all stay consistent.

## Evaluate the model

```ts
// R² on rows that include the target (ground truth)
const r2 = reg.score(students); // ~1.0 on this perfect fit

// Train/test split + cross-validation for a robust choice
import { trainTestSplit, crossValScore } from "@dnax/ml";

const { train, test } = trainTestSplit(students, {
  testSize: 0.3,
  randomState: 42,
});
reg.fit(train, { features: ["note", "admis"], target: "note" });
console.log(reg.score(test)); // quality on unseen rows
```

## What next?

- [Core concepts](/11-core-concepts) — the JSON-first spec, value encoding, missing values, scoped imports
- [Linear models](/03-linear) — regression and classification
- [Evaluation](/09-evaluation) — score, metrics, splits, cross-validation
- [Monitoring](/07-monitoring) — time-series drift detection
- [Spatial scan](/08-scan) — geographic cluster detection
- [API reference](/12-api-reference) — every model and method
