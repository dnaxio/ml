# Introduction

> Install @dnax/ml and train your first model on JSON rows.

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

Both import styles work everywhere — scoped `new linear.LinearRegression()`
and flat `new LinearRegression()` are identical (see
[Core concepts](/01-getting-started/03-core-concepts)).

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

- [Architecture](/01-getting-started/02-architecture) — how the facade is structured
- [Core concepts](/01-getting-started/03-core-concepts) — the JSON-first spec, value encoding, missing values, scoped imports
- [Linear models](/04-linear) — regression and classification
- [Neighbors](/05-neighbors) — instance-based kNN classification and regression
- [Evaluation](/11-evaluation) — score, metrics, splits, cross-validation
- [Feature selection](/12-feature-selection) — rank and select the informative columns
- [Monitoring](/09-monitoring) — time-series drift detection
- [Spatial scan](/10-scan) — geographic cluster detection
- [Examples](/15-examples) — ready-made 5-row datasets and expected outputs for every model
- [API reference](/14-api-reference) — every model and method
