---
title: Neighbors
description: 2 examples — kNN classification and regression from the nearest labeled examples.
---

## `KNeighborsClassifier` — fraudulent order? (by neighborhood)

**When to use it** — **no training phase**: predictions come from the nearest
labeled examples. Use as a non-parametric baseline, or when the decision
boundary is very local (a fraud looks like the frauds near it). Distance-based
→ `scale: true` is required.

```ts
const orders = [
  { qty: 30, amount: 90, fraud: false },
  { qty: 60, amount: 180, fraud: false },
  { qty: 420, amount: 3200, fraud: true },
  { qty: 480, amount: 3900, fraud: true },
  { qty: 90, amount: 250, fraud: false },
];
const clf = new neighbors.KNeighborsClassifier({ kNeighbors: 3 });
clf.fit(orders, {
  features: ["qty", "amount"], target: "fraud",
  options: { scale: true }, // distance-based → scaling is required
});
clf.predict([{ qty: 450, amount: 3600 }]); // → [1] (close to the frauds)
clf.classes;                               // → [0, 1]
```

## `KNeighborsRegressor` — house price from surface area

**When to use it** — same principle for a **continuous target**: the prediction
is the average of the k nearest known values. Ideal for very local patterns
(price per m² in a neighborhood, delivery time per route).

```ts
const houses = [
  { area: 45, price: 120000 },
  { area: 60, price: 155000 },
  { area: 75, price: 190000 },
  { area: 90, price: 230000 },
  { area: 110, price: 280000 },
];
const reg = new neighbors.KNeighborsRegressor({ nNeighbors: 2 });
reg.fit(houses, { features: ["area"], target: "price", options: { scale: true } });
reg.predict([{ area: 70 }]); // → ≈ (155000 + 190000) / 2 = 172500
```
