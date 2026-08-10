---
title: Linear
description: 9 examples — regression, classification, feature selection, counts and curves.
---

## `LinearRegression` — predict salary from seniority

**When to use it** — the relationship looks like a straight line and you need a
**continuous number** out: salary, price, sales volume, delivery time. Use the
`coef`/`intercept` getters to explain the impact of each feature.

```ts
const employees = [
  { years: 1, salary: 30000 },
  { years: 2, salary: 36000 },
  { years: 3, salary: 42000 },
  { years: 4, salary: 48000 },
  { years: 5, salary: 54000 },
];
const reg = new linear.LinearRegression();
reg.fit(employees, { features: ["years"], target: "salary" });
reg.predict([{ years: 6 }]);  // → [~60000]
reg.coef;                     // → [~6000]  (1 year = +6000 €)
reg.intercept;                // → ~24000
```

## `LogisticRegression` — will the client buy? (boolean target)

**When to use it** — the answer is a **yes/no** (or a class) and you want the
**probability**, not just the label: churn, fraud, will-buy, disease risk. Use
`predict_proba` to rank rows or trigger alerts.

```ts
const clients = [
  { age: 20, balance: 100, bought: false },
  { age: 35, balance: 5000, bought: true },
  { age: 45, balance: 9000, bought: true },
  { age: 25, balance: 300, bought: false },
  { age: 30, balance: 1200, bought: false },
];
const clf = new linear.LogisticRegression();
clf.fit(clients, { features: ["age", "balance"], target: "bought" });
clf.predict([{ age: 38, balance: 7000 }]);         // → [1]
clf.predict_proba([{ age: 38, balance: 7000 }]);   // → [[~0, ~1]] (confident on this tiny set)
clf.fill_predict(clients); // → rows + bought: true | false
```

## `RidgeRegression` — revenue with correlated features (ads ≈ stores×2)

**When to use it** — like `LinearRegression` but your features are
**correlated** (multicollinearity): it shrinks the coefficients (L2) so they
stay stable and interpretable instead of exploding.

```ts
const sales = [
  { ads: 1, stores: 2, revenue: 100 },
  { ads: 2, stores: 4, revenue: 210 },
  { ads: 3, stores: 6, revenue: 305 },
  { ads: 4, stores: 8, revenue: 415 },
  { ads: 5, stores: 10, revenue: 500 },
];
const reg = new linear.RidgeRegression({ alpha: 1 });
reg.fit(sales, { features: ["ads", "stores"], target: "revenue" });
reg.coef; // → STABLE coefficients despite ads ≈ stores×2 correlation
```

## `LassoRegression` — feature selection (shoe_size dropped)

**When to use it** — you have many features and suspect most are noise. The L1
penalty **zeroes the useless ones**, giving you a smaller, interpretable model
— a built-in feature selector.

```ts
const data = [
  { age: 30, experience: 5, shoe_size: 42, salary: 38000 },
  { age: 35, experience: 8, shoe_size: 44, salary: 48000 },
  { age: 40, experience: 12, shoe_size: 43, salary: 57000 },
  { age: 45, experience: 16, shoe_size: 41, salary: 66000 },
  { age: 50, experience: 20, shoe_size: 45, salary: 75000 },
];
const reg = new linear.LassoRegression({ alpha: 1 });
reg.fit(data, { features: ["age", "experience", "shoe_size"], target: "salary" });
reg.coef; // → shoe_size ≈ 0 (uninformative → zeroed out)
```

## `ElasticNet` — L1 + L2 combined

**When to use it** — both worlds: **correlated features AND useless columns**.
L1 + L2 is the robust compromise when you don't know which of the two problems
you have.

```ts
const data = [
  { age: 30, experience: 5, noise: 1, salary: 38000 },
  { age: 35, experience: 8, noise: 0, salary: 48000 },
  { age: 40, experience: 12, noise: 2, salary: 57000 },
  { age: 45, experience: 16, noise: 1, salary: 66000 },
  { age: 50, experience: 20, noise: 3, salary: 75000 },
];
const reg = new linear.ElasticNet({ alpha: 1, l1Ratio: 0.5 });
reg.fit(data, { features: ["age", "experience", "noise"], target: "salary" });
reg.coef; // → noise ≈ 0, age/experience stay stable
```

## `RidgeClassifier` — admitted or not (labels, no probabilities)

**When to use it** — fast **label-only** classifier. Use as a quick baseline or
when you only need a yes/no decision (not a probability) on large data.

```ts
const candidates = [
  { grade: 8, hours: 5, admitted: false },
  { grade: 12, hours: 8, admitted: false },
  { grade: 14, hours: 10, admitted: true },
  { grade: 16, hours: 12, admitted: true },
  { grade: 18, hours: 15, admitted: true },
];
const clf = new linear.RidgeClassifier({ alpha: 1 });
clf.fit(candidates, { features: ["grade", "hours"], target: "admitted" });
clf.predict([{ grade: 15, hours: 11 }]); // → [1]
clf.classes;                             // → [0, 1]
```

## `RANSACRegressor` — robust to outliers (bad measurement ignored)

**When to use it** — your data contains **bad measurements** (sensor glitch,
manual entry error, one-off event): it fits the healthy line and ignores the
noise. `inlierMask` flags the bad rows for investigation.

```ts
const measurements = [
  { x: 1, y: 3.1 },
  { x: 2, y: 30 },   // ← outlier (bad measurement)
  { x: 3, y: 9.2 },
  { x: 4, y: 12.1 },
  { x: 5, y: 15.0 },
];
const reg = new linear.RANSACRegressor({ randomState: 42 });
reg.fit(measurements, { features: ["x"], target: "y" });
reg.inlierMask; // → 4 true + 1 false — the bad measurement is rejected
reg.coef;       // → ≈ 3 (healthy slope, outlier ignored)
```

## `PoissonRegressor` — counting sales (predictions ≥ 0)

**When to use it** — the target is a **count**: units sold, incidents,
patients per day, calls per hour. Predictions stay ≥ 0 and the model handles
count data correctly, where `LinearRegression` could predict negative numbers.

```ts
const sales = [
  { ads: 0, price: 5, units: 8 },
  { ads: 1, price: 5, units: 14 },
  { ads: 2, price: 4, units: 23 },
  { ads: 3, price: 4, units: 31 },
  { ads: 4, price: 3, units: 45 },
];
const reg = new linear.PoissonRegressor({ alpha: 0.1 });
reg.fit(sales, { features: ["ads", "price"], target: "units" });
reg.predict([{ ads: 5, price: 3 }]); // → ~69 (always ≥ 0)
```

## `PolynomialRegression` — nonlinear cost (quadratic curve)

**When to use it** — the data **bends**: quadratic cost curves, price
elasticities, diminishing returns. `degree` 2+ captures curvature a straight
line misses.

```ts
const costs = [
  { quantity: 1, cost: 11 },
  { quantity: 2, cost: 24 },
  { quantity: 3, cost: 41 },
  { quantity: 4, cost: 60 },
  { quantity: 5, cost: 83 },
];
const reg = new linear.PolynomialRegression({ degree: 2 });
reg.fit(costs, { features: ["quantity"], target: "cost" });
reg.predict([{ quantity: 6 }]); // → ≈ 108 (degree-2 least-squares fit)
reg.degree;                      // → 2
```
