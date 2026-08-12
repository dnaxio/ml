# Tree

> 4 examples — DecisionTree and ExtraTree classifiers and regressors.

## `DecisionTreeClassifier` — flu? (readable if/then rules)

**When to use it** — you need **readable if/then rules** you can explain to a
human (regulatory audits, medical triage, loan decisions). Best with a small
set of features.

```ts
const patients = [
  { fever: 40, cough: true, flu: true },
  { fever: 37, cough: false, flu: false },
  { fever: 39, cough: true, flu: true },
  { fever: 36.5, cough: false, flu: false },
  { fever: 38.5, cough: false, flu: false },
];
const clf = new tree.DecisionTreeClassifier({ max_depth: 3 });
clf.fit(patients, { features: ["fever", "cough"], target: "flu" });
clf.predict([{ fever: 39.5, cough: true }]); // → [1]
clf.featureImportances;                      // → [cough, fever] — fever dominates
                                             //   (perfect fever split on 5 rows)
```

## `DecisionTreeRegressor` — salary by tiers (leaf mean, not a line)

**When to use it** — the relationship is **stepwise**: salary bands, tax
brackets, quantity discounts. It predicts the mean of the bucket, not a smooth
line — perfect when real life jumps in levels.

```ts
const employees = [
  { years: 1, salary: 30000 },
  { years: 3, salary: 36000 },
  { years: 5, salary: 45000 },
  { years: 10, salary: 70000 },
  { years: 15, salary: 92000 },
];
const reg = new tree.DecisionTreeRegressor({ max_depth: 3 });
reg.fit(employees, { features: ["years"], target: "salary" });
reg.predict([{ years: 8 }]); // → mean of the leaf group (piecewise, not linear)
```

## `ExtraTreeClassifier` — subscription churn (with probabilities)

**When to use it** — same family but **randomized** and it **gives
probabilities** (`predict_proba`, `rocAucScore`). Use when you need a
probability output from a cheap tree model.

```ts
const subscribers = [
  { months: 1, calls: 20, churned: false },
  { months: 3, calls: 55, churned: false },
  { months: 8, calls: 120, churned: true },
  { months: 12, calls: 200, churned: true },
  { months: 5, calls: 40, churned: false },
];
const clf = new tree.ExtraTreeClassifier({ max_depth: 3 });
clf.fit(subscribers, { features: ["months", "calls"], target: "churned" });
clf.predict_proba([{ months: 10, calls: 180 }]); // → [[~0, ~1]] (leaf fully confident)
```

## `ExtraTreeRegressor` — tomorrow's temperature

**When to use it** — randomized tree for a **continuous target**: a solid
cheap base model (forecasts, sensor values). `randomState` keeps it
reproducible.

```ts
const days = [
  { day: 1, temperature: 12 },
  { day: 2, temperature: 14 },
  { day: 3, temperature: 13 },
  { day: 4, temperature: 17 },
  { day: 5, temperature: 19 },
];
const reg = new tree.ExtraTreeRegressor({ max_depth: 3, randomState: 42 });
reg.fit(days, { features: ["day"], target: "temperature" });
reg.predict([{ day: 6 }]); // → mean of the nearest leaf
```
