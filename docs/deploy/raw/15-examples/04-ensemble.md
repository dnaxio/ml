# Ensemble

> 9 examples — forests, boosting and IsolationForest.

## `IsolationForest` — abnormal transaction (unsupervised)

**When to use it** — **anomaly detection without labels**: flag the
transactions/rows that stand out (fraud pre-screening, stock discrepancies,
defective batches). `predict` → 1 = anomaly; `anomaly_score` ranks them.

```ts
const transactions = [
  { amount: 120, hour: 14 },
  { amount: 90, hour: 10 },
  { amount: 3000, hour: 3 },   // ← anomaly
  { amount: 150, hour: 18 },
  { amount: 80, hour: 9 },
];
const model = new ensemble.IsolationForest({ random_state: 42 });
model.fit(transactions, { features: ["amount", "hour"] });
model.predict(transactions); // → [0, 0, 1, 0, 0] (1 = anomaly)
model.anomaly_score({ amount: 3000, hour: 3 }); // → high continuous score
```

## `RandomForestClassifier` — fraudulent prescription

**When to use it** — the **robust default** for tabular classification: handles
nonlinearity and feature interactions with little tuning. Labels only (no
`predict_proba`).

```ts
const rxSales = [
  { qty: 60, age: 30, fraud: false },
  { qty: 30, age: 25, fraud: false },
  { qty: 420, age: 67, fraud: true },
  { qty: 480, age: 71, fraud: true },
  { qty: 90, age: 33, fraud: false },
];
const clf = new ensemble.RandomForestClassifier({ nEstimators: 50, maxDepth: 3, randomState: 42 });
clf.fit(rxSales, { features: ["qty", "age"], target: "fraud" });
clf.predict([{ qty: 400, age: 65 }]); // → [1]
```

## `RandomForestRegressor` — selling price from cost + margin

**When to use it** — same default for **regression**: solid results without
much tuning, harder to overfit than a single tree. Good when features interact
(cost × margin → price).

```ts
const products = [
  { cost: 5, margin: 2, price: 10 },
  { cost: 8, margin: 3, price: 16 },
  { cost: 12, margin: 5, price: 24 },
  { cost: 20, margin: 8, price: 40 },
  { cost: 30, margin: 10, price: 60 },
];
const reg = new ensemble.RandomForestRegressor({ nEstimators: 50, randomState: 42 });
reg.fit(products, { features: ["cost", "margin"], target: "price" });
reg.predict([{ cost: 15, margin: 6 }]); // → ≈ 30
```

## `AdaBoostClassifier` — loan default risk

**When to use it** — sequential boosting of weak learners. Try it when
`RandomForestClassifier` underperforms; it returns **probabilities** and works
well on moderate-size tabular data.

```ts
const loans = [
  { income: 1500, term: 12, default: false },
  { income: 1800, term: 24, default: false },
  { income: 900, term: 36, default: true },
  { income: 1200, term: 48, default: true },
  { income: 2200, term: 12, default: false },
];
const clf = new ensemble.AdaBoostClassifier({ nEstimators: 20, randomState: 42 });
clf.fit(loans, { features: ["income", "term"], target: "default" });
clf.predict_proba([{ income: 1000, term: 36 }]); // → [[~0.3, ~0.7]]
```

## `AdaBoostRegressor` — forecasting demand

**When to use it** — boosting for a **continuous target**: good for demand,
price and load forecasts when the features are clean.

```ts
const weeks = [
  { week: 1, demand: 100 },
  { week: 2, demand: 105 },
  { week: 3, demand: 98 },
  { week: 4, demand: 112 },
  { week: 5, demand: 118 },
];
const reg = new ensemble.AdaBoostRegressor({ nEstimators: 20, randomState: 42 });
reg.fit(weeks, { features: ["week"], target: "demand" });
reg.predict([{ week: 6 }]); // → ≈ 120
```

## `GradientBoostingClassifier` — cardiovascular risk (calibrated probability)

**When to use it** — the **strongest off-the-shelf tabular classifier** with
calibrated probabilities. Use when accuracy matters most and you can tune
`learningRate` / `maxDepth`.

```ts
const patients = [
  { age: 30, chol: 1.8, risk: false },
  { age: 45, chol: 2.2, risk: false },
  { age: 60, chol: 2.8, risk: true },
  { age: 70, chol: 3.1, risk: true },
  { age: 50, chol: 2.1, risk: false },
];
const clf = new ensemble.GradientBoostingClassifier({ nEstimators: 20, learningRate: 0.1, maxDepth: 2, randomState: 42 });
clf.fit(patients, { features: ["age", "chol"], target: "risk" });
clf.predict_proba([{ age: 65, chol: 3.0 }]); // → [[~0.1, ~0.9]]
```

## `GradientBoostingRegressor` — monthly sales

**When to use it** — often the **best default regressor** on tabular data:
nonlinear trends + feature interactions with well-calibrated errors.

```ts
const months = [
  { month: 1, sales: 120 },
  { month: 2, sales: 135 },
  { month: 3, sales: 128 },
  { month: 4, sales: 150 },
  { month: 5, sales: 160 },
];
const reg = new ensemble.GradientBoostingRegressor({ nEstimators: 20, learningRate: 0.1, maxDepth: 2, randomState: 42 });
reg.fit(months, { features: ["month"], target: "sales" });
reg.predict([{ month: 6 }]); // → ≈ 168
```

## `XGBoostClassifier` — bank fraud (strong separation)

**When to use it** — regularized boosting, **state of the art on large tabular
data** (hundreds+ rows). On tiny sets it underfits (see the pitfall below).

```ts
const operations = [
  { amount: 100, time: 10, fraud: false },
  { amount: 90, time: 8, fraud: false },
  { amount: 9000, time: 2, fraud: true },
  { amount: 7500, time: 1, fraud: true },
  { amount: 8200, time: 3, fraud: true },
];
const clf = new ensemble.XGBoostClassifier({ nEstimators: 100, learningRate: 0.3, maxDepth: 3, randomState: 42 });
clf.fit(operations, { features: ["amount", "time"], target: "fraud" });
clf.predict([{ amount: 8500, time: 2 }]);         // → [1]
clf.predict_proba([{ amount: 8500, time: 2 }]);   // → [[~0.4, ~0.6]]
// ⚠️ On 5 rows XGBoost underfits (probabilities stay near the prior) —
//    it needs hundreds of rows to shine. See the pitfalls in /08-ensemble.
```

## `XGBoostRegressor` — stock price

**When to use it** — same for regression: the go-to choice on **big datasets
with many features**; overkill on 5 rows.

```ts
const days = [
  { day: 1, price: 100 },
  { day: 2, price: 102 },
  { day: 3, price: 101 },
  { day: 4, price: 105 },
  { day: 5, price: 108 },
];
const reg = new ensemble.XGBoostRegressor({ nEstimators: 20, maxDepth: 2, randomState: 42 });
reg.fit(days, { features: ["day"], target: "price" });
reg.predict([{ day: 6 }]); // → ≈ 109
```
