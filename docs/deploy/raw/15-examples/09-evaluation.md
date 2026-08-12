# Evaluation

> 2 examples — model leaderboard, and detectTask + splits + pure metrics.

## `compareModels` — which model for this case? (leaderboard)

**When to use it** — **rank models** on your data with cross-validation: pick
the best family without relying on one arbitrary split.

```ts
const employees = [
  { years: 1, salary: 30000 },
  { years: 2, salary: 36000 },
  { years: 3, salary: 42000 },
  { years: 4, salary: 48000 },
  { years: 5, salary: 54000 },
];
const leaderboard = evaluation.compareModels(
  {
    linear: () => new linear.LinearRegression(),
    ridge: () => new linear.RidgeRegression({ alpha: 1 }),
    tree: () => new tree.DecisionTreeRegressor({ max_depth: 2 }),
  },
  employees,
  { features: ["years"], target: "salary" },
  { cv: 3, randomState: 42 },
);
// → [{ name: "linear", mean: ~1.00, std: ~0.00, scores: [...] },
//     { name: "ridge", mean: ~0.64, ... }, ...] — sorted best first
```

## `detectTask` + `trainTestSplit` + pure metrics

**When to use them** — `detectTask` tells you if a spec is classification or
regression (pick the right family/scoring). `trainTestSplit` holds out rows to
measure generalization. The pure metrics below score predictions directly.

```ts
const employees = [
  { years: 1, salary: 30000 },
  { years: 2, salary: 36000 },
  { years: 3, salary: 42000 },
  { years: 4, salary: 48000 },
  { years: 5, salary: 54000 },
];
const clients = [
  { age: 20, balance: 100, bought: false },
  { age: 35, balance: 5000, bought: true },
  { age: 45, balance: 9000, bought: true },
  { age: 25, balance: 300, bought: false },
  { age: 30, balance: 1200, bought: false },
];
evaluation.detectTask(employees, { features: ["years"], target: "salary" }); // → "regression"
evaluation.detectTask(clients, { features: ["age", "balance"], target: "bought" }); // → "classification"

const { train, test } = evaluation.trainTestSplit(employees, { testSize: 0.4, randomState: 42 });
// → { train: 3 rows, test: 2 rows }

const reg = new linear.LinearRegression();
reg.fit(employees, { features: ["years"], target: "salary" });
const preds = reg.predict(employees);
const truth = employees.map((e) => e.salary);

evaluation.rmse(preds, truth);                // → ≈ 0 (near-perfect fit)
evaluation.mape(preds, truth);                // → ≈ 0
evaluation.mcc([0, 1, 1, 0], [0, 1, 1, 0]);  // → 1 (perfect classification)
```
