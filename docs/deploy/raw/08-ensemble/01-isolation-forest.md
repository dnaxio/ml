# IsolationForest

> Unsupervised anomaly detection — flags points that are isolated quickly.

Unsupervised: builds random partition trees and flags points that are
isolated quickly (short path length) as anomalies. **No target** — only
`features`. Returns binary labels and a continuous `anomaly_score`.

**When to use it** — **anomaly detection without labels**: flag the
transactions/rows that stand out — **fraud**, **IoT sensor faults**, or
**data cleaning** before supervised training. `predict` → 1 = anomaly;
`anomaly_score` ranks them.

```ts
import { ensemble } from "@dnax/ml";

const model = new ensemble.IsolationForest({ random_state: 42 });
model.fit(data, { features: ["montant", "heure"] }); // build the forest
const labels = model.predict(data); // → [0, 0, 0, 1, 0, 1] — 1 = anomaly

const score = model.anomaly_score({ montant: 9000, heure: 3 });
// → 0.62 — continuous anomaly score (custom thresholds)
```

Methods: `fit`, `predict` (1 = anomaly), `anomaly_score(row)`, `fill_predict`,
`export` / `load` · getters `columnNames`, `droppedRows`.
