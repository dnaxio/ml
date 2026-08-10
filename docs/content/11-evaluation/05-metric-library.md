---
title: Metric Library
description: Pure functions on arrays — regression, classification and probability metrics.
---

Pure functions on arrays (exported from `@dnax/ml`), useful for custom
workflows and thresholds. Convention: **predictions first, truth second** for
label metrics; **truth first** for probability/curve metrics (kml style).

## Regression

```ts
import { rmse, mape, medianAbsoluteError } from "@dnax/ml";

rmse(preds, truth);            // √MSE — same units as the target (euros, sales…)
mape(preds, truth);            // average % off (0.05 = 5% off on average)
medianAbsoluteError(preds, truth); // robust to outliers (median of |y − p|)
```

## Classification

```ts
import { mcc, balancedAccuracy, logLoss, prAucScore, rocCurve, optimalThreshold } from "@dnax/ml";

mcc(preds, truth);             // Matthews −1..+1 — the robust number for imbalanced data
balancedAccuracy(preds, truth); // macro recall — "honest" accuracy on imbalance

// Needs a predict_proba model (proba = model.predict_proba(data)[:, 1]):
logLoss(truth, proba);         // cross-entropy — punishes confident & wrong probas
prAucScore(truth, proba);      // precision-recall AUC — THE metric for epidemic/fraud
rocCurve(truth, proba);        // { fpr, tpr, thresholds } — full curve for plots
optimalThreshold(truth, proba); // Youden's J — best decision threshold
```

```ts
// Pharma example: turn probabilities into alerts at the optimal threshold
const proba = clf.predict_proba(newSales).map((p) => p[1]);
const seuil = optimalThreshold(historiqueVrai, historiqueProba);
const alertes = proba.map((p) => p >= seuil); // → boolean alerts
```

**Which one for imbalanced detection?** `prAucScore` (and `mcc`) drop when
false alarms grow, unlike `rocAucScore` which stays optimistic — prefer them
for epidemic/fraud cases.
