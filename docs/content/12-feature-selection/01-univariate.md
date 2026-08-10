---
title: Univariate Scoring
description: Score each feature independently against the target — linear and nonlinear measures.
---

One result per feature, **sorted by score** (most informative first):

| Function | Target | Returns | Captures |
| -------- | ------ | ------- | -------- |
| `chi2(data, spec)` | classification | score + `pValue` | linear (requires non-negative features) |
| `fClassif(data, spec)` | classification | score + `pValue` | linear (ANOVA F) |
| `mutualInfoClassif(data, spec, opts?)` | classification | score | **non-linear** links |
| `mutualInfoRegression(data, spec, opts?)` | regression | score | **non-linear** links |

```ts
import { featureSelection } from "@dnax/ml";

const ranking = featureSelection.mutualInfoClassif(data, {
  features: ["age", "solde", "nb_visites", "inutile"],
  target: "achete",
});
// → [
//     { feature: "age", score: 0.42 },
//     { feature: "nb_visites", score: 0.18 },
//     { feature: "solde", score: 0.09 },
//     { feature: "inutile", score: 0.01 },
//   ]  — most informative first

const top3 = ranking.slice(0, 3).map((r) => r.feature);
clf.fit(data, { features: top3, target: "achete" });
```

`mutualInfo*` options: `discreteFeatures` (boolean or per-feature), `nNeighbors`
(default 5), `randomState`.
