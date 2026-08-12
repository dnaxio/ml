# Model Scoring

> score, mse, mae, classificationReport and rocAucScore on rows with ground truth.

```ts
// Regressors → R² + MSE + MAE
reg.score(data); // number — R² (1 = perfect fit)
reg.mse(data); // number — mean squared error (0 = perfect)
reg.mae(data); // number — mean absolute error (0 = perfect)

// Classifiers → accuracy + full report
clf.score(data); // number — accuracy
clf.classificationReport(data); // { accuracy, precision, recall, fScore,
//   support, confusionMatrix }

// Fβ — beta weights recall (default 1 = F1; 2 = recall twice as important,
// 0.5 = precision twice as important)
clf.classificationReport(data, 2).fScore; // F2
import { fbetaFromPrecisionRecall } from "@dnax/ml";
fbetaFromPrecisionRecall(0.5, 0.8, 2); // 0.714 — F2 from precision/recall

// Classifiers with predict_proba (LogisticRegression, ExtraTreeClassifier,
// AdaBoostClassifier, GradientBoostingClassifier, XGBoostClassifier) → ROC
clf.rocAucScore(data); // number — AUC (1 = perfect, 0.5 = random, binary only)
```

## Reading the scores

- **R²**: 1 = perfect, 0 = no better than predicting the mean, negative =
worse than the mean. Compare models with the CV **mean**, and check the
fold spread (a stable model has tight folds).
- **AUC**: threshold-free ranking quality — 1 = perfect separation,
0.5 = random.
- **Fβ** (classification): `classificationReport(data, beta)` — `fScore` is
F1 by default; β > 1 favors recall (missed positives are costly),
β < 1 favors precision (false alarms are costly). Never rely on accuracy
alone on imbalanced data.
- On small datasets, single-fold scores are noisy: prefer the mean of a
repeated k-fold over one arbitrary split.
