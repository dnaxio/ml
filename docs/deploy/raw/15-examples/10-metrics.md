# Metric Cheat Sheet

> What score, mse, mae, rmse, mape, mcc, prAucScore… measure, and when to prefer each one.

Model methods are available on the models themselves (`reg.score(data)`,
`clf.mae(data)`…); the pure functions are exported from the package root and
called `evaluation.xxx(...)`. Convention: **predictions first, truth second**
for label/regression metrics; **truth first** for probability/curve metrics
(kml style).

## Regression — `score`, `mse`, `mae`, `rmse`, `mape`, `medianAbsoluteError`

<table>
<thead>
  <tr>
    <th>
      Metric
    </th>
    
    <th>
      What it measures
    </th>
    
    <th>
      When to prefer it
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        score
      </code>
      
       (R²)
    </td>
    
    <td>
      share of variance explained — 1 = perfect, 0 = no better than the mean
    </td>
    
    <td>
      <strong>
        overall fit quality
      </strong>
      
      , comparing models
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        mse
      </code>
    </td>
    
    <td>
      mean squared error — 0 = perfect; large errors weigh a lot
    </td>
    
    <td>
      <strong>
        big mistakes are costly
      </strong>
      
       (a 100-unit error is worse than 10× 10-unit)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        rmse
      </code>
    </td>
    
    <td>
      √MSE — same <strong>
        units as the target
      </strong>
    </td>
    
    <td>
      you want the typical error in euros/days/sales
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        mae
      </code>
    </td>
    
    <td>
      mean absolute error — same units, <strong>
        outlier-robust
      </strong>
    </td>
    
    <td>
      the typical error you will actually see in practice
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        mape
      </code>
    </td>
    
    <td>
      average <strong>
        % error
      </strong>
      
       (0.05 = 5% off on average)
    </td>
    
    <td>
      relative error matters (compare small and large items fairly)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        medianAbsoluteError
      </code>
    </td>
    
    <td>
      median of |y − ŷ|
    </td>
    
    <td>
      outliers must not move the metric
    </td>
  </tr>
</tbody>
</table>

## Classification — `score`, `classificationReport`, `mcc`, `balancedAccuracy`

<table>
<thead>
  <tr>
    <th>
      Metric
    </th>
    
    <th>
      What it measures
    </th>
    
    <th>
      When to prefer it
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        score
      </code>
      
       (accuracy)
    </td>
    
    <td>
      share of correct labels
    </td>
    
    <td>
      only on <strong>
        balanced
      </strong>
      
       classes; misleading otherwise
    </td>
  </tr>
  
  <tr>
    <td>
      precision
    </td>
    
    <td>
      of the <em>
        predicted
      </em>
      
       positives, how many are right
    </td>
    
    <td>
      <strong>
        false alarms are costly
      </strong>
      
       (a false fraud alert wastes an investigation)
    </td>
  </tr>
  
  <tr>
    <td>
      recall
    </td>
    
    <td>
      of the <em>
        actual
      </em>
      
       positives, how many were found
    </td>
    
    <td>
      <strong>
        missed positives are costly
      </strong>
      
       (fraud, epidemics — you must not miss cases)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        fScore
      </code>
      
       (F1 / Fβ)
    </td>
    
    <td>
      harmonic balance of precision/recall — β > 1 favors recall
    </td>
    
    <td>
      a single number when you want <strong>
        balance
      </strong>
      
       (default F1)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        classificationReport
      </code>
    </td>
    
    <td>
      accuracy + precision + recall + Fβ + support + confusion matrix
    </td>
    
    <td>
      full picture of a classifier in one call
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        mcc
      </code>
    </td>
    
    <td>
      Matthews correlation −1..+1 — robust to imbalance
    </td>
    
    <td>
      <strong>
        THE number for imbalanced data
      </strong>
      
       (fraud, rare diseases)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        balancedAccuracy
      </code>
    </td>
    
    <td>
      macro recall (mean recall per class)
    </td>
    
    <td>
      “honest” accuracy when classes are skewed
    </td>
  </tr>
</tbody>
</table>

## Probability — `rocAucScore`, `prAucScore`, `logLoss`, `rocCurve`, `optimalThreshold`

<table>
<thead>
  <tr>
    <th>
      Metric
    </th>
    
    <th>
      What it measures
    </th>
    
    <th>
      When to prefer it
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        rocAucScore
      </code>
    </td>
    
    <td>
      ranking quality — 1 = perfect separation, 0.5 = random (binary)
    </td>
    
    <td>
      global ranking quality of a <code>
        predict_proba
      </code>
      
       model
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        prAucScore
      </code>
    </td>
    
    <td>
      precision-recall AUC
    </td>
    
    <td>
      <strong>
        rare events
      </strong>
      
       (epidemic, fraud): drops when false alarms grow, unlike ROC
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        logLoss
      </code>
    </td>
    
    <td>
      cross-entropy — punishes confident <strong>
        and wrong
      </strong>
      
       probabilities
    </td>
    
    <td>
      probability calibration matters (risk scores, betting)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        rocCurve
      </code>
    </td>
    
    <td>
      <code>
        { fpr, tpr, thresholds }
      </code>
      
       — the full curve
    </td>
    
    <td>
      plotting, or choosing where the curve bends
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        optimalThreshold
      </code>
    </td>
    
    <td>
      Youden's J — the threshold that maximizes tpr − fpr
    </td>
    
    <td>
      turning probabilities into <strong>
        alerts/decisions
      </strong>
    </td>
  </tr>
</tbody>
</table>

## Workflow helpers

<table>
<thead>
  <tr>
    <th>
      Function
    </th>
    
    <th>
      What it does
    </th>
    
    <th>
      When to use it
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        trainTestSplit
      </code>
    </td>
    
    <td>
      splits rows into train/test (seeded, stratifiable)
    </td>
    
    <td>
      measure generalization on <strong>
        unseen
      </strong>
      
       rows
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        crossValScore
      </code>
    </td>
    
    <td>
      k-fold scores — one score per fold
    </td>
    
    <td>
      robust model comparison (mean <strong>
        and
      </strong>
      
       fold spread)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        compareModels
      </code>
    </td>
    
    <td>
      trains & ranks several models with CV
    </td>
    
    <td>
      <strong>
        choose the best model family
      </strong>
      
       for your case
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        detectTask
      </code>
    </td>
    
    <td>
      <code>
        "classification"
      </code>
      
       | <code>
        "regression"
      </code>
      
       from a spec
    </td>
    
    <td>
      auto-pick a family and a default scoring
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        predictStream
      </code>
      
       / <code>
        fillPredictStream
      </code>
    </td>
    
    <td>
      chunked inference (generators OK)
    </td>
    
    <td>
      run predictions over <strong>
        millions of rows
      </strong>
      
       without loading them all
    </td>
  </tr>
</tbody>
</table>

**Which metric first?** For regression start with **R²** (overall quality),
then **MAE** (typical error in target units); add **MAPE** if you compare
items of very different sizes. For classification on imbalanced data (fraud,
epidemics) never trust accuracy alone — look at **recall**, **prAucScore** and
**mcc**.
