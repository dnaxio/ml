# Overview

> Score your models — R², MSE, MAE, accuracy, classification reports, ROC/AUC, train/test splits and cross-validation.

> **Scoped vs flat import** — scoped: `evaluation.compareModels({...}, data, spec)` ·
> flat: `import { compareModels } from "@dnax/ml"`. Both are identical.

Every supervised model scores itself on rows that include the `target` field
(ground truth), reusing the transformation learned at fit time (rows dropped
by the `'drop'` strategy are excluded).

Implemented on: 14 regressors (`score`/`mse`/`mae`) and 9 classifiers
(`score`/`classificationReport(data, beta?)`, + `rocAucScore` on the 5 with
`predict_proba`). Clustering, IsolationForest, monitoring and scan have no
target → no `score`.

The ground truth convention matches kml: **predictions first, truth second**.

<table>
<thead>
  <tr>
    <th>
      Page
    </th>
    
    <th>
      What it covers
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <a href="/11-evaluation/01-model-scoring">
        Model scoring
      </a>
    </td>
    
    <td>
      <code>
        score
      </code>
      
      , <code>
        mse
      </code>
      
      , <code>
        mae
      </code>
      
      , <code>
        classificationReport
      </code>
      
      , <code>
        rocAucScore
      </code>
      
       + reading the scores
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/11-evaluation/02-splits-and-cv">
        Splits & cross-validation
      </a>
    </td>
    
    <td>
      <code>
        trainTestSplit
      </code>
      
      , <code>
        crossValScore
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/11-evaluation/03-leaderboard">
        Leaderboard
      </a>
    </td>
    
    <td>
      <code>
        compareModels
      </code>
      
       + <code>
        detectTask
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/11-evaluation/04-streaming">
        Streaming
      </a>
    </td>
    
    <td>
      <code>
        predictStream
      </code>
      
      , <code>
        fillPredictStream
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <a href="/11-evaluation/05-metric-library">
        Metric library
      </a>
    </td>
    
    <td>
      <code>
        rmse
      </code>
      
      , <code>
        mape
      </code>
      
      , <code>
        mcc
      </code>
      
      , <code>
        prAucScore
      </code>
      
      , <code>
        optimalThreshold
      </code>
      
      …
    </td>
  </tr>
</tbody>
</table>
