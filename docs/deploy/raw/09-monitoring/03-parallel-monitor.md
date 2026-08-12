# ParallelMonitor

> Monitor several series at once — one chart per field, per-field alerts.

Monitors several series simultaneously with one chart per field. An alert
fires when **at least one** chart fires, and `alertFields` reports **which**
fields triggered (interpretable alarms).

**When to use it** — you watch **several series at once** (many products,
meters, branches): it monitors each field and tells you **which ones**
triggered.

```ts
import { monitoring } from "@dnax/ml";

const pm = new monitoring.ParallelMonitor(); // one CUSUM per field (default)
pm.fit(days, { fields: ["paracetamol", "ibuprofen"] });

const alerts = pm.predict(days); // 1 when at least one chart fires
const fields = pm.alertFields(days); // ["paracetamol"] per row — which fields
const tracked = pm.fill_predict(days); // + alert: boolean + alertFields: string[]

// EWMA charts, with a controlled family-wise false-alarm rate (Bonferroni):
const pmE = new monitoring.ParallelMonitor({ model: "ewma", familyError: 0.05 });
pmE.fit(days, { fields: ["paracetamol", "ibuprofen"] });
```

**Why the Bonferroni correction matters**: with N independent charts, the
chance of at least one false alarm grows with N (5 charts at 5% each → ~23%
family-wise). `familyError` divides the risk across charts (each chart gets
the two-sided level `familyError / N`). When omitted, each chart keeps its
own conservative defaults (`h = 5·σ`, `L = 3`).

<table>
<thead>
  <tr>
    <th>
      Param
    </th>
    
    <th>
      Default
    </th>
    
    <th>
      Role
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        model
      </code>
    </td>
    
    <td>
      <code>
        'cusum'
      </code>
    </td>
    
    <td>
      chart type per field: <code>
        'cusum'
      </code>
      
       or <code>
        'ewma'
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        familyError
      </code>
    </td>
    
    <td>
      —
    </td>
    
    <td>
      family-wise false-alarm rate (Bonferroni across fields)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        alertField
      </code>
    </td>
    
    <td>
      <code>
        'alert'
      </code>
    </td>
    
    <td>
      boolean field filled by <code>
        fill_predict
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        alertFieldsField
      </code>
    </td>
    
    <td>
      <code>
        'alertFields'
      </code>
    </td>
    
    <td>
      field listing the triggering fields
    </td>
  </tr>
</tbody>
</table>

Per-field overrides go in the spec: `{ fields, params: { paracetamol: { k, h } } }`
(user values always win over the Bonferroni adjustment).

Methods: `fit`, `predict`, `scores`, `fill_predict`, `alertFields(data)`,
`update(row)`, `export` / `load` · getters `monitoredFields`, `targets`,
`params`.
