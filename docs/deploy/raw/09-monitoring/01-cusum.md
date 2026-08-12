# CUSUM

> Cumulative sum — very sensitive to small sustained shifts.

Accumulates deviations from a reference mean: `S_n = max(0, S_{n-1} + (x_n − μ0 − k))`
and alerts when `S_n > h`. Very sensitive to **small sustained shifts**.

**When to use it** — detect the **onset of a sustained shift** in a series:
early epidemic warnings, sales spikes, rising defect rates. Fit on a
**known-normal baseline** first.

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
        target
      </code>
    </td>
    
    <td>
      estimated
    </td>
    
    <td>
      reference mean μ0 (deviations are measured against it)
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        std
      </code>
    </td>
    
    <td>
      estimated
    </td>
    
    <td>
      series standard deviation σ
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        k
      </code>
    </td>
    
    <td>
      <code>
        0.5·σ
      </code>
    </td>
    
    <td>
      allowable slack — sub-k shifts are ignored as background noise
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        h
      </code>
    </td>
    
    <td>
      <code>
        5·σ
      </code>
    </td>
    
    <td>
      alert threshold — lower = earlier but noisier alarms
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        direction
      </code>
    </td>
    
    <td>
      <code>
        'increase'
      </code>
    </td>
    
    <td>
      shift to detect: <code>
        'increase'
      </code>
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
      field name filled by <code>
        fill_predict
      </code>
    </td>
  </tr>
</tbody>
</table>

Methods: `fit`, `predict`, `scores`, `fill_predict`, `changePoint(data)`,
`update(row)`, `reset()`, `export` / `load` · getters `target`, `std`,
`slack`, `threshold`, `direction`.
