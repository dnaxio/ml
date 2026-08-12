# EWMA

> Exponentially weighted moving average — smooths noise, fewer false alarms.

Smooths the series with a decaying memory (`z_n = λ·x_n + (1−λ)·z_{n−1}`) and
alerts when the smoothed value leaves `μ0 ± L·σ_z(n)`. Recovers quickly after
isolated spikes.

**When to use it** — same goal as CUSUM but it **smooths the noise** first:
fewer false alarms on jittery series. `limits()` gives the control band to
draw on a chart.

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
        lambda
      </code>
    </td>
    
    <td>
      <code>
        0.25
      </code>
    </td>
    
    <td>
      smoothing factor λ ∈ (0,1] — small = smooth, sensitive to trends
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        limit
      </code>
    </td>
    
    <td>
      <code>
        3
      </code>
    </td>
    
    <td>
      control-limit width L (in σ units)
    </td>
  </tr>
  
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
      reference mean μ0
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

Methods: `fit`, `predict`, `scores`, `fill_predict`, `limits(data)`,
`update(row)`, `reset()`, `export` / `load` · getters `target`, `std`,
`lambda`, `limit`.
