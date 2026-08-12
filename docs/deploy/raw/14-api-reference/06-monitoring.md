# Monitoring

> CUSUM, EWMA, ParallelMonitor and SeasonalMonitor — params and extras.

`MonitorSpec` = `{ field, missing? }`. Common surface: `fit`, `predict`,
`scores`, `fill_predict`, `export`, `load`.

<table>
<thead>
  <tr>
    <th>
      Class
    </th>
    
    <th>
      Params
    </th>
    
    <th>
      Extra
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        CUSUM
      </code>
    </td>
    
    <td>
      <code>
        target
      </code>
      
      , <code>
        std
      </code>
      
      , <code>
        k
      </code>
      
       (0.5σ), <code>
        h
      </code>
      
       (5σ), <code>
        direction
      </code>
      
       (<code>
        'increase'
      </code>
      
      |<code>
        'decrease'
      </code>
      
      |<code>
        'both'
      </code>
      
      ), <code>
        robust
      </code>
      
      , <code>
        alertField
      </code>
    </td>
    
    <td>
      <code>
        changePoint(data)
      </code>
      
      , <code>
        update(row)
      </code>
      
      , <code>
        reset()
      </code>
      
       · getters <code>
        target
      </code>
      
      , <code>
        std
      </code>
      
      , <code>
        slack
      </code>
      
      , <code>
        threshold
      </code>
      
      , <code>
        direction
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        EWMA
      </code>
    </td>
    
    <td>
      <code>
        lambda
      </code>
      
       (0.25), <code>
        limit
      </code>
      
       (3), <code>
        target
      </code>
      
      , <code>
        std
      </code>
      
      , <code>
        robust
      </code>
      
      , <code>
        alertField
      </code>
    </td>
    
    <td>
      <code>
        limits(data)
      </code>
      
      , <code>
        update(row)
      </code>
      
      , <code>
        reset()
      </code>
      
       · getters <code>
        target
      </code>
      
      , <code>
        std
      </code>
      
      , <code>
        lambda
      </code>
      
      , <code>
        limit
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        ParallelMonitor
      </code>
    </td>
    
    <td>
      <code>
        model
      </code>
      
       (<code>
        'cusum'
      </code>
      
      |<code>
        'ewma'
      </code>
      
      ), <code>
        familyError
      </code>
      
      , <code>
        alertField
      </code>
      
      , <code>
        alertFieldsField
      </code>
    </td>
    
    <td>
      <code>
        ParallelSpec
      </code>
      
       = <code>
        { fields, params?, missing? }
      </code>
      
       · <code>
        alertFields(data)
      </code>
      
      , <code>
        update(row)
      </code>
      
       · getters <code>
        monitoredFields
      </code>
      
      , <code>
        targets
      </code>
      
      , <code>
        params
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        SeasonalMonitor
      </code>
    </td>
    
    <td>
      <code>
        model
      </code>
      
      , <code>
        alertField
      </code>
      
      , <code>
        chart
      </code>
    </td>
    
    <td>
      <code>
        SeasonalSpec
      </code>
      
       = <code>
        { field, dateField, missing? }
      </code>
      
       · <code>
        changePoint
      </code>
      
      , <code>
        limits
      </code>
      
       (per model), <code>
        update(row)
      </code>
      
       · getter <code>
        dayProfile
      </code>
    </td>
  </tr>
</tbody>
</table>
