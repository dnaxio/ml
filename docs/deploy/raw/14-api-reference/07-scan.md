# Spatial Scan

> SpatialScan (Kulldorff) and GetisOrd (Gi*) — params, methods and result shapes.

`ScanSpec` = `{ zone, coordinates: [xField, yField], population, cases }` ·
`HotspotSpec` = `{ zone, coordinates: [xField, yField], cases }`.

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
      Methods & getters
    </th>
  </tr>
</thead>

<tbody>
  <tr>
    <td>
      <code>
        SpatialScan
      </code>
    </td>
    
    <td>
      <code>
        replications
      </code>
      
       (199), <code>
        significance
      </code>
      
       (0.05), <code>
        maxWindowFraction
      </code>
      
       (0.5), <code>
        randomState
      </code>
      
      , <code>
        clusterField
      </code>
    </td>
    
    <td>
      <code>
        fit
      </code>
      
      , <code>
        cluster(data)
      </code>
      
       → <code>
        ScanCluster | null
      </code>
      
      , <code>
        predict
      </code>
      
      , <code>
        fill_predict
      </code>
      
      , <code>
        export
      </code>
      
      , <code>
        load
      </code>
      
       · getters <code>
        zonesList
      </code>
      
      , <code>
        expectedRate
      </code>
    </td>
  </tr>
  
  <tr>
    <td>
      <code>
        GetisOrd
      </code>
    </td>
    
    <td>
      <code>
        distance
      </code>
      
      , <code>
        significance
      </code>
      
       (0.05), <code>
        hotField
      </code>
    </td>
    
    <td>
      <code>
        fit
      </code>
      
      , <code>
        hotspots(data)
      </code>
      
       → <code>
        HotspotResult[]
      </code>
      
      , <code>
        predict
      </code>
      
      , <code>
        fill_predict
      </code>
      
      , <code>
        export
      </code>
      
      , <code>
        load
      </code>
      
       · getters <code>
        zonesList
      </code>
      
      , <code>
        distance
      </code>
      
      , <code>
        significance
      </code>
    </td>
  </tr>
</tbody>
</table>

`ScanCluster` = `{ zones: string[], cases, expected, llr, pValue }` ·
`HotspotResult` = `{ zone, zScore, pValue, hot, cold }`.
