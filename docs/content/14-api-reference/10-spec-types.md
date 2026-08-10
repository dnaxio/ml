---
title: Spec Types
description: JsonFitSpec, ClusterSpec, MonitorSpec, ParallelSpec, SeasonalSpec, ScanSpec, HotspotSpec and JsonTransformOptions.
---

```ts
// Supervised
interface JsonFitSpec { features: string[]; target: string; options?: JsonTransformOptions }

// Unsupervised
interface ClusterSpec { features: string[]; options?: JsonTransformOptions }
```

> **Supervised vs unsupervised**: supervised models (`JsonFitSpec`, 23 models)
> have a `target` and expose `score`/`mse`/`mae`/`classificationReport`. Unsupervised
> models take only `features` (`ClusterSpec`) and have no `score` — see
> [Core concepts](/01-getting-started/03-core-concepts) for the full classification.

```ts
// Monitoring
interface MonitorSpec { field: string; missing?: "throw" | "fill" }
interface ParallelSpec { fields: string[]; params?: Record<string, unknown>; missing?: "throw" | "fill" }
interface SeasonalSpec { field: string; dateField: string; missing?: "throw" | "fill" }

// Spatial scan
interface ScanSpec {
  zone: string;
  coordinates: [string, string];
  population: string;
  cases: string;
}

// Getis-Ord Gi* hotspot analysis
interface HotspotSpec {
  zone: string;
  coordinates: [string, string];
  cases: string;
}

// Options
interface JsonTransformOptions {
  oneHot?: boolean;      // default false
  dropFirst?: boolean;   // default true
  missing?: "throw" | "fill0" | "drop";  // default 'throw'
  scale?: boolean;       // default false
  noise?: number;        // Gaussian jittering, training only
  noiseSeed?: number;    // reproducibility
}
```
