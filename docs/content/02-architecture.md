---
title: Architecture
description: How @dnax/ml is structured — the JSON-first facade over @kanaries/ml, and why it stays dependency-light.
navigation:
  icon: lucide:blocks
---

`@dnax/ml` is a **JSON-first facade** over the `@kanaries/ml` engine (pinned
to an exact version — `1.1.0` — to freeze its internal serialization format):

```
algorithm/
├── core/            ← the SINGLE import point of @kanaries/ml
│   ├── kml.ts         (engine namespace + factories)
│   └── state.ts       (centralized internal-state access + serialization)
├── linear/ clusters/ tree/ ensemble/   ← wrappers (never import kml directly)
├── monitoring/ scan/                   ← pure JS, zero dependency
└── transformation/                     ← JSON → matrix + scaler via core
```

Swapping the engine (or vendoring it) only requires rewriting
`algorithm/core/` — the 25 wrappers never touch the package directly.

## Dependencies

`@dnax/ml` declares a single runtime dependency: `@kanaries/ml` (pinned to an
exact version), which is itself **dependency-free** — the whole runtime
footprint is one package. The monitoring and scan families are pure
JavaScript with zero dependencies.

## Design principles

1. **JSON-first** — the same shape in and out: `fit(data, spec)`, rows come
   in, rows (enriched) come out. No manual matrix conversion.
2. **sklearn vocabulary** — `coef`, `intercept`, `features` / `target`,
   `fit` / `predict` / `score`, `getParams()` / `setParams()`.
3. **One API everywhere** — every model exposes `fit` → `predict` →
   `fill_predict` → `export` / `load`. A single pattern to learn.
4. **Versioned persistence** — every export is tagged with a format version;
   `load` refuses incompatible files.
5. **Deterministic by default** — alphabetical feature ordering, seeded
   randomness everywhere (`randomState` / `random_state` / `noiseSeed`).

## License

MIT
