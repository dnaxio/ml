---
title: Architecture
description: How @dnax/ml is structured — the JSON-first facade over @kanaries/ml, and why it stays dependency-light.
---

`@dnax/ml` is a **JSON-first facade** over the `@kanaries/ml` engine (pinned
to an exact version — `1.1.0` — to freeze its internal serialization format):
all direct access to the engine is centralized in a single internal module
(the **core facade**), so swapping the engine — or vendoring it — only
requires rewriting that one layer, never the models themselves.

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
