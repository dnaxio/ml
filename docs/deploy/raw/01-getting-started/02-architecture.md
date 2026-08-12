# Architecture

> How @dnax/ml is structured — the JSON-first core facade, and why it stays dependency-light.

`@dnax/ml` is a **JSON-first facade**: all direct access to the underlying
engine is centralized in a single internal module (the **core facade**), so
swapping the engine — or vendoring it — only requires rewriting that one
layer, never the models themselves.

## Dependencies

`@dnax/ml` declares a single runtime dependency — pinned to an exact version
so its internal serialization format stays frozen — and that dependency is
itself **dependency-free**: the whole runtime footprint is one package. The
monitoring and scan families are pure JavaScript with zero dependencies.

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
