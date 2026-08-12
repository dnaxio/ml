# Streaming

> predictStream and fillPredictStream — memory-bounded inference over millions of rows.

`predictStream` and `fillPredictStream` process rows **in chunks**, so you can
run inference over millions of rows (file, database, API) without loading
them all into memory. They accept arrays, generators and async generators:

```ts
import { predictStream, fillPredictStream } from "@dnax/ml";

// read rows from anywhere (NDJSON file, DB cursor, API)
async function* rows() {
  for await (const line of Bun.stdin.stream()) {
    yield JSON.parse(line.toString());
  }
}

// stream predictions, one per row
for await (const pred of predictStream(reg, rows(), { chunkSize: 1000 })) {
  // → one prediction per row, as they arrive
}

// stream enriched rows (like fill_predict, memory bounded)
await fillPredictStream(reg, rows(), (row) => {
  console.log(row); // { ..., target: prediction }
});
```

Works with **every supervised model** (one generic implementation — no
per-model duplication). `chunkSize` controls the memory/compute trade-off.
