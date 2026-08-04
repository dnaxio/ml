import { linear, monitoring, predictStream, fillPredictStream } from "./index";

// --- Smoke : le package fonctionne ---
const students = [
  { name: "koffi", note: 1, admis: true },
  { name: "Jules", note: 2, admis: false },
  { name: "Aya", note: 3, admis: true },
  { name: "Sam", note: 4, admis: false },
  { name: "Léo", note: 5, admis: true },
];
const reg = new linear.LinearRegression();
reg.fit(students, { features: ["note", "admis"], target: "note" });
const batch = reg.predict(students);
console.log("Smoke — fit/predict OK:", batch.length === 5 && typeof batch[0] === "number");

// --- Test S1 : predictStream == predict par lots ---
async function* gen(rows: typeof students) {
  for (const r of rows) yield r;
}
const streamed: number[] = [];
for await (const p of predictStream(reg, gen(students), { chunkSize: 2 })) {
  streamed.push(p);
}
console.log(
  "S1 — predictStream == batch:",
  streamed.length === batch.length && streamed.every((p, i) => p === batch[i]),
);

// --- Test S2 : fillPredictStream remplit les rows ---
const filled: Record<string, unknown>[] = [];
await fillPredictStream(
  reg,
  students,
  (row) => {
    filled.push(row);
  },
  { chunkSize: 3 },
);
console.log(
  "S2 — fillPredictStream remplit le target:",
  filled.length === 5 && filled.every((r) => typeof r.note === "number" && "note" in r),
);

// --- Test O1 : CUSUM.update == scores() sur la même série ---
const days = Array.from({ length: 18 }, (_, i) => ({
  day: i + 1,
  sales: i < 12 ? 120 + (i % 3) : 120 + (i - 11) * 20,
}));
const cusum = new monitoring.CUSUM();
cusum.fit(days.slice(0, 12), { field: "sales" });
const batchScores = cusum.scores(days);
const onlineScores: number[] = [];
const onlineAlerts: boolean[] = [];
for (const d of days) {
  const r = cusum.update(d);
  onlineScores.push(r.score);
  onlineAlerts.push(r.alert);
}
const batchAlerts = cusum.predict(days).map((a) => a === 1);
console.log(
  "O1 — CUSUM.update == scores():",
  onlineScores.every((s, i) => s === batchScores[i]),
);
console.log("O1 — CUSUM.update == predict():", onlineAlerts.every((a, i) => a === batchAlerts[i]));
console.log("O1 — alertes détectées:", onlineAlerts.filter(Boolean).length >= 1);

// --- Test O2 : EWMA.update == predict() ---
const ewma = new monitoring.EWMA({ lambda: 0.25, limit: 3 });
ewma.fit(days.slice(0, 12), { field: "sales" });
const ewmaAlerts: boolean[] = [];
for (const d of days) ewmaAlerts.push(ewma.update(d).alert);
console.log(
  "O2 — EWMA.update == predict():",
  ewmaAlerts.every((a, i) => a === (ewma.predict(days)[i] === 1)),
);

// --- Test O3 : reset() relance la surveillance ---
cusum.reset();
const first = cusum.update(days[0]!);
console.log("O3 — reset relance (premier point non alerté):", first.score === batchScores[0] && !first.alert);

// --- Test O4 : ParallelMonitor.update ---
const pm = new monitoring.ParallelMonitor();
const multi = days.map((d, i) => ({
  ...d,
  paracetamol: d.sales,
  ibuprofen: 60 + (i % 4) * 2,
}));
pm.fit(multi.slice(0, 12), { fields: ["paracetamol", "ibuprofen"] });
const pmOnline: string[][] = [];
for (const m of multi) pmOnline.push(pm.update(m).alertFields);
console.log(
  "O4 — ParallelMonitor.update (paracetamol déclenche):",
  pmOnline.slice(13).every((f) => f.includes("paracetamol")),
);

// --- Test O5 : SeasonalMonitor.update (smoke) ---
const baseDate = new Date(2026, 6, 27);
const weeks = Array.from({ length: 21 }, (_, i) => ({
  date: new Date(baseDate.getTime() + i * 86400000),
  sales: i < 14 ? 100 : 100 + (i - 13) * 20,
}));
const sm = new monitoring.SeasonalMonitor();
sm.fit(weeks.slice(0, 14), { field: "sales", dateField: "date" });
const smAlerts: boolean[] = [];
for (const w of weeks) smAlerts.push(sm.update(w).alert);
console.log(
  "O5 — SeasonalMonitor.update détecte la montée:",
  smAlerts.slice(14).some(Boolean) && smAlerts.slice(0, 14).every((a) => !a),
);
