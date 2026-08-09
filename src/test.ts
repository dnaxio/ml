import { linear, monitoring, clusters, scan, predictStream, fillPredictStream, meanAbsoluteError } from "./index";

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

// --- Test M1 : mae() retourne l'erreur absolue moyenne ---
// Données parfaitement expliquées par le modèle → prédiction == vérité → MAE ≈ 0
const maePerfect = reg.mae(students);
console.log("M1 — mae() sur données de train (≈0):", maePerfect < 1);

// MAE manuel sur une cible connue : erreurs [3, 1] → (3+1)/2 = 2
const maeHelper = meanAbsoluteError([10, 20], [7, 21]);
console.log("M1 — meanAbsoluteError([10,20],[7,21]) == 2:", maeHelper === 2);

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

// --- Test H1 : HDBSCAN détecte 2 clusters + bruit ---
const pts = [
  { x: 0.0, y: 0.0 },
  { x: 0.1, y: 0.1 },
  { x: 0.2, y: 0.0 },
  { x: -0.1, y: 0.2 },
  { x: 10.0, y: 10.0 },
  { x: 10.1, y: 10.2 },
  { x: 9.9, y: 9.8 },
  { x: 10.2, y: 9.9 },
  { x: 5.0, y: 5.0 }, // point isolé → bruit (-1)
];
const hdb = new clusters.HDBSCAN({ min_cluster_size: 3 });
const hdbLabels = hdb.fit_predict(pts, { features: ["x", "y"] });
const hdbDistinct = new Set(hdbLabels.filter((l) => l !== -1)).size;
console.log(
  "H1 — HDBSCAN: labels + 2 clusters détectés:",
  hdbLabels.length === 9 && hdbDistinct === 2,
);
console.log(
  "H1 — HDBSCAN: point isolé = probabilité ~0:",
  (hdb.probabilities[8] ?? 1) < 0.1,
);
console.log(
  "H1 — HDBSCAN probabilities alignées:",
  hdb.probabilities.length === 9 &&
    hdb.probabilities.every((p) => p >= 0 && p <= 1),
);
console.log(
  "H1 — HDBSCAN labels_ == fit_predict:",
  hdb.labels_.every((l, i) => l === hdbLabels[i]),
);

// --- Test G1 : Getis-Ord Gi* détecte le hotspot d'un coin ---
const grid: Record<string, unknown>[] = [];
for (let x = 0; x < 4; x++) {
  for (let y = 0; y < 4; y++) {
    grid.push({
      zone: `z${x}${y}`,
      lon: x,
      lat: y,
      cases: x === 0 && y === 0 ? 40 : 2, // coin (0,0) anormalement chaud
    });
  }
}
const gi = new scan.GetisOrd({ distance: 1 });
const giSpec = {
  zone: "zone",
  coordinates: ["lon", "lat"] as [string, string],
  cases: "cases",
};
gi.fit(grid, giSpec);
const giHots = gi.hotspots(grid);
const corner = giHots.find((r) => r.zone === "z00")!;
console.log(
  "G1 — GetisOrd: coin significatif (z>0, p≤0.05):",
  corner.hot && corner.zScore > 0 && corner.pValue <= 0.05,
);
console.log(
  "G1 — GetisOrd: seules les zones du hotspot sont flaguées:",
  giHots.filter((r) => r.zone !== "z00").every((r) => !r.hot),
);
console.log(
  "G1 — GetisOrd: predict binaire (1 seul hotspot):",
  gi.predict(grid).join("") === "1" + "0".repeat(15),
);
const giUniform = grid.map((r) => ({ ...r, cases: 2 }));
console.log(
  "G1 — GetisOrd: données uniformes → aucun hotspot:",
  gi.hotspots(giUniform).every((r) => !r.hot),
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
