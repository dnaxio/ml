import { linear, tree, monitoring, clusters, scan, neighbors, featureSelection, evaluation, predictStream, fillPredictStream, meanAbsoluteError, fbetaFromPrecisionRecall, compareModels, detectTask, rmse, mape, medianAbsoluteError, mcc, balancedAccuracy, logLoss, prAucScore, rocCurve, optimalThreshold } from "./index";

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

// --- Test N1 : KNeighborsClassifier classifie par voisinage ---
const knnData = [
  { age: 20, solde: 100, achete: false },
  { age: 21, solde: 150, achete: false },
  { age: 22, solde: 120, achete: false },
  { age: 35, solde: 5000, achete: true },
  { age: 36, solde: 5200, achete: true },
  { age: 37, solde: 4800, achete: true },
  { age: 55, solde: 9000, achete: true },
  { age: 30, solde: 3000, achete: false },
];
const knn = new neighbors.KNeighborsClassifier({ kNeighbors: 3 });
knn.fit(knnData, { features: ["age", "solde"], target: "achete" });
const knnPreds = knn.predict([
  { age: 21, solde: 130 },  // proche des non-acheteurs → 0
  { age: 36, solde: 5100 }, // proche des acheteurs → 1
]);
console.log(
  "N1 — KNN classifier prédit par voisinage:",
  knnPreds[0] === 0 && knnPreds[1] === 1,
);
console.log(
  "N1 — KNN classifier score + report:",
  knn.score(knnData) >= 0.5 &&
    knn.classificationReport(knnData).accuracy === knn.score(knnData),
);
console.log(
  "N1 — KNN classifier fill_predict (boolean):",
  knn.fill_predict([{ age: 36, solde: 5100 }])[0]!.achete === true,
);

// --- Test N2 : KNeighborsRegressor moyenne les voisins ---
const knnRegData = [
  { x: 1, y: 2 },
  { x: 2, y: 4 },
  { x: 3, y: 6 },
  { x: 4, y: 8 },
  { x: 5, y: 10 },
  { x: 6, y: 12 },
];
const knnReg = new neighbors.KNeighborsRegressor({ nNeighbors: 2 });
knnReg.fit(knnRegData, { features: ["x"], target: "y" });
const knnRegPreds = knnReg.predict([{ x: 2.1 }, { x: 5.9 }]);
// k=2 uniform : x=2.1 → moyenne de y(2)=4 et y(3)=6 = 5 ; x=5.9 → (12+10)/2 = 11
console.log(
  "N2 — KNN regressor moyenne les voisins (≈ y = 2x):",
  Math.abs(knnRegPreds[0]! - 5) < 0.5 && Math.abs(knnRegPreds[1]! - 11) < 0.5,
);
console.log(
  "N2 — KNN regressor mae/mse cohérents:",
  knnReg.mae(knnRegData) > 0 &&
    knnReg.mse(knnRegData) > 0 &&
    knnReg.score(knnRegData) > 0.9,
);

// --- Test N3 : alias sklearn (nNeighbors/metric/weights sur le classifieur) ---
const knnAlias = new neighbors.KNeighborsClassifier({
  nNeighbors: 3,     // alias de kNeighbors
  metric: "euclidean", // alias de distanceType
  weights: "distance", // alias de weightType
});
knnAlias.fit(knnData, { features: ["age", "solde"], target: "achete" });
console.log(
  "N3 — alias sklearn sur le classifieur:",
  knnAlias.predict([{ age: 21, solde: 130 }])[0] === 0,
);
// kml naming sur le régresseur (kNeighbors/distanceType/weightType) :
const knnRegAlias = new neighbors.KNeighborsRegressor({
  kNeighbors: 2,      // alias de nNeighbors
  distanceType: "euclidean", // alias de metric
  weightType: "uniform",     // alias de weights
});
knnRegAlias.fit(knnRegData, { features: ["x"], target: "y" });
console.log(
  "N3 — alias kml sur le régresseur:",
  Math.abs(knnRegAlias.predict([{ x: 3.1 }])[0]! - 7) < 0.5,
);

// --- Test N4 : getParams / setParams (sklearn-style) ---
const g = knn.getParams();
console.log(
  "N4 — getParams retourne kNeighbors:",
  g.kNeighbors === 3 && typeof g.distanceType === "string",
);
knn.setParams({ nNeighbors: 1 }); // setParams rebuilds unfitted
let threw = false;
try {
  knn.predict(knnData);
} catch {
  threw = true;
}
console.log("N4 — setParams force le refit (predict → erreur):", threw);
knn.fit(knnData, { features: ["age", "solde"], target: "achete" });
console.log(
  "N4 — refit après setParams:",
  knn.predict([{ age: 21, solde: 130 }])[0] === 0,
);

// --- Test N5 : classes getter (surface classifieur alignée SDK) ---
console.log(
  "N5 — KNN classifier classes:",
  knn.classes.length === 2 && knn.classes[0] === 0 && knn.classes[1] === 1,
);
const knnNoFit = new neighbors.KNeighborsClassifier();
let classesThrew = false;
try {
  knnNoFit.classes;
} catch {
  classesThrew = true;
}
console.log("N5 — classes avant fit → erreur:", classesThrew);

// --- Test B1 : fbeta — beta pèse le rappel (F2 > F1 quand recall > precision) ---
// precision 0.5 / recall 0.8 → F1 = 0.615, F2 = 0.714, F0.5 = 0.556
const f1 = fbetaFromPrecisionRecall(0.5, 0.8, 1);
const f2 = fbetaFromPrecisionRecall(0.5, 0.8, 2);
const f05 = fbetaFromPrecisionRecall(0.5, 0.8, 0.5);
console.log(
  "B1 — fbeta(0.5,0.8): F1<F2 et F0.5<F1:",
  Math.abs(f1 - 0.61538) < 0.001 &&
    Math.abs(f2 - 0.71429) < 0.001 &&
    Math.abs(f05 - 0.54054) < 0.001 &&
    f05 < f1 && f1 < f2,
);

// classificationReport(beta) sur un classifieur réel
const ordonnancesB = [
  { age: 24, quantite: 60, jour: 3, fraude: false },
  { age: 34, quantite: 30, jour: 5, fraude: false },
  { age: 67, quantite: 420, jour: 1, fraude: true },
  { age: 45, quantite: 120, jour: 4, fraude: false },
  { age: 58, quantite: 360, jour: 2, fraude: true },
  { age: 29, quantite: 60, jour: 6, fraude: false },
  { age: 71, quantite: 480, jour: 1, fraude: true },
  { age: 40, quantite: 90, jour: 5, fraude: false },
  { age: 33, quantite: 150, jour: 4, fraude: false },
  { age: 62, quantite: 300, jour: 2, fraude: true },
  { age: 36, quantite: 60, jour: 3, fraude: true },
  { age: 60, quantite: 270, jour: 2, fraude: false },
];
const clfB = new linear.LogisticRegression();
clfB.fit(ordonnancesB, { features: ["age", "quantite", "jour"], target: "fraude" });
const repF1 = clfB.classificationReport(ordonnancesB);
const repF10 = clfB.classificationReport(ordonnancesB, 10);
console.log(
  "B1 — classificationReport(beta) applique Fβ:",
  repF10.fScore > repF1.fScore === repF10.recall > repF1.precision &&
    typeof repF1.fScore === "number",
);
console.log(
  "B1 — beta=1 == ancien F1:",
  Math.abs(repF1.fScore - 2 * ((repF1.precision * repF1.recall) / (repF1.precision + repF1.recall))) < 1e-9,
);

// --- Test E2 : namespace evaluation (evaluation.mcc, evaluation.rmse, ...) ---
console.log(
  "E2 — evaluation.mcc / rmse / compareModels:",
  evaluation.mcc([0, 1], [0, 1]) === 1 &&
    Math.abs(evaluation.rmse([10, 20], [7, 21]) - Math.sqrt(5)) < 1e-9 &&
    typeof evaluation.compareModels === "function" &&
    typeof evaluation.detectTask === "function",
);

// --- Test E1 : métriques expertes (régression + classification) ---
// rmse : prédictions [10,20] vs vérité [7,21] → mse 5 → rmse √5 ≈ 2.236
console.log(
  "E1 — rmse ≈ √5:",
  Math.abs(rmse([10, 20], [7, 21]) - Math.sqrt(5)) < 1e-9,
);
// mape : (|1|/1 + |4|/8)/2 = (1 + 0.5)/2 = 0.75
console.log(
  "E1 — mape = 0.75:",
  Math.abs(mape([2, 4], [1, 8]) - 0.75) < 1e-9,
);
// medianAbsoluteError : erreurs [3, 1, 1] → médiane 1
console.log(
  "E1 — medianAbsoluteError = 1:",
  medianAbsoluteError([10, 20, 30], [7, 21, 29]) === 1,
);
// mcc : parfait = 1, inversé = -1
console.log(
  "E1 — mcc: parfait 1 / inversé -1:",
  mcc([0, 1, 1, 0], [0, 1, 1, 0]) === 1 && mcc([1, 0, 0, 1], [0, 1, 1, 0]) === -1,
);
// balancedAccuracy : prédictions parfaites → 1
console.log(
  "E1 — balancedAccuracy parfait = 1:",
  balancedAccuracy([0, 1, 1, 0], [0, 1, 1, 0]) === 1,
);
// logLoss : vérité [0,1] proba [0.01, 0.99] → -ln(0.99) ≈ 0.01005
console.log(
  "E1 — logLoss ≈ 0.01005:",
  Math.abs(logLoss([0, 1], [0.01, 0.99]) - 0.010050335853501449) < 1e-9,
);
// prAucScore : séparation parfaite → 1 ; aléatoire → faible
console.log(
  "E1 — prAucScore parfait = 1:",
  Math.abs(prAucScore([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1) < 1e-9,
);
// rocCurve : structure correcte (tpr croissant)
const curve = rocCurve([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]);
console.log(
  "E1 — rocCurve: fpr/tpr/thresholds alignés:",
  curve.fpr.length === curve.tpr.length &&
    curve.thresholds.length === curve.tpr.length &&
    curve.tpr[curve.tpr.length - 1]! === 1,
);
// optimalThreshold : entre 0.2 et 0.8 (Youden), classes bien séparées
const tOpt = optimalThreshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]);
console.log(
  "E1 — optimalThreshold (Youden) dans ]0.2, 0.8]:",
  tOpt > 0.2 && tOpt <= 0.8,
);

// --- Test C1 : compareModels classe les modèles (leaderboard) ---
const benchData = [
  { x: 1, y: 2.1 }, { x: 2, y: 3.9 }, { x: 3, y: 6.2 }, { x: 4, y: 7.8 },
  { x: 5, y: 10.3 }, { x: 6, y: 11.9 }, { x: 7, y: 14.2 }, { x: 8, y: 15.8 },
  { x: 9, y: 18.1 }, { x: 10, y: 19.7 },
];
const benchSpec = { features: ["x"], target: "y" };
const leaderboard = compareModels(
  {
    linear: () => new linear.LinearRegression(),
    tree: () => new linear.PolynomialRegression({ degree: 2 }),
  },
  benchData,
  benchSpec,
  { cv: 5, randomState: 42 },
);
console.log(
  "C1 — leaderboard trié desc (meilleur en premier):",
  leaderboard.length === 2 &&
    leaderboard[0]!.mean >= leaderboard[1]!.mean &&
    leaderboard[0]!.scores.length === 5 &&
    typeof leaderboard[0]!.std === "number",
);
console.log(
  "C1 — leaderboard mse trié asc (plus bas mieux):",
  compareModels(
    { linear: () => new linear.LinearRegression(), tree: () => new linear.PolynomialRegression({ degree: 2 }) },
    benchData,
    benchSpec,
    { cv: 5, scoring: "mse", randomState: 42 },
  )[0]!.mean <= 0.5,
);
console.log(
  "C1 — detectTask: régression vs classification:",
  detectTask(benchData, benchSpec) === "regression" &&
    detectTask(ordonnancesB, { features: ["age", "quantite", "jour"], target: "fraude" }) === "classification",
);
// leaderboard classification (accuracy) :
const clfBoard = compareModels(
  {
    logistic: () => new linear.LogisticRegression(),
    tree: () => new tree.DecisionTreeClassifier({ max_depth: 3 }),
  },
  ordonnancesB,
  { features: ["age", "quantite", "jour"], target: "fraude" },
  { cv: 3, randomState: 42 },
);
console.log(
  "C1 — leaderboard classification (accuracy):",
  clfBoard.length === 2 && clfBoard[0]!.mean >= clfBoard[1]!.mean,
);

// --- Test F1 : scores univariés classent le signal avant le bruit ---
// Classification target (fClassif / chi2 / mutualInfoClassif) :
const fsClfData = [
  { signal: 1, bruit: 0.12, achete: false },
  { signal: 2, bruit: 0.87, achete: false },
  { signal: 3, bruit: 0.43, achete: false },
  { signal: 4, bruit: 0.91, achete: false },
  { signal: 6, bruit: 0.25, achete: true },
  { signal: 7, bruit: 0.64, achete: true },
  { signal: 8, bruit: 0.33, achete: true },
  { signal: 9, bruit: 0.77, achete: true },
];
const fsClfSpec = { features: ["signal", "bruit"], target: "achete" };
const fRank = featureSelection.fClassif(fsClfData, fsClfSpec);
console.log(
  "F1 — fClassif: signal classé avant bruit:",
  fRank[0]!.feature === "signal" && fRank[0]!.score > fRank[1]!.score,
);
const cRank = featureSelection.chi2(fsClfData, fsClfSpec);
console.log(
  "F1 — chi2: signal classé avant bruit:",
  cRank[0]!.feature === "signal" && typeof cRank[0]!.pValue === "number",
);
const miClfRank = featureSelection.mutualInfoClassif(fsClfData, fsClfSpec);
console.log(
  "F1 — mutualInfoClassif: signal classé avant bruit:",
  miClfRank[0]!.feature === "signal" && miClfRank[0]!.score > miClfRank[1]!.score,
);
// Regression target (mutualInfoRegression) :
const fsRegData = [
  { x: 1, signal: 10, bruit: 0.12, cible: 19 },
  { x: 2, signal: 21, bruit: 0.87, cible: 44 },
  { x: 3, signal: 29, bruit: 0.43, cible: 57 },
  { x: 4, signal: 41, bruit: 0.91, cible: 84 },
  { x: 5, signal: 49, bruit: 0.25, cible: 97 },
  { x: 6, signal: 62, bruit: 0.64, cible: 125 },
];
const fsRegSpec = { features: ["signal", "bruit"], target: "cible" };
const miRank = featureSelection.mutualInfoRegression(fsRegData, fsRegSpec);
console.log(
  "F1 — mutualInfoRegression: signal classé avant bruit:",
  miRank[0]!.feature === "signal" && miRank[0]!.score > miRank[1]!.score,
);

// --- Test F2 : SelectFromModel garde les features utiles (Lasso) ---
const fsSelData = [
  { age: 30, salaire: 38000, bruit: 7 },
  { age: 35, salaire: 46000, bruit: 2 },
  { age: 40, salaire: 55000, bruit: 9 },
  { age: 45, salaire: 62000, bruit: 4 },
  { age: 50, salaire: 71000, bruit: 6 },
  { age: 55, salaire: 79000, bruit: 3 },
  { age: 60, salaire: 88000, bruit: 8 },
  { age: 65, salaire: 95000, bruit: 5 },
];
const lasso = new linear.LassoRegression({ alpha: 1 });
const sel = new featureSelection.SelectFromModel({ estimator: lasso });
const fsSelSpec = { features: ["salaire", "bruit"], target: "age" };
sel.fit(fsSelData, fsSelSpec);
console.log(
  "F2 — SelectFromModel (Lasso): salaire gardé, bruit éliminé:",
  sel.selectedFeatures.includes("salaire") && !sel.selectedFeatures.includes("bruit"),
);
console.log(
  "F2 — support aligné + fittedEstimator:",
  sel.support.length === 2 && sel.featureScores.length === 2 && sel.fittedEstimator === lasso,
);
const selTop1 = new featureSelection.SelectFromModel({
  estimator: lasso,
  threshold: 0,
  maxFeatures: 1,
});
selTop1.fit(fsSelData, fsSelSpec);
console.log(
  "F2 — maxFeatures: 1 → top-1 par importance:",
  selTop1.selectedFeatures.length === 1 && selTop1.selectedFeatures[0] === "salaire",
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
