import { LinearRegression } from "./algorithm/linear/LinearRegression";
import { LogisticRegression } from "./algorithm/linear/LogisticRegression";
import { RidgeRegression } from "./algorithm/linear/RidgeRegression";
import { LassoRegression } from "./algorithm/linear/LassoRegression";
import { ElasticNet } from "./algorithm/linear/ElasticNet";
import { RidgeClassifier } from "./algorithm/linear/RidgeClassifier";
import { RANSACRegressor } from "./algorithm/linear/RANSACRegressor";
import { PoissonRegressor } from "./algorithm/linear/PoissonRegressor";
import { PolynomialRegression } from "./algorithm/linear/PolynomialRegression";
import { KMeans } from "./algorithm/clusters/KMeans";
import { DBSCAN } from "./algorithm/clusters/DBSCAN";
import { DecisionTreeClassifier } from "./algorithm/tree/DecisionTreeClassifier";
import { DecisionTreeRegressor } from "./algorithm/tree/DecisionTreeRegressor";
import { ExtraTreeClassifier } from "./algorithm/tree/ExtraTreeClassifier";
import { ExtraTreeRegressor } from "./algorithm/tree/ExtraTreeRegressor";
import { IsolationForest } from "./algorithm/ensemble/IsolationForest";
import { RandomForestClassifier } from "./algorithm/ensemble/RandomForestClassifier";
import { RandomForestRegressor } from "./algorithm/ensemble/RandomForestRegressor";
import { AdaBoostClassifier } from "./algorithm/ensemble/AdaBoostClassifier";
import { AdaBoostRegressor } from "./algorithm/ensemble/AdaBoostRegressor";
import { GradientBoostingClassifier } from "./algorithm/ensemble/GradientBoostingClassifier";
import { GradientBoostingRegressor } from "./algorithm/ensemble/GradientBoostingRegressor";
import { XGBoostClassifier } from "./algorithm/ensemble/XGBoostClassifier";
import { XGBoostRegressor } from "./algorithm/ensemble/XGBoostRegressor";
import { CUSUM } from "./algorithm/monitoring/CUSUM";
import { EWMA } from "./algorithm/monitoring/EWMA";
import { ParallelMonitor } from "./algorithm/monitoring/ParallelMonitor";
import { SeasonalMonitor } from "./algorithm/monitoring/SeasonalMonitor";
import { SpatialScan } from "./algorithm/scan/SpatialScan";

// Namespaced access (kml-style): new linear.LinearRegression(), new clusters.KMeans(), ...
import * as linear from "./algorithm/linear";
import * as clusters from "./algorithm/clusters";
import * as tree from "./algorithm/tree";
import * as ensemble from "./algorithm/ensemble";
import * as monitoring from "./algorithm/monitoring";
import * as scan from "./algorithm/scan";

export {
  LinearRegression,
  LogisticRegression,
  RidgeRegression,
  LassoRegression,
  ElasticNet,
  RidgeClassifier,
  RANSACRegressor,
  PoissonRegressor,
  PolynomialRegression,
  KMeans,
  DBSCAN,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  ExtraTreeClassifier,
  ExtraTreeRegressor,
  IsolationForest,
  RandomForestClassifier,
  RandomForestRegressor,
  AdaBoostClassifier,
  AdaBoostRegressor,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  XGBoostClassifier,
  XGBoostRegressor,
  CUSUM,
  EWMA,
  ParallelMonitor,
  SeasonalMonitor,
  SpatialScan,
};

export { linear, clusters, tree, ensemble, monitoring, scan };

export { trainTestSplit, crossValScore, predictStream, fillPredictStream } from "./algorithm/evaluation";
export type {
  TrainTestSplitOptions,
  CrossValOptions,
  StreamOptions,
} from "./algorithm/evaluation";
export type { LinearParams } from "./algorithm/linear/LinearRegression";
export type { LogisticParams } from "./algorithm/linear/LogisticRegression";
export type { RidgeParams } from "./algorithm/linear/RidgeRegression";
export type { LassoParams } from "./algorithm/linear/LassoRegression";
export type { ElasticNetParams } from "./algorithm/linear/ElasticNet";
export type { RidgeClassifierParams } from "./algorithm/linear/RidgeClassifier";
export type { RANSACParams } from "./algorithm/linear/RANSACRegressor";
export type { PoissonParams } from "./algorithm/linear/PoissonRegressor";
export type { PolynomialParams } from "./algorithm/linear/PolynomialRegression";
export type { KMeansParams } from "./algorithm/clusters/KMeans";
export type { DBSCANParams } from "./algorithm/clusters/DBSCAN";
export type { DecisionTreeClassifierParams } from "./algorithm/tree/DecisionTreeClassifier";
export type { DecisionTreeRegressorParams } from "./algorithm/tree/DecisionTreeRegressor";
export type { ExtraTreeClassifierParams } from "./algorithm/tree/ExtraTreeClassifier";
export type { ExtraTreeRegressorParams } from "./algorithm/tree/ExtraTreeRegressor";
export type { IsolationForestParams } from "./algorithm/ensemble/IsolationForest";
export type { RandomForestClassifierParams } from "./algorithm/ensemble/RandomForestClassifier";
export type { RandomForestRegressorParams } from "./algorithm/ensemble/RandomForestRegressor";
export type { AdaBoostClassifierParams } from "./algorithm/ensemble/AdaBoostClassifier";
export type { AdaBoostRegressorParams } from "./algorithm/ensemble/AdaBoostRegressor";
export type { GradientBoostingClassifierParams } from "./algorithm/ensemble/GradientBoostingClassifier";
export type { GradientBoostingRegressorParams } from "./algorithm/ensemble/GradientBoostingRegressor";
export type { XGBoostClassifierParams } from "./algorithm/ensemble/XGBoostClassifier";
export type { XGBoostRegressorParams } from "./algorithm/ensemble/XGBoostRegressor";
export type { CUSUMParams } from "./algorithm/monitoring/CUSUM";
export type { EWMAParams } from "./algorithm/monitoring/EWMA";
export type { ParallelMonitorParams } from "./algorithm/monitoring/ParallelMonitor";
export type { SeasonalMonitorParams } from "./algorithm/monitoring/SeasonalMonitor";
export type { SpatialScanParams } from "./algorithm/scan/SpatialScan";
export type {
  JsonFitSpec,
  ClusterSpec,
  MonitorSpec,
  ParallelSpec,
  SeasonalSpec,
  ScanSpec,
  ScanCluster,
  JsonRow,
  JsonTransformOptions,
  JsonTransformResult,
} from "./@types/json";
