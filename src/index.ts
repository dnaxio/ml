import { LinearRegression } from "./linear/LinearRegression";
import { LogisticRegression } from "./linear/LogisticRegression";
import { RidgeRegression } from "./linear/RidgeRegression";
import { LassoRegression } from "./linear/LassoRegression";
import { ElasticNet } from "./linear/ElasticNet";
import { RidgeClassifier } from "./linear/RidgeClassifier";
import { RANSACRegressor } from "./linear/RANSACRegressor";
import { PoissonRegressor } from "./linear/PoissonRegressor";
import { PolynomialRegression } from "./linear/PolynomialRegression";
import { KMeans } from "./clusters/KMeans";
import { DBSCAN } from "./clusters/DBSCAN";
import { HDBSCAN } from "./clusters/HDBSCAN";
import { DecisionTreeClassifier } from "./tree/DecisionTreeClassifier";
import { DecisionTreeRegressor } from "./tree/DecisionTreeRegressor";
import { ExtraTreeClassifier } from "./tree/ExtraTreeClassifier";
import { ExtraTreeRegressor } from "./tree/ExtraTreeRegressor";
import { IsolationForest } from "./ensemble/IsolationForest";
import { RandomForestClassifier } from "./ensemble/RandomForestClassifier";
import { RandomForestRegressor } from "./ensemble/RandomForestRegressor";
import { AdaBoostClassifier } from "./ensemble/AdaBoostClassifier";
import { AdaBoostRegressor } from "./ensemble/AdaBoostRegressor";
import { GradientBoostingClassifier } from "./ensemble/GradientBoostingClassifier";
import { GradientBoostingRegressor } from "./ensemble/GradientBoostingRegressor";
import { XGBoostClassifier } from "./ensemble/XGBoostClassifier";
import { XGBoostRegressor } from "./ensemble/XGBoostRegressor";
import { CUSUM } from "./monitoring/CUSUM";
import { EWMA } from "./monitoring/EWMA";
import { ParallelMonitor } from "./monitoring/ParallelMonitor";
import { SeasonalMonitor } from "./monitoring/SeasonalMonitor";
import { SpatialScan } from "./scan/SpatialScan";
import { GetisOrd } from "./scan/GetisOrd";

// Namespaced access (kml-style): new linear.LinearRegression(), new clusters.KMeans(), ...
import * as linear from "./linear";
import * as clusters from "./clusters";
import * as tree from "./tree";
import * as ensemble from "./ensemble";
import * as monitoring from "./monitoring";
import * as scan from "./scan";

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
  HDBSCAN,
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
  GetisOrd,
};

export { linear, clusters, tree, ensemble, monitoring, scan };

export { trainTestSplit, crossValScore, predictStream, fillPredictStream, meanAbsoluteError } from "./evaluation";
export type {
  TrainTestSplitOptions,
  CrossValOptions,
  StreamOptions,
} from "./evaluation";
export type { LinearParams } from "./linear/LinearRegression";
export type { LogisticParams } from "./linear/LogisticRegression";
export type { RidgeParams } from "./linear/RidgeRegression";
export type { LassoParams } from "./linear/LassoRegression";
export type { ElasticNetParams } from "./linear/ElasticNet";
export type { RidgeClassifierParams } from "./linear/RidgeClassifier";
export type { RANSACParams } from "./linear/RANSACRegressor";
export type { PoissonParams } from "./linear/PoissonRegressor";
export type { PolynomialParams } from "./linear/PolynomialRegression";
export type { KMeansParams } from "./clusters/KMeans";
export type { DBSCANParams } from "./clusters/DBSCAN";
export type { HDBSCANParams } from "./clusters/HDBSCAN";
export type { DecisionTreeClassifierParams } from "./tree/DecisionTreeClassifier";
export type { DecisionTreeRegressorParams } from "./tree/DecisionTreeRegressor";
export type { ExtraTreeClassifierParams } from "./tree/ExtraTreeClassifier";
export type { ExtraTreeRegressorParams } from "./tree/ExtraTreeRegressor";
export type { IsolationForestParams } from "./ensemble/IsolationForest";
export type { RandomForestClassifierParams } from "./ensemble/RandomForestClassifier";
export type { RandomForestRegressorParams } from "./ensemble/RandomForestRegressor";
export type { AdaBoostClassifierParams } from "./ensemble/AdaBoostClassifier";
export type { AdaBoostRegressorParams } from "./ensemble/AdaBoostRegressor";
export type { GradientBoostingClassifierParams } from "./ensemble/GradientBoostingClassifier";
export type { GradientBoostingRegressorParams } from "./ensemble/GradientBoostingRegressor";
export type { XGBoostClassifierParams } from "./ensemble/XGBoostClassifier";
export type { XGBoostRegressorParams } from "./ensemble/XGBoostRegressor";
export type { CUSUMParams } from "./monitoring/CUSUM";
export type { EWMAParams } from "./monitoring/EWMA";
export type { ParallelMonitorParams } from "./monitoring/ParallelMonitor";
export type { SeasonalMonitorParams } from "./monitoring/SeasonalMonitor";
export type { SpatialScanParams } from "./scan/SpatialScan";
export type { GetisOrdParams, HotspotResult } from "./scan/GetisOrd";
export type {
  JsonFitSpec,
  ClusterSpec,
  MonitorSpec,
  ParallelSpec,
  SeasonalSpec,
  ScanSpec,
  ScanCluster,
  HotspotSpec,
  JsonRow,
  JsonTransformOptions,
  JsonTransformResult,
} from "./@types/json";
