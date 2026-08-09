import { kml } from "./kml";

/**
 * Centralized access to the kml internal state we read/write. The fragile
 * `as unknown as ...` casts live here only — if kml's internal shapes
 * change, this is the single file to adapt.
 */

/** Loose shape of the internal kml state we touch. */
interface KmState {
  coefState?: number[];
  intercept?: number;
  fitted?: boolean;
  classes?: number[];
  toJSON?: () => unknown;
}

/** Coefficients of a fitted linear model (kml stores them in `coefState`). */
export function coefOf(model: unknown): number[] {
  return (model as KmState).coefState ?? [];
}

/** Intercept of a fitted linear model. */
export function interceptOf(model: unknown): number {
  return (model as KmState).intercept ?? 0;
}

/** Writes coefficients (used to restore a linear model from parameters). */
export function setCoef(model: unknown, coef: number[]): void {
  (model as KmState).coefState = coef;
}

/** Writes the intercept (used to restore a linear model from parameters). */
export function setIntercept(model: unknown, intercept: number): void {
  (model as KmState).intercept = intercept;
}

/** Marks the kml model as fitted (kml guards its own predict on this flag). */
export function setFitted(model: unknown, fitted = true): void {
  (model as KmState).fitted = fitted;
}

/** Whether the kml model is marked as fitted. */
export function fittedOf(model: unknown): boolean {
  return (model as KmState).fitted ?? false;
}

/** Class labels of a fitted classifier (semantics of predict_proba). */
export function classesOf(model: unknown): number[] {
  return (model as KmState).classes ?? [];
}

/** Reads internal configuration keys (for persistence, e.g. alpha/maxIter). */
export function configOf(
  model: unknown,
  keys: string[],
): Record<string, unknown> {
  const m = model as Record<string, unknown>;
  const out: Record<string, unknown> = {};
  for (const key of keys) out[key] = m[key];
  return out;
}

/** Official kml serializer. */
export function toJSONOf(model: unknown): unknown {
  return (model as KmState).toJSON?.();
}

/** Official kml deserializer. */
export function loadModel<T>(json: unknown): T {
  return kml.loadModel(json as Parameters<typeof kml.loadModel>[0]) as T;
}

/** StandardScaler factory (preprocessing used by the JSON transformer). */
export function createScaler(): InstanceType<
  typeof kml.utils.Preprocessing.StandardScaler
> {
  return new kml.utils.Preprocessing.StandardScaler();
}

/** Offloads a computation to a worker (kml asyncMode). */
export function asyncMode<P extends any[], R>(
  fn: (...args: P) => R,
): (...args: P) => Promise<R> {
  return kml.utils.asyncMode(fn);
}
