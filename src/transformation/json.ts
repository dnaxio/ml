import { kml, createScaler, loadModel, toJSONOf } from "../core";
import type {
  TransformerSpec,
  JsonRow,
  JsonTransformResult,
} from "../@types/json";

/**
 * Transforms JSON rows (row objects) into a feature matrix `number[][]`
 * and an output vector `number[]`, ready for @kanaries/ml models.
 *
 * Value encoding:
 * - number        → as-is
 * - boolean       → 1 / 0
 * - numeric string → Number()
 * - non-numeric string → one-hot if `oneHot: true`, otherwise an error
 * - undefined/null → depends on `missing` ('throw' | 'drop' | 'fill0')
 *
 * Fields not selected in `spec.features` / `spec.target` are ignored.
 *
 * Features are sorted alphabetically (and one-hot categories as well), so
 * permuting the `features` array has no effect on the column layout.
 */
export class JsonTransformer {
  private spec: TransformerSpec;
  /** Features sorted alphabetically → order-invariant layout. */
  private features: string[];
  /** field -> list of learned categories (one-hot, sorted alphabetically). */
  private categories = new Map<string, string[]>();
  /** rows removed by the 'drop' strategy in the last transform. */
  private lastDropped = 0;
  /** indices (in the input data) of the rows kept by the last transform. */
  private lastKeptIndices: number[] = [];
  /** Fitted scaler (when `options.scale` is enabled). */
  private scaler: kml.utils.Preprocessing.StandardScaler | null = null;
  /** Whether the target field is boolean in the training data. */
  private targetIsBoolean = false;

  constructor(spec: TransformerSpec) {
    if (spec.features.length === 0)
      throw new Error("JsonFitSpec.features must contain at least one field.");
    this.spec = spec;
    // Sort alphabetically so permuting the features array has no effect on the layout.
    this.features = [...spec.features].sort();
  }

  /**
   * Learns the categories (and fits the scaler when `scale` is enabled),
   * then builds X and Y. Use during training (fitTransform).
   */
  fitTransform(data: JsonRow[]): JsonTransformResult {
    this.learnCategories(data);
    this.learnTargetType(data);
    const result = this.build(data);
    if (this.scaleEnabled()) {
      this.scaler = createScaler();
      this.scaler.fit(result.X);
    }
    const X = this.applyScale(result).X;
    // Noise is a training-time data augmentation: it perturbs the features
    // used to fit the model, but `transform()` (inference) stays deterministic.
    return { X: this.addNoise(X), Y: result.Y };
  }

  /**
   * Builds X and Y reusing the learned categories (and the fitted scaler).
   * Use during inference (transform).
   */
  transform(data: JsonRow[]): JsonTransformResult {
    return this.applyScale(this.build(data));
  }

  /** Builds the raw X matrix and Y vector (before scaling). */
  private build(data: JsonRow[]): JsonTransformResult {
    const X: number[][] = [];
    const Y: number[] = [];
    const kept: number[] = [];
    let dropped = 0;

    for (const [i, row] of data.entries()) {
      const yField = this.spec.target;

      // With 'drop', skip the whole row if a required field is missing.
      if (this.isMissingStrategy() === "drop") {
        const missing = this.findMissingField(row);
        if (missing) {
          dropped++;
          continue;
        }
      }

      const xRow = this.features.flatMap((field) =>
        this.encodeField(row, field),
      );
      X.push(xRow);
      kept.push(i);

      // The target is optional (clustering has none).
      if (yField !== undefined) {
        const yValue = this.encodeOutput(row, yField);
        if (yValue !== undefined) Y.push(yValue);
      }
    }

    this.lastDropped = dropped;
    this.lastKeptIndices = kept;
    return { X, Y };
  }

  /** Number of rows removed by the 'drop' strategy in the last transform. */
  get droppedRows(): number {
    return this.lastDropped;
  }

  /** Indices (in the input data) of the rows kept by the last transform. */
  get keptIndices(): number[] {
    return this.lastKeptIndices;
  }

  /** The output (target) field name of the spec ("" for clustering). */
  get targetField(): string {
    return this.spec.target ?? "";
  }

  /** Whether the target field is boolean in the training data. */
  get targetBoolean(): boolean {
    return this.targetIsBoolean;
  }

  /** Learns whether the target field holds boolean values. */
  private learnTargetType(data: JsonRow[]): void {
    const y = this.spec.target;
    if (!y) {
      this.targetIsBoolean = false;
      return;
    }
    this.targetIsBoolean =
      data.length > 0 && data.every((row) => typeof row[y] === "boolean");
  }

  /** Applies the fitted scaler to X when present. */
  private applyScale(result: JsonTransformResult): JsonTransformResult {
    if (!this.scaler) return result;
    return { X: this.scaler.transform(result.X), Y: result.Y };
  }

  /**
   * Adds Gaussian noise (`options.noise`) to continuous feature columns.
   * One-hot columns (binary indicators) are left untouched so categorical
   * encodings stay valid. No-op when `noise` is absent, 0 or negative.
   */
  private addNoise(X: number[][]): number[][] {
    const std = this.spec.options?.noise;
    if (!std || std <= 0) return X;
    const continuous = this.continuousMask();
    const rng = mulberry32(
      this.spec.options?.noiseSeed ?? Math.floor(Math.random() * 2 ** 32),
    );
    return X.map((row) =>
      row.map((v, j) => (continuous[j] ? v + gaussian(rng) * std : v)),
    );
  }

  /**
   * Boolean mask aligned with the X columns: true = continuous column
   * (noise applies), false = one-hot column (kept intact).
   */
  private continuousMask(): boolean[] {
    const mask: boolean[] = [];
    for (const field of this.features) {
      const cats = this.categories.get(field);
      if (cats) {
        for (let i = 0; i < this.columnWidth(field); i++) mask.push(false);
      } else {
        mask.push(true);
      }
    }
    return mask;
  }

  /** Whether feature standardization is enabled. */
  private scaleEnabled(): boolean {
    return this.spec.options?.scale ?? false;
  }

  /** X column names (useful to interpret the coefficients). */
  get columnNames(): string[] {
    const names: string[] = [];
    for (const field of this.features) {
      const cats = this.categories.get(field);
      if (cats) {
        const drop = this.dropFirst();
        const cols = drop ? cats.slice(1) : cats;
        for (const c of cols) names.push(`${field}_${c}`);
      } else {
        names.push(field);
      }
    }
    return names;
  }

  private learnCategories(data: JsonRow[]): void {
    for (const field of this.features) {
      const seen = new Set<string>();
      for (const row of data) {
        const v = row[field];
        if (typeof v === "string" && !isNumeric(v)) {
          seen.add(v);
        }
      }
      // Sort categories alphabetically → stable reference & column order.
      if (seen.size > 0) this.categories.set(field, [...seen].sort());
    }
  }

  /** Returns the missing strategy ('throw' | 'drop' | 'fill0'). */
  private isMissingStrategy(): "throw" | "drop" | "fill0" {
    return this.spec.options?.missing ?? "throw";
  }

  /** Returns the first missing required field, or undefined if none. */
  private findMissingField(row: JsonRow): string | undefined {
    for (const field of this.features) {
      if (row[field] === undefined || row[field] === null) return field;
    }
    const y = this.spec.target;
    if (y && (row[y] === undefined || row[y] === null)) return y;
    return undefined;
  }

  /** Encodes the output Y. Absent/undefined → undefined (not required at inference). */
  private encodeOutput(row: JsonRow, field: string): number | undefined {
    const v = row[field];
    if (v === undefined || v === null) return undefined;
    if (typeof v === "number") return v;
    if (typeof v === "boolean") return v ? 1 : 0;
    if (typeof v === "string") {
      if (isNumeric(v)) return Number(v);
      return undefined; // categorical output not supported for regression
    }
    return undefined;
  }

  /** Encodes a field into a list of numbers (1 value, or n for one-hot). */
  private encodeField(row: JsonRow, field: string): number[] {
    const v = row[field];

    if (v === undefined || v === null) {
      const missing = this.isMissingStrategy();
      if (missing === "throw") {
        throw new Error(
          `Field "${field}" is missing in the row ${JSON.stringify(row)}.`,
        );
      }
      // 'fill0' => fill with 0 (the row is kept)
      return Array(this.columnWidth(field)).fill(0);
    }

    if (typeof v === "number") return [v];
    if (typeof v === "boolean") return [v ? 1 : 0];

    if (typeof v === "string") {
      if (isNumeric(v)) return [Number(v)];
      // categorical → one-hot (one column per learned category, minus the reference if dropFirst)
      if (!this.spec.options?.oneHot) {
        throw new Error(
          `Categorical value "${v}" for field "${field}" while one-hot is not enabled (options.oneHot: true expected).`,
        );
      }
      const cats = this.categories.get(field) ?? [];
      if (cats.length === 0) {
        throw new Error(
          `Categorical value "${v}" for field "${field}": no category learned.`,
        );
      }
      const drop = this.dropFirst();
      const cols = drop ? cats.slice(1) : cats;
      return cols.map((c) => (c === v ? 1 : 0));
    }

    throw new Error(
      `Unsupported type for field "${field}": ${typeof v} (${JSON.stringify(v)}).`,
    );
  }

  /** Width (number of columns) of a field: 1 or the number of categories. */
  private columnWidth(field: string): number {
    const cats = this.categories.get(field);
    if (!cats) return 1;
    return this.dropFirst() ? cats.length - 1 : cats.length;
  }

  /** Should the first category be dropped in one-hot? (default true) */
  private dropFirst(): boolean {
    return this.spec.options?.dropFirst ?? true;
  }

  /** Serializes the transformer (spec + categories + scaler) for persistence. */
  toJSON(): JsonTransformerState {
    return {
      spec: this.spec,
      categories: Object.fromEntries(this.categories),
      scaler: this.scaler ? (toJSONOf(this.scaler) as SerializedScaler) : null,
      targetBoolean: this.targetIsBoolean,
    };
  }

  /** Revives a transformer serialized by `toJSON()`. */
  static fromJSON(state: JsonTransformerState): JsonTransformer {
    const t = new JsonTransformer(state.spec);
    t.categories = new Map(Object.entries(state.categories ?? {}));
    if (state.scaler) {
      t.scaler = loadModel<kml.utils.Preprocessing.StandardScaler>(
        state.scaler,
      );
    }
    if (state.targetBoolean !== undefined)
      t.targetIsBoolean = state.targetBoolean;
    return t;
  }
}

/** Serialized shape of a JsonTransformer (see `toJSON`). */
export interface JsonTransformerState {
  spec: TransformerSpec;
  categories: Record<string, string[]>;
  /** Serialized StandardScaler (from its `toJSON()`) when `scale` is enabled. */
  scaler?: SerializedScaler | null;
  /** Whether the target field is boolean (restored on load). */
  targetBoolean?: boolean;
}

/** Serialized shape of the kml StandardScaler. */
type SerializedScaler = ReturnType<
  InstanceType<typeof kml.utils.Preprocessing.StandardScaler>["toJSON"]
>;

function isNumeric(s: string): boolean {
  return s.trim() !== "" && !Number.isNaN(Number(s));
}

/** mulberry32 — tiny seeded PRNG returning uniform values in [0, 1). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller transform: standard normal (mean 0, std 1) from a uniform PRNG. */
function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
