/**
 * THE single import point for the @kanaries/ml engine.
 *
 * Every model accesses kml through this facade: swapping the engine (or
 * vendoring the code) only requires rewriting this file (and `core/state.ts`
 * if the internal shapes change) — the 25 wrappers never touch the package.
 */
import * as kml from "@kanaries/ml";

export { kml };
