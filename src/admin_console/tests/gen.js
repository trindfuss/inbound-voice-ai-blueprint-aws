// Seeded generators and forAll for the property-based tests. No dependency (so no fast-check)
// and no shrinking: generators stay small (depth <= 3, arrays <= 6) so a failing
// case is readable as printed, and its seed replays it: OWL_SEED=<seed> node --test <file>.
//
// A generator is a function rng => value, made with gen() so oneOf can tell it from a constant.
import { inspect } from "node:util";

const GEN = Symbol("gen");

/** Mark fn (rng => value) as a generator. */
export function gen(fn) {
  fn[GEN] = true;
  return fn;
}

export const isGen = (x) => typeof x === "function" && x[GEN] === true;

/** The mulberry32 PRNG: seed (a 32-bit integer) => () => a float in [0, 1). */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** An integer in [min, max], both inclusive. */
export function int(min, max) {
  if (!Number.isSafeInteger(min) || !Number.isSafeInteger(max) || min > max) {
    throw new RangeError(`int(${min}, ${max}): need integers with min <= max`);
  }
  return gen((rng) => (Math.floor(rng() * (max - min + 1)) + min) || 0);
}

/** One of the options, uniformly. An option that is a generator is run; anything else is returned as is. */
export function oneOf(...options) {
  if (options.length === 0) throw new RangeError("oneOf() needs at least one option");
  const pick = int(0, options.length - 1);
  return gen((rng) => {
    const o = options[pick(rng)];
    return isGen(o) ? o(rng) : o;
  });
}

export const ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

function lengthGen(minLength, maxLength) {
  if (minLength > maxLength) throw new RangeError(`length: min ${minLength} > max ${maxLength}`);
  return int(minLength, maxLength);
}

/** A string of minLength..maxLength characters, each from `chars` (default ALNUM). */
export function string({ minLength = 0, maxLength = 12, chars = ALNUM } = {}) {
  const alphabet = [...chars];
  const len = lengthGen(minLength, maxLength);
  const ch = int(0, alphabet.length - 1);
  return gen((rng) => Array.from({ length: len(rng) }, () => alphabet[ch(rng)]).join(""));
}

// Code point ranges for unicodeString: controls, ASCII, Latin, combining marks, Hebrew and Arabic
// (right to left), invisible and line-separator characters, CJK, emoji (astral) and the rest of
// the astral planes. No surrogate code points, so every string is well formed.
const RANGES = [
  [0x00, 0x1F], [0x7F, 0x7F], [0x20, 0x7E], [0x20, 0x7E], [0xA0, 0x24F], [0x300, 0x36F],
  [0x590, 0x6FF], [0x200B, 0x200F], [0x2028, 0x2029], [0xFEFF, 0xFEFF], [0x4E00, 0x9FFF],
  [0x1F300, 0x1FAFF], [0x10000, 0x10FFFF],
];

/** A string of minLength..maxLength code points from all over Unicode (always well formed). */
export function unicodeString({ minLength = 0, maxLength = 12 } = {}) {
  const len = lengthGen(minLength, maxLength);
  const range = int(0, RANGES.length - 1);
  return gen((rng) => {
    let out = "";
    for (let n = len(rng); n > 0; n--) {
      const [lo, hi] = RANGES[range(rng)];
      out += String.fromCodePoint(int(lo, hi)(rng));
    }
    return out;
  });
}

/** An array of minLength..maxLength values from g. */
export function array(g, { minLength = 0, maxLength = 6 } = {}) {
  const len = lengthGen(minLength, maxLength);
  return gen((rng) => Array.from({ length: len(rng) }, () => g(rng)));
}

/** An object with a value from each generator in shape; keys in `optional` are left out half the time. */
export function record(shape, { optional = [] } = {}) {
  const coin = int(0, 1);
  return gen((rng) => {
    const out = {};
    for (const [k, g] of Object.entries(shape)) {
      if (optional.includes(k) && coin(rng) === 0) continue;
      out[k] = isGen(g) ? g(rng) : g;
    }
    return out;
  });
}

/**
 * A JSON value: null, a boolean, a finite number (never -0), a string, or an array or object
 * of at most maxLength entries, nested at most maxDepth levels. JSON.parse(JSON.stringify(v))
 * deep-equals v.
 */
export function json({ maxDepth = 3, maxLength = 6 } = {}) {
  const scalar = oneOf(null, true, false, int(-1000, 1000),
    gen((rng) => int(-100000, 100000)(rng) / 100 || 0), unicodeString({ maxLength: 8 }), string());
  const key = string({ minLength: 1, maxLength: 8 });
  const kind = int(0, 3);
  const at = (d) => gen((rng) => {
    const k = d === 0 ? 0 : kind(rng);
    if (k <= 1) return scalar(rng);
    const items = array(at(d - 1), { maxLength })(rng);
    return k === 2 ? items : Object.fromEntries(items.map((v) => [key(rng), v]));
  });
  return at(maxDepth);
}

/** Nesting depth of a JSON-like value: 0 for a scalar, 1 + the deepest child for a container. */
export function depth(v) {
  if (v === null || typeof v !== "object") return 0;
  return 1 + Math.max(0, ...Object.values(v).map(depth));
}

function resolveSeed(seed) {
  if (seed !== undefined) return seed >>> 0;
  const env = process.env.OWL_SEED;
  if (env !== undefined && env !== "") {
    const n = Number(env);
    if (!Number.isInteger(n) || n < 0 || n > 0xFFFFFFFF) {
      throw new RangeError(`OWL_SEED must be an integer in 0..4294967295, got ${JSON.stringify(env)}`);
    }
    return n;
  }
  return Math.floor(Math.random() * 0x100000000);
}

/**
 * Check prop on `runs` values from g, drawn from one PRNG seeded with `seed` (else OWL_SEED,
 * else random). The property fails when it throws or returns false; forAll then throws an Error
 * naming the seed, the run and the case, with .seed, .run, .case and .cause. If prop returns a
 * promise, forAll returns one too (await it).
 */
export function forAll(g, prop, { runs = 200, seed } = {}) {
  if (typeof g !== "function") throw new TypeError("forAll: the first argument must be a generator");
  const s = resolveSeed(seed);
  const rng = mulberry32(s);
  const fail = (run, cause) => {
    const replay = mulberry32(s);
    let value;
    for (let k = 0; k <= run; k++) value = g(replay); // the case as generated, before prop saw it
    const why = cause instanceof Error ? cause : new Error("the property returned false");
    const err = new Error(`Property failed on run ${run + 1} of ${runs} (seed ${s}). `
      + `Replay: OWL_SEED=${s} node --test <file>. case: ${inspect(value, { depth: 8, breakLength: Infinity })}. `
      + `${why.message}`, { cause: why });
    return Object.assign(err, { seed: s, run, case: value });
  };
  const check = (run, result) => { if (result === false) throw fail(run); };
  for (let run = 0; run < runs; run++) {
    let result;
    try {
      result = prop(g(rng));
    } catch (e) {
      throw fail(run, e);
    }
    if (result && typeof result.then === "function") {
      return (async () => {
        for (let r = run; r < runs; r++) {
          let out;
          try {
            out = await (r === run ? result : prop(g(rng)));
          } catch (e) {
            throw fail(r, e);
          }
          check(r, out);
        }
      })();
    }
    check(run, result);
  }
  return undefined;
}
