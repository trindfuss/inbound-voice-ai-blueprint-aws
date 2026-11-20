// compareRows: the field-by-field rows of the conflict panel. Example tests from the
// real 409 shapes, then a property over generated bodies with {value, source} wrappers.
import assert from "node:assert/strict";
import { test } from "node:test";

import { compareRows, isWrapper } from "../lib/compare.js";
import { forAll, gen, int, json, oneOf, record, string } from "./gen.js";

// A tool's 409 `current` (config_routes.py `_tool_view`): mask.after_ms and handoff_wait_ms are
// {value, source} wrappers; `yours` is the PUT body tools.js sends ({rev, ...body}).
const TOOL_CURRENT = {
  name: "transfer_funds", kind: "write", min_level: 2, phrase_key: "tool.transfer_funds",
  confirm: "always", confirm_source: "tool", rev: 7,
  mask: {
    phrase_key: "mask.transfer",
    after_ms: { value: 800, source: "tool" },
    handoff_wait_ms: { value: 3000, source: "default" },
    pool: [{ key: "mask.transfer.1", texts: { "en-US": "One moment.", "es-US": "Un momento." } }],
  },
  problems: [],
};

test("tool conflict: rows for each leaf of yours, wrappers unwrapped, rev skipped", () => {
  const yours = { rev: 6, min_level: 3, confirm: "always", mask: { phrase_key: "mask.transfer", after_ms: 1200 } };
  assert.deepEqual(compareRows(TOOL_CURRENT, yours), [
    { path: "min_level", current: 2, yours: 3, changed: true, found: true },
    { path: "confirm", current: "always", yours: "always", changed: false, found: true },
    { path: "mask.phrase_key", current: "mask.transfer", yours: "mask.transfer", changed: false, found: true },
    { path: "mask.after_ms", current: 800, yours: 1200, changed: true, found: true },
  ]);
});

test("a wrapped value equal to yours is not changed", () => {
  const rows = compareRows(TOOL_CURRENT, { rev: 1, mask: { after_ms: 800, handoff_wait_ms: 3000 } });
  assert.deepEqual(rows.map((r) => [r.path, r.changed]), [["mask.after_ms", false], ["mask.handoff_wait_ms", false]]);
});

test("a path missing from current is changed, with current undefined and found false", () => {
  const rows = compareRows(TOOL_CURRENT, { rev: 7, mask: null, extra: { deep: 1 } });
  assert.deepEqual(rows, [
    { path: "mask", current: { ...TOOL_CURRENT.mask, after_ms: 800, handoff_wait_ms: 3000 }, yours: null, changed: true, found: true },
    { path: "extra.deep", current: undefined, yours: 1, changed: true, found: false },
  ]);
});

test("masking conflict: yours.masking against current.masking", () => {
  const current = { masking: { tool_after_ms: 900, handoff_wait_ms: 2500 }, effective: {}, rev: 3 };
  const rows = compareRows(current, { rev: 2, masking: { tool_after_ms: 900, handoff_wait_ms: 4000 } });
  assert.deepEqual(rows.map((r) => [r.path, r.current, r.yours, r.changed]), [
    ["masking.tool_after_ms", 900, 900, false],
    ["masking.handoff_wait_ms", 2500, 4000, true],
  ]);
});

test("a restore body matches no path of current, so the caller falls back to the JSON blocks", () => {
  const yours = { audit_pk: "audit#tool#transfer_funds", audit_sk: "2026-10-01T00:00:00Z#abc", rev: 7 };
  const rows = compareRows(TOOL_CURRENT, yours);
  assert.deepEqual(rows.map((r) => r.path), ["audit_pk", "audit_sk"]);
  assert.ok(rows.every((r) => !r.found && r.changed));
});

test("rev is skipped only at the top level", () => {
  assert.deepEqual(compareRows({ rev: 9 }, { rev: 1 }), []);
  assert.deepEqual(compareRows({ a: { rev: 1 } }, { rev: 1, a: { rev: 2 } }).map((r) => r.path), ["a.rev"]);
  assert.deepEqual(compareRows({}, { rev: { a: 1 }, b: 2 }).map((r) => r.path), ["b"]); // its whole subtree
});

test("arrays and empty objects are leaves; arrays compare as a whole", () => {
  const rows = compareRows({ list: [1, 2], o: {} }, { list: [1, 2], o: {}, l2: [] });
  assert.deepEqual(rows.map((r) => [r.path, r.changed, r.found]),
    [["list", false, true], ["o", false, true], ["l2", true, false]]);
});

test("unwrapping is deep: nested wrappers and wrappers inside a whole-object leaf", () => {
  const rows = compareRows(TOOL_CURRENT, { mask: null });
  assert.equal(rows[0].current.after_ms, 800);
  assert.equal(rows[0].current.handoff_wait_ms, 3000);
  assert.deepEqual(compareRows({ a: { value: { value: 1, source: "tool" }, source: "tool" } }, { a: 1 })[0],
    { path: "a", current: 1, yours: 1, changed: false, found: true });
});

test("lookup goes through a wrapper in current, and never through an array or the prototype", () => {
  assert.equal(compareRows({ a: { value: { b: 1 }, source: "tool" } }, { a: { b: 1 } })[0].changed, false);
  assert.equal(compareRows({ a: [5] }, { a: { 0: 5 } })[0].found, false);
  const proto = compareRows({}, { constructor: 1, toString: { x: 1 } });
  assert.ok(proto.every((r) => !r.found && r.current === undefined));
});

test("isWrapper: exactly the keys value and source, with a string source", () => {
  assert.equal(isWrapper({ value: 1, source: "tool" }), true);
  assert.equal(isWrapper({ value: null, source: "default" }), true);
  assert.equal(isWrapper({ value: 1, source: 2 }), false);
  assert.equal(isWrapper({ value: 1, source: "tool", x: 1 }), false);
  assert.equal(isWrapper({ value: 1 }), false);
  assert.equal(isWrapper([1, "tool"]), false);
  assert.equal(isWrapper(null), false);
});

test("non-object inputs give no rows or found-false rows, and never throw", () => {
  for (const bad of [null, undefined, 1, "x", true, [1, 2]]) {
    assert.deepEqual(compareRows({ a: 1 }, bad), []);
    assert.deepEqual(compareRows(bad, { a: 1 }), [{ path: "a", current: undefined, yours: 1, changed: true, found: false }]);
  }
  const big = { n: 10n };
  assert.doesNotThrow(() => compareRows(big, big)); // not JSON, but still no throw
});

// ---------------------------------------------------------------- property

const SOURCE = oneOf("tool", "default");
const KEY = string({ minLength: 1, maxLength: 4, chars: "abcxyzrev" }); // collides often, sometimes "rev"

/** JSON-like values with {value, source} wrappers mixed in at any depth. */
function wrapped(d) {
  const scalar = json({ maxDepth: 0 });
  return gen((rng) => {
    const k = d === 0 ? int(0, 1)(rng) : int(0, 4)(rng);
    if (k === 0) return scalar(rng);
    if (k === 1) return { value: d === 0 ? scalar(rng) : wrapped(d - 1)(rng), source: SOURCE(rng) };
    if (k === 2) return json({ maxDepth: 1, maxLength: 3 })(rng); // arrays and plain objects
    const n = int(0, 4)(rng);
    const o = {};
    for (let i = 0; i < n; i++) o[KEY(rng)] = wrapped(d - 1)(rng);
    return o;
  });
}

const BODY = gen((rng) => {
  const n = int(0, 5)(rng);
  const o = {};
  for (let i = 0; i < n; i++) o[KEY(rng)] = wrapped(2)(rng);
  return o;
});

// The oracle: a direct reading of the conflict-comparison rule, written separately from lib/compare.js.
const plain = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const wrap = (v) => plain(v) && Object.keys(v).length === 2 && "value" in v && typeof v.source === "string";
const unwrap = (v) => {
  if (wrap(v)) return unwrap(v.value);
  if (Array.isArray(v)) return v.map(unwrap);
  return plain(v) ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, unwrap(x)])) : v;
};
function leaves(v, prefix, out) {
  for (const k of Object.keys(v)) {
    const p = [...prefix, k];
    const x = v[k];
    if (plain(x) && !wrap(x) && Object.keys(x).length > 0) leaves(x, p, out);
    else out.push(p);
  }
  return out;
}
function lookup(root, path) {
  let cur = root;
  for (const k of path) {
    while (wrap(cur)) cur = cur.value;
    if (!plain(cur) || !Object.hasOwn(cur, k)) return { found: false, value: undefined };
    cur = cur[k];
  }
  return { found: true, value: unwrap(cur) };
}

/** Derive current from yours: keep, change, drop, wrap or unwrap each entry, so paths match. */
function derive(v, rng, { valueSame = false } = {}) {
  if (!plain(v) || wrap(v)) return v;
  const out = {};
  for (const [k, x] of Object.entries(v)) {
    const op = int(0, valueSame ? 2 : 4)(rng);
    if (op === 0) out[k] = derive(x, rng, { valueSame });
    else if (op === 1) out[k] = wrap(x) ? x.value : { value: derive(x, rng, { valueSame }), source: SOURCE(rng) };
    else if (op === 2) out[k] = derive(x, rng, { valueSame });
    else if (op === 3) out[k] = wrapped(1)(rng);
    // op 4: dropped
  }
  return out;
}

const CASE = gen((rng) => {
  const yours = BODY(rng);
  const mode = int(0, 2)(rng);
  const current = mode === 0 ? BODY(rng) : mode === 1 ? derive(yours, rng) : oneOf(null, 3, "s", [1])(rng);
  return { yours, current };
});

test("rows are the leaf paths of yours minus top-level rev; changed iff unwrapped values differ", () => {
  forAll(CASE, ({ yours, current }) => {
    const rows = compareRows(current, yours);
    const expected = leaves(yours, [], []).filter((p) => p[0] !== "rev"); // the whole top-level rev
    assert.deepEqual(rows.map((r) => r.path), expected.map((p) => p.join(".")));
    rows.forEach((r, i) => {
      const yv = unwrap(expected[i].reduce((o, k) => o[k], yours));
      const { found, value } = lookup(current, expected[i]);
      assert.deepEqual(r.yours, yv);
      assert.equal(r.found, found);
      assert.deepEqual(r.current, value);
      assert.equal(r.changed, JSON.stringify(value) !== JSON.stringify(yv));
    });
  }, { runs: 300 });
});

test("compareRows(x, x) and wrap-only differences have no changed rows", () => {
  forAll(BODY, (x) => compareRows(x, x).every((r) => !r.changed && r.found));
  forAll(gen((rng) => {
    const yours = BODY(rng);
    return { yours, current: derive(yours, rng, { valueSame: true }) };
  }), ({ yours, current }) => compareRows(current, yours).every((r) => !r.changed));
});

test("never throws, for any JSON-like pair", () => {
  const ANY = record({ a: oneOf(BODY, json(), wrapped(2)), b: oneOf(BODY, json(), wrapped(2)) });
  forAll(ANY, ({ a, b }) => Array.isArray(compareRows(a, b)), { runs: 300 });
});
