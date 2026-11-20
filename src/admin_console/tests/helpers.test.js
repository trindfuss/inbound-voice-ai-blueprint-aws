// The test helpers themselves: the fake DOM, the seeded generators and forAll. They
// are the harness for the property tests, so no property applies here; the PBT checks that
// every generator stays within its declared bounds.
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { h, field, announce } from "../lib/dom.js";
import { confirmDialog, save } from "../lib/ui.js";
import { createStorage, install, uninstall } from "./fakedom.js";
import {
  ALNUM, array, depth, forAll, int, isGen, json, mulberry32, oneOf, record, string, unicodeString,
} from "./gen.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const GLOBALS = ["document", "window", "Node", "Element", "Text", "Event", "CustomEvent", "CSS",
  "localStorage", "sessionStorage"];

/** Run fn with the fake DOM installed, and always uninstall. */
function withDom(fn, opts) {
  const env = install(opts);
  try {
    return fn(env);
  } finally {
    uninstall();
  }
}

// ----------------------------------------------------------------: no dependencies

test("package.json has no dependencies and the folder has no lockfile", () => {
  const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8"));
  for (const k of ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]) {
    assert.equal(k in pkg, false, `package.json must not have ${k}`);
  }
  for (const lock of ["package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb"]) {
    assert.equal(existsSync(join(ROOT, lock)), false, `${lock} must not exist`);
  }
  assert.equal(existsSync(join(ROOT, "node_modules")), false, "no node_modules");
});

test("the helpers import only node: built-ins and each other", () => {
  for (const name of readdirSync(join(ROOT, "tests")).filter((n) => n.endsWith(".js"))) {
    const text = readFileSync(join(ROOT, "tests", name), "utf8");
    for (const m of text.matchAll(/^import .* from "([^"]+)";$/gm)) {
      assert.match(m[1], /^(node:|\.{1,2}\/)/, `${name}: ${m[1]}`);
    }
  }
});

// ---------------------------------------------------------------- fakedom: install

test("install() sets the globals and uninstall() leaves none behind", () => {
  const before = Object.fromEntries(GLOBALS.map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  const names = Object.getOwnPropertyNames(globalThis).sort();
  const env = install();
  assert.equal(globalThis.document, env.document);
  assert.equal(globalThis.window.document, env.document);
  assert.equal(globalThis.localStorage, env.localStorage);
  assert.equal(globalThis.sessionStorage, env.sessionStorage);
  assert.equal(typeof globalThis.CSS.escape, "function");
  assert.ok(env.document.createElement("p") instanceof globalThis.Node);
  assert.throws(() => install(), /already installed/);
  uninstall();
  uninstall(); // idempotent
  for (const k of GLOBALS) {
    assert.deepEqual(Object.getOwnPropertyDescriptor(globalThis, k), before[k], `global ${k} restored`);
  }
  assert.deepEqual(Object.getOwnPropertyNames(globalThis).sort(), names);
  assert.equal(typeof globalThis.document, "undefined");
});

test("install() can share a storage across installs (a reload keeps localStorage)", () => {
  const storage = createStorage();
  withDom(() => localStorage.setItem("k", "v"), { localStorage: storage });
  withDom(() => assert.equal(localStorage.getItem("k"), "v"), { localStorage: storage });
  withDom(() => assert.equal(localStorage.getItem("k"), null));
});

// ---------------------------------------------------------------- fakedom: tree

test("elements, text, attributes, id, classList, value and checked", () => withDom(({ document }) => {
  const p = document.createElement("P");
  assert.equal(p.tagName, "P");
  assert.equal(p.localName, "p");
  p.id = "intro";
  assert.equal(p.getAttribute("id"), "intro");
  p.setAttribute("data-x", "1");
  assert.equal(p.hasAttribute("data-x"), true);
  assert.equal(p.dataset.x, "1");
  p.dataset.fooBar = "baz";
  assert.equal(p.getAttribute("data-foo-bar"), "baz");
  p.removeAttribute("data-x");
  assert.equal(p.getAttribute("data-x"), null);
  assert.deepEqual(p.attributes.map((a) => a.name), ["id", "data-foo-bar"]);

  p.classList.add("a", "b");
  p.classList.add("a");
  assert.equal(p.className, "a b");
  assert.equal(p.getAttribute("class"), "a b");
  assert.equal(p.classList.contains("b"), true);
  p.classList.remove("a");
  assert.equal(p.classList.toggle("c"), true);
  assert.equal(p.classList.toggle("c"), false);
  assert.deepEqual([...p.classList], ["b"]);
  assert.equal(p.classList.length, 1);
  p.className = "x  y";
  assert.deepEqual([...p.classList], ["x", "y"]);

  p.append("Hello ", document.createElement("b"), document.createTextNode("world"));
  p.children[0].textContent = "big";
  assert.equal(p.textContent, "Hello bigworld");
  assert.equal(p.childNodes.length, 3);
  assert.equal(p.children.length, 1);
  assert.equal(p.firstChild.nodeType, 3);
  assert.equal(p.children[0].parentNode, p);
  p.textContent = "plain";
  assert.equal(p.childNodes.length, 1);
  assert.equal(p.textContent, "plain");

  const ul = document.createElement("ul");
  const li = document.createElement("li");
  ul.append(li);
  p.replaceChildren(li, "t");
  assert.equal(ul.children.length, 0, "append moves a node");
  assert.equal(li.parentNode, p);
  li.remove();
  assert.equal(p.childNodes.length, 1);
  p.replaceChildren();
  assert.equal(p.childNodes.length, 0);

  const input = document.createElement("input");
  assert.equal(input.value, "");
  input.setAttribute("value", "start");
  assert.equal(input.value, "start");
  input.value = 42;
  assert.equal(input.value, "42");
  assert.equal(input.getAttribute("value"), "start", "the value property is not the attribute");
  assert.equal(input.checked, false);
  input.checked = 1;
  assert.equal(input.checked, true);
  input.hidden = true;
  assert.equal(input.getAttribute("hidden"), "");
  input.hidden = false;
  assert.equal(input.hasAttribute("hidden"), false);
}));

test("createElementNS with <use href>, and node checks", () => withDom(({ document }) => {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  const use = document.createElementNS(NS, "use");
  use.setAttribute("href", "icons.svg#tools");
  svg.classList.add("icon");
  svg.append(use);
  document.body.append(svg);
  assert.equal(svg.namespaceURI, NS);
  assert.equal(use.tagName, "use", "SVG keeps its case");
  assert.equal(document.querySelector('svg.icon > use[href="icons.svg#tools"]'), use);
  assert.ok(use instanceof Node && use instanceof Element);
  assert.ok(document.createTextNode("x") instanceof Text);
  assert.ok(document.body.contains(use));
}));

// ---------------------------------------------------------------- fakedom: selectors

function tree(document) {
  const main = document.createElement("main");
  main.id = "main";
  const header = document.createElement("header");
  header.className = "page-header";
  const h1 = document.createElement("h1");
  h1.id = "page-title";
  header.append(h1);
  const form = document.createElement("form");
  const a = document.createElement("input");
  a.setAttribute("data-field", "a.b");
  a.setAttribute("required", "");
  const b = document.createElement("input");
  b.setAttribute("data-field", 'we"ird');
  b.className = "wide";
  const sel = document.createElement("select");
  form.append(a, b, sel);
  main.append(header, form);
  document.body.append(main);
  return { main, header, h1, form, a, b, sel };
}

test("querySelector: tag, .class, #id, [attr], [attr=v], comma lists, combinators", () => withDom(({ document }) => {
  const { main, header, h1, form, a, b, sel } = tree(document);
  assert.equal(document.querySelector("header"), header);
  assert.equal(document.querySelector("HEADER"), header, "tag names are case-insensitive");
  assert.equal(document.querySelector(".page-header"), header);
  assert.equal(document.querySelector("#page-title"), h1);
  assert.equal(document.getElementById("page-title"), h1);
  assert.equal(document.querySelector("h1#page-title"), h1);
  assert.equal(document.querySelector("header.page-header"), header);
  assert.deepEqual(document.querySelectorAll("[data-field]"), [a, b]);
  assert.equal(document.querySelector("[required]"), a);
  assert.equal(document.querySelector('[data-field="a.b"]'), a);
  assert.equal(document.querySelector("[data-field='a.b']"), a);
  assert.equal(document.querySelector(`[data-field="${CSS.escape('we"ird')}"]`), b);
  assert.equal(document.querySelector('input.wide[data-field="we\\"ird"]'), b);
  assert.equal(document.querySelector('[data-field="nope"]'), null);
  assert.deepEqual(document.querySelectorAll("select, input"), [a, b, sel], "document order");
  assert.deepEqual(document.querySelectorAll("main input"), [a, b]);
  assert.deepEqual(document.querySelectorAll("main > input"), []);
  assert.deepEqual(document.querySelectorAll("main > form > input.wide"), [b]);
  assert.equal(main.querySelector("main"), null, "querySelector searches descendants only");
  assert.deepEqual(form.querySelectorAll("*"), [a, b, sel]);
  assert.equal(b.matches("form .wide"), true);
  assert.equal(b.closest("main"), main);
  assert.equal(b.closest(".nope"), null);
  assert.throws(() => document.querySelector("a:hover"), SyntaxError);
  assert.throws(() => document.querySelector("[x"), SyntaxError);
  assert.throws(() => document.querySelector(""), SyntaxError);
}));

test("CSS.escape follows the CSSOM algorithm", () => withDom(() => {
  assert.equal(CSS.escape("a.b"), "a\\.b");
  assert.equal(CSS.escape('we"ird'), 'we\\"ird');
  assert.equal(CSS.escape("1x"), "\\31 x");
  assert.equal(CSS.escape("-"), "\\-");
  assert.equal(CSS.escape("-1"), "-\\31 ");
  assert.equal(CSS.escape("\0"), "\uFFFD");
  assert.equal(CSS.escape("a\u007Fb"), "a\\7f b");
  assert.equal(CSS.escape("é_-z9"), "é_-z9");
}));

// ---------------------------------------------------------------- fakedom: events and focus

test("events: listeners, bubbling, once, stopPropagation, preventDefault", () => withDom(({ document, window }) => {
  const outer = document.createElement("div");
  const inner = document.createElement("button");
  outer.append(inner);
  document.body.append(outer);
  const seen = [];
  outer.addEventListener("click", (e) => seen.push(["outer", e.target === inner, e.currentTarget === outer]));
  inner.addEventListener("click", () => seen.push(["inner-once"]), { once: true });
  inner.dispatchEvent(new Event("click", { bubbles: true }));
  inner.dispatchEvent(new Event("click", { bubbles: true }));
  assert.deepEqual(seen, [["inner-once"], ["outer", true, true], ["outer", true, true]]);

  seen.length = 0;
  inner.dispatchEvent(new Event("click"));
  assert.deepEqual(seen, [], "not bubbling by default");

  const stop = (e) => e.stopPropagation();
  inner.addEventListener("click", stop);
  inner.dispatchEvent(new Event("click", { bubbles: true }));
  assert.deepEqual(seen, []);
  inner.removeEventListener("click", stop);

  const form = document.createElement("form");
  form.addEventListener("submit", (e) => e.preventDefault());
  const ev = new Event("submit", { cancelable: true });
  assert.equal(form.dispatchEvent(ev), false);
  assert.equal(ev.defaultPrevented, true);

  const custom = [];
  window.addEventListener("hashchange", (e) => custom.push(e.detail));
  window.dispatchEvent(new CustomEvent("hashchange", { detail: 7 }));
  assert.deepEqual(custom, [7]);
}));

test("focus sets document.activeElement and fires focus and blur", () => withDom(({ document }) => {
  assert.equal(document.activeElement, document.body);
  const a = document.createElement("input");
  const b = document.createElement("button");
  document.body.append(a, b);
  const log = [];
  a.addEventListener("focus", () => log.push("a focus"));
  a.addEventListener("blur", () => log.push("a blur"));
  a.focus();
  assert.equal(document.activeElement, a);
  b.focus();
  assert.equal(document.activeElement, b);
  b.blur();
  assert.equal(document.activeElement, document.body);
  assert.deepEqual(log, ["a focus", "a blur"]);
}));

test("dialog: showModal and close, with the close event", () => withDom(({ document }) => {
  const dlg = document.createElement("dialog");
  document.body.append(dlg);
  const events = [];
  dlg.addEventListener("close", () => events.push("close"));
  assert.equal(dlg.open, false);
  dlg.showModal();
  assert.equal(dlg.open, true);
  assert.equal(dlg.hasAttribute("open"), true);
  assert.throws(() => dlg.showModal(), /already open/);
  dlg.close("ok");
  assert.equal(dlg.open, false);
  assert.equal(dlg.returnValue, "ok");
  dlg.close();
  assert.deepEqual(events, ["close"], "closing a closed dialog fires nothing");
}));

test("the console's own lib/dom.js and lib/ui.js run under fakedom", async () => {
  const { document } = install();
  try {
    const live = h("div#live");
    const dlg = h("dialog#confirm");
    document.body.append(live, dlg);
    const { wrap, input } = field("Name", h("input", { "data-field": "name", value: "x" }), "A hint");
    const form = h("form", { on: { submit: (e) => e.preventDefault() } }, wrap);
    document.body.append(form);
    assert.equal(document.querySelector(`label[for="${input.id}"]`).textContent, "Name");
    assert.equal(input.getAttribute("aria-describedby"), `${input.id}-hint`);
    assert.equal(input.value, "x");
    assert.throws(() => h("p", { style: "color: red" }), /CSP/);
    announce("hello");
    assert.equal(live.textContent, "hello");

    const answer = confirmDialog({ title: "Sure?", message: "Really", confirmLabel: "Yes" });
    assert.equal(dlg.open, true);
    assert.equal(document.activeElement.textContent, "Cancel");
    dlg.querySelector("button.primary").click();
    assert.equal(await answer, true);
    assert.equal(dlg.open, false);

    const status = h("div");
    const err = Object.assign(new Error("Bad name"), { kind: "invalid", field: "name", title: "Invalid" });
    assert.equal(await save(status, async () => { throw err; }, { form }), null);
    assert.equal(input.getAttribute("aria-invalid"), "true");
    assert.equal(document.activeElement, input);
    // A 400 naming a form field marks that field and leaves the status empty.
    assert.equal(status.childNodes.length, 0);
    assert.equal(form.querySelector(".field-error").textContent, "Bad name");
  } finally {
    uninstall();
  }
});

// ---------------------------------------------------------------- fakedom: storage

test("storage: the Storage API, and a switch that makes every call throw", () => withDom(({ localStorage, sessionStorage }) => {
  assert.notEqual(localStorage, sessionStorage);
  localStorage.setItem("a", 1);
  localStorage.setItem("b", "two");
  assert.equal(localStorage.getItem("a"), "1");
  assert.equal(localStorage.length, 2);
  assert.equal(localStorage.key(0), "a");
  assert.equal(localStorage.key(5), null);
  assert.deepEqual(localStorage.keys(), ["a", "b"]);
  localStorage.removeItem("a");
  assert.equal(localStorage.getItem("a"), null);
  assert.equal(sessionStorage.length, 0);

  localStorage.throws = true;
  for (const call of [() => localStorage.getItem("b"), () => localStorage.setItem("c", "x"),
    () => localStorage.removeItem("b"), () => localStorage.clear(), () => localStorage.key(0)]) {
    assert.throws(call, (e) => e.name === "SecurityError");
  }
  localStorage.throws = false;
  assert.equal(localStorage.getItem("b"), "two", "a failed call changes nothing");
  localStorage.clear();
  assert.equal(localStorage.length, 0);
}));

test("storage: install can make the accessor itself throw", () => withDom(() => {
  assert.throws(() => globalThis.localStorage, (e) => e.name === "SecurityError");
  assert.equal(sessionStorage.getItem("x"), null);
}, { storageThrows: "localStorage" }));

// ---------------------------------------------------------------- gen and forAll

test("mulberry32 is deterministic, in [0, 1), and seeds differ", () => {
  const a = mulberry32(1);
  const b = mulberry32(1);
  const c = mulberry32(2);
  const xs = Array.from({ length: 50 }, () => a());
  assert.deepEqual(xs, Array.from({ length: 50 }, () => b()));
  assert.notDeepEqual(xs, Array.from({ length: 50 }, () => c()));
  for (const x of xs) assert.ok(x >= 0 && x < 1);
});

test("forAll replays the same cases for the same seed", () => {
  const g = record({ n: int(0, 1000), s: string(), xs: array(json()) });
  const run = (seed) => {
    const cases = [];
    forAll(g, (v) => { cases.push(v); }, { runs: 30, seed });
    return cases;
  };
  assert.equal(run(7).length, 30);
  assert.deepEqual(run(7), run(7));
  assert.notDeepEqual(run(7), run(8));
});

test("forAll reads OWL_SEED when no seed is given; an explicit seed wins", () => {
  const saved = process.env.OWL_SEED;
  const first = (opts) => {
    let got;
    forAll(int(0, 1e9), (v) => { got ??= v; }, { runs: 1, ...opts });
    return got;
  };
  try {
    process.env.OWL_SEED = "12345";
    assert.equal(first(), first({ seed: 12345 }));
    assert.notEqual(first({ seed: 1 }), first({ seed: 12345 }));
    process.env.OWL_SEED = "not a number";
    assert.throws(() => first(), /OWL_SEED/);
  } finally {
    if (saved === undefined) delete process.env.OWL_SEED;
    else process.env.OWL_SEED = saved;
  }
});

test("a planted failing property reports its seed and the failing case", () => {
  let err;
  try {
    forAll(int(0, 100), (x) => assert.ok(x < 50, "too big"), { seed: 42 });
  } catch (e) {
    err = e;
  }
  assert.ok(err, "the property must fail");
  assert.equal(err.seed, 42);
  assert.ok(err.case >= 50);
  assert.match(err.message, /seed 42\b/);
  assert.match(err.message, /OWL_SEED=42/);
  assert.match(err.message, new RegExp(`case: ${err.case}\\b`));
  assert.match(err.message, /too big/);
  assert.ok(err.cause instanceof Error);
  // The same seed fails on the same run with the same case.
  assert.throws(() => forAll(int(0, 100), (x) => assert.ok(x < 50), { seed: 42 }),
    (e) => e.run === err.run && e.case === err.case);
  // A returned false also fails.
  assert.throws(() => forAll(int(0, 9), () => false, { seed: 1 }), /seed 1\b/);
});

test("forAll supports async properties", async () => {
  let n = 0;
  await forAll(int(0, 9), async () => { n++; }, { runs: 20, seed: 3 });
  assert.equal(n, 20);
  await assert.rejects(forAll(int(0, 9), async (x) => assert.ok(x < 5), { seed: 3 }), /seed 3\b/);
});

test("oneOf picks generators or constants", () => {
  const g = oneOf("a", int(10, 12), null);
  const seen = new Set();
  const rng = mulberry32(9);
  for (let i = 0; i < 200; i++) seen.add(g(rng));
  assert.deepEqual([...seen].sort(), [10, 11, 12, "a", null].sort());
  assert.ok(isGen(g));
  assert.equal(isGen(() => 1), false);
  assert.throws(() => oneOf(), RangeError);
  assert.throws(() => int(3, 2), RangeError);
});

// ---------------------------------------------------------------- PBT: generator bounds

const isScalarValue = (cp) => cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);

function jsonBounded(v, maxLength) {
  if (v === null || typeof v === "boolean" || typeof v === "string") return true;
  if (typeof v === "number") return Number.isFinite(v) && !Object.is(v, -0);
  if (Array.isArray(v)) return v.length <= maxLength && v.every((x) => jsonBounded(x, maxLength));
  if (typeof v === "object" && Object.getPrototypeOf(v) === Object.prototype) {
    const vals = Object.values(v);
    return vals.length <= maxLength && vals.every((x) => jsonBounded(x, maxLength));
  }
  return false;
}

// For 200 seeds: random bounds, and each generator's output within them.
const BOUNDS = record({
  seed: int(0, 2 ** 32 - 1),
  lo: int(-1000, 1000),
  span: int(0, 1000),
  minLen: int(0, 3),
  extra: int(0, 8),
  maxDepth: int(0, 3),
});

test("PBT: every generator stays within its declared bounds (200 seeds)", () => {
  forAll(BOUNDS, ({ seed, lo, span, minLen, extra, maxDepth }) => {
    const rng = mulberry32(seed);
    const hi = lo + span;
    const maxLen = minLen + extra;

    const n = int(lo, hi)(rng);
    assert.ok(Number.isInteger(n) && n >= lo && n <= hi, `int ${n} in [${lo}, ${hi}]`);

    const s = string({ minLength: minLen, maxLength: maxLen })(rng);
    assert.ok(s.length >= minLen && s.length <= maxLen, `string length ${s.length}`);
    for (const ch of s) assert.ok(ALNUM.includes(ch), `string char ${JSON.stringify(ch)}`);

    const hex = string({ maxLength: maxLen, chars: "0123456789abcdef" })(rng);
    assert.match(hex, /^[0-9a-f]*$/);

    const u = unicodeString({ minLength: minLen, maxLength: maxLen })(rng);
    const cps = [...u].map((c) => c.codePointAt(0));
    assert.ok(cps.length >= minLen && cps.length <= maxLen, `unicodeString length ${cps.length}`);
    for (const cp of cps) assert.ok(isScalarValue(cp), `scalar value ${cp.toString(16)}`);
    assert.equal(u.isWellFormed(), true);
    assert.doesNotThrow(() => encodeURIComponent(u));

    const xs = array(int(0, 3), { minLength: minLen, maxLength: maxLen })(rng);
    assert.ok(xs.length >= minLen && xs.length <= maxLen, `array length ${xs.length}`);
    assert.ok(xs.every((x) => x >= 0 && x <= 3));

    const defaultArray = array(int(0, 1))(rng);
    assert.ok(defaultArray.length <= 6, "arrays default to at most 6");

    const r = record({ a: int(0, 1), b: string() }, { optional: ["b"] })(rng);
    assert.ok(Object.keys(r).every((k) => k === "a" || k === "b"));
    assert.ok("a" in r);
    assert.ok(r.b === undefined || typeof r.b === "string");

    const j = json({ maxDepth })(rng);
    assert.ok(depth(j) <= maxDepth, `json depth ${depth(j)} > ${maxDepth}`);
    assert.ok(jsonBounded(j, 6), "json values, at most 6 entries");
    assert.deepEqual(JSON.parse(JSON.stringify(j)), j, "json round-trips");
    assert.ok(depth(json()(rng)) <= 3, "json defaults to depth 3");

    const pick = oneOf("x", int(lo, hi))(rng);
    assert.ok(pick === "x" || (pick >= lo && pick <= hi));
  }, { runs: 200 });
});

test("depth counts nesting levels", () => {
  assert.equal(depth(1), 0);
  assert.equal(depth([]), 1);
  assert.equal(depth({ a: [1, { b: null }] }), 3);
});
