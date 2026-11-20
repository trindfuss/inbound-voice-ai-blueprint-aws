// The shared view checker. The view tests and the whole-console
// check render under fakedom with renderUnder() and then assert:
//   assertNoInlineStyle  no node has a style or on* attribute;
//   assertClassesStyled  every class used appears as a .class selector in a stylesheet;
//   assertViewStructure  page header and h1, labels, aria-describedby, tables in cards.
// Each assert throws an AssertionError naming every problem, so a planted fault fails.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { install, uninstall } from "./fakedom.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** The stylesheets a class may be styled in: the five console stylesheets. */
export const STYLE_SOURCES = ["tokens.css", "base.css", "layout.css", "components.css", "views.css"];

/** Every element in the tree, the root included. */
export function allElements(root) {
  return [root, ...root.querySelectorAll("*")];
}

// ---------------------------------------------------------------- renderUnder

/**
 * Install a fresh fakedom with the shell's containers (main#main labelled by page-title, as in
 * index.html; #toasts; dialog#confirm.dialog; #live), run `render(env)` (sync or
 * async), then `check(env)` while the DOM is still installed, and always uninstall.
 * env = {document, window, localStorage, sessionStorage, main, toasts, confirm, live}.
 * Returns what `check` returns (or what `render` returns when there is no check).
 */
export async function renderUnder(render, { check, install: opts } = {}) {
  const env = install(opts);
  try {
    const { document } = env;
    const el = (tag, attrs) => {
      const n = document.createElement(tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      return n;
    };
    env.main = el("main", { id: "main", tabindex: "-1", "aria-labelledby": "page-title" });
    env.toasts = el("div", { id: "toasts", class: "toasts" });
    env.confirm = el("dialog", { id: "confirm", class: "dialog" });
    env.live = el("div", { id: "live", class: "visually-hidden", role: "status", "aria-live": "polite" });
    document.body.append(env.main, env.toasts, env.confirm, env.live);
    const out = await render(env);
    return check ? await check(env, out) : out;
  } finally {
    uninstall();
  }
}

// ----------------------------------------------------------------

/** No element in the tree has a style attribute or an on* handler attribute. */
export function assertNoInlineStyle(root) {
  const problems = [];
  for (const el of allElements(root)) {
    for (const { name } of el.attributes) {
      if (name.toLowerCase() === "style" || /^on/i.test(name)) problems.push(`<${el.tagName.toLowerCase()}> has ${name}`);
    }
  }
  assert.deepEqual(problems, [], `inline style or handler attributes:\n${problems.join("\n")}`);
}

// ----------------------------------------------------------------

/** The class names in a stylesheet's selectors (comments, strings and declarations ignored). */
export function selectorClasses(css) {
  const text = css.replace(/\/\*[\s\S]*?\*\//g, "").replace(/"[^"]*"|'[^']*'/g, '""');
  const out = new Set();
  let prelude = "";
  const take = (sel) => {
    if (sel.trim().startsWith("@")) return; // @media and @keyframes preludes hold no classes
    for (const m of sel.matchAll(/\.(-?[_a-zA-Z][_a-zA-Z0-9-]*)/g)) out.add(m[1]);
  };
  // Only text outside declaration blocks is a selector. @media bodies hold rules, so track
  // which open braces belong to at-rules.
  const stack = [];
  for (const ch of text) {
    if (ch === "{") {
      const isAt = prelude.trim().startsWith("@");
      if (stack.every((s) => s === "at")) take(prelude);
      stack.push(isAt ? "at" : "rule");
      prelude = "";
    } else if (ch === "}") {
      stack.pop();
      prelude = "";
    } else if (stack.every((s) => s === "at")) {
      prelude += ch;
      if (ch === ";") prelude = ""; // e.g. a declaration-less at-statement
    }
  }
  return out;
}

const cache = new Map();

/** The set of classes styled by the given stylesheets (relative to the console folder). */
export function styledClasses(sources = STYLE_SOURCES) {
  const key = sources.join("\n");
  if (!cache.has(key)) {
    const set = new Set();
    for (const f of sources) for (const c of selectorClasses(readFileSync(join(ROOT, f), "utf8"))) set.add(c);
    cache.set(key, set);
  }
  return cache.get(key);
}

/** Every class used in the tree (from the class attribute, so SVG elements count too). */
export function usedClasses(root) {
  const out = new Set();
  for (const el of allElements(root)) {
    for (const c of (el.getAttribute("class") || "").split(/\s+/)) if (c) out.add(c);
  }
  return out;
}

/** Every class used in the tree is a .class selector in one of `sources`. */
export function assertClassesStyled(root, { sources = STYLE_SOURCES, styled } = {}) {
  const known = styled ?? styledClasses(sources);
  const missing = [...usedClasses(root)].filter((c) => !known.has(c)).sort();
  assert.deepEqual(missing, [], `classes with no selector in ${sources.join(", ")}: ${missing.join(", ")}`);
}

// ----------------------------------------------------------------

const CONTROLS = ["input", "select", "textarea"];

/**
 * Structure of a rendered view in `main`: exactly one h1, h1#page-title, inside header.page-header,
 * and that header is the first element in main (a .page-header-sentinel may come before it);
 * main is aria-labelledby="page-title"; every input, select and textarea (not type=hidden) has
 * a label[for], an enclosing label, an aria-label or an aria-labelledby that resolves; every
 * aria-describedby id resolves; every table is inside .table-wrap inside .card.
 */
export function assertViewStructure(main) {
  const problems = [];
  const doc = main.ownerDocument;
  const byId = (id) => doc.getElementById(id);

  if (main.getAttribute("aria-labelledby") !== "page-title") problems.push('main is not aria-labelledby="page-title"');
  const h1s = main.querySelectorAll("h1");
  if (h1s.length !== 1) problems.push(`main has ${h1s.length} h1 elements, not 1`);
  const h1 = h1s[0];
  if (h1 && h1.id !== "page-title") problems.push("the h1 is not #page-title");
  const header = h1 ? h1.closest("header") : null;
  if (h1 && !(header && header.classList.contains("page-header"))) problems.push("the h1 is not inside header.page-header");
  const first = main.children.find((c) => !c.classList.contains("page-header-sentinel"));
  if (header && first !== header) problems.push("header.page-header is not first in main");

  for (const el of allElements(main)) {
    const tag = el.tagName.toLowerCase();
    if (CONTROLS.includes(tag) && (el.getAttribute("type") || "").toLowerCase() !== "hidden") {
      const id = el.id;
      const forLabel = id && allElements(doc.body).some((n) => n.tagName === "LABEL" && n.getAttribute("for") === id);
      const wrapped = Boolean(el.parentNode && el.parentNode.closest && el.parentNode.closest("label"));
      const aria = (el.getAttribute("aria-label") || "").trim() !== "";
      const ids = (el.getAttribute("aria-labelledby") || "").split(/\s+/).filter(Boolean);
      const labelledBy = ids.length > 0 && ids.every((i) => byId(i));
      if (!(forLabel || wrapped || aria || labelledBy)) problems.push(`<${tag}${id ? `#${id}` : ""}> has no label`);
    }
    for (const ref of (el.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean)) {
      if (!byId(ref)) problems.push(`aria-describedby="${ref}" on <${tag}> does not resolve`);
    }
    if (tag === "table") {
      const wrap = el.parentNode && el.parentNode.closest ? el.parentNode.closest(".table-wrap") : null;
      const card = wrap && wrap.parentNode && wrap.parentNode.closest ? wrap.parentNode.closest(".card") : null;
      if (!wrap) problems.push("a table is not inside .table-wrap");
      else if (!card) problems.push("a .table-wrap is not inside .card");
    }
  }
  assert.deepEqual(problems, [], `view structure:\n${problems.join("\n")}`);
}
