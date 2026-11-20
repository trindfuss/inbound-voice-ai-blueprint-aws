// Component and view stylesheets and the view checker in viewkit.js.
// Examples: every component class and every view class has a selector; disabled
// controls use --color-text-muted with no opacity; button text is token-based in both themes;
// reduced motion and forced colours are handled.
// viewkit's own examples include a planted unstyled class and a planted style attribute.
// Property test (builder part): the builders, rendered with generated arguments, use only
// classes that assertClassesStyled finds. The token scan runs in tokens.test.js and
// picks up both new files by itself.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { array, forAll, gen, int, oneOf, string, unicodeString } from "./gen.js";
import {
  STYLE_SOURCES, assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder,
  selectorClasses, styledClasses,
} from "./viewkit.js";
import { field, h } from "../lib/dom.js";
import {
  badge, card, compareCards, emptyState, envBadge, langBadge, pageHeader, panel, skeleton,
  tableWrap, toast,
} from "../lib/components.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (f) => readFileSync(join(ROOT, f), "utf8");
const strip = (css) => css.replace(/\/\*[\s\S]*?\*\//g, "");

/** Top-level and @media rules as {media, selector, body}. */
function rules(css, media = null) {
  const text = strip(css);
  const out = [];
  let i = 0;
  while (i < text.length) {
    const open = text.indexOf("{", i);
    if (open < 0) break;
    const prelude = text.slice(i, open).trim();
    let depth = 1;
    let j = open + 1;
    while (j < text.length && depth > 0) {
      if (text[j] === "{") depth++;
      else if (text[j] === "}") depth--;
      j++;
    }
    const body = text.slice(open + 1, j - 1);
    if (prelude.startsWith("@")) out.push(...rules(body, prelude));
    else out.push({ media, selector: prelude, body });
    i = j;
  }
  return out;
}

const components = read("components.css");
const views = read("views.css");

// The classes the builders emit, each needing a selector in components.css.
const COMPONENT_CLASSES = [
  // buttons
  "primary", "danger", "quiet", "sm", "is-busy", "button",
  // field and inputs
  "field", "hint", "note", "form-note", "field-required", "field-error", "select-wrap", "select-chevron",
  "inline", "switch",
  // table
  "table-wrap", "num",
  // page header (inner parts; the sticky shell is in layout.css)
  "back-link", "page-header-text", "page-description", "page-actions", "page-header-sentinel",
  // card
  "card", "flush", "card-header", "card-heading", "card-title", "card-description", "card-actions", "card-body",
  // badges
  "badge", ...["neutral", "brand", "success", "warning", "danger", "info"].map((t) => `badge--${t}`),
  "env-test", "env-prod", "lang-badge",
  // panel
  "panel", ...["error", "warning", "info", "success"].map((t) => `panel--${t}`), "panel-body", "panel-title",
  // empty state
  "empty-state", "empty-state-icon", "empty-state-title", "empty-state-message", "empty-state-action",
  // loading
  "loading", "skeleton", ...["page", "table", "form", "inline"].map((k) => `skeleton--${k}`),
  "skeleton-bar", ...["title", "line", "row", "label", "input"].map((k) => `skeleton-bar--${k}`), "skeleton-card",
  // toast
  "toasts", "toast", ...["error", "warning", "info", "success"].map((t) => `toast--${t}`),
  "toast-body", "toast-title", "toast-message", "toast-dismiss",
  // dialog
  "dialog", "dialog-header", "dialog-title", "dialog-body", "dialog-details", "dialog-footer",
  // compare
  "compare-group", "compare", "compare-list", "compare-row", "is-changed", "compare-value", "compare-details", "compare-json",
];

// The per-view arrangements, each needing a selector in views.css.
const VIEW_CLASSES = ["field-grid", "lang-pair", "preview", "is-security", "is-invalid", "search-row", "changes",
  "change-label", "change-value", "change-before", "change-after"];

test("components.css: every component class has a selector", () => {
  const have = selectorClasses(components);
  const missing = COMPONENT_CLASSES.filter((c) => !have.has(c));
  assert.deepEqual(missing, [], `no selector in components.css for: ${missing.join(", ")}`);
});

test("components.css: bare button, input, select, textarea, table and dialog take the new look", () => {
  const selectors = rules(components).map((r) => r.selector.split(",").map((s) => s.trim())).flat();
  for (const el of ["button", 'input[type="text"]', "select", "textarea", 'input[type="checkbox"]', "table", "th", "thead th"]) {
    assert.ok(selectors.includes(el), `a rule for bare ${el}`);
  }
  assert.ok(selectors.includes(".dialog") && selectors.includes(".dialog::backdrop"), "the dialog and its backdrop");
});

test("views.css: every per-view arrangement has a selector", () => {
  const have = selectorClasses(views);
  const missing = VIEW_CLASSES.filter((c) => !have.has(c));
  assert.deepEqual(missing, [], `no selector in views.css for: ${missing.join(", ")}`);
  assert.ok(rules(views).some((r) => r.media === "@media (min-width: 960px)" && r.selector === ".field-grid"),
    "the Masking field grid has two columns at 960px and wider");
  assert.match(views, /\.change-value\s*\{\s*text-decoration:\s*line-through/, '"Before" is struck through');
});

test("disabled controls use --color-text-muted (through the button token) and never opacity", () => {
  const disabled = rules(components).filter((r) => /:disabled|\[aria-disabled="true"\]/.test(r.selector));
  assert.ok(disabled.length > 0, "there are disabled rules");
  const buttons = disabled.filter((r) => /^button/.test(r.selector));
  assert.ok(buttons.some((r) => /(^|;)\s*color:\s*var\(--button-disabled-text\)/.test(r.body)), "disabled buttons use the disabled text token");
  const tokens = read("tokens.css");
  assert.match(tokens, /--button-disabled-text:\s*var\(--color-text-muted\);/, "which is --color-text-muted");
  for (const f of ["base.css", "layout.css", "components.css", "views.css"]) {
    for (const r of rules(read(f))) {
      if (/:disabled|aria-disabled/.test(r.selector)) assert.doesNotMatch(r.body, /opacity/, `${f}: ${r.selector}`);
    }
  }
  // Each button variant is listed in the disabled rule, so it wins over the variant's hover.
  const list = buttons.map((r) => r.selector).join(",");
  for (const v of ["primary", "danger", "quiet"]) {
    assert.match(list, new RegExp(`button\\.${v}:disabled`), `button.${v}:disabled`);
    assert.match(list, new RegExp(`button\\.${v}\\[aria-disabled="true"\\]`), `button.${v}[aria-disabled]`);
  }
});

test("button text colours are tokens, so dark mode keeps its contrast (no #fff)", () => {
  const body = (sel) => rules(components).filter((r) => r.selector.split(",").map((s) => s.trim()).includes(sel)).map((r) => r.body).join(";");
  assert.match(body("button.primary"), /color:\s*var\(--button-primary-text\)/);
  assert.match(body("button.danger"), /color:\s*var\(--button-danger-text\)/);
});

test("busy buttons show a spinner, whether marked by .is-busy or aria-busy", () => {
  const sel = rules(components).filter((r) => /::before/.test(r.selector) && /content:/.test(r.body)).map((r) => r.selector).join(",");
  assert.match(sel, /button\.is-busy::before/);
  assert.match(sel, /button\[aria-busy="true"\]::before/);
});

test("reduced motion stops the shimmer, the spinner and the press; forced colours are handled", () => {
  const reduced = rules(components).filter((r) => /prefers-reduced-motion:\s*reduce/.test(r.media ?? ""));
  const stopped = reduced.filter((r) => /animation:\s*none/.test(r.body)).map((r) => r.selector).join(",");
  for (const s of [".skeleton-bar", "button.is-busy::before", ".toast"]) assert.ok(stopped.includes(s), `${s} has no animation`);
  assert.ok(reduced.some((r) => /button:active/.test(r.selector) && /transform:\s*none/.test(r.body)), "no press translate");
  assert.ok(rules(components).some((r) => /forced-colors:\s*active/.test(r.media ?? "")), "a forced-colours block");
  // The animated rules live outside the reduced-motion block.
  assert.ok(rules(components).some((r) => r.media === null && r.selector === ".skeleton-bar" && /animation:/.test(r.body)));
});

test("components.css doesn't redefine .icon (base.css owns its sizing)", () => {
  for (const r of rules(components).concat(rules(views))) {
    for (const s of r.selector.split(",")) assert.notEqual(s.trim(), ".icon", "a bare .icon rule");
  }
});

// ---------------------------------------------------------------- viewkit examples

const goodView = () => {
  const input = h("input", { type: "text", "data-field": "x" });
  const f = field("Name", input, "A hint.");
  return [
    pageHeader({ title: "Tools", description: "A view." }),
    card({ title: "Table", flush: true }, tableWrap(h("table", { "aria-label": "Rows" }, h("tbody", {}, h("tr", {}, h("td.num", {}, "1")))))),
    card({ title: "Form" }, h("form", {}, f.wrap, h("label.inline", {}, h("input", { type: "checkbox" }), " Wrapped"))),
  ];
};

test("viewkit: a well-formed view passes all three checks", async () => {
  await renderUnder(({ main }) => main.append(...goodView()), {
    check: ({ main }) => {
      assertNoInlineStyle(main);
      assertClassesStyled(main);
      assertViewStructure(main);
    },
  });
});

test("viewkit: a planted unstyled class fails assertClassesStyled, naming the class", async () => {
  await renderUnder(({ main }) => main.append(h("div.card", {}, h("span.not-a-styled-class", {}, "x"))), {
    check: ({ main }) => assert.throws(() => assertClassesStyled(main), /not-a-styled-class/),
  });
  // A class only in a declaration value or a comment doesn't count as styled.
  const set = selectorClasses('/* .ghost */ .real { content: ".fake"; width: 0.5rem; } @media (min-width: 9px) { .inner { gap: 0 } }');
  assert.deepEqual([...set].sort(), ["inner", "real"]);
  assert.deepEqual(STYLE_SOURCES, ["tokens.css", "base.css", "layout.css", "components.css", "views.css"]);
  assert.ok(styledClasses().has("badge--warning"));
});

test("viewkit: a planted style attribute or handler attribute fails assertNoInlineStyle", async () => {
  await renderUnder(({ main, document }) => {
    const el = document.createElement("div");
    el.setAttribute("style", "color: red");
    main.append(el);
  }, { check: ({ main }) => assert.throws(() => assertNoInlineStyle(main), /has style/) });
  await renderUnder(({ main, document }) => {
    const el = document.createElement("button");
    el.setAttribute("onclick", "x()");
    main.append(h("div", {}, el));
  }, { check: ({ main }) => assert.throws(() => assertNoInlineStyle(main), /has onclick/) });
});

test("viewkit: assertViewStructure catches each planted structure fault", async () => {
  const cases = [
    [() => [h("h2", {}, "No h1")], /0 h1/],
    [() => [h("p", {}, "first"), pageHeader({ title: "Late" })], /not first in main/],
    [() => [h("header", {}, h("h1#page-title", {}, "Bare header"))], /not inside header\.page-header/],
    [() => [pageHeader({ title: "T" }), h("input", { type: "text" })], /has no label/],
    [() => [pageHeader({ title: "T" }), h("select", { "aria-labelledby": "nowhere" })], /has no label/],
    [() => [pageHeader({ title: "T" }), h("textarea", { "aria-label": "x", "aria-describedby": "gone" })], /does not resolve/],
    [() => [pageHeader({ title: "T" }), card({ title: "C" }, h("table"))], /not inside \.table-wrap/],
    [() => [pageHeader({ title: "T" }), tableWrap(h("table"))], /not inside \.card/],
  ];
  for (const [build, re] of cases) {
    await renderUnder(({ main }) => main.append(...build()), { check: ({ main }) => assert.throws(() => assertViewStructure(main), re) });
  }
  // main must be labelled by the page title.
  await renderUnder(({ main }) => { main.removeAttribute("aria-labelledby"); main.append(pageHeader({ title: "T" })); },
    { check: ({ main }) => assert.throws(() => assertViewStructure(main), /aria-labelledby/) });
  // A sentinel before the header is allowed (design, implementation notes), and hidden inputs need no label.
  await renderUnder(({ main }) => main.append(h("div.page-header-sentinel", { "aria-hidden": "true" }), pageHeader({ title: "T" }),
    h("input", { type: "hidden", name: "rev" })), { check: ({ main }) => assertViewStructure(main) });
});

test("viewkit: renderUnder uninstalls the DOM even when the render throws", async () => {
  await assert.rejects(renderUnder(() => { throw new Error("boom"); }), /boom/);
  assert.equal(typeof globalThis.document, "undefined");
});

// ---------------------------------------------------------------- PBT (builder part)

const label = unicodeString({ minLength: 0, maxLength: 8 });
const someText = gen((rng) => {
  const s = unicodeString({ minLength: 1, maxLength: 8 })(rng);
  return s.trim() ? s : `x${s}`;
});
const ICONS = ["tools", "info", "prompts", "history", "users"];
const leaf = oneOf(null, true, false, int(-5, 5), string({ maxLength: 3 }));
const wrapped = gen((rng) => ({ value: leaf(rng), source: oneOf("policy", "tool")(rng) }));
const obj = gen((rng) => {
  const keys = array(oneOf("rev", "a", "b", "mask", "after_ms"), { maxLength: 4 })(rng);
  return Object.fromEntries(keys.map((k) => [k, oneOf(leaf, wrapped, gen((r) => ({ a: leaf(r), after_ms: wrapped(r) })))(rng)]));
});

const builders = oneOf(
  gen((rng) => () => pageHeader({ title: label(rng), description: oneOf(undefined, label)(rng),
    back: oneOf(undefined, { href: "#/tools", label: "All tools" })(rng),
    actions: [h("button", { class: oneOf(null, "primary", "danger", "quiet", "sm")(rng) }, someText(rng))] })),
  gen((rng) => () => card({ title: oneOf(undefined, someText)(rng), description: oneOf(undefined, label)(rng),
    actions: oneOf([], [h("button.sm", {}, "Act")])(rng), flush: oneOf(true, false)(rng), label: oneOf(undefined, someText)(rng) }, label(rng))),
  gen((rng) => () => card({ flush: true }, tableWrap(h("table", { "aria-label": oneOf(undefined, someText)(rng) })))),
  gen((rng) => () => badge(someText(rng), oneOf("neutral", "brand", "success", "warning", "danger", "info")(rng),
    { icon: oneOf(undefined, ...ICONS)(rng) })),
  gen((rng) => () => envBadge(oneOf("dev", "test", "prod", label)(rng))),
  gen((rng) => () => langBadge(oneOf("en-US", "es-US", "fr", string({ minLength: 1, maxLength: 5 }))(rng))),
  gen((rng) => () => panel({ tone: oneOf("error", "warning", "info", "success")(rng), title: oneOf(undefined, label)(rng) }, label(rng))),
  gen((rng) => () => emptyState({ icon: oneOf(undefined, ...ICONS)(rng), title: label(rng), message: oneOf(undefined, label)(rng),
    action: oneOf(undefined, h("button.primary", {}, "Go"))(rng) })),
  gen((rng) => () => skeleton(oneOf("page", "table", "form", "inline")(rng))),
  gen((rng) => () => compareCards(obj(rng), obj(rng))),
  gen((rng) => () => toast({ tone: oneOf("error", "warning", "info", "success")(rng), title: label(rng), message: label(rng) }).element),
);

test("PBT (builder part): every class the builders use has a selector", async () => {
  const styled = styledClasses();
  await forAll(array(builders, { minLength: 1, maxLength: 4 }), (builds) => renderUnder(({ main }) => {
    for (const build of builds) main.append(build());
  }, {
    check: ({ main, toasts }) => {
      assertClassesStyled(main, { styled });
      assertClassesStyled(toasts, { styled });
      assertNoInlineStyle(main);
    },
  }));
});
