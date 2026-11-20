// Prompts view and the prompt editor.
// Characterization (written before the restyle, unchanged after it, apart from the one call
// line noted at runView): the view's and the editor's controls per language, the placeholder
// picker, the preview's validation messages, the save messages and the API calls. The language
// fields' labels are the one label that differs (from the code to a language badge and the
// language's name), so the editor's characterization leaves those labels out and a separate
// example checks them.
// PBT (no inline style, every class styled, view structure) over generated catalogs (missing languages, placeholders, invalid
// preview rows), with an editor open.
import assert from "node:assert/strict";
import { test } from "node:test";

import { ApiError, mapError } from "../lib/api.js";
import { checkPromptTexts } from "../lib/validate.js";
import { Event } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string, unicodeString } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder } from "./viewkit.js";
import { OTHER_PREFIXES, promptEditor, promptsView } from "../views/prompts.js";

// ---------------------------------------------------------------- helpers

const flush = async (n = 8) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };
const norm = (s) => s.replace(/\s+/g, " ").trim();

/** A recording fake of lib/api.js's client: route(method, path, arg) answers each call. */
function fakeApi(route) {
  const calls = [];
  const verb = (method) => async (segments, arg) => {
    const path = segments.join("/");
    calls.push(arg === undefined ? [method, path] : [method, path, structuredClone(arg)]);
    return route(method, path, arg);
  };
  return { calls, get: verb("get"), put: verb("put"), post: verb("post"), patch: verb("patch"), del: verb("del") };
}

/** Text a screen reader reads: aria-hidden subtrees and form controls left out. */
function accText(node) {
  if (node.nodeType === 3) return node.textContent;
  if (node.getAttribute("aria-hidden") === "true") return "";
  if (["INPUT", "SELECT", "TEXTAREA"].includes(node.tagName)) return "";
  return node.childNodes.map(accText).join("");
}

function labelOf(el) {
  if (el.getAttribute("aria-label")) return el.getAttribute("aria-label");
  const forLabel = el.id && el.ownerDocument.body.querySelectorAll("label").find((l) => l.getAttribute("for") === el.id);
  if (forLabel) return norm(accText(forLabel));
  const wrap = el.closest("label");
  if (wrap) return norm(accText(wrap));
  return el.tagName === "BUTTON" ? norm(accText(el)) : "";
}

const byJson = (a, b) => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1);

/** The view's controls as {tag, type, field, label, options, disabled}, sorted. */
function controls(root) {
  return root.querySelectorAll("input, select, textarea, button").map((el) => {
    const d = { tag: el.localName };
    if (el.getAttribute("type")) d.type = el.getAttribute("type");
    if (el.getAttribute("data-field")) d.field = el.getAttribute("data-field");
    d.label = labelOf(el);
    if (el.localName === "select") d.options = el.querySelectorAll("option").map((o) => o.value);
    if (el.disabled) d.disabled = true;
    return d;
  }).sort(byJson);
}

const links = (root) => root.querySelectorAll("a[href]").map((a) => [a.getAttribute("href"), norm(accText(a))]).sort(byJson);
/** Each error panel as "title | body". */
const errorTexts = (root) => root.querySelectorAll(".panel--error")
  .map((p) => p.querySelectorAll(".panel-body > *").map((c) => norm(c.textContent)).join(" | "));

function submit(form, submitter) {
  const ev = new Event("submit", { bubbles: true, cancelable: true });
  ev.submitter = submitter;
  form.dispatchEvent(ev);
}

function type(el, value) {
  el.value = value;
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

const ADMIN = { groups: ["admin"], email: "admin@example.com", username: "admin" };
const VIEWER = { groups: ["admin-viewer"], email: "viewer@example.com", username: "viewer" };

// ---------------------------------------------------------------- fixtures

const LANGUAGES = ["en-US", "es-US"];

const MASKS = {
  languages: LANGUAGES,
  takes_effect_s: 60,
  prompts: [
    { key: "mask.get_balance", owner_tool: "get_balance", rev: 2, texts: { "en-US": "Your {account} balance.", "es-US": null } },
    { key: "mask.ack", owner_tool: null, rev: 1, texts: { "en-US": "Okay.", "es-US": "Vale." } },
  ],
};
const ARGS = {
  languages: LANGUAGES,
  prompts: [{ key: "arg.account.checking", rev: 5, texts: { "en-US": "checking", "es-US": "corriente" } }],
};
const TOOLS = {
  tools: [
    { name: "get_balance", mask: { pool: [{ key: "mask.get_balance", texts: MASKS.prompts[0].texts }] } },
    { name: "transfer", mask: null },
  ],
};
const MASKING = { ack_pool: [{ key: "mask.ack", texts: MASKS.prompts[1].texts }] };
const PLACEHOLDERS = {
  placeholders: [{
    name: "account", placeholder: "{account}",
    values: [
      { value: "checking", key: "arg.account.checking", words: { "en-US": "checking", "es-US": "corriente" } },
      { value: "savings", key: "arg.account.savings", words: { "en-US": "savings" } },
    ],
    missing: ["arg.account.savings[es-US]"],
  }],
};

/** Other groups, only for the tests that ask for them (via setup's `groups`). */
let GROUPS = {};
const SLOTS = {
  prefix: "slot.", languages: LANGUAGES, takes_effect_s: 60,
  prompts: [{ key: "slot.zip.ask", rev: 3, editable: true, texts: { "en-US": "Your ZIP, {zip}?", "es-US": "Su ZIP, {zip}?" } }],
};
const CONFIRMS = {
  prefix: "confirm.", languages: LANGUAGES, takes_effect_s: 60,
  prompts: [{ key: "confirm.pay", rev: 1, editable: false, reason: "protected read-back",
    texts: { "en-US": "Pay {amount}?", "es-US": "Pagar {amount}?" } }],
};

/** The placeholders with every word present, so a phrase using {account} previews in full. */
const COMPLETE = structuredClone(PLACEHOLDERS);
COMPLETE.placeholders[0].values[1].words["es-US"] = "ahorros";
COMPLETE.placeholders[0].missing = [];

function promptRoutes({ complete = false } = {}) {
  return (method, path, arg) => {
    if (complete && method === "get" && path === "placeholders/get_balance") return structuredClone(COMPLETE);
    if (method === "get" && path === "prompts" && GROUPS[arg.prefix]) return structuredClone(GROUPS[arg.prefix]);
    if (method === "get" && path === "prompts" && !["mask.", "arg."].includes(arg.prefix)) throw new Error("no such prefix");
    if (method === "get" && path === "prompts") return structuredClone(arg.prefix === "mask." ? MASKS : ARGS);
    if (method === "get" && path === "tools") return structuredClone(TOOLS);
    if (method === "get" && path === "masking") return structuredClone(MASKING);
    if (method === "get" && path === "placeholders/get_balance") return structuredClone(PLACEHOLDERS);
    if (method === "put") return { changed: true, takes_effect_s: 60, prompt: { key: path, rev: 9, texts: arg.texts } };
    throw new Error(`unexpected ${method} ${path}`);
  };
}

/** The one line that follows the view's signature: (main, ctx, {params}). */
const runView = (main, ctx, params) => promptsView(main, ctx, { params });

function setup(user, query = "", route = promptRoutes()) {
  const api = fakeApi(route);
  const ctx = { api, user };
  return { api, ctx, run: (main) => runView(main, ctx, new URLSearchParams(query)) };
}

const buttonNamed = (root, name) => root.querySelectorAll("button").filter((b) => norm(accText(b)) === name);

async function openEditor(main, index) {
  buttonNamed(main, "Edit")[index].click();
  await flush();
  return main.querySelector("#prompt-editor");
}

/** The preview's rows as [language, values, caller hears], badges left out. */
function previewRows(root) {
  const table = root.querySelector("table.preview");
  return table.querySelectorAll("tbody tr").map((tr) => tr.querySelectorAll("td").map((td) =>
    norm(td.childNodes.map((c) => (c.nodeType === 1 && c.classList.contains("badge") ? "" : c.textContent)).join(""))));
}

const withoutLangLabels = (list) => list.map((c) => (c.field && c.field.startsWith("texts[") ? (({ label, ...rest }) => rest)(c) : c));

// ---------------------------------------------------------------- characterization

test("characterization: the view's GETs (four, then one per other prefix), its Edit buttons, and the tool filter", async () => {
  const { api, run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.deepEqual(api.calls, [["get", "prompts", { prefix: "mask." }], ["get", "prompts", { prefix: "arg." }],
      ["get", "tools"], ["get", "masking"], ...OTHER_PREFIXES.map((prefix) => ["get", "prompts", { prefix }])]);
    assert.deepEqual(controls(main), Array(3).fill({ tag: "button", type: "button", label: "Edit" }));
    assert.equal(main.querySelectorAll("table").length, 2, "no group for a prefix that answers nothing");
    assert.deepEqual(links(main), []);
    assert.deepEqual(main.querySelectorAll("summary").map((s) => norm(s.textContent)),
      ["get_balance: 1 acknowledgements x 1 phrases"]);
  });
  const filtered = setup(VIEWER, "tool=transfer");
  await renderUnder(async ({ main }) => {
    await filtered.run(main);
    assert.deepEqual(controls(main), [{ tag: "button", type: "button", label: "View" }]);
    assert.deepEqual(links(main), [["#/prompts", "Show all"]]);
    assert.match(main.textContent, /Showing transfer's phrases\./);
  });
});

test("characterization: a phrase editor has a field per language, the picker and Save; it GETs the placeholders", async () => {
  const { api, run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const editor = await openEditor(main, 0);
    assert.deepEqual(api.calls.at(-1), ["get", "placeholders/get_balance"]);
    assert.deepEqual(withoutLangLabels(controls(editor)), [
      { tag: "button", type: "button", label: "{account}" },
      { tag: "button", type: "submit", label: "Save" },
      { tag: "textarea", field: "texts[en-US]" },
      { tag: "textarea", field: "texts[es-US]" },
    ]);
    assert.match(editor.textContent, /Missing words \(fill these in under arg\.\*\): arg\.account\.savings\[es-US\]/);
    assert.deepEqual(previewRows(editor), [
      ["en-US", "account=checking", "Your checking balance."],
      ["en-US", "account=savings", "Your savings balance."],
      ["es-US", "(no placeholders)", "The text is empty."],
    ]);
  });
});

test("characterization: the preview's messages follow the text; the picker inserts at the caret", async () => {
  const { run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const editor = await openEditor(main, 0);
    const es = editor.querySelector('[data-field="texts[es-US]"]');
    type(es, "Su saldo {account} {bogus}");
    assert.deepEqual(previewRows(editor).slice(2), [
      ["es-US", "account=checking", "{bogus} isn't one of this tool's placeholders."],
      ["es-US", "account=savings", "{bogus} isn't one of this tool's placeholders."],
    ]);
    type(es, "Su saldo ");
    es.focus();
    es.selectionStart = es.value.length;
    es.selectionEnd = es.value.length;
    es.setSelectionRange = () => {};
    buttonNamed(editor, "{account}")[0].click();
    assert.equal(es.value, "Su saldo {account}");
    assert.deepEqual(previewRows(editor).slice(2), [
      ["es-US", "account=checking", "Su saldo corriente"],
      ["es-US", "account=savings", "Missing word for arg.account.savings in es-US."],
    ]);
  });
});

test("characterization: Save checks every language and the preview, then PUTs the texts with the rev", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { api, run } = setup(ADMIN, "", promptRoutes({ complete: true }));
  await renderUnder(async ({ main }) => {
    await run(main);
    const editor = await openEditor(main, 0);
    const form = editor.querySelector("form");
    submit(form, buttonNamed(editor, "Save")[0]);
    await flush();
    assert.deepEqual(errorTexts(editor), ["Not saved | texts[es-US]: every language needs its text; texts[es-US]: The text is empty."]);
    type(editor.querySelector('[data-field="texts[es-US]"]'), "Su saldo {account}.");
    submit(form, buttonNamed(editor, "Save")[0]);
    await flush();
    assert.deepEqual(api.calls.at(-1), ["put", "prompts/mask.get_balance",
      { rev: 2, texts: { "en-US": "Your {account} balance.", "es-US": "Su saldo {account}." } }]);
  });
});

test("characterization: a word editor has a text input per language, no preview, and rejects braces", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { api, run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const before = api.calls.length;
    const editor = await openEditor(main, 2);
    assert.equal(api.calls.length, before, "no placeholders GET for a word");
    assert.deepEqual(withoutLangLabels(controls(editor)), [
      { tag: "button", type: "submit", label: "Save" },
      { tag: "input", type: "text", field: "texts[en-US]" },
      { tag: "input", type: "text", field: "texts[es-US]" },
    ]);
    assert.equal(editor.querySelector("table.preview"), null);
    type(editor.querySelector('[data-field="texts[en-US]"]'), "{x}");
    submit(editor.querySelector("form"), buttonNamed(editor, "Save")[0]);
    await flush();
    assert.deepEqual(errorTexts(editor), ["Not saved | texts[en-US]: a word can't contain { or }"]);
    assert.equal(api.calls.length, before);
  });
});

test("characterization: a viewer's editor is read-only with no Save", async () => {
  const { run } = setup(VIEWER);
  await renderUnder(async ({ main }) => {
    await run(main);
    buttonNamed(main, "View")[0].click();
    await flush();
    const editor = main.querySelector("#prompt-editor");
    assert.deepEqual(withoutLangLabels(controls(editor)), [
      { tag: "button", type: "button", label: "{account}", disabled: true },
      { tag: "textarea", field: "texts[en-US]", disabled: true },
      { tag: "textarea", field: "texts[es-US]", disabled: true },
    ]);
  });
});

// ---------------------------------------------------------------- the restyled anatomy

test("each editor has exactly one .lang-pair, a field per language labelled with a language badge and name", async () => {
  const { run } = setup(ADMIN);
  await renderUnder(async ({ main, document }) => {
    await run(main);
    for (const index of [0, 2]) {
      const editor = await openEditor(main, index);
      const pairs = editor.querySelectorAll(".lang-pair");
      assert.equal(pairs.length, 1);
      const fields = pairs[0].querySelectorAll(".field");
      assert.equal(fields.length, 2);
      for (const [i, f] of fields.entries()) {
        const control = f.querySelector("textarea, input");
        const label = document.body.querySelectorAll("label").find((l) => l.getAttribute("for") === control.id);
        assert.ok(label.querySelector(".lang-badge"), "the label holds a language badge");
        assert.equal(norm(accText(label)), ["English", "Spanish"][i], "and the language name");
        assert.equal(control.getAttribute("data-field"), `texts[${LANGUAGES[i]}]`);
      }
      assert.ok(editor.querySelector("section.card"), "the editor is a card");
    }
  });
});

test("anatomy: page header, cards, langBadge headers, missing and placeholders badges, invalid rows, mono picker", async () => {
  const { run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.equal(norm(main.querySelector("header.page-header h1").textContent), "Prompts");
    assert.ok(main.querySelectorAll("section.card").length >= 3);
    assert.ok(main.querySelector("thead .lang-badge"));
    const badges = main.querySelectorAll(".badge").map((b) => norm(accText(b)));
    assert.ok(badges.includes("missing") && badges.includes("placeholders"));
    const editor = await openEditor(main, 0);
    const invalid = editor.querySelectorAll("tr.is-invalid");
    assert.equal(invalid.length, 1);
    assert.equal(norm(invalid[0].querySelector(".badge--warning").textContent), "invalid");
    const pick = buttonNamed(editor, "{account}")[0];
    assert.ok(pick.classList.contains("sm") && pick.querySelector("code"), "a small button in mono");
    assertViewStructure(main);
  });
});

test("anatomy: opening shows a form skeleton until the placeholders arrive; Save passes ev.submitter", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const gate = () => { let open; const p = new Promise((r) => { open = r; }); return { p, open }; };
  const placeholders = gate();
  const put = gate();
  const base = promptRoutes({ complete: true });
  const { run } = setup(ADMIN, "", async (m, p, a) => {
    if (m === "get" && p.startsWith("placeholders")) await placeholders.p;
    if (m === "put") await put.p;
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(main);
    buttonNamed(main, "Edit")[0].click();
    await flush();
    assert.ok(main.querySelector("#prompt-editor .skeleton--form"), "a form skeleton while loading");
    placeholders.open();
    await flush();
    const editor = main.querySelector("#prompt-editor");
    type(editor.querySelector('[data-field="texts[es-US]"]'), "Su saldo {account}.");
    const btn = buttonNamed(editor, "Save")[0];
    submit(editor.querySelector("form"), btn);
    await flush();
    assert.equal(btn.getAttribute("aria-disabled"), "true");
    assert.ok(btn.classList.contains("is-busy"));
    put.open();
    await flush();
    assert.ok(!btn.classList.contains("is-busy"));
  });
});

test("anatomy: an invalid client-side save marks the language field", async () => {
  const { run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const editor = await openEditor(main, 0);
    submit(editor.querySelector("form"), buttonNamed(editor, "Save")[0]);
    await flush();
    assert.equal(editor.querySelector('[data-field="texts[es-US]"]').getAttribute("aria-invalid"), "true");
  });
});

// ---------------------------------------------------------------- PBT

const LANGS = oneOf(["en-US", "es-US"], ["en-US"], ["en-US", "es-US", "fr-FR"]);
const userGen = oneOf(ADMIN, VIEWER, { groups: ["admin-editor"], email: "e@example.com", username: "e" });
const phraseText = oneOf(null, "", unicodeString({ maxLength: 6 }), "Your {account}", "{nope} x", "a { b", "Hi {account} {when}");
const word = oneOf(undefined, "", "checking", "{x}", unicodeString({ minLength: 1, maxLength: 4 }));

const catalogGen = gen((rng) => {
  const languages = LANGS(rng);
  const tools = array(string({ minLength: 1, maxLength: 6, chars: "abcxyz_" }), { maxLength: 3 })(rng);
  const texts = (r) => Object.fromEntries(languages.filter(() => int(0, 4)(r) > 0).map((l) => [l, phraseText(r)]));
  const masks = array(gen((r) => ({
    key: `mask.${string({ minLength: 1, maxLength: 6 })(r)}`,
    owner_tool: tools.length && int(0, 2)(r) ? oneOf(...tools)(r) : null,
    rev: int(1, 9)(r),
    texts: texts(r),
  })), { maxLength: 4 })(rng);
  const args = array(gen((r) => ({ key: `arg.${string({ minLength: 1, maxLength: 6 })(r)}`, rev: 1, texts: texts(r) })), { maxLength: 3 })(rng);
  const placeholders = array(gen((r) => {
    const name = oneOf("account", "when", "amount")(r);
    return {
      name, placeholder: `{${name}}`,
      values: array(gen((r2) => ({
        value: string({ minLength: 1, maxLength: 4 })(r2),
        key: `arg.${name}.x`,
        words: Object.fromEntries(languages.map((l) => [l, word(r2)])),
      })), { maxLength: 3 })(r),
      missing: array(string({ minLength: 1, maxLength: 6 }), { maxLength: 2 })(r),
    };
  }), { maxLength: 2 })(rng);
  return {
    languages,
    masks: { languages, takes_effect_s: 60, prompts: masks },
    args: { languages, prompts: args },
    tools: { tools: tools.map((name) => ({ name, mask: int(0, 1)(rng) ? { pool: masks.filter((m) => m.owner_tool === name) } : null })) },
    masking: { ack_pool: masks.filter((m) => !m.owner_tool) },
    placeholders: { placeholders },
    only: tools.length && int(0, 3)(rng) === 0 ? tools[0] : null,
    open: int(0, 6)(rng),
  };
});

test("PBT: the view for generated catalogs, with an editor open", async () => {
  await forAll(record({ c: catalogGen, user: userGen }), async ({ c, user }) => {
    const api = fakeApi((method, path, arg) => {
      if (path === "prompts") return structuredClone(arg.prefix === "mask." ? c.masks : c.args);
      if (path === "tools") return structuredClone(c.tools);
      if (path === "masking") return structuredClone(c.masking);
      return structuredClone(c.placeholders);
    });
    await renderUnder(async ({ main }) => {
      await runView(main, { api, user }, new URLSearchParams(c.only ? `tool=${c.only}` : ""));
      const opens = buttonNamed(main, "Edit").concat(buttonNamed(main, "View"));
      if (opens.length) {
        opens[c.open % opens.length].click();
        await flush();
      }
      assertNoInlineStyle(main);
      assertClassesStyled(main);
      assertViewStructure(main);
    });
  });
});

test("PBT: promptEditor alone, for generated prompts and placeholders", async () => {
  await forAll(record({ c: catalogGen, user: userGen, isWord: oneOf(true, false) }), async ({ c, user, isWord }) => {
    const prompt = { key: isWord ? "arg.x.y" : "mask.x", owner_tool: isWord ? null : "x", rev: 1,
      texts: Object.fromEntries(c.languages.map((l, i) => [l, i ? null : "Hi {account}"])) };
    await renderUnder(async ({ main }) => {
      main.append(promptEditor({ api: fakeApi(() => ({})), user }, prompt, c.languages, { placeholders: c.placeholders.placeholders }));
      assertNoInlineStyle(main);
      assertClassesStyled(main);
    });
  });
});

// ---------------------------------------------------------------- keyed-phone login groups

const withGroups = async (groups, fn) => {
  GROUPS = groups;
  try { await fn(); } finally { GROUPS = {}; }
};
const groupTitles = (root) => root.querySelectorAll(".card h2, .card-title, .card [class*=title]")
  .map((e) => norm(e.textContent)).filter((t) => t.startsWith("Other prompts"));

test("other keys are listed in a group per prefix, with the read-only badge and reason", async () => {
  await withGroups({ "slot.": SLOTS, "confirm.": CONFIRMS }, async () => {
    const { run } = setup(ADMIN);
    await renderUnder(async ({ main }) => {
      await run(main);
      const tables = main.querySelectorAll("table[aria-label]").map((t) => t.getAttribute("aria-label"));
      assert.deepEqual(tables.filter((t) => t.startsWith("Prompts (")), ["Prompts (confirm.*)", "Prompts (slot.*)"]);
      assert.match(main.textContent, /slot\.zip\.ask/);
      assert.match(main.textContent, /read-only\s*protected read-back/);
      assert.equal(buttonNamed(main, "View").length, 1);
      assert.equal(buttonNamed(main, "Edit").length, 4);
    });
  });
});

test("a protected key opens read-only with the reason and no Save", async () => {
  await withGroups({ "confirm.": CONFIRMS }, async () => {
    const { api, run } = setup(ADMIN);
    await renderUnder(async ({ main }) => {
      await run(main);
      buttonNamed(main, "View")[0].click();
      await flush();
      const editor = main.querySelector("#prompt-editor");
      assert.deepEqual(withoutLangLabels(controls(editor)), [
        { tag: "textarea", field: "texts[en-US]", disabled: true },
        { tag: "textarea", field: "texts[es-US]", disabled: true },
      ]);
      assert.match(editor.textContent, /Read-only: protected read-back\./);
      assert.equal(api.calls.some((c) => c[0] === "put"), false);
    });
  });
});

const LANGUAGE_KEYS = {
  prefix: "language.", languages: LANGUAGES, takes_effect_s: 60,
  prompts: [{ key: "language.select", rev: 1, editable: true, languages: ["es-US"],
    texts: { "en-US": null, "es-US": "Para español, oprima dos." } }],
};

test("a single-language key's editor has only its language and saves only that text", async () => {
  await withGroups({ "language.": LANGUAGE_KEYS }, async () => {
    const { api, run } = setup(ADMIN);
    await renderUnder(async ({ main }) => {
      await run(main);
      const editor = await openEditor(main, 3);
      assert.deepEqual(withoutLangLabels(controls(editor)).filter((c) => c.field),
        [{ tag: "textarea", field: "texts[es-US]" }]);
      type(editor.querySelector('[data-field="texts[es-US]"]'), "Para español, oprima el dos.");
      submit(editor.querySelector("form"), buttonNamed(editor, "Save")[0]);
      await flush();
      assert.deepEqual(errorTexts(editor), []);
      assert.deepEqual(api.calls.at(-1), ["put", "prompts/language.select",
        { rev: 1, texts: { "es-US": "Para español, oprima el dos." } }]);
    });
  });
});

test("checkPromptTexts checks only the languages it is given", () => {
  assert.deepEqual(checkPromptTexts("language.select", { "es-US": "Oprima dos." }, ["es-US"]), []);
  assert.deepEqual(checkPromptTexts("language.select", { "es-US": "Oprima dos." }, LANGUAGES).map((e) => e.field),
    ["texts[en-US]"]);
});

test("checkPromptTexts' placeholder rule matches the API's", () => {
  const cur = { "en-US": "Pay {amount} from {from}", "es-US": null };
  const langs = ["en-US", "es-US"];
  const fields = (t) => checkPromptTexts("confirm.x", t, langs, cur).map((e) => e.field);
  assert.deepEqual(fields({ "en-US": "Pagar {from} {amount}", "es-US": "Pagar {amount} de {from}" }), [], "order is free");
  assert.deepEqual(fields({ "en-US": "Pay {amount}", "es-US": "Pagar {amount} {from}" }), ["texts[en-US]"], "one dropped");
  assert.deepEqual(fields({ "en-US": "Pay {amount} {from} {to}", "es-US": "x {amount} {from}" }), ["texts[en-US]"], "one added");
  assert.deepEqual(fields({ "en-US": "Pay {amount} {from}", "es-US": "sin variables" }), ["texts[es-US]"], "falls back to the first language's set");
  assert.deepEqual(fields({ "en-US": "Pay {amount} {from} {", "es-US": "x {amount} {from}" }), ["texts[en-US]"], "stray brace");
  assert.deepEqual(fields({ "en-US": "<p>Pay {amount} {from}</p>", "es-US": "x {amount} {from}" }), ["texts[en-US]"], "markup");
  assert.deepEqual(fields({ "en-US": "x".repeat(601) + "{amount}{from}", "es-US": "x {amount} {from}" }), ["texts[en-US]"], "too long");
  assert.deepEqual(checkPromptTexts("slot.x", { "en-US": "free", "es-US": "libre" }, langs), [], "no current texts: no placeholder rule");
});

test("mask.* phrases skip the same-set rule but keep the others", () => {
  const cur = { "en-US": "Your {account} balance.", "es-US": "Su saldo." };
  const langs = ["en-US", "es-US"];
  const fields = (key, t) => checkPromptTexts(key, t, langs, cur).map((e) => e.field);
  assert.deepEqual(fields("mask.get_balance", { "en-US": "Your balance.", "es-US": "Su saldo de {account}." }), [],
    "a mask phrase may lose or gain a placeholder");
  assert.deepEqual(fields("greeting.personal", { "en-US": "Your balance.", "es-US": "Su saldo." }), ["texts[en-US]"],
    "any other key keeps its set");
  assert.deepEqual(fields("agent.call_started", { "en-US": "Your {account} balance.", "es-US": "Su saldo." }), ["texts[en-US]"],
    "an agent.* key has its own allow-list instead (none here)");
  assert.deepEqual(fields("mask.get_balance", { "en-US": "Your {account} balance }", "es-US": "Su saldo." }), ["texts[en-US]"], "stray brace");
  assert.deepEqual(fields("mask.get_balance", { "en-US": "<b>Your</b> balance.", "es-US": "Su saldo." }), ["texts[en-US]"], "markup");
  assert.deepEqual(fields("mask.get_balance", { "en-US": "x".repeat(601), "es-US": "Su saldo." }), ["texts[en-US]"], "too long");
  assert.deepEqual(fields("mask.get_balance", { "en-US": "Your balance.", "es-US": " " }), ["texts[es-US]"], "empty");
});

test("the editor blocks a changed placeholder set before any PUT", async () => {
  await withGroups({ "slot.": SLOTS }, async () => {
    const { api, run } = setup(ADMIN);
    await renderUnder(async ({ main }) => {
      await run(main);
      const editor = await openEditor(main, 3);
      type(editor.querySelector('[data-field="texts[en-US]"]'), "Your ZIP?");
      submit(editor.querySelector("form"), buttonNamed(editor, "Save")[0]);
      await flush();
      assert.match(errorTexts(editor)[0], /texts\[en-US\]: must keep exactly the placeholders of the current text \(\{zip\}\)/);
      assert.equal(api.calls.some((c) => c[0] === "put"), false);
    });
  });
});

for (const [status, body, shown] of [
  [400, { error: "placeholders", message: "must keep exactly the placeholders of the current text ({zip})", field: "texts[en-US]" },
    /texts\[en-US\]: must keep exactly/],
  [403, { error: "protected", message: "this prompt is read-only (protected read-back)", field: "key", reason: "protected read-back" },
    /this prompt is read-only \(protected read-back\)/],
]) {
  test(`a ${status} on save is shown to the editor`, async () => {
    await withGroups({ "slot.": SLOTS }, async () => {
      const base = promptRoutes();
      const { api, run } = setup(ADMIN, "", (m, p, a) => {
        if (m === "put") throw mapped(status, body);
        return base(m, p, a);
      });
      await renderUnder(async ({ main }) => {
        await run(main);
        const editor = await openEditor(main, 3);
        type(editor.querySelector('[data-field="texts[en-US]"]'), "Your {zip}, please?");
        submit(editor.querySelector("form"), buttonNamed(editor, "Save")[0]);
        await flush();
        assert.equal(api.calls.at(-1)[0], "put");
        const field = editor.querySelector('[data-field="texts[en-US]"]');
        const text = status === 400 ? field.getAttribute("aria-invalid") + " " + editor.textContent : errorTexts(editor).join(" ");
        assert.match(text, shown);
      });
    });
  });
}

/** What the API client throws for an error response. */
function mapped(status, body) {
  return new ApiError(mapError(status, body));
}

test("OTHER_PREFIXES lists every non-mask, non-arg prefix in the seeded catalog", async () => {
  const { readFile } = await import("node:fs/promises");
  const catalog = JSON.parse(await readFile(new URL("../../../data/prompt_catalog.json", import.meta.url), "utf8"));
  const prefixes = new Set(catalog.entries.map((e) => `${e.key.split(".")[0]}.`));
  const missing = [...prefixes].filter((p) => p !== "mask." && p !== "arg." && !OTHER_PREFIXES.includes(p));
  assert.deepEqual(missing, [], "a catalog prefix the console never lists: add it to OTHER_PREFIXES");
});
