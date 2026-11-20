// Masking view: the policy-wide values and the acknowledgement pool.
// Characterization (written before the restyle, unchanged after it): the MASKING_FIELDS
// controls and their "Use the default" checkboxes, the client-side validation messages, and
// the API calls on load, save and opening an acknowledgement in the prompt editor.
// PBT (no inline style, every class styled, view structure) over generated masking payloads (missing languages, empty pools,
// problems, set and unset fields).
import assert from "node:assert/strict";
import { test } from "node:test";

import { Event } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string, unicodeString } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder } from "./viewkit.js";
import { MASKING_FIELDS } from "../lib/validate.js";
import { maskingView } from "../views/masking.js";

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

const DEFAULTS = {
  after_ms: 700, tool_after_ms: 1200, handoff_wait_ms: 1500, end_mode: "wait", chars_per_s: 15,
  silence_timeout_ms: 6000, silence_max_repeats: 2, silence_action: "escalate", greeting_silence_timeout_ms: 8000,
};

const MASKING = {
  rev: 3,
  masking: { after_ms: 500, end_mode: "immediate" },
  effective: { ...DEFAULTS, after_ms: 500, end_mode: "immediate" },
  defaults: DEFAULTS,
  ack_pool: [
    { key: "mask.ack", texts: { "en-US": "Okay.", "es-US": null } },
    { key: "mask.ack.1", texts: { "en-US": "Sure.", "es-US": "Claro." } },
  ],
  problems: ["mask.ack is missing es-US"],
  takes_effect_s: 60,
};

const PROMPT = { key: "mask.ack", rev: 4, texts: { "en-US": "Okay.", "es-US": "" } };

function maskingRoutes(data = MASKING) {
  return (method, path, arg) => {
    if (method === "get" && path === "masking") return structuredClone(data);
    if (method === "get" && path === "prompts") {
      return { languages: ["en-US", "es-US"], prompts: arg.prefix === "mask.ack" ? [structuredClone(PROMPT)] : [] };
    }
    if (method === "put") return { changed: true, takes_effect_s: 60, rev: data.rev + 1 };
    throw new Error(`unexpected ${method} ${path}`);
  };
}

function setup(user, route = maskingRoutes()) {
  const api = fakeApi(route);
  return { api, run: (main) => maskingView(main, { api, user }, {}) };
}

const fieldControls = (editable) => Object.entries(MASKING_FIELDS).flatMap(([name, spec]) => {
  const isSet = name in MASKING.masking;
  const control = spec.type === "enum"
    ? { tag: "select", field: `masking.${name}`, label: spec.label, options: spec.values }
    : { tag: "input", type: "number", field: `masking.${name}`, label: spec.label };
  if (!editable || !isSet) control.disabled = true;
  const useDefault = { tag: "input", type: "checkbox", label: "Use the default" };
  if (!editable) useDefault.disabled = true;
  return [control, useDefault];
});

// ---------------------------------------------------------------- characterization

test("characterization: the controls, the History link and one GET /masking", async () => {
  const { api, run } = await setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.deepEqual(api.calls, [["get", "masking"]]);
    const expected = [...fieldControls(true), { tag: "button", type: "submit", label: "Save" },
      { tag: "button", type: "button", label: "Edit" }, { tag: "button", type: "button", label: "Edit" }].sort(byJson);
    assert.deepEqual(controls(main), expected);
    assert.deepEqual(links(main), [["#/audit?item=policy%23masking", "History"]]);
    const after = main.querySelector('[data-field="masking.after_ms"]');
    assert.equal(after.value, "500");
    assert.equal(main.querySelector('[data-field="masking.chars_per_s"]').value, "15");
  });
});

test("characterization: a viewer sees every control disabled, View buttons and no Save", async () => {
  const { run } = await setup(VIEWER);
  await renderUnder(async ({ main }) => {
    await run(main);
    const expected = [...fieldControls(false),
      { tag: "button", type: "button", label: "View" }, { tag: "button", type: "button", label: "View" }].sort(byJson);
    assert.deepEqual(controls(main), expected);
    assert.match(main.textContent, /Your group can view only\./);
  });
});

test("characterization: \"Use the default\" disables the field and puts the default back", async () => {
  const { run } = await setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const input = main.querySelector('[data-field="masking.after_ms"]');
    const box = input.closest(".field").querySelector('input[type="checkbox"]');
    box.checked = true;
    box.dispatchEvent(new Event("change", { bubbles: true }));
    assert.equal(input.disabled, true);
    assert.equal(input.value, "700");
  });
});

test("characterization: client-side validation messages, and no PUT", async () => {
  const { api, run } = await setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    type(main.querySelector('[data-field="masking.after_ms"]'), "abc");
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    await flush();
    assert.deepEqual(errorTexts(main), ["Check the form | masking.after_ms: must be a whole number from 0 to 10000"]);
    assert.deepEqual(api.calls, [["get", "masking"]]);
  });
});

test("characterization: Save PUTs only the set fields with the rev", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { api, run } = await setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    type(main.querySelector('[data-field="masking.after_ms"]'), "900");
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    await flush();
    assert.deepEqual(api.calls, [["get", "masking"],
      ["put", "masking", { rev: 3, masking: { after_ms: 900, end_mode: "immediate" } }]]);
  });
});

test("characterization: Edit loads the prompt by prefix and opens the editor; an unknown key says so", async () => {
  const { api, run } = await setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const edits = main.querySelectorAll("button").filter((b) => norm(b.textContent) === "Edit");
    edits[0].click();
    await flush();
    assert.deepEqual(api.calls.at(-1), ["get", "prompts", { prefix: "mask.ack" }]);
    assert.ok(main.querySelector('[data-field="texts[en-US]"]'), "the prompt editor is open");
    edits[1].click();
    await flush();
    assert.deepEqual(api.calls.at(-1), ["get", "prompts", { prefix: "mask.ack.1" }]);
    assert.match(main.textContent, /mask\.ack\.1 isn't in the catalog yet\./);
  });
});

// ---------------------------------------------------------------- the restyled anatomy

test("anatomy: header with History and Save for masking-form, a field grid, langBadge headers, missing badges", async () => {
  const { run } = await setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const header = main.querySelector("header.page-header");
    assert.equal(norm(header.querySelector("h1").textContent), "Masking");
    assert.ok(header.querySelector("a.button.quiet"), "History is a quiet link-button");
    assert.equal(header.querySelector("button.primary").getAttribute("form"), "masking-form");
    assert.equal(main.querySelector("form").id, "masking-form");
    assert.ok(main.querySelector(".panel--warning"), "the problems are a warning panel");
    assert.ok(main.querySelector(".card .field-grid .field"), "the policy-wide values sit in a field grid in a card");
    assert.equal(main.querySelectorAll("thead .lang-badge").length, 2);
    assert.equal(main.querySelectorAll("tbody .badge--warning").map((b) => norm(b.textContent)).join(), "missing");
  });
});

test("anatomy: Save passes ev.submitter; a client-side error marks the field", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  let release;
  const pending = new Promise((r) => { release = r; });
  const base = maskingRoutes();
  const { run } = await setup(ADMIN, async (m, p, a) => (m === "put" ? (await pending, base(m, p, a)) : base(m, p, a)));
  await renderUnder(async ({ main }) => {
    await run(main);
    const btn = main.querySelector('button[type="submit"]');
    submit(main.querySelector("form"), btn);
    await flush();
    assert.ok(btn.classList.contains("is-busy"));
    release();
    await flush();
    assert.ok(!btn.classList.contains("is-busy"));
    const input = main.querySelector('[data-field="masking.after_ms"]');
    type(input, "abc");
    submit(main.querySelector("form"), btn);
    await flush();
    assert.equal(input.getAttribute("aria-invalid"), "true");
  });
});

test("anatomy: an empty acknowledgement pool shows an empty state", async () => {
  const { run } = await setup(ADMIN, maskingRoutes({ ...MASKING, ack_pool: [], problems: [] }));
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.ok(main.querySelector(".card .empty-state"));
    assert.equal(main.querySelector("table"), null);
    assertViewStructure(main);
  });
});

test("anatomy: with the prompt editor open (from prompts.js), the view still passes the style, class and structure checks", async () => {
  const { run } = await setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    main.querySelectorAll("button").find((b) => norm(b.textContent) === "Edit").click();
    await flush();
    assert.ok(main.querySelector("section.card .lang-pair"), "the editor card is open");
    assertNoInlineStyle(main);
    assertClassesStyled(main);
    assertViewStructure(main);
  });
});

// ---------------------------------------------------------------- PBT

const LANGS = oneOf(["en-US", "es-US"], ["en-US"], ["en-US", "es-US", "fr-FR"]);
const text = oneOf(null, unicodeString({ minLength: 1, maxLength: 8 }));
const userGen = oneOf(ADMIN, VIEWER, { groups: ["admin-editor"], email: "e@example.com", username: "e" });

const maskingGen = gen((rng) => {
  const langs = LANGS(rng);
  const masking = {};
  for (const [name, spec] of Object.entries(MASKING_FIELDS)) {
    if (int(0, 1)(rng)) masking[name] = spec.type === "enum" ? oneOf(...spec.values)(rng) : int(spec.range[0], spec.range[1])(rng);
  }
  return {
    rev: int(1, 99)(rng),
    masking: oneOf(masking, null)(rng),
    effective: { ...DEFAULTS, ...masking },
    defaults: DEFAULTS,
    ack_pool: array(gen((r) => ({
      key: `mask.ack${int(0, 1)(r) ? `.${int(1, 9)(r)}` : ""}${string({ maxLength: 3, chars: "abc" })(r)}`,
      texts: Object.fromEntries(langs.filter(() => int(0, 4)(r) > 0).map((l) => [l, text(r)])),
    })), { maxLength: 4 })(rng),
    problems: array(unicodeString({ minLength: 1, maxLength: 8 }), { maxLength: 3 })(rng),
    takes_effect_s: 60,
  };
});

test("PBT: the view for generated masking payloads", async () => {
  await forAll(record({ data: maskingGen, user: userGen }), async ({ data, user }) => {
    const api = fakeApi(() => structuredClone(data));
    await renderUnder(async ({ main }) => {
      await maskingView(main, { api, user }, {});
      assertNoInlineStyle(main);
      assertClassesStyled(main);
      assertViewStructure(main);
    });
  });
});
