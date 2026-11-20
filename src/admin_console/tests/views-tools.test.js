// Tools view: the list and the detail page.
// Characterization (written before the restyle, unchanged after it): the controls (tag, type,
// data-field, label, options, disabled), the links, the client-side validation messages and
// the API calls on load and save, through a recording fake API.
// PBT (no inline style, every class styled, view structure) through viewkit, over generated tools payloads (with and without
// mask, problems, own or default sources, empty lists).
import assert from "node:assert/strict";
import { test } from "node:test";

import { Event } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string, unicodeString } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder } from "./viewkit.js";
import { toolsView } from "../views/tools.js";

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

const TOOL = {
  name: "get_balance", kind: "read", min_level: 2, confirm: "none", confirm_source: "default", rev: 7,
  mask: {
    phrase_key: "mask.get_balance",
    after_ms: { value: 800, source: "tool" },
    handoff_wait_ms: { value: 1500, source: "default" },
    pool: [{ key: "mask.get_balance", texts: { "en-US": "One moment.", "es-US": null } }],
  },
  problems: ["phrase mask.get_balance.1 is missing es-US"],
  placeholders: [{ placeholder: "{account}" }],
};

const LIST = {
  takes_effect_s: 60,
  tools: [
    { ...TOOL },
    { name: "transfer", kind: "write", min_level: 3, confirm: "deterministic", confirm_source: "tool", mask: null, problems: [] },
  ],
};

function toolRoutes(tool = TOOL) {
  return (method, path, arg) => {
    if (method === "get" && path === "tools") return structuredClone(LIST);
    if (method === "get" && path === `tools/${tool.name}`) return { tool: structuredClone(tool), takes_effect_s: 60 };
    if (method === "put") return { changed: true, takes_effect_s: 60, tool: { ...tool, ...arg, rev: tool.rev + 1 } };
    throw new Error(`unexpected ${method} ${path}`);
  };
}

async function renderTools(user, arg, route = toolRoutes()) {
  const api = fakeApi(route);
  return { api, ctx: { api, user }, run: (main) => toolsView(main, { api, user }, { arg }) };
}

// ---------------------------------------------------------------- characterization

test("characterization: the list has the tool links and one GET /tools", async () => {
  const { api, run } = await renderTools(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.deepEqual(api.calls, [["get", "tools"]]);
    assert.deepEqual(controls(main), []);
    assert.deepEqual(links(main), [["#/tools/get_balance", "get_balance"], ["#/tools/transfer", "transfer"]]);
  });
});

const DETAIL_CONTROLS = [
  { tag: "button", type: "submit", label: "Save" },
  { tag: "input", type: "checkbox", label: "Mask this tool's wait with a phrase" },
  { tag: "input", type: "checkbox", label: "Use the policy-wide default" },
  { tag: "input", type: "checkbox", label: "Use the policy-wide default" },
  { tag: "input", type: "number", field: "tools.get_balance.mask.after_ms", label: "Phrase after (after_ms)" },
  { tag: "input", type: "number", field: "tools.get_balance.mask.handoff_wait_ms", label: "Hand-off wait (handoff_wait_ms)", disabled: true },
  { tag: "input", type: "number", field: "tools.get_balance.min_level", label: "Minimum verification level" },
  { tag: "input", type: "text", field: "tools.get_balance.mask.phrase_key", label: "Phrase key" },
  { tag: "select", field: "confirm", label: "Confirmation", options: ["", "none", "deterministic"] },
].sort(byJson);

const DETAIL_LINKS = [
  ["#/audit?item=policy%23tool%23get_balance", "History"],
  ["#/prompts?tool=get_balance", "Edit the phrases and preview them"],
  ["#/tools", "All tools"],
].sort(byJson);

test("characterization: the detail's controls, links and its one GET", async () => {
  const { api, run } = await renderTools(ADMIN, "get_balance");
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.deepEqual(api.calls, [["get", "tools/get_balance"]]);
    assert.deepEqual(controls(main), DETAIL_CONTROLS);
    assert.deepEqual(links(main), DETAIL_LINKS);
  });
});

test("characterization: a viewer sees every control disabled and no Save", async () => {
  const { run } = await renderTools(VIEWER, "get_balance");
  await renderUnder(async ({ main }) => {
    await run(main);
    const expected = DETAIL_CONTROLS.filter((c) => c.tag !== "button").map((c) => ({ ...c, disabled: true }));
    assert.deepEqual(controls(main), expected);
    assert.match(main.textContent, /Your group can view only\./);
  });
});

test("characterization: client-side validation messages, and no PUT", async () => {
  const { api, run } = await renderTools(ADMIN, "get_balance");
  await renderUnder(async ({ main }) => {
    await run(main);
    type(main.querySelector('[data-field="tools.get_balance.min_level"]'), "x");
    type(main.querySelector('[data-field="tools.get_balance.mask.phrase_key"]'), "bad key");
    type(main.querySelector('[data-field="tools.get_balance.mask.after_ms"]'), "20000");
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    await flush();
    assert.deepEqual(errorTexts(main), ["Check the form | min_level: must be a whole number, 0 or more; "
      + "mask.phrase_key: must be a catalog key starting with mask.; mask.after_ms: must be a whole number from 0 to 10000"]);
    assert.deepEqual(api.calls, [["get", "tools/get_balance"]]);
  });
});

test("characterization: a save that isn't a security change PUTs the body with the rev", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { api, run } = await renderTools(ADMIN, "get_balance");
  await renderUnder(async ({ main, confirm }) => {
    await run(main);
    type(main.querySelector('[data-field="tools.get_balance.mask.phrase_key"]'), "mask.balance");
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    await flush();
    assert.equal(confirm.open, false);
    assert.deepEqual(api.calls, [["get", "tools/get_balance"],
      ["put", "tools/get_balance", { rev: 7, min_level: 2, confirm: null, mask: { phrase_key: "mask.balance", after_ms: 800 } }]]);
  });
});

test("characterization: a security change asks first, with the danger variant; Cancel sends nothing", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const { api, run } = await renderTools(ADMIN, "get_balance");
    await renderUnder(async ({ main, confirm }) => {
      await run(main);
      type(main.querySelector('[data-field="tools.get_balance.min_level"]'), "3");
      main.querySelector('[data-field="confirm"]').value = "deterministic";
      submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
      await flush();
      assert.equal(confirm.open, true);
      assert.equal(norm(confirm.querySelector(".dialog-title").textContent), "This changes security behaviour");
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent),
        ["Minimum level: 2 -> 3", "Confirmation: default -> deterministic"]);
      const ok = confirm.querySelector("button.danger");
      assert.equal(norm(ok.textContent), "Save security change");
      (answer === "confirm" ? ok : confirm.querySelectorAll("button").find((b) => norm(b.textContent) === "Cancel")).click();
      await flush();
      const puts = api.calls.filter((c) => c[0] === "put");
      assert.deepEqual(puts, answer === "confirm"
        ? [["put", "tools/get_balance", { rev: 7, min_level: 3, confirm: "deterministic", mask: { phrase_key: "mask.get_balance", after_ms: 800 } }]]
        : []);
    });
  }
});

// ---------------------------------------------------------------- the restyled anatomy

test("detail: the page header holds the back link, the History link-button and Save for the form", async () => {
  const { run } = await renderTools(ADMIN, "get_balance");
  await renderUnder(async ({ main }) => {
    await run(main);
    const header = main.querySelector("header.page-header");
    assert.equal(norm(header.querySelector("h1#page-title").textContent), "get_balance");
    assert.equal(header.querySelector("a.back-link").getAttribute("href"), "#/tools");
    assert.ok(header.querySelector("a.button.quiet svg.icon"), "History is a quiet link-button with an icon");
    const saveBtn = header.querySelector("button.primary");
    assert.equal(saveBtn.getAttribute("form"), "tool-form");
    assert.equal(main.querySelector("form").id, "tool-form");
    assert.ok(main.querySelector(".panel--warning"), "the problems are a warning panel");
    assert.ok(main.querySelector('input.switch[role="switch"]'), "masked is a switch");
    assert.ok(main.querySelector(".lang-badge"), "language lines use langBadge");
  });
});

test("detail: Save passes ev.submitter to save(), so the button is busy while saving", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  let release;
  const pending = new Promise((r) => { release = r; });
  const base = toolRoutes();
  const { run } = await renderTools(ADMIN, "get_balance", async (m, p, a) => (m === "put" ? (await pending, base(m, p, a)) : base(m, p, a)));
  await renderUnder(async ({ main }) => {
    await run(main);
    const btn = main.querySelector('button[type="submit"]');
    submit(main.querySelector("form"), btn);
    await flush();
    assert.equal(btn.getAttribute("aria-disabled"), "true");
    assert.ok(btn.classList.contains("is-busy"));
    release();
    await flush();
    assert.equal(btn.getAttribute("aria-disabled"), null);
  });
});

test("detail: a client-side error also marks each named field invalid", async () => {
  const { run } = await renderTools(ADMIN, "get_balance");
  await renderUnder(async ({ main }) => {
    await run(main);
    const lvl = main.querySelector('[data-field="tools.get_balance.min_level"]');
    type(lvl, "x");
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    await flush();
    assert.equal(lvl.getAttribute("aria-invalid"), "true");
  });
});

test("list: the table is in a flush card with .num columns, badges, and an empty state when there are no tools", async () => {
  const { run } = await renderTools(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.ok(main.querySelector(".card.flush .table-wrap table"));
    assert.equal(main.querySelectorAll("th.num").length, 3);
    assert.ok(main.querySelector(".badge--warning"), "problems are a warning badge");
    assert.ok(main.querySelector(".badge--success"), "masking on is a success badge");
  });
  const empty = fakeApi(() => ({ tools: [], takes_effect_s: 60 }));
  await renderUnder(async ({ main }) => {
    await toolsView(main, { api: empty, user: ADMIN }, {});
    assert.equal(norm(main.querySelector(".empty-state-title").textContent), "No tools");
    assertViewStructure(main);
  });
});

// ---------------------------------------------------------------- PBT

const LANGS = oneOf(["en-US", "es-US"], ["en-US"], ["en-US", "es-US", "fr-FR"]);
const toolName = string({ minLength: 1, maxLength: 10, chars: "abcdefghij_" });
const delay = oneOf(null, record({ value: int(0, 10000), source: oneOf("tool", "default") }));
const text = oneOf(null, unicodeString({ minLength: 1, maxLength: 8 }));

const toolGen = gen((rng) => {
  const langs = LANGS(rng);
  const name = toolName(rng);
  const pool = array(gen((r) => ({
    key: `mask.${name}${int(0, 3)(r) ? `.${int(1, 3)(r)}` : ""}`,
    texts: Object.fromEntries(langs.map((l) => [l, text(r)])),
  })), { maxLength: 3 })(rng);
  return record({
    name,
    kind: oneOf("read", "write", null),
    min_level: oneOf(int(0, 3), null),
    confirm: oneOf("none", "deterministic", null),
    confirm_source: oneOf("tool", "default"),
    rev: int(1, 50),
    mask: oneOf(null, record({ phrase_key: `mask.${name}`, after_ms: delay, handoff_wait_ms: delay, pool })),
    problems: array(unicodeString({ minLength: 1, maxLength: 8 }), { maxLength: 3 }),
    placeholders: array(record({ placeholder: oneOf("{account}", "{amount}", "{when}") }), { maxLength: 2 }),
  })(rng);
});
const userGen = oneOf(ADMIN, VIEWER, { groups: ["admin-editor"], email: "e@example.com", username: "e" });

function checkView(main) {
  assertNoInlineStyle(main);
  assertClassesStyled(main);
  assertViewStructure(main);
}

test("PBT: the list for generated tools", async () => {
  await forAll(record({ tools: array(toolGen, { maxLength: 5 }), user: userGen }), async ({ tools, user }) => {
    const api = fakeApi(() => ({ tools, takes_effect_s: 60 }));
    await renderUnder(async ({ main }) => {
      await toolsView(main, { api, user }, {});
      checkView(main);
    });
  });
});

test("PBT: the detail for a generated tool", async () => {
  await forAll(record({ tool: toolGen, user: userGen }), async ({ tool, user }) => {
    const api = fakeApi(() => ({ tool: structuredClone(tool), takes_effect_s: 60 }));
    await renderUnder(async ({ main }) => {
      await toolsView(main, { api, user }, { arg: tool.name });
      checkView(main);
    });
  });
});
