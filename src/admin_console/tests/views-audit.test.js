// Audit log view.
// Characterization (written before the restyle, unchanged after it, apart from the one call
// line noted at runView): the search field and its known items, the item check's message, the
// Restore flow with its confirmation, and the API calls.
// Examples: a security change has the danger badge with text; each changed field is a dt
// with the field name and a dd with labelled Before and After values, nested values via show().
// PBT (no inline style, every class styled, view structure) over generated entries (with and without security_change, nested
// changes, restores, none).
import assert from "node:assert/strict";
import { test } from "node:test";

import { Event } from "./fakedom.js";
import { array, forAll, gen, int, json, oneOf, record, string, unicodeString } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder } from "./viewkit.js";
import { show } from "../lib/dom.js";
import { auditView } from "../views/audit.js";

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

const ITEM = "policy#tool#get_balance";
const ENTRIES = [
  {
    pk: ITEM, sk: "2026-10-01T10:00:00Z#1", ts: "2026-10-01T10:00:00Z", action: "update",
    actor: { email: "a@example.com" }, target: { kind: "tool", id: "get_balance" }, security_change: true,
    changes: [{ field: "min_level", old: 2, new: 3 }, { field: "mask", old: { after_ms: 800 }, new: null }],
  },
  {
    pk: ITEM, sk: "2026-09-30T09:00:00Z#2", ts: "2026-09-30T09:00:00Z", action: "update", restored_from: "x",
    actor: { sub: "u-1" }, target: { kind: "user", id: "bob" },
    changes: [{ field: "group", old: "admin-viewer", new: "admin" }],
  },
];

function auditRoutes(entries = ENTRIES) {
  return (method, path, arg) => {
    if (method === "get" && path === "tools") return { tools: [{ name: "get_balance" }] };
    if (method === "get" && path === "prompts") return { prompts: [{ key: `${arg.prefix}x` }] };
    if (method === "get" && path === "audit") return { entries: structuredClone(entries) };
    if (method === "get" && path === "tools/get_balance") return { tool: { rev: 7 } };
    if (method === "post" && path === "restore") return { changed: true, takes_effect_s: 60 };
    throw new Error(`unexpected ${method} ${path}`);
  };
}

/** The one line that follows the view's signature: (main, ctx, {params}). */
const runView = (main, ctx, params) => auditView(main, ctx, { params });

function setup(user, item, route = auditRoutes()) {
  const api = fakeApi(route);
  const ctx = { api, user };
  const params = new URLSearchParams(item === undefined ? "" : `item=${encodeURIComponent(item)}`);
  return { api, run: (main) => runView(main, ctx, params) };
}

/** location.hash, which the search form sets (fakedom has no location). */
function withLocation(t) {
  const before = Object.getOwnPropertyDescriptor(globalThis, "location");
  const loc = { hash: "" };
  Object.defineProperty(globalThis, "location", { value: loc, configurable: true, writable: true });
  t.after(() => {
    if (before) Object.defineProperty(globalThis, "location", before);
    else delete globalThis.location;
  });
  return loc;
}

const KNOWN = [["get", "tools"], ["get", "prompts", { prefix: "mask." }], ["get", "prompts", { prefix: "arg." }]];

// ---------------------------------------------------------------- characterization

test("characterization: with no item, the search field, its known items and no audit GET", async () => {
  const { api, run } = setup(ADMIN);
  await renderUnder(async ({ main, document }) => {
    await run(main);
    await flush();
    assert.deepEqual(api.calls, KNOWN);
    assert.deepEqual(controls(main), [
      { tag: "button", type: "submit", label: "Show changes" },
      { tag: "input", type: "text", field: "item", label: "Item" },
    ]);
    assert.equal(main.querySelector("input").getAttribute("list"), "audit-items");
    assert.deepEqual(document.getElementById("audit-items").querySelectorAll("option").map((o) => o.value),
      ["policy#masking", "policy#tool#get_balance", "catalog#mask.x", "catalog#arg.x"]);
  });
});

test("characterization: the item check's message; a good item goes to its route", async (t) => {
  const loc = withLocation(t);
  const { api, run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    const input = main.querySelector('[data-field="item"]');
    type(input, "nonsense");
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    await flush();
    assert.deepEqual(errorTexts(main), ["Check the item | choose an item such as policy#tool#get_balance"]);
    assert.equal(loc.hash, "");
    type(input, ` ${ITEM} `);
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    assert.equal(loc.hash, "#/audit?item=policy%23tool%23get_balance");
    assert.ok(!api.calls.some((c) => c[1] === "audit"));
  });
});

test("characterization: an item's entries: the audit GET, the who, and Restore only where restorable", async () => {
  const { api, run } = setup(ADMIN, ITEM);
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.deepEqual(api.calls, [...KNOWN, ["get", "audit", { item: ITEM, limit: 100 }]]);
    assert.equal(main.querySelector('[data-field="item"]').value, ITEM);
    assert.deepEqual(controls(main), [
      { tag: "button", type: "button", label: "Restore" },
      { tag: "button", type: "submit", label: "Show changes" },
      { tag: "input", type: "text", field: "item", label: "Item" },
    ]);
    const who = main.querySelectorAll("tbody tr").map((tr) => norm(tr.querySelectorAll("td")[2].textContent));
    assert.deepEqual(who, ["a@example.com", "u-1"]);
  });
  const viewer = setup(VIEWER, ITEM);
  await renderUnder(async ({ main }) => {
    await viewer.run(main);
    assert.ok(!controls(main).some((c) => c.label === "Restore"), "a viewer can't restore");
  });
});

test("characterization: Restore asks first, then reads the rev and POSTs the restore; Cancel sends nothing", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const { api, run } = setup(ADMIN, ITEM);
    await renderUnder(async ({ main, confirm }) => {
      await run(main);
      const n = api.calls.length;
      main.querySelectorAll("button").find((b) => norm(b.textContent) === "Restore").click();
      await flush();
      assert.equal(confirm.open, true);
      assert.equal(norm(confirm.querySelector(".dialog-title").textContent), "Restore the previous values?");
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent),
        ["min_level: 3 -> 2", `mask: (none) -> ${show({ after_ms: 800 })}`]);
      const buttons = confirm.querySelectorAll("button");
      buttons.find((b) => norm(b.textContent) === (answer === "confirm" ? "Restore" : "Cancel")).click();
      await flush();
      assert.deepEqual(api.calls.slice(n, n + 2), answer === "confirm"
        ? [["get", "tools/get_balance"], ["post", "restore", { audit_pk: ITEM, audit_sk: ENTRIES[0].sk, rev: 7 }]]
        : []);
      if (answer === "confirm") assert.deepEqual(api.calls.at(-1), ["get", "audit", { item: ITEM, limit: 100 }], "it reloads");
    });
  }
});

test("characterization: no entries says so", async () => {
  const { run } = setup(ADMIN, ITEM, auditRoutes([]));
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.match(main.textContent, /No changes recorded for this item\./);
    assert.equal(main.querySelector("table"), null);
  });
});

// ---------------------------------------------------------------- the restyled anatomy

test("a security change has the danger badge with text; changes are a dl with labelled Before and After", async () => {
  const { run } = setup(ADMIN, ITEM);
  await renderUnder(async ({ main }) => {
    await run(main);
    const [sec, restored] = main.querySelectorAll("tbody tr");
    assert.ok(sec.classList.contains("is-security"));
    const b = sec.querySelector(".badge--danger");
    assert.equal(norm(b.textContent), "Security change");
    assert.ok(b.querySelector("svg.icon"), "with the shield-alert icon");
    assert.ok(!restored.classList.contains("is-security"));
    assert.equal(norm(restored.querySelector(".badge--info").textContent), "restore");
    const dl = sec.querySelector("dl.changes");
    assert.deepEqual(dl.querySelectorAll("dt").map((dt) => norm(dt.textContent)), ["min_level", "mask"]);
    const dds = dl.querySelectorAll("dd");
    const part = (dd, cls) => [norm(dd.querySelector(`.${cls} .change-label`).textContent), dd.querySelector(`.${cls} .change-value`).textContent];
    assert.deepEqual(part(dds[0], "change-before"), ["Before", "2"]);
    assert.deepEqual(part(dds[0], "change-after"), ["After", "3"]);
    assert.deepEqual(part(dds[1], "change-before"), ["Before", show({ after_ms: 800 })]);
    assert.deepEqual(part(dds[1], "change-after"), ["After", "(none)"]);
    assert.ok(main.querySelector("button.sm"), "Restore is a small button");
  });
});

test("anatomy: page header, the search card on one row, and empty states for no item and no entries", async () => {
  for (const [item, route, title] of [[undefined, auditRoutes(), "Choose an item"], [ITEM, auditRoutes([]), "No changes recorded"]]) {
    const { run } = setup(ADMIN, item, route);
    await renderUnder(async ({ main }) => {
      await run(main);
      await flush();
      assert.equal(norm(main.querySelector("header.page-header h1").textContent), "Audit log");
      assert.ok(main.querySelector(".card form.search-row"));
      assert.equal(norm(main.querySelector(".empty-state-title").textContent), title);
      assertViewStructure(main);
    });
  }
});

test("anatomy: a bad item also marks the field invalid", async (t) => {
  withLocation(t);
  const { run } = setup(ADMIN);
  await renderUnder(async ({ main }) => {
    await run(main);
    await flush();
    const input = main.querySelector('[data-field="item"]');
    type(input, "nonsense");
    submit(main.querySelector("form"), main.querySelector('button[type="submit"]'));
    assert.equal(input.getAttribute("aria-invalid"), "true");
  });
});

// ---------------------------------------------------------------- PBT

const userGen = oneOf(ADMIN, VIEWER, { groups: ["admin-editor"], email: "e@example.com", username: "e" });
const value = json({ maxDepth: 2, maxLength: 3 });
const entryGen = record({
  pk: ITEM,
  sk: string({ minLength: 1, maxLength: 8 }),
  ts: oneOf("2026-10-01T10:00:00Z", unicodeString({ maxLength: 6 })),
  action: oneOf("update", "create", "delete", "restore"),
  restored_from: oneOf(undefined, "x"),
  actor: oneOf(null, { email: "a@example.com" }, { sub: "u-2" }, {}),
  target: oneOf(undefined, { kind: "tool", id: "t" }, { kind: "masking", id: "m" }, { kind: "prompt", id: "p" }, { kind: "user", id: "u" }),
  security_change: oneOf(true, false, undefined),
  changes: oneOf(undefined, array(record({ field: string({ minLength: 1, maxLength: 8 }), old: value, new: value }), { maxLength: 4 })),
});

test("PBT: the view for generated entries", async () => {
  await forAll(record({ entries: array(entryGen, { maxLength: 5 }), user: userGen, item: oneOf(undefined, ITEM) }),
    async ({ entries, user, item }) => {
      const { run } = setup(user, item, auditRoutes(entries));
      await renderUnder(async ({ main }) => {
        await run(main);
        await flush();
        assertNoInlineStyle(main);
        assertClassesStyled(main);
        assertViewStructure(main);
      });
    });
});
