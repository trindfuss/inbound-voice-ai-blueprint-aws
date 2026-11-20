// Users view: admins manage console users.
// Characterization (written before the restyle, unchanged after it): the row actions and their
// confirmations, the create form and its message, the last-admin and self-delete rules as the
// view shows them, and the API calls.
// PBT (no inline style, every class styled, view structure) over generated user lists in every status and group, including the
// current user.
import assert from "node:assert/strict";
import { test } from "node:test";

import { Event } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder } from "./viewkit.js";
import { GROUPS } from "../lib/validate.js";
import { usersView } from "../views/users.js";

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

const USERS = {
  groups: GROUPS,
  users: [
    { username: "admin", email: "admin@example.com", group: "admin", status: "CONFIRMED", enabled: true, created: "2026-01-01T00:00:00Z" },
    { username: "bob", email: "bob@example.com", group: "admin-viewer", status: "FORCE_CHANGE_PASSWORD", enabled: true },
    { username: "carol", group: "", status: "CONFIRMED", enabled: false },
  ],
};

function userRoutes({ fail } = {}) {
  return (method, path) => {
    if (method === "get" && path === "users") return structuredClone(USERS);
    if (fail && method === fail) throw Object.assign(new Error("The console must keep at least one enabled admin."), { kind: "invalid", title: "Not saved" });
    if (["patch", "post", "del"].includes(method)) return { changed: true };
    throw new Error(`unexpected ${method} ${path}`);
  };
}

function setup(route = userRoutes()) {
  const api = fakeApi(route);
  return { api, run: (main) => usersView(main, { api, user: ADMIN }, {}) };
}

const named = (root, name) => root.querySelectorAll("button").find((b) => norm(accText(b)) === name);
const rowOf = (main, who) => main.querySelectorAll("tbody tr").find((tr) => norm(accText(tr.querySelector("th"))).startsWith(who));

async function answer(confirm, label) {
  await flush();
  assert.equal(confirm.open, true, "the confirmation is open");
  const title = norm(confirm.querySelector(".dialog-title").textContent);
  const message = norm(confirm.querySelector(".dialog-body p").textContent);
  const ok = confirm.querySelectorAll(".dialog-footer button").at(-1);
  const danger = ok.classList.contains("danger");
  confirm.querySelectorAll("button").find((b) => norm(b.textContent) === label).click();
  await flush();
  return { title, message, danger };
}

// ---------------------------------------------------------------- characterization

test("characterization: the row controls, the create form, the rules sentence and one GET /users", async () => {
  const { api, run } = setup();
  await renderUnder(async ({ main }) => {
    await run(main);
    assert.deepEqual(api.calls, [["get", "users"]]);
    const sel = (who) => ({ tag: "select", label: `Group of ${who}`, options: GROUPS });
    const btn = (label, type = "button") => ({ tag: "button", type, label });
    assert.deepEqual(controls(main), [
      sel("admin@example.com"), btn("Disable"),
      sel("bob@example.com"), btn("Disable"), btn("Resend invitation"), btn("Delete"),
      sel("carol"), btn("Enable"), btn("Delete"),
      { tag: "input", type: "email", field: "email", label: "Email" },
      { tag: "select", field: "group", label: "Group", options: GROUPS },
      btn("Create user", "submit"),
    ].sort(byJson));
    assert.equal(main.querySelector('[data-field="group"]').value, "admin-viewer");
    assert.match(main.textContent, /The console keeps at least one enabled admin and doesn't let you delete yourself\./);
    assert.match(main.textContent, /Cognito emails them a temporary password; they set their own at first sign-in\./);
    assert.equal(named(rowOf(main, "admin@example.com"), "Delete"), undefined, "no Delete for yourself");
  });
});

test("characterization: each row action asks first, then calls the API and reloads", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const cases = [
    ["Disable", "Disable", ["patch", "users/bob", { enabled: false }],
      { title: "Disable user?", message: "bob@example.com won't be able to sign in until enabled again.", danger: true }],
    ["Resend invitation", "Resend", ["post", "users/bob/resend"],
      { title: "Resend the invitation?", message: "Cognito emails bob@example.com a new temporary password.", danger: false }],
    ["Delete", "Delete", ["del", "users/bob"],
      { title: "Delete user?", message: "bob@example.com is removed from the console for good. This can't be undone.", danger: true }],
  ];
  for (const [label, confirmLabel, call, dialog] of cases) {
    const { api, run } = setup();
    await renderUnder(async ({ main, confirm }) => {
      await run(main);
      named(rowOf(main, "bob@example.com"), label).click();
      assert.deepEqual(await answer(confirm, "Cancel"), dialog);
      assert.equal(api.calls.length, 1, `${label}: Cancel sends nothing`);
      named(rowOf(main, "bob@example.com"), label).click();
      await answer(confirm, confirmLabel);
      assert.deepEqual(api.calls.slice(1), [call, ["get", "users"]]);
    });
  }
});

test("characterization: a group change reverts the select, asks (danger for admin), then PATCHes", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { api, run } = setup();
  await renderUnder(async ({ main, confirm }) => {
    await run(main);
    const sel = rowOf(main, "bob@example.com").querySelector("select");
    sel.value = "admin";
    sel.dispatchEvent(new Event("change", { bubbles: true }));
    assert.equal(sel.value, "admin-viewer", "the select shows the current group until it's saved");
    assert.deepEqual(await answer(confirm, "Change group"),
      { title: "Change group?", message: "bob@example.com: admin-viewer -> admin. This changes what they can do.", danger: true });
    assert.deepEqual(api.calls.slice(1), [["patch", "users/bob", { group: "admin" }], ["get", "users"]]);
  });
});

test("characterization: the API's last-admin refusal shows as an error panel, and nothing reloads", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { api, run } = setup(userRoutes({ fail: "patch" }));
  await renderUnder(async ({ main, confirm }) => {
    await run(main);
    named(rowOf(main, "admin@example.com"), "Disable").click();
    await answer(confirm, "Disable");
    assert.deepEqual(errorTexts(main), ["Not saved | The console must keep at least one enabled admin."]);
    assert.deepEqual(api.calls.map((c) => c[0]), ["get", "patch"]);
  });
});

test("characterization: the create form's message, then a confirmed POST /users", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { api, run } = setup();
  await renderUnder(async ({ main, confirm }) => {
    await run(main);
    const form = main.querySelector('[data-field="email"]').closest("form");
    type(main.querySelector('[data-field="email"]'), "not-an-email");
    submit(form, named(main, "Create user"));
    await flush();
    assert.deepEqual(errorTexts(main), ["Check the form | email: must be an email address"]);
    assert.equal(api.calls.length, 1);
    type(main.querySelector('[data-field="email"]'), " new@example.com ");
    main.querySelector('[data-field="group"]').value = "admin";
    submit(form, named(main, "Create user"));
    assert.deepEqual(await answer(confirm, "Create"),
      { title: "Create user?", message: "new@example.com joins as admin.", danger: true });
    assert.deepEqual(api.calls.slice(1), [["post", "users", { email: "new@example.com", group: "admin" }], ["get", "users"]]);
  });
});

// ---------------------------------------------------------------- the restyled anatomy

test("anatomy: header with no action, the table card, badges, small row buttons, the Add a user card", async () => {
  const { run } = setup();
  await renderUnder(async ({ main }) => {
    await run(main);
    const header = main.querySelector("header.page-header");
    assert.equal(norm(header.querySelector("h1").textContent), "Users");
    assert.equal(header.querySelector("button"), null, "the view has no single main action");
    assert.ok(main.querySelector(".card.flush .table-wrap table"));
    const badges = (who) => rowOf(main, who).querySelectorAll(".badge").map((b) => [norm(b.textContent), b.className.split(" ")[1]]);
    assert.deepEqual(badges("admin@example.com"), [["you", "badge--brand"], ["enabled", "badge--success"]]);
    assert.deepEqual(badges("bob@example.com"), [["invited", "badge--warning"], ["enabled", "badge--success"]]);
    assert.deepEqual(badges("carol"), [["disabled", "badge--danger"]]);
    for (const b of main.querySelectorAll("tbody button")) assert.ok(b.classList.contains("sm"));
    assert.ok(named(rowOf(main, "bob@example.com"), "Delete").classList.contains("danger"));
    const create = named(main, "Create user");
    assert.ok(create.classList.contains("primary") && create.closest(".card"));
  });
});

test("anatomy: a row action and Create user pass their button to save(), so it is busy while saving", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  let release;
  const pending = new Promise((r) => { release = r; });
  const base = userRoutes();
  const { run } = setup(async (m, p, a) => (m === "get" ? base(m, p, a) : (await pending, base(m, p, a))));
  await renderUnder(async ({ main, confirm }) => {
    await run(main);
    const btn = named(rowOf(main, "bob@example.com"), "Disable");
    btn.click();
    await answer(confirm, "Disable");
    assert.ok(btn.classList.contains("is-busy"));
    const create = named(main, "Create user");
    type(main.querySelector('[data-field="email"]'), "new@example.com");
    submit(create.closest("form"), create);
    await answer(confirm, "Create");
    assert.ok(create.classList.contains("is-busy"));
    release();
    await flush();
  });
});

test("anatomy: a bad email also marks the field invalid", async () => {
  const { run } = setup();
  await renderUnder(async ({ main }) => {
    await run(main);
    const email = main.querySelector('[data-field="email"]');
    type(email, "x");
    submit(email.closest("form"), named(main, "Create user"));
    await flush();
    assert.equal(email.getAttribute("aria-invalid"), "true");
  });
});

// ---------------------------------------------------------------- PBT

const userRow = record({
  username: string({ minLength: 1, maxLength: 8 }),
  email: oneOf(undefined, null, "x@example.com", "admin@example.com"),
  group: oneOf(...GROUPS, "", undefined),
  status: oneOf("CONFIRMED", "FORCE_CHANGE_PASSWORD", "RESET_REQUIRED", undefined, ""),
  enabled: oneOf(true, false),
  created: oneOf(undefined, "2026-01-01T00:00:00Z"),
}, { optional: ["email", "created"] });

const listGen = gen((rng) => {
  const users = array(userRow, { maxLength: 6 })(rng);
  if (int(0, 1)(rng)) users.push({ ...ADMIN, group: "admin", status: "CONFIRMED", enabled: true });
  return { users, groups: oneOf(undefined, GROUPS)(rng) };
});

test("PBT: the view for generated user lists", async () => {
  await forAll(listGen, async (data) => {
    const api = fakeApi(() => structuredClone(data));
    await renderUnder(async ({ main }) => {
      await usersView(main, { api, user: ADMIN }, {});
      assertNoInlineStyle(main);
      assertClassesStyled(main);
      assertViewStructure(main);
    });
  });
});
