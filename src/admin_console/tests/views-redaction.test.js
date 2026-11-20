// Redaction view. Rendered
// under fakedom with a recording fake API:
//   admin only: no nav entry for a non-admin, and the route shows the denied empty state with no API
//   call; the controls reflect a GET /redaction payload (the enabled switch, the entity types as
//   checkboxes grouped by type, each group heading selecting or clearing its group and shown
//   indeterminate when mixed, min_confidence, mask and on_failure); no type selected is valid and
//   saves as enabled: false with entity_types: [] (the server refuses enabled: true with no types),
//   shown as off: the switch off and "Redaction is off (no entity types selected)";
//   switching off, removing a type, raising min_confidence and setting on_failure to allow each show
//   a warning and a danger confirmDialog before saving, and Cancel sends nothing; a 409
//   shows the compare cards; a 400 marks the named field.
// PBT (gen.js): the checkboxes equal entity_types; the displayed state is
// off iff enabled is false or the list is empty; the PUT body equals the edited config (and is never
// one the server refuses), and the view repainted from the save shows what was shown before it (every
// displayed state round-trips); the style, class and structure checks through viewkit.
import assert from "node:assert/strict";
import { test } from "node:test";

import { Event } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string, unicodeString } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder } from "./viewkit.js";
import { SECTIONS, render } from "../app.js";
import { ApiError, mapError } from "../lib/api.js";
import { renderNav } from "../lib/shell.js";
import {
  ALLOW_WARNING, CONFIDENCE_WARNING, ENTITY_GROUPS, OFF_NO_TYPES, OFF_WARNING, REMOVED_WARNING, groupTypes,
  redactionView,
} from "../views/redaction.js";

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

const puts = (api) => api.calls.filter((c) => c[0] === "put");

const ADMIN = { groups: ["admin"], email: "admin@example.com", username: "admin" };
const EDITOR = { groups: ["admin-editor"], email: "e@example.com", username: "e" };
const VIEWER = { groups: ["admin-viewer"], email: "viewer@example.com", username: "viewer" };

/** pii.COMPREHEND_PII_ENTITY_TYPES, sorted, as GET /redaction sends it. */
const AVAILABLE = [
  "ADDRESS", "AGE", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "BANK_ACCOUNT_NUMBER", "BANK_ROUTING",
  "CA_HEALTH_NUMBER", "CA_SOCIAL_INSURANCE_NUMBER", "CREDIT_DEBIT_CVV", "CREDIT_DEBIT_EXPIRY",
  "CREDIT_DEBIT_NUMBER", "DATE_TIME", "DRIVER_ID", "EMAIL", "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
  "IN_AADHAAR", "IN_NREGA", "IN_PERMANENT_ACCOUNT_NUMBER", "IN_VOTER_NUMBER", "IP_ADDRESS",
  "LICENSE_PLATE", "MAC_ADDRESS", "NAME", "PASSPORT_NUMBER", "PASSWORD", "PHONE", "PIN", "SSN",
  "SWIFT_CODE", "UK_NATIONAL_HEALTH_SERVICE_NUMBER", "UK_NATIONAL_INSURANCE_NUMBER",
  "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER", "URL", "USERNAME",
  "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "VEHICLE_IDENTIFICATION_NUMBER",
];
const DEFAULT_TYPES = ["SSN", "CREDIT_DEBIT_NUMBER", "CREDIT_DEBIT_CVV", "CREDIT_DEBIT_EXPIRY",
  "BANK_ACCOUNT_NUMBER", "BANK_ROUTING", "PIN", "PASSWORD"];
const DEFAULTS = { enabled: true, entity_types: DEFAULT_TYPES, min_confidence: "0.5", mask: "[REDACTED:{type}]", on_failure: "withhold" };

const SERVER_OFF = { code: "redaction_off", message: "Transcript redaction is off: server text." };
const SERVER_REMOVED = (types) => ({ code: "entity_type_removed", message: "Removing an entity type: server text.", entity_types: types });

/** A GET /redaction view (redaction_routes._view) for a stored config. */
function viewOf(cfg, { rev = 4, problems = [], effective } = {}) {
  const eff = effective ?? cfg;
  return {
    redaction: cfg ?? {}, effective: structuredClone(eff), defaults: DEFAULTS, problems, rev,
    entity_types_available: AVAILABLE, on_failure_values: ["allow", "withhold"], takes_effect_s: 60,
    warnings: eff.enabled ? [] : [SERVER_OFF],
  };
}

const ON = { enabled: true, entity_types: ["SSN", "PIN", "NAME"], min_confidence: "0.5", mask: "[REDACTED:{type}]", on_failure: "withhold" };
const PAYLOAD = viewOf(ON);

/**
 * GET /redaction answers `payload`; PUT acts like redaction_routes.put_redaction: the body over
 * the stored fields, refused (400) when enabled with no types, rev checked and bumped.
 */
function redactionRoutes(payload = PAYLOAD) {
  let view = structuredClone(payload);
  return (method, path, arg) => {
    if (method === "get" && path === "redaction") return structuredClone(view);
    if (method === "put" && path === "redaction") {
      const { rev, ...fields } = arg;
      if (rev !== view.rev) throw new ApiError(mapError(409, { error: "conflict", message: "changed", current: view }));
      const next = { ...view.effective, ...fields };
      if (next.enabled && !next.entity_types.length) {
        throw new ApiError(mapError(400, { error: "invalid", message: "must not be empty while enabled", field: "redaction.entity_types" }));
      }
      next.min_confidence = String(Number(next.min_confidence));
      const removed = view.effective.entity_types.filter((t) => !next.entity_types.includes(t)).sort();
      const before = view;
      view = viewOf(next, { rev: rev + 1 });
      const warnings = [...(next.enabled ? [] : [SERVER_OFF]), ...(removed.length ? [SERVER_REMOVED(removed)] : [])];
      const changed = JSON.stringify(before.effective) !== JSON.stringify(next);
      return { ...structuredClone(view), changed, security_change: changed, audit: { pk: "policy#redaction", sk: "s" }, warnings };
    }
    throw new Error(`unexpected ${method} ${path}`);
  };
}

const run = (api, user = ADMIN) => (main) => redactionView(main, { api, user });

const enabledSwitch = (main) => main.querySelector('input[data-field="redaction.enabled"]');
const typeBox = (main, t) => main.querySelector(`input[data-entity-type="${t}"]`);
const typeBoxes = (main) => main.querySelectorAll("input[data-entity-type]");
const checkedTypes = (main) => typeBoxes(main).filter((b) => b.checked).map((b) => b.getAttribute("data-entity-type"));
const groupBox = (main, g) => main.querySelector(`input[data-entity-group="${g}"]`);
const groupOf = (t) => ENTITY_GROUPS.find((g) => g.types.includes(t))?.name ?? "Other";
const form = (main) => main.querySelector("form[data-form=redaction]");
const fieldOf = (main, f) => main.querySelector(`[data-field="redaction.${f}"]`);

function toggle(input) {
  input.checked = !input.checked;
  input.indeterminate = false; // as a browser does on a click
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function setValue(el, value) {
  el.value = value;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

function submit(f) {
  const ev = new Event("submit", { bubbles: true, cancelable: true });
  ev.submitter = f.querySelector('button[type="submit"]');
  f.dispatchEvent(ev);
}

const confirmButton = (dlg) => dlg.querySelectorAll(".dialog-footer button").find((b) => norm(b.textContent) !== "Cancel");
const cancelButton = (dlg) => dlg.querySelectorAll("button").find((b) => norm(b.textContent) === "Cancel");
const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const stateOf = (main) => main.querySelector("[data-redaction-state]");

/** What the form shows: the switch, the state line, the ticked types and the detection fields. */
function displayed(main) {
  return {
    switch: enabledSwitch(main).checked,
    switchText: norm(enabledSwitch(main).closest("label").textContent),
    state: stateOf(main).getAttribute("data-redaction-state"),
    stateText: norm(stateOf(main).textContent),
    types: checkedTypes(main).sort(),
    min_confidence: Number(fieldOf(main, "min_confidence").value),
    mask: fieldOf(main, "mask").value,
    on_failure: fieldOf(main, "on_failure").value,
  };
}

// ---------------------------------------------------------------- the registry and admin-only

test("SECTIONS has one redaction entry: the reserved redaction icon, admins only, with denied text", () => {
  const found = SECTIONS.filter((s) => s.id === "redaction");
  assert.equal(found.length, 1);
  const [s] = found;
  assert.equal(s.label, "Redaction");
  assert.equal(s.icon, "redaction");
  assert.equal(s.render, redactionView);
  assert.equal(s.when(ADMIN), true);
  assert.equal(s.when(EDITOR), false);
  assert.equal(s.when(VIEWER), false);
  assert.equal(typeof s.denied, "string");
  assert.ok(s.denied.trim().length > 0);
});

test("a non-admin sees no Redaction nav entry, and the route shows the denied empty state with no API call", async () => {
  for (const user of [EDITOR, VIEWER, ADMIN]) {
    const api = fakeApi(redactionRoutes());
    await renderUnder(async ({ document, main }) => {
      const nav = document.createElement("nav");
      document.body.prepend(nav);
      await render({ api, user }, { main, nav, hash: "#/redaction" });
      await flush();
      const hrefs = nav.querySelectorAll("a").map((a) => a.getAttribute("href"));
      const admin = user === ADMIN;
      assert.equal(hrefs.includes("#/redaction"), admin, `nav entry for ${user.username}`);
      if (admin) {
        assert.deepEqual(api.calls, [["get", "redaction"]]);
        assert.ok(form(main));
      } else {
        assert.deepEqual(api.calls, [], "no API call for a section the user can't see");
        const denied = SECTIONS.find((s) => s.id === "redaction").denied;
        assert.equal(norm(main.querySelector(".empty-state .empty-state-title").textContent), denied);
        assert.equal(main.querySelectorAll("input, select, textarea").length, 0);
      }
      assertViewStructure(main);
    });
  }
  // The nav alone, through the real registry.
  await renderUnder(async ({ document }) => {
    const nav = document.createElement("nav");
    renderNav(nav, SECTIONS, "tools", EDITOR);
    assert.equal(nav.querySelectorAll('a[href="#/redaction"]').length, 0);
    renderNav(nav, SECTIONS, "tools", ADMIN);
    assert.equal(nav.querySelectorAll('a[href="#/redaction"]').length, 1);
  });
});

// ---------------------------------------------------------------- rendering

test("groupTypes: every available type in exactly one group, in ENTITY_GROUPS order, unknown types under Other", () => {
  const groups = groupTypes(AVAILABLE, []);
  const all = groups.flatMap((g) => g.types);
  assert.deepEqual([...all].sort(), [...AVAILABLE].sort());
  assert.equal(new Set(all).size, all.length);
  assert.equal(groups.some((g) => g.name === "Other"), false, "every Comprehend type has a named group");
  for (const g of groups) assert.ok(g.types.length > 0, `${g.name} isn't empty`);
  const extra = groupTypes(["SSN"], ["NEW_TYPE", "SSN"]);
  assert.deepEqual(extra.map((g) => [g.name, g.types]), [[groupOf("SSN"), ["SSN"]], ["Other", ["NEW_TYPE"]]]);
});

test("the controls reflect the payload: switch, grouped checkboxes, min_confidence, mask, on_failure", async () => {
  const api = fakeApi(redactionRoutes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.deepEqual(api.calls, [["get", "redaction"]]);
    assert.equal(norm(main.querySelector("h1#page-title").textContent), "Redaction");
    const sw = enabledSwitch(main);
    assert.equal(sw.getAttribute("role"), "switch");
    assert.ok(sw.classList.contains("switch"));
    assert.equal(sw.checked, true);
    assert.match(norm(sw.closest("label").textContent), /On/);
    assert.deepEqual(checkedTypes(main).sort(), ["NAME", "PIN", "SSN"]);
    assert.deepEqual(typeBoxes(main).map((b) => b.getAttribute("data-entity-type")).sort(), [...AVAILABLE].sort());
    // Each type sits in its group's fieldset, whose legend holds the group heading checkbox.
    for (const b of typeBoxes(main)) {
      const t = b.getAttribute("data-entity-type");
      const fs = b.closest("fieldset[data-group]");
      assert.equal(fs.getAttribute("data-group"), groupOf(t));
      assert.ok(fs.querySelector("legend").contains(groupBox(main, groupOf(t))));
    }
    const credentials = groupBox(main, groupOf("PIN"));
    assert.equal(credentials.checked, false);
    assert.equal(credentials.indeterminate, true, "a mixed group is indeterminate");
    assert.match(norm(credentials.closest("label").textContent), /1 of \d+/, "a visible count, not the box alone");
    const vehicle = groupBox(main, groupOf("LICENSE_PLATE"));
    assert.equal(vehicle.checked, false);
    assert.equal(vehicle.indeterminate, false);
    assert.equal(fieldOf(main, "min_confidence").value, "0.5");
    assert.equal(fieldOf(main, "mask").value, "[REDACTED:{type}]");
    const sel = fieldOf(main, "on_failure");
    assert.equal(sel.value, "withhold");
    assert.deepEqual(sel.querySelectorAll("option").map((o) => o.value), ["allow", "withhold"]);
    assert.ok(main.querySelector('a[href="#/audit?item=policy%23redaction"]'), "the History link");
    assert.ok(main.querySelector(".panel--info"), "the takes-effect note");
  });
});

test("a full group shows its heading checked; the heading selects and clears its whole group", async () => {
  const api = fakeApi(redactionRoutes(viewOf({ ...ON, entity_types: ["IP_ADDRESS", "MAC_ADDRESS", "URL", "SSN"] })));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const net = groupBox(main, groupOf("URL"));
    assert.equal(net.checked, true);
    assert.equal(net.indeterminate, false);
    toggle(net); // clear the group
    for (const t of ["IP_ADDRESS", "MAC_ADDRESS", "URL"]) assert.equal(typeBox(main, t).checked, false);
    assert.equal(net.checked, false);
    assert.equal(net.indeterminate, false);
    assert.equal(typeBox(main, "SSN").checked, true, "other groups untouched");
    // A mixed group's heading selects all of it.
    const fin = groupBox(main, groupOf("SWIFT_CODE"));
    toggle(typeBox(main, "SWIFT_CODE"));
    assert.equal(fin.indeterminate, true);
    toggle(fin);
    const finTypes = ENTITY_GROUPS.find((g) => g.name === groupOf("SWIFT_CODE")).types;
    for (const t of finTypes) assert.equal(typeBox(main, t).checked, true, t);
    assert.equal(fin.checked, true);
    assert.equal(fin.indeterminate, false);
    // Unticking the last box of a group from inside it updates the heading.
    toggle(typeBox(main, "SWIFT_CODE"));
    assert.equal(fin.checked, false);
    assert.equal(fin.indeterminate, true);
    assert.deepEqual(puts(api), [], "nothing is sent until Save");
  });
});

test("a stored (enabled: false, entity_types: []) loads with the switch off: \"Redaction is off (no entity types selected)\"", async () => {
  assert.equal(OFF_NO_TYPES, "Redaction is off (no entity types selected)");
  const api = fakeApi(redactionRoutes(viewOf({ ...ON, enabled: false, entity_types: [] })));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.equal(enabledSwitch(main).checked, false, "off, not \"on with nothing selected\"");
    assert.match(norm(enabledSwitch(main).closest("label").textContent), /Off/);
    assert.deepEqual(checkedTypes(main), []);
    assert.equal(stateOf(main).getAttribute("data-redaction-state"), "off");
    assert.equal(norm(stateOf(main).textContent), OFF_NO_TYPES);
    assert.equal(norm(main.querySelector("[data-pending-warnings]").textContent), "", "no edit, no warning");
    // Saving it unchanged sends nothing.
    submit(form(main));
    await flush();
    assert.deepEqual(puts(api), []);
  });
});

test("a stored (enabled: false) with types shows the switch off and the types kept", async () => {
  const api = fakeApi(redactionRoutes(viewOf({ ...ON, enabled: false })));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.equal(enabledSwitch(main).checked, false);
    assert.match(norm(enabledSwitch(main).closest("label").textContent), /Off/);
    assert.deepEqual(checkedTypes(main).sort(), ["NAME", "PIN", "SSN"]);
    assert.equal(main.querySelector("[data-redaction-state]").getAttribute("data-redaction-state"), "off");
    assert.match(norm(main.querySelector("[data-server-warnings]").textContent), /server text/, "the GET's redaction_off warning");
  });
});

test("a missing or invalid item shows its problems and the defaults the components run with", async () => {
  const payload = viewOf({ enabled: "yes" }, { rev: 2, effective: DEFAULTS,
    problems: [{ field: "redaction.enabled", message: "redaction.enabled: must be true or false" }] });
  const api = fakeApi(redactionRoutes(payload));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const problems = main.querySelector("[data-problems]");
    assert.ok(problems.classList.contains("panel--warning"));
    assert.match(norm(problems.textContent), /redaction\.enabled: must be true or false/);
    assert.deepEqual(checkedTypes(main).sort(), [...DEFAULT_TYPES].sort());
  });
});

test("a 503 on load shows the error panel under the page header", async () => {
  const api = fakeApi(() => { throw new ApiError(mapError(503, { error: "not_seeded" })); });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.ok(main.querySelector(".panel--error"));
    assertViewStructure(main);
  });
});

// ---------------------------------------------------------------- saving

test("a tightening change saves at once with the whole edited config and rev; the response repaints", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(redactionRoutes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    toggle(typeBox(main, "EMAIL"));
    setValue(fieldOf(main, "min_confidence"), "0.4");
    setValue(fieldOf(main, "on_failure"), "withhold");
    submit(form(main));
    await flush();
    assert.equal(confirm.open, false, "adding a type asks nothing");
    assert.deepEqual(puts(api), [["put", "redaction", {
      rev: 4, enabled: true, entity_types: ["SSN", "PIN", "NAME", "EMAIL"], min_confidence: "0.4",
      mask: "[REDACTED:{type}]", on_failure: "withhold",
    }]]);
    assert.ok(typeBox(main, "EMAIL").checked);
    // The next save carries the response's rev.
    setValue(fieldOf(main, "mask"), "[{type}]");
    submit(form(main));
    await flush();
    assert.equal(puts(api)[1][2].rev, 5);
    assert.equal(puts(api)[1][2].mask, "[{type}]");
  });
});

test("nothing changed sends nothing", async () => {
  const api = fakeApi(redactionRoutes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    submit(form(main));
    await flush();
    assert.deepEqual(puts(api), []);
    assert.match(norm(main.textContent), /Nothing to save/);
  });
});

test("switching off shows a warning, then a danger confirmDialog; Cancel sends nothing, Save sends enabled: false", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const api = fakeApi(redactionRoutes());
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      toggle(enabledSwitch(main));
      const pending = main.querySelector("[data-pending-warnings]");
      assert.match(norm(pending.textContent), new RegExp(OFF_WARNING.slice(0, 30).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
      assert.equal(main.querySelector("[data-redaction-state]").getAttribute("data-redaction-state"), "off");
      submit(form(main));
      await flush();
      assert.equal(confirm.open, true);
      assert.ok(confirm.querySelector(".dialog-header svg.icon"), "the danger header icon");
      assert.ok(confirm.querySelector("button.danger"));
      assert.match(norm(confirm.textContent), /security change/);
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent), ["enabled: true -> false"]);
      (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      if (answer === "cancel") {
        assert.deepEqual(puts(api), []);
        assert.equal(enabledSwitch(main).checked, false, "the edit stays in the form, unsaved");
      } else {
        assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, enabled: false }]]);
        assert.match(norm(main.querySelector("[data-save-warnings]").textContent), /server text/);
      }
    });
  }
});

test("removing a type shows a warning naming it, then a confirmDialog; Cancel sends nothing", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const api = fakeApi(redactionRoutes());
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      toggle(typeBox(main, "PIN"));
      const pending = main.querySelector("[data-pending-warnings]");
      assert.match(norm(pending.textContent), new RegExp(REMOVED_WARNING.slice(0, 30).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
      assert.match(norm(pending.textContent), /PIN/);
      submit(form(main));
      await flush();
      assert.equal(confirm.open, true);
      assert.ok(confirm.querySelector("button.danger"));
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent), ["entity_types: PIN removed"]);
      (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      if (answer === "cancel") assert.deepEqual(puts(api), []);
      else assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, entity_types: ["SSN", "NAME"] }]]);
    });
  }
});

test("no type selected: a warning, a confirmDialog, and the PUT is enabled: false with entity_types: [] (never refused)", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(redactionRoutes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    for (const g of ENTITY_GROUPS) {
      const box = groupBox(main, g.name);
      if (box && (box.checked || box.indeterminate)) {
        if (!box.checked) toggle(box); // mixed -> all
        toggle(box); // all -> none
      }
    }
    assert.deepEqual(checkedTypes(main), []);
    assert.equal(enabledSwitch(main).checked, false, "no types means off: the switch follows");
    assert.equal(stateOf(main).getAttribute("data-redaction-state"), "off");
    assert.equal(norm(stateOf(main).textContent), OFF_NO_TYPES);
    const pending = norm(main.querySelector("[data-pending-warnings]").textContent);
    assert.match(pending, new RegExp(escape(OFF_WARNING.slice(0, 30))));
    assert.match(pending, new RegExp(escape(OFF_NO_TYPES)));
    const before = displayed(main);
    submit(form(main));
    await flush();
    assert.equal(confirm.open, true);
    const details = confirm.querySelectorAll("li").map((li) => li.textContent);
    assert.deepEqual(details, ["enabled: true -> false", "entity_types: NAME, PIN, SSN removed"]);
    confirmButton(confirm).click();
    await flush();
    assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, enabled: false, entity_types: [] }]]);
    assert.equal(main.querySelectorAll(".panel--error").length, 0, "the server accepted it");
    // Repainted from the response: the same off state as before the save (it round-trips).
    assert.deepEqual(displayed(main), before);
  });
});

test("from (false, []): ticking a type keeps it off; switching on then saves enabled: true without asking", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(redactionRoutes(viewOf({ ...ON, enabled: false, entity_types: [] })));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    toggle(typeBox(main, "SSN"));
    assert.equal(enabledSwitch(main).checked, false, "ticking a type doesn't switch it on");
    assert.equal(stateOf(main).getAttribute("data-redaction-state"), "off");
    assert.notEqual(norm(stateOf(main).textContent), OFF_NO_TYPES, "a type is selected now");
    toggle(enabledSwitch(main));
    assert.equal(stateOf(main).getAttribute("data-redaction-state"), "on");
    submit(form(main));
    await flush();
    assert.equal(confirm.open, false);
    assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, enabled: true, entity_types: ["SSN"] }]]);
  });
});

test("switching on with no type selected ticks the default types, so it is never on with nothing to mask", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(redactionRoutes(viewOf({ ...ON, enabled: false, entity_types: [] })));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    toggle(enabledSwitch(main));
    assert.equal(enabledSwitch(main).checked, true);
    assert.deepEqual(checkedTypes(main).sort(), [...DEFAULT_TYPES].sort());
    assert.equal(stateOf(main).getAttribute("data-redaction-state"), "on");
    const before = displayed(main);
    submit(form(main));
    await flush();
    assert.equal(confirm.open, false, "turning on is tightening");
    const domOrder = typeBoxes(main).map((b) => b.getAttribute("data-entity-type"));
    assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, enabled: true,
      entity_types: domOrder.filter((t) => DEFAULT_TYPES.includes(t)) }]]);
    assert.deepEqual(displayed(main), before, "repainted as it was shown");
  });
  // With no defaults in the view, the switch can't be turned on with nothing selected.
  const api2 = fakeApi(redactionRoutes({ ...viewOf({ ...ON, enabled: false, entity_types: [] }), defaults: {} }));
  await renderUnder(async ({ main }) => {
    await run(api2)(main);
    toggle(enabledSwitch(main));
    assert.equal(enabledSwitch(main).checked, false);
    assert.equal(norm(stateOf(main).textContent), OFF_NO_TYPES);
  });
});

test("raising min_confidence shows a warning, then a danger confirmDialog; Cancel sends nothing", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const api = fakeApi(redactionRoutes());
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      setValue(fieldOf(main, "min_confidence"), "0.7");
      const pending = norm(main.querySelector("[data-pending-warnings]").textContent);
      assert.match(pending, new RegExp(escape(CONFIDENCE_WARNING.slice(0, 30))));
      submit(form(main));
      await flush();
      assert.equal(confirm.open, true);
      assert.ok(confirm.querySelector("button.danger"));
      assert.ok(confirm.querySelector(".dialog-header svg.icon"), "the danger header icon");
      assert.match(norm(confirm.textContent), /security change/);
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent), ["min_confidence: 0.5 -> 0.7"]);
      (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      if (answer === "cancel") {
        assert.deepEqual(puts(api), []);
        assert.equal(fieldOf(main, "min_confidence").value, "0.7", "the edit stays in the form, unsaved");
      } else {
        assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, min_confidence: "0.7" }]]);
      }
    });
  }
  // The same value written differently ("0.50") isn't a raise, and lowering asks nothing.
  for (const v of ["0.50", "0.2"]) {
    const api = fakeApi(redactionRoutes());
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      setValue(fieldOf(main, "min_confidence"), v);
      assert.equal(norm(main.querySelector("[data-pending-warnings]").textContent), "", v);
      submit(form(main));
      await flush();
      assert.equal(confirm.open, false, v);
    });
  }
});

test("setting on_failure to allow shows a warning, then a danger confirmDialog; Cancel sends nothing", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const api = fakeApi(redactionRoutes());
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      setValue(fieldOf(main, "on_failure"), "allow");
      const pending = norm(main.querySelector("[data-pending-warnings]").textContent);
      assert.match(pending, new RegExp(escape(ALLOW_WARNING.slice(0, 30))));
      submit(form(main));
      await flush();
      assert.equal(confirm.open, true);
      assert.ok(confirm.querySelector("button.danger"));
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent), ["on_failure: withhold -> allow"]);
      (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      if (answer === "cancel") assert.deepEqual(puts(api), []);
      else assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, on_failure: "allow" }]]);
    });
  }
  // allow -> withhold is tightening: no confirmation.
  const api = fakeApi(redactionRoutes(viewOf({ ...ON, on_failure: "allow" })));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    setValue(fieldOf(main, "on_failure"), "withhold");
    submit(form(main));
    await flush();
    assert.equal(confirm.open, false);
    assert.deepEqual(puts(api), [["put", "redaction", { rev: 4, ...ON, on_failure: "withhold" }]]);
  });
});

test("several weakening edits are listed together in one confirmation", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(redactionRoutes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    toggle(typeBox(main, "PIN"));
    setValue(fieldOf(main, "min_confidence"), "0.9");
    setValue(fieldOf(main, "on_failure"), "allow");
    submit(form(main));
    await flush();
    assert.equal(confirm.open, true);
    assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent),
      ["entity_types: PIN removed", "min_confidence: 0.5 -> 0.9", "on_failure: withhold -> allow"]);
    cancelButton(confirm).click();
    await flush();
    assert.deepEqual(puts(api), []);
  });
});

test("a 409 shows the compare cards with the current config next to yours, and Reload reads again", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const current = viewOf({ ...ON, mask: "[X]" }, { rev: 9 });
  const base = redactionRoutes();
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(409, { error: "conflict", message: "changed since read", current }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    setValue(fieldOf(main, "mask"), "[M:{type}]");
    submit(form(main));
    await flush();
    assert.equal(puts(api).length, 1);
    const cmp = main.querySelector(".panel--error .compare");
    assert.ok(cmp, "the Current and Yours cards");
    const changed = cmp.querySelectorAll(".compare-row.is-changed").map((r) => norm(r.querySelector("dt code").textContent));
    assert.ok(changed.includes("mask"), `mask is a changed row: ${changed}`);
    assert.match(norm(cmp.textContent), /\[X\]/);
    const reload = main.querySelectorAll("button").find((b) => norm(b.textContent) === "Reload current value");
    reload.click();
    await flush();
    assert.deepEqual(api.calls.at(-1), ["get", "redaction"]);
  });
});

test("a 400 naming redaction.<field> marks that field; a bad min_confidence or empty mask is caught before sending", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = redactionRoutes();
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(400, { error: "invalid", message: "may contain only the {type} placeholder", field: "redaction.mask" }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    setValue(fieldOf(main, "mask"), "[{name}]");
    submit(form(main));
    await flush();
    assert.equal(puts(api).length, 1);
    assert.equal(fieldOf(main, "mask").getAttribute("aria-invalid"), "true");
    for (const [f, v] of [["min_confidence", "1.5"], ["min_confidence", "abc"], ["min_confidence", ""], ["mask", ""]]) {
      await run(api)(main);
      setValue(fieldOf(main, f), v);
      submit(form(main));
      await flush();
      assert.equal(fieldOf(main, f).getAttribute("aria-invalid"), "true", `${f}=${JSON.stringify(v)}`);
    }
    assert.equal(puts(api).length, 1, "the client-side checks sent nothing");
  });
});

test("the controls are disabled for a non-admin who reaches the view directly", async () => {
  const api = fakeApi(redactionRoutes());
  await renderUnder(async ({ main }) => {
    await run(api, VIEWER)(main);
    for (const el of main.querySelectorAll("input, select")) assert.equal(el.disabled, true);
    assert.equal(main.querySelectorAll('button[type="submit"]').length, 0);
  });
});

// ---------------------------------------------------------------- PBT

const typeName = oneOf(...AVAILABLE);
const configGen = gen((rng) => {
  const types = [...new Set(array(typeName, { maxLength: 6 })(rng))];
  return {
    enabled: types.length ? oneOf(true, false)(rng) : false, // the server never stores (true, [])
    entity_types: types,
    min_confidence: oneOf("0.5", "0.0", "1.0", "0.75", "0.33")(rng),
    mask: oneOf("[REDACTED:{type}]", "[{type}]", "***")(rng),
    on_failure: oneOf("withhold", "allow")(rng),
  };
});
const payloadGen = gen((rng) => {
  const cfg = configGen(rng);
  const invalid = int(0, 4)(rng) === 0;
  return viewOf(invalid ? { apply_to: ["x"] } : cfg, {
    rev: int(0, 50)(rng),
    effective: invalid ? DEFAULTS : cfg,
    problems: invalid ? [{ field: "redaction.apply_to", message: unicodeString({ minLength: 1, maxLength: 8 })(rng) }] : [],
  });
});

test("PBT: the checkboxes equal entity_types, and each group heading matches its group", async () => {
  await forAll(payloadGen, async (payload) => {
    const api = fakeApi(redactionRoutes(payload));
    await renderUnder(async ({ main }) => {
      await run(api)(main);
      const eff = payload.effective;
      assert.deepEqual(checkedTypes(main).sort(), [...eff.entity_types].sort());
      assert.equal(enabledSwitch(main).checked, eff.enabled && eff.entity_types.length > 0);
      for (const fs of main.querySelectorAll("fieldset[data-group]")) {
        const boxes = fs.querySelectorAll("input[data-entity-type]");
        const n = boxes.filter((b) => b.checked).length;
        const head = groupBox(main, fs.getAttribute("data-group"));
        assert.equal(head.checked, n === boxes.length);
        assert.equal(head.indeterminate === true, n > 0 && n < boxes.length);
      }
    });
  });
});

/** Any (enabled, entity_types) pair, (true, []) included, which the server never stores. */
const anyConfigGen = gen((rng) => ({ ...configGen(rng), enabled: oneOf(true, false)(rng) }));

test("PBT the displayed state is off iff enabled is false or the list is empty", async () => {
  await forAll(anyConfigGen, async (cfg) => {
    const api = fakeApi(redactionRoutes(viewOf(cfg)));
    await renderUnder(async ({ main }) => {
      await run(api)(main);
      const off = !cfg.enabled || cfg.entity_types.length === 0;
      const d = displayed(main);
      assert.equal(d.switch, !off);
      assert.match(d.switchText, off ? /Off/ : /On/);
      assert.equal(d.state, off ? "off" : "on");
      assert.equal(d.stateText === OFF_NO_TYPES, cfg.entity_types.length === 0, "the no-types line iff no types");
      assert.deepEqual(d.types, [...cfg.entity_types].sort());
      // Nothing edited: Save sends nothing, whatever was loaded.
      submit(form(main));
      await flush();
      assert.deepEqual(puts(api), []);
    });
  }, { runs: 150 });
});

/** An edit script: switch flips, type toggles, group toggles and new field values. */
const editGen = record({
  flips: int(0, 2),
  toggles: array(typeName, { maxLength: 6 }),
  groups: array(oneOf(...ENTITY_GROUPS.map((g) => g.name)), { maxLength: 3 }),
  clearAll: oneOf(false, false, true), // then untick every type, so the empty selection is reached often
  min_confidence: oneOf(undefined, "0.1", "0.9", "1", "0"),
  mask: oneOf(undefined, "[{type}]", "#", string({ minLength: 1, maxLength: 6 })),
  on_failure: oneOf(undefined, "allow", "withhold"),
  answer: oneOf("confirm", "cancel"),
});

test("PBT: the PUT body equals the edited config, never (enabled: true, []), and round-trips; Cancel sends nothing", async () => {
  await forAll(record({ payload: payloadGen, edit: editGen }), async ({ payload, edit }) => {
    const api = fakeApi(redactionRoutes(payload));
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      // Apply the edit through the UI, and keep an independent model of it.
      const eff = payload.effective;
      let on = eff.enabled && eff.entity_types.length > 0;
      const selected = new Set(eff.entity_types);
      const noTypesOff = () => { if (!selected.size) on = false; }; // no types means off
      for (let i = 0; i < edit.flips; i++) {
        toggle(enabledSwitch(main));
        on = !on;
        if (on && !selected.size) for (const t of payload.defaults.entity_types) selected.add(t);
        noTypesOff();
      }
      for (const t of edit.toggles) {
        toggle(typeBox(main, t));
        if (selected.has(t)) selected.delete(t); else selected.add(t);
        noTypesOff();
      }
      for (const g of edit.groups) {
        const box = groupBox(main, g);
        const types = typeBoxes(main).filter((b) => b.closest("fieldset[data-group]").getAttribute("data-group") === g)
          .map((b) => b.getAttribute("data-entity-type"));
        const all = types.every((t) => selected.has(t));
        toggle(box);
        for (const t of types) if (all) selected.delete(t); else selected.add(t);
        noTypesOff();
      }
      if (edit.clearAll) {
        for (const b of typeBoxes(main)) if (b.checked) toggle(b);
        selected.clear();
        noTypesOff();
      }
      for (const f of ["min_confidence", "mask", "on_failure"]) if (edit[f] !== undefined) setValue(fieldOf(main, f), edit[f]);
      assert.deepEqual(checkedTypes(main).sort(), [...selected].sort(), "the boxes follow the model");
      assert.equal(enabledSwitch(main).checked, on, "the switch follows the model, never on with no type");

      const domOrder = typeBoxes(main).map((b) => b.getAttribute("data-entity-type"));
      const entity_types = [...eff.entity_types.filter((t) => selected.has(t)),
        ...domOrder.filter((t) => selected.has(t) && !eff.entity_types.includes(t))];
      const expected = {
        rev: payload.rev,
        enabled: on && entity_types.length > 0,
        entity_types,
        min_confidence: edit.min_confidence ?? eff.min_confidence,
        mask: edit.mask ?? eff.mask,
        on_failure: edit.on_failure ?? eff.on_failure,
      };
      const { rev: _r, ...exp } = expected;
      const changed = JSON.stringify({ ...exp, min_confidence: Number(exp.min_confidence) })
        !== JSON.stringify({ ...eff, min_confidence: Number(eff.min_confidence) });
      const removed = eff.entity_types.filter((t) => !selected.has(t));
      const raised = Number(expected.min_confidence) > Number(eff.min_confidence);
      const allow = expected.on_failure === "allow" && eff.on_failure !== "allow";
      const asks = changed && ((eff.enabled && !expected.enabled) || removed.length > 0 || raised || allow);
      const before = displayed(main);

      submit(form(main));
      await flush();
      assert.equal(confirm.open, asks,
        "a confirmation iff it switches off, removes a type, raises min_confidence or sets on_failure to allow");
      if (asks) {
        (edit.answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
        await flush();
      }
      const sent = puts(api);
      if (!changed || (asks && edit.answer === "cancel")) {
        assert.deepEqual(sent, []);
        return;
      }
      assert.deepEqual(sent, [["put", "redaction", expected]]);
      assert.ok(!(expected.enabled && expected.entity_types.length === 0), "never a body the server refuses");
      assert.equal(main.querySelectorAll(".panel--error").length, 0, "the fake server accepted it");
      assert.deepEqual(displayed(main), before, "the saved state reads back as it was shown (round trip)");
    });
  }, { runs: 200 });
});

const userGen = oneOf(ADMIN, EDITOR, VIEWER, { groups: [], email: "n@example.com", username: "n" });

test("PBT: the view for generated payloads and users", async () => {
  await forAll(record({ payload: payloadGen, user: userGen }), async ({ payload, user }) => {
    const api = fakeApi(redactionRoutes(payload));
    await renderUnder(async ({ document, main }) => {
      await run(api, user)(main);
      assertNoInlineStyle(document.body);
      assertClassesStyled(document.body);
      assertViewStructure(main);
    });
  });
});

test("PBT: with the confirmation open, after a save with warnings, a 400 and a 409", async () => {
  const caseGen = record({ payload: payloadGen, outcome: oneOf("confirm-open", "saved", "400", "409") });
  await forAll(caseGen, async ({ payload, outcome }) => {
    const base = redactionRoutes(payload);
    const api = fakeApi((m, p, a) => {
      if (m === "put" && outcome === "409") throw new ApiError(mapError(409, { error: "conflict", current: payload }));
      if (m === "put" && outcome === "400") throw new ApiError(mapError(400, { error: "invalid", message: "bad", field: "redaction.mask" }));
      return base(m, p, a);
    });
    await renderUnder(async ({ document, main, confirm }) => {
      await run(api)(main);
      const anyChecked = typeBoxes(main).find((b) => b.checked);
      if (outcome === "confirm-open" && anyChecked) toggle(anyChecked);
      else setValue(fieldOf(main, "mask"), "[changed:{type}]");
      submit(form(main));
      await flush();
      if (outcome !== "confirm-open" && confirm.open) {
        confirmButton(confirm).click();
        await flush();
      }
      assertNoInlineStyle(document.body);
      assertClassesStyled(document.body);
      assertViewStructure(main);
    });
  }, { runs: 100 });
});
