// Slots view (phases 1 and 2). Rendered under fakedom with a recording fake API:
//   rows and switches reflect a GET /slots payload; maxRetries has no control; a PUT body holds
//   only the changed field and rev; turning sensitive or keypad_only off asks first (danger) and
//   Cancel sends nothing; a 409 shows the compare cards; a 400 marks the named switch; the
//   response's warnings are shown.
// Property test (view half): the "effective keypad-only" badge appears iff the payload's
// effective_keypad_only is true and keypad_only is false (the view never recomputes it).
// PBT (no inline style, every class styled, view structure) through viewkit over generated payloads.
// Phase 2: the mode, effective-mode and confirm columns, the
// slot_resolution card (generative_enabled for admins only, "no effect" below a generative
// ceiling), a choice slot's spoken names (PUT /prompts/option.*) and aliases; a mode change shows
// POST /slots/preview in its confirmation (danger when the preview warns of a move away from
// deterministic) and Cancel sends nothing; a PUT body holds only the changed fields and rev; a 409
// shows the compare cards; an alias conflict's 400 is shown on its field.
// Property test (view half): the badges equal the payload's effective-mode table (never recomputed).
import assert from "node:assert/strict";
import { test } from "node:test";

import { Event } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string, unicodeString } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder } from "./viewkit.js";
import { SECTIONS } from "../app.js";
import { ApiError, mapError } from "../lib/api.js";
import {
  CHOICE_NEVER_HINT, EFFECTIVE_KEYPAD_ONLY, GENERATIVE_NO_EFFECT, GENERATIVE_WARNING, slotsView,
} from "../views/slots.js";

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

const NOT_SENSITIVE = "This slot is a verification factor and isn't sensitive: its value may be logged, journaled, "
  + "read back and passed to the AI agent. Production deployments should flag every verification factor as "
  + "sensitive; the seed leaves ZIP unflagged for demos only.";

const slot = (over) => ({
  name: "x", type: "digits", sensitive: false, keypad_only: false, barge_in: true, maxRetries: 2,
  is_factor: false, effective_keypad_only: false, flow: null, rev: 1, loaded: true, problems: [], warnings: [], ...over,
});

const ZIP = slot({ name: "zip", is_factor: true, rev: 3,
  warnings: [{ code: "verification_not_sensitive", slot: "zip", message: NOT_SENSITIVE }] });
const SSN4 = slot({ name: "ssn4", is_factor: true, sensitive: true, keypad_only: true, effective_keypad_only: true, rev: 1 });
const OTP = slot({ name: "otp", is_factor: true, sensitive: true, rev: 0 });
const YES_NO = slot({ name: "yes_no", type: "yes_no", maxRetries: 1, flow: "confirm", rev: 2 });

const PAYLOAD = { acceptSpokenFactors: true, takes_effect_s: 60, slots: [ZIP, SSN4, OTP, YES_NO] };

/** GET /slots answers `payload`; PUT merges the body into the slot and bumps its rev. */
function slotRoutes(payload = PAYLOAD, { putWarnings = () => [] } = {}) {
  return (method, path, arg) => {
    if (method === "get" && path === "slots") return structuredClone(payload);
    if (method === "put" && path.startsWith("slots/")) {
      const name = path.slice(6);
      const cur = payload.slots.find((s) => s.name === name);
      const { rev, ...fields } = arg;
      const next = { ...cur, ...fields, rev: rev + 1, warnings: [] };
      if (next.is_factor && !next.sensitive) next.warnings = [{ code: "verification_not_sensitive", slot: name, message: NOT_SENSITIVE }];
      return { changed: true, takes_effect_s: 60, security_change: false, audit: { pk: `policy#slot#${name}`, sk: "s" },
        slot: next, warnings: putWarnings(name, fields, cur) };
    }
    throw new Error(`unexpected ${method} ${path}`);
  };
}

const run = (api, user = ADMIN) => (main) => slotsView(main, { api, user });

const rowOf = (main, name) => main.querySelectorAll("tbody tr").find((tr) => tr.getAttribute("data-slot") === name);
const switchOf = (main, name, f) => main.querySelector(`input[data-field="slot#${name}.${f}"]`);
const cellTexts = (tr) => tr.children.map((c) => norm(c.textContent));

function toggle(input) {
  input.checked = !input.checked;
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

const confirmButton = (dlg) => dlg.querySelector("button.danger");
const cancelButton = (dlg) => dlg.querySelectorAll("button").find((b) => norm(b.textContent) === "Cancel");

// ---------------------------------------------------------------- the registry

test("SECTIONS has one slots entry with the reserved slots icon", () => {
  const found = SECTIONS.filter((s) => s.id === "slots");
  assert.equal(found.length, 1);
  assert.equal(found[0].label, "Slots");
  assert.equal(found[0].icon, "slots");
  assert.equal(found[0].render, slotsView);
});

// ---------------------------------------------------------------- rendering

test("rows and switches reflect the GET /slots payload; maxRetries is text, with no control", async () => {
  const api = fakeApi(slotRoutes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.deepEqual(api.calls, [["get", "slots"]]);
    assert.equal(norm(main.querySelector("h1#page-title").textContent), "Slots");
    assert.deepEqual(main.querySelectorAll("thead th").map((th) => norm(th.textContent)),
      ["Slot", "Type", "Sensitive", "Keypad only", "Barge-in", "Mode", "Effective mode", "Confirm", "Max retries", "History"]);
    for (const s of PAYLOAD.slots) {
      const tr = rowOf(main, s.name);
      assert.ok(tr, `a row for ${s.name}`);
      for (const f of ["sensitive", "keypad_only", "barge_in"]) {
        const sw = switchOf(main, s.name, f);
        assert.ok(sw, `${s.name}.${f} has a switch`);
        assert.equal(sw.getAttribute("role"), "switch");
        assert.equal(sw.getAttribute("type"), "checkbox");
        assert.ok(sw.classList.contains("switch"));
        assert.equal(sw.checked, s[f], `${s.name}.${f}`);
        assert.equal(sw.disabled, false);
        assert.ok(sw.getAttribute("aria-label").startsWith(`${s.name}: `));
        assert.ok(tr.contains(sw));
      }
      const cells = cellTexts(tr);
      assert.equal(cells[1], s.type);
      assert.equal(cells[8], String(s.maxRetries));
      assert.equal(tr.children[8].querySelectorAll("input, select, textarea, button").length, 0, "maxRetries is read-only");
      assert.ok(tr.children[8].classList.contains("num"));
      const history = tr.querySelector("a");
      assert.equal(history.getAttribute("href"), `#/audit?item=${encodeURIComponent(`policy#slot#${s.name}`)}`);
    }
    // The visible On/Off text next to each switch (not colour alone).
    assert.match(norm(rowOf(main, "ssn4").children[2].textContent), /On/);
    assert.match(norm(rowOf(main, "zip").children[2].textContent), /Off/);
    // Factors are marked.
    assert.match(norm(rowOf(main, "zip").children[0].textContent), /factor/);
    assert.doesNotMatch(norm(rowOf(main, "yes_no").children[0].textContent), /factor/);
    assert.ok(main.querySelector(".card.flush .table-wrap table"));
    assert.ok(main.querySelector(".panel--info"), "the takes-effect and ceiling notes");
  });
});

test("the acceptSpokenFactors ceiling is a read-only info panel above the table", async () => {
  for (const asf of [true, false]) {
    const api = fakeApi(slotRoutes({ ...PAYLOAD, acceptSpokenFactors: asf }));
    await renderUnder(async ({ main }) => {
      await run(api)(main);
      const ceiling = main.querySelector('[data-ceiling="acceptSpokenFactors"]');
      assert.ok(ceiling.classList.contains("panel--info"));
      assert.equal(ceiling.querySelectorAll("input, select, textarea, button").length, 0);
      const text = norm(ceiling.textContent);
      assert.match(text, /acceptSpokenFactors/);
      assert.match(text, asf ? /\bon\b/ : /\boff\b/);
      if (!asf) assert.match(text, /every verification factor is keypad-only/);
      const card = main.querySelector(".card.flush");
      const kids = main.children;
      assert.ok(kids.indexOf(ceiling) < kids.indexOf(card), "the ceiling panel comes before the table's card");
    });
  }
});

test("the 9.8 warning: a warning panel on each slot whose GET carries it, with the server's text", async () => {
  const api = fakeApi(slotRoutes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const panels = main.querySelectorAll("tbody .panel--warning");
    assert.equal(panels.length, 1);
    assert.equal(panels[0].closest("tr").getAttribute("data-warning-for"), "zip");
    assert.match(norm(panels[0].textContent), /Production deployments should flag every verification factor/);
    // It sits right after zip's row.
    const rows = main.querySelectorAll("tbody tr");
    assert.equal(rows[rows.indexOf(rowOf(main, "zip")) + 1], panels[0].closest("tr"));
  });
});

test("the effective keypad-only badge shows the GET's effective_keypad_only, when keypad_only is off", async () => {
  const forced = slot({ name: "zip", is_factor: true, sensitive: true, effective_keypad_only: true });
  const api = fakeApi(slotRoutes({ ...PAYLOAD, acceptSpokenFactors: false, slots: [forced, SSN4, YES_NO] }));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const badges = (name) => rowOf(main, name).querySelectorAll(".badge").filter((b) => norm(b.textContent) === EFFECTIVE_KEYPAD_ONLY);
    assert.equal(badges("zip").length, 1);
    assert.ok(rowOf(main, "zip").children[3].contains(badges("zip")[0]), "in the Keypad only cell");
    assert.equal(badges("ssn4").length, 0, "its own keypad_only is on: no badge");
    assert.equal(badges("yes_no").length, 0);
  });
});

test("a slot that didn't load shows its problems and no switches", async () => {
  const broken = slot({ name: "card", loaded: false, type: null, sensitive: null, keypad_only: null, barge_in: null,
    maxRetries: null, is_factor: null, effective_keypad_only: null, problems: ["slot#card.slotType: unknown slot type"] });
  const api = fakeApi(slotRoutes({ ...PAYLOAD, slots: [YES_NO, broken] }));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const tr = rowOf(main, "card");
    assert.equal(tr.querySelectorAll("input").length, 0);
    assert.match(norm(tr.textContent), /1 problem/);
    const problems = main.querySelector('[data-problems="slots"]');
    assert.ok(problems.classList.contains("panel--warning"));
    assert.match(norm(problems.textContent), /card: slot#card\.slotType: unknown slot type/);
    assertViewStructure(main);
  });
});

test("no slots: an empty state", async () => {
  const api = fakeApi(slotRoutes({ ...PAYLOAD, slots: [] }));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.equal(norm(main.querySelector(".empty-state-title").textContent), "No slots");
    assert.equal(main.querySelectorAll("table").length, 0);
    assertViewStructure(main);
  });
});

test("a 503 not_seeded on load shows the error panel", async () => {
  const api = fakeApi(() => { throw new ApiError(mapError(503, { error: "not_seeded", message: "the policy table has no meta item" })); });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.ok(main.querySelector(".panel--error"));
    assert.match(norm(main.textContent), /the policy table has no meta item/);
  });
});

test("a viewer sees every switch disabled, the view-only note, and can't PUT", async () => {
  const api = fakeApi(slotRoutes());
  await renderUnder(async ({ main }) => {
    await run(api, VIEWER)(main);
    const switches = main.querySelectorAll("input.switch");
    assert.equal(switches.length, 12);
    assert.ok(switches.every((s) => s.disabled));
    assert.match(main.textContent, /Your group can view only\./);
  });
});

// ---------------------------------------------------------------- saving

test("a change that isn't a security change PUTs only that field and the rev, without asking", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(slotRoutes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api, EDITOR)(main);
    toggle(switchOf(main, "zip", "barge_in"));
    await flush();
    assert.equal(confirm.open, false);
    toggle(switchOf(main, "zip", "sensitive")); // false -> true: tightening, no confirmation
    await flush();
    assert.equal(confirm.open, false);
    toggle(switchOf(main, "yes_no", "keypad_only"));
    await flush();
    assert.deepEqual(puts(api), [
      ["put", "slots/zip", { rev: 3, barge_in: false }],
      ["put", "slots/zip", { rev: 4, sensitive: true }], // the rev from the first save's response
      ["put", "slots/yes_no", { rev: 2, keypad_only: true }],
    ]);
    // The row follows the response: zip is now sensitive, so the warning panel is gone.
    assert.equal(switchOf(main, "zip", "sensitive").checked, true);
    assert.equal(main.querySelectorAll('tr[data-warning-for="zip"]').length, 0);
  });
});

test("turning sensitive off asks first with the danger variant; Cancel sends nothing and resets the switch", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const api = fakeApi(slotRoutes());
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      const sw = switchOf(main, "ssn4", "sensitive");
      toggle(sw);
      await flush();
      assert.equal(confirm.open, true);
      assert.match(norm(confirm.querySelector(".dialog-title").textContent), /sensitive/i);
      assert.ok(confirm.querySelector(".dialog-header svg.icon"), "the danger header icon");
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent), ["sensitive: true -> false"]);
      assert.match(norm(confirm.textContent), /security change/);
      (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      if (answer === "cancel") {
        assert.deepEqual(puts(api), []);
        assert.equal(switchOf(main, "ssn4", "sensitive").checked, true, "back to its saved value");
      } else {
        assert.deepEqual(puts(api), [["put", "slots/ssn4", { rev: 1, sensitive: false }]]);
        assert.equal(switchOf(main, "ssn4", "sensitive").checked, false);
      }
    });
  }
});

test("turning keypad_only off asks first with the danger variant; Cancel sends nothing", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const api = fakeApi(slotRoutes({ ...PAYLOAD, acceptSpokenFactors: false }));
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      toggle(switchOf(main, "ssn4", "keypad_only"));
      await flush();
      assert.equal(confirm.open, true);
      assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent), ["keypad_only: true -> false"]);
      assert.match(norm(confirm.textContent), /speech recognition/);
      assert.match(norm(confirm.textContent), /acceptSpokenFactors is off/, "a factor stays keypad-only under the ceiling");
      (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      assert.deepEqual(puts(api), answer === "confirm" ? [["put", "slots/ssn4", { rev: 1, keypad_only: false }]] : []);
      assert.equal(switchOf(main, "ssn4", "keypad_only").checked, answer !== "confirm");
    });
  }
});

test("the response's warnings are shown after a save", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const putWarnings = (name) => [
    { code: "sensitive_off", slot: name, message: "Turning sensitive off is a security change: server text." },
    { code: "verification_not_sensitive", slot: name, message: NOT_SENSITIVE },
  ];
  const api = fakeApi(slotRoutes(PAYLOAD, { putWarnings }));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    toggle(switchOf(main, "otp", "sensitive"));
    await flush();
    confirmButton(confirm).click();
    await flush();
    const shown = main.querySelector('[data-save-warnings="otp"]');
    assert.ok(shown.classList.contains("panel--warning"));
    const items = shown.querySelectorAll("li").map((li) => norm(li.textContent));
    assert.deepEqual(items, ["otp: Turning sensitive off is a security change: server text.", `otp: ${NOT_SENSITIVE}`]);
    // And the row's warning panel now comes from the response's slot.
    assert.equal(main.querySelectorAll('tr[data-warning-for="otp"]').length, 1);
  });
});

test("a 409 shows the compare cards and resets the switch", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = slotRoutes();
  const current = { ...YES_NO, barge_in: false, rev: 9 };
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(409, { error: "conflict", message: "the slot changed since it was read", current }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    toggle(switchOf(main, "yes_no", "keypad_only"));
    await flush();
    assert.deepEqual(puts(api), [["put", "slots/yes_no", { rev: 2, keypad_only: true }]]);
    assert.ok(main.querySelector(".panel--error .compare"), "the Current and Yours cards");
    assert.match(norm(main.querySelector(".compare").textContent), /keypad_only/);
    assert.equal(switchOf(main, "yes_no", "keypad_only").checked, false);
    const reload = main.querySelectorAll("button").find((b) => norm(b.textContent) === "Reload current value");
    reload.click();
    await flush();
    assert.deepEqual(api.calls.at(-1), ["get", "slots"]);
  });
});

test("a 400 naming slot#<name>.<field> marks that switch invalid", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = slotRoutes();
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(400, { error: "invalid", message: "must be true or false", field: "slot#yes_no.barge_in" }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const sw = switchOf(main, "yes_no", "barge_in");
    toggle(sw);
    await flush();
    assert.equal(sw.getAttribute("aria-invalid"), "true");
    assert.equal(sw.checked, true, "reset to its saved value");
    assertViewStructure(main);
  });
});

test("a switch is busy while its save is pending, and a second change then is undone without a PUT", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  let release;
  const pending = new Promise((r) => { release = r; });
  const base = slotRoutes();
  const api = fakeApi(async (m, p, a) => (m === "put" ? (await pending, base(m, p, a)) : base(m, p, a)));
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const sw = switchOf(main, "yes_no", "barge_in");
    toggle(sw);
    await flush();
    assert.equal(sw.getAttribute("aria-busy"), "true");
    const other = switchOf(main, "yes_no", "keypad_only");
    toggle(other);
    await flush();
    assert.equal(other.checked, false, "undone");
    assert.equal(puts(api).length, 1);
    release();
    await flush();
    assert.equal(puts(api).length, 1);
    assert.equal(switchOf(main, "yes_no", "barge_in").checked, false);
  });
});

// ---------------------------------------------------------------- generators

const NAMES = ["zip", "ssn4", "otp", "yes_no", "card", "dob", "pin"];
const warningGen = (name) => record({ code: "verification_not_sensitive", slot: name, message: unicodeString({ minLength: 1, maxLength: 10 }) });

/** One GET /slots slot: loaded or not, any mix of settings, problems and warnings. */
const slotGen = (name) => gen((rng) => {
  const loaded = int(0, 5)(rng) > 0;
  const problems = array(unicodeString({ minLength: 1, maxLength: 8 }), { maxLength: 2 })(rng);
  if (!loaded) {
    return slot({ name, loaded, problems, type: null, sensitive: null, keypad_only: null, barge_in: null,
      maxRetries: null, is_factor: null, effective_keypad_only: null, rev: int(0, 50)(rng) });
  }
  const b = oneOf(true, false);
  return {
    name, loaded, problems,
    type: oneOf("digits", "yes_no", "choice", unicodeString({ minLength: 1, maxLength: 6 }))(rng),
    sensitive: b(rng), keypad_only: b(rng), barge_in: b(rng),
    maxRetries: oneOf(int(0, 5), null)(rng), is_factor: b(rng), effective_keypad_only: b(rng),
    flow: oneOf(null, "confirm", string({ minLength: 1, maxLength: 6 }))(rng), rev: int(0, 50)(rng),
    warnings: array(warningGen(name), { maxLength: 1 })(rng),
  };
});

const payloadGen = gen((rng) => {
  const names = NAMES.filter(() => int(0, 2)(rng) > 0);
  return {
    acceptSpokenFactors: oneOf(true, false)(rng),
    takes_effect_s: oneOf(60, 30)(rng),
    slots: names.map((n) => slotGen(n)(rng)),
  };
});
const userGen = oneOf(ADMIN, EDITOR, VIEWER);

// ---------------------------------------------------------------- property: effective keypad-only badge

test("PBT: the keypad-only badge appears iff effective_keypad_only is true and keypad_only is false", async () => {
  await forAll(payloadGen, async (payload) => {
    const api = fakeApi(() => structuredClone(payload));
    await renderUnder(async ({ main }) => {
      await run(api)(main);
      for (const s of payload.slots) {
        const n = rowOf(main, s.name).querySelectorAll(".badge").filter((b) => norm(b.textContent) === EFFECTIVE_KEYPAD_ONLY).length;
        assert.equal(n, s.effective_keypad_only === true && s.keypad_only === false ? 1 : 0, s.name);
      }
    });
  });
});

// ---------------------------------------------------------------- PBT: a PUT body is the one field and rev

const FIELDS = ["sensitive", "keypad_only", "barge_in"];

test("PBT: toggling any switch PUTs exactly {rev, <field>: new value}, confirmed when it turns sensitive or keypad_only off", async () => {
  const loadedPayload = gen((rng) => {
    const p = payloadGen(rng);
    p.slots = p.slots.filter((s) => s.loaded);
    if (!p.slots.length) p.slots = [slotGen("zip")(rng)].map((s) => ({ ...slot({ name: "zip" }), ...s, loaded: true, sensitive: true, keypad_only: true, barge_in: true }));
    return p;
  });
  const caseGen = record({ payload: loadedPayload, pick: int(0, 99), field: oneOf(...FIELDS), answer: oneOf("confirm", "cancel") });
  await forAll(caseGen, async ({ payload, pick, field, answer }) => {
    const target = payload.slots[pick % payload.slots.length];
    const api = fakeApi(slotRoutes(payload));
    await renderUnder(async ({ main, confirm }) => {
      await run(api)(main);
      toggle(switchOf(main, target.name, field));
      await flush();
      const asks = field !== "barge_in" && target[field] === true;
      assert.equal(confirm.open, asks);
      if (asks) (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      const sent = asks && answer === "cancel" ? [] : [["put", `slots/${target.name}`, { rev: target.rev, [field]: !target[field] }]];
      assert.deepEqual(puts(api), sent);
    });
  }, { runs: 100 });
});

// ---------------------------------------------------------------- PBT

test("PBT: the view for generated payloads and users", async () => {
  await forAll(record({ payload: payloadGen, user: userGen }), async ({ payload, user }) => {
    const api = fakeApi(() => structuredClone(payload));
    await renderUnder(async ({ main }) => {
      await run(api, user)(main);
      assertNoInlineStyle(main);
      assertClassesStyled(main);
      assertViewStructure(main);
    });
  });
});

test("PBT: after a save with warnings, and after a 409", async () => {
  const caseGen = record({ payload: payloadGen, conflict: oneOf(true, false) });
  await forAll(caseGen, async ({ payload, conflict }) => {
    const target = payload.slots.find((s) => s.loaded);
    if (!target) return;
    const base = slotRoutes(payload, { putWarnings: (name) => [{ code: "sensitive_off", slot: name, message: "m" }] });
    const api = fakeApi((m, p, a) => {
      if (m === "put" && conflict) throw new ApiError(mapError(409, { error: "conflict", current: target }));
      return base(m, p, a);
    });
    await renderUnder(async ({ document, main, confirm }) => {
      await run(api)(main);
      toggle(switchOf(main, target.name, "barge_in"));
      await flush();
      assert.equal(confirm.open, false);
      assertNoInlineStyle(document.body);
      assertClassesStyled(document.body);
      assertViewStructure(main);
    });
  }, { runs: 100 });
});

// ================================================================ phase 2

const LANGS = ["en-US", "es-US"];
const decision = (configured, source, ceiling, effective, capped = null) => ({ configured, source, ceiling, effective, capped });

const P2_ZIP = { ...ZIP, mode: null, confirm: "when_inferred",
  modes: { login: decision("deterministic", "flow", "assisted", "deterministic"), stepup: decision("assisted", "default", "assisted", "assisted") } };
const P2_SSN4 = { ...SSN4, mode: "generative", confirm: "never",
  modes: { login: decision("generative", "slot", "assisted", "assisted", "ceiling"), stepup: decision("generative", "slot", "assisted", "assisted", "ceiling") } };
const P2_YES_NO = { ...YES_NO, mode: null, confirm: "when_inferred", options: null,
  modes: { confirm: decision("deterministic", "default", "assisted", "deterministic") } };
const P2_CARD = slot({ name: "card", type: "choice", maxRetries: 2, flow: "confirm", rev: 5, mode: null, confirm: "when_inferred",
  modes: { confirm: decision("assisted", "default", "assisted", "assisted") },
  options: [
    { value: "red", dtmf: "1", spoken_names: { "en-US": "Red card", "es-US": "tarjeta roja" }, aliases: { "en-US": ["red"], "es-US": ["roja"] } },
    { value: "purple", dtmf: null, spoken_names: { "en-US": "Purple card", "es-US": "tarjeta morada" }, aliases: { "en-US": ["purple"], "es-US": ["morada"] } },
  ] });

const RESOLUTION = { rev: 2, default_mode: "assisted", flows: { login: "deterministic" }, generative_enabled: false,
  resolver_budget_ms: 1500, problems: [], generative_enabled_set: null, warnings: [] };

const P2 = { acceptSpokenFactors: true, takes_effect_s: 60, ceiling: "assisted", languages: LANGS,
  slot_resolution: RESOLUTION, slots: [P2_ZIP, P2_SSN4, P2_YES_NO, P2_CARD] };

const PROMPT_REVS = { "option.card.red": 3, "option.card.purple": 1 };
const AWAY = { code: "away_from_deterministic", slot: "slot_resolution", message: "Moving away from deterministic is a security change: server text." };
const PREVIEW = { ceiling: "assisted", current: {}, proposed: {},
  calls_model: [{ slot: "yes_no", flow: "confirm", effective: "assisted" }],
  changes: [{ slot: "yes_no", flow: "confirm", from: "deterministic", to: "assisted" }], warnings: [AWAY] };

/** The phase 2 routes: GET /slots, GET /prompts, POST /slots/preview, and the three PUTs. */
function p2Routes(payload = P2, { preview = () => PREVIEW, putWarnings = () => [], fresh } = {}) {
  let res = structuredClone(payload.slot_resolution);
  return (method, path, arg) => {
    if (method === "get" && path === "slots") return structuredClone(fresh || payload);
    if (method === "get" && path === "prompts") {
      const slotName = arg.prefix.split(".")[1];
      const s = payload.slots.find((x) => x.name === slotName);
      return { prefix: arg.prefix, languages: LANGS, prompts: (s.options || []).map((o) => ({
        key: `option.${slotName}.${o.value}`, texts: o.spoken_names, rev: PROMPT_REVS[`option.${slotName}.${o.value}`] ?? 0 })) };
    }
    if (method === "post" && path === "slots/preview") return preview(arg);
    if (method === "put" && path === "slots/resolution") {
      const { rev, ...fields } = arg;
      if (fields.flows) fields.flows = Object.fromEntries(Object.entries(fields.flows).map(([f, v]) => [f, v.mode]));
      res = { ...res, ...fields, rev: rev + 1 };
      return { changed: true, takes_effect_s: 60, security_change: true, slot_resolution: res, warnings: [] };
    }
    if (method === "put" && path.startsWith("prompts/")) {
      return { changed: true, takes_effect_s: 60, prompt: { key: path.slice(8), texts: arg.texts, rev: arg.rev + 1 } };
    }
    if (method === "put" && path.startsWith("slots/")) {
      const name = path.slice(6);
      const cur = payload.slots.find((x) => x.name === name);
      const { rev, ...fields } = arg;
      return { changed: true, takes_effect_s: 60, security_change: false, audit: { pk: `policy#slot#${name}`, sk: "s" },
        slot: { ...cur, ...fields, rev: rev + 1 }, warnings: putWarnings(name, fields, cur) };
    }
    throw new Error(`unexpected ${method} ${path}`);
  };
}

const posts = (api) => api.calls.filter((c) => c[0] === "post");
const modeSelect = (main, name) => main.querySelector(`select[data-field="slot#${name}.mode"]`);
const confirmSelect = (main, name) => main.querySelector(`select[data-field="slot#${name}.confirm"]`);
const resForm = (main) => main.querySelector('form[data-form="slot_resolution"]');
const resField = (main, f) => main.querySelector(`[data-field="slot_resolution.${f}"]`);
const effectiveLines = (main, name) => main.querySelector(`td[data-effective-for="${name}"]`).querySelectorAll("div[data-flow]");
const primaryButton = (dlg) => dlg.querySelector("button.primary");

function choose(select, value) {
  select.value = value;
  select.dispatchEvent(new Event("change", { bubbles: true }));
}

function submit(form) {
  const ev = new Event("submit", { bubbles: true, cancelable: true });
  ev.submitter = form.querySelector('button[type="submit"]');
  form.dispatchEvent(ev);
}

function type(input, value) {
  input.value = value;
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

// ---------------------------------------------------------------- rendering

test("phase 2 controls reflect the payload: mode, effective modes, confirm, slot_resolution, options", async () => {
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.deepEqual(api.calls, [["get", "slots"], ["get", "prompts", { prefix: "option.card." }]]);
    // The mode ceiling, read-only.
    const ceiling = main.querySelector('[data-ceiling="mode"]');
    assert.ok(ceiling.classList.contains("panel--info"));
    assert.match(norm(ceiling.textContent), /Mode ceiling: assisted/);
    assert.equal(ceiling.querySelectorAll("input, select, textarea, button").length, 0);
    // Mode and confirm selects.
    assert.equal(modeSelect(main, "zip").value, "", "inherit");
    assert.equal(modeSelect(main, "ssn4").value, "generative");
    assert.deepEqual(modeSelect(main, "zip").querySelectorAll("option").map((o) => o.value), ["", "deterministic", "assisted", "generative"]);
    assert.equal(confirmSelect(main, "zip").value, "when_inferred");
    assert.equal(confirmSelect(main, "ssn4").value, "never");
    assert.deepEqual(confirmSelect(main, "card").querySelectorAll("option").map((o) => o.value), ["when_inferred", "always", "never"]);
    // The effective modes, per flow, with where the configured one came from and the cap.
    const zip = effectiveLines(main, "zip");
    assert.deepEqual(zip.map((d) => d.getAttribute("data-flow")), ["login", "stepup"]);
    assert.deepEqual(zip[0].querySelectorAll(".badge").map((b) => norm(b.textContent)),
      ["Configured: deterministic (the flow's setting)", "Effective: deterministic"]);
    const ssn4 = effectiveLines(main, "ssn4");
    assert.deepEqual(ssn4[0].querySelectorAll(".badge").map((b) => norm(b.textContent)),
      ["Configured: generative (set on the slot)", "Effective: assisted", "Capped by deployment (ceiling assisted)"]);
    assert.ok(ssn4[0].querySelector(".badge--warning"));
    // slot_resolution.
    const form = resForm(main);
    assert.ok(form.closest(".card"));
    assert.equal(resField(main, "default_mode").value, "assisted");
    assert.equal(resField(main, "flows.login.mode").value, "deterministic");
    assert.equal(resField(main, "flows.stepup.mode").value, "");
    assert.equal(resField(main, "flows.confirm.mode").value, "");
    assert.equal(resField(main, "resolver_budget_ms").value, "1500");
    const gen = resField(main, "generative_enabled");
    assert.equal(gen.getAttribute("type"), "checkbox");
    assert.equal(gen.checked, false);
    assert.match(norm(form.textContent), /no longer validated catalog text/);
    assert.ok(form.querySelector("[data-generative-no-effect]"), "the ceiling isn't generative");
    assert.match(norm(form.querySelector("[data-generative-no-effect]").textContent), new RegExp(GENERATIVE_NO_EFFECT));
    // The options of the choice slot: spoken names and aliases per language.
    const options = main.querySelector('[data-options-for="card"]');
    assert.ok(options.classList.contains("card"));
    const red = options.querySelector('form[data-spoken="option.card.red"]');
    assert.equal(red.querySelector('input[data-field="texts[en-US]"]').value, "Red card");
    assert.equal(red.querySelector('input[data-field="texts[es-US]"]').value, "tarjeta roja");
    assert.match(norm(red.querySelector("legend").textContent), /red \(key 1\)/);
    const aliases = options.querySelector('form[data-aliases="card"]');
    assert.equal(aliases.querySelector('input[data-field="slot#card.options[1].aliases.es-US"]').value, "morada");
    assert.ok(aliases.querySelector('fieldset[data-field="slot#card.options.aliases.en-US"]'));
    // No options card for the other slots.
    assert.equal(main.querySelectorAll("[data-options-for]").length, 1);
    assertViewStructure(main);
  });
});

test("generative_enabled: admins only, with the who and when; no effect shown unless the ceiling is generative", async () => {
  const set = { value: true, by: { sub: "u-1", email: "admin@example.com" }, at: "2026-10-20T10:00:00Z" };
  for (const ceiling of ["deterministic", "assisted", "generative"]) {
    for (const user of [ADMIN, EDITOR, VIEWER]) {
      const payload = { ...P2, ceiling, slot_resolution: { ...RESOLUTION, generative_enabled: true, generative_enabled_set: set } };
      const api = fakeApi(p2Routes(payload));
      await renderUnder(async ({ main }) => {
        await run(api, user)(main);
        const control = resField(main, "generative_enabled");
        const form = resForm(main);
        if (user === ADMIN) {
          assert.ok(control, "admin sees the control");
          assert.equal(control.checked, true);
          assert.match(norm(form.textContent), /no longer validated/);
        } else {
          assert.equal(control, null, "a non-admin sees no generative_enabled control");
          assert.equal(form.querySelector("[data-generative]").querySelectorAll("input, select, textarea, button").length, 0);
          assert.match(norm(form.querySelector('[data-generative="read-only"]').textContent), /Generative mode: On\. Only admins can change it\./);
        }
        assert.match(norm(form.textContent), /Last set to on by admin@example\.com at 2026-10-20T10:00:00Z/);
        assert.equal(Boolean(form.querySelector("[data-generative-no-effect]")), ceiling !== "generative");
        assertViewStructure(main);
      });
    }
  }
});

test("a viewer sees every phase 2 control disabled and no Save buttons", async () => {
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main }) => {
    await run(api, VIEWER)(main);
    const controls = main.querySelectorAll("select, input");
    assert.ok(controls.length > 12);
    assert.ok(controls.every((c) => c.disabled));
    assert.equal(main.querySelectorAll('button[type="submit"]').length, 0);
  });
});

test("a phase 1 payload (no ceiling, no slot_resolution) renders without the phase 2 cards", async () => {
  const api = fakeApi(slotRoutes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    assert.equal(main.querySelector('[data-ceiling="mode"]'), null);
    assert.equal(resForm(main), null);
    assert.equal(main.querySelectorAll("select").length, 0);
    assertViewStructure(main);
  });
});

// ---------------------------------------------------------------- a slot's mode: preview, confirm, PUT

test("a mode change shows the preview in a danger confirmation, then PUTs only {rev, mode}", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api, EDITOR)(main);
    choose(modeSelect(main, "yes_no"), "assisted");
    await flush();
    assert.deepEqual(posts(api), [["post", "slots/preview", { slots: { yes_no: { mode: "assisted" } } }]]);
    assert.equal(confirm.open, true);
    assert.deepEqual(puts(api), [], "nothing saved before the confirmation");
    const text = norm(confirm.textContent);
    assert.match(text, /yes_no in confirm \(assisted\)/, "which slots and flows would call a model");
    assert.match(text, /Moving away from deterministic is a security change: server text\./);
    assert.match(text, /security change/);
    assert.ok(confirm.querySelector(".dialog-header svg.icon"), "the danger variant");
    assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent),
      ["yes_no.mode: inherit -> assisted", "yes_no (confirm) effective: deterministic -> assisted"]);
    confirmButton(confirm).click();
    await flush();
    assert.deepEqual(puts(api), [["put", "slots/yes_no", { rev: 2, mode: "assisted" }]]);
    assert.equal(modeSelect(main, "yes_no").value, "assisted");
  });
});

test("a mode change the preview doesn't flag uses the plain confirmation; Inherit sends mode: null", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const preview = () => ({ ceiling: "assisted", current: {}, proposed: {}, calls_model: [], changes: [], warnings: [] });
  const api = fakeApi(p2Routes(P2, { preview }));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    choose(modeSelect(main, "ssn4"), "");
    await flush();
    assert.equal(confirm.open, true);
    assert.equal(confirm.querySelector("button.danger"), null);
    assert.match(norm(confirm.textContent), /no slot would call a model/);
    primaryButton(confirm).click();
    await flush();
    assert.deepEqual(puts(api), [["put", "slots/ssn4", { rev: 1, mode: null }]]);
  });
});

test("Cancel on a mode change sends no PUT and resets the select", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    choose(modeSelect(main, "zip"), "deterministic");
    await flush();
    cancelButton(confirm).click();
    await flush();
    assert.deepEqual(puts(api), []);
    assert.equal(modeSelect(main, "zip").value, "");
  });
});

test("a 400 from the preview marks the mode select and opens no confirmation", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const preview = () => { throw new ApiError(mapError(400, { error: "invalid", message: "generative needs a non-sensitive choice slot", field: "slot#ssn4.mode" })); };
  const api = fakeApi(p2Routes(P2, { preview }));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    const sel = modeSelect(main, "zip");
    choose(modeSelect(main, "ssn4"), "assisted");
    await flush();
    assert.equal(confirm.open, false);
    assert.equal(modeSelect(main, "ssn4").getAttribute("aria-invalid"), "true");
    assert.equal(sel.getAttribute("aria-invalid"), null);
    assert.equal(modeSelect(main, "ssn4").value, "generative", "reset");
    assert.deepEqual(puts(api), []);
    assertViewStructure(main);
  });
});

test("a 409 on a mode PUT shows the compare cards", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = p2Routes();
  const current = { ...P2_YES_NO, mode: "deterministic", rev: 7 };
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(409, { error: "conflict", message: "the slot changed since it was read", current }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    choose(modeSelect(main, "yes_no"), "assisted");
    await flush();
    confirmButton(confirm).click();
    await flush();
    assert.ok(main.querySelector(".panel--error .compare"));
    assert.match(norm(main.querySelector(".compare").textContent), /mode/);
    assert.equal(modeSelect(main, "yes_no").value, "");
  });
});

test("a choice slot's confirm select has the \"Never\" hint next to it; digit and yes/no slots don't", async () => {
  assert.equal(CHOICE_NEVER_HINT,
    "Never: skipped only for a keypad press or an exact name; a model match is always confirmed");
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const select = confirmSelect(main, "card");
    const hint = main.querySelector('[data-confirm-hint="card"]');
    assert.ok(hint, "the hint for the card slot");
    assert.ok(hint.classList.contains("hint"));
    assert.equal(norm(hint.textContent), CHOICE_NEVER_HINT);
    assert.equal(hint.closest("td"), select.closest("td"), "in the same cell as the select");
    assert.equal(select.getAttribute("aria-describedby"), hint.id, "linked to the select");
    for (const name of ["zip", "ssn4", "yes_no"]) {
      assert.equal(main.querySelector(`[data-confirm-hint="${name}"]`), null, name);
      assert.equal(confirmSelect(main, name).getAttribute("aria-describedby"), null, name);
    }
    // Shown whatever the current value is, and the select still saves as before.
    assert.equal(select.value, "when_inferred");
    assertViewStructure(main);
  });
});

test("the confirm select saves at once with {rev, confirm}", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api, EDITOR)(main);
    choose(confirmSelect(main, "zip"), "always");
    await flush();
    assert.equal(confirm.open, false);
    assert.deepEqual(posts(api), []);
    assert.deepEqual(puts(api), [["put", "slots/zip", { rev: 3, confirm: "always" }]]);
    // The next save uses the response's rev.
    choose(modeSelect(main, "zip"), "deterministic");
    await flush();
    confirmButton(confirm).click(); // PREVIEW warns away_from_deterministic: the danger variant
    await flush();
    assert.deepEqual(puts(api).at(-1), ["put", "slots/zip", { rev: 4, mode: "deterministic" }]);
  });
});

test("the response's warnings are shown after a mode save", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const putWarnings = (name) => [{ code: "away_from_deterministic", slot: name, message: "server away text" }];
  const api = fakeApi(p2Routes(P2, { putWarnings }));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    choose(modeSelect(main, "yes_no"), "assisted");
    await flush();
    confirmButton(confirm).click();
    await flush();
    assert.deepEqual(main.querySelector('[data-save-warnings="yes_no"]').querySelectorAll("li").map((li) => norm(li.textContent)),
      ["yes_no: server away text"]);
  });
});

// ---------------------------------------------------------------- slot_resolution

test("slot_resolution: a default_mode change previews, confirms, PUTs only {rev, default_mode} and rereads the slots", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const fresh = { ...P2, slots: [{ ...P2_ZIP, modes: { login: decision("deterministic", "default", "assisted", "deterministic") } }, P2_SSN4, P2_YES_NO, P2_CARD] };
  const preview = () => ({ ...PREVIEW, warnings: [] });
  const api = fakeApi(p2Routes(P2, { preview, fresh }));
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    resField(main, "default_mode").value = "deterministic";
    submit(resForm(main));
    await flush();
    assert.deepEqual(posts(api), [["post", "slots/preview", { slot_resolution: { default_mode: "deterministic" } }]]);
    assert.equal(confirm.open, true);
    assert.deepEqual(confirm.querySelectorAll("li").map((li) => li.textContent)[0], "default_mode: assisted -> deterministic");
    primaryButton(confirm).click();
    await flush();
    assert.deepEqual(puts(api), [["put", "slots/resolution", { rev: 2, default_mode: "deterministic" }]]);
    assert.deepEqual(api.calls.at(-1), ["get", "slots"], "the effective modes are read again, not recomputed");
    assert.deepEqual(effectiveLines(main, "zip").map((d) => d.getAttribute("data-source")), ["default"]);
    // The card follows the response (its rev is 3 now).
    resField(main, "resolver_budget_ms").value = "2000";
    submit(resForm(main));
    await flush();
    assert.deepEqual(puts(api).at(-1), ["put", "slots/resolution", { rev: 3, resolver_budget_ms: 2000 }]);
  });
});

test("slot_resolution: a budget-only change needs no preview; flows are sent whole; nothing changed sends nothing", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main, confirm }) => {
    await run(api, EDITOR)(main);
    submit(resForm(main));
    await flush();
    assert.deepEqual(puts(api), []);
    assert.deepEqual(posts(api), []);
    assert.match(norm(resForm(main).textContent), /Nothing to save/);
    resField(main, "resolver_budget_ms").value = "2500";
    submit(resForm(main));
    await flush();
    assert.equal(confirm.open, false);
    assert.deepEqual(posts(api), []);
    assert.deepEqual(puts(api), [["put", "slots/resolution", { rev: 2, resolver_budget_ms: 2500 }]]);
    resField(main, "flows.stepup.mode").value = "assisted";
    submit(resForm(main));
    await flush();
    assert.deepEqual(posts(api).at(-1), ["post", "slots/preview", { slot_resolution: { flows: { login: { mode: "deterministic" }, stepup: { mode: "assisted" } } } }]);
    confirmButton(confirm).click();
    await flush();
    assert.deepEqual(puts(api).at(-1), ["put", "slots/resolution", { rev: 3, flows: { login: { mode: "deterministic" }, stepup: { mode: "assisted" } } }]);
  });
});

test("slot_resolution: a budget that isn't a whole number is marked and not sent", async () => {
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const budget = resField(main, "resolver_budget_ms");
    budget.value = "1.5e3";
    submit(resForm(main));
    await flush();
    assert.equal(budget.getAttribute("aria-invalid"), "true");
    assert.deepEqual(puts(api), []);
    assertViewStructure(main);
  });
});

test("slot_resolution: an admin turning generative on gets the danger confirmation with the wording warning", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  for (const answer of ["cancel", "confirm"]) {
    const preview = () => ({ ceiling: "assisted", current: {}, proposed: {}, calls_model: [], changes: [],
      warnings: [{ code: "generative_no_effect", slot: "slot_resolution", message: "server no-effect text" }] });
    const api = fakeApi(p2Routes(P2, { preview }));
    await renderUnder(async ({ main, confirm }) => {
      await run(api, ADMIN)(main);
      resField(main, "generative_enabled").checked = true;
      submit(resForm(main));
      await flush();
      assert.deepEqual(posts(api), [["post", "slots/preview", { slot_resolution: { generative_enabled: true } }]]);
      assert.equal(confirm.open, true);
      assert.ok(confirmButton(confirm), "danger");
      assert.ok(norm(confirm.textContent).includes(GENERATIVE_WARNING));
      assert.match(norm(confirm.textContent), /server no-effect text/);
      (answer === "confirm" ? confirmButton(confirm) : cancelButton(confirm)).click();
      await flush();
      assert.deepEqual(puts(api), answer === "confirm" ? [["put", "slots/resolution", { rev: 2, generative_enabled: true }]] : [],
        "Cancel sends nothing");
    });
  }
});

test("slot_resolution: a 409 shows the compare cards", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = p2Routes();
  const current = { ...RESOLUTION, resolver_budget_ms: 3000, rev: 4 };
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(409, { error: "conflict", message: "slot_resolution changed since it was read", current }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    resField(main, "resolver_budget_ms").value = "2000";
    submit(resForm(main));
    await flush();
    const compare = resForm(main).querySelector(".panel--error .compare");
    assert.ok(compare, "the Current and Yours cards");
    assert.match(norm(compare.textContent), /resolver_budget_ms/);
    const reload = main.querySelectorAll("button").find((b) => norm(b.textContent) === "Reload current value");
    reload.click();
    await flush();
    assert.deepEqual(api.calls.at(-1)[1], "prompts", "the view loads again");
  });
});

test("slot_resolution: a 400 naming a flow marks that select", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = p2Routes();
  const api = fakeApi((m, p, a) => {
    if (m === "post") throw new ApiError(mapError(400, { error: "invalid", message: "must be one of the modes", field: "slot_resolution.flows.confirm.mode" }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main, confirm }) => {
    await run(api)(main);
    resField(main, "flows.confirm.mode").value = "generative";
    submit(resForm(main));
    await flush();
    assert.equal(confirm.open, false);
    assert.equal(resField(main, "flows.confirm.mode").getAttribute("aria-invalid"), "true");
    assert.deepEqual(puts(api), []);
    assertViewStructure(main);
  });
});

// ---------------------------------------------------------------- options: spoken names and aliases

test("spoken names: PUT /prompts/option.<slot>.<value> with the prompt's rev and every language", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main }) => {
    await run(api, EDITOR)(main);
    const form = main.querySelector('form[data-spoken="option.card.red"]');
    submit(form);
    await flush();
    assert.deepEqual(puts(api), [], "nothing changed: nothing sent");
    type(form.querySelector('input[data-field="texts[es-US]"]'), "la tarjeta roja");
    submit(form);
    await flush();
    assert.deepEqual(puts(api), [["put", "prompts/option.card.red", { rev: 3, texts: { "en-US": "Red card", "es-US": "la tarjeta roja" } }]]);
    type(form.querySelector('input[data-field="texts[en-US]"]'), "The red card");
    submit(form);
    await flush();
    assert.deepEqual(puts(api).at(-1), ["put", "prompts/option.card.red", { rev: 4, texts: { "en-US": "The red card", "es-US": "la tarjeta roja" } }]);
  });
});

test("spoken names: a 400 naming texts[<lang>] marks that language's field", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = p2Routes();
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(400, { error: "invalid", message: "a spoken name or alias would belong to more than one option (purple, red)", field: "texts[en-US]" }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const form = main.querySelector('form[data-spoken="option.card.red"]');
    type(form.querySelector('input[data-field="texts[en-US]"]'), "purple");
    submit(form);
    await flush();
    assert.equal(form.querySelector('input[data-field="texts[en-US]"]').getAttribute("aria-invalid"), "true");
    assert.equal(main.querySelector('form[data-spoken="option.card.purple"] input[data-field="texts[en-US]"]').getAttribute("aria-invalid"), null);
    assertViewStructure(main);
  });
});

test("spoken names are read-only when their prompts can't be read", async () => {
  const base = p2Routes();
  const api = fakeApi((m, p, a) => {
    if (p === "prompts") throw new ApiError(mapError(500, {}));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api, EDITOR)(main);
    const form = main.querySelector('form[data-spoken="option.card.red"]');
    assert.ok(form.querySelectorAll("input").every((i) => i.disabled));
    assert.equal(form.querySelector('button[type="submit"]'), null);
    assert.match(norm(main.querySelector('[data-options-for="card"]').textContent), /couldn't be read/);
    assertViewStructure(main);
  });
});

test("aliases: the PUT holds only {rev, options}, with each option's aliases per language and its dtmf", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const api = fakeApi(p2Routes());
  await renderUnder(async ({ main }) => {
    await run(api, EDITOR)(main);
    const form = main.querySelector('form[data-aliases="card"]');
    submit(form);
    await flush();
    assert.deepEqual(puts(api), [], "nothing changed: nothing sent");
    type(form.querySelector('input[data-field="slot#card.options[0].aliases.en-US"]'), "red,  crimson , ");
    submit(form);
    await flush();
    assert.deepEqual(puts(api), [["put", "slots/card", { rev: 5, options: [
      { value: "red", aliases: { "en-US": ["red", "crimson"], "es-US": ["roja"] }, dtmf: "1" },
      { value: "purple", aliases: { "en-US": ["purple"], "es-US": ["morada"] } },
    ] }]]);
    // The row follows the response's rev.
    toggle(switchOf(main, "card", "barge_in"));
    await flush();
    assert.deepEqual(puts(api).at(-1), ["put", "slots/card", { rev: 6, barge_in: false }]);
  });
});

test("aliases: an alias conflict's 400 is shown on its language's field", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const base = p2Routes();
  const api = fakeApi((m, p, a) => {
    if (m === "put") throw new ApiError(mapError(400, { error: "invalid", message: "a spoken name or alias belongs to more than one option (purple, red)", field: "slot#card.options.aliases.es-US" }));
    return base(m, p, a);
  });
  await renderUnder(async ({ main }) => {
    await run(api)(main);
    const form = main.querySelector('form[data-aliases="card"]');
    type(form.querySelector('input[data-field="slot#card.options[0].aliases.es-US"]'), "roja, morada");
    submit(form);
    await flush();
    const es = form.querySelector('fieldset[data-field="slot#card.options.aliases.es-US"]');
    assert.equal(es.getAttribute("aria-invalid"), "true");
    const msg = main.ownerDocument.getElementById(es.getAttribute("aria-describedby"));
    assert.match(norm(msg.textContent), /belongs to more than one option \(purple, red\)/);
    assert.equal(form.querySelector('fieldset[data-field="slot#card.options.aliases.en-US"]').getAttribute("aria-invalid"), null);
    assert.equal(form.querySelectorAll(".panel--error").length, 0, "on the field, not as a panel");
    assertViewStructure(main);
  });
});

// ---------------------------------------------------------------- phase 2 generators

const MODE_VALUES = ["deterministic", "assisted", "generative"];
const SOURCES = ["slot", "flow", "default", "invalid"];
const CAPS = [null, "ceiling", "generative_disabled", "type", "sensitive"];
const FLOWS_ALL = ["login", "stepup", "confirm"];

/** Any decision: not necessarily one the front door would make, so a recomputing view fails. */
const decisionGen = record({ configured: oneOf(...MODE_VALUES), source: oneOf(...SOURCES), ceiling: oneOf(...MODE_VALUES),
  effective: oneOf(...MODE_VALUES), capped: oneOf(...CAPS) });

const optionGen = (value) => record({
  value, dtmf: oneOf(null, "1", "2", "9"),
  spoken_names: gen((rng) => Object.fromEntries(LANGS.filter(() => int(0, 3)(rng) > 0).map((l) => [l, unicodeString({ minLength: 1, maxLength: 8 })(rng)]))),
  aliases: gen((rng) => Object.fromEntries(LANGS.filter(() => int(0, 1)(rng) > 0).map((l) => [l, array(string({ minLength: 1, maxLength: 6 }), { maxLength: 2 })(rng)]))),
});

const p2SlotGen = (name) => gen((rng) => {
  const base = slotGen(name)(rng);
  if (!base.loaded) return { ...base, mode: null, confirm: null, options: null, modes: {} };
  const flows = FLOWS_ALL.filter(() => int(0, 2)(rng) > 0);
  const choice = name === "card" || base.type === "choice";
  return {
    ...base,
    type: choice ? "choice" : base.type,
    mode: oneOf(null, ...MODE_VALUES)(rng),
    confirm: oneOf("when_inferred", "always", "never")(rng),
    modes: Object.fromEntries(flows.map((f) => [f, decisionGen(rng)])),
    options: choice ? ["red", "purple", "rewards"].filter(() => int(0, 2)(rng) > 0).map((v) => optionGen(v)(rng)) : null,
    warnings: array(oneOf(
      record({ code: "verification_not_sensitive", slot: name, message: unicodeString({ minLength: 1, maxLength: 10 }) }),
      record({ code: "generative_no_effect", slot: name, message: unicodeString({ minLength: 1, maxLength: 10 }) })), { maxLength: 2 })(rng),
  };
});
const resolutionGen = record({
  rev: int(0, 20), default_mode: oneOf(...MODE_VALUES),
  flows: gen((rng) => Object.fromEntries(FLOWS_ALL.filter(() => int(0, 1)(rng) > 0).map((f) => [f, oneOf(...MODE_VALUES)(rng)]))),
  generative_enabled: oneOf(true, false), resolver_budget_ms: int(500, 4000),
  problems: array(unicodeString({ minLength: 1, maxLength: 8 }), { maxLength: 1 }),
  generative_enabled_set: oneOf(null, { value: true, by: { email: "a@example.com" }, at: "2026-10-20T10:00:00Z" }, { value: false, by: {}, at: null }),
  warnings: array(record({ code: "generative_no_effect", slot: "slot_resolution", message: unicodeString({ minLength: 1, maxLength: 10 }) }), { maxLength: 1 }),
});

const p2PayloadGen = gen((rng) => {
  const names = NAMES.filter(() => int(0, 2)(rng) > 0);
  return {
    acceptSpokenFactors: oneOf(true, false)(rng),
    takes_effect_s: 60,
    ceiling: oneOf(...MODE_VALUES)(rng),
    languages: oneOf(LANGS, ["en-US"], [])(rng),
    slot_resolution: resolutionGen(rng),
    slots: names.map((n) => p2SlotGen(n)(rng)),
  };
});

/** A route answering GET /slots with the payload and GET /prompts from its options. */
function genRoutes(payload) {
  const base = p2Routes(payload);
  return (m, p, a) => (m === "get" && p === "slots" ? structuredClone(payload) : base(m, p, a));
}

// ---------------------------------------------------------------- property: effective-mode badges

test("PBT: the effective-mode badges equal the payload's table, never recomputed", async () => {
  await forAll(p2PayloadGen, async (payload) => {
    const api = fakeApi(genRoutes(payload));
    await renderUnder(async ({ main }) => {
      await run(api)(main);
      for (const s of payload.slots) {
        const cell = main.querySelector(`td[data-effective-for="${s.name}"]`);
        const lines = cell.querySelectorAll("div[data-flow]");
        const table = s.loaded ? Object.entries(s.modes) : [];
        assert.deepEqual(lines.map((d) => d.getAttribute("data-flow")), table.map(([f]) => f), s.name);
        table.forEach(([, d], i) => {
          const line = lines[i];
          assert.equal(line.getAttribute("data-configured"), d.configured);
          assert.equal(line.getAttribute("data-source"), d.source);
          assert.equal(line.getAttribute("data-effective"), d.effective);
          assert.equal(line.getAttribute("data-capped"), d.capped ?? "-");
          const texts = line.querySelectorAll(".badge").map((b) => norm(b.textContent));
          assert.equal(texts.length, d.capped ? 3 : 2);
          assert.ok(texts[0].startsWith(`Configured: ${d.configured} (`), texts[0]);
          assert.equal(texts[1], `Effective: ${d.effective}`);
          if (d.capped === "ceiling") assert.equal(texts[2], `Capped by deployment (ceiling ${d.ceiling})`);
          else if (d.capped) assert.ok(texts[2].startsWith("Capped"), texts[2]);
        });
        if (s.loaded) assert.equal(modeSelect(main, s.name).value, s.mode ?? "");
      }
    });
  });
});

// ---------------------------------------------------------------- PBT: a resolution PUT is the changed fields and rev

test("PBT: a slot_resolution save PUTs exactly rev and the changed fields; Cancel sends nothing", async () => {
  const caseGen = record({
    payload: p2PayloadGen, user: oneOf(ADMIN, EDITOR),
    def: oneOf(null, ...MODE_VALUES), flow: oneOf(null, ...FLOWS_ALL), flowMode: oneOf("", ...MODE_VALUES),
    budget: oneOf(null, int(500, 4000)), generative: oneOf(null, true, false), answer: oneOf("confirm", "cancel"),
  });
  await forAll(caseGen, async ({ payload, user, def, flow, flowMode, budget, generative, answer }) => {
    const r = payload.slot_resolution;
    const preview = () => ({ ceiling: payload.ceiling, current: {}, proposed: {}, calls_model: [], changes: [], warnings: [] });
    const api = fakeApi((m, p, a) => (m === "get" && p === "slots" ? structuredClone(payload) : p2Routes(payload, { preview })(m, p, a)));
    await renderUnder(async ({ main, confirm }) => {
      await run(api, user)(main);
      if (!payload.slots.length) {
        // No slots: the view shows only the "No slots" empty state (seed the table first), so there
        // is no resolution form to save. Found by seed 1209297847 (the intermittent console failure).
        assert.equal(resField(main, "default_mode"), null);
        assert.ok(main.textContent.includes("No slots"));
        return;
      }
      if (def) resField(main, "default_mode").value = def;
      if (flow) resField(main, `flows.${flow}.mode`).value = flowMode;
      if (budget !== null) resField(main, "resolver_budget_ms").value = String(budget);
      if (generative !== null && user === ADMIN) resField(main, "generative_enabled").checked = generative;
      submit(resForm(main));
      await flush();
      const expected = {};
      if (def && def !== r.default_mode) expected.default_mode = def;
      if (flow && (r.flows[flow] ?? "") !== flowMode) {
        const next = { ...r.flows, [flow]: flowMode };
        expected.flows = Object.fromEntries(FLOWS_ALL.filter((f) => next[f]).map((f) => [f, { mode: next[f] }]));
      }
      if (budget !== null && budget !== r.resolver_budget_ms) expected.resolver_budget_ms = budget;
      if (generative !== null && user === ADMIN && generative !== r.generative_enabled) expected.generative_enabled = generative;
      const asks = ["default_mode", "flows", "generative_enabled"].some((k) => Object.hasOwn(expected, k));
      assert.equal(confirm.open, asks);
      if (asks) {
        const modeFields = Object.fromEntries(Object.entries(expected).filter(([k]) => k !== "resolver_budget_ms"));
        assert.deepEqual(posts(api), [["post", "slots/preview", { slot_resolution: modeFields }]]);
        (answer === "confirm" ? (confirm.querySelector("button.danger") || primaryButton(confirm)) : cancelButton(confirm)).click();
        await flush();
      }
      const sent = !Object.keys(expected).length || (asks && answer === "cancel") ? [] : [["put", "slots/resolution", { rev: r.rev, ...expected }]];
      assert.deepEqual(puts(api), sent);
      assert.ok(puts(api).every((c) => user === ADMIN || !Object.hasOwn(c[2], "generative_enabled")));
    });
  }, { runs: 150 });
});

// ---------------------------------------------------------------- PBT (phase 2)

test("PBT: the phase 2 view for generated payloads and users", async () => {
  await forAll(record({ payload: p2PayloadGen, user: userGen }), async ({ payload, user }) => {
    const api = fakeApi(genRoutes(payload));
    await renderUnder(async ({ document, main }) => {
      await run(api, user)(main);
      assertNoInlineStyle(document.body);
      assertClassesStyled(document.body);
      assertViewStructure(main);
    });
  });
});

test("PBT: with the preview confirmation open, and after a 400 or 409 on an options save", async () => {
  const caseGen = record({ payload: p2PayloadGen, outcome: oneOf("preview", "400", "409") });
  await forAll(caseGen, async ({ payload, outcome }) => {
    const target = payload.slots.find((s) => s.loaded);
    const choice = payload.slots.find((s) => s.loaded && Array.isArray(s.options) && s.options.length);
    const base = genRoutes(payload);
    const api = fakeApi((m, p, a) => {
      if (m === "put" && outcome === "409") throw new ApiError(mapError(409, { error: "conflict", current: choice }));
      if (m === "put" && outcome === "400") {
        throw new ApiError(mapError(400, { error: "invalid", message: "conflict", field: `slot#${choice.name}.options.aliases.${payload.languages[0] ?? "en-US"}` }));
      }
      return base(m, p, a);
    });
    await renderUnder(async ({ document, main }) => {
      await run(api)(main);
      if (outcome === "preview" && target) {
        choose(modeSelect(main, target.name), (target.mode ?? "") === "assisted" ? "deterministic" : "assisted");
        await flush();
      } else if (choice) {
        const form = main.querySelector(`form[data-aliases="${choice.name}"]`);
        const input = form.querySelector("input");
        if (input) {
          type(input, `${input.value}, extra`);
          submit(form);
          await flush();
        }
      }
      assertNoInlineStyle(document.body);
      assertClassesStyled(document.body);
      assertViewStructure(main);
    });
  }, { runs: 100 });
});
