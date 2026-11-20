// Save, load, error and dialog feedback in lib/ui.js, and the field pieces of lib/dom.js
// (Field, Select, Dialog).
// Examples: save's toasts and inline detail for each ApiError kind, the 409 comparison, a 400
// with and without a matching data-field, load's skeleton and failure, confirmDialog (confirm,
// Cancel, Esc, danger), field()'s required marker, form note and select chevron.
// PBT: a second submit while saving is ignored; the busy state comes and goes.
// Examples per control kind: markInvalid / clearInvalid.
import assert from "node:assert/strict";
import { test } from "node:test";

import { Event, install, uninstall } from "./fakedom.js";
import { forAll, int, oneOf, record } from "./gen.js";
import { mapError } from "../lib/api.js";
import { clearInvalid, field, h, markInvalid } from "../lib/dom.js";
import { confirmDialog, errorPanel, load, save, sourceMark, takesEffect } from "../lib/ui.js";

/** A fresh fakedom holding #main, #toasts, #confirm and #live; async-aware. */
async function withDom(fn) {
  const env = install();
  try {
    env.document.body.append(h("main#main"), h("div#toasts.toasts"), h("dialog#confirm.dialog"),
      h("div#live.visually-hidden", { role: "status", "aria-live": "polite" }));
    return await fn(env);
  } finally {
    uninstall();
  }
}

/** Toasts start timers (and announce does): keep them on the mock clock. */
const mockClock = (t) => t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });

const all = (root) => [root, ...root.querySelectorAll("*")];
const text = (el) => el.textContent.trim();
const iconName = (svg) => svg.querySelector("use").getAttribute("href").replace("icons.svg#", "");
const toasts = () => document.getElementById("toasts").querySelectorAll(".toast");
const noStyle = (root) => all(root).forEach((el) => {
  assert.ok(!el.hasAttribute("style"), `${el.tagName} has a style attribute`);
  for (const a of el.attributes) assert.ok(!/^on/i.test(a.name), `${el.tagName} has ${a.name}`);
});

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

// ---------------------------------------------------------------- save: success

test("save success: a success toast whose message is the takes-effect text; status is cleared", async (t) => {
  mockClock(t);
  await withDom(async () => {
    const status = h("div", {}, h("p", {}, "an old error"));
    const res = await save(status, async () => ({ changed: true, takes_effect_s: 60 }));
    assert.deepEqual(res, { changed: true, takes_effect_s: 60 });
    assert.equal(status.childNodes.length, 0);
    const [tst] = toasts();
    assert.equal(toasts().length, 1);
    assert.ok(tst.classList.contains("toast--success"));
    assert.equal(text(tst.querySelector(".toast-title")), "Saved.");
    assert.equal(text(tst.querySelector(".toast-message")), "It takes effect within 60 seconds.");
  });
});

test("save success with changed: false says there was no change; no takes_effect_s, no message", async (t) => {
  mockClock(t);
  await withDom(async () => {
    const status = h("div");
    await save(status, async () => ({ changed: false }));
    const [tst] = toasts();
    assert.equal(text(tst.querySelector(".toast-title")), "No change: the values are the same.");
    assert.equal(tst.querySelector(".toast-message"), null);
  });
});

// ---------------------------------------------------------------- save: errors

const KINDS = [
  [400, { error: "invalid", message: "min_level must be 0-3" }],
  [401, { error: "unauthorized" }],
  [403, { error: "forbidden" }],
  [403, { error: "self_delete" }],
  [404, { error: "not_found" }],
  [409, { error: "conflict" }], // no current: shown as invalid
  [429, {}],
  [500, {}],
  [503, {}],
  [0, {}],
];

for (const [status, body] of KINDS) {
  test(`save error ${status} ${body.error ?? ""}: an error toast with the title and reason, and an inline error panel`, async (t) => {
    mockClock(t);
    await withDom(async () => {
      const err = mapError(status, body);
      const box = h("div");
      const res = await save(box, async () => { throw err; });
      assert.equal(res, null);
      const [tst] = toasts();
      assert.equal(toasts().length, 1);
      assert.ok(tst.classList.contains("toast--error"));
      assert.equal(text(tst.querySelector(".toast-title")), err.title);
      assert.equal(text(tst.querySelector(".toast-message")), err.message);
      const p = box.querySelector(".panel.panel--error");
      assert.ok(p, "an inline error panel");
      assert.equal(p.getAttribute("role"), "alert");
      assert.equal(text(p.querySelector(".panel-title")), err.title);
      assert.ok(text(p).includes(err.message));
      noStyle(box);
    });
  });
}

test("save 409 conflict: the panel, the Current and Yours cards, and the reload button", async (t) => {
  mockClock(t);
  await withDom(async () => {
    const err = mapError(409, { error: "conflict", current: { rev: 4, min_level: 3, confirm: "always" } });
    const yours = { rev: 3, min_level: 2, confirm: "always" };
    let reloads = 0;
    const box = h("div");
    assert.equal(await save(box, async () => { throw err; }, { yours, onReload: () => { reloads++; } }), null);
    const p = box.querySelector(".panel--error");
    assert.ok(text(p).includes("Your change was not saved. Here is the current value next to yours; reload, then redo your change."));
    const cards = p.querySelectorAll(".compare > .card");
    assert.equal(cards.length, 2);
    assert.deepEqual(cards.map((c) => text(c.querySelector(".badge"))), ["Current", "Yours"]);
    const changed = p.querySelectorAll(".compare-row.is-changed");
    assert.equal(changed.length, 2); // min_level in each card
    assert.ok(p.querySelector("details.compare-details"));
    const reload = p.querySelectorAll("button").find((b) => text(b) === "Reload current value");
    assert.ok(reload);
    reload.click();
    assert.equal(reloads, 1);
    assert.equal(text(toasts()[0].querySelector(".toast-title")), "Not saved: someone else changed this");
    noStyle(box);
  });
});

test("save 400 with a matching data-field: markInvalid on it and focus; no inline panel", async (t) => {
  mockClock(t);
  await withDom(async ({ document }) => {
    const ctl = field("Minimum level", h("input", { type: "number", "data-field": "min_level" }), "0 to 3.");
    const form = h("form", {}, ctl.wrap);
    const box = h("div");
    document.body.append(form, box);
    const err = mapError(400, { error: "invalid", field: "min_level", message: "must be 0-3" });
    assert.equal(await save(box, async () => { throw err; }, { form }), null);
    assert.equal(ctl.input.getAttribute("aria-invalid"), "true");
    assert.equal(document.activeElement, ctl.input);
    const msg = form.querySelector(".field-error");
    assert.equal(text(msg), "min_level: must be 0-3");
    assert.equal(box.querySelector(".panel"), null);
    assert.equal(text(toasts()[0].querySelector(".toast-message")), "min_level: must be 0-3");
  });
});

test("save 400 naming a field the form doesn't have: the inline panel instead", async (t) => {
  mockClock(t);
  await withDom(async ({ document }) => {
    const ctl = field("Minimum level", h("input", { type: "number", "data-field": "min_level" }));
    const form = h("form", {}, ctl.wrap);
    const box = h("div");
    document.body.append(form, box);
    const err = mapError(400, { error: "invalid", field: "phrase_key", message: "unknown key" });
    await save(box, async () => { throw err; }, { form });
    assert.equal(ctl.input.getAttribute("aria-invalid"), null);
    assert.equal(form.querySelector(".field-error"), null);
    assert.ok(text(box.querySelector(".panel--error")).includes("phrase_key: unknown key"));
  });
});

// ---------------------------------------------------------------- PBT

test("submits during one pending save call fn once; busy while pending, not after", async (t) => {
  mockClock(t);
  await forAll(record({ n: int(2, 10), outcome: oneOf("ok", "error") }), async ({ n, outcome }) => {
    await withDom(async ({ document }) => {
      const button = h("button.primary", { type: "submit" }, "Save");
      const form = h("form", {}, button);
      const status = h("div");
      document.body.append(form, status);
      const d = deferred();
      let calls = 0;
      const runs = [];
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        runs.push(save(status, () => { calls++; return d.promise; }, { button: ev.submitter, form }));
      });
      for (let i = 0; i < n; i++) {
        const ev = new Event("submit", { bubbles: true, cancelable: true });
        ev.submitter = button;
        form.dispatchEvent(ev);
        assert.equal(button.getAttribute("aria-disabled"), "true");
        assert.ok(button.classList.contains("is-busy"));
        assert.equal(button.getAttribute("aria-busy"), "true");
      }
      assert.equal(calls, 1);
      if (outcome === "ok") d.resolve({ changed: true });
      else d.reject(mapError(500, {}));
      const results = await Promise.all(runs);
      assert.equal(calls, 1);
      assert.equal(results.slice(1).every((r) => r === null), true);
      assert.equal(button.hasAttribute("aria-disabled"), false);
      assert.equal(button.classList.contains("is-busy"), false);
      assert.equal(button.hasAttribute("aria-busy"), false);
      assert.equal(text(button), "Save"); // the label is kept
      assert.equal(toasts().length, 1);
    });
  }, { runs: 60 });
});

// ---------------------------------------------------------------- load

test("load: a skeleton and aria-busy while pending, the value after, aria-busy gone", async () => {
  await withDom(async () => {
    const main = document.getElementById("main");
    const d = deferred();
    const p = load(main, () => d.promise);
    assert.equal(main.getAttribute("aria-busy"), "true");
    assert.ok(main.querySelector(".skeleton.skeleton--page"));
    assert.equal(text(main.querySelector("[role=status]")), "Loading…");
    d.resolve({ ok: 1 });
    assert.deepEqual(await p, { ok: 1 });
    assert.equal(main.hasAttribute("aria-busy"), false);
  });
});

test("load: the kind picks the skeleton", async () => {
  await withDom(async () => {
    const box = h("div");
    const d = deferred();
    const p = load(box, () => d.promise, { kind: "table" });
    assert.ok(box.querySelector(".skeleton--table"));
    d.resolve(1);
    await p;
  });
});

test("load failure: an error panel in the container, null returned, aria-busy gone", async () => {
  await withDom(async () => {
    const box = h("div");
    const err = mapError(503, {});
    assert.equal(await load(box, async () => { throw err; }), null);
    const p = box.querySelector(".panel--error");
    assert.equal(text(p.querySelector(".panel-title")), "Not available");
    assert.ok(text(p).includes(err.message));
    assert.equal(box.querySelector(".skeleton"), null);
    assert.equal(box.hasAttribute("aria-busy"), false);
  });
});

test("load failure with a title: the page header (h1#page-title, description, back link), then the error panel", async () => {
  await withDom(async () => {
    const box = h("div");
    const err = mapError(500, {});
    const back = { href: "#/tools", label: "All tools" };
    assert.equal(await load(box, async () => { throw err; }, { title: "Tools", description: "About tools.", back }), null);
    const [header, panelEl, ...rest] = box.children.filter((c) => !c.classList.contains("page-header-sentinel"));
    assert.deepEqual(rest, []);
    assert.ok(header.classList.contains("page-header"));
    const h1 = header.querySelector("h1");
    assert.equal(h1.id, "page-title");
    assert.equal(text(h1), "Tools");
    assert.equal(text(header.querySelector(".page-description")), "About tools.");
    assert.equal(header.querySelector("a.back-link").getAttribute("href"), "#/tools");
    assert.ok(panelEl.classList.contains("panel--error"));
    assert.equal(box.hasAttribute("aria-busy"), false);
  });
});

test("load success with a title: no header is added (the view renders its own)", async () => {
  await withDom(async () => {
    const box = h("div");
    assert.deepEqual(await load(box, async () => ({ ok: 1 }), { title: "Tools" }), { ok: 1 });
    assert.equal(box.querySelector("h1"), null);
  });
});

// ---------------------------------------------------------------- errorPanel, takesEffect, sourceMark

test("errorPanel takes a plain {title, message} as the views pass it", () => withDom(() => {
  const p = errorPanel({ title: "Check the form", message: "min_level: must be 0-3" });
  assert.ok(p.classList.contains("panel") && p.classList.contains("panel--error"));
  assert.equal(p.getAttribute("role"), "alert");
  assert.equal(text(p.querySelector(".panel-title")), "Check the form");
  assert.ok(text(p).includes("min_level: must be 0-3"));
  assert.equal(text(errorPanel(new Error("boom")).querySelector(".panel-title")), "Error");
}));

test("takesEffect: an info note with an icon and the existing text", () => withDom(() => {
  const n = takesEffect(30);
  assert.ok(n.classList.contains("panel--info"));
  assert.equal(n.getAttribute("role"), "note");
  assert.ok(n.querySelector("svg.icon"));
  assert.equal(text(n.querySelector(".panel-body")),
    "Saved changes take effect within 30 seconds, on a call's next turn (the bridge's policy cache).");
  assert.ok(text(takesEffect()).includes("within 60 seconds"));
}));

test("sourceMark: a badge with text", () => withDom(() => {
  const own = sourceMark("tool");
  const def = sourceMark("default");
  assert.ok(own.classList.contains("badge"));
  assert.equal(text(own), "own");
  assert.ok(def.classList.contains("badge"));
  assert.equal(text(def), "default");
}));

// ---------------------------------------------------------------- confirmDialog

const dialog = () => document.getElementById("confirm");
const buttons = () => dialog().querySelectorAll(".dialog-footer > button");

test("confirmDialog: anatomy, Cancel first and focused, confirm resolves true", async () => {
  await withDom(async () => {
    const p = confirmDialog({ title: "Restore?", message: "This writes back.", details: ["min_level: 2 -> 3"], confirmLabel: "Restore" });
    const dlg = dialog();
    assert.equal(dlg.open, true);
    assert.ok(dlg.classList.contains("dialog"));
    assert.equal(dlg.getAttribute("aria-labelledby"), "confirm-title");
    assert.equal(text(dlg.querySelector(".dialog-header > h2#confirm-title")), "Restore?");
    assert.equal(dlg.querySelector(".dialog-header svg.icon"), null);
    assert.equal(text(dlg.querySelector(".dialog-body > p")), "This writes back.");
    assert.equal(text(dlg.querySelector(".dialog-body ul.dialog-details > li > code")), "min_level: 2 -> 3");
    const [cancel, ok] = buttons();
    assert.equal(text(cancel), "Cancel");
    assert.equal(text(ok), "Restore");
    assert.ok(ok.classList.contains("primary"));
    assert.equal(document.activeElement, cancel);
    noStyle(dlg);
    ok.click();
    assert.equal(await p, true);
    assert.equal(dlg.open, false);
  });
});

test("confirmDialog: Cancel resolves false", async () => {
  await withDom(async () => {
    const p = confirmDialog({ title: "Delete?", message: "Gone." });
    buttons()[0].click();
    assert.equal(await p, false);
    assert.equal(dialog().open, false);
  });
});

test("confirmDialog: Esc (the cancel event) resolves false, and a later dialog is unaffected", async () => {
  await withDom(async () => {
    const p = confirmDialog({ title: "Delete?", message: "Gone." });
    dialog().dispatchEvent(new Event("cancel", { cancelable: true }));
    dialog().close(); // the browser closes it after an uncancelled cancel event
    assert.equal(await p, false);
    const q = confirmDialog({ title: "Again?", message: "Sure." });
    buttons()[1].click();
    assert.equal(await q, true);
  });
});

test("confirmDialog danger: the confirm button is .danger and the header has an alert-triangle icon", async () => {
  await withDom(async () => {
    const p = confirmDialog({ title: "This changes security behaviour", message: "m", confirmLabel: "Save security change", danger: true });
    const [, ok] = buttons();
    assert.ok(ok.classList.contains("danger"));
    assert.ok(!ok.classList.contains("primary"));
    assert.equal(iconName(dialog().querySelector(".dialog-header svg.icon")), "alert-triangle");
    ok.click();
    assert.equal(await p, true);
  });
});

// ---------------------------------------------------------------- field()

test("field: a required control gets an aria-hidden * in its label and the form note", () => withDom(() => {
  const f = field("Email", h("input", { type: "email", required: true }), "Their work address.");
  const label = f.wrap.querySelector("label");
  assert.equal(label.getAttribute("for"), f.input.id);
  const mark = label.querySelector(".field-required");
  assert.equal(text(mark), "*");
  assert.equal(mark.getAttribute("aria-hidden"), "true");
  assert.equal(text(f.wrap.querySelector(".form-note")), "* Required");
  assert.equal(f.input.getAttribute("aria-describedby"), `${f.input.id}-hint`);
  const quiet = field("Email", h("input", { type: "email", required: true }), null, { requiredNote: false });
  assert.ok(quiet.wrap.querySelector(".field-required"));
  assert.equal(quiet.wrap.querySelector(".form-note"), null);
  const plain = field("Name", h("input", { type: "text" }));
  assert.equal(plain.wrap.querySelector(".field-required"), null);
  assert.equal(plain.wrap.querySelector(".form-note"), null);
  assert.equal(plain.input.hasAttribute("aria-describedby"), false);
}));

test("field: a select sits in .select-wrap next to a chevron-down icon; other controls don't", () => withDom(() => {
  const f = field("Group", h("select", {}, h("option", { value: "a" }, "a")));
  const wrap = f.wrap.querySelector(".select-wrap");
  assert.ok(wrap);
  assert.equal(wrap.children[0], f.input);
  const chevron = wrap.children[1];
  assert.equal(iconName(chevron), "chevron-down");
  assert.equal(chevron.getAttribute("aria-hidden"), "true");
  assert.equal(field("Name", h("input")).wrap.querySelector(".select-wrap, svg"), null);
  noStyle(f.wrap);
}));

// ---------------------------------------------------------------- examples per control kind

const CONTROLS = {
  text: () => h("input", { type: "text" }),
  number: () => h("input", { type: "number" }),
  email: () => h("input", { type: "email" }),
  textarea: () => h("textarea"),
  select: () => h("select", {}, h("option", { value: "a" }, "a")),
  checkbox: () => h("input", { type: "checkbox" }),
};

for (const [kind, make] of Object.entries(CONTROLS)) {
  for (const withHint of [true, false]) {
    test(`${kind}${withHint ? " with a hint" : ""}: markInvalid, then clearInvalid on the next input`, () => withDom(({ document }) => {
      const f = field(`A ${kind}`, make(), withHint ? "A hint." : null);
      document.body.append(f.wrap);
      const ctl = f.input;
      const before = ctl.getAttribute("aria-describedby");
      markInvalid(ctl, "must be 0-3");
      markInvalid(ctl, "must be 0-3 (again)"); // a second error replaces the first
      assert.equal(ctl.getAttribute("aria-invalid"), "true");
      const msgs = f.wrap.querySelectorAll(".field-error");
      assert.equal(msgs.length, 1);
      const [msg] = msgs;
      assert.ok(msg.id);
      const ids = ctl.getAttribute("aria-describedby").split(/\s+/);
      assert.ok(ids.includes(msg.id));
      if (withHint) {
        assert.ok(ids.includes(before));
        const hint = f.wrap.querySelector(".hint");
        assert.equal(hint.nextSibling, msg, "the message sits below the hint");
      }
      assert.equal(ids.length, withHint ? 2 : 1);
      assert.equal(iconName(msg.querySelector("svg.icon")), "alert-circle");
      assert.equal(text(msg), "must be 0-3 (again)");
      noStyle(f.wrap);

      ctl.dispatchEvent(new Event("input", { bubbles: true }));
      assert.equal(ctl.hasAttribute("aria-invalid"), false);
      assert.equal(ctl.getAttribute("aria-describedby"), before);
      assert.equal(f.wrap.querySelector(".field-error"), null);

      markInvalid(ctl, "again"); // it can be marked again after a clear
      assert.equal(f.wrap.querySelectorAll(".field-error").length, 1);
      clearInvalid(ctl);
      clearInvalid(ctl); // idempotent
      assert.equal(ctl.getAttribute("aria-describedby"), before);
    }));
  }
}

test("markInvalid works on a control outside field()", () => withDom(({ document }) => {
  const input = h("input", { type: "text", "aria-describedby": "elsewhere" });
  const box = h("div", {}, input);
  document.body.append(box);
  markInvalid(input, "bad");
  const msg = box.querySelector(".field-error");
  assert.equal(input.nextSibling, msg);
  assert.deepEqual(input.getAttribute("aria-describedby").split(" "), ["elsewhere", msg.id]);
  input.dispatchEvent(new Event("input"));
  assert.equal(input.getAttribute("aria-describedby"), "elsewhere");
}));
