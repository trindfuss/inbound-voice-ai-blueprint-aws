// The presentational builders of lib/components.js and the
// re-announcing live region (announce in lib/dom.js).
// Examples: each builder's anatomy and ARIA.
// PBT: envBadge (node part), toasts (a state machine under mock.timers) and
// the style check (builder part: no style or on* attribute in any built tree, and every badge has text).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { mock, test } from "node:test";
import { fileURLToPath } from "node:url";

import { install, uninstall, Event } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string, unicodeString } from "./gen.js";
import { announce, ANNOUNCE_DELAY_MS, h } from "../lib/dom.js";
import {
  MAX_TOASTS, TOAST_TIMEOUT_MS, badge, card, compareCards, emptyState, envBadge, langBadge,
  langName, pageHeader, panel, skeleton, tableWrap, toast,
} from "../lib/components.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** Run fn with a fresh fakedom holding #main, #toasts and #live. */
function withDom(fn) {
  const env = install();
  try {
    env.document.body.append(h("main#main"), h("div#toasts.toasts"),
      h("div#live.visually-hidden", { role: "status", "aria-live": "polite" }));
    return fn(env);
  } finally {
    uninstall();
  }
}

/** Every element in the tree, root included. */
const all = (root) => [root, ...root.querySelectorAll("*")];
const text = (el) => el.textContent.trim();
const iconName = (svg) => svg.querySelector("use").getAttribute("href").replace("icons.svg#", "");

/** Record every write to #live's textContent, in order. */
function recordLive(live) {
  const proto = Object.getPrototypeOf(live);
  let desc;
  for (let p = proto; p && !desc; p = Object.getPrototypeOf(p)) desc = Object.getOwnPropertyDescriptor(p, "textContent");
  const writes = [];
  Object.defineProperty(live, "textContent", {
    configurable: true,
    get() { return desc.get.call(this); },
    set(v) { writes.push(String(v)); desc.set.call(this, v); },
  });
  return writes;
}

// ---------------------------------------------------------------- pageHeader

test("pageHeader: header.page-header > h1#page-title, a back link, description and actions", () => withDom(() => {
  const save = h("button.primary", { type: "submit" }, "Save");
  const el = pageHeader({ title: "transfer_funds", description: "Kind: tool",
    back: { href: "#/tools", label: "All tools" }, actions: [save] });
  assert.equal(el.tagName, "HEADER");
  assert.ok(el.classList.contains("page-header"));
  const h1 = el.querySelectorAll("h1");
  assert.equal(h1.length, 1);
  assert.equal(h1[0].id, "page-title");
  assert.equal(text(h1[0]), "transfer_funds");
  const back = el.querySelector("a.back-link");
  assert.equal(back.getAttribute("href"), "#/tools");
  assert.equal(text(back), "All tools");
  const svg = back.querySelector("svg.icon");
  assert.equal(iconName(svg), "chevron-left");
  assert.equal(svg.getAttribute("aria-hidden"), "true");
  assert.equal(text(el.querySelector("p.page-description")), "Kind: tool");
  assert.equal(el.querySelector(".page-actions > button"), save);
}));

test("pageHeader: no back link, description or actions when none are given", () => withDom(() => {
  const el = pageHeader({ title: "Tools" });
  assert.equal(el.querySelector("a"), null);
  assert.equal(el.querySelector("p"), null);
  assert.equal(el.querySelector(".page-actions"), null);
  assert.equal(text(el.querySelector("h1#page-title")), "Tools");
}));

test("pageHeader: without IntersectionObserver it builds and adds no sentinel", () => withDom(({ document }) => {
  assert.equal(globalThis.IntersectionObserver, undefined);
  const main = document.getElementById("main");
  main.append(pageHeader({ title: "Tools" }));
  assert.equal(main.querySelector(".page-header-sentinel"), null);
  assert.equal(main.firstElementChild.tagName, "HEADER");
}));

/** A fake IntersectionObserver: records instances; fire() delivers entries to the callback. */
class FakeIO {
  static made = [];
  constructor(cb) { Object.assign(this, { cb, targets: new Set(), disconnected: false }); FakeIO.made.push(this); }
  observe(t) { this.targets.add(t); }
  unobserve(t) { this.targets.delete(t); }
  disconnect() { this.targets.clear(); this.disconnected = true; }
  fire(target, { isIntersecting, top = 0 }) {
    this.cb([{ target, isIntersecting, boundingClientRect: { top }, rootBounds: { top: 0 } }], this);
  }
}

test("pageHeader: one IntersectionObserver places a sentinel before the header and toggles .is-stuck", () => withDom(({ document }) => {
  FakeIO.made = [];
  globalThis.IntersectionObserver = FakeIO;
  try {
    const header = pageHeader({ title: "Tools" });
    assert.equal(FakeIO.made.length, 1);
    const io = FakeIO.made[0];
    assert.ok(io.targets.has(header));
    io.fire(header, { isIntersecting: false }); // not connected yet: nothing to place
    assert.equal(header.parentNode, null);
    const main = document.getElementById("main");
    main.append(header);
    io.fire(header, { isIntersecting: true });
    const sentinel = main.firstElementChild;
    assert.ok(sentinel.classList.contains("page-header-sentinel"));
    assert.equal(sentinel.getAttribute("aria-hidden"), "true");
    assert.equal(sentinel.nextSibling, header);
    assert.ok(io.targets.has(sentinel) && !io.targets.has(header));
    io.fire(sentinel, { isIntersecting: false, top: -5 });
    assert.ok(header.classList.contains("is-stuck"));
    io.fire(sentinel, { isIntersecting: true, top: 0 });
    assert.ok(!header.classList.contains("is-stuck"));
    io.fire(sentinel, { isIntersecting: false, top: 900 }); // below the viewport: not stuck
    assert.ok(!header.classList.contains("is-stuck"));
    pageHeader({ title: "Next view" });
    assert.ok(io.disconnected, "a new page header disconnects the previous observer");
  } finally {
    delete globalThis.IntersectionObserver;
  }
}));

// ---------------------------------------------------------------- card and tableWrap

test("card: section.card with a header (h2, description, actions) and a body", () => withDom(() => {
  const btn = h("button", { type: "button" }, "Add");
  const el = card({ title: "Permissions", description: "Who may call it.", actions: [btn] }, h("p", {}, "body"));
  assert.equal(el.tagName, "SECTION");
  assert.ok(el.classList.contains("card"));
  assert.ok(!el.classList.contains("flush"));
  const h2 = el.querySelector(".card-header h2.card-title");
  assert.equal(text(h2), "Permissions");
  assert.equal(el.getAttribute("aria-labelledby"), h2.id);
  assert.ok(h2.id);
  assert.equal(text(el.querySelector(".card-header p.card-description")), "Who may call it.");
  assert.equal(el.querySelector(".card-actions > button"), btn);
  assert.equal(text(el.querySelector(".card-body > p")), "body");
}));

test("card: flush, an aria-label without a title, no header when nothing goes in it", () => withDom(() => {
  const el = card({ flush: true, label: "Tools" }, h("table"));
  assert.ok(el.classList.contains("flush"));
  assert.equal(el.getAttribute("aria-label"), "Tools");
  assert.equal(el.querySelector(".card-header"), null);
  assert.ok(el.querySelector(".card-body > table"));
  const bare = card();
  assert.ok(bare.classList.contains("card"));
  const titled = card({ title: h("code", {}, "ack.pool") });
  assert.equal(text(titled.querySelector("h2 > code")), "ack.pool");
}));

test("tableWrap: div.table-wrap, a focusable named region holding the table", () => withDom(() => {
  const table = h("table", { "aria-label": "Tools" });
  const el = tableWrap(table);
  assert.ok(el.classList.contains("table-wrap"));
  assert.equal(el.firstElementChild, table);
  assert.equal(el.getAttribute("tabindex"), "0");
  assert.equal(el.getAttribute("role"), "region");
  assert.equal(el.getAttribute("aria-label"), "Tools");
  const captioned = tableWrap(h("table", {}, h("caption", {}, "Audit entries")));
  assert.equal(captioned.getAttribute("aria-label"), "Audit entries");
  assert.equal(tableWrap(h("table")).getAttribute("aria-label"), "Table");
}));

// ---------------------------------------------------------------- badges

test("badge: span.badge.badge--<tone> with text, an optional decorative icon", () => withDom(() => {
  const b = badge("changed", "warning");
  assert.equal(b.tagName, "SPAN");
  assert.ok(b.classList.contains("badge") && b.classList.contains("badge--warning"));
  assert.equal(text(b), "changed");
  assert.ok(badge("x").classList.contains("badge--neutral"));
  const withIcon = badge("security", "danger", { icon: "shield-alert" });
  assert.equal(iconName(withIcon.querySelector("svg.icon")), "shield-alert");
  assert.equal(text(withIcon), "security");
  for (const tone of ["neutral", "brand", "success", "warning", "danger", "info"]) {
    assert.ok(badge("t", tone).classList.contains(`badge--${tone}`));
  }
}));

test("badge: refuses an empty label and an unknown tone (every badge carries text, 3.4)", () => withDom(() => {
  for (const blank of ["", "   ", null, undefined]) assert.throws(() => badge(blank), /text/);
  assert.throws(() => badge("x", "purple"), /tone/);
}));

test("envBadge: dev and test are env-test, prod is env-prod, with the stage as text", () => withDom(() => {
  for (const [stage, cls] of [["dev", "env-test"], ["test", "env-test"], ["prod", "env-prod"]]) {
    const b = envBadge(stage);
    assert.ok(b.classList.contains("badge") && b.classList.contains(cls), stage);
    assert.equal(text(b), stage);
  }
  for (const other of [undefined, null, "", "staging", "PROD", 3]) {
    const b = envBadge(other);
    assert.ok(b.classList.contains("badge--neutral"), String(other));
    assert.equal(text(b), "unknown stage");
  }
}));

test("PBT: any other stage string gives a neutral badge with text, never an env class", () => withDom(() => {
  const stage = oneOf(unicodeString({ maxLength: 10 }), string({ maxLength: 6 }), "dev ", " prod", "Test");
  forAll(stage, (s) => {
    const b = envBadge(s);
    if (["dev", "test", "prod"].includes(s)) return true;
    return b.classList.contains("badge") && b.classList.contains("badge--neutral")
      && !b.classList.contains("env-test") && !b.classList.contains("env-prod") && text(b) !== "";
  });
}));

test("langName: English and Spanish by primary subtag, else the code", () => {
  assert.equal(langName("en-US"), "English");
  assert.equal(langName("es-US"), "Spanish");
  assert.equal(langName("es-MX"), "Spanish");
  assert.equal(langName("EN"), "English");
  assert.equal(langName("fr-FR"), "fr-FR");
  assert.equal(langName(""), "");
});

test("langBadge: visible primary subtag (hidden from AT) plus the language name for screen readers", () => withDom(() => {
  const b = langBadge("es-US");
  assert.ok(b.classList.contains("badge") && b.classList.contains("badge--info"));
  const [visible, sr] = b.children;
  assert.equal(visible.textContent, "ES");
  assert.equal(visible.getAttribute("aria-hidden"), "true");
  assert.ok(sr.classList.contains("visually-hidden"));
  assert.equal(sr.textContent, "Spanish");
  const fr = langBadge("fr-FR");
  assert.equal(fr.children[0].textContent, "FR");
  assert.equal(fr.children[1].textContent, "fr-FR");
}));

// ---------------------------------------------------------------- panel, emptyState, skeleton

test("panel: div.panel.panel--<tone> with an icon, a title, a body, and the role for its tone", () => withDom(() => {
  const cases = [["error", "alert", "alert-circle"], ["warning", "note", "alert-triangle"],
    ["info", "note", "info"], ["success", "status", "check-circle"]];
  for (const [tone, role, ic] of cases) {
    const p = panel({ tone, title: "Problems" }, h("p", {}, "One"), "two");
    assert.ok(p.classList.contains("panel") && p.classList.contains(`panel--${tone}`));
    assert.equal(p.getAttribute("role"), role, tone);
    assert.equal(iconName(p.querySelector("svg.icon")), ic);
    assert.equal(p.querySelector("svg.icon").getAttribute("aria-hidden"), "true");
    assert.equal(text(p.querySelector(".panel-title")), "Problems");
    assert.equal(text(p.querySelector(".panel-body")), "ProblemsOnetwo");
  }
  assert.equal(panel({ tone: "info", role: "region" }).getAttribute("role"), "region");
  assert.equal(panel({ tone: "info" }).querySelector(".panel-title"), null);
  assert.throws(() => panel({ tone: "loud" }), /tone/);
}));

test("emptyState: icon in a circle, a title, a message and an optional action", () => withDom(() => {
  const action = h("button", { type: "button" }, "Add a phrase");
  const el = emptyState({ icon: "prompts", title: "No phrases yet", message: "Add one to start.", action });
  assert.ok(el.classList.contains("empty-state"));
  assert.equal(iconName(el.querySelector(".empty-state-icon > svg.icon")), "prompts");
  assert.equal(text(el.querySelector(".empty-state-title")), "No phrases yet");
  assert.equal(text(el.querySelector(".empty-state-message")), "Add one to start.");
  assert.equal(el.querySelector(".empty-state-action > button"), action);
  const plain = emptyState({ title: "Nothing here" });
  assert.equal(iconName(plain.querySelector("svg.icon")), "info");
  assert.equal(plain.querySelector(".empty-state-message"), null);
  assert.equal(plain.querySelector(".empty-state-action"), null);
}));

test("skeleton: aria-hidden bars of each kind plus a role=status 'Loading…' for screen readers", () => withDom(() => {
  const rows = { page: 6, table: 5, form: 0, inline: 0 };
  for (const kind of ["page", "table", "form", "inline"]) {
    const el = skeleton(kind);
    const bars = el.querySelector(`.skeleton.skeleton--${kind}`);
    assert.equal(bars.getAttribute("aria-hidden"), "true", kind);
    assert.ok(bars.querySelectorAll(".skeleton-bar").length >= 1, kind);
    assert.equal(bars.querySelectorAll(".skeleton-bar--row").length, rows[kind], kind);
    const status = el.querySelector('[role="status"]');
    assert.equal(status.textContent, "Loading…");
    assert.ok(status.classList.contains("visually-hidden"));
    assert.equal(bars.contains(status), false, "the status is not inside the hidden subtree");
  }
  assert.equal(skeleton("form").querySelectorAll(".skeleton-bar--input").length, 3);
  assert.equal(skeleton().querySelector(".skeleton--page") !== null, true);
  assert.throws(() => skeleton("wide"), /kind/);
}));

// ---------------------------------------------------------------- compareCards

test("compareCards: two cards, changed rows marked with is-changed and a 'changed' badge, the JSON kept", () => withDom(() => {
  const current = { rev: 7, allowed: true, mask: { masked: true, after_ms: { value: 1500, source: "policy" } } };
  const yours = { rev: 6, allowed: true, mask: { masked: true, after_ms: 2000 } };
  const el = compareCards(current, yours);
  const cards = el.querySelectorAll(".compare > section.card");
  assert.equal(cards.length, 2);
  const [cur, mine] = cards;
  assert.equal(text(cur.querySelector("h2 .badge")), "Current");
  assert.ok(cur.querySelector("h2 .badge").classList.contains("badge--neutral"));
  assert.equal(text(mine.querySelector("h2 .badge")), "Yours");
  assert.ok(mine.querySelector("h2 .badge").classList.contains("badge--brand"));
  for (const c of cards) {
    const rows = c.querySelectorAll("dl > div.compare-row");
    assert.deepEqual(rows.map((r) => text(r.querySelector("dt code"))), ["allowed", "mask.masked", "mask.after_ms"]);
    const changed = rows.filter((r) => r.classList.contains("is-changed"));
    assert.deepEqual(changed.map((r) => text(r.querySelector("dt code"))), ["mask.after_ms"]);
    const b = changed[0].querySelector(".badge");
    assert.equal(text(b), "changed");
    assert.ok(b.classList.contains("badge--warning"));
    assert.equal(rows[0].querySelector(".badge"), null);
  }
  assert.equal(text(cur.querySelector(".is-changed dd")), "1500");
  assert.equal(text(mine.querySelector(".is-changed dd")), "2000");
  const details = el.querySelector("details");
  assert.equal(text(details.querySelector("summary")), "Full current value (JSON)");
  assert.deepEqual(JSON.parse(details.querySelector("pre").textContent), current);
}));

test("compareCards: a path missing from current shows (none) and is changed", () => withDom(() => {
  const el = compareCards({ a: 1 }, { a: 1, b: 2 });
  const cur = el.querySelectorAll(".compare > section.card")[0];
  const row = cur.querySelectorAll(".compare-row")[1];
  assert.ok(row.classList.contains("is-changed"));
  assert.equal(text(row.querySelector("dd")), "(none)");
}));

test("compareCards: falls back to two JSON blocks when no path of yours is in current", () => withDom(() => {
  const current = { kind: "policy", value: { masking: {} } };
  const yours = { entry_id: "abc", target: "policy#masking" };
  const el = compareCards(current, yours);
  assert.equal(el.querySelector("dl"), null);
  assert.equal(el.querySelector(".is-changed"), null);
  const pres = el.querySelectorAll(".compare > section.card pre");
  assert.equal(pres.length, 2);
  assert.deepEqual(JSON.parse(pres[0].textContent), current);
  assert.deepEqual(JSON.parse(pres[1].textContent), yours);
  assert.equal(el.querySelector("details"), null);
  const odd = compareCards(undefined, "text");
  assert.equal(odd.querySelectorAll("pre").length, 2);
  assert.equal(odd.querySelectorAll("pre")[0].textContent, "(none)");
}));

// ---------------------------------------------------------------- announce

test("announce: a different text is set at once; the same text again is cleared, then re-set", (t) => withDom(({ document }) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const live = document.getElementById("live");
  const writes = recordLive(live);
  announce("Saved.");
  assert.equal(live.textContent, "Saved.");
  t.mock.timers.tick(ANNOUNCE_DELAY_MS);
  announce("Saved.");
  assert.equal(live.textContent, "", "emptied first");
  t.mock.timers.tick(ANNOUNCE_DELAY_MS);
  assert.equal(live.textContent, "Saved.", "then set on a later task");
  t.mock.timers.tick(ANNOUNCE_DELAY_MS);
  announce("A");
  announce("B");
  assert.equal(live.textContent, "A", "B waits while A is held");
  t.mock.timers.tick(ANNOUNCE_DELAY_MS);
  assert.equal(live.textContent, "B");
  assert.deepEqual(writes, ["Saved.", "", "Saved.", "A", "B"]);
}));

test("announce: does nothing without #live", () => {
  install();
  try { assert.doesNotThrow(() => announce("x")); } finally { uninstall(); }
});

// ---------------------------------------------------------------- toast examples

test("toast: a card in #toasts with a tone icon, title, message and a dismiss button; announced", (t) => withDom(({ document }) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const live = document.getElementById("live");
  const handle = toast({ tone: "success", title: "Saved.", message: "Takes effect within 60 seconds." });
  const el = document.querySelector("#toasts > .toast");
  assert.equal(handle.element, el);
  assert.ok(el.classList.contains("toast--success"));
  assert.equal(iconName(el.querySelector("svg.icon")), "check-circle");
  assert.equal(text(el.querySelector(".toast-title")), "Saved.");
  assert.equal(text(el.querySelector(".toast-message")), "Takes effect within 60 seconds.");
  assert.equal(el.getAttribute("role"), null, "#toasts is not a live region; #live announces");
  const dismiss = el.querySelector("button.toast-dismiss");
  assert.equal(dismiss.getAttribute("type"), "button");
  assert.equal(dismiss.getAttribute("aria-label"), "Dismiss notification");
  assert.equal(iconName(dismiss.querySelector("svg.icon")), "x");
  assert.equal(live.textContent, "Saved. Takes effect within 60 seconds.", "no colon after a title that ends in punctuation");
  dismiss.click();
  assert.equal(el.parentNode, null);
  assert.doesNotThrow(() => handle.dismiss(), "dismissing twice is harmless");
  toast({ tone: "info", title: "Only a title" });
  t.mock.timers.tick(ANNOUNCE_DELAY_MS * 2);
  assert.equal(live.textContent, "Only a title");
  assert.equal(document.querySelector(".toast .toast-message"), null);
  toast({ tone: "info", title: "Heads up", message: "The pool changed." });
  t.mock.timers.tick(ANNOUNCE_DELAY_MS * 2);
  assert.equal(live.textContent, "Heads up: The pool changed.", "a colon after a title without end punctuation");
  for (const [tone, ic] of [["error", "alert-circle"], ["warning", "alert-triangle"], ["info", "info"]]) {
    assert.equal(iconName(toast({ tone, title: tone }).element.querySelector("svg.icon")), ic);
  }
  assert.throws(() => toast({ tone: "loud", title: "x" }), /tone/);
}));

test("toast: success goes after TOAST_TIMEOUT_MS, error stays, hover pauses the timer", (t) => withDom(({ document }) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  assert.equal(TOAST_TIMEOUT_MS, 5000);
  assert.equal(MAX_TOASTS, 3);
  const ok = toast({ tone: "success", title: "Saved." }).element;
  const bad = toast({ tone: "error", title: "Failed" }).element;
  t.mock.timers.tick(2000);
  ok.dispatchEvent(new Event("mouseenter"));
  t.mock.timers.tick(10000);
  assert.ok(ok.isConnected, "paused while hovered");
  ok.dispatchEvent(new Event("mouseleave"));
  t.mock.timers.tick(2999);
  assert.ok(ok.isConnected);
  t.mock.timers.tick(1);
  assert.ok(!ok.isConnected, "gone at 5 s of unpaused time");
  t.mock.timers.tick(60000);
  assert.ok(bad.isConnected, "errors stay until dismissed");
  const custom = toast({ tone: "info", title: "Quick", timeoutMs: 100 }).element;
  t.mock.timers.tick(100);
  assert.ok(!custom.isConnected);
  assert.equal(document.querySelectorAll(".toast").length, 1);
}));

test("toast: focus inside pauses it; focus moving within the toast keeps it paused", (t) => withDom(() => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const el = toast({ tone: "info", title: "Hi" }).element;
  const btn = el.querySelector("button");
  btn.dispatchEvent(new Event("focusin", { bubbles: true }));
  t.mock.timers.tick(9000);
  const out = new Event("focusout", { bubbles: true });
  out.relatedTarget = el.querySelector(".toast-body");
  btn.dispatchEvent(out);
  t.mock.timers.tick(9000);
  assert.ok(el.isConnected);
  btn.dispatchEvent(new Event("focusout", { bubbles: true }));
  t.mock.timers.tick(5000);
  assert.ok(!el.isConnected);
}));

test("toast: at most MAX_TOASTS; the oldest success or info goes first, then warnings, then errors", (t) => withDom(({ document }) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const titles = () => document.querySelectorAll("#toasts .toast-title").map(text);
  toast({ tone: "error", title: "e1" });
  toast({ tone: "success", title: "s1" });
  toast({ tone: "warning", title: "w1" });
  toast({ tone: "info", title: "i1" });
  assert.deepEqual(titles(), ["e1", "w1", "i1"]);
  toast({ tone: "error", title: "e2" });
  assert.deepEqual(titles(), ["e1", "w1", "e2"]);
  toast({ tone: "error", title: "e3" });
  assert.deepEqual(titles(), ["e1", "e2", "e3"]);
  toast({ tone: "error", title: "e4" });
  assert.deepEqual(titles(), ["e2", "e3", "e4"]);
}));

test("toast: works without #toasts (still announced, not attached)", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
  const { document } = install();
  try {
    document.body.append(h("div#live"));
    const handle = toast({ tone: "error", title: "Oops" });
    assert.equal(handle.element.parentNode, null);
    assert.equal(document.getElementById("live").textContent, "Oops");
    handle.dismiss();
  } finally {
    uninstall();
  }
});

// ---------------------------------------------------------------- PBT

const TONES = ["success", "info", "warning", "error"];
const step = oneOf(
  record({ op: "toast", tone: oneOf(...TONES), title: oneOf("Saved.", "Done!", "Failed", "Heads up", string({ minLength: 1, maxLength: 4 })), message: oneOf("", "x", "Takes effect.") }),
  record({ op: "advance", ms: oneOf(int(0, 300), int(0, 3000), int(4000, 6000)) }),
  record({ op: oneOf("hover", "unhover", "focus", "blur", "dismiss"), pick: int(0, 5) }),
);

test("PBT: toasts time out only when they should, at most MAX_TOASTS, and every one is announced", () => {
  forAll(array(step, { minLength: 1, maxLength: 25 }), (steps) => {
    mock.timers.enable({ apis: ["setTimeout", "Date"], now: 0 });
    const env = install();
    try {
      const { document } = env;
      const container = h("div#toasts");
      const live = h("div#live");
      document.body.append(container, live);
      const writes = recordLive(live);
      const model = []; // {el, tone, created, paused, hovered, focused, gone}
      const announced = [];
      let now = 0;
      const visible = () => model.filter((m) => !m.gone);
      const pick = (i) => visible()[i % Math.max(1, visible().length)];
      for (const s of steps) {
        if (s.op === "toast") {
          const { element } = toast({ tone: s.tone, title: s.title, message: s.message });
          const sep = /[.!?:;…]$/u.test(s.title) ? " " : ": ";
          announced.push(s.message ? `${s.title}${sep}${s.message}` : s.title);
          model.push({ el: element, tone: s.tone, created: now, paused: 0, hovered: false, focused: false, gone: false });
          const shown = visible();
          if (shown.length > MAX_TOASTS) {
            const victim = shown.find((m) => m.tone === "success" || m.tone === "info")
              || shown.find((m) => m.tone === "warning") || shown[0];
            victim.gone = "evicted";
          }
        } else if (s.op === "advance") {
          for (const m of visible()) if (m.hovered || m.focused) m.paused += s.ms;
          mock.timers.tick(s.ms);
          now += s.ms;
          for (const m of visible()) {
            const active = now - m.created - m.paused;
            const timed = m.tone === "success" || m.tone === "info";
            if (!m.el.isConnected) {
              assert.ok(timed, `a ${m.tone} toast was removed by time`);
              assert.ok(active >= TOAST_TIMEOUT_MS, `removed early: ${active} ms of unpaused time`);
              m.gone = "time";
            } else if (timed) {
              assert.ok(active < TOAST_TIMEOUT_MS, `still shown after ${active} ms of unpaused time`);
            }
          }
        } else {
          const m = pick(s.pick);
          if (!m) continue;
          const target = m.el.querySelector("button");
          if (s.op === "hover") { m.el.dispatchEvent(new Event("mouseenter")); m.hovered = true; }
          if (s.op === "unhover") { m.el.dispatchEvent(new Event("mouseleave")); m.hovered = false; }
          if (s.op === "focus") { target.dispatchEvent(new Event("focusin", { bubbles: true })); m.focused = true; }
          if (s.op === "blur") { target.dispatchEvent(new Event("focusout", { bubbles: true })); m.focused = false; }
          if (s.op === "dismiss") { target.click(); m.gone = "dismissed"; }
        }
        const shownEls = container.querySelectorAll(".toast");
        assert.ok(shownEls.length <= MAX_TOASTS, `${shownEls.length} toasts visible`);
        assert.deepEqual(shownEls, visible().map((m) => m.el), "the visible toasts are the model's");
      }
      // mock.timers runs one level of chained timers per tick, so drain the queue step by step.
      for (let i = 0; i < 2 * announced.length + 2; i++) mock.timers.tick(ANNOUNCE_DELAY_MS);
      const said = writes.filter((w) => w !== "");
      assert.deepEqual(said, announced, "every toast's text reaches #live, in order");
      writes.forEach((w, i) => {
        const before = i === 0 ? "" : writes[i - 1]; // #live starts empty
        if (w !== "") assert.notEqual(before, w, "each announcement is a change (same text: emptied first)");
      });
      return true;
    } finally {
      uninstall();
      mock.timers.reset();
    }
  }, { runs: 200 });
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
    back: oneOf(undefined, { href: "#/tools", label: "All tools" })(rng), actions: [h("button", {}, label(rng))] })),
  gen((rng) => () => card({ title: oneOf(undefined, someText)(rng), description: oneOf(undefined, label)(rng),
    flush: oneOf(true, false)(rng), label: oneOf(undefined, someText)(rng) }, label(rng))),
  gen((rng) => () => tableWrap(h("table", { "aria-label": oneOf(undefined, someText)(rng) }))),
  gen((rng) => () => badge(someText(rng), oneOf("neutral", "brand", "success", "warning", "danger", "info")(rng),
    { icon: oneOf(undefined, ...ICONS)(rng) })),
  gen((rng) => () => envBadge(oneOf("dev", "test", "prod", label)(rng))),
  gen((rng) => () => langBadge(oneOf("en-US", "es-US", "fr", string({ minLength: 1, maxLength: 5 }))(rng))),
  gen((rng) => () => panel({ tone: oneOf("error", "warning", "info", "success")(rng), title: oneOf(undefined, label)(rng) }, label(rng))),
  gen((rng) => () => emptyState({ icon: oneOf(undefined, ...ICONS)(rng), title: label(rng), message: oneOf(undefined, label)(rng) })),
  gen((rng) => () => skeleton(oneOf("page", "table", "form", "inline")(rng))),
  gen((rng) => () => compareCards(obj(rng), obj(rng))),
  gen((rng) => () => toast({ tone: oneOf(...TONES)(rng), title: label(rng), message: label(rng) }).element),
);

test("PBT: no built node has a style or on* attribute, and every badge has non-empty text", () => {
  forAll(builders, (build) => withDom(() => {
    const root = build();
    for (const el of all(root)) {
      for (const { name } of el.attributes) {
        assert.ok(name !== "style" && !/^on/i.test(name), `${el.tagName} has ${name}`);
      }
    }
    for (const b of all(root).filter((el) => el.classList.contains("badge"))) {
      assert.ok(text(b) !== "", "a badge without text");
    }
    return true;
  }));
});

// ---------------------------------------------------------------- static

test("lib/components.js builds no SVG from markup and sets no inline style", () => {
  const src = readFileSync(join(ROOT, "lib", "components.js"), "utf8");
  for (const banned of ["<svg", "<path", "innerHTML", "outerHTML", ".style", "setAttribute(\"style\""]) {
    assert.equal(src.includes(banned), false, banned);
  }
});
