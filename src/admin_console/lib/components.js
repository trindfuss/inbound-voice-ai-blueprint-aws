// Presentational builders: page header, card, table wrapper, badges, panel, empty
// state, skeleton, toast and the conflict comparison. They only build nodes with h() and icon(),
// so text stays text and no inline style or handler attribute is ever set. Icons are
// decorative: every one sits next to a text label. Every badge carries text.

import { announce, h, icon, show } from "./dom.js";
import { compareRows } from "./compare.js";

// ---------------------------------------------------------------- page header

let stuckObserver = null; // one IntersectionObserver, for the current page header

/**
 * header.page-header: an optional back link (chevron-left and its label), h1#page-title, an
 * optional description and the actions. Where IntersectionObserver exists, a zero-height
 * sentinel goes just before the header once it is in the page, and the header gets .is-stuck
 * while the sentinel has scrolled above the viewport. Building the next header disconnects the
 * previous observer.
 */
export function pageHeader({ title, description, back, actions = [] } = {}) {
  const header = h("header.page-header", {},
    back ? h("a.back-link", { href: back.href }, icon("chevron-left"), back.label) : null,
    h("div.page-header-text", {},
      h("h1#page-title", {}, title),
      description ? h("p.page-description", {}, description) : null),
    actions.length ? h("div.page-actions", {}, actions) : null);
  watchStuck(header);
  return header;
}

function watchStuck(header) {
  if (stuckObserver) stuckObserver.disconnect();
  stuckObserver = null;
  const IO = globalThis.IntersectionObserver;
  if (typeof IO !== "function") return;
  const sentinel = h("div.page-header-sentinel", { "aria-hidden": "true" });
  const io = new IO((entries, observer) => {
    for (const e of entries) {
      if (e.target === header) {
        if (!header.parentNode) continue; // not in the page yet; wait for the next notice
        header.parentNode.insertBefore(sentinel, header);
        observer.unobserve(header);
        observer.observe(sentinel);
      } else if (e.target === sentinel) {
        const top = e.rootBounds ? e.rootBounds.top : 0;
        header.classList.toggle("is-stuck", !e.isIntersecting && e.boundingClientRect.top < top);
      }
    }
  });
  io.observe(header);
  stuckObserver = io;
}

// ---------------------------------------------------------------- card and table

let cardId = 0;

/**
 * section.card: a header row (h2 title, muted description, actions) when any of them is given,
 * then div.card-body with the children. `flush` drops the body padding (tables). Named by its
 * title, else by `label`.
 */
export function card({ title, description, actions = [], flush = false, label } = {}, ...children) {
  const titleId = title !== undefined && title !== null && title !== "" ? `card-title-${++cardId}` : null;
  const hasHeader = titleId || description || actions.length;
  const el = h("section.card", {
    class: flush ? "flush" : null,
    "aria-labelledby": titleId,
    "aria-label": titleId ? null : label,
  },
  hasHeader ? h("div.card-header", {},
    titleId || description ? h("div.card-heading", {},
      titleId ? h("h2.card-title", { id: titleId }, title) : null,
      description ? h("p.card-description", {}, description) : null) : null,
    actions.length ? h("div.card-actions", {}, actions) : null) : null,
  h("div.card-body", {}, children));
  return el;
}

/**
 * div.table-wrap around a table: it scrolls both ways (so the sticky th works) and is a
 * focusable, named region, so keyboard users can scroll it.
 */
export function tableWrap(table) {
  const caption = table.querySelector("caption");
  const name = table.getAttribute("aria-label") || (caption && caption.textContent.trim()) || "Table";
  return h("div.table-wrap", { role: "region", tabindex: "0", "aria-label": name }, table);
}

// ---------------------------------------------------------------- badges

const BADGE_TONES = ["neutral", "brand", "success", "warning", "danger", "info"];

function labelText(text) {
  const s = text === undefined || text === null ? "" : String(text);
  if (s.trim() === "") throw new Error("a badge needs text");
  return s;
}

/** span.badge.badge--<tone> with its text and an optional decorative leading icon. */
export function badge(text, tone = "neutral", { icon: name } = {}) {
  if (!BADGE_TONES.includes(tone)) throw new Error(`not a badge tone: ${String(tone)}`);
  return h(`span.badge.badge--${tone}`, {}, name ? icon(name) : null, labelText(text));
}

/** The environment badge: dev and test are env-test, prod is env-prod; else "unknown stage". */
export function envBadge(stage) {
  if (stage === "dev" || stage === "test") return h("span.badge.env-test", {}, stage);
  if (stage === "prod") return h("span.badge.env-prod", {}, stage);
  return badge("unknown stage", "neutral");
}

const LANGUAGES = { en: "English", es: "Spanish" };

const primarySubtag = (code) => String(code ?? "").trim().split(/[-_]/)[0];

/** "en-US" -> "English", "es-US" -> "Spanish" (by primary subtag); any other code is returned as is. */
export function langName(code) {
  return LANGUAGES[primarySubtag(code).toLowerCase()] ?? String(code ?? "");
}

/** An info badge showing the primary subtag ("ES"), with the language name for screen readers. */
export function langBadge(code) {
  const short = labelText(primarySubtag(code).toUpperCase());
  return h("span.badge.badge--info.lang-badge", {},
    h("span", { "aria-hidden": "true" }, short),
    h("span.visually-hidden", {}, langName(code)));
}

// ---------------------------------------------------------------- panel, empty state, skeleton

const TONE_ROLE = { error: "alert", warning: "note", info: "note", success: "status" };

function toneIcon(tone) {
  if (tone === "error") return icon("alert-circle");
  if (tone === "warning") return icon("alert-triangle");
  if (tone === "success") return icon("check-circle");
  return icon("info");
}

/**
 * div.panel.panel--<tone>: an icon, then a body with the optional title and the children.
 * Errors are role="alert", warning and info role="note", success role="status"; `role` overrides.
 */
export function panel({ tone, title, role } = {}, ...children) {
  if (!Object.hasOwn(TONE_ROLE, tone)) throw new Error(`not a panel tone: ${String(tone)}`);
  return h(`div.panel.panel--${tone}`, { role: role || TONE_ROLE[tone] },
    toneIcon(tone),
    h("div.panel-body", {},
      title !== undefined && title !== null && title !== "" ? h("p.panel-title", {}, title) : null,
      children));
}

/** A centred empty state: an icon in a circle, a title, an optional message and action. */
export function emptyState({ icon: name = "info", title, message, action } = {}) {
  return h("div.empty-state", {},
    h("span.empty-state-icon", {}, icon(name)),
    h("p.empty-state-title", {}, title),
    message ? h("p.empty-state-message", {}, message) : null,
    action ? h("div.empty-state-action", {}, action) : null);
}

const bar = (kind) => h(`div.skeleton-bar.skeleton-bar--${kind}`);
const times = (n, fn) => Array.from({ length: n }, fn);

const SKELETONS = {
  page: () => [bar("title"), bar("line"), h("div.skeleton-card", {}, times(6, () => bar("row")))],
  table: () => times(5, () => bar("row")),
  form: () => times(3, () => [bar("label"), bar("input")]),
  inline: () => [bar("line")],
};

/**
 * Loading placeholder: aria-hidden bars shaped like what's coming (page, table, form or
 * inline), next to a visually hidden role="status" "Loading…".
 */
export function skeleton(kind = "page") {
  if (!Object.hasOwn(SKELETONS, kind)) throw new Error(`not a skeleton kind: ${String(kind)}`);
  return h("div.loading", {},
    h(`div.skeleton.skeleton--${kind}`, { "aria-hidden": "true" }, SKELETONS[kind]()),
    h("p.visually-hidden", { role: "status" }, "Loading…"));
}

// ---------------------------------------------------------------- toasts

export const TOAST_TIMEOUT_MS = 5000;
export const MAX_TOASTS = 3;

const stacks = new WeakMap(); // #toasts element -> the open toasts, oldest first

/** Which open toast to drop for a new one: the oldest success or info, then warning, then error. */
function evictionVictim(open) {
  return open.find((t) => t.tone === "success" || t.tone === "info")
    || open.find((t) => t.tone === "warning")
    || open[0];
}

/**
 * Show a toast in #toasts and announce "title: message" through #live, or "title message"
 * when the title already ends in punctuation ("Saved. It takes effect…"). Success and info
 * close after `timeoutMs` (default TOAST_TIMEOUT_MS) of time not hovered or focused; error and
 * warning stay until dismissed. At most MAX_TOASTS are shown. Returns {element, dismiss()}.
 */
const ENDS_IN_PUNCTUATION = /[.!?:;…]$/u;

export function toast({ tone = "info", title, message = "", timeoutMs } = {}) {
  if (!Object.hasOwn(TONE_ROLE, tone)) throw new Error(`not a toast tone: ${String(tone)}`);
  const dismissBtn = h("button.quiet.toast-dismiss", { type: "button", "aria-label": "Dismiss notification" }, icon("x"));
  const el = h(`div.toast.toast--${tone}`, {},
    toneIcon(tone),
    h("div.toast-body", {},
      h("p.toast-title", {}, title),
      message ? h("p.toast-message", {}, message) : null),
    dismissBtn);

  const container = document.getElementById("toasts");
  let open = [];
  if (container) {
    open = stacks.get(container) || [];
    stacks.set(container, open);
  }
  const timed = tone === "success" || tone === "info";
  let remaining = timeoutMs ?? TOAST_TIMEOUT_MS;
  let startedAt = 0;
  let timer = null;
  let hovered = false;
  let focused = false;
  let gone = false;

  const entry = { tone, dismiss };
  function dismiss() {
    if (gone) return;
    gone = true;
    if (timer !== null) clearTimeout(timer);
    timer = null;
    const i = open.indexOf(entry);
    if (i >= 0) open.splice(i, 1);
    el.remove();
  }
  const run = () => {
    if (!timed || gone || timer !== null || hovered || focused) return;
    startedAt = Date.now();
    timer = setTimeout(dismiss, Math.max(0, remaining));
  };
  const pause = () => {
    if (timer === null) return;
    clearTimeout(timer);
    timer = null;
    remaining -= Date.now() - startedAt;
  };
  const sync = () => { if (hovered || focused) pause(); else run(); };

  el.addEventListener("mouseenter", () => { hovered = true; sync(); });
  el.addEventListener("mouseleave", () => { hovered = false; sync(); });
  el.addEventListener("focusin", () => { focused = true; sync(); });
  el.addEventListener("focusout", (ev) => {
    if (ev.relatedTarget && el.contains(ev.relatedTarget)) return; // focus moved within the toast
    focused = false;
    sync();
  });
  dismissBtn.addEventListener("click", dismiss);

  if (container) {
    open.push(entry);
    container.append(el);
    while (open.length > MAX_TOASTS) evictionVictim(open).dismiss();
  }
  run();
  const said = String(title ?? "");
  announce(message ? `${said}${ENDS_IN_PUNCTUATION.test(said) ? "" : ":"} ${message}` : said);
  return { element: el, dismiss };
}

// ---------------------------------------------------------------- conflict comparison

function compareCard(heading, tone, body) {
  return card({ title: badge(heading, tone) }, body);
}

function rowList(rows, side) {
  return h("dl.compare-list", {}, rows.map((r) => h("div.compare-row", { class: r.changed ? "is-changed" : null },
    h("dt", {}, h("code", {}, r.path), r.changed ? badge("changed", "warning") : null),
    h("dd", {}, h("code.compare-value", {}, show(r[side]))))));
}

/**
 * The conflict comparison: "Current" and "Yours" cards listing each leaf path of `yours`
 * (rev skipped) with changed rows marked by is-changed and a "changed" badge, then the full
 * current value under a <details>. When no path of `yours` is in `current` (a restore body, for
 * example) it shows the two values as JSON blocks instead.
 */
export function compareCards(current, yours) {
  const rows = compareRows(current, yours);
  if (!rows.some((r) => r.found)) {
    return h("div.compare-group", {},
      h("div.compare", {},
        compareCard("Current", "neutral", h("pre.compare-json", {}, show(current))),
        compareCard("Yours", "brand", h("pre.compare-json", {}, show(yours)))));
  }
  return h("div.compare-group", {},
    h("div.compare", {},
      compareCard("Current", "neutral", rowList(rows, "current")),
      compareCard("Yours", "brand", rowList(rows, "yours"))),
    h("details.compare-details", {},
      h("summary", {}, "Full current value (JSON)"),
      h("pre.compare-json", {}, show(current))));
}
