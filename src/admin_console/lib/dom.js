// Small DOM helpers. Text is always added as text nodes, so nothing from the API is ever parsed
// as HTML. No inline styles or handler attributes (the CSP allows neither): h() refuses them.

/** h("tag.class#id", {attr: value, on: {click: fn}}, ...children) */
export function h(spec, attrs = {}, ...children) {
  const [, tag = "div", rest = ""] = /^([a-z0-9]+)?(.*)$/i.exec(spec);
  const el = document.createElement(tag);
  for (const m of rest.matchAll(/([.#])([A-Za-z0-9_-]+)/g)) {
    if (m[1] === ".") el.classList.add(m[2]);
    else el.id = m[2];
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "style" || /^on./i.test(k)) throw new Error(`inline ${k} is not allowed (CSP)`);
    if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "class") el.classList.add(...String(v).split(/\s+/).filter(Boolean));
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = Boolean(v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

const SVG_NS = "http://www.w3.org/2000/svg";

/**
 * An icon from the sprite: an svg.icon element holding one use element that points at
 * icons.svg#name, built with createElementNS (never from markup). Decorative (aria-hidden)
 * unless `label` is given, in which case it is role="img" with that aria-label.
 */
export function icon(name, { label } = {}) {
  if (typeof name !== "string" || !/^[a-z0-9]+(-[a-z0-9]+)*$/.test(name)) {
    throw new Error(`not an icon name: ${String(name)}`);
  }
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("focusable", "false");
  if (label) {
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", String(label));
  } else {
    svg.setAttribute("aria-hidden", "true");
  }
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `icons.svg#${name}`);
  svg.append(use);
  return svg;
}

export function clear(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

let uid = 0;
const newId = () => `f${++uid}`;

/**
 * A labelled form control: returns {wrap, input}. The label sits above; a required
 * control (control.required) gets an aria-hidden "*" in its label and, unless `requiredNote` is
 * false, a "* Required" note (the control's own `required` is what assistive technology hears).
 * The hint goes below, linked by aria-describedby. A select sits in .select-wrap next to a
 * chevron-down icon, which replaces the native arrow (CSS can't use url()).
 */
export function field(label, control, hint, { requiredNote = true } = {}) {
  const id = control.id || newId();
  control.id = id;
  const required = Boolean(control.required);
  const hintEl = hint ? h("p.hint", { id: `${id}-hint` }, hint) : null;
  if (hintEl) control.setAttribute("aria-describedby", hintEl.id);
  const isSelect = control.localName === "select";
  const body = isSelect ? h("span.select-wrap", {}, control, selectChevron()) : control;
  return {
    wrap: h("div.field", {},
      h("label", { for: id }, label, required ? h("span.field-required", { "aria-hidden": "true" }, "*") : null),
      body,
      hintEl,
      required && requiredNote ? h("p.form-note", { "aria-hidden": "true" }, "* Required") : null),
    input: control,
  };
}

function selectChevron() {
  const svg = icon("chevron-down");
  svg.classList.add("select-chevron");
  return svg;
}

const invalid = new WeakMap(); // control -> {describedBy (the original, or null), message, onInput}

/**
 * Mark a control invalid (Field): aria-invalid="true" and one message below its hint
 * (an alert-circle icon and the text, with an id joined into aria-describedby, the hint's id
 * kept). Marking it again replaces the message. The next "input" (or "change") clears it.
 */
export function markInvalid(control, message) {
  const state = invalid.get(control);
  if (state) {
    clear(state.message, icon("alert-circle"), h("span", {}, message));
    return state.message;
  }
  if (!control.id) control.id = newId();
  const msg = h("p.field-error", { id: `${control.id}-error` }, icon("alert-circle"), h("span", {}, message));
  const describedBy = control.getAttribute("aria-describedby");
  const parent = control.parentNode;
  const anchor = (parent && parent.classList && parent.classList.contains("select-wrap")) ? parent : control;
  const wrap = control.closest ? control.closest(".field") : null;
  const hint = wrap && wrap.querySelector(`[id="${CSS.escape(`${control.id}-hint`)}"]`);
  const after = hint || anchor;
  if (after.parentNode) after.parentNode.insertBefore(msg, after.nextSibling);
  control.setAttribute("aria-invalid", "true");
  control.setAttribute("aria-describedby", [describedBy, msg.id].filter(Boolean).join(" "));
  const onInput = () => clearInvalid(control);
  control.addEventListener("input", onInput);
  control.addEventListener("change", onInput);
  invalid.set(control, { describedBy, message: msg, onInput });
  return msg;
}

/** Undo markInvalid: aria-invalid and the message go, aria-describedby is as it was. */
export function clearInvalid(control) {
  const state = invalid.get(control);
  if (!state) return;
  invalid.delete(control);
  control.removeEventListener("input", state.onInput);
  control.removeEventListener("change", state.onInput);
  control.removeAttribute("aria-invalid");
  if (state.describedBy === null) control.removeAttribute("aria-describedby");
  else control.setAttribute("aria-describedby", state.describedBy);
  state.message.remove();
}

/** Pretty JSON for showing values (audit, conflicts). */
export function show(value) {
  if (value === undefined || value === null) return "(none)";
  return typeof value === "string" ? value : JSON.stringify(value, null, 1);
}

/** How long each announcement holds #live, and how long it stays empty before a repeat. */
export const ANNOUNCE_DELAY_MS = 50;

const liveQueues = new WeakMap(); // the #live element -> {items, busy}; a new document starts fresh

/**
 * Show a status message in the page's live region (screen readers announce it). Every call is
 * a change to the region, so the same text twice is announced twice: a text equal to what the
 * region shows is emptied first and set on a later task. Calls made while one announcement is
 * held queue up, each held ANNOUNCE_DELAY_MS, so a quick second message doesn't hide the first.
 * A different text with nothing queued is set at once.
 */
export function announce(text) {
  const live = document.getElementById("live");
  if (!live) return;
  let q = liveQueues.get(live);
  if (!q) liveQueues.set(live, (q = { items: [], busy: false }));
  q.items.push(String(text ?? ""));
  if (!q.busy) pumpLive(live, q);
}

function pumpLive(live, q) {
  q.busy = true;
  const next = q.items.shift();
  const hold = () => setTimeout(() => {
    if (q.items.length) pumpLive(live, q);
    else q.busy = false;
  }, ANNOUNCE_DELAY_MS);
  if (live.textContent !== next) {
    live.textContent = next;
    hold();
    return;
  }
  live.textContent = "";
  setTimeout(() => {
    live.textContent = next;
    hold();
  }, ANNOUNCE_DELAY_MS);
}
