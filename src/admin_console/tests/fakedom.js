// A small fake DOM for `node --test`: enough of the browser for the console's modules to
// build and query their trees, with no dependency (so no jsdom). It covers elements and
// text nodes, attributes, `id`, `classList`, `dataset`, `value` and `checked`, `append` and
// `replaceChildren`, `querySelector(All)` (tag, `*`, `.class`, `#id`, `[attr]`, `[attr="v"]`,
// descendant and `>` combinators, comma lists), events with bubbling, `focus` and
// `document.activeElement`, `<dialog>`, `createElementNS`, `CSS.escape`, and `localStorage` and
// `sessionStorage` with a "throws" switch. Anything it doesn't support throws rather than
// pretending. `install()` sets the globals and `uninstall()` restores them exactly.

const HTML_NS = "http://www.w3.org/1999/xhtml";
const GLOBALS = ["document", "window", "Node", "Element", "Text", "Event", "CustomEvent", "CSS",
  "localStorage", "sessionStorage"];

const securityError = () => new DOMException("The operation is insecure.", "SecurityError");

// ---------------------------------------------------------------- events

export class Event {
  constructor(type, { bubbles = false, cancelable = false } = {}) {
    Object.assign(this, { type: String(type), bubbles, cancelable, defaultPrevented: false });
    this.cancelBubble = false;
    this.target = null;
    this.currentTarget = null;
  }
  preventDefault() { if (this.cancelable) this.defaultPrevented = true; }
  stopPropagation() { this.cancelBubble = true; }
  stopImmediatePropagation() { this.cancelBubble = true; this._stopNow = true; }
}

export class CustomEvent extends Event {
  constructor(type, init = {}) {
    super(type, init);
    this.detail = init.detail ?? null;
  }
}

class Target {
  #listeners = new Map();
  addEventListener(type, fn, opts) {
    if (typeof fn !== "function") return;
    const list = this.#listeners.get(type) || [];
    if (!list.some((l) => l.fn === fn)) list.push({ fn, once: Boolean(opts && opts.once) });
    this.#listeners.set(type, list);
  }
  removeEventListener(type, fn) {
    const list = this.#listeners.get(type) || [];
    this.#listeners.set(type, list.filter((l) => l.fn !== fn));
  }
  _fire(ev) {
    for (const l of [...(this.#listeners.get(ev.type) || [])]) {
      if (l.once) this.removeEventListener(ev.type, l.fn);
      l.fn.call(this, ev);
      if (ev._stopNow) break;
    }
  }
  _eventParent() { return null; }
  dispatchEvent(ev) {
    const set = (k, v) => Object.defineProperty(ev, k, { value: v, configurable: true, writable: true });
    set("target", this);
    const path = [this];
    if (ev.bubbles) for (let n = this._eventParent(); n; n = n._eventParent()) path.push(n);
    for (const node of path) {
      set("currentTarget", node);
      node._fire(ev);
      if (ev.cancelBubble) break;
    }
    set("currentTarget", null);
    return !ev.defaultPrevented;
  }
}

// ---------------------------------------------------------------- nodes

export class Node extends Target {
  static ELEMENT_NODE = 1;
  static TEXT_NODE = 3;
  static DOCUMENT_NODE = 9;
  constructor(doc) {
    super();
    this.ownerDocument = doc;
    this.parentNode = null;
    this.childNodes = [];
  }
  _eventParent() { return this.parentNode || (this.nodeType === 9 ? this._window : null); }
  get parentElement() { return this.parentNode instanceof Element ? this.parentNode : null; }
  get firstChild() { return this.childNodes[0] ?? null; }
  get lastChild() { return this.childNodes.at(-1) ?? null; }
  get nextSibling() { return this.#sibling(1); }
  get previousSibling() { return this.#sibling(-1); }
  #sibling(d) {
    const kids = this.parentNode ? this.parentNode.childNodes : [];
    return kids[kids.indexOf(this) + d] ?? null;
  }
  contains(other) {
    for (let n = other; n; n = n.parentNode) if (n === this) return true;
    return false;
  }
  get isConnected() { return Boolean(this.ownerDocument && this.ownerDocument.contains(this)); }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  removeChild(child) {
    const i = this.childNodes.indexOf(child);
    if (i < 0) throw new DOMException("not a child", "NotFoundError");
    this.childNodes.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  insertBefore(child, ref) {
    const node = this.#adopt(child);
    const i = ref ? this.childNodes.indexOf(ref) : -1;
    if (ref && i < 0) throw new DOMException("ref is not a child", "NotFoundError");
    this.childNodes.splice(i < 0 ? this.childNodes.length : i, 0, node);
    node.parentNode = this;
    return node;
  }
  appendChild(child) { return this.insertBefore(child, null); }
  append(...nodes) { for (const n of nodes) this.appendChild(n); }
  prepend(...nodes) { const first = this.firstChild; for (const n of nodes) this.insertBefore(n, first); }
  replaceChildren(...nodes) {
    for (const c of [...this.childNodes]) this.removeChild(c);
    this.append(...nodes);
  }
  #adopt(child) {
    const node = child instanceof Node ? child : this.ownerDocument.createTextNode(String(child));
    if (node.contains(this)) throw new DOMException("cycle", "HierarchyRequestError");
    node.remove();
    return node;
  }
  /** Every descendant element, in document order. */
  _descendants() {
    return this.childNodes.flatMap((c) => (c instanceof Element ? [c, ...c._descendants()] : []));
  }
  querySelectorAll(selector) {
    const sel = parseSelector(selector);
    return this._descendants().filter((el) => matches(el, sel));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }
}

export class Text extends Node {
  constructor(doc, data) {
    super(doc);
    this.data = String(data);
  }
  get nodeType() { return 3; }
  get nodeName() { return "#text"; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v ?? ""); }
}

function classList(el) {
  const tokens = () => (el.getAttribute("class") || "").split(/\s+/).filter(Boolean);
  const write = (list) => el.setAttribute("class", [...new Set(list)].join(" "));
  const check = (t) => {
    if (t === "") throw new DOMException("empty token", "SyntaxError");
    if (/\s/.test(t)) throw new DOMException("token has whitespace", "InvalidCharacterError");
    return t;
  };
  return {
    add: (...ts) => write([...tokens(), ...ts.map(check)]),
    remove: (...ts) => write(tokens().filter((t) => !ts.map(check).includes(t))),
    contains: (t) => tokens().includes(t),
    toggle(t, force) {
      const on = force === undefined ? !this.contains(check(t)) : Boolean(force);
      if (on) this.add(t);
      else this.remove(t);
      return on;
    },
    replace(a, b) {
      if (!this.contains(a)) return false;
      write(tokens().map((t) => (t === a ? check(b) : t)));
      return true;
    },
    item: (i) => tokens()[i] ?? null,
    get length() { return tokens().length; },
    get value() { return tokens().join(" "); },
    toString() { return this.value; },
    [Symbol.iterator]: () => tokens()[Symbol.iterator](),
  };
}

const dataName = (k) => `data-${k.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;

export class Element extends Node {
  #attrs = new Map();
  #value;
  #checked;
  constructor(doc, ns, name) {
    super(doc);
    this.namespaceURI = ns;
    this.localName = ns === HTML_NS ? name.toLowerCase() : name;
    this.classList = classList(this);
    this.dataset = new Proxy({}, {
      get: (_, k) => (typeof k === "string" ? this.getAttribute(dataName(k)) ?? undefined : undefined),
      set: (_, k, v) => { this.setAttribute(dataName(k), v); return true; },
      deleteProperty: (_, k) => { this.removeAttribute(dataName(k)); return true; },
    });
  }
  get nodeType() { return 1; }
  get tagName() { return this.namespaceURI === HTML_NS ? this.localName.toUpperCase() : this.localName; }
  get nodeName() { return this.tagName; }
  #name(n) { return this.namespaceURI === HTML_NS ? String(n).toLowerCase() : String(n); }
  getAttribute(n) { return this.#attrs.get(this.#name(n)) ?? null; }
  setAttribute(n, v) { this.#attrs.set(this.#name(n), String(v)); }
  hasAttribute(n) { return this.#attrs.has(this.#name(n)); }
  removeAttribute(n) { this.#attrs.delete(this.#name(n)); }
  toggleAttribute(n, force) {
    const on = force === undefined ? !this.hasAttribute(n) : Boolean(force);
    if (on) this.setAttribute(n, this.getAttribute(n) ?? "");
    else this.removeAttribute(n);
    return on;
  }
  get attributes() { return [...this.#attrs].map(([name, value]) => ({ name, value })); }
  get id() { return this.getAttribute("id") ?? ""; }
  set id(v) { this.setAttribute("id", v); }
  get className() { return this.getAttribute("class") ?? ""; }
  set className(v) { this.setAttribute("class", v); }
  get children() { return this.childNodes.filter((c) => c instanceof Element); }
  get firstElementChild() { return this.children[0] ?? null; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) {
    this.replaceChildren();
    if (v !== null && v !== undefined && String(v) !== "") this.append(String(v));
  }
  get value() {
    if (this.localName === "option") return this.getAttribute("value") ?? this.textContent;
    if (this.#value !== undefined) return this.#value;
    if (this.localName === "select") {
      const opts = this.querySelectorAll("option");
      return (opts.find((o) => o.hasAttribute("selected")) || opts[0])?.value ?? "";
    }
    if (this.localName === "textarea") return this.textContent;
    return this.getAttribute("value") ?? "";
  }
  set value(v) {
    if (this.localName === "option") this.setAttribute("value", v);
    else this.#value = String(v ?? "");
  }
  get checked() { return this.#checked ?? this.hasAttribute("checked"); }
  set checked(v) { this.#checked = Boolean(v); }
  matches(selector) { return matches(this, parseSelector(selector)); }
  closest(selector) {
    const sel = parseSelector(selector);
    for (let n = this; n instanceof Element; n = n.parentNode) if (matches(n, sel)) return n;
    return null;
  }
  focus() { this.ownerDocument._focus(this); }
  blur() { if (this.ownerDocument.activeElement === this) this.ownerDocument._focus(null); }
  click() { this.dispatchEvent(new Event("click", { bubbles: true, cancelable: true })); }
}
for (const name of ["hidden", "disabled", "inert", "required", "open"]) {
  Object.defineProperty(Element.prototype, name, {
    get() { return this.hasAttribute(name); },
    set(v) { this.toggleAttribute(name, Boolean(v)); },
  });
}

class DialogElement extends Element {
  returnValue = "";
  show() { this.open = true; }
  showModal() {
    if (this.open) throw new DOMException("The dialog is already open.", "InvalidStateError");
    this.open = true;
  }
  close(returnValue) {
    if (!this.open) return;
    if (returnValue !== undefined) this.returnValue = String(returnValue);
    this.open = false;
    this.dispatchEvent(new Event("close"));
  }
}

export class Document extends Node {
  #active = null;
  constructor() {
    super(null);
    this.ownerDocument = null;
    const html = this.createElement("html");
    html.append(this.createElement("head"), this.createElement("body"));
    this.appendChild(html);
  }
  get nodeType() { return 9; }
  get nodeName() { return "#document"; }
  get documentElement() { return this.childNodes[0]; }
  get head() { return this.documentElement.children[0]; }
  get body() { return this.documentElement.children[1]; }
  get activeElement() { return this.#active && this.contains(this.#active) ? this.#active : this.body; }
  createElement(tag) { return this.createElementNS(HTML_NS, tag); }
  createElementNS(ns, name) {
    const El = ns === HTML_NS && String(name).toLowerCase() === "dialog" ? DialogElement : Element;
    return new El(this, ns, String(name));
  }
  createTextNode(data) { return new Text(this, data); }
  getElementById(id) { return this._descendants().find((el) => el.id === String(id)) ?? null; }
  _focus(el) {
    if (el && !this.contains(el)) return;
    const prev = this.#active && this.contains(this.#active) ? this.#active : null;
    if (prev === el) return;
    this.#active = el;
    if (prev) prev.dispatchEvent(new Event("blur"));
    if (el) el.dispatchEvent(new Event("focus"));
  }
}

// ---------------------------------------------------------------- selectors

const IDENT = /[A-Za-z0-9_\-\u0080-\uFFFF]/;

/** Parse a selector list into [[{tag, ids, classes, attrs, comb}]]; throws SyntaxError. */
function parseSelector(text) {
  const s = String(text);
  let i = 0;
  const fail = () => { throw new SyntaxError(`fakedom: unsupported selector ${JSON.stringify(s)}`); };
  const escape = () => {
    i++; // the backslash
    const hex = /^[0-9a-fA-F]{1,6}/.exec(s.slice(i));
    if (hex) {
      i += hex[0].length;
      if (/\s/.test(s[i] || "")) i++;
      return String.fromCodePoint(parseInt(hex[0], 16) || 0xFFFD);
    }
    if (i >= s.length) fail();
    return s[i++];
  };
  const ident = () => {
    let out = "";
    while (i < s.length && (s[i] === "\\" || IDENT.test(s[i]))) out += s[i] === "\\" ? escape() : s[i++];
    return out || fail();
  };
  const quoted = () => {
    const q = s[i++];
    let out = "";
    while (i < s.length && s[i] !== q) out += s[i] === "\\" ? escape() : s[i++];
    if (s[i++] !== q) fail();
    return out;
  };
  const ws = () => { const start = i; while (/\s/.test(s[i] || "")) i++; return i > start; };
  const list = [];
  let parts = [];
  let comb = null;
  ws();
  while (i <= s.length) {
    const c = { tag: null, ids: [], classes: [], attrs: [], comb };
    let any = false;
    for (;;) {
      if (s[i] === "*") { i++; any = true; } else if (s[i] === ".") { i++; c.classes.push(ident()); any = true; } else if (s[i] === "#") { i++; c.ids.push(ident()); any = true; } else if (s[i] === "[") {
        i++; ws();
        const name = ident().toLowerCase();
        ws();
        let value = null;
        if (s[i] === "=") {
          i++; ws();
          value = s[i] === '"' || s[i] === "'" ? quoted() : ident();
          ws();
        }
        if (s[i++] !== "]") fail();
        c.attrs.push({ name, value });
        any = true;
      } else if (!any && i < s.length && (IDENT.test(s[i]) || s[i] === "\\")) { c.tag = ident().toLowerCase(); any = true; } else break;
    }
    if (!any) fail();
    parts.push(c);
    const spaced = ws();
    if (i >= s.length) { list.push(parts); break; }
    if (s[i] === ",") { i++; ws(); list.push(parts); parts = []; comb = null; } else if (s[i] === ">") { i++; ws(); comb = ">"; } else if (spaced) comb = " ";
    else fail();
  }
  return list;
}

function matchCompound(el, c) {
  return (!c.tag || el.localName.toLowerCase() === c.tag)
    && c.ids.every((id) => el.id === id)
    && c.classes.every((k) => el.classList.contains(k))
    && c.attrs.every((a) => (a.value === null ? el.hasAttribute(a.name) : el.getAttribute(a.name) === a.value));
}

function matchParts(el, parts, i) {
  if (!matchCompound(el, parts[i])) return false;
  if (i === 0) return true;
  const up = (n) => (n.parentNode instanceof Element ? n.parentNode : null);
  if (parts[i].comb === ">") return Boolean(up(el)) && matchParts(up(el), parts, i - 1);
  for (let n = up(el); n; n = up(n)) if (matchParts(n, parts, i - 1)) return true;
  return false;
}

function matches(el, list) { return list.some((parts) => matchParts(el, parts, parts.length - 1)); }

// ---------------------------------------------------------------- CSS.escape and storage

/** CSS.escape, as the CSSOM spec defines it. */
export function cssEscape(value) {
  const cps = [...String(value)];
  return cps.map((ch, i) => {
    const cp = ch.codePointAt(0);
    const hex = () => `\\${cp.toString(16)} `;
    if (cp === 0) return "\uFFFD";
    if ((cp >= 1 && cp <= 0x1F) || cp === 0x7F) return hex();
    if (/[0-9]/.test(ch) && (i === 0 || (i === 1 && cps[0] === "-"))) return hex();
    if (i === 0 && ch === "-" && cps.length === 1) return "\\-";
    if (cp >= 0x80 || /[-_0-9A-Za-z]/.test(ch)) return ch;
    return `\\${ch}`;
  }).join("");
}

/** A Storage. Set `throws = true` to make every call throw a SecurityError (blocked storage). */
export class Storage {
  #map = new Map();
  throws = false;
  #guard() { if (this.throws) throw securityError(); }
  get length() { this.#guard(); return this.#map.size; }
  key(i) { this.#guard(); return [...this.#map.keys()][i] ?? null; }
  getItem(k) { this.#guard(); return this.#map.get(String(k)) ?? null; }
  setItem(k, v) { this.#guard(); this.#map.set(String(k), String(v)); }
  removeItem(k) { this.#guard(); this.#map.delete(String(k)); }
  clear() { this.#guard(); this.#map.clear(); }
  /** Test helper (not in the Storage API): the stored keys, ignoring `throws`. */
  keys() { return [...this.#map.keys()]; }
}

export const createStorage = () => new Storage();
export const createDocument = () => new Document();

// ---------------------------------------------------------------- install

let saved = null;

/**
 * Install a fresh document, window, the DOM classes, CSS and both storages as globals.
 * Options: `localStorage` / `sessionStorage` (reuse a Storage, e.g. across a "reload"), and
 * `storageThrows: "localStorage" | "sessionStorage" | "both"` (the accessor itself throws, as in
 * a browser with site data blocked). Returns {document, window, localStorage, sessionStorage}.
 */
export function install({ localStorage = createStorage(), sessionStorage = createStorage(), storageThrows } = {}) {
  if (saved) throw new Error("fakedom is already installed; call uninstall() first");
  const document = createDocument();
  const window = new Target();
  const CSS = { escape: cssEscape };
  const throwing = (k) => storageThrows === "both" || storageThrows === k;
  const props = {
    document: { value: document },
    window: { value: window },
    Node: { value: Node },
    Element: { value: Element },
    Text: { value: Text },
    Event: { value: Event },
    CustomEvent: { value: CustomEvent },
    CSS: { value: CSS },
    localStorage: throwing("localStorage") ? { get() { throw securityError(); } } : { value: localStorage },
    sessionStorage: throwing("sessionStorage") ? { get() { throw securityError(); } } : { value: sessionStorage },
  };
  document._window = window;
  saved = Object.fromEntries(GLOBALS.map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  for (const [k, d] of Object.entries(props)) {
    const desc = { configurable: true, enumerable: false, ...d, ...("value" in d ? { writable: true } : {}) };
    Object.defineProperty(globalThis, k, desc);
    if (k !== "window") Object.defineProperty(window, k, desc);
  }
  return { document, window, localStorage, sessionStorage };
}

/** Restore every global install() touched to exactly what it was. Safe to call twice. */
export function uninstall() {
  if (!saved) return;
  for (const [k, d] of Object.entries(saved)) {
    if (d) Object.defineProperty(globalThis, k, d);
    else delete globalThis[k];
  }
  saved = null;
}
