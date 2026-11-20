// The icon sprite, the owl mark and icon().
// Every icon the code or the HTML references is a <symbol> in icons.svg, and both SVGs
// parse as XML. The scan is live, so icons added by later tasks are checked too.
// SVG and JS part: the SVG files carry no style, script, handler, foreignObject, image or
// external reference, and no .js builds SVG from markup strings.
// No PBT: the sprite is a fixed file, so checking every symbol is exhaustive.
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { icon } from "../lib/dom.js";
import { install, uninstall } from "./fakedom.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SVG_NS = "http://www.w3.org/2000/svg";

/** The 20 symbols. */
const ICONS = ["menu", "x", "chevron-left", "chevron-down", "tools", "masking", "prompts", "history",
  "users", "slots", "redaction", "sun", "moon", "sign-out", "check-circle", "alert-circle",
  "alert-triangle", "info", "shield-alert", "external"];

function files(dir = ROOT) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return name === "tests" ? [] : files(p);
    return [p];
  });
}
const read = (name) => readFileSync(join(ROOT, name), "utf8");

// ---------------------------------------------------------------- a minimal XML parser

/**
 * Parse a well-formed XML document into {name, attrs, children, text} elements. Throws on
 * anything malformed: unbalanced or mismatched tags, unquoted or duplicate attributes, stray
 * `<` or `&`, text or a second element outside the root. Enough for two hand-written SVG files.
 */
export function parseXml(src) {
  let i = 0;
  const fail = (msg) => { throw new Error(`XML: ${msg} at offset ${i}`); };
  const ws = () => { while (/\s/.test(src[i] ?? "")) i++; };
  const NAME = /[A-Za-z_][\w.:-]*/y;
  const name = () => {
    NAME.lastIndex = i;
    const m = NAME.exec(src);
    if (!m) fail("expected a name");
    i += m[0].length;
    return m[0];
  };
  const checkText = (t) => {
    if (/</.test(t)) fail("stray <");
    for (const m of t.matchAll(/&/g)) {
      if (!/^&(lt|gt|amp|quot|apos|#\d+|#x[0-9a-f]+);/i.test(t.slice(m.index))) fail("bad entity");
    }
    return t;
  };
  const skipMisc = () => {
    for (;;) {
      ws();
      if (src.startsWith("<?", i)) { const e = src.indexOf("?>", i); if (e < 0) fail("open <?"); i = e + 2; }
      else if (src.startsWith("<!--", i)) { const e = src.indexOf("-->", i); if (e < 0) fail("open comment"); i = e + 3; }
      else return;
    }
  };
  const element = () => {
    if (src[i] !== "<") fail("expected <");
    i++;
    const el = { name: name(), attrs: {}, children: [], text: "" };
    for (;;) {
      const before = i;
      ws();
      if (src.startsWith("/>", i)) { i += 2; return el; }
      if (src[i] === ">") { i++; break; }
      if (i === before) fail("expected whitespace before an attribute");
      const a = name();
      ws();
      if (src[i] !== "=") fail(`attribute ${a} has no value`);
      i++;
      ws();
      const q = src[i];
      if (q !== '"' && q !== "'") fail(`attribute ${a} is not quoted`);
      const end = src.indexOf(q, i + 1);
      if (end < 0) fail("unterminated attribute");
      if (a in el.attrs) fail(`duplicate attribute ${a}`);
      el.attrs[a] = checkText(src.slice(i + 1, end));
      i = end + 1;
    }
    for (;;) {
      const lt = src.indexOf("<", i);
      if (lt < 0) fail(`unclosed <${el.name}>`);
      el.text += checkText(src.slice(i, lt));
      i = lt;
      if (src.startsWith("<!--", i)) { const e = src.indexOf("-->", i); if (e < 0) fail("open comment"); i = e + 3; continue; }
      if (src.startsWith("</", i)) {
        i += 2;
        const n = name();
        if (n !== el.name) fail(`</${n}> closes <${el.name}>`);
        ws();
        if (src[i] !== ">") fail("expected >");
        i++;
        return el;
      }
      el.children.push(element());
    }
  };
  skipMisc();
  const root = element();
  skipMisc();
  if (i !== src.length) fail("content after the root element");
  return root;
}

const walk = (el) => [el, ...el.children.flatMap(walk)];

test("parseXml accepts well-formed XML and rejects malformed XML", () => {
  const ok = parseXml('<?xml version="1.0"?>\n<a x="1"><!-- c --><b y=\'2\'/>t &amp; u</a>\n');
  assert.equal(ok.name, "a");
  assert.equal(ok.children[0].attrs.y, "2");
  for (const bad of ["<a>", "<a></b>", "<a x=1/>", '<a x="1" x="2"/>', "<a/><b/>", "<a>&</a>",
    '<a x="<"/>', "text<a/>", "<a><b></a></b>", '<ax="1"/>', "<a x/>"]) {
    assert.throws(() => parseXml(bad), /XML:/, bad);
  }
});

// ---------------------------------------------------------------- the sprite

test("icons.svg parses as XML and has exactly the 20 symbols", () => {
  const root = parseXml(read("icons.svg"));
  assert.equal(root.name, "svg");
  assert.equal(root.attrs.xmlns, SVG_NS);
  const symbols = root.children.filter((c) => c.name === "symbol");
  assert.equal(symbols.length, root.children.length, "only <symbol> children at the top level");
  assert.deepEqual(symbols.map((s) => s.attrs.id).sort(), [...ICONS].sort());
});

test("every symbol is a 24-unit stroke icon with presentation attributes", () => {
  const root = parseXml(read("icons.svg"));
  for (const s of root.children) {
    const id = s.attrs.id;
    assert.equal(s.attrs.viewBox, "0 0 24 24", id);
    assert.equal(s.attrs.fill, "none", id);
    assert.equal(s.attrs.stroke, "currentColor", id);
    assert.equal(s.attrs["stroke-width"], "1.75", id);
    assert.equal(s.attrs["stroke-linecap"], "round", id);
    assert.equal(s.attrs["stroke-linejoin"], "round", id);
    assert.ok(s.children.length > 0, `${id} draws something`);
    for (const el of walk(s).slice(1)) {
      assert.ok(["path", "circle", "rect", "line", "polyline", "g"].includes(el.name), `${id}: <${el.name}>`);
      for (const k of ["fill", "stroke"]) {
        if (k in el.attrs) assert.ok(["none", "currentColor"].includes(el.attrs[k]), `${id}: ${k}=${el.attrs[k]}`);
      }
    }
  }
});

// ---------------------------------------------------------------- the owl mark

test("owl.svg parses as XML, has its title, a 32-unit viewBox and fixed colours", () => {
  const root = parseXml(read("owl.svg"));
  assert.equal(root.name, "svg");
  assert.equal(root.attrs.xmlns, SVG_NS);
  assert.equal(root.attrs.viewBox, "0 0 32 32");
  const title = root.children[0];
  assert.equal(title?.name, "title", "the <title> comes first");
  assert.equal(title.text, "OwlBank");
  const colours = new Set();
  for (const el of walk(root)) {
    for (const k of ["fill", "stroke"]) {
      if (!(k in el.attrs)) continue;
      assert.notEqual(el.attrs[k], "currentColor", "the mark doesn't change with the theme");
      if (el.attrs[k] !== "none") colours.add(el.attrs[k].toUpperCase());
    }
  }
  // amber-400 tile and beak, navy-900 owl, cream eyes, amber-300 breast
  assert.deepEqual([...colours].sort(), ["#10233F", "#F2A93B", "#F5C46B", "#FFF8EB"]);
  assert.match(walk(root).find((e) => e.name === "rect")?.attrs.rx ?? "", /^7$/, "the tile is rx 7");
  const text = read("owl.svg");
  assert.doesNotMatch(text, /<text\b/, "no letters");
});

test("size budget: sprite under 8 KB, mark under 2 KB", () => {
  assert.ok(statSync(join(ROOT, "icons.svg")).size < 8 * 1024, "icons.svg < 8 KB");
  assert.ok(statSync(join(ROOT, "owl.svg")).size < 2 * 1024, "owl.svg < 2 KB");
});

// ---------------------------------------------------------------- SVG and JS part

test("the SVG files have no style, script, handlers, foreignObject, image or external href", () => {
  const svgs = files().filter((f) => f.endsWith(".svg"));
  assert.ok(svgs.length >= 2);
  for (const f of svgs) {
    const name = relative(ROOT, f);
    const text = readFileSync(f, "utf8");
    assert.doesNotMatch(text, /\sstyle\s*=/i, `${name}: style=`);
    assert.doesNotMatch(text, /<style\b/i, `${name}: <style>`);
    assert.doesNotMatch(text, /<script\b/i, `${name}: <script>`);
    assert.doesNotMatch(text, /\son[a-z]+\s*=/i, `${name}: on*=`);
    assert.doesNotMatch(text, /<foreignObject\b/i, `${name}: <foreignObject>`);
    assert.doesNotMatch(text, /<image\b/i, `${name}: <image>`);
    for (const el of walk(parseXml(text))) {
      for (const [k, v] of Object.entries(el.attrs)) {
        if (/(^|:)href$/i.test(k)) assert.match(v, /^#[\w-]+$/, `${name}: external ${k}="${v}"`);
        assert.doesNotMatch(v, /url\(\s*['"]?(?!#)/i, `${name}: external url() in ${k}`);
      }
    }
  }
});

test("no .js contains <svg or <path markup strings", () => {
  for (const f of files().filter((p) => p.endsWith(".js"))) {
    assert.doesNotMatch(readFileSync(f, "utf8"), /<\s*(svg|path)\b/i, relative(ROOT, f));
  }
});

// ----------------------------------------------------------------: every referenced icon exists

/** Icon names referenced in a JS source: icon("…") literals and `icon: "…"` (SECTIONS, badge, emptyState). */
export function jsIconRefs(text) {
  const refs = [];
  for (const m of text.matchAll(/\bicon\(\s*(["'`])([^"'`]*)\1/g)) refs.push(m[2]);
  for (const m of text.matchAll(/\bicon\s*:\s*(["'`])([^"'`]*)\1/g)) refs.push(m[2]);
  return refs;
}

/** Icon names referenced by `<use href="icons.svg#…">` in HTML. Any other use href is returned as-is. */
export function htmlIconRefs(text) {
  return [...text.matchAll(/<use\b[^>]*\bhref\s*=\s*"([^"]*)"/gi)].map((m) => m[1].replace(/^icons\.svg#/, ""));
}

test("the reference scanners find literals, SECTIONS entries and <use> hrefs", () => {
  assert.deepEqual(jsIconRefs('icon("menu"); icon( \'x\', {label:"Close"}); const S = [{ id: "a", icon: "tools" }];'
    + " badge(t, 'info', { icon: \"info\" }); icon(name);"), ["menu", "x", "tools", "info"]);
  assert.deepEqual(htmlIconRefs('<svg class="icon" aria-hidden="true"><use href="icons.svg#menu"/></svg>'
    + '<use href="other.svg#x"></use>'), ["menu", "other.svg#x"]);
});

test("every icon referenced in .js, SECTIONS and index.html is a symbol in icons.svg", () => {
  const ids = new Set(parseXml(read("icons.svg")).children.map((s) => s.attrs.id));
  const refs = [];
  for (const f of files()) {
    const name = relative(ROOT, f);
    if (f.endsWith(".js")) refs.push(...jsIconRefs(readFileSync(f, "utf8")).map((r) => [name, r]));
    if (f.endsWith(".html")) refs.push(...htmlIconRefs(readFileSync(f, "utf8")).map((r) => [name, r]));
  }
  for (const [name, r] of refs) assert.ok(ids.has(r), `${name}: icon "${r}" is not in icons.svg`);
});

test("a planted unknown icon would fail the icon check", () => {
  const ids = new Set(ICONS);
  assert.equal(jsIconRefs('icon("no-such-icon")').every((r) => ids.has(r)), false);
  assert.equal(htmlIconRefs('<use href="icons.svg#nope"/>').every((r) => ids.has(r)), false);
});

// ---------------------------------------------------------------- icon()

function withDom(fn) {
  install();
  try {
    return fn();
  } finally {
    uninstall();
  }
}

test("icon(name) is a decorative svg.icon > use[href=icons.svg#name], built with createElementNS", () => {
  withDom(() => {
    const svg = icon("tools");
    assert.equal(svg.namespaceURI, SVG_NS);
    assert.equal(svg.localName, "svg");
    assert.ok(svg.classList.contains("icon"));
    assert.equal(svg.getAttribute("aria-hidden"), "true");
    assert.equal(svg.getAttribute("focusable"), "false");
    assert.equal(svg.hasAttribute("role"), false);
    assert.equal(svg.hasAttribute("aria-label"), false);
    assert.equal(svg.childNodes.length, 1);
    const use = svg.firstElementChild;
    assert.equal(use.namespaceURI, SVG_NS);
    assert.equal(use.localName, "use");
    assert.equal(use.getAttribute("href"), "icons.svg#tools");
    assert.equal(svg.querySelector("use").getAttribute("href"), "icons.svg#tools");
    for (const el of [svg, use]) {
      for (const { name } of el.attributes) assert.doesNotMatch(name, /^(style|on.*)$/i);
    }
    // it slots into h() like any node
    const parent = document.createElement("span");
    parent.append(svg);
    assert.equal(parent.querySelector("svg.icon > use").getAttribute("href"), "icons.svg#tools");
  });
});

test("icon(name, {label}) is announced: role=img and aria-label, not aria-hidden", () => {
  withDom(() => {
    const svg = icon("alert-triangle", { label: "Warning" });
    assert.equal(svg.getAttribute("role"), "img");
    assert.equal(svg.getAttribute("aria-label"), "Warning");
    assert.equal(svg.hasAttribute("aria-hidden"), false);
    assert.ok(svg.classList.contains("icon"));
    assert.equal(svg.firstElementChild.getAttribute("href"), "icons.svg#alert-triangle");
    // an empty label is no label
    assert.equal(icon("info", { label: "" }).getAttribute("aria-hidden"), "true");
  });
});

test("icon() works for every symbol and refuses names that aren't symbol ids", () => {
  withDom(() => {
    for (const n of ICONS) assert.equal(icon(n).firstElementChild.getAttribute("href"), `icons.svg#${n}`);
    for (const bad of ["", "Menu", "a b", "x\"", "../evil", "http://x/y#z", "a#b", undefined, 7]) {
      assert.throws(() => icon(bad), /icon name/, String(bad));
    }
  });
});
