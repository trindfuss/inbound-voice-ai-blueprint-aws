// The site's files keep to the CSP (default-src 'self', no inline script or style, no
// external scripts or fonts) and every module resolves.
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

function files(dir = ROOT) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return name === "tests" ? [] : files(p);
    return [p];
  });
}

// Exactly one classic script, theme.js, in <head> before the stylesheets, with no async or
// defer (it must run before first paint). Every other script is a module. No inline script.
// Returns the list of violations so planted bad pages can be checked too.
function scriptViolations(html) {
  const out = [];
  const headEnd = html.search(/<\/head>/i);
  const firstSheet = html.search(/<link\b[^>]*rel="stylesheet"/i);
  const classic = [];
  for (const m of html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
    const attrs = m[1];
    if (!/\bsrc="[a-z0-9_./-]+\.js"/.test(attrs)) out.push("a script must have a relative src");
    if (m[2].trim() !== "") out.push("no inline script");
    if (/type="module"/.test(attrs)) continue;
    classic.push({ attrs, at: m.index });
  }
  if (classic.length !== 1) out.push(`exactly one classic script (theme.js), found ${classic.length}`);
  for (const c of classic) {
    if (!/^\s*src="theme\.js"\s*$/.test(c.attrs)) out.push(`the classic script is <script src="theme.js"> with no other attribute: ${c.attrs.trim()}`);
    if (headEnd < 0 || c.at > headEnd) out.push("theme.js is in <head>");
    if (firstSheet >= 0 && c.at > firstSheet) out.push("theme.js comes before the stylesheets");
  }
  return out;
}

test("index.html: one classic theme.js, otherwise module scripts and stylesheets from this origin, no inline code", () => {
  const html = readFileSync(join(ROOT, "index.html"), "utf8");
  assert.deepEqual(scriptViolations(html), []);
  assert.doesNotMatch(html, /<style\b/i);
  assert.doesNotMatch(html, /\sstyle=/i);
  assert.doesNotMatch(html, /\son[a-z]+=/i, "no inline event handlers");
  assert.doesNotMatch(html, /(src|href)="(https?:)?\/\//i, "no external resources");
});

// LocalStorage appears only in theme.js and lib/theme.js, and there only with the theme key
// (static part). Every other file, lib/auth.js among them, stays banned.
const STORAGE_FILES = new Set(["theme.js", "lib/theme.js"]);
const THEME_KEY_ARG = /^\s*(THEME_KEY|"owlbank-admin-theme"|'owlbank-admin-theme')\s*$/;
function storageViolations(name, text) {
  const file = name.split(sep).join("/");
  if (!STORAGE_FILES.has(file)) {
    return /localStorage/.test(text) ? [`${file}: localStorage is only for the theme (theme.js, lib/theme.js)`] : [];
  }
  const out = [];
  for (const m of text.matchAll(/\.(getItem|setItem|removeItem)\(([^,)]*)/g)) {
    if (!THEME_KEY_ARG.test(m[2])) out.push(`${file}: ${m[1]}(${m[2].trim()}) is not the theme key`);
  }
  if (/localStorage\s*(\[|\.(?!getItem\b|setItem\b|removeItem\b))/.test(text)) {
    out.push(`${file}: localStorage is used only through getItem/setItem/removeItem with the theme key`);
  }
  if (/\.(clear|key)\(/.test(text)) out.push(`${file}: no clear() or key() on storage`);
  return out;
}

// url( is allowed only as url("fonts/<name>.woff2") inside @font-face. No scheme, no data:,
// no other path, no @import (anything else could load from elsewhere or bypass the font budget).
const FONT_URL = /^url\("fonts\/[A-Za-z0-9][A-Za-z0-9_.-]*\.woff2"\)/;
function cssUrlViolations(css) {
  const text = css.replace(/\/\*[\s\S]*?\*\//g, "");
  const out = [];
  if (/@import/i.test(text)) out.push("no @import");
  const faces = [...text.matchAll(/@font-face\s*\{[^}]*\}/gi)].map((m) => [m.index, m.index + m[0].length]);
  for (const m of text.matchAll(/url\(/gi)) {
    const inFace = faces.some(([a, b]) => m.index > a && m.index < b);
    const tail = text.slice(m.index);
    if (!inFace) out.push(`url( outside @font-face: ${tail.slice(0, 40)}`);
    else if (!FONT_URL.test(tail) || /\.\./.test(tail.match(/^url\([^)]*\)/)?.[0] ?? "")) {
      out.push(`@font-face url must be url("fonts/<name>.woff2"): ${tail.slice(0, 40)}`);
    }
  }
  return out;
}

test("the site rules catch planted violations", () => {
  const page = (head) => `<!doctype html><html><head>${head}</head><body></body></html>`;
  const sheet = '<link rel="stylesheet" href="tokens.css">';
  const mod = '<script type="module" src="app.js"></script>';
  const ok = page(`<script src="theme.js"></script>${sheet}${mod}`);
  assert.deepEqual(scriptViolations(ok), []);
  assert.notDeepEqual(scriptViolations(page(`<script src="theme.js"></script><script src="other.js"></script>${sheet}${mod}`)), [], "a second classic script");
  assert.notDeepEqual(scriptViolations(page(`<script src="theme.js" defer></script>${sheet}${mod}`)), [], "defer on theme.js");
  assert.notDeepEqual(scriptViolations(page(`<script async src="theme.js"></script>${sheet}${mod}`)), [], "async on theme.js");
  assert.notDeepEqual(scriptViolations(page(`${sheet}${mod}`)), [], "no theme.js");
  assert.notDeepEqual(scriptViolations(page(`${sheet}<script src="theme.js"></script>${mod}`)), [], "theme.js after the stylesheets");
  assert.notDeepEqual(scriptViolations(`<html><head>${sheet}${mod}</head><body><script src="theme.js"></script></body></html>`), [], "theme.js in <body>");
  assert.notDeepEqual(scriptViolations(page(`<script src="theme.js"></script>${sheet}<script type="module">go()</script>`)), [], "inline script");
  assert.notDeepEqual(scriptViolations(page(`<script src="theme.js">go()</script>${sheet}${mod}`)), [], "inline code in theme.js");

  assert.deepEqual(storageViolations("theme.js", 'try { t = window.localStorage.getItem("owlbank-admin-theme"); } catch (e) {}'), []);
  assert.deepEqual(storageViolations("lib/theme.js", "storage?.setItem(THEME_KEY, v); storage?.getItem(THEME_KEY);"), []);
  assert.notDeepEqual(storageViolations("lib/auth.js", 'localStorage.setItem("tokens", t)'), [], "localStorage in lib/auth.js");
  assert.notDeepEqual(storageViolations("lib/shell.js", 'safeStorage("localStorage")'), [], "localStorage elsewhere");
  assert.notDeepEqual(storageViolations("theme.js", 'localStorage.setItem("owlbank-admin-token", t)'), [], "another key in theme.js");
  assert.notDeepEqual(storageViolations("lib/theme.js", "localStorage.token = t"), [], "property write");
  assert.notDeepEqual(storageViolations("lib/theme.js", 'localStorage["x"] = t'), [], "bracket write");
  assert.notDeepEqual(storageViolations("lib/theme.js", "localStorage.clear()"), [], "clear()");

  const face = (src) => `@font-face { font-family: "Inter"; font-display: swap; src: ${src} format("woff2"); }`;
  assert.deepEqual(cssUrlViolations(face('url("fonts/Inter-4.1-latin-var.woff2")')), []);
  assert.deepEqual(cssUrlViolations('/* url(https://x) in a comment */ .x { color: var(--color-text); }'), []);
  assert.notDeepEqual(cssUrlViolations(face('url("data:font/woff2;base64,AAAA")')), [], "url(data:) in @font-face");
  assert.notDeepEqual(cssUrlViolations(face("url(data:font/woff2;base64,AAAA)")), [], "unquoted url(data:)");
  assert.notDeepEqual(cssUrlViolations(face('url("https://fonts.example/x.woff2")')), [], "url(https:) in @font-face");
  assert.notDeepEqual(cssUrlViolations(face('url("//cdn.example/fonts/x.woff2")')), [], "protocol-relative url");
  assert.notDeepEqual(cssUrlViolations(face('url("fonts/../x.woff2")')), [], "a path out of fonts/");
  assert.notDeepEqual(cssUrlViolations(face('url("fonts/x.ttf")')), [], "not a woff2");
  assert.notDeepEqual(cssUrlViolations(face("url(fonts/x.woff2)")), [], "unquoted");
  assert.notDeepEqual(cssUrlViolations('.x { background: url("fonts/x.woff2"); }'), [], "url( outside @font-face");
  assert.notDeepEqual(cssUrlViolations('.x { background-image: url(data:image/png;base64,AAAA); }'), [], "url(data:) in a rule");
  assert.notDeepEqual(cssUrlViolations('.x { background: URL(https://evil.example/a.png); }'), [], "upper-case URL(https:)");
  assert.notDeepEqual(cssUrlViolations('@import "other.css";'), [], "@import");
  assert.notDeepEqual(cssUrlViolations('@import url("https://evil.example/a.css");'), [], "@import url()");
});

test("CSS and JS load nothing from elsewhere and never write HTML", () => {
  for (const f of files()) {
    const text = readFileSync(f, "utf8");
    const name = relative(ROOT, f);
    if (f.endsWith(".css")) {
      assert.deepEqual(cssUrlViolations(text), [], name);
    }
    if (f.endsWith(".js")) {
      assert.doesNotMatch(text, /\binnerHTML\b|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function\(/, name);
      assert.doesNotMatch(text, /\.style\.|setAttribute\(\s*["']style/, name);
      assert.deepEqual(storageViolations(name, text), [], `${name}: tokens are kept in memory only`);
      for (const m of text.matchAll(/^import .* from "([^"]+)";$/gm)) {
        assert.match(m[1], /^\.{1,2}\//, `${name}: only relative imports (no packages, no CDNs)`);
      }
    }
  }
});

test("no config.js in the repository (generated at deploy)", () => {
  assert.equal(files().some((f) => relative(ROOT, f) === "config.js"), false);
});

test("every view and library module resolves", async () => {
  for (const f of files().filter((p) => p.endsWith(".js") && !p.endsWith("app.js"))) {
    await import(pathToFileURL(f).href);
  }
});
