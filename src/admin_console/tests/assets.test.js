// Node part: the static budget and the font wiring. fonts/*, icons.svg and
// owl.svg total at most 300 KB, and the @font-face url and the preload href name the same file,
// which exists. The pytest part (content types, WAF) is in infra/tests. No PBT: fixed files.
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (name) => readFileSync(join(ROOT, name), "utf8");
const BUDGET = 300 * 1000;

const css = read("base.css").replace(/\/\*[\s\S]*?\*\//g, "");
const faces = [...css.matchAll(/@font-face\s*\{([^}]*)\}/g)].map((m) => m[1]);
const preloads = [...read("index.html").matchAll(/<link\b[^>]*rel="preload"[^>]*>/g)].map((m) => m[0]);

test("fonts/*, icons.svg and owl.svg total at most 300 KB", () => {
  const fonts = readdirSync(join(ROOT, "fonts")).map((n) => join("fonts", n));
  const sizes = [...fonts, "icons.svg", "owl.svg"].map((n) => [n, statSync(join(ROOT, n)).size]);
  const total = sizes.reduce((n, [, b]) => n + b, 0);
  assert.ok(total <= BUDGET, `${total} bytes > ${BUDGET}: ${JSON.stringify(sizes)}`);
});

test("base.css has one @font-face for Inter: url in fonts/, swap, and the README's unicode-range", () => {
  assert.equal(faces.length, 1, "one @font-face");
  const face = faces[0];
  assert.match(face, /font-family:\s*"Inter"/);
  assert.match(face, /font-display:\s*swap/);
  assert.match(face, /font-weight:\s*400 700/);
  const readme = read("fonts/README.md").match(/unicode-range:\s*([^;]+);/)?.[1];
  assert.ok(readme, "fonts/README.md states the unicode-range");
  const range = face.match(/unicode-range:\s*([^;]+)/)?.[1];
  assert.equal(range?.replace(/\s+/g, ""), readme.replace(/\s+/g, ""), "the @font-face range is the subset's range");
});

test("the @font-face url and the font preload name the same file, and it exists", () => {
  const urls = [...faces[0].matchAll(/url\("([^"]+)"\)/g)].map((m) => m[1]);
  assert.deepEqual(urls, ["fonts/Inter-4.1-latin-var.woff2"]);
  assert.equal(preloads.length, 1, "one preload");
  const href = preloads[0].match(/\shref="([^"]+)"/)?.[1];
  assert.equal(href, urls[0], "the preload fetches the file the @font-face uses (else it is fetched twice)");
  const file = join(ROOT, urls[0]);
  assert.ok(existsSync(file), `${urls[0]} exists`);
  const head = readFileSync(file).subarray(0, 4).toString("latin1");
  assert.equal(head, "wOF2", "the file is a WOFF2");
});

test("the favicon and the sidebar mark name owl.svg, which exists", () => {
  assert.match(read("index.html"), /<link rel="icon" href="owl\.svg"/);
  assert.ok(existsSync(join(ROOT, "owl.svg")));
});
