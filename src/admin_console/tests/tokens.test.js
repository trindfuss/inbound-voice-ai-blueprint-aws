// Design tokens. Every declared colour pair meets its WCAG 2.x
// contrast minimum in both themes, and both themes define the same semantic tokens.
// Outside tokens.css no stylesheet holds a colour
// literal, a primitive, or a size that isn't a token; nothing follows prefers-color-scheme.
// No PBT: the pair list is finite, so the exhaustive check is stronger.
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const TOKENS = join(ROOT, "tokens.css");

// The primitives, exactly as the design lists them. Primitives are `--owl-<hue>-<step>`.
export const PALETTE = {
  navy: { 50: "#E6EDF7", 200: "#B7CEF1", 300: "#A9BEDC", 600: "#1F4E8C", 700: "#24457A", 800: "#1B365D", 850: "#18304F", 900: "#10233F", 950: "#0A1628" },
  amber: { 50: "#FDF0D5", 300: "#F5C46B", 400: "#F2A93B", 800: "#7A4600", 950: "#3A2A0C" },
  grey: {
    0: "#FFFFFF", 50: "#F5F7FA", 100: "#EEF1F5", 200: "#DCE1E8", 250: "#C9D1DC", 300: "#A7B1C0", 400: "#7A8494",
    600: "#4D5768", 700: "#3A4352", 750: "#2B3442", 800: "#262E3B", 850: "#1D2430", 900: "#161B24", 950: "#0D1117",
  },
  green: { 50: "#E4F4EA", 300: "#7ED4A0", 700: "#13693A", 950: "#123222" },
  red: { 50: "#FDE9E7", 200: "#F6A8A1", 300: "#F28B82", 700: "#B42318", 800: "#912018", 950: "#3D1513", 975: "#2A0A07" },
  blue: { 50: "#E5EFFC", 300: "#8DB6F2", 600: "#1F5FBF", 700: "#1B5AA6", 950: "#13284A" },
};

// The semantic colour tokens each theme must define.
export const SEMANTIC_COLOURS = [
  "bg", "surface", "surface-sunken", "text", "text-muted", "border", "border-control", "link",
  "primary", "primary-hover", "on-primary", "danger", "danger-hover", "on-danger", "focus",
  ...["success", "warning", "danger", "info", "neutral", "brand"].flatMap((t) => [`${t}-text`, `${t}-bg`]),
  "changed-bg", "skeleton", "overlay",
  "sidebar-bg", "sidebar-hover", "sidebar-active", "sidebar-text", "sidebar-muted", "sidebar-indicator",
  "sidebar-focus", "sidebar-border", "env-test-bg", "env-test-text", "env-prod-bg", "env-prod-text",
].map((n) => `--color-${n}`);

// The colour pairs plus the tinted-row pairs. [fg, bg, minimum, design ratio light, design ratio dark].
export const PAIRS = [
  ["text", "bg", 4.5, 16.09, 16.70],
  ["text", "surface", 4.5, 17.26, 15.24],
  ["text", "surface-sunken", 4.5, 15.24, 13.76],
  ["text-muted", "surface", 4.5, 7.29, 7.97],
  ["text-muted", "bg", 4.5, 6.80, 8.74],
  ["text-muted", "surface-sunken", 4.5, 6.44, 7.19],
  ["link", "surface", 4.5, 8.31, 8.31],
  ["link", "bg", 4.5, 7.74, 9.11],
  ["on-primary", "primary", 4.5, 12.12, 11.32],
  ["on-primary", "primary-hover", 4.5, 15.74, 15.39],
  ["on-danger", "danger", 4.5, 6.57, 7.68],
  ["on-danger", "danger-hover", 4.5, 8.66, 9.64],
  ["danger-text", "surface", 4.5, 6.57, 7.23],
  ["border-control", "surface", 3, 3.78, 4.57],
  ["border-control", "bg", 3, 3.52, 5.01],
  ["border-control", "surface-sunken", 3, 3.34, 4.12],
  ["focus", "surface", 3, 6.09, 8.31],
  ["focus", "bg", 3, 5.68, 9.11],
  ["focus", "surface-sunken", 3, 5.38, 7.50],
  ["success-text", "success-bg", 4.5, 5.93, 7.85],
  ["warning-text", "warning-bg", 4.5, 6.88, 8.57],
  ["danger-text", "danger-bg", 4.5, 5.63, 6.68],
  ["info-text", "info-bg", 4.5, 5.91, 7.07],
  ["neutral-text", "neutral-bg", 4.5, 8.81, 8.88],
  ["brand-text", "brand-bg", 4.5, 10.29, 7.57],
  ["success-text", "surface", 4.5, 6.75, 9.71],
  ["warning-text", "surface", 4.5, 7.77, 10.68],
  ["info-text", "surface", 4.5, 6.86, 8.31],
  ["text", "changed-bg", 4.5, 15.29, 12.22],
  ["sidebar-text", "sidebar-bg", 4.5, 13.36, 15.39],
  ["sidebar-muted", "sidebar-bg", 4.5, 8.31, 9.57],
  ["sidebar-text", "sidebar-hover", 4.5, 11.32, 13.36],
  ["sidebar-text", "sidebar-active", 4.5, 8.07, 10.29],
  ["sidebar-muted", "sidebar-active", 4.5, 5.02, 6.40],
  ["sidebar-indicator", "sidebar-active", 3, 4.76, 6.07],
  ["sidebar-focus", "sidebar-bg", 3, 7.88, 9.08],
  ["sidebar-focus", "sidebar-active", 3, 4.76, 6.07],
  ["env-test-text", "env-test-bg", 4.5, 7.88, 7.88],
  ["env-prod-text", "env-prod-bg", 4.5, 6.57, 6.57],
  // Tinted table rows (after the table).
  ["text", "danger-bg", 4.5, 14.79, 14.09],
  ["text-muted", "warning-bg", 4.5, 6.46, 6.39],
];

// Tokens that live on :root (shadows are per theme, see below).
const SCALE_TOKENS = [
  "--font-sans", "--font-mono",
  ...["xs", "sm", "base", "md", "lg", "xl"].flatMap((s) => [`--text-${s}`, `--leading-${s}`]),
  "--weight-regular", "--weight-medium", "--weight-semibold", "--weight-bold",
  ...[1, 2, 3, 4, 5, 6, 8, 10, 12].map((n) => `--space-${n}`),
  "--radius-sm", "--radius-md", "--radius-lg", "--radius-pill",
  "--focus-width", "--focus-offset",
  "--control-height", "--control-height-sm", "--control-padding-x",
  "--sidebar-width", "--content-max", "--text-max", "--topbar-height",
  "--motion-fast", "--motion-base", "--ease",
];
const THEME_SCALE_TOKENS = ["--shadow-sm", "--shadow-md", "--shadow-lg"];
const COMPONENT_PREFIXES = ["--button-", "--input-", "--card-", "--table-", "--badge-", "--sidebar-", "--toast-", "--panel-"];

// ---- A small CSS reader: rules (with their enclosing @media) and declarations. ----

function stripComments(css) {
  return css.replace(/\/\*[\s\S]*?\*\//g, "");
}

/** Every rule as {media, selector, decls: [[prop, value]]}. Handles one level of @media nesting. */
export function parseRules(css, media = null) {
  const text = stripComments(css);
  const rules = [];
  let i = 0;
  while (i < text.length) {
    const open = text.indexOf("{", i);
    if (open < 0) break;
    const prelude = text.slice(i, open).trim();
    let depth = 1;
    let j = open + 1;
    while (j < text.length && depth > 0) {
      if (text[j] === "{") depth++;
      else if (text[j] === "}") depth--;
      j++;
    }
    const body = text.slice(open + 1, j - 1);
    if (prelude.startsWith("@")) rules.push(...parseRules(body, prelude));
    else rules.push({ media, selector: prelude, decls: parseDecls(body) });
    i = j;
  }
  return rules;
}

function parseDecls(body) {
  const decls = [];
  let depth = 0;
  let start = 0;
  const push = (s) => {
    const k = s.indexOf(":");
    if (k > 0) decls.push([s.slice(0, k).trim(), s.slice(k + 1).trim()]);
  };
  for (let k = 0; k < body.length; k++) {
    if (body[k] === "(") depth++;
    else if (body[k] === ")") depth--;
    else if (body[k] === ";" && depth === 0) {
      push(body.slice(start, k));
      start = k + 1;
    }
  }
  push(body.slice(start));
  return decls;
}

function block(rules, selector, media = null) {
  const found = rules.filter((r) => r.selector === selector && r.media === media);
  assert.equal(found.length, 1, `exactly one ${selector} block${media ? ` in ${media}` : ""} in tokens.css`);
  return Object.fromEntries(found[0].decls);
}

/** Resolve var() references (with fallbacks) against vars, recursively. */
export function resolve(value, vars, seen = new Set()) {
  let out = "";
  let i = 0;
  while (i < value.length) {
    const k = value.indexOf("var(", i);
    if (k < 0) { out += value.slice(i); break; }
    out += value.slice(i, k);
    let depth = 1;
    let j = k + 4;
    while (j < value.length && depth > 0) {
      if (value[j] === "(") depth++;
      else if (value[j] === ")") depth--;
      j++;
    }
    const inner = value.slice(k + 4, j - 1);
    const comma = inner.indexOf(",");
    const name = (comma < 0 ? inner : inner.slice(0, comma)).trim();
    if (seen.has(name)) throw new Error(`var() cycle at ${name}`);
    if (name in vars) out += resolve(vars[name], vars, new Set([...seen, name]));
    else if (comma >= 0) out += resolve(inner.slice(comma + 1).trim(), vars, seen);
    else throw new Error(`undefined ${name}`);
    i = j;
  }
  return out.trim();
}

// ---- WCAG 2.x relative luminance and contrast. ----

function luminance(hex) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) throw new Error(`not a 6-digit hex colour: ${hex}`);
  const [r, g, b] = [0, 2, 4].map((o) => {
    const c = parseInt(m[1].slice(o, o + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

// ---- The static scan for stylesheets other than tokens.css. ----

const NAMED_COLOURS = new Set(`aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue
blueviolet brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue
darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid
darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink
deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite gold
goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki lavender lavenderblush
lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow lightgray lightgreen lightgrey
lightpink lightsalmon lightseagreen lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime
limegreen linen magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen
mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream mistyrose moccasin
navajowhite navy oldlace olive olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise
palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red rosybrown royalblue
saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue slategray slategrey snow
springgreen steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow yellowgreen`
  .split(/\s+/));
const SIZED = /^(font-size|padding(-[a-z-]+)?|margin(-[a-z-]+)?|gap|row-gap|column-gap|border(-[a-z]+)*-radius|box-shadow)$/;

/** Split a value into top-level space/comma-separated parts (var() kept whole). */
function parts(value) {
  const out = [];
  let depth = 0;
  let cur = "";
  for (const ch of value) {
    if (ch === "(") depth++;
    if (ch === ")") depth--;
    if (depth === 0 && /[\s,]/.test(ch)) {
      if (cur) out.push(cur);
      cur = "";
    } else cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

/** Violations in one stylesheet's text, as readable strings. */
export function scanCss(css) {
  const problems = [];
  if (/prefers-color-scheme/i.test(css)) problems.push("prefers-color-scheme");
  if (/--owl-/.test(stripComments(css))) problems.push("references a primitive (--owl-*)");
  for (const { selector, decls } of parseRules(css)) {
    for (const [prop, raw] of decls) {
      const value = raw.replace(/"[^"]*"|'[^']*'/g, '""');
      const where = `${selector} { ${prop}: ${raw} }`;
      if (/#[0-9a-f]{3,8}\b/i.test(value)) problems.push(`hex colour in ${where}`);
      if (/\b(rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(/i.test(value)) problems.push(`colour function in ${where}`);
      const words = value.replace(/var\([^)]*\)/g, " ").match(/(?<![\w-])[a-z]+(?![\w-])/gi) ?? [];
      for (const w of words) if (NAMED_COLOURS.has(w.toLowerCase())) problems.push(`named colour "${w}" in ${where}`);
      if (SIZED.test(prop) && !parts(raw.replace(/\s*!important$/, "")).every((p) => /^var\(.*\)$/.test(p) || p === "0" || p === "auto")) {
        problems.push(`size not a token in ${where}`);
      }
    }
  }
  return problems;
}

function files(dir = ROOT) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return name === "tests" || name === "node_modules" ? [] : files(p);
    return [p];
  });
}

// ---- Tests ----

const rules = parseRules(readFileSync(TOKENS, "utf8"));
const root = block(rules, ":root");
const light = block(rules, '[data-theme="light"]');
const dark = block(rules, '[data-theme="dark"]');
const themes = { light: { ...root, ...light }, dark: { ...root, ...dark } };

test("tokens.css: the primitives on :root, with the exact hex values", () => {
  for (const [hue, steps] of Object.entries(PALETTE)) {
    for (const [step, hex] of Object.entries(steps)) {
      assert.equal(root[`--owl-${hue}-${step}`]?.toUpperCase(), hex, `--owl-${hue}-${step}`);
    }
  }
  const primitives = Object.keys(root).filter((k) => k.startsWith("--owl-"));
  assert.equal(primitives.length, Object.values(PALETTE).reduce((n, s) => n + Object.keys(s).length, 0), "no extra primitives");
});

test("tokens.css: primitives and colour literals stay on :root and in the theme blocks", () => {
  for (const r of rules) {
    const isTheme = r.media === null && [":root", '[data-theme="light"]', '[data-theme="dark"]'].includes(r.selector);
    for (const [prop, value] of r.decls) {
      if (prop.startsWith("--owl-")) assert.equal(r.selector, ":root", `${prop} only on :root`);
      if (!isTheme) assert.doesNotMatch(value, /#[0-9a-f]{3,8}\b|--owl-|rgb\(/i, `${r.selector} { ${prop} }`);
    }
  }
  // Tier 3 references tier 2 only: component tokens never name a primitive or a literal colour.
  for (const [prop, value] of Object.entries({ ...root, ...light, ...dark })) {
    if (COMPONENT_PREFIXES.some((p) => prop.startsWith(p))) {
      assert.doesNotMatch(value, /#[0-9a-f]{3,8}\b|--owl-|rgb\(/i, `${prop} uses semantic tokens only`);
    }
  }
});

test("tokens.css: both themes define the same semantic tokens, each with color-scheme", () => {
  assert.equal(light["color-scheme"], "light");
  assert.equal(dark["color-scheme"], "dark");
  assert.deepEqual(Object.keys(light).sort(), Object.keys(dark).sort());
  for (const name of [...SEMANTIC_COLOURS, ...THEME_SCALE_TOKENS]) {
    assert.ok(name in light, `${name} in the light theme`);
  }
  for (const name of Object.keys(light).filter((n) => n.startsWith("--color-"))) {
    assert.ok(SEMANTIC_COLOURS.includes(name), `${name} is a semantic colour token`);
  }
  for (const name of SEMANTIC_COLOURS) assert.ok(!(name in root), `${name} is per theme, not on :root`);
});

test("tokens.css: the scales, the component tokens and the panel borders", () => {
  for (const name of SCALE_TOKENS) assert.ok(name in root, `${name} on :root`);
  for (const p of COMPONENT_PREFIXES) {
    assert.ok(Object.keys(root).some((k) => k.startsWith(p)), `a ${p}* component token`);
  }
  for (const tone of ["error", "warning", "info", "success"]) {
    assert.match(root[`--panel-${tone}-border`] ?? "", /^color-mix\(in srgb, var\(--color-[a-z]+-text\) 30%, transparent\)$/);
  }
  for (const theme of Object.values(themes)) {
    for (const name of Object.keys(theme)) resolve(`var(${name})`, theme); // every chain resolves, no cycles
  }
});

test("tokens.css: reduced motion turns the motion tokens off", () => {
  const reduced = rules.filter((r) => /prefers-reduced-motion:\s*reduce/.test(r.media ?? ""));
  const vars = Object.fromEntries(reduced.find((r) => r.selector === ":root")?.decls ?? []);
  assert.equal(vars["--motion-fast"], "0ms");
  assert.equal(vars["--motion-base"], "0ms");
  assert.equal(Object.fromEntries(reduced.find((r) => r.selector === ".skeleton")?.decls ?? []).animation, "none");
});

test("every colour pair meets its minimum in both themes (and matches the design's ratio)", () => {
  assert.equal(PAIRS.length, 41, "39 table pairs plus the two tinted-row pairs");
  const table = [];
  for (const [fg, bg, min, ...expected] of PAIRS) {
    ["light", "dark"].forEach((name, k) => {
      const f = resolve(`var(--color-${fg})`, themes[name]).toUpperCase();
      const b = resolve(`var(--color-${bg})`, themes[name]).toUpperCase();
      const ratio = contrast(f, b);
      table.push({ pair: `${fg} / ${bg}`, theme: name, f, b, ratio: ratio.toFixed(2), min });
      assert.ok(ratio >= min, `${fg} / ${bg} (${name}): ${f} on ${b} is ${ratio.toFixed(2)}:1, needs ${min}:1`);
      assert.ok(Math.abs(ratio - expected[k]) < 0.015, `${fg} / ${bg} (${name}): ${ratio.toFixed(2)} vs design ${expected[k]}`);
    });
  }
  if (process.env.OWL_CONTRAST_TABLE) console.table(table);
});

test("the scan catches a planted colour, primitive, size and system theme", () => {
  assert.deepEqual(scanCss(".ok { color: var(--color-text); padding: var(--space-2) 0; margin: 0 auto; }"), []);
  assert.match(scanCss(".x { color: #fff; }").join(), /hex colour/);
  assert.match(scanCss(".x { background: rgb(0 0 0 / .5); }").join(), /colour function/);
  assert.match(scanCss(".x { border: 1px solid white; }").join(), /named colour/);
  assert.match(scanCss(".x { color: var(--owl-navy-900); }").join(), /primitive/);
  assert.match(scanCss(".x { padding: 4px 8px; }").join(), /size not a token/);
  assert.match(scanCss("@media (min-width: 960px) { .x { gap: 1rem; } }").join(), /size not a token/);
  assert.match(scanCss("@media (prefers-color-scheme: dark) { .x { color: var(--color-text); } }").join(), /prefers-color-scheme/);
  assert.deepEqual(scanCss("#main, #nav { color: transparent; border-color: currentColor; fill: inherit; }"), []);
});

test("every stylesheet except tokens.css uses tokens only", () => {
  const sheets = files().filter((f) => f.endsWith(".css") && relative(ROOT, f) !== "tokens.css");
  assert.ok(sheets.length >= 4, `the other stylesheets are scanned: ${sheets.map((f) => relative(ROOT, f)).join()}`);
  for (const f of sheets) assert.deepEqual(scanCss(readFileSync(f, "utf8")), [], relative(ROOT, f));
});

test("nothing follows prefers-color-scheme", () => {
  for (const f of files().filter((p) => /\.(css|html|js)$/.test(p))) {
    assert.doesNotMatch(readFileSync(f, "utf8"), /prefers-color-scheme/i, relative(ROOT, f));
  }
});

test("index.html: tokens.css loads first, and the page opens light", () => {
  const html = readFileSync(join(ROOT, "index.html"), "utf8");
  assert.match(html, /<html\b[^>]*\sdata-theme="light"/);
  const sheets = [...html.matchAll(/<link rel="stylesheet" href="([^"]+)">/g)].map((m) => m[1]);
  assert.equal(sheets[0], "tokens.css", sheets.join());
});
