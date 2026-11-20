// Whole-console checks. style.css is gone: index.html links exactly the five
// stylesheets of the console, in order, and its main is labelled by page-title. The legacy classes
// (.top, .scroll, .warn, .mark, .busy, .ok, .error) appear in no stylesheet, no page and no
// module. Property: every route rendered through SECTIONS with generated
// payloads, users and API failures, and every bare page from start(), uses only classes the
// five stylesheets style; the style and structure checks run on the same trees.
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { SECTIONS, render, start } from "../app.js";
import { ApiError, mapError } from "../lib/api.js";
import { GROUPS, MASKING_FIELDS } from "../lib/validate.js";
import { array, forAll, gen, int, json, mulberry32, oneOf, record, string, unicodeString } from "./gen.js";
import {
  STYLE_SOURCES, allElements, assertClassesStyled, assertNoInlineStyle, assertViewStructure, renderUnder,
  selectorClasses, usedClasses,
} from "./viewkit.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (f) => readFileSync(join(ROOT, f), "utf8");
const STYLESHEETS = ["tokens.css", "base.css", "layout.css", "components.css", "views.css"];
const LEGACY = ["top", "scroll", "warn", "mark", "busy", "ok", "error"];

function files(dir = ROOT) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return name === "tests" || name === "node_modules" ? [] : files(p);
    return [p];
  });
}

const flush = async (n = 8) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };

/** Legacy classes used in a tree. */
const legacyIn = (root) => [...usedClasses(root)].filter((c) => LEGACY.includes(c)).sort();

// ---------------------------------------------------------------- static

test("style.css is gone, and the five stylesheets are the style sources", () => {
  assert.equal(existsSync(join(ROOT, "style.css")), false);
  assert.deepEqual(STYLE_SOURCES, STYLESHEETS);
  const sheets = files().filter((f) => f.endsWith(".css")).map((f) => relative(ROOT, f)).sort();
  assert.deepEqual(sheets, [...STYLESHEETS].sort(), "no other stylesheet");
});

test("index.html links exactly the five stylesheets in order, and main is labelled by page-title", () => {
  const html = read("index.html");
  const sheets = [...html.matchAll(/<link\b[^>]*rel="stylesheet"[^>]*>/g)].map((m) => /href="([^"]+)"/.exec(m[0])[1]);
  assert.deepEqual(sheets, STYLESHEETS);
  const mains = html.match(/<main\b[^>]*>/g) ?? [];
  assert.equal(mains.length, 1);
  assert.match(mains[0], /\sid="main"/);
  assert.match(mains[0], /\saria-labelledby="page-title"/);
  assert.doesNotMatch(html, /\sstyle=/i, "no style attribute");
});

test("the legacy classes appear in no stylesheet, in index.html or in any module", () => {
  for (const f of STYLESHEETS) {
    const found = [...selectorClasses(read(f))].filter((c) => LEGACY.includes(c));
    assert.deepEqual(found, [], `${f} styles ${found.join(", ")}`);
  }
  const html = read("index.html");
  for (const m of html.matchAll(/\sclass="([^"]*)"/g)) {
    const found = m[1].split(/\s+/).filter((c) => LEGACY.includes(c));
    assert.deepEqual(found, [], `index.html class="${m[1]}"`);
  }
  // In the modules: h("tag.a.b") selectors, class: "…" attributes and classList calls.
  for (const f of files().filter((p) => p.endsWith(".js"))) {
    const src = readFileSync(f, "utf8");
    const problems = [];
    for (const m of src.matchAll(/\bh\(\s*["'`]([a-z][a-z0-9]*(?:[.#][\w-]+)*)["'`]/g)) {
      for (const c of m[1].split(".").slice(1).map((x) => x.split("#")[0])) if (LEGACY.includes(c)) problems.push(m[0]);
    }
    for (const m of src.matchAll(/\bclass(?:Name)?\s*:\s*["'`]([^"'`]*)["'`]/g)) {
      if (m[1].split(/\s+/).some((c) => LEGACY.includes(c))) problems.push(m[0]);
    }
    for (const m of src.matchAll(/classList\.(?:add|toggle|contains)\(\s*["'`]([\w-]+)["'`]/g)) {
      if (LEGACY.includes(m[1])) problems.push(m[0]);
    }
    assert.deepEqual(problems, [], relative(ROOT, f));
  }
});

test("the legacy-class scan catches a planted class", async () => {
  await renderUnder(({ document, main }) => {
    const p = document.createElement("p");
    p.setAttribute("class", "hint warn");
    main.append(p);
  }, { check: ({ document }) => assert.deepEqual(legacyIn(document.body), ["warn"]) });
});

// ---------------------------------------------------------------- generated worlds

const ADMIN = { groups: ["admin"], email: "admin@example.com", username: "admin" };
const VIEWER = { groups: ["admin-viewer"], email: "viewer@example.com", username: "viewer" };
const EDITOR = { groups: ["admin-editor"], email: "e@example.com", username: "e" };
const DEFAULTS = {
  after_ms: 700, tool_after_ms: 1200, handoff_wait_ms: 1500, end_mode: "wait", chars_per_s: 15,
  silence_timeout_ms: 6000, silence_max_repeats: 2, silence_action: "escalate", greeting_silence_timeout_ms: 8000,
};
const LANGS = oneOf(["en-US", "es-US"], ["en-US"], ["en-US", "es-US", "fr-FR"]);
const text = oneOf(null, "", unicodeString({ minLength: 1, maxLength: 8 }), "Your {account}", "{nope} x");
const delay = oneOf(null, record({ value: int(0, 10000), source: oneOf("tool", "default") }));

/** Everything the API can answer, consistent across routes: one generated console. */
const worldGen = gen((rng) => {
  const languages = LANGS(rng);
  const texts = (r) => Object.fromEntries(languages.filter(() => int(0, 4)(r) > 0).map((l) => [l, text(r)]));
  const names = array(string({ minLength: 1, maxLength: 8, chars: "abcdefgh_" }), { minLength: 1, maxLength: 3 })(rng);
  const tools = [...new Set(names)].map((name) => record({
    name,
    kind: oneOf("read", "write", null),
    min_level: oneOf(int(0, 3), null),
    confirm: oneOf("none", "deterministic", null),
    confirm_source: oneOf("tool", "default"),
    rev: int(1, 50),
    mask: oneOf(null, record({
      phrase_key: `mask.${name}`, after_ms: delay, handoff_wait_ms: delay,
      pool: array(gen((r) => ({ key: `mask.${name}${int(0, 1)(r) ? `.${int(1, 3)(r)}` : ""}`, owner_tool: name, rev: 1, texts: texts(r) })), { maxLength: 3 }),
    })),
    problems: array(unicodeString({ minLength: 1, maxLength: 8 }), { maxLength: 2 }),
    placeholders: array(record({ placeholder: oneOf("{account}", "{amount}", "{when}") }), { maxLength: 2 }),
  })(rng));
  const masking = {};
  for (const [name, spec] of Object.entries(MASKING_FIELDS)) {
    if (int(0, 1)(rng)) masking[name] = spec.type === "enum" ? oneOf(...spec.values)(rng) : int(spec.range[0], spec.range[1])(rng);
  }
  const ackPool = array(gen((r) => ({ key: `mask.ack${int(0, 1)(r) ? `.${int(1, 9)(r)}` : ""}`, owner_tool: null, rev: 1, texts: texts(r) })), { maxLength: 3 })(rng);
  const masks = [...ackPool, ...tools.flatMap((t) => (t.mask ? t.mask.pool : []))];
  const args = array(gen((r) => ({ key: `arg.${string({ minLength: 1, maxLength: 6, chars: "abc" })(r)}`, rev: 1, texts: texts(r) })), { maxLength: 3 })(rng);
  const value = json({ maxDepth: 2, maxLength: 3 });
  const item = oneOf(`policy#tool#${tools[0].name}`, "policy#masking", `prompt#${masks[0]?.key ?? "mask.x"}`)(rng);
  return {
    languages,
    tools,
    masking: {
      rev: int(1, 99)(rng), masking: oneOf(masking, null)(rng), effective: { ...DEFAULTS, ...masking }, defaults: DEFAULTS,
      ack_pool: ackPool, problems: array(unicodeString({ minLength: 1, maxLength: 8 }), { maxLength: 2 })(rng), takes_effect_s: 60,
    },
    masks: { languages, takes_effect_s: 60, prompts: masks },
    args: { languages, prompts: args },
    placeholders: {
      placeholders: array(gen((r) => {
        const name = oneOf("account", "when", "amount")(r);
        return {
          name, placeholder: `{${name}}`,
          values: array(record({ value: string({ minLength: 1, maxLength: 4 }), key: `arg.${name}.x`, words: gen((r2) => Object.fromEntries(languages.map((l) => [l, text(r2)]))) }), { maxLength: 2 })(r),
          missing: array(string({ minLength: 1, maxLength: 6 }), { maxLength: 2 })(r),
        };
      }), { maxLength: 2 })(rng),
    },
    item,
    entries: array(record({
      pk: item,
      sk: string({ minLength: 1, maxLength: 8 }),
      ts: oneOf("2026-10-01T10:00:00Z", unicodeString({ maxLength: 6 })),
      action: oneOf("update", "create", "delete", "restore"),
      restored_from: oneOf(undefined, "x"),
      actor: oneOf(null, { email: "a@example.com" }, { sub: "u-2" }, {}),
      target: oneOf(undefined, { kind: "tool", id: tools[0].name }, { kind: "masking", id: "m" }, { kind: "user", id: "u" }),
      security_change: oneOf(true, false, undefined),
      changes: oneOf(undefined, array(record({ field: string({ minLength: 1, maxLength: 8 }), old: value, new: value }), { maxLength: 3 })),
    }), { maxLength: 4 })(rng),
    slots: {
      acceptSpokenFactors: oneOf(true, false)(rng),
      takes_effect_s: 60,
      slots: [...new Set(array(oneOf("zip", "ssn4", "otp", "yes_no"), { maxLength: 4 })(rng))].map((name) => record({
        name, type: oneOf("digits", "yes_no"), sensitive: oneOf(true, false), keypad_only: oneOf(true, false),
        barge_in: oneOf(true, false), maxRetries: int(0, 3), is_factor: oneOf(true, false),
        effective_keypad_only: oneOf(true, false), flow: oneOf(null, "confirm"), rev: int(0, 9), loaded: true,
        problems: array(unicodeString({ minLength: 1, maxLength: 8 }), { maxLength: 1 }),
        warnings: oneOf([], [{ code: "verification_not_sensitive", slot: name, message: "m" }]),
      })(rng)),
    },
    redaction: gen((r) => {
      const types = [...new Set(array(oneOf("SSN", "PIN", "NAME", "EMAIL", "URL", "NEW_TYPE"), { maxLength: 4 })(r))];
      const effective = { enabled: types.length ? oneOf(true, false)(r) : false, entity_types: types, min_confidence: "0.5",
        mask: oneOf("[REDACTED:{type}]", unicodeString({ minLength: 1, maxLength: 6 }))(r), on_failure: oneOf("withhold", "allow")(r) };
      return {
        redaction: oneOf(effective, {}, { apply_to: ["x"] })(r), effective, defaults: effective, rev: int(0, 9)(r),
        problems: array(record({ field: "redaction.apply_to", message: unicodeString({ minLength: 1, maxLength: 8 }) }), { maxLength: 1 })(r),
        entity_types_available: ["EMAIL", "NAME", "PIN", "SSN", "URL"], on_failure_values: ["allow", "withhold"], takes_effect_s: 60,
        warnings: effective.enabled ? [] : [{ code: "redaction_off", message: "off" }],
      };
    })(rng),
    users: {
      users: array(record({
        username: string({ minLength: 1, maxLength: 8 }),
        email: oneOf(undefined, null, "x@example.com", ADMIN.email),
        group: oneOf(...GROUPS, "", undefined),
        status: oneOf("CONFIRMED", "FORCE_CHANGE_PASSWORD", undefined, ""),
        enabled: oneOf(true, false),
        created: oneOf(undefined, "2026-01-01T00:00:00Z"),
      }), { maxLength: 4 })(rng),
      groups: oneOf(undefined, GROUPS)(rng),
    },
  };
});

/** A fake lib/api.js client answering from `world`; `fail` (an HTTP status) fails every call. */
function worldApi(world, fail) {
  const answer = (method, path, arg) => {
    if (fail !== null) throw new ApiError(mapError(fail, {}));
    if (method !== "get") return { changed: true, takes_effect_s: 60 };
    if (path === "tools") return { tools: world.tools, takes_effect_s: 60 };
    if (path.startsWith("tools/")) {
      const tool = world.tools.find((t) => t.name === path.slice(6));
      if (!tool) throw new ApiError(mapError(404, {}));
      return { tool, takes_effect_s: 60 };
    }
    if (path === "masking") return world.masking;
    if (path === "prompts") {
      const all = arg?.prefix?.startsWith("arg.") ? world.args : world.masks;
      return { ...all, prompts: all.prompts.filter((p) => p.key.startsWith(arg?.prefix ?? "")) };
    }
    if (path.startsWith("placeholders/")) return world.placeholders;
    if (path === "audit") return { entries: world.entries };
    if (path === "users") return world.users;
    if (path === "slots") return world.slots;
    if (path === "redaction") return world.redaction;
    throw new ApiError(mapError(404, {}));
  };
  const verb = (method) => async (segments, arg) => structuredClone(answer(method, segments.join("/"), arg));
  return { get: verb("get"), put: verb("put"), post: verb("post"), patch: verb("patch"), del: verb("del") };
}

/** The routes a world can show, every section's included (and an unknown one). */
const routesOf = (world) => [
  "", "#/nope", "#/tools", `#/tools/${world.tools[0].name}`, "#/tools/missing_tool", "#/masking", "#/prompts",
  `#/prompts?tool=${world.tools[0].name}`, "#/audit", `#/audit?item=${encodeURIComponent(world.item)}`, "#/users",
  ...SECTIONS.map((s) => `#/${s.id}`),
];

/** A view whose first load failed: load() (lib/ui.js) leaves its page header and the error panel. */
const failedFirstLoad = (main) => main.querySelector(".panel--error") !== null && main.querySelectorAll(".skeleton").length === 0
  && main.children.filter((c) => !c.classList.contains("page-header-sentinel")).length === 2;

/** The checks on a rendered console: the style and class checks on the whole page, the structure check on main. */
function checkConsole(document, main) {
  assertNoInlineStyle(document.body);
  assertClassesStyled(document.body);
  assert.deepEqual(legacyIn(document.body), [], "no legacy class");
  assertViewStructure(main);
}

async function renderRoute({ world, user, hash, fail, open }) {
  await renderUnder(async ({ document, main }) => {
    const nav = document.createElement("nav");
    nav.setAttribute("id", "nav");
    nav.setAttribute("aria-label", "Sections");
    document.body.prepend(nav);
    await render({ api: worldApi(world, fail), user }, { main, nav, hash });
    await flush();
    // Open an editor or a history panel where the view has one (Edit, View, History).
    const opens = main.querySelectorAll("button").filter((b) => /^(Edit|View|History)\b/.test(b.textContent.trim()));
    if (opens.length && open !== null) {
      opens[open % opens.length].click();
      await flush();
    }
  }, {
    check: ({ document, main }) => {
      // With no failure injected, only the unknown tool fails its load. The structure check runs on every route.
      if (fail === null && hash !== "#/tools/missing_tool") assert.equal(failedFirstLoad(main), false, `${hash} loaded`);
      checkConsole(document, main);
    },
  });
}

test("every SECTIONS route for each kind of user, with a fixed world and with API failures", async () => {
  const world = worldGen(mulberry32(22));
  for (const user of [ADMIN, EDITOR, VIEWER]) {
    for (const hash of routesOf(world)) {
      for (const fail of [null, 500, 503, 403]) await renderRoute({ world, user, hash, fail, open: 0 });
    }
  }
});

test("PBT (whole console): every route rendered through SECTIONS with generated payloads uses only styled classes", async () => {
  await forAll(record({
    world: worldGen,
    user: oneOf(ADMIN, EDITOR, VIEWER),
    pick: int(0, 100),
    fail: oneOf(null, null, null, 400, 404, 409, 429, 500, 503),
    open: oneOf(null, int(0, 5)),
  }), async ({ world, user, pick, fail, open }) => {
    const routes = routesOf(world);
    await renderRoute({ world, user, hash: routes[pick % routes.length], fail, open });
  }, { runs: 200 });
});

test("after a failed first load: each view keeps its page header, so aria-labelledby=\"page-title\" resolves", async () => {
  const world = worldGen(mulberry32(22));
  const views = [
    ["#/tools", "Tools"], [`#/tools/${world.tools[0].name}`, world.tools[0].name], ["#/tools/missing_tool", "missing_tool"],
    ["#/masking", "Masking"], ["#/prompts", "Prompts"], ["#/slots", "Slots"], ["#/redaction", "Redaction"], ["#/users", "Users"],
  ];
  for (const [hash, title] of views) {
    for (const fail of [500, 403]) {
      if (hash === "#/tools/missing_tool" && fail !== 500) continue;
      await renderUnder(async ({ document, main }) => {
        const nav = document.createElement("nav");
        document.body.prepend(nav);
        await render({ api: worldApi(world, hash === "#/tools/missing_tool" ? null : fail), user: ADMIN }, { main, nav, hash });
        await flush();
      }, {
        check: ({ main }) => {
          assert.ok(failedFirstLoad(main), `${hash} failed its first load`);
          assertViewStructure(main);
          assert.equal(main.ownerDocument.getElementById("page-title").textContent, title, hash);
        },
      });
    }
  }
});

// ---------------------------------------------------------------- bare pages

/** The shell's other containers, as in index.html (renderUnder supplies main and the rest). */
function shellParts(document) {
  const el = (tag, attrs) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  };
  const sidebar = el("div", { id: "sidebar", class: "sidebar" });
  sidebar.append(el("span", { id: "stage", class: "stage" }), el("nav", { id: "nav", "aria-label": "Sections" }), el("div", { id: "account", class: "account" }));
  const topbar = el("header", { class: "topbar" });
  topbar.append(el("button", { id: "menu", type: "button", class: "quiet", hidden: "" }), el("span", { id: "topbar-stage", class: "stage" }));
  document.body.prepend(sidebar, topbar, el("div", { id: "scrim", class: "scrim", hidden: "" }));
  document.body.setAttribute("data-shell", "loading");
}

/** A fake lib/auth.js for one load. */
function fakeAuth({ complete, signedIn = false, user = ADMIN }) {
  return {
    configure() {},
    completeSignIn: async () => complete(),
    isSignedIn: () => signedIn,
    signIn: async () => {},
    signOut: async () => {},
    user: () => user,
    idToken: () => "t",
    refresh: async () => false,
  };
}

const CFG = { stage: "dev", apiBase: "/api", redirectUri: "https://example.com/" };

const bareGen = oneOf(
  record({ kind: "not-configured" }),
  record({ kind: "error", code: oneOf("access_denied", "state_mismatch", "token_endpoint_500", "storage_unavailable", "TypeError", "", unicodeString({ maxLength: 6 })), description: oneOf(undefined, unicodeString({ maxLength: 10 }), "<b>x</b>") }),
  record({ kind: "loop" }),
  record({ kind: "no-access" }),
  record({ kind: "session-ended" }),
);

async function renderBare(page) {
  await renderUnder(async ({ document, sessionStorage }) => {
    shellParts(document);
    let onAuthLost = null;
    const env = {
      location: { hash: "", search: "", pathname: "/", assign() {} },
      storage: sessionStorage,
      now: () => 1_000_000,
      loadConfig: async () => CFG,
      win: { addEventListener() {}, removeEventListener() {} },
      createApiClient: (opts) => { onAuthLost = opts.onAuthLost; return {}; },
    };
    let expected;
    if (page.kind === "not-configured") {
      env.loadConfig = async () => { throw new Error("no config.js"); };
      env.auth = fakeAuth({ complete: () => null });
      expected = "not-configured";
    } else if (page.kind === "error") {
      env.auth = fakeAuth({ complete: () => { throw Object.assign(new Error(page.description ?? ""), { code: page.code, description: page.description }); } });
      expected = "error";
    } else if (page.kind === "loop") {
      sessionStorage.setItem("owlbank-admin-signin-guard", JSON.stringify({ count: 2, first: 1_000_000 }));
      env.auth = fakeAuth({ complete: () => null });
      expected = "loop";
    } else if (page.kind === "no-access") {
      env.auth = fakeAuth({ complete: () => "signed-in", signedIn: true, user: { ...ADMIN, groups: [] } });
      expected = "no-access";
    } else {
      env.auth = fakeAuth({ complete: () => "signed-in", signedIn: true });
      env.sections = [{ id: "tools", label: "Tools", icon: "tools", render: () => onAuthLost() }];
      expected = "app";
    }
    const out = await start(env);
    assert.equal(out, page.kind === "error" && page.code === "loop" ? "loop" : expected);
    assert.equal(document.body.getAttribute("data-shell"), "bare", page.kind);
  }, {
    check: ({ document, main }) => {
      assertNoInlineStyle(document.body);
      assertClassesStyled(document.body);
      assert.deepEqual(legacyIn(document.body), [], "no legacy class");
      assert.equal(main.getAttribute("aria-labelledby"), "page-title");
      const h1s = allElements(main).filter((e) => e.tagName === "H1");
      assert.equal(h1s.length, 1);
      assert.equal(h1s[0].id, "page-title");
    },
  });
}

test("each bare page uses only styled classes and is labelled by its h1#page-title", async () => {
  for (const kind of ["not-configured", "loop", "no-access", "session-ended"]) await renderBare({ kind });
  await renderBare({ kind: "error", code: "access_denied", description: "nope" });
});

test("PBT (bare pages): generated sign-in errors and the other bare pages use only styled classes", async () => {
  await forAll(bareGen, (page) => renderBare(page), { runs: 100 });
});
