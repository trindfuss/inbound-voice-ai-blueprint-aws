// The page shell. Static part: landmarks and the skip
// link exist before any script runs, the stylesheets load in order, and body[data-shell]
// switches between loading, app and bare (no PBT: static markup and a three-value switch).
// Runtime part: the nav from SECTIONS (a property test), the drawer menu (examples: a small fixed
// state machine), the theme toggle, the stage badge and the account block.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { render, SECTIONS } from "../app.js";
import { h } from "../lib/dom.js";
import { parseRoute, VIEWS } from "../lib/route.js";
import { initMenu, renderAccount, renderNav, renderStage, SHELL_STATES, setShell, themeToggle, visibleSections } from "../lib/shell.js";
import { readTheme, THEME_KEY } from "../lib/theme.js";
import { Event, install, uninstall } from "./fakedom.js";
import { array, forAll, gen, int, oneOf, record, string } from "./gen.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(ROOT, "index.html"), "utf8");
const body = html.slice(html.search(/<body\b/i));

/** The opening tag of the element with this id, or undefined. */
const tagWithId = (id) => html.match(new RegExp(`<[a-z]+\\b[^>]*\\sid="${id}"[^>]*>`, "i"))?.[0];

test("index.html: the title, the font preload and the owl favicon", () => {
  assert.match(html, /<title>OwlBank Console<\/title>/);
  const preload = html.match(/<link\b[^>]*rel="preload"[^>]*>/i)?.[0] ?? "";
  assert.match(preload, /\shref="fonts\/Inter-4\.1-latin-var\.woff2"/);
  assert.match(preload, /\sas="font"/);
  assert.match(preload, /\stype="font\/woff2"/);
  assert.match(preload, /\scrossorigin(\s|>|=)/, "fonts are fetched in CORS mode, so the preload needs crossorigin");
  assert.match(html, /<link rel="icon" href="owl\.svg" type="image\/svg\+xml">/);
  assert.doesNotMatch(html, /href="data:/);
});

test("index.html: exactly the five stylesheets, in order", () => {
  const sheets = [...html.matchAll(/<link rel="stylesheet" href="([^"]+)">/g)].map((m) => m[1]);
  assert.deepEqual(sheets, ["tokens.css", "base.css", "layout.css", "components.css", "views.css"]);
});

test("index.html: the skip link is first in <body> and points at #main", () => {
  assert.match(body, /^<body\b[^>]*>\s*<a class="skip" href="#main">Skip to content<\/a>/);
});

test("index.html: one labelled nav, one main#main, and every container", () => {
  const navs = [...html.matchAll(/<nav\b[^>]*>/gi)].map((m) => m[0]);
  assert.equal(navs.length, 1, "one nav");
  assert.match(navs[0], /\saria-label="[^"]+"/, "the nav is labelled");
  assert.match(navs[0], /\sid="nav"/);
  const mains = [...html.matchAll(/<main\b[^>]*>/gi)].map((m) => m[0]);
  assert.equal(mains.length, 1, "one main");
  assert.match(mains[0], /\sid="main"/);
  assert.match(mains[0], /\stabindex="-1"/, "main takes focus after a route change");
  for (const id of ["sidebar", "stage", "nav", "account", "menu", "main", "scrim", "toasts", "confirm", "live"]) {
    assert.ok(tagWithId(id), `#${id} is in the shell`);
    assert.equal(html.match(new RegExp(`\\sid="${id}"`, "g")).length, 1, `#${id} once`);
  }
  assert.doesNotMatch(html, /\sid="who"/, "#who is now #account");
  assert.match(tagWithId("sidebar"), /class="sidebar"/);
  assert.match(tagWithId("confirm"), /^<dialog\b/);
  assert.match(tagWithId("live"), /role="status"/);
  assert.match(tagWithId("live"), /aria-live="polite"/);
  assert.match(tagWithId("scrim"), /\shidden(\s|>)/);
  assert.match(html, /<header class="topbar">/);
});

test("index.html: #menu is a hidden button for the sidebar until the shell wires it", () => {
  const menu = tagWithId("menu");
  assert.match(menu, /^<button\b/);
  assert.match(menu, /\stype="button"/);
  assert.match(menu, /\saria-expanded="false"/);
  assert.match(menu, /\saria-controls="sidebar"/);
  assert.match(menu, /\shidden(\s|>)/);
});

test("index.html: the sidebar mark is decorative next to the product name", () => {
  const sidebar = html.slice(html.indexOf('id="sidebar"'), html.indexOf('<nav'));
  assert.match(sidebar, /<img\b[^>]*\ssrc="owl\.svg"[^>]*\salt=""/);
  assert.match(sidebar, /OwlBank Console/);
});

test("index.html: no style attribute; the body opens in the loading state with a skeleton", () => {
  assert.doesNotMatch(html, /\sstyle=/i);
  assert.match(body, /^<body data-shell="loading">/);
  const main = html.slice(html.search(/<main\b/), html.search(/<\/main>/));
  assert.match(main, /class="skeleton skeleton--page" aria-hidden="true"/);
  assert.match(main, /class="visually-hidden">Loading…</);
});

test("app.js fills #account (not #who) and switches the shell", () => {
  const app = readFileSync(join(ROOT, "app.js"), "utf8");
  assert.match(app, /getElementById\("account"\)/);
  assert.doesNotMatch(app, /getElementById\("who"\)/);
  assert.match(app, /setShell\("app"\)/);
  assert.match(app, /setShell\("bare"\)/);
});

function withDom(fn) {
  install();
  let result;
  try {
    result = fn();
  } catch (e) {
    uninstall();
    throw e;
  }
  if (result && typeof result.then === "function") return result.finally(uninstall);
  uninstall();
  return result;
}

test("setShell sets body[data-shell] for each state", () => withDom(() => {
  assert.deepEqual([...SHELL_STATES], ["loading", "app", "bare"]);
  for (const s of ["app", "bare", "loading", "app"]) {
    setShell(s);
    assert.equal(document.body.getAttribute("data-shell"), s);
  }
}));

test("setShell refuses anything that isn't a state, and leaves the shell as it was", () => withDom(() => {
  setShell("bare");
  for (const bad of ["", "App", "hidden", undefined, null, 1]) {
    assert.throws(() => setShell(bad), /not a shell state/);
    assert.equal(document.body.getAttribute("data-shell"), "bare");
  }
}));

// ---------------------------------------------------------------- SECTIONS and the nav

const SYMBOLS = [...readFileSync(join(ROOT, "icons.svg"), "utf8").matchAll(/<symbol\b[^>]*\sid="([^"]+)"/g)].map((m) => m[1]);
const GROUPS = ["admin", "admin-editor", "viewer"];
const userOf = (groups) => ({ email: "someone@example.test", username: "someone", groups });
const ADMIN = userOf(["admin"]);
const VIEWER = userOf(["viewer"]);

/** The links renderNav produced, as plain data. */
function navLinks(nav) {
  return nav.querySelectorAll("a.nav-link").map((a) => ({
    a,
    href: a.getAttribute("href"),
    current: a.getAttribute("aria-current"),
    icon: a.querySelector("svg.icon > use")?.getAttribute("href") ?? null,
    label: a.querySelector(".nav-label")?.textContent ?? "",
  }));
}

/** Check the nav rule for one (sections, user, current) case. */
function checkNav(sections, user, currentId) {
  const nav = h("nav");
  renderNav(nav, sections, currentId, user);
  const visible = sections.filter((s) => !s.when || s.when(user));
  assert.deepEqual(visibleSections(sections, user), visible);
  const links = navLinks(nav);
  assert.equal(links.length, visible.length, "one link per visible section");
  assert.equal(nav.querySelectorAll("ul > li > a.nav-link").length, visible.length, "ul > li > a.nav-link");
  links.forEach((l, i) => {
    assert.equal(l.href, `#/${visible[i].id}`);
    assert.equal(l.icon, `icons.svg#${visible[i].icon}`);
    assert.equal(l.label, visible[i].label);
    assert.ok(l.label.trim() !== "", "a non-empty label");
  });
  const current = links.filter((l) => l.current !== null);
  const matches = visible.some((s) => s.id === currentId);
  assert.equal(current.length, matches ? 1 : 0, "exactly one current link when the route matches, else none");
  if (matches) {
    assert.equal(current[0].current, "page");
    assert.equal(current[0].href, `#/${currentId}`);
    assert.ok(current[0].a.classList.contains("is-current"));
  }
  const ids = sections.map((s) => s.id);
  for (const id of ids) {
    assert.equal(parseRoute(`#/${id}`, ids).view, id, `parseRoute accepts ${id}`);
    assert.equal(parseRoute(`#/${id}/x?y=1`, ids).arg, "x");
  }
}

const kebabId = string({ minLength: 1, maxLength: 10, chars: "abcdefghijklmnopqrstuvwxyz0123456789-" });
const sectionsGen = gen((rng) => {
  const raw = array(record({
    id: kebabId,
    label: gen((r) => `${string({ minLength: 1, maxLength: 4 })(r)} ${string({ maxLength: 8, chars: "abc XYZ 019" })(r)}`.trim()),
    icon: oneOf(...SYMBOLS),
    when: oneOf(undefined, () => true, () => false, (u) => u.groups.includes("admin")),
  }), { minLength: 0, maxLength: 6 })(rng);
  const seen = new Set();
  return raw.filter((s) => !seen.has(s.id) && seen.add(s.id)).map((s) => ({ ...s, render: () => {} }));
});
const userGen = gen((rng) => userOf(GROUPS.filter(() => int(0, 1)(rng) === 1)));
const caseGen = gen((rng) => {
  const sections = sectionsGen(rng);
  const ids = sections.map((s) => s.id);
  const currentId = ids.length && int(0, 3)(rng) > 0 ? ids[int(0, ids.length - 1)(rng)] : `zz-${kebabId(rng)}`;
  return { sections, user: userGen(rng), currentId };
});

test("SECTIONS is the registry: ids, labels, sprite icons, one render signature, Users for admins", () => {
  assert.deepEqual(SECTIONS.map((s) => s.id), ["tools", "masking", "prompts", "slots", "redaction", "audit", "users"]);
  assert.deepEqual(SECTIONS.map((s) => s.label), ["Tools", "Masking", "Prompts", "Slots", "Redaction", "Audit log", "Users"]);
  assert.deepEqual(SECTIONS.map((s) => s.icon), ["tools", "masking", "prompts", "slots", "redaction", "history", "users"]);
  for (const s of SECTIONS) {
    assert.ok(SYMBOLS.includes(s.icon), `${s.id}: icon in the sprite`);
    assert.equal(typeof s.render, "function");
  }
  assert.deepEqual(SECTIONS.filter((s) => s.when).map((s) => s.id), ["redaction", "users"]);
  assert.deepEqual(VIEWS, SECTIONS.map((s) => s.id), "route.js's default ids match the registry");
});

test("PBT: renderNav over the real SECTIONS for generated users and routes", () => withDom(() => {
  forAll(gen((rng) => ({ user: userGen(rng), currentId: oneOf(...SECTIONS.map((s) => s.id), "nope", "")(rng) })),
    ({ user, currentId }) => {
      checkNav(SECTIONS, user, currentId);
      const nav = h("nav");
      renderNav(nav, SECTIONS, currentId, user);
      const hasUsers = navLinks(nav).some((l) => l.href === "#/users");
      assert.equal(hasUsers, user.groups.includes("admin"), "Users only for admins");
    });
}));

test("PBT: renderNav over generated section lists", () => withDom(() => {
  forAll(caseGen, ({ sections, user, currentId }) => checkNav(sections, user, currentId));
}));

test("renderNav replaces the previous links, and a hidden current section marks nothing", () => withDom(() => {
  const nav = h("nav");
  renderNav(nav, SECTIONS, "tools", ADMIN);
  renderNav(nav, SECTIONS, "users", VIEWER);
  const links = navLinks(nav);
  assert.equal(links.length, 5);
  assert.equal(links.filter((l) => l.current).length, 0);
}));

test("parseRoute(hash, ids): the ids decide the views; unknown routes fall back to the first id", () => {
  assert.equal(parseRoute("#/slots", ["slots", "tools"]).view, "slots");
  assert.equal(parseRoute("#/tools", ["slots"]).view, "slots");
  assert.equal(parseRoute("#/tools", ["slots"]).arg, null);
  assert.equal(parseRoute("#/users").view, "users", "the default ids are VIEWS");
  assert.equal(parseRoute("#/nope").view, "tools");
});

// ---------------------------------------------------------------- render through the registry

test("render: the title is '<label> · OwlBank Console'; a section hidden by when() shows the empty state", () => withDom(async () => {
  const main = h("main#main", { tabindex: "-1" });
  const nav = h("nav#nav");
  document.body.append(nav, main);
  const calls = [];
  const ctx = { user: VIEWER, api: { get: async (p) => { calls.push(p); return {}; } } };
  await render(ctx, { main, nav, hash: "#/users" });
  assert.equal(document.title, "Users · OwlBank Console");
  assert.equal(main.querySelector("header.page-header h1#page-title").textContent, "Users");
  assert.equal(main.querySelector(".empty-state .empty-state-title").textContent, "Only admins manage users.");
  assert.deepEqual(calls, [], "no API call for a section the user can't see");
  assert.equal(document.activeElement, main, "main takes focus after the route change");
  assert.equal(navLinks(nav).length, 5);
}));

test("render dispatches to the section's render with {arg, params}", () => withDom(async () => {
  const main = h("main#main", { tabindex: "-1" });
  const nav = h("nav#nav");
  document.body.append(nav, main);
  const seen = [];
  const sections = [
    { id: "alpha", label: "Alpha", icon: "tools", render: async (m, ctx, route) => { seen.push(route); m.append(h("p", {}, "alpha")); } },
    { id: "beta", label: "Beta", icon: "users", render: async () => {} },
  ];
  await render({ user: VIEWER }, { main, nav, hash: "#/alpha/one%20two?x=1", sections });
  assert.equal(document.title, "Alpha · OwlBank Console");
  assert.equal(seen.length, 1);
  assert.equal(seen[0].arg, "one two");
  assert.equal(seen[0].params.get("x"), "1");
  assert.equal(navLinks(nav).find((l) => l.current).href, "#/alpha");
}));

test("app.js doesn't touch document at import time, and starts only in a browser", () => {
  const app = readFileSync(join(ROOT, "app.js"), "utf8");
  assert.match(app, /if \(typeof document !== "undefined"\) start\(\);/);
  const topLevel = app.split("\n").filter((l) => /^(const|let|var) /.test(l));
  for (const l of topLevel) assert.doesNotMatch(l, /document\.|window\.|location\./, l);
});

// ---------------------------------------------------------------- the drawer menu

const keydown = (key) => Object.assign(new Event("keydown", { bubbles: true, cancelable: true }), { key });

function fakeMql(matches = false) {
  const listeners = [];
  return {
    matches,
    addEventListener: (type, fn) => { if (type === "change") listeners.push(fn); },
    removeEventListener: () => {},
    cross(to) { this.matches = to; for (const fn of listeners) fn({ matches: to }); },
  };
}

function menuShell({ mql = fakeMql(false), current = "masking" } = {}) {
  const button = h("button#menu", { type: "button", "aria-expanded": "false", "aria-controls": "sidebar", hidden: true }, "Menu");
  const topbar = h("header.topbar", {}, button);
  const nav = h("nav#nav");
  const sidebar = h("div.sidebar#sidebar", {}, nav);
  const main = h("main#main", { tabindex: "-1" });
  const scrim = h("div.scrim#scrim", { hidden: true });
  document.body.append(h("div.app", {}, sidebar, h("div.content", {}, topbar, main)), scrim);
  renderNav(nav, SECTIONS, current, ADMIN);
  const menu = initMenu({ button, sidebar, scrim, inertTargets: [topbar, main], mql });
  return { menu, button, topbar, sidebar, main, scrim, nav, mql };
}

function assertOpen(s) {
  assert.equal(s.menu.isOpen(), true);
  assert.ok(s.sidebar.classList.contains("is-open"));
  assert.equal(s.button.getAttribute("aria-expanded"), "true");
  assert.equal(s.scrim.hidden, false);
  assert.equal(s.topbar.inert, true, "the top bar is inert");
  assert.equal(s.main.inert, true, "main is inert");
  assert.equal(s.sidebar.inert, false);
}

function assertClosed(s, { focusOnButton = true } = {}) {
  assert.equal(s.menu.isOpen(), false);
  assert.ok(!s.sidebar.classList.contains("is-open"));
  assert.equal(s.button.getAttribute("aria-expanded"), "false");
  assert.equal(s.scrim.hidden, true);
  assert.equal(s.topbar.inert, false);
  assert.equal(s.main.inert, false);
  if (focusOnButton) assert.equal(document.activeElement, s.button, "focus returns to the menu button");
}

test("initMenu shows the button; clicking it opens the drawer, makes the rest inert and focuses the current link", () => withDom(() => {
  const s = menuShell();
  assert.equal(s.button.hidden, false, "#menu is un-hidden from script");
  assertClosed(s, { focusOnButton: false });
  s.button.focus();
  s.button.click();
  assertOpen(s);
  assert.equal(document.activeElement, s.nav.querySelector('a.nav-link[aria-current="page"]'));
  assert.equal(document.activeElement.getAttribute("href"), "#/masking");
  s.button.click();
  assertClosed(s);
}));

test("with no current link, opening focuses the first nav link", () => withDom(() => {
  const s = menuShell({ current: "nope" });
  s.menu.open();
  assert.equal(document.activeElement, s.nav.querySelector("a.nav-link"));
}));

test("Esc closes the drawer and returns focus to the button; other keys and a closed drawer do nothing", () => withDom(() => {
  const s = menuShell();
  s.button.click();
  document.activeElement.dispatchEvent(keydown("Enter"));
  assertOpen(s);
  document.activeElement.dispatchEvent(keydown("Escape"));
  assertClosed(s);
  const other = h("button", { type: "button" }, "other");
  document.body.append(other);
  other.focus();
  other.dispatchEvent(keydown("Escape"));
  assert.equal(document.activeElement, other, "Esc with the drawer closed leaves focus alone");
}));

test("a scrim click closes the drawer and returns focus to the button", () => withDom(() => {
  const s = menuShell();
  s.button.click();
  s.scrim.click();
  assertClosed(s);
}));

test("a nav-link click closes the drawer and returns focus to the button", () => withDom(() => {
  const s = menuShell();
  s.button.click();
  s.nav.querySelectorAll("a.nav-link")[2].querySelector(".nav-label").click();
  assertClosed(s);
}));

test("crossing 960 px closes the drawer (a fake matchMedia), in either direction", () => withDom(() => {
  const s = menuShell();
  s.button.click();
  s.mql.cross(true);
  assertClosed(s, { focusOnButton: false });
  assert.ok(s.sidebar.contains(document.activeElement), "at desktop width focus stays in the (now persistent) sidebar");
  s.mql.cross(false);
  assertClosed(s, { focusOnButton: false });
  s.button.click();
  assertOpen(s);
  s.mql.cross(false);
  assertClosed(s);
}));

test("at 960 px and wider open() does nothing (the sidebar is not a drawer there)", () => withDom(() => {
  const s = menuShell({ mql: fakeMql(true) });
  s.menu.open();
  assertClosed(s, { focusOnButton: false });
}));

test("initMenu works without matchMedia", () => withDom(() => {
  const s = menuShell({ mql: null });
  s.menu.open();
  assertOpen(s);
  s.menu.close();
  assertClosed(s);
}));

// ---------------------------------------------------------------- theme toggle

test("themeToggle: the label and icon name the mode a click switches to, and the theme persists", () => {
  const { localStorage } = install();
  try {
    const root = document.documentElement;
    root.setAttribute("data-theme", "light");
    const btn = themeToggle(root, localStorage);
    assert.equal(btn.localName, "button");
    assert.equal(btn.getAttribute("type"), "button");
    assert.equal(btn.textContent, "Dark mode");
    assert.equal(btn.getAttribute("aria-pressed"), null);
    assert.equal(btn.querySelector("svg.icon > use").getAttribute("href"), "icons.svg#moon");
    btn.click();
    assert.equal(root.getAttribute("data-theme"), "dark");
    assert.equal(btn.textContent, "Light mode");
    assert.equal(btn.querySelectorAll("svg.icon").length, 1, "one icon after the swap");
    assert.equal(btn.querySelector("svg.icon > use").getAttribute("href"), "icons.svg#sun");
    assert.equal(readTheme(localStorage), "dark");
    assert.equal(localStorage.getItem(THEME_KEY), "dark");
    btn.click();
    assert.equal(root.getAttribute("data-theme"), "light");
    assert.equal(btn.textContent, "Dark mode");
    assert.equal(btn.querySelector("svg.icon > use").getAttribute("href"), "icons.svg#moon");
    assert.equal(readTheme(localStorage), "light");
  } finally { uninstall(); }
});

test("themeToggle starts as Light mode in dark, defaults to the page root and theme storage, and survives blocked storage", () => {
  install({ storageThrows: "localStorage" });
  try {
    document.documentElement.setAttribute("data-theme", "dark");
    const btn = themeToggle();
    assert.equal(btn.textContent, "Light mode");
    btn.click();
    assert.equal(document.documentElement.getAttribute("data-theme"), "light");
    assert.equal(btn.textContent, "Dark mode");
  } finally { uninstall(); }
  const { localStorage } = install();
  try {
    themeToggle().click();
    assert.equal(localStorage.getItem(THEME_KEY), "dark", "the default storage is the theme's");
  } finally { uninstall(); }
});

// ---------------------------------------------------------------- stage badge and account

test("renderStage puts the env badge into #stage and the top-bar copy", () => withDom(() => {
  const stage = h("span#stage.stage");
  const top = h("span#topbar-stage.stage");
  document.body.append(stage, top);
  for (const [s, cls, text] of [["dev", "env-test", "dev"], ["test", "env-test", "test"], ["prod", "env-prod", "prod"],
    [undefined, "badge--neutral", "unknown stage"]]) {
    renderStage(s);
    for (const el of [stage, top]) {
      assert.equal(el.querySelectorAll(".badge").length, 1, "one badge, replacing the last");
      assert.ok(el.querySelector(".badge").classList.contains(cls), `${s}: ${cls}`);
      assert.equal(el.textContent, text);
    }
  }
  const other = h("span");
  renderStage("prod", [other]);
  assert.equal(other.textContent, "prod");
  assert.equal(stage.textContent, "unknown stage", "explicit targets only");
}));

test("renderStage tolerates a missing top-bar copy", () => withDom(() => {
  document.body.append(h("span#stage"));
  renderStage("dev");
  assert.equal(document.getElementById("stage").textContent, "dev");
}));

test("renderAccount: the email, one badge per group, the theme toggle and Sign out", () => withDom(() => {
  const el = h("div#account");
  document.body.append(el);
  let out = 0;
  renderAccount(el, userOf(["admin", "viewer"]), { onSignOut: () => { out++; } });
  assert.equal(el.querySelector(".account-user").textContent, "someone@example.test");
  assert.deepEqual(el.querySelectorAll(".badge").map((b) => b.textContent), ["admin", "viewer"]);
  const toggle = el.querySelector("button.theme-toggle");
  assert.ok(toggle, "the theme toggle");
  const signOut = el.querySelectorAll("button").find((b) => /Sign out/.test(b.textContent));
  assert.ok(signOut.querySelector('use[href="icons.svg#sign-out"]'));
  signOut.click();
  assert.equal(out, 1);

  renderAccount(el, { username: "u1", groups: [] }, { onSignOut: () => {} });
  assert.equal(el.querySelector(".account-user").textContent, "u1", "the username when there is no email");
  assert.deepEqual(el.querySelectorAll(".badge").map((b) => b.textContent), ["no group"]);
  assert.equal(el.querySelectorAll("button.theme-toggle").length, 1, "re-rendering replaces the block");
}));

test("no style attribute anywhere in what the shell builds", () => withDom(() => {
  const s = menuShell();
  const acc = h("div#account");
  const stage = h("span#stage");
  s.sidebar.append(acc, stage);
  renderAccount(acc, ADMIN, { onSignOut: () => {} });
  renderStage("dev");
  s.menu.open();
  const all = [document.body, ...document.body.querySelectorAll("*")];
  for (const el of all) {
    for (const a of el.attributes) {
      assert.notEqual(a.name, "style");
      assert.doesNotMatch(a.name, /^on/);
    }
  }
}));
