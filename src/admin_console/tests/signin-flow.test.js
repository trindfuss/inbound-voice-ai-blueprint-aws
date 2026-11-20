// app.js start(): the sign-in redirect, the loop guard, the bare pages and
// "Sign in again". Each page load runs under a fresh fakedom and a fresh lib/auth.js module
// (tokens live in its memory, so a reload forgets them, as in a browser); sessionStorage and
// localStorage carry over between loads. location, history and fetch are fakes; crypto is Node's.
// PBT: callback errors never redirect by themselves, and (flow part) the guard bounds
// automatic redirects. Runtime part: no token in web storage after a sign-in and a save.
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import { start } from "../app.js";
import { h } from "../lib/dom.js";
import { GUARD_KEY, GUARD_MAX, GUARD_WINDOW_MS, signInReason } from "../lib/signin.js";
import { save } from "../lib/ui.js";
import { createStorage, install, uninstall } from "./fakedom.js";
import { ALNUM, array, forAll, int, oneOf, record, string, unicodeString } from "./gen.js";
import { assertClassesStyled, assertNoInlineStyle } from "./viewkit.js";

const cfg = {
  apiBase: "https://api.example.test/dev",
  clientId: "client123",
  loginDomain: "owlbank-dev-admin-111111111111.auth.us-east-1.amazoncognito.com",
  redirectUri: "https://d111.cloudfront.net/",
  stage: "dev",
};
const LOGIN = `https://${cfg.loginDomain}`;
const AUTHORIZE = `${LOGIN}/oauth2/authorize`;
const TOKEN = `${LOGIN}/oauth2/token`;
const REVOKE = `${LOGIN}/oauth2/revoke`;
const PENDING = "owlbank-admin-pkce";
const THEME_KEY = "owlbank-admin-theme";
const JWT_RE = /eyJ[\w-]+\./;

const b64url = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
const jwt = (claims) => `${b64url({ alg: "RS256" })}.${b64url(claims)}.sig`;
const idToken = (groups) => jwt({ sub: "u1", email: "a@example.com", exp: 4_000_000_000, "cognito:groups": groups });
const ACCESS_TOKEN = jwt({ sub: "u1", token_use: "access", exp: 4_000_000_000 });
const REFRESH_TOKEN = "refresh-opaque-token-value";

function reply(status, body) {
  const text = body === undefined ? "" : JSON.stringify(body);
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      if (body === undefined) throw new SyntaxError("no body");
      return body;
    },
    text: async () => text,
  };
}

// ---------------------------------------------------------------- the fake browser

let browser;
let saved;
let loadNo = 0;

/** A tab: storage that survives reloads, the fake location/history/fetch, and the clock. */
function newBrowser() {
  return {
    sessionStorage: createStorage(),
    localStorage: createStorage(),
    clock: 1_000_000,
    assigned: [], // {url, shell, mainText} per location.assign, for the current load
    fetches: [],
    tokenReply: () => reply(200, { id_token: idToken(["editors"]), access_token: ACCESS_TOKEN, refresh_token: REFRESH_TOKEN }),
    apiReply: () => reply(200, {}),
    env: null,
  };
}

function shellDom(document) {
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    n.append(...kids);
    return n;
  };
  document.body.setAttribute("data-shell", "loading");
  const menu = el("button", { id: "menu", type: "button", hidden: "" });
  document.body.append(
    el("div", { class: "app" },
      el("div", { id: "sidebar", class: "sidebar" },
        el("span", { id: "stage", class: "stage" }),
        el("nav", { id: "nav", "aria-label": "Sections" }),
        el("div", { id: "account", class: "account" })),
      el("div", { class: "content" },
        el("header", { class: "topbar" }, menu, el("span", { id: "topbar-stage", class: "stage" })),
        el("main", { id: "main", tabindex: "-1" },
          el("div", { class: "skeleton skeleton--page", "aria-hidden": "true" }),
          el("p", { class: "visually-hidden" }, document.createTextNode("Loading…"))))),
    el("div", { id: "scrim", class: "scrim", hidden: "" }),
    el("div", { id: "toasts", class: "toasts" }),
    el("dialog", { id: "confirm", class: "dialog" }),
    el("div", { id: "live", class: "visually-hidden", role: "status", "aria-live": "polite" }));
}

function defineGlobal(k, v) {
  Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
}

/**
 * A page load: a fresh DOM and a fresh auth module, the address `hash` and `search`, then
 * start(). Returns start()'s outcome. `storageThrows` blocks sessionStorage for this load.
 */
async function load({ hash = "", search = "", storageThrows, sections = SECTIONS_FAKE, loadConfig } = {}) {
  uninstall();
  const b = browser;
  b.assigned = [];
  b.env = install({ sessionStorage: b.sessionStorage, localStorage: b.localStorage, storageThrows });
  shellDom(b.env.document);
  const location = {
    hash,
    search,
    pathname: "/",
    assign(url) {
      const main = b.env.document.getElementById("main");
      b.assigned.push({ url: String(url), shell: b.env.document.body.getAttribute("data-shell"), mainText: main.textContent });
    },
  };
  b.location = location;
  defineGlobal("location", location);
  defineGlobal("history", {
    replaceState(_s, _t, url) {
      const u = new URL(String(url), "https://d111.cloudfront.net");
      location.pathname = u.pathname;
      location.search = u.search;
      location.hash = u.hash;
    },
  });
  defineGlobal("fetch", async (url, init) => {
    const u = String(url);
    b.fetches.push({ url: u, init });
    if (u === TOKEN) return b.tokenReply(init);
    if (u === REVOKE) return reply(200, {});
    if (u.startsWith(cfg.apiBase)) return b.apiReply(u, init);
    throw new TypeError(`unexpected fetch ${u}`);
  });
  const auth = await import(`../lib/auth.js?load=${++loadNo}`);
  b.auth = auth;
  return start({
    auth,
    location,
    now: () => b.clock,
    loadConfig: loadConfig || (async () => cfg),
    sections,
  });
}

// Sections that stand in for the views, so these tests don't depend on them.
let rendered = [];
const SECTIONS_FAKE = [
  { id: "tools", label: "Tools", icon: "tools", render: (main, _ctx, r) => { rendered.push(["tools", r.arg]); main.replaceChildren(h("h1#page-title", {}, "Tools")); } },
  { id: "masking", label: "Masking", icon: "masking", render: (main) => { rendered.push(["masking"]); main.replaceChildren(h("h1#page-title", {}, "Masking")); } },
];

const doc = () => browser.env.document;
const shell = () => doc().body.getAttribute("data-shell");
const main = () => doc().getElementById("main");
const title = () => main().querySelector("h1#page-title")?.textContent;
const codeText = () => main().querySelector("code")?.textContent;
const buttons = () => main().querySelectorAll("button");
const button = (label) => buttons().find((b) => b.textContent.trim() === label);
const authorizeCalls = () => browser.assigned.filter((a) => a.url.startsWith(`${AUTHORIZE}?`));

/** Let pending promises (crypto.subtle, fetch fakes) run until `cond()` holds or the budget ends. */
async function settle(cond = () => false, rounds = 50) {
  for (let i = 0; i < rounds && !cond(); i++) await new Promise((r) => setImmediate(r));
}

/**
 * Click `label` and let the page settle. A click that should leave the page (`leaves`, the
 * default) is waited for by time as well as by rounds: the PKCE challenge goes through
 * WebCrypto on the thread pool, which can take longer than 50 rounds when the suite runs in
 * parallel. A click that shouldn't leave gets the rounds only.
 */
async function clickAndSettle(label, { leaves = true } = {}) {
  const btn = button(label);
  assert.ok(btn, `a "${label}" button on the page (have: ${buttons().map((b) => b.textContent).join(", ")})`);
  const before = browser.assigned.length;
  btn.click();
  await settle(() => browser.assigned.length > before);
  const deadline = Date.now() + 5_000;
  while (leaves && browser.assigned.length === before && Date.now() < deadline) await new Promise((r) => setTimeout(r, 1));
  await settle(() => false, 5);
}

/** The state value the last automatic or Try again sign-in put in the authorize URL. */
const lastState = () => new URL(browser.assigned.at(-1).url).searchParams.get("state");

/** Load once with no session (an automatic redirect), and return the pending state for the callback. */
async function beginSignIn(hash = "") {
  const out = await load({ hash });
  assert.equal(out, "redirect");
  return lastState();
}

beforeEach(() => {
  saved = Object.fromEntries(["location", "history", "fetch"].map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  browser = newBrowser();
  rendered = [];
});

afterEach(() => {
  uninstall();
  for (const [k, d] of Object.entries(saved)) {
    if (d) Object.defineProperty(globalThis, k, d);
    else delete globalThis[k];
  }
});

function assertBare() {
  assert.equal(shell(), "bare");
  const mark = main().querySelector("img");
  assert.ok(mark, "the bare page shows the mark");
  assert.equal(mark.getAttribute("src"), "owl.svg");
  assert.equal(mark.getAttribute("alt"), "OwlBank");
  assert.equal(main().querySelectorAll("h1").length, 1);
  assert.equal(main().querySelector("h1").id, "page-title");
  assert.equal(doc().getElementById("nav").childNodes.length, 0);
  assert.equal(doc().getElementById("account").childNodes.length, 0);
  assertNoInlineStyle(doc().body);
  assertClassesStyled(main());
}

// ---------------------------------------------------------------- examples

test("no session: one location.assign to the authorize URL, with no page rendered first", async () => {
  const out = await load({ hash: "#/masking" });
  assert.equal(out, "redirect");
  assert.equal(browser.assigned.length, 1);
  const [a] = browser.assigned;
  assert.ok(a.url.startsWith(`${AUTHORIZE}?`), a.url);
  assert.equal(a.shell, "loading", "nothing but the skeleton shows before the redirect");
  assert.equal(a.mainText, "Loading…");
  assert.equal(shell(), "loading");
  const pending = JSON.parse(browser.sessionStorage.getItem(PENDING));
  assert.equal(pending.hash, "#/masking");
  assert.deepEqual(JSON.parse(browser.sessionStorage.getItem(GUARD_KEY)), { count: 1, first: browser.clock });
  assert.equal(doc().getElementById("stage").textContent, "dev", "the stage shows before sign-in");
});

test("a signed-in visit renders the route, restores the hash and resets the guard", async () => {
  const state = await beginSignIn("#/masking");
  const out = await load({ search: `?code=abc&state=${state}` });
  assert.equal(out, "app");
  assert.equal(shell(), "app");
  assert.equal(browser.location.hash, "#/masking");
  assert.equal(browser.location.search, "");
  assert.deepEqual(rendered, [["masking"]]);
  assert.equal(title(), "Masking");
  assert.equal(browser.assigned.length, 0);
  assert.equal(browser.sessionStorage.getItem(GUARD_KEY), null);
  assert.equal(browser.sessionStorage.getItem(PENDING), null);
  assert.match(doc().getElementById("account").textContent, /a@example\.com/);
  assert.equal(doc().getElementById("menu").hidden, false);
  assert.equal(doc().title, "Masking · OwlBank Console");
});

test("a user in no console group gets the bare No access page with Sign out", async () => {
  browser.tokenReply = () => reply(200, { id_token: idToken([]), access_token: ACCESS_TOKEN, refresh_token: REFRESH_TOKEN });
  const state = await beginSignIn();
  assert.equal(await load({ search: `?code=abc&state=${state}` }), "no-access");
  assertBare();
  assert.equal(title(), "No access");
  assert.match(main().textContent, /isn't in a console group/);
  assert.ok(button("Sign out"));
  assert.deepEqual(rendered, []);
  await clickAndSettle("Sign out");
  assert.equal(browser.assigned.length, 1);
  assert.ok(browser.assigned[0].url.startsWith(`${LOGIN}/logout?`));
});

test("a missing config.js shows Not configured and never redirects", async () => {
  const out = await load({ loadConfig: async () => { throw new TypeError("Failed to fetch dynamically imported module"); } });
  assert.equal(out, "not-configured");
  assertBare();
  assert.equal(title(), "Not configured");
  assert.match(main().textContent, /config\.js is missing/);
  assert.equal(browser.assigned.length, 0);
  assert.equal(buttons().length, 0);
});

test("onAuthLost shows Session ended and never redirects; Sign in again redirects once", async () => {
  const sections = [{
    id: "tools", label: "Tools", icon: "tools",
    async render(m, ctx) {
      m.replaceChildren(h("h1#page-title", {}, "Tools"));
      try { await ctx.api.get(["tools"]); } catch { /* the view would show an error panel */ }
    },
  }];
  const state = await beginSignIn();
  browser.apiReply = () => reply(401, { message: "Unauthorized" });
  let refreshed = 0;
  browser.tokenReply = (init) => {
    if (String(init.body).includes("grant_type=refresh_token")) {
      refreshed++;
      return reply(400, { error: "invalid_grant" });
    }
    return reply(200, { id_token: idToken(["editors"]), access_token: ACCESS_TOKEN, refresh_token: REFRESH_TOKEN });
  };
  await load({ search: `?code=abc&state=${state}`, sections });
  await settle(() => shell() === "bare");
  assert.equal(refreshed, 1);
  assertBare();
  assert.equal(title(), "Your session has ended");
  assert.equal(browser.assigned.length, 0, "no automatic redirect");
  // Navigating doesn't bring the view back.
  browser.location.hash = "#/masking";
  browser.env.window.dispatchEvent(new Event("hashchange"));
  await settle(() => false, 5);
  assert.equal(title(), "Your session has ended");
  browser.sessionStorage.setItem(GUARD_KEY, JSON.stringify({ count: GUARD_MAX - 1, first: browser.clock }));
  await clickAndSettle("Sign in again");
  assert.equal(authorizeCalls().length, 1);
  assert.equal(browser.sessionStorage.getItem(GUARD_KEY), null);
});

test("Sign out still revokes the refresh token and goes to logoutUrl", async () => {
  const state = await beginSignIn();
  await load({ search: `?code=abc&state=${state}` });
  doc().getElementById("account").querySelector("button.sign-out").click();
  await settle(() => browser.assigned.length > 0);
  const revoke = browser.fetches.find((f) => f.url === REVOKE);
  assert.ok(revoke, "the refresh token is revoked");
  assert.match(String(revoke.init.body), new RegExp(`token=${REFRESH_TOKEN}`));
  assert.equal(browser.assigned.length, 1);
  const out = new URL(browser.assigned[0].url);
  assert.equal(`${out.origin}${out.pathname}`, `${LOGIN}/logout`);
  assert.equal(out.searchParams.get("logout_uri"), cfg.redirectUri);
});

test("a callback error shows the reason and the code as text, and Try again redirects once", async () => {
  const state = await beginSignIn();
  const desc = '<img src=x onerror="alert(1)"> nope';
  await load({ search: `?${new URLSearchParams({ error: "access_denied", error_description: desc, state })}` });
  assertBare();
  assert.equal(title(), "Couldn't sign you in");
  assert.match(main().textContent, /Sign-in was cancelled or refused\./);
  assert.equal(codeText(), "access_denied");
  assert.equal(main().querySelectorAll("img").length, 1, "only the mark; nothing is parsed as HTML");
  assert.equal(browser.assigned.length, 0);
  assert.equal(doc().title, "Couldn't sign you in · OwlBank Console");
  await clickAndSettle("Try again");
  assert.equal(authorizeCalls().length, 1);
});

test("an unknown error code shows its raw description", async () => {
  const state = await beginSignIn();
  await load({ search: `?${new URLSearchParams({ error: "server_error", error_description: "Pool <b>down</b>", state })}` });
  assert.match(main().textContent, /Pool <b>down<\/b>/);
  assert.equal(codeText(), "server_error");
});

test("the guard: the 3rd automatic redirect within 120 s shows Sign-in didn't complete; Try again resets it", async () => {
  for (let i = 0; i < GUARD_MAX - 1; i++) {
    assert.equal(await load(), "redirect");
    browser.clock += 10_000;
  }
  assert.equal(await load(), "loop");
  assertBare();
  assert.equal(title(), "Sign-in didn't complete");
  assert.match(main().textContent, /Sign-in didn't complete after several tries\./);
  assert.equal(codeText(), "loop");
  assert.equal(browser.assigned.length, 0);
  await clickAndSettle("Try again");
  assert.equal(authorizeCalls().length, 1);
  assert.equal(browser.sessionStorage.getItem(GUARD_KEY), null);
  assert.equal(await load(), "redirect", "after Try again the guard allows again");
});

test("a tripped guard allows a redirect again once its 120 s window has passed", async () => {
  for (let i = 0; i < GUARD_MAX - 1; i++) assert.equal(await load(), "redirect");
  assert.equal(await load(), "loop");
  browser.clock += GUARD_WINDOW_MS;
  assert.equal(await load(), "loop", "still within the window");
  browser.clock += 1;
  assert.equal(await load(), "redirect", "the window has passed");
  assert.deepEqual(JSON.parse(browser.sessionStorage.getItem(GUARD_KEY)), { count: 1, first: browser.clock });
});

test("redirects more than 120 s apart don't trip the guard", async () => {
  for (let i = 0; i < 5; i++) {
    assert.equal(await load(), "redirect");
    browser.clock += GUARD_WINDOW_MS + 1;
  }
});

test("blocked sessionStorage on an ordinary load shows the storage error, not a redirect", async () => {
  assert.equal(await load({ storageThrows: "sessionStorage" }), "error");
  assertBare();
  assert.equal(codeText(), "storage_unavailable");
  assert.equal(browser.assigned.length, 0);
  await clickAndSettle("Try again", { leaves: false });
  assert.equal(browser.assigned.length, 0, "sign-in can't start without session storage");
  assert.equal(codeText(), "storage_unavailable");
});

test("a double click on Try again starts one sign-in", async () => {
  const state = await beginSignIn();
  await load({ search: `?error=access_denied&state=${state}` });
  const btn = button("Try again");
  btn.click();
  btn.click();
  await settle(() => browser.assigned.length > 1, 20);
  assert.equal(authorizeCalls().length, 1);
});

// ----------------------------------------------------------------

const CODE_CHARS = `${ALNUM}_`;
const outcome = oneOf(
  record({ kind: "error", code: string({ minLength: 1, maxLength: 16, chars: CODE_CHARS }), description: unicodeString({ maxLength: 24 }) }),
  record({ kind: "error", code: oneOf("access_denied", "invalid_request", "server_error", "temporarily_unavailable"), description: string({ maxLength: 10 }) }),
  record({ kind: "mismatch", state: string({ minLength: 0, maxLength: 12 }) }),
  record({ kind: "token", status: int(400, 599), body: oneOf(undefined, {}, record({ error: string({ minLength: 1, maxLength: 12, chars: CODE_CHARS }) })) }),
  record({ kind: "network" }),
  record({ kind: "storage" }),
);

/** The {title, message, code} the page must show for outcome `o`. */
function expected(o) {
  const coded = (code, message) => Object.assign(new Error(message || code), { code });
  if (o.kind === "error") return signInReason(coded(o.code, o.description));
  if (o.kind === "mismatch") return signInReason({ code: "state_mismatch" });
  if (o.kind === "token") {
    const code = (o.body && o.body.error) || `token_endpoint_${o.status}`;
    return signInReason(coded(code, code));
  }
  if (o.kind === "network") return signInReason(new TypeError("Failed to fetch"));
  return signInReason({ code: "storage_unavailable" });
}

test("PBT: a callback error ends on the bare error page and never redirects until Try again, which redirects once", async () => {
  await forAll(outcome, async (o) => {
    browser = newBrowser();
    const state = await beginSignIn("#/tools");
    let search = `?${new URLSearchParams({ code: "abc", state })}`;
    if (o.kind === "error") search = `?${new URLSearchParams({ error: o.code, error_description: o.description, state })}`;
    if (o.kind === "mismatch") search = `?${new URLSearchParams({ code: "abc", state: o.state === state ? `${state}x` : o.state })}`;
    if (o.kind === "token") browser.tokenReply = () => reply(o.status, o.body);
    if (o.kind === "network") browser.tokenReply = () => { throw new TypeError("Failed to fetch"); };
    const out = await load({ search, storageThrows: o.kind === "storage" ? "sessionStorage" : undefined });
    const want = expected(o);
    assert.equal(out, "error");
    assertBare();
    assert.equal(title(), want.title);
    assert.ok(main().textContent.includes(want.message), `the message "${want.message}" shows`);
    assert.equal(codeText(), want.code);
    await settle(() => false, 5);
    assert.equal(browser.assigned.length, 0, "no automatic redirect");
    await clickAndSettle("Try again", { leaves: o.kind !== "storage" });
    if (o.kind === "storage") {
      assert.equal(browser.assigned.length, 0, "blocked storage: sign-in can't start");
      assert.equal(codeText(), "storage_unavailable");
    } else {
      assert.equal(browser.assigned.length, 1);
      assert.equal(authorizeCalls().length, 1);
    }
  }, { runs: 100 });
});

// ---------------------------------------------------------------- flow part

const flowEvent = oneOf(
  record({ op: "load-no-session" }),
  record({ op: "load-no-session" }),
  record({ op: "callback-ok" }),
  record({ op: "callback-error" }),
  record({ op: "tryAgain" }),
  record({ op: "advance", ms: int(0, 200_000) }),
  record({ op: "advance", ms: int(0, 30_000) }),
);

// The model keeps its own guard window: it opens at an automatic sign-in when none is open or the
// open one has passed (more than GUARD_WINDOW_MS since its first sign-in); a reset closes it.
test("PBT (flow): automatic sign-ins stay within GUARD_MAX - 1 per 120 s window since a reset, and a tripped guard shows the loop page until a reset or the window passes", async () => {
  await forAll(array(flowEvent, { minLength: 1, maxLength: 24 }), async (events) => {
    browser = newBrowser();
    let autos = []; // times of automatic sign-ins since the last reset
    let win = null; // the model's open window: {first, count}
    let lastPending = null; // a state from the last sign-in that started
    const fakePending = () => {
      // A sign-in that started earlier (this tab, or before the events began): its pending record.
      const st = "s".repeat(43);
      browser.sessionStorage.setItem(PENDING, JSON.stringify({ verifier: "v".repeat(64), state: st, hash: "#/tools" }));
      return st;
    };
    for (const e of events) {
      if (e.op === "advance") {
        browser.clock += e.ms;
      } else if (e.op === "load-no-session") {
        const passed = !win || browser.clock - win.first > GUARD_WINDOW_MS;
        const allowed = passed || win.count < GUARD_MAX - 1;
        const out = await load();
        if (allowed) {
          assert.equal(out, "redirect", passed ? "a passed window allows" : "within the window, under GUARD_MAX - 1");
          assert.equal(browser.assigned.length, 1);
          assert.equal(browser.assigned[0].shell, "loading");
          if (passed) win = { first: browser.clock, count: 0 };
          win.count++;
          autos.push(browser.clock);
          lastPending = lastState();
        } else {
          assert.equal(out, "loop", "a tripped guard stays tripped until a reset or the window passes");
          assert.equal(browser.assigned.length, 0);
          assert.equal(title(), "Sign-in didn't complete");
        }
      } else if (e.op === "callback-ok" || e.op === "callback-error") {
        const state = lastPending && browser.sessionStorage.getItem(PENDING) ? lastPending : fakePending();
        const search = e.op === "callback-ok" ? `?code=abc&state=${state}` : `?error=access_denied&state=${state}`;
        const out = await load({ search });
        lastPending = null;
        assert.equal(browser.assigned.length, 0, "a callback never redirects by itself");
        if (e.op === "callback-ok") {
          assert.equal(out, "app");
          assert.equal(browser.sessionStorage.getItem(GUARD_KEY), null);
          autos = [];
          win = null;
        } else {
          assert.equal(out, "error");
          assert.equal(title(), "Couldn't sign you in");
        }
      } else if (e.op === "tryAgain") {
        // Only an error or loop page that is still showing has it (after a redirect the tab has left).
        if (!browser.env || browser.assigned.length || !button("Try again")) continue;
        await clickAndSettle("Try again");
        assert.equal(browser.assigned.length, 1, "Try again redirects exactly once");
        assert.equal(browser.sessionStorage.getItem(GUARD_KEY), null);
        lastPending = lastState();
        autos = [];
        win = null;
      }
      if (win) assert.ok(win.count <= GUARD_MAX - 1, `${win.count} automatic sign-ins in the window from ${win.first}`);
      // Windows don't overlap, so any 120 s span meets at most two of them.
      for (const t of autos) {
        const inSpan = autos.filter((u) => u >= t && u - t <= GUARD_WINDOW_MS).length;
        assert.ok(inSpan <= 2 * (GUARD_MAX - 1), `${inSpan} automatic sign-ins within 120 s of ${t}: ${autos}`);
      }
    }
  }, { runs: 60 });
});

// ---------------------------------------------------------------- runtime part

test("after a full sign-in, a theme change and one save, no token is in web storage", async () => {
  const sections = [{
    id: "tools", label: "Tools", icon: "tools",
    render(m, ctx) {
      const status = h("div");
      const submit = h("button.primary", { type: "submit" }, "Save");
      const form = h("form", { on: { submit: (ev) => { ev.preventDefault(); save(status, () => ctx.api.put(["tools", "x"], { masked: true }), { form, button: submit }); } } }, submit);
      m.replaceChildren(h("h1#page-title", {}, "Tools"), form, status);
    },
  }];
  browser.apiReply = (_u, init) => reply(200, init.method === "PUT" ? { changed: true } : {});
  const state = await beginSignIn("#/tools");
  await load({ search: `?code=abc&state=${state}`, sections });
  assert.equal(shell(), "app");
  doc().getElementById("account").querySelector("button.theme-toggle").click();
  main().querySelector("form").dispatchEvent(new Event("submit", { cancelable: true }));
  await settle(() => browser.fetches.some((f) => f.init && f.init.method === "PUT"));
  await settle(() => false, 10);
  const put = browser.fetches.find((f) => f.init && f.init.method === "PUT");
  assert.ok(put, "the save reached the API");
  assert.match(put.init.headers.Authorization, JWT_RE, "the token is sent, from memory");
  for (const k of browser.localStorage.keys()) assert.ok([THEME_KEY].includes(k), `unexpected localStorage key ${k}`);
  for (const s of [browser.localStorage, browser.sessionStorage]) {
    for (const k of s.keys()) assert.doesNotMatch(s.getItem(k), JWT_RE, `a token in storage under ${k}`);
  }
});
