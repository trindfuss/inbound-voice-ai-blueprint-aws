// The OwlBank admin console. Plain ES modules, no dependencies, no inline script.
// SECTIONS is the one registry of the console's sections: the nav, the route ids,
// the page title and the admin-only gating all come from it. Adding a section is one entry here
// plus its <symbol> in icons.svg. The module touches no DOM at import time, so tests import it.

import { createClient } from "./lib/api.js";
import * as authModule from "./lib/auth.js";
import { card, emptyState, pageHeader } from "./lib/components.js";
import { clear, h } from "./lib/dom.js";
import { parseRoute } from "./lib/route.js";
import { initMenu, renderAccount, renderNav, renderStage, setShell } from "./lib/shell.js";
import { clearGuard, guardAllows, guardRecord, readGuard, signInReason, writeGuard } from "./lib/signin.js";
import { isAdmin } from "./lib/ui.js";
import { auditView } from "./views/audit.js";
import { maskingView } from "./views/masking.js";
import { promptsView } from "./views/prompts.js";
import { redactionView } from "./views/redaction.js";
import { slotsView } from "./views/slots.js";
import { toolsView } from "./views/tools.js";
import { usersView } from "./views/users.js";

/**
 * {id, label, icon, render(main, ctx, {arg, params}), when?(user), denied?}. A section whose
 * `when(user)` is false is left out of the nav, and its route shows `denied` in an empty state.
 */
export const SECTIONS = [
  { id: "tools", label: "Tools", icon: "tools", render: toolsView },
  { id: "masking", label: "Masking", icon: "masking", render: maskingView },
  { id: "prompts", label: "Prompts", icon: "prompts", render: promptsView },
  { id: "slots", label: "Slots", icon: "slots", render: slotsView },
  { id: "redaction", label: "Redaction", icon: "redaction", render: redactionView, when: isAdmin, denied: "Only admins manage transcript redaction." },
  { id: "audit", label: "Audit log", icon: "history", render: auditView },
  { id: "users", label: "Users", icon: "users", render: usersView, when: isAdmin, denied: "Only admins manage users." },
];

const PRODUCT = "OwlBank Console";

/** Render the route in `hash` into `main` through the registry, and the nav for it. */
export async function render(ctx, { main, nav, hash, sections = SECTIONS }) {
  const { view, arg, params } = parseRoute(hash, sections.map((s) => s.id));
  const section = sections.find((s) => s.id === view);
  renderNav(nav, sections, view, ctx.user);
  document.title = `${section.label} · ${PRODUCT}`;
  if (section.when && !section.when(ctx.user)) {
    clear(main,
      pageHeader({ title: section.label }),
      card({ label: section.label },
        emptyState({ icon: section.icon, title: section.denied || "You don't have access to this section." })));
  } else {
    await section.render(main, ctx, { arg, params });
  }
  main.focus();
}

const defaultLoadConfig = async () => (await import("./config.js")).default;

/**
 * The page load. Finishes a sign-in callback if this is one; with no session it
 * redirects to the login page at once (no page in between) unless the loop guard refuses; every
 * sign-in error, the loop, a session that ends mid-use, no access and a missing config.js end on a
 * bare page and never redirect by themselves. env (all optional, for tests): `auth` (lib/auth.js),
 * `location`, `storage` (the guard's; sessionStorage by default), `now()`, `loadConfig()`,
 * `sections`, `win` (for hashchange) and `createApiClient`. Resolves to the outcome: "redirect",
 * "error", "loop", "no-access", "not-configured" or "app".
 */
export async function start({
  auth = authModule,
  location = globalThis.location,
  storage,
  now = () => Date.now(),
  loadConfig = defaultLoadConfig,
  sections = SECTIONS,
  win = globalThis.window,
  createApiClient = createClient,
} = {}) {
  const main = document.getElementById("main");
  const nav = document.getElementById("nav");
  const account = document.getElementById("account");

  /** The bare centred card: the mark, h1#page-title, the message, the raw code, one action. */
  function barePage({ title, message, code, action }) {
    setShell("bare");
    clear(nav);
    clear(account);
    document.title = `${title} · ${PRODUCT}`;
    clear(main, card({ label: title },
      h("img.brand-mark", { src: "owl.svg", alt: "OwlBank", width: 48, height: 48 }),
      h("h1#page-title", {}, title),
      message ? h("p", {}, message) : null,
      code ? h("p", {}, "Code: ", h("code", {}, code)) : null,
      action ? h("button.primary", { type: "button", on: { click: action.run } }, action.label) : null));
    main.focus();
  }

  // Try again and Sign in again: a user's choice, so the guard starts over. One sign-in at a time.
  let signingIn = false;
  async function signInAgain() {
    if (signingIn) return;
    signingIn = true;
    clearGuard(storage);
    try {
      await auth.signIn();
    } catch (err) {
      signingIn = false;
      errorPage(err);
    }
  }

  function errorPage(err) {
    const { title, message, code } = signInReason(err);
    barePage({ title, message, code, action: { label: "Try again", run: signInAgain } });
    return code === "loop" ? "loop" : "error";
  }

  let config;
  try {
    config = await loadConfig();
  } catch {
    barePage({ title: "Not configured", message: "config.js is missing: it is generated when the OwlBankAdmin stack deploys." });
    return "not-configured";
  }
  renderStage(config.stage);
  auth.configure(config);

  let result;
  try {
    result = await auth.completeSignIn();
  } catch (err) {
    return errorPage(err); // a callback error never redirects by itself
  }
  if (result === "signed-in") {
    clearGuard(storage);
  } else if (!auth.isSignedIn()) {
    const guard = readGuard(storage);
    if (!guardAllows(guard, now())) return errorPage({ code: "loop" });
    writeGuard(storage, guardRecord(guard, now()));
    try {
      await auth.signIn(); // straight to the login page
    } catch (err) {
      return errorPage(err);
    }
    return "redirect";
  }

  let go = null;
  const api = createApiClient({
    base: config.apiBase,
    getToken: () => auth.idToken(),
    refresh: () => auth.refresh(),
    onAuthLost: () => {
      // A page, not a redirect, so nothing loops on a 401.
      if (go) win.removeEventListener("hashchange", go);
      go = null;
      barePage({
        title: "Your session has ended",
        message: "Sign in again to carry on.",
        action: { label: "Sign in again", run: signInAgain },
      });
    },
  });
  const ctx = { api, user: auth.user() };
  if (!ctx.user.groups.length) {
    barePage({
      title: "No access",
      message: "Your user isn't in a console group. Ask an admin to add you.",
      action: { label: "Sign out", run: () => auth.signOut() },
    });
    return "no-access";
  }
  setShell("app");
  renderAccount(account, ctx.user, { onSignOut: () => auth.signOut() });
  initMenu({
    button: document.getElementById("menu"),
    sidebar: document.getElementById("sidebar"),
    scrim: document.getElementById("scrim"),
    inertTargets: [document.querySelector(".topbar"), main],
  });
  go = () => render(ctx, { main, nav, hash: location.hash, sections });
  win.addEventListener("hashchange", go);
  await go();
  return "app";
}

if (typeof document !== "undefined") start();
