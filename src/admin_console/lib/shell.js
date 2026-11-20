// The page shell. index.html holds the static shell; body[data-shell] says which
// part of it shows, and layout.css alone decides what that means, so no style is ever set.
//   loading: only the skeleton main (before the sign-in redirect, nothing flashes)
//   app:     sidebar, top bar and main
//   bare:    a centred card (sign-in errors, no access, session ended, not configured)
// This module also fills the sidebar: the nav from SECTIONS, the account block with the theme
// toggle, the environment badge, and the drawer menu below 960px.

import { badge, envBadge } from "./components.js";
import { clear, h, icon } from "./dom.js";
import { themeStorage, toggleTheme } from "./theme.js";

export const SHELL_STATES = Object.freeze(["loading", "app", "bare"]);

/** Switch the shell to `state`. Throws on anything else, so a typo can't hide the console. */
export function setShell(state) {
  if (!SHELL_STATES.includes(state)) throw new Error(`not a shell state: ${String(state)}`);
  document.body.setAttribute("data-shell", state);
}

// ---------------------------------------------------------------- nav

/** The sections this user sees: those without `when`, or whose `when(user)` is true. */
export function visibleSections(sections, user) {
  return sections.filter((s) => !s.when || s.when(user));
}

/**
 * ul > li > a.nav-link[href=#/id] (icon and label) for each visible section. The link for
 * `currentId` has aria-current="page" and .is-current (the active fill, the amber bar, weight 600).
 */
export function renderNav(nav, sections, currentId, user) {
  clear(nav, h("ul.nav-list", {}, visibleSections(sections, user).map((s) => {
    const current = s.id === currentId;
    return h("li", {}, h("a.nav-link", {
      href: `#/${s.id}`,
      class: current ? "is-current" : null,
      "aria-current": current ? "page" : null,
    }, icon(s.icon), h("span.nav-label", {}, s.label)));
  })));
  return nav;
}

// ---------------------------------------------------------------- theme toggle

const THEME_ICON = { light: "moon", dark: "sun" };

/**
 * The theme toggle button: it names the mode a click switches to ("Dark mode" with a moon in
 * light, "Light mode" with a sun in dark), and a click switches and remembers the theme through
 * lib/theme.js.
 */
export function themeToggle(root = document.documentElement, storage = themeStorage()) {
  const btn = h("button.sidebar-button.theme-toggle", { type: "button" });
  const paint = () => {
    // The label names the mode the click switches to, so no aria-pressed.
    const dark = root.getAttribute("data-theme") === "dark";
    clear(btn, icon(THEME_ICON[dark ? "dark" : "light"]), h("span", {}, dark ? "Light mode" : "Dark mode"));
  };
  btn.addEventListener("click", () => {
    toggleTheme(root, storage);
    paint();
  });
  paint();
  return btn;
}

// ---------------------------------------------------------------- account and stage

/** The account block: who is signed in, one badge per group (or "no group"), the theme toggle, Sign out. */
export function renderAccount(el, user, { onSignOut, root, storage } = {}) {
  const groups = user.groups || [];
  clear(el,
    h("p.account-user", {}, user.email || user.username || ""),
    h("div.account-roles", {}, groups.length ? groups.map((g) => badge(g, "neutral")) : badge("no group", "neutral")),
    h("div.account-actions", {},
      themeToggle(root, storage),
      h("button.sidebar-button.sign-out", { type: "button", on: { click: () => onSignOut && onSignOut() } },
        icon("sign-out"), h("span", {}, "Sign out"))));
  return el;
}

/** The environment badge (envBadge) into each target: by default #stage and the top-bar copy. */
export function renderStage(stage, targets = [document.getElementById("stage"), document.getElementById("topbar-stage")]) {
  for (const el of targets) if (el) clear(el, envBadge(stage));
}

// ---------------------------------------------------------------- the drawer menu

const defaultMql = () => (typeof globalThis.matchMedia === "function" ? globalThis.matchMedia("(min-width: 960px)") : null);

/**
 * Wire the menu button to the sidebar drawer below 960px. Opening adds .is-open, shows the scrim,
 * sets `inert` on `inertTargets` (the top bar and main, so focus stays in the drawer) and focuses
 * the current nav link. Esc, a scrim click, a nav-link click and crossing 960px close it; closing
 * clears `inert` and returns focus to the button (except at desktop width, where the button is
 * hidden and focus stays in the now persistent sidebar). The button starts `hidden` in the HTML,
 * so it appears only once this has run. Returns {open(), close(), isOpen()}.
 */
export function initMenu({ button, sidebar, scrim, inertTargets = [], mql = defaultMql(), doc = document }) {
  let open = false;
  const wide = () => Boolean(mql && mql.matches);

  function setOpen(on) {
    open = on;
    sidebar.classList.toggle("is-open", on);
    button.setAttribute("aria-expanded", on ? "true" : "false");
    if (scrim) scrim.hidden = !on;
    for (const el of inertTargets) if (el) el.inert = on;
  }

  const api = {
    isOpen: () => open,
    open() {
      if (open || wide()) return;
      setOpen(true);
      const target = sidebar.querySelector('a.nav-link[aria-current="page"]') || sidebar.querySelector("a.nav-link");
      if (target) target.focus();
    },
    close({ returnFocus = true } = {}) {
      if (!open) return;
      setOpen(false);
      if (returnFocus && !wide()) button.focus();
    },
  };

  button.hidden = false;
  button.addEventListener("click", () => (open ? api.close() : api.open()));
  if (scrim) scrim.addEventListener("click", () => api.close());
  sidebar.addEventListener("click", (ev) => {
    if (ev.target && typeof ev.target.closest === "function" && ev.target.closest("a.nav-link")) api.close();
  });
  doc.addEventListener("keydown", (ev) => {
    if (open && ev.key === "Escape") {
      ev.preventDefault();
      api.close();
    }
  });
  if (mql && typeof mql.addEventListener === "function") mql.addEventListener("change", () => api.close());
  return api;
}
