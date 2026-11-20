// The console theme: pure logic for the sidebar toggle. The theme lives on
// <html data-theme="light|dark">; theme.js sets it before first paint. Light is the default and
// the system setting is never followed. Every storage access is inside try/catch, so blocked
// or throwing storage reads as light and a failed write only affects this page.

export const THEME_KEY = "owlbank-admin-theme";

/** "dark" stays dark; any other value (null, garbage, another case) is light. */
export const normalise = (v) => (v === "dark" ? "dark" : "light");

/** The named Web Storage, or null when the accessor throws (blocked site data) or it is missing. */
export function safeStorage(name) {
  try { return globalThis[name] ?? null; } catch { return null; }
}

/** The store the theme is remembered in (the only use of localStorage outside theme.js). */
export const themeStorage = () => safeStorage("localStorage");

export function readTheme(storage) {
  try { return normalise(storage?.getItem(THEME_KEY)); } catch { return "light"; }
}

/** Persist the theme. False when there is no storage or the write throws (nothing is persisted). */
export function writeTheme(storage, t) {
  if (!storage) return false;
  try { storage.setItem(THEME_KEY, normalise(t)); return true; } catch { return false; }
}

export function applyTheme(root, t) { root.setAttribute("data-theme", normalise(t)); }

/** Switch the page's theme and try to remember it. Returns the new theme. */
export function toggleTheme(root, storage) {
  const next = normalise(root.getAttribute("data-theme")) === "dark" ? "light" : "dark";
  applyTheme(root, next);
  writeTheme(storage, next);
  return next;
}
