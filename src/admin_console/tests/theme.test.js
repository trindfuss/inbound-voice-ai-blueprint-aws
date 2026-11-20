// The theme: `theme.js` sets data-theme before first paint from localStorage, and
// `lib/theme.js` holds the pure logic the sidebar toggle uses. Light is the default, the system
// setting is never followed, and blocked or garbage storage reads as light.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { runInThisContext } from "node:vm";
import { createStorage, install, uninstall } from "./fakedom.js";
import { array, gen, int, oneOf, string, unicodeString, forAll } from "./gen.js";
import {
  THEME_KEY, applyTheme, normalise, readTheme, safeStorage, themeStorage, toggleTheme, writeTheme,
} from "../lib/theme.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BOOT = readFileSync(join(ROOT, "theme.js"), "utf8");

/** Run theme.js as the browser would: a classic script against the installed globals. */
const boot = () => runInThisContext(BOOT, { filename: "theme.js" });
const theme = () => document.documentElement.getAttribute("data-theme");

/** Make the localStorage accessor itself throw (site data blocked), on globalThis and window. */
function blockAccessor() {
  const desc = { configurable: true, enumerable: false, get() { throw new DOMException("insecure", "SecurityError"); } };
  Object.defineProperty(globalThis, "localStorage", desc);
  Object.defineProperty(window, "localStorage", desc);
}

function unblockAccessor(storage) {
  const desc = { configurable: true, enumerable: false, writable: true, value: storage };
  Object.defineProperty(globalThis, "localStorage", desc);
  Object.defineProperty(window, "localStorage", desc);
}

function withDom(opts, fn) {
  const env = install(opts);
  try { return fn(env); } finally { uninstall(); }
}

// ---------------------------------------------------------------- lib/theme.js examples

test("THEME_KEY is the documented key", () => {
  assert.equal(THEME_KEY, "owlbank-admin-theme");
});

test("normalise: only 'dark' is dark; everything else is light", () => {
  assert.equal(normalise("dark"), "dark");
  for (const v of ["light", null, undefined, "", "Dark", "DARK", " dark", "system", 1, {}]) {
    assert.equal(normalise(v), "light", String(v));
  }
});

test("readTheme: light for an empty, missing, throwing or garbage store; dark only when stored", () => {
  const s = createStorage();
  assert.equal(readTheme(s), "light");
  assert.equal(readTheme(null), "light");
  assert.equal(readTheme(undefined), "light");
  s.setItem(THEME_KEY, "dark");
  assert.equal(readTheme(s), "dark");
  s.setItem(THEME_KEY, "purple");
  assert.equal(readTheme(s), "light");
  s.setItem(THEME_KEY, "dark");
  s.throws = true;
  assert.equal(readTheme(s), "light");
});

test("writeTheme: stores the normalised value under the key only; false when it can't", () => {
  const s = createStorage();
  assert.equal(writeTheme(s, "dark"), true);
  assert.equal(s.getItem(THEME_KEY), "dark");
  assert.equal(writeTheme(s, "garbage"), true);
  assert.equal(s.getItem(THEME_KEY), "light");
  assert.deepEqual(s.keys(), [THEME_KEY]);
  s.throws = true;
  assert.equal(writeTheme(s, "dark"), false);
  assert.equal(writeTheme(null, "dark"), false, "no storage: nothing persisted");
});

test("safeStorage / themeStorage: the store, or null when the accessor throws or is missing", () => {
  withDom({}, (env) => {
    assert.equal(safeStorage("localStorage"), env.localStorage);
    assert.equal(themeStorage(), env.localStorage);
  });
  withDom({ storageThrows: "localStorage" }, () => {
    assert.equal(safeStorage("localStorage"), null);
    assert.equal(themeStorage(), null);
  });
  assert.equal(safeStorage("noSuchStorageHere"), null);
});

test("applyTheme / toggleTheme: set data-theme, persist, and keep switching when storage fails", () => {
  withDom({}, (env) => {
    const root = document.documentElement;
    applyTheme(root, "bogus");
    assert.equal(theme(), "light");
    assert.equal(toggleTheme(root, env.localStorage), "dark");
    assert.equal(theme(), "dark");
    assert.equal(env.localStorage.getItem(THEME_KEY), "dark");
    env.localStorage.throws = true;
    assert.equal(toggleTheme(root, env.localStorage), "light", "the page still switches");
    assert.equal(theme(), "light");
    env.localStorage.throws = false;
    assert.equal(env.localStorage.getItem(THEME_KEY), "dark", "nothing was persisted");
    assert.equal(toggleTheme(root, null), "dark");
  });
});

// ---------------------------------------------------------------- theme.js examples

test("theme.js: a fresh browser is light (never the system setting)", () => {
  withDom({}, () => {
    boot();
    assert.equal(theme(), "light");
  });
});

test("theme.js: stored dark is dark; stored light or garbage is light", () => {
  for (const [stored, want] of [["dark", "dark"], ["light", "light"], ["Dark", "light"], ["", "light"], ["{}", "light"]]) {
    const s = createStorage();
    s.setItem(THEME_KEY, stored);
    withDom({ localStorage: s }, () => {
      boot();
      assert.equal(theme(), want, JSON.stringify(stored));
    });
  }
});

test("theme.js: light, with no error, when storage calls throw or the accessor throws", () => {
  const s = createStorage();
  s.setItem(THEME_KEY, "dark");
  s.throws = true;
  withDom({ localStorage: s }, () => { boot(); assert.equal(theme(), "light"); });
  s.throws = false;
  withDom({ localStorage: s, storageThrows: "localStorage" }, () => { boot(); assert.equal(theme(), "light"); });
  withDom({ localStorage: s }, () => { blockAccessor(); boot(); assert.equal(theme(), "light"); });
});

test("theme.js: a classic script (no import or export) that only touches data-theme and the theme key", () => {
  assert.doesNotMatch(BOOT, /^\s*(import|export)\b/m);
  assert.doesNotMatch(BOOT, /prefers-color-scheme|matchMedia/, "never follows the system setting");
  assert.match(BOOT, /try\s*\{[^}]*localStorage[^}]*\}\s*catch/, "storage access is inside try/catch");
});

test("lib/theme.js never follows the system setting", () => {
  const src = readFileSync(join(ROOT, "lib/theme.js"), "utf8");
  assert.doesNotMatch(src, /prefers-color-scheme|matchMedia/);
});

// ---------------------------------------------------------------- properties

test("theme.js and readTheme agree on every stored value", () => {
  const value = oneOf("dark", "light", string({ maxLength: 8 }), unicodeString({ maxLength: 8 }),
    gen((rng) => (rng() < 0.5 ? " dark" : "dark ")));
  forAll(value, (stored) => {
    const s = createStorage();
    s.setItem(THEME_KEY, stored);
    return withDom({ localStorage: s }, () => {
      boot();
      return theme() === readTheme(s) && theme() === (stored === "dark" ? "dark" : "light");
    });
  });
});

test("toggleTheme twice is the identity on data-theme", () => {
  const start = oneOf("dark", "light", null, string({ maxLength: 6 }));
  const store = oneOf("ok", "throws", "none");
  forAll(gen((rng) => ({ start: start(rng), store: store(rng) })), ({ start: v, store: mode }) =>
    withDom({}, (env) => {
      const root = document.documentElement;
      if (v !== null) root.setAttribute("data-theme", v);
      env.localStorage.throws = mode === "throws";
      const storage = mode === "none" ? null : env.localStorage;
      const before = normalise(root.getAttribute("data-theme"));
      toggleTheme(root, storage);
      toggleTheme(root, storage);
      return theme() === before;
    }));
});

// The state machine. The model tracks what the page shows, what the store holds, and whether the
// last toggled value has reached the store unchanged (every write since it succeeded, no garbage).
const op = oneOf(
  { kind: "toggle" },
  { kind: "reload" },
  { kind: "throws", on: true, accessor: false },
  { kind: "throws", on: true, accessor: true },
  { kind: "throws", on: false },
  gen((rng) => ({ kind: "garbage", value: oneOf(string({ maxLength: 8 }), "dark", "light", "")(rng) })),
);

test("the theme round-trips through toggle, reload, blocked storage and garbage", () => {
  forAll(array(op, { minLength: 1, maxLength: 24 }), (ops) => {
    const storage = createStorage(); // survives reloads, as a browser's does
    const m = { page: "light", stored: null, mode: "ok", lastToggled: null, intact: true };
    const load = () => {
      install({ localStorage: storage });
      storage.throws = m.mode === "calls";
      if (m.mode === "accessor") blockAccessor();
      boot();
    };
    load();
    try {
      assert.equal(theme(), "light", "a fresh browser is light");
      for (const o of ops) {
        if (o.kind === "toggle") {
          const next = toggleTheme(document.documentElement, themeStorage());
          assert.equal(next, m.page === "dark" ? "light" : "dark");
          m.page = next;
          m.lastToggled = next;
          if (m.mode === "ok") { m.stored = next; m.intact = true; } else m.intact = false;
        } else if (o.kind === "reload") {
          uninstall();
          load();
          m.page = m.mode === "ok" ? normalise(m.stored) : "light";
          assert.equal(theme(), m.page, "reload shows what the store holds, or light");
          // The spec's form: the last toggled value if every write since succeeded.
          if (m.lastToggled !== null && m.intact && m.mode === "ok") assert.equal(theme(), m.lastToggled);
        } else if (o.kind === "throws") {
          m.mode = o.on ? (o.accessor ? "accessor" : "calls") : "ok";
          // Both take effect on the next page load; "calls" also applies to this page at once.
          storage.throws = m.mode === "calls";
          if (m.mode === "accessor") blockAccessor();
          else unblockAccessor(storage);
        } else if (o.kind === "garbage") {
          const was = storage.throws;
          storage.throws = false;
          storage.setItem(THEME_KEY, o.value);
          storage.throws = was;
          m.stored = o.value;
          m.intact = false;
        }
        assert.ok(["light", "dark"].includes(theme()), `data-theme is ${theme()}`);
        assert.equal(theme(), m.page);
      }
      // localStorage holds only the theme (for the theme's own writes).
      assert.ok(storage.keys().every((k) => k === THEME_KEY));
    } finally {
      uninstall();
    }
  }, { runs: 300 });
});

test("a write that fails is never read back as the toggled value", () => {
  forAll(int(1, 6), (n) => withDom({}, (env) => {
    const root = document.documentElement;
    env.localStorage.throws = true;
    for (let i = 0; i < n; i++) toggleTheme(root, env.localStorage);
    env.localStorage.throws = false;
    return readTheme(env.localStorage) === "light";
  }));
});
