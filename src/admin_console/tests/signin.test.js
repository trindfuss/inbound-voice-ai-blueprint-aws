// lib/signin.js: the redirect loop guard, safeHash and the sign-in error reasons.
// PBT: the guard part and the safeHash part. The flow parts are
// in auth.test.js and signin-flow.test.js.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  GUARD_KEY, GUARD_MAX, GUARD_WINDOW_MS, clearGuard, guardAllows, guardRecord, readGuard, safeHash,
  signInReason, writeGuard,
} from "../lib/signin.js";
import { createStorage } from "./fakedom.js";
import { ALNUM, array, forAll, gen, int, oneOf, record, string, unicodeString } from "./gen.js";

const throwing = {
  getItem() { throw new Error("SecurityError"); },
  setItem() { throw new Error("QuotaExceededError"); },
  removeItem() { throw new Error("SecurityError"); },
};

// ---------------------------------------------------------------- constants

test("the guard constants match the design", () => {
  assert.equal(GUARD_KEY, "owlbank-admin-signin-guard");
  assert.equal(GUARD_MAX, 3);
  assert.equal(GUARD_WINDOW_MS, 120_000);
});

// ---------------------------------------------------------------- guard storage

test("writeGuard, readGuard and clearGuard round-trip through sessionStorage-like storage", () => {
  const s = createStorage();
  assert.equal(readGuard(s), null);
  const state = { count: 2, first: 1_000 };
  assert.deepEqual(writeGuard(s, state), state);
  assert.equal(typeof s.getItem(GUARD_KEY), "string");
  assert.deepEqual(readGuard(s), state);
  assert.equal(clearGuard(s), true);
  assert.equal(s.getItem(GUARD_KEY), null);
  assert.equal(readGuard(s), null);
});

test("readGuard, writeGuard and clearGuard return null when storage throws", () => {
  assert.equal(readGuard(throwing), null);
  assert.equal(writeGuard(throwing, { count: 1, first: 0 }), null);
  assert.equal(clearGuard(throwing), null);
});

test("the guard helpers return null when the storage accessor itself throws", () => {
  const blocked = () => { throw new Error("SecurityError"); };
  assert.equal(readGuard(blocked), null);
  assert.equal(writeGuard(blocked, { count: 1, first: 0 }), null);
  assert.equal(clearGuard(blocked), null);
  assert.equal(readGuard(null), null);
  // Omitted storage means the global sessionStorage, which Node 25+ defines: hide it for this case.
  const desc = Object.getOwnPropertyDescriptor(globalThis, "sessionStorage");
  try {
    Object.defineProperty(globalThis, "sessionStorage", { configurable: true, value: undefined });
    assert.equal(writeGuard(undefined, { count: 1, first: 0 }), null);
  } finally {
    if (desc) Object.defineProperty(globalThis, "sessionStorage", desc);
    else delete globalThis.sessionStorage;
  }
});

test("readGuard defaults to the global sessionStorage inside the try/catch", () => {
  const desc = Object.getOwnPropertyDescriptor(globalThis, "sessionStorage");
  try {
    Object.defineProperty(globalThis, "sessionStorage", { configurable: true, get() { throw new Error("blocked"); } });
    assert.equal(readGuard(), null);
    assert.equal(writeGuard(undefined, { count: 1, first: 0 }), null);
    assert.equal(clearGuard(), null);
    const s = createStorage();
    Object.defineProperty(globalThis, "sessionStorage", { configurable: true, value: s });
    assert.deepEqual(writeGuard(undefined, { count: 1, first: 5 }), { count: 1, first: 5 });
    assert.deepEqual(readGuard(), { count: 1, first: 5 });
    assert.equal(clearGuard(), true);
  } finally {
    if (desc) Object.defineProperty(globalThis, "sessionStorage", desc);
    else delete globalThis.sessionStorage;
  }
});

test("readGuard returns null for garbage or malformed records", () => {
  for (const raw of ["", "{", "null", "42", '"x"', "[]", '{"count":"2","first":0}', '{"count":1}',
    '{"count":-1,"first":0}', '{"count":1.5,"first":0}', '{"count":1,"first":"0"}']) {
    const s = createStorage();
    s.setItem(GUARD_KEY, raw);
    assert.equal(readGuard(s), null, raw);
  }
});

// ---------------------------------------------------------------- guard logic

test("guardAllows allows a first redirect and refuses the third within the window", () => {
  assert.equal(guardAllows(null, 0), true);
  let st = guardRecord(null, 0);
  assert.deepEqual(st, { count: 1, first: 0 });
  assert.equal(guardAllows(st, 1_000), true);
  st = guardRecord(st, 1_000);
  assert.deepEqual(st, { count: 2, first: 0 });
  assert.equal(guardAllows(st, 2_000), false);
});

test("guardRecord starts a new window once the old one has passed", () => {
  const st = guardRecord({ count: 1, first: 0 }, GUARD_WINDOW_MS + 1);
  assert.deepEqual(st, { count: 1, first: GUARD_WINDOW_MS + 1 });
});

test("a tripped guard refuses for the rest of its 120 s window, then allows a new window", () => {
  const st = { count: GUARD_MAX - 1, first: 0 };
  for (const now of [0, 1_000, GUARD_WINDOW_MS]) assert.equal(guardAllows(st, now), false, `at ${now}`);
  for (const now of [GUARD_WINDOW_MS + 1, GUARD_WINDOW_MS * 10, 1e12]) assert.equal(guardAllows(st, now), true, `at ${now}`);
  assert.deepEqual(guardRecord(st, GUARD_WINDOW_MS + 1), { count: 1, first: GUARD_WINDOW_MS + 1 });
});

// Guard part. A model of the start flow: `record` is an automatic redirect, made only
// when guardAllows says so (as app.js does), `clear` is a success or Try again, `advance` moves
// the clock. The state goes through real storage each step, as across page loads. The model
// keeps its own window: it opens at a redirect when none is open or the open one has passed
// (more than GUARD_WINDOW_MS since its first redirect), and a clear closes it.
const guardEvent = oneOf(
  record({ op: "record" }),
  record({ op: "clear" }),
  record({ op: "advance", ms: int(0, 200_000) }),
  record({ op: "advance", ms: int(0, 5_000) }),
);

test("PBT (guard): at most GUARD_MAX - 1 automatic redirects per 120 s window since the last clear; a refusal lasts until a clear or the window passes", () => {
  forAll(array(guardEvent, { minLength: 1, maxLength: 40 }), (events) => {
    const s = createStorage();
    let now = 0;
    let win = null; // the model's open window: {first, times}
    let redirects = []; // times of automatic redirects since the last clear
    for (const e of events) {
      if (e.op === "advance") {
        now += e.ms;
      } else if (e.op === "clear") {
        clearGuard(s);
        win = null;
        redirects = [];
      } else {
        const passed = !win || now - win.first > GUARD_WINDOW_MS;
        const expected = passed || win.times.length < GUARD_MAX - 1;
        const st = readGuard(s);
        const ok = guardAllows(st, now);
        assert.equal(ok, expected, passed ? "a passed window allows" : "within the window, only GUARD_MAX - 1 redirects");
        if (ok) {
          writeGuard(s, guardRecord(st, now));
          if (passed) win = { first: now, times: [] };
          win.times.push(now);
          redirects.push(now);
        }
      }
      if (win) {
        assert.ok(win.times.length <= GUARD_MAX - 1, `${win.times.length} redirects in the window from ${win.first}`);
        assert.ok(win.times.every((t) => t - win.first <= GUARD_WINDOW_MS), "a window's redirects are within 120 s of its first");
      }
      // Windows don't overlap, so any 120 s span meets at most two of them.
      for (const t of redirects) {
        const inSpan = redirects.filter((u) => u >= t && u <= t + GUARD_WINDOW_MS).length;
        assert.ok(inSpan <= 2 * (GUARD_MAX - 1), `${inSpan} redirects within 120 s of ${t}: ${redirects}`);
      }
    }
    // Repeated attempts without any time passing trip the guard.
    let st = readGuard(s);
    let n = 0;
    while (guardAllows(st, now)) {
      st = guardRecord(st, now);
      n++;
      assert.ok(n < GUARD_MAX, "the guard never tripped");
    }
  });
});

// ---------------------------------------------------------------- safeHash

test("safeHash keeps the console's real routes", () => {
  for (const h of ["#/tools", "#/tools/lookup_account", "#/masking", "#/prompts", "#/prompts?tool=lookup_account",
    "#/audit?item=policy%23tool%23lookup_account", "#/audit?item=policy%23masking", "#/users",
    `#/audit?item=${encodeURIComponent("José ✓ 日本 🦉")}`]) {
    assert.equal(safeHash(h), h);
  }
});

test("safeHash rejects hostile and malformed hashes", () => {
  for (const h of [undefined, null, 42, {}, "", "#", "#/", "tools", "/tools", "#tools",
    "//evil.example", "#//evil.example", "#/\\evil", "#/%2F%2Fevil", "#/%5Cevil", "#/%2fevil",
    "javascript:alert(1)", "#/javascript:alert(1)", "#/tools/JavaScript%3Aalert(1)", "#/x?y=vbscript:1",
    "#/tools\n", "#/tools\u0000", "#/tools%0A", "#/tools%00", "#/tools%7F", "#/tools%C2%85",
    "#/tools%E2%80%A8", "#/tools%E2%80%AE", "#/tools/<script>", '#/tools/"x"', "#/tools/a b",
    "#/tools#again", "#/tools/%", "#/tools/%zz", "#/tools/%E0%A4%A", "#/tools/日本",
    `#/${"a".repeat(511)}`]) {
    assert.equal(safeHash(h), "#/tools", JSON.stringify(h));
  }
  assert.equal(safeHash(`#/${"a".repeat(510)}`).length, 512);
});

const NAME = `${ALNUM}_-.`;
const noControls = (s) => s.replace(/[\p{Cc}\u2028\u2029\u202A-\u202E\u2066-\u2069]/gu, "");
const validHash = oneOf(
  oneOf("#/tools", "#/masking", "#/prompts", "#/audit", "#/users"),
  gen((rng) => `#/tools/${encodeURIComponent(string({ minLength: 1, maxLength: 40, chars: NAME })(rng))}`),
  gen((rng) => `#/prompts?tool=${encodeURIComponent(string({ minLength: 1, maxLength: 40, chars: NAME })(rng))}`),
  gen((rng) => `#/audit?item=${encodeURIComponent(noControls(unicodeString({ minLength: 1, maxLength: 30 })(rng)) || "x")}`),
);

const CONTROL = [...Array(0x20).keys()].map((c) => String.fromCharCode(c)).join("") + "\u007F\u0085\u2028\u202E";
const hostileHash = oneOf(
  gen((rng) => `${validHash(rng)}${string({ minLength: 1, maxLength: 3, chars: CONTROL })(rng)}`),
  gen((rng) => `#/tools/${string({ minLength: 1, maxLength: 3, chars: CONTROL })(rng)}x`),
  gen((rng) => `#/tools/${encodeURIComponent(string({ minLength: 1, maxLength: 2, chars: CONTROL })(rng))}`),
  gen((rng) => `${oneOf("", "#", "#/", "/", "https:")(rng)}//${string({ minLength: 1, maxLength: 20 })(rng)}`),
  gen((rng) => `${oneOf("", "#/", "#/tools/", "#/x?y=")(rng)}${oneOf("javascript:", "JavaScript:", "javascript%3A", "vbscript:")(rng)}${string()(rng)}`),
  gen((rng) => `#/${string({ minLength: 511, maxLength: 700 })(rng)}`),
  gen((rng) => `#/%2F${string({ minLength: 1 })(rng)}`),
);

test("PBT (safeHash): valid routes are unchanged", () => {
  forAll(validHash, (h) => safeHash(h) === h);
});

test("PBT (safeHash): hostile strings become #/tools", () => {
  forAll(hostileHash, (h) => safeHash(h) === "#/tools");
});

test("PBT (safeHash): idempotent, always starts with #/ and fits in 512 characters", () => {
  forAll(oneOf(validHash, hostileHash, unicodeString({ maxLength: 40 }), string({ maxLength: 40, chars: "#/%?=:\\.aZ09" })), (h) => {
    const out = safeHash(h);
    assert.ok(out.startsWith("#/"), out);
    assert.ok(out.length <= 512);
    assert.ok(!/[\p{Cc}]/u.test(out));
    assert.ok(!out.startsWith("#//"));
    assert.equal(safeHash(out), out);
  });
});

// ---------------------------------------------------------------- signInReason

const codeErr = (code, message = "raw description") => Object.assign(new Error(message), { code });

test("signInReason has a fixed message for each known code", () => {
  const cases = [
    [codeErr("access_denied"), "Sign-in was cancelled or refused.", "access_denied"],
    [codeErr("state_mismatch"), "The sign-in response didn't match this browser's request.", "state_mismatch"],
    [codeErr("token_endpoint_400"), "The sign-in code couldn't be exchanged. Try again.", "token_endpoint_400"],
    [codeErr("token_endpoint_503"), "The sign-in code couldn't be exchanged. Try again.", "token_endpoint_503"],
    [codeErr("invalid_grant"), "The sign-in code couldn't be exchanged. Try again.", "invalid_grant"],
    [new TypeError("Failed to fetch"), "Can't reach the sign-in service.", "TypeError"],
    [{ code: "storage_unavailable" }, "This browser blocks session storage, which sign-in needs.", "storage_unavailable"],
    [{ code: "loop" }, "Sign-in didn't complete after several tries.", "loop"],
  ];
  for (const [err, message, code] of cases) {
    const r = signInReason(err);
    assert.equal(r.message, message, code);
    assert.equal(r.code, code);
    assert.equal(typeof r.title, "string");
    assert.ok(r.title.length > 0);
  }
  assert.equal(signInReason({ code: "loop" }).title, "Sign-in didn't complete");
});

test("signInReason gives the raw description for any other code", () => {
  assert.deepEqual(
    { ...signInReason(codeErr("server_error", "Cognito is down")) },
    { title: signInReason(codeErr("x")).title, message: "Cognito is down", code: "server_error" });
  assert.equal(signInReason({ code: "invalid_request", description: "Bad scope" }).message, "Bad scope");
});

test("signInReason copes with odd inputs", () => {
  for (const err of [undefined, null, "boom", 42, {}, new Error(""), { code: "" }]) {
    const r = signInReason(err);
    for (const k of ["title", "message", "code"]) {
      assert.equal(typeof r[k], "string", `${k} of ${String(err)}`);
      assert.ok(r[k].length > 0, `${k} of ${String(err)}`);
    }
  }
  assert.equal(signInReason(new Error("plain")).message, "plain");
  // A TypeError that carries a code (e.g.) is classified by the code.
  assert.equal(signInReason(Object.assign(new TypeError("x"), { code: "access_denied" })).code, "access_denied");
});

test("PBT signInReason: unknown codes give the raw description and the raw code, as strings", () => {
  const KNOWN = /^(access_denied|state_mismatch|invalid_grant|storage_unavailable|loop|token_endpoint_\d+)$/;
  forAll(record({ code: unicodeString({ minLength: 1, maxLength: 20 }), description: unicodeString({ minLength: 1, maxLength: 40 }) }), ({ code, description }) => {
    if (KNOWN.test(code)) return true;
    const r = signInReason(codeErr(code, description));
    return r.code === code && r.message === description && typeof r.title === "string";
  });
});
