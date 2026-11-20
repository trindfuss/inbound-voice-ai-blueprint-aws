// lib/auth.js: the pending route goes through safeHash both ways, and sign-in
// failures carry a typed `code` for signInReason. PBT: the redirect round trip.
// Tokens stay in memory: no storage value ever holds one.
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import * as auth from "../lib/auth.js";
import { safeHash, signInReason } from "../lib/signin.js";
import { install, uninstall } from "./fakedom.js";
import { ALNUM, forAll, gen, int, oneOf, string, unicodeString } from "./gen.js";

const PENDING = "owlbank-admin-pkce";
const cfg = {
  clientId: "client123",
  loginDomain: "owlbank-dev-admin-111111111111.auth.us-east-1.amazoncognito.com",
  redirectUri: "https://d111.cloudfront.net/",
};
const PATH = "/";
const STATE_TEXT = "The sign-in response didn't match this browser's request.";

const b64url = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
const jwt = (claims) => `${b64url({ alg: "RS256" })}.${b64url(claims)}.sig`;
const ID_TOKEN = jwt({ sub: "u1", email: "a@example.com", exp: 4_000_000_000 });
const ACCESS_TOKEN = jwt({ sub: "u1", token_use: "access", exp: 4_000_000_000 });
const REFRESH_TOKEN = "refresh-opaque-token-value";

let dom;
let assigned;
let replaced;
let fetches;
let fetchImpl;
let saved;

function reply(status, body) {
  return async () => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => {
      if (body === undefined) throw new SyntaxError("no body");
      return body;
    },
  });
}
const okTokens = reply(200, { id_token: ID_TOKEN, access_token: ACCESS_TOKEN, refresh_token: REFRESH_TOKEN });

function setup(opts) {
  dom = install(opts);
  assigned = [];
  replaced = [];
  fetches = [];
  fetchImpl = okTokens;
  const location = {
    hash: "",
    search: "",
    pathname: PATH,
    assign: (url) => { assigned.push(String(url)); },
  };
  const history = { replaceState: (_s, _t, url) => { replaced.push(String(url)); } };
  const fetch = (url, init) => { fetches.push({ url: String(url), init }); return fetchImpl(url, init); };
  for (const [k, v] of Object.entries({ location, history, fetch })) {
    Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  }
  auth.configure(cfg);
}

beforeEach(() => {
  saved = Object.fromEntries(["location", "history", "fetch"].map((k) => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));
  setup();
});

afterEach(() => {
  uninstall();
  for (const [k, d] of Object.entries(saved)) {
    if (d) Object.defineProperty(globalThis, k, d);
    else delete globalThis[k];
  }
});

const pending = () => JSON.parse(dom.sessionStorage.getItem(PENDING));

/** signIn() from `hash`, then come back with `search` built from the stored state. */
async function roundTrip(hash, makeSearch = (state) => `?code=abc&state=${encodeURIComponent(state)}`) {
  location.hash = hash;
  location.search = "";
  await auth.signIn();
  const { state } = pending();
  location.hash = "";
  location.search = makeSearch(state);
  return state;
}

function noTokensStored() {
  for (const s of [dom.sessionStorage, dom.localStorage]) {
    for (const k of s.keys()) {
      const v = s.getItem(k);
      assert.ok(!/eyJ[\w-]+\./.test(v), `${k} holds a JWT`);
      assert.ok(!v.includes(REFRESH_TOKEN), `${k} holds the refresh token`);
    }
  }
}

// ---------------------------------------------------------------- signIn

test("signIn stores the verifier, state and a safe hash, then goes to the login page", async () => {
  location.hash = "#/prompts?tool=lookup";
  await auth.signIn();
  const p = pending();
  assert.equal(p.hash, "#/prompts?tool=lookup");
  assert.equal(typeof p.verifier, "string");
  assert.equal(typeof p.state, "string");
  assert.equal(assigned.length, 1);
  const url = new URL(assigned[0]);
  assert.equal(url.pathname, "/oauth2/authorize");
  assert.equal(url.searchParams.get("state"), p.state);
});

test("signIn stores #/tools in place of a hostile hash", async () => {
  for (const h of ["", "#//evil.example", "#/javascript:alert(1)", `#/a${String.fromCharCode(1)}b`, "#/a%0Ab"]) {
    location.hash = h;
    await auth.signIn();
    assert.equal(pending().hash, "#/tools", JSON.stringify(h));
  }
});

test("signIn throws storage_unavailable and does not redirect when setItem throws", async () => {
  dom.sessionStorage.throws = true;
  await assert.rejects(auth.signIn(), (e) => e.code === "storage_unavailable" && e instanceof Error && e.message !== "");
  assert.deepEqual(assigned, []);
  assert.equal(signInReason(await auth.signIn().catch((e) => e)).code, "storage_unavailable");
});

test("signIn throws storage_unavailable when the sessionStorage accessor itself throws", async () => {
  uninstall();
  setup({ storageThrows: "sessionStorage" });
  await assert.rejects(auth.signIn(), (e) => e.code === "storage_unavailable");
  assert.deepEqual(assigned, []);
});

// ---------------------------------------------------------------- completeSignIn

test("completeSignIn returns none when the page is not a callback", async () => {
  location.search = "?foo=1";
  assert.equal(await auth.completeSignIn(), "none");
  assert.deepEqual(replaced, []);
  assert.deepEqual(fetches, []);
});

test("a successful callback signs in, restores the route, removes the pending record and stores no token", async () => {
  await roundTrip("#/tools/lookup_account");
  assert.equal(await auth.completeSignIn(), "signed-in");
  assert.ok(auth.isSignedIn());
  assert.deepEqual(replaced, [`${PATH}#/tools/lookup_account`]);
  assert.equal(dom.sessionStorage.getItem(PENDING), null);
  assert.equal(fetches.length, 1);
  assert.equal(assigned.length, 1, "only the signIn redirect");
  noTokensStored();
});

test("an error callback throws its code and description, and never redirects", async () => {
  await roundTrip("#/audit", (state) => `?error=access_denied&error_description=User+cancelled&state=${state}`);
  const err = await auth.completeSignIn().catch((e) => e);
  assert.ok(err instanceof Error);
  assert.equal(err.code, "access_denied");
  assert.equal(err.message, "User cancelled");
  assert.equal(signInReason(err).code, "access_denied");
  assert.equal(dom.sessionStorage.getItem(PENDING), null);
  assert.deepEqual(replaced, [`${PATH}#/audit`]);
  assert.deepEqual(fetches, []);
  assert.equal(assigned.length, 1, "only the signIn redirect");
});

test("an error callback without a description uses the code as the message", async () => {
  await roundTrip("#/audit", (state) => `?error=server_error&state=${state}`);
  const err = await auth.completeSignIn().catch((e) => e);
  assert.equal(err.code, "server_error");
  assert.equal(err.message, "server_error");
});

test("a state mismatch throws state_mismatch with the pkce.js text", async () => {
  await roundTrip("#/audit", () => "?code=abc&state=forged");
  const err = await auth.completeSignIn().catch((e) => e);
  assert.equal(err.code, "state_mismatch");
  assert.equal(err.message, STATE_TEXT);
  assert.equal(dom.sessionStorage.getItem(PENDING), null);
  assert.deepEqual(fetches, []);
});

test("a callback with no pending record is a state mismatch", async () => {
  location.search = "?code=abc&state=whatever";
  const err = await auth.completeSignIn().catch((e) => e);
  assert.equal(err.code, "state_mismatch");
  assert.deepEqual(replaced, [`${PATH}#/tools`]);
});

test("a malformed pending record is a state mismatch, not a crash", async () => {
  dom.sessionStorage.setItem(PENDING, "{not json");
  location.search = "?code=abc&state=x";
  const err = await auth.completeSignIn().catch((e) => e);
  assert.equal(err.code, "state_mismatch");
  assert.equal(dom.sessionStorage.getItem(PENDING), null);
});

test("a callback when sessionStorage throws gives storage_unavailable", async () => {
  await roundTrip("#/audit");
  dom.sessionStorage.throws = true;
  const err = await auth.completeSignIn().catch((e) => e);
  assert.equal(err.code, "storage_unavailable");
  assert.deepEqual(fetches, []);
  assert.deepEqual(replaced, [`${PATH}#/tools`], "code and state still leave the address bar");
});

test("a callback when the sessionStorage accessor throws gives storage_unavailable", async () => {
  uninstall();
  setup({ storageThrows: "sessionStorage" });
  location.search = "?code=abc&state=x";
  const err = await auth.completeSignIn().catch((e) => e);
  assert.equal(err.code, "storage_unavailable");
});

test("a non-callback page with blocked storage is still none", async () => {
  uninstall();
  setup({ storageThrows: "sessionStorage" });
  assert.equal(await auth.completeSignIn(), "none");
});

for (const [status, body, code] of [
  [400, { error: "invalid_grant" }, "invalid_grant"],
  [400, { error: "invalid_client" }, "invalid_client"],
  [500, undefined, "token_endpoint_500"],
  [503, {}, "token_endpoint_503"],
  [200, { access_token: "x" }, "token_endpoint_200"],
]) {
  test(`the token endpoint answering ${status} ${JSON.stringify(body)} throws ${code}`, async () => {
    await roundTrip("#/audit");
    fetchImpl = reply(status, body);
    const err = await auth.completeSignIn().catch((e) => e);
    assert.ok(err instanceof Error);
    assert.equal(err.code, code);
    assert.ok(err.message !== "");
    assert.equal(signInReason(err).code, code);
    assert.equal(dom.sessionStorage.getItem(PENDING), null);
    assert.equal(assigned.length, 1, "only the signIn redirect");
  });
}

test("a network failure surfaces as the fetch TypeError", async () => {
  await roundTrip("#/audit");
  fetchImpl = async () => { throw new TypeError("Failed to fetch"); };
  const err = await auth.completeSignIn().catch((e) => e);
  assert.ok(err instanceof TypeError);
  assert.equal(signInReason(err).code, "TypeError");
  assert.equal(assigned.length, 1);
});

// ---------------------------------------------------------------- PBT

const NAME = string({ minLength: 1, maxLength: 20, chars: `${ALNUM}-_.` });
const BAD_CP = /[\p{Cc}\u2028\u2029\u202A-\u202E\u2066-\u2069]/gu;
const textItem = gen((rng) => unicodeString({ minLength: 1, maxLength: 12 })(rng).replace(BAD_CP, "") || "x");
const validHash = oneOf(
  gen((rng) => `#/tools/${NAME(rng)}`),
  gen((rng) => `#/prompts?tool=${NAME(rng)}`),
  gen((rng) => `#/audit?item=${encodeURIComponent(textItem(rng))}`),
);
const CTRL = gen((rng) => String.fromCharCode(oneOf(int(0, 31), 127)(rng)));
const hostileHash = oneOf(
  gen((rng) => `#/tools/${NAME(rng)}${CTRL(rng)}`),
  gen((rng) => `#//${NAME(rng)}.example`),
  gen((rng) => `#/${oneOf("javascript:", "JavaScript:", "vbscript:")(rng)}${NAME(rng)}`),
  gen((rng) => `#/${"a".repeat(int(511, 700)(rng))}`),
  gen((rng) => `#/audit?item=%${oneOf("00", "0A", "1F", "7F")(rng)}${NAME(rng)}`),
  gen((rng) => `#/audit?item=${unicodeString({ minLength: 1, maxLength: 6 })(rng).replace(/[\x20-\x7E]/g, "")}${String.fromCharCode(0x2028)}`),
  oneOf("", "#", "#/", "#tools", "/tools"),
);
const anyHash = oneOf(validHash, hostileHash, gen((rng) => `#/${unicodeString({ maxLength: 20 })(rng)}`));

test("a valid route survives signIn -> pending -> completeSignIn -> replaceState unchanged", () =>
  forAll(validHash, async (h) => {
    assert.equal(safeHash(h), h, "the generator makes valid routes");
    replaced = [];
    await roundTrip(h);
    assert.equal(pending().hash, h);
    assert.equal(await auth.completeSignIn(), "signed-in");
    assert.deepEqual(replaced, [PATH + h]);
    assert.equal(dom.sessionStorage.getItem(PENDING), null);
    noTokensStored();
  }, { runs: 100 }));

test("a hostile hash comes back as #/tools", () =>
  forAll(hostileHash, async (h) => {
    replaced = [];
    await roundTrip(h);
    assert.equal(pending().hash, "#/tools");
    await auth.completeSignIn();
    assert.deepEqual(replaced, [`${PATH}#/tools`]);
  }, { runs: 100 }));

test("any hash, even one forged into the pending record, is restored through safeHash", () =>
  forAll(anyHash, async (h) => {
    replaced = [];
    await roundTrip("#/tools");
    // A record tampered with after it was written is checked again when read.
    dom.sessionStorage.setItem(PENDING, JSON.stringify({ ...pending(), hash: h }));
    await auth.completeSignIn();
    assert.equal(replaced.length, 1);
    assert.equal(replaced[0], PATH + safeHash(h));
    assert.ok(replaced[0].startsWith(`${PATH}#/`));
  }, { runs: 150 }));
