import assert from "node:assert/strict";
import { test } from "node:test";

import {
  authorizeUrl, base64url, codeChallenge, codeExchangeBody, expiresSoon, groupsOf, jwtClaims,
  logoutUrl, parseCallback, randomString, refreshBody,
} from "../lib/pkce.js";

const cfg = {
  clientId: "client123",
  loginDomain: "owlbank-dev-admin-111111111111.auth.us-east-1.amazoncognito.com",
  redirectUri: "https://d111.cloudfront.net/",
};

test("S256 challenge matches RFC 7636 appendix B", async () => {
  const verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
  assert.equal(await codeChallenge(globalThis.crypto, verifier), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
});

test("base64url has no padding or +/ characters", () => {
  assert.equal(base64url(new Uint8Array([251, 255, 191])), "-_-_");
  assert.equal(base64url(new Uint8Array([1])), "AQ");
});

test("verifiers are unreserved characters of the requested length and differ", () => {
  const seen = new Set();
  for (let i = 0; i < 200; i++) {
    const v = randomString(globalThis.crypto, 64);
    assert.match(v, /^[A-Za-z0-9\-._~]{64}$/);
    seen.add(v);
  }
  assert.equal(seen.size, 200);
  assert.throws(() => randomString(globalThis.crypto, 42), RangeError);
  assert.throws(() => randomString(globalThis.crypto, 129), RangeError);
});

test("challenge is always 43 base64url characters (property over random verifiers)", async () => {
  for (let i = 0; i < 100; i++) {
    const len = 43 + (i % 86);
    const c = await codeChallenge(globalThis.crypto, randomString(globalThis.crypto, len));
    assert.match(c, /^[A-Za-z0-9_-]{43}$/);
  }
});

test("authorize URL asks for the code flow with S256 and the console's redirect", () => {
  const url = new URL(authorizeUrl(cfg, { state: "st", challenge: "ch" }));
  assert.equal(url.origin, `https://${cfg.loginDomain}`);
  assert.equal(url.pathname, "/oauth2/authorize");
  const q = Object.fromEntries(url.searchParams);
  assert.deepEqual(q, {
    response_type: "code", client_id: "client123", redirect_uri: cfg.redirectUri,
    scope: "openid email profile", state: "st", code_challenge: "ch", code_challenge_method: "S256",
  });
});

test("token bodies: code exchange with the verifier, refresh; never a client secret", () => {
  const code = new URLSearchParams(codeExchangeBody(cfg, { code: "c0de", verifier: "v" }));
  assert.equal(code.get("grant_type"), "authorization_code");
  assert.equal(code.get("code_verifier"), "v");
  assert.equal(code.get("redirect_uri"), cfg.redirectUri);
  assert.equal(code.has("client_secret"), false);
  const ref = new URLSearchParams(refreshBody(cfg, "r"));
  assert.equal(ref.get("grant_type"), "refresh_token");
  assert.equal(ref.get("refresh_token"), "r");
});

test("logout URL returns to the console", () => {
  const url = new URL(logoutUrl(cfg));
  assert.equal(url.pathname, "/logout");
  assert.equal(url.searchParams.get("logout_uri"), cfg.redirectUri);
  assert.equal(url.searchParams.get("client_id"), "client123");
});

test("callback: code only with the matching state", () => {
  assert.equal(parseCallback("", "s"), null);
  assert.equal(parseCallback("?tab=1", "s"), null);
  assert.deepEqual(parseCallback("?code=abc&state=s", "s"), { code: "abc" });
  assert.equal(parseCallback("?code=abc&state=other", "s").error, "state_mismatch");
  assert.equal(parseCallback("?code=abc&state=s", null).error, "state_mismatch");
  assert.equal(parseCallback("?code=abc", "s").error, "state_mismatch");
  assert.deepEqual(parseCallback("?error=access_denied&error_description=no&state=s", "s"),
    { error: "access_denied", description: "no" });
});

function jwt(claims) {
  const enc = (o) => base64url(new TextEncoder().encode(JSON.stringify(o)));
  return `${enc({ alg: "RS256" })}.${enc(claims)}.sig`;
}

test("claims, groups and expiry from an ID token", () => {
  const claims = jwtClaims(jwt({ sub: "u1", email: "é@example.com", "cognito:groups": ["admin-editor"], exp: 1000 }));
  assert.equal(claims.email, "é@example.com");
  assert.deepEqual(groupsOf(claims), ["admin-editor"]);
  assert.deepEqual(groupsOf({}), []);
  assert.equal(expiresSoon(claims, 1000 * 1000 - 61_000), false);
  assert.equal(expiresSoon(claims, 1000 * 1000 - 59_000), true);
  assert.equal(expiresSoon({}, 0), true);
  assert.deepEqual(jwtClaims("garbage"), {});
});
