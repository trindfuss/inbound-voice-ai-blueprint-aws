// Sign-in with Cognito managed login: authorization code + PKCE.
//
// Tokens live in this module's memory only, never in web storage; a reload
// signs in again (Cognito's own session on the login domain makes that one redirect). Only the
// PKCE verifier and the state, which must survive the redirect to the login page and back, are
// kept in sessionStorage, and removed as soon as the code is exchanged. The route to return to
// travels with them, checked by safeHash() when written and again when read.
//
// Failures carry a `code` for signInReason() (lib/signin.js): the callback's `error`,
// "state_mismatch", the token endpoint's `error` or "token_endpoint_<status>",
// "storage_unavailable", or a fetch TypeError as it is. `message` stays human-readable.

import {
  authorizeUrl, codeChallenge, codeExchangeBody, expiresSoon, groupsOf, jwtClaims, logoutUrl,
  parseCallback, randomString, refreshBody, revokeUrl, tokenUrl,
} from "./pkce.js";
import { safeHash, signInReason } from "./signin.js";

const PENDING = "owlbank-admin-pkce";
let cfg = null;
let tokens = null; // {idToken, accessToken, refreshToken, claims}
let refreshing = null;

export function configure(config) {
  cfg = config;
}

function codedError(code, message) {
  return Object.assign(new Error(message || code), { code });
}

const storageUnavailable = () => codedError("storage_unavailable", signInReason({ code: "storage_unavailable" }).message);

function setTokens(t) {
  tokens = { ...tokens, ...t, claims: jwtClaims(t.idToken) };
}

async function tokenRequest(body) {
  const res = await fetch(tokenUrl(cfg), {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
    credentials: "omit",
    cache: "no-store",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data || !data.id_token) {
    const code = (data && typeof data.error === "string" && data.error) || `token_endpoint_${res.status}`;
    throw codedError(code, (data && data.error_description) || code);
  }
  return data;
}

/** Go to the managed login page. */
export async function signIn() {
  const verifier = randomString(crypto, 64);
  const state = randomString(crypto, 43);
  try {
    sessionStorage.setItem(PENDING, JSON.stringify({ verifier, state, hash: safeHash(location.hash) }));
  } catch {
    throw storageUnavailable(); // the sessionStorage accessor or setItem threw: never redirect
  }
  location.assign(authorizeUrl(cfg, { state, challenge: await codeChallenge(crypto, verifier) }));
}

/**
 * On page load: finish a sign-in redirect if this is one. Returns "signed-in", "none" (not a
 * redirect) or throws with the reason.
 */
export async function completeSignIn() {
  let pending = null;
  let storageOk = true;
  try {
    const raw = sessionStorage.getItem(PENDING);
    try {
      pending = JSON.parse(raw || "null");
    } catch {
      pending = null; // malformed: treated as missing, so the callback is a state mismatch
    }
    if (pending === null || typeof pending !== "object") pending = null;
  } catch {
    storageOk = false;
  }
  const result = parseCallback(location.search, pending && pending.state);
  if (!result) return "none";
  try {
    sessionStorage.removeItem(PENDING);
  } catch {
    storageOk = false;
  }
  // Drop the code and state from the address bar and history before anything else.
  history.replaceState(null, "", location.pathname + safeHash(pending && pending.hash));
  if (!storageOk) throw storageUnavailable();
  if (result.error) throw codedError(result.error, result.description);
  const data = await tokenRequest(codeExchangeBody(cfg, { code: result.code, verifier: pending.verifier }));
  setTokens({ idToken: data.id_token, accessToken: data.access_token, refreshToken: data.refresh_token });
  return "signed-in";
}

/** Use the refresh token; resolves true when new tokens were issued. */
export function refresh() {
  if (!tokens || !tokens.refreshToken) return Promise.resolve(false);
  refreshing ??= tokenRequest(refreshBody(cfg, tokens.refreshToken))
    .then((data) => {
      setTokens({ idToken: data.id_token, accessToken: data.access_token });
      return true;
    })
    .catch(() => false)
    .finally(() => {
      refreshing = null;
    });
  return refreshing;
}

/** A valid ID token, refreshed when it expires within a minute. Rejects when signed out. */
export async function idToken() {
  if (!tokens) throw new Error("signed out");
  if (expiresSoon(tokens.claims, Date.now()) && !(await refresh())) throw new Error("session ended");
  return tokens.idToken;
}

export function isSignedIn() {
  return tokens !== null;
}

/** The signed-in user: {email, username, sub, groups}. */
export function user() {
  const c = (tokens && tokens.claims) || {};
  return { email: c.email || null, username: c["cognito:username"] || null, sub: c.sub || null, groups: groupsOf(c) };
}

/** Revoke the refresh token (best effort), forget the tokens and end the Cognito session. */
export async function signOut() {
  const refreshToken = tokens && tokens.refreshToken;
  tokens = null;
  if (refreshToken) {
    await fetch(revokeUrl(cfg), {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ token: refreshToken, client_id: cfg.clientId }).toString(),
      credentials: "omit",
    }).catch(() => {});
  }
  location.assign(logoutUrl(cfg));
}
