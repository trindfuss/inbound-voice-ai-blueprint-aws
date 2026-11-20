// OAuth 2.0 authorization code + PKCE helpers for Cognito managed login.
// Pure functions: no DOM, no storage. The caller passes `crypto` (window.crypto or Node's
// globalThis.crypto) so the tests run under `node --test`.

const VERIFIER_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
export const SCOPES = ["openid", "email", "profile"];

/** base64url without padding (RFC 4648 section 5) of a byte array. */
export function base64url(bytes) {
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/** A random string of `length` characters from the RFC 7636 unreserved set. */
export function randomString(crypto, length = 64) {
  if (length < 43 || length > 128) throw new RangeError("a PKCE verifier is 43 to 128 characters");
  const out = [];
  // Rejection sampling: 66 characters, so bytes >= 198 (3 * 66) are skipped to avoid bias.
  while (out.length < length) {
    const buf = crypto.getRandomValues(new Uint8Array(length * 2));
    for (const b of buf) {
      if (b < 198 && out.length < length) out.push(VERIFIER_CHARS[b % VERIFIER_CHARS.length]);
    }
  }
  return out.join("");
}

/** The S256 code challenge: base64url(SHA-256(ASCII(verifier))). */
export async function codeChallenge(crypto, verifier) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  return base64url(new Uint8Array(digest));
}

function origin(loginDomain) {
  return loginDomain.startsWith("https://") ? loginDomain.replace(/\/+$/, "") : `https://${loginDomain}`;
}

/** The managed login's authorize URL. */
export function authorizeUrl(cfg, { state, challenge }) {
  const q = new URLSearchParams({
    response_type: "code",
    client_id: cfg.clientId,
    redirect_uri: cfg.redirectUri,
    scope: SCOPES.join(" "),
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
  });
  return `${origin(cfg.loginDomain)}/oauth2/authorize?${q}`;
}

export function tokenUrl(cfg) {
  return `${origin(cfg.loginDomain)}/oauth2/token`;
}

export function revokeUrl(cfg) {
  return `${origin(cfg.loginDomain)}/oauth2/revoke`;
}

/** The managed login's logout URL; it ends the Cognito session and returns to the console. */
export function logoutUrl(cfg) {
  const q = new URLSearchParams({ client_id: cfg.clientId, logout_uri: cfg.redirectUri });
  return `${origin(cfg.loginDomain)}/logout?${q}`;
}

/** Form body for exchanging the authorization code (public client: no secret). */
export function codeExchangeBody(cfg, { code, verifier }) {
  return new URLSearchParams({
    grant_type: "authorization_code",
    client_id: cfg.clientId,
    code,
    redirect_uri: cfg.redirectUri,
    code_verifier: verifier,
  }).toString();
}

export function refreshBody(cfg, refreshToken) {
  return new URLSearchParams({
    grant_type: "refresh_token",
    client_id: cfg.clientId,
    refresh_token: refreshToken,
  }).toString();
}

/**
 * Read the redirect's query string. Returns null when it isn't an OAuth redirect, else
 * {code} or {error}. The state must equal the one sent; anything else is an error.
 */
export function parseCallback(search, expectedState) {
  const q = new URLSearchParams(search);
  if (!q.has("code") && !q.has("error")) return null;
  if (!expectedState || q.get("state") !== expectedState) {
    return { error: "state_mismatch", description: "The sign-in response didn't match this browser's request." };
  }
  if (q.has("error")) {
    return { error: q.get("error"), description: q.get("error_description") || "" };
  }
  return { code: q.get("code") };
}

/** The claims of a JWT (no signature check: API Gateway verifies the token; this is for display). */
export function jwtClaims(token) {
  const part = String(token || "").split(".")[1];
  if (!part) return {};
  const b64 = part.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(part.length / 4) * 4, "=");
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return {};
  }
}

/** The console groups in a token's `cognito:groups` claim. */
export function groupsOf(claims) {
  const raw = claims && claims["cognito:groups"];
  return Array.isArray(raw) ? raw.filter((g) => typeof g === "string") : [];
}

/** True when the token expires within `skewSeconds` of `nowMs`. */
export function expiresSoon(claims, nowMs, skewSeconds = 60) {
  const exp = Number(claims && claims.exp);
  return !Number.isFinite(exp) || exp * 1000 - skewSeconds * 1000 <= nowMs;
}
