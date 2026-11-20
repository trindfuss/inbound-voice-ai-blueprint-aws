// Sign-in helpers. Pure apart from the guard's storage, which is always read and
// written inside try/catch: a browser that blocks session storage gets null, never a throw.
//
// The loop guard counts automatic redirects to the login page that haven't been followed by a
// successful sign-in. Within GUARD_WINDOW_MS of the first, only GUARD_MAX - 1 are allowed; the
// next attempt is refused, and stays refused until clearGuard() (a success or Try again) or
// until the window has passed, when the next redirect starts a new window.

export const GUARD_KEY = "owlbank-admin-signin-guard";
export const GUARD_MAX = 3;
export const GUARD_WINDOW_MS = 120_000;

const isCount = (n) => Number.isSafeInteger(n) && n >= 0;
const validState = (s) => s !== null && typeof s === "object" && !Array.isArray(s)
  && isCount(s.count) && Number.isFinite(s.first);

/** true when an automatic redirect may start. state: {count, first} | null. A passed window allows. */
export function guardAllows(state, now) {
  if (!validState(state) || now - state.first > GUARD_WINDOW_MS) return true;
  return state.count < GUARD_MAX - 1;
}

/** The state after an automatic redirect at `now`. A new window starts once the old one has passed. */
export function guardRecord(state, now) {
  if (!validState(state) || now - state.first > GUARD_WINDOW_MS) return { count: 1, first: now };
  return { count: state.count + 1, first: state.first };
}

// `storage` is a Storage, a function returning one, or omitted for sessionStorage. Resolving it
// happens inside the caller's try, because the sessionStorage accessor itself can throw.
function resolve(storage) {
  const s = typeof storage === "function" ? storage() : storage === undefined ? globalThis.sessionStorage : storage;
  if (!s) throw new Error("no storage");
  return s;
}

/** The stored guard state, or null when there is none, it is malformed, or storage throws. */
export function readGuard(storage) {
  try {
    const st = JSON.parse(resolve(storage).getItem(GUARD_KEY));
    return validState(st) ? { count: st.count, first: st.first } : null;
  } catch {
    return null;
  }
}

/** Store the state; returns it, or null when storage throws. */
export function writeGuard(storage, state) {
  try {
    resolve(storage).setItem(GUARD_KEY, JSON.stringify({ count: state.count, first: state.first }));
    return state;
  } catch {
    return null;
  }
}

/** Remove the guard; true, or null when storage throws. */
export function clearGuard(storage) {
  try {
    resolve(storage).removeItem(GUARD_KEY);
    return true;
  } catch {
    return null;
  }
}

// What a hash may contain, as the browser reports it (non-ASCII is percent-encoded): the URL
// unreserved and sub-delim characters, ":", "@", "/", "?" and %XX escapes. No space, quotes,
// angle brackets, backslash or a second "#".
const HASH_CHARS = /^#\/(?:[A-Za-z0-9\-._~!$&'()*+,;=:@/?]|%[0-9A-Fa-f]{2})+$/;
// Checked on the decoded text, so escapes can't smuggle them in.
const UNSAFE_DECODED = /[\p{Cc}\u2028\u2029\u202A-\u202E\u2066-\u2069]/u;
const SCHEME = /(?:java|vb)script:/i;
const FALLBACK = "#/tools";

/**
 * hash if it is a safe in-app route ("#/" + 1..510 allowed characters, not "#//" or "#/\",
 * no control characters, no script scheme, also after percent-decoding), else "#/tools".
 */
export function safeHash(hash) {
  if (typeof hash !== "string" || hash.length < 3 || hash.length > 512 || !HASH_CHARS.test(hash)) return FALLBACK;
  let decoded;
  try {
    decoded = decodeURIComponent(hash);
  } catch {
    return FALLBACK;
  }
  if (UNSAFE_DECODED.test(decoded) || SCHEME.test(decoded) || /^#\/[/\\]/.test(decoded)) return FALLBACK;
  return hash;
}

const TITLE = "Couldn't sign you in";
const EXCHANGE = "The sign-in code couldn't be exchanged. Try again.";
const REASONS = {
  access_denied: "Sign-in was cancelled or refused.",
  state_mismatch: "The sign-in response didn't match this browser's request.", // pkce.js parseCallback
  invalid_grant: EXCHANGE,
  TypeError: "Can't reach the sign-in service.",
  storage_unavailable: "This browser blocks session storage, which sign-in needs.",
  loop: "Sign-in didn't complete after several tries.",
};

/**
 * {title, message, code} for the sign-in error page. Known codes get fixed text; any other code
 * gets the raw description. Callers render all three as text only.
 */
export function signInReason(err) {
  const e = err !== null && typeof err === "object" ? err : {};
  let code = typeof e.code === "string" && e.code ? e.code : "";
  if (!code && err instanceof TypeError) code = "TypeError";
  const raw = [e.description, e.message, typeof err === "string" ? err : ""]
    .find((x) => typeof x === "string" && x !== "");
  if (!code) code = "unknown";
  const title = code === "loop" ? "Sign-in didn't complete" : TITLE;
  let message;
  if (Object.hasOwn(REASONS, code)) message = REASONS[code];
  else if (/^token_endpoint_\d+$/.test(code)) message = EXCHANGE;
  else message = raw || "Sign-in returned an error.";
  return { title, message, code };
}
