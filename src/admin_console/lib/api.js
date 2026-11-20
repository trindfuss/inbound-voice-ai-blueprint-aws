// The console's API client. Same origin: requests go to `${apiBase}${path}` on
// the console's own CloudFront domain, which forwards /api/* to API Gateway. The ID token goes
// in the Authorization header (the Cognito authorizer reads it; its claims carry the groups).

/** A failed API call, mapped for display (see mapError). */
export class ApiError extends Error {
  constructor(info) {
    super(info.message);
    Object.assign(this, info);
  }
}

const TITLES = {
  invalid: "Not saved: a value is invalid",
  conflict: "Not saved: someone else changed this",
  auth: "Your session has ended",
  forbidden: "Not allowed",
  not_found: "Not found",
  throttled: "Too many requests",
  unavailable: "Not available",
  server: "Something went wrong",
  network: "Can't reach the console's API",
};

/**
 * Map a status and JSON body ({error, message?, field?, current?}) to
 * {status, kind, code, title, message, field, current}. Pure.
 *
 * kind: invalid (400, 413), auth (401: sign in again), forbidden (403), not_found (404),
 * conflict (409 with `current`: show it next to the editor's values), throttled (429),
 * unavailable (503: e.g. the tables aren't seeded), server (other 5xx).
 */
export function mapError(status, body) {
  const b = body && typeof body === "object" ? body : {};
  const code = typeof b.error === "string" ? b.error : null;
  let kind;
  if (status === 400 || status === 413 || status === 405) kind = "invalid";
  else if (status === 401) kind = "auth";
  else if (status === 403) kind = "forbidden";
  else if (status === 404) kind = "not_found";
  else if (status === 409) kind = b.current !== undefined ? "conflict" : "invalid";
  else if (status === 429) kind = "throttled";
  else if (status === 503) kind = "unavailable";
  else if (status === 0) kind = "network";
  else kind = "server";
  let message = typeof b.message === "string" ? b.message : null;
  if (!message) {
    message = {
      auth: "Sign in again to continue.",
      forbidden: "Your group can't do this. Ask an admin if you need access.",
      not_found: "It may have been removed.",
      throttled: "Wait a moment and try again.",
      network: "Check your connection and try again.",
      server: "The change may not have been saved. Reload to check.",
      unavailable: "The configuration tables aren't ready. Has the stack been seeded?",
    }[kind] ?? "The request was refused.";
  }
  if (status === 403 && code === "self_delete") message = "You can't delete your own user.";
  const field = typeof b.field === "string" ? b.field : null;
  if (kind === "invalid" && field) message = `${field}: ${message}`;
  return { status, kind, code, title: TITLES[kind], message, field, current: b.current };
}

/** Build `${base}${path}?query` with each path segment and query value encoded. */
export function apiUrl(base, segments, query) {
  const path = segments.map((s) => encodeURIComponent(String(s))).join("/");
  const q = query ? new URLSearchParams(Object.entries(query).filter(([, v]) => v !== undefined && v !== null && v !== "")) : null;
  const qs = q && [...q].length ? `?${q}` : "";
  return `${base.replace(/\/+$/, "")}/${path}${qs}`;
}

/**
 * Make a client. `getToken()` resolves to a valid ID token (refreshing it when needed);
 * `onAuthLost()` is called once a 401 survives one refresh (the console returns to sign-in).
 */
export function createClient({ base, getToken, refresh, onAuthLost, fetchImpl = globalThis.fetch.bind(globalThis) }) {
  async function call(method, segments, { query, body } = {}, retried = false) {
    const token = await getToken();
    const headers = { Authorization: token, Accept: "application/json" };
    const init = { method, headers, credentials: "omit", cache: "no-store" };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetchImpl(apiUrl(base, segments, query), init);
    } catch {
      throw new ApiError(mapError(0, null));
    }
    let data = null;
    const text = await res.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    if (res.ok) return data ?? {};
    if (res.status === 401 && !retried && refresh) {
      const ok = await refresh().catch(() => false);
      if (ok) return call(method, segments, { query, body }, true);
    }
    const err = new ApiError(mapError(res.status, data));
    if (err.kind === "auth" && onAuthLost) onAuthLost();
    throw err;
  }
  return {
    get: (segments, query) => call("GET", segments, { query }),
    put: (segments, body) => call("PUT", segments, { body }),
    post: (segments, body) => call("POST", segments, { body }),
    patch: (segments, body) => call("PATCH", segments, { body }),
    del: (segments) => call("DELETE", segments),
  };
}
