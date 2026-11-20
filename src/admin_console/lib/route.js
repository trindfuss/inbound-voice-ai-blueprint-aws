// Hash routes: #/<section>, #/<section>/<arg>, #/<section>?<params>; for example #/tools/<name>,
// #/prompts?tool=<name>, #/audit?item=<key>. Pure. The section ids come from SECTIONS in app.js;
// VIEWS is the default for callers that don't pass them, and a test keeps the two equal.

export const VIEWS = ["tools", "masking", "prompts", "slots", "redaction", "audit", "users"];

/** Parse a location hash into {view, arg, params}; unknown routes fall back to the first id. */
export function parseRoute(hash, ids = VIEWS) {
  const raw = String(hash || "").replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?", 2);
  const parts = path.split("/").filter(Boolean).map((p) => {
    try {
      return decodeURIComponent(p);
    } catch {
      return p;
    }
  });
  const known = ids.includes(parts[0]);
  const view = known ? parts[0] : ids[0];
  return { view, arg: known ? parts[1] ?? null : null, params: new URLSearchParams(query) };
}
