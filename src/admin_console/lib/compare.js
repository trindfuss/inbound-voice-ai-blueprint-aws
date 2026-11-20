// The conflict panel's field-by-field comparison. Pure, no DOM.
//
// On a 409 the server sends `current`, its view of the item, while the console holds `yours`,
// the body it tried to save. The shapes differ (a PUT body against a view), so this isn't a deep
// diff of both: it walks the leaf paths of `yours` and looks each one up in `current`.

/** A view's {value, source} wrapper (for example mask.after_ms, config_routes.py `_tool_view`). */
export function isWrapper(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v)
    && Object.keys(v).length === 2 && Object.hasOwn(v, "value") && typeof v.source === "string";
}

const isPlain = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

/** Strip every {value, source} wrapper in v, at any depth (arrays and objects included). */
function unwrap(v) {
  while (isWrapper(v)) v = v.value;
  if (Array.isArray(v)) return v.map(unwrap);
  if (isPlain(v)) return Object.fromEntries(Object.keys(v).map((k) => [k, unwrap(v[k])]));
  return v;
}

function same(a, b) {
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return Object.is(a, b); // not JSON (a BigInt, a cycle): never throw
  }
}

/** Look up `keys` in `root`, unwrapping a wrapper at every step; arrays and prototypes don't count. */
function lookup(root, keys) {
  let cur = root;
  for (const k of keys) {
    while (isWrapper(cur)) cur = cur.value;
    if (!isPlain(cur) || !Object.hasOwn(cur, k)) return { found: false, value: undefined };
    cur = cur[k];
  }
  return { found: true, value: unwrap(cur) };
}

/**
 * One row per leaf path of `yours`, in key order, skipping the top-level `rev` (it always
 * differs). A leaf is anything but a non-empty plain object: a scalar, an array (compared whole),
 * an empty object, or a {value, source} wrapper. Each row is
 *   {path: "mask.after_ms", current, yours, changed, found}
 * where `current` and `yours` have every wrapper stripped (at any depth), `found` says the path
 * exists in `current` (else `current` is undefined), and `changed` is JSON.stringify inequality.
 * When no row is `found` (a restore body, for example), the caller shows the two JSON blocks
 * instead. A non-object `yours` gives []. It never throws.
 */
export function compareRows(current, yours) {
  const rows = [];
  if (!isPlain(yours)) return rows;
  const walk = (obj, prefix) => {
    for (const k of Object.keys(obj)) {
      if (prefix.length === 0 && k === "rev") continue;
      const keys = [...prefix, k];
      const v = obj[k];
      if (isPlain(v) && !isWrapper(v) && Object.keys(v).length > 0) {
        walk(v, keys);
        continue;
      }
      const mine = unwrap(v);
      const { found, value } = lookup(current, keys);
      rows.push({ path: keys.join("."), current: value, yours: mine, changed: !same(value, mine), found });
    }
  };
  walk(yours, []);
  return rows;
}
