// Form checks before a save. The server validates every write with the bridge's own
// rules and is the authority; these catch typing mistakes early with the same ranges as
// owlbank.masking_config and owlbank.policy. Each check returns null or {field, message}.

export const DELAY_MS = [0, 10000];
export const GREETING_SILENCE_MS = [0, 30000];
export const SILENCE_MAX_REPEATS = [0, 5];
export const CHARS_PER_S = [8, 30];
export const END_MODES = ["wait", "immediate"];
export const SILENCE_ACTIONS = ["escalate", "hangup"];
export const CONFIRM_MODES = ["none", "deterministic"];
export const GROUPS = ["admin", "admin-editor", "admin-viewer"];

/** The masking fields (MaskingSettings) and how each is checked. */
export const MASKING_FIELDS = {
  after_ms: { type: "int", range: DELAY_MS, label: "Acknowledgement after (ms)" },
  tool_after_ms: { type: "int", range: DELAY_MS, label: "Tool phrase after (ms)" },
  handoff_wait_ms: { type: "int", range: DELAY_MS, label: "Hand-off wait (ms)" },
  end_mode: { type: "enum", values: END_MODES, label: "End mode" },
  chars_per_s: { type: "number", range: CHARS_PER_S, label: "Speech rate (characters per second)" },
  silence_timeout_ms: { type: "int", range: DELAY_MS, label: "Silence timeout (ms)" },
  silence_max_repeats: { type: "int", range: SILENCE_MAX_REPEATS, label: "Silence repeats" },
  silence_action: { type: "enum", values: SILENCE_ACTIONS, label: "After the last repeat" },
  greeting_silence_timeout_ms: { type: "int", range: GREETING_SILENCE_MS, label: "Silence timeout after the opening greeting (ms)" },
};

/** Parse an integer typed in a form field: digits only (no 1e3, no 2.5, no blanks). */
export function parseIntField(raw) {
  const s = String(raw ?? "").trim();
  return /^-?\d+$/.test(s) ? Number(s) : null;
}

export function parseNumberField(raw) {
  const s = String(raw ?? "").trim();
  return /^\d+(\.\d+)?$/.test(s) ? Number(s) : null;
}

export function checkInt(field, raw, [lo, hi]) {
  const n = parseIntField(raw);
  if (n === null || n < lo || n > hi) return { field, message: `must be a whole number from ${lo} to ${hi}` };
  return null;
}

export function checkNumber(field, raw, [lo, hi]) {
  const n = parseNumberField(raw);
  if (n === null || n < lo || n > hi) return { field, message: `must be a number from ${lo} to ${hi}` };
  return null;
}

export function checkEnum(field, raw, values) {
  return values.includes(raw) ? null : { field, message: `must be one of ${values.join(", ")}` };
}

/**
 * The masking form: `values` maps each field to its typed value, or undefined for "use the
 * default". Returns {errors, masking}: the body's `masking` object holds only the set fields.
 */
export function checkMaskingForm(values) {
  const errors = [];
  const masking = {};
  for (const [name, spec] of Object.entries(MASKING_FIELDS)) {
    const raw = values[name];
    if (raw === undefined || raw === null) continue;
    const field = `masking.${name}`;
    let err = null;
    if (spec.type === "int") err = checkInt(field, raw, spec.range);
    else if (spec.type === "number") err = checkNumber(field, raw, spec.range);
    else err = checkEnum(field, raw, spec.values);
    if (err) errors.push(err);
    else masking[name] = spec.type === "enum" ? raw : Number(String(raw).trim());
  }
  return { errors, masking };
}

/**
 * A tool's form. `form`: {min_level, confirm ("" = the default), masked, phrase_key,
 * after_ms (undefined = default), handoff_wait_ms (undefined = default)}. Returns
 * {errors, body} where body holds min_level, confirm and mask as PUT /tools/{name} takes them.
 */
export function checkToolForm(form) {
  const errors = [];
  const body = {};
  const lvl = parseIntField(form.min_level);
  if (lvl === null || lvl < 0) errors.push({ field: "min_level", message: "must be a whole number, 0 or more" });
  else body.min_level = lvl;
  if (form.confirm === "" || form.confirm === undefined || form.confirm === null) body.confirm = null;
  else {
    const err = checkEnum("confirm", form.confirm, CONFIRM_MODES);
    if (err) errors.push(err);
    else body.confirm = form.confirm;
  }
  if (!form.masked) {
    body.mask = null;
  } else {
    const mask = {};
    const key = String(form.phrase_key ?? "").trim();
    if (!/^mask\.[A-Za-z0-9_.-]{1,123}$/.test(key)) {
      errors.push({ field: "mask.phrase_key", message: "must be a catalog key starting with mask." });
    } else mask.phrase_key = key;
    for (const f of ["after_ms", "handoff_wait_ms"]) {
      if (form[f] === undefined || form[f] === null) continue;
      const err = checkInt(`mask.${f}`, form[f], DELAY_MS);
      if (err) errors.push(err);
      else mask[f] = parseIntField(form[f]);
    }
    body.mask = mask;
  }
  return { errors, body };
}

/** True when a tool edit changes security behaviour: its level or confirmation. */
export function isSecurityChange(before, body) {
  const confirmBefore = before.confirm_source === "tool" ? before.confirm : null;
  return ("min_level" in body && body.min_level !== before.min_level)
    || ("confirm" in body && body.confirm !== confirmBefore);
}

export const MAX_PROMPT_CHARS = 600;
/** agent.* texts are model instructions, never spoken; as the API's limit. */
export const MAX_AGENT_PROMPT_CHARS = 16000;
/** Empty or only what Python's str.strip() removes (the API's rule; String.trim differs slightly). */
const PY_SPACE = new Set([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029,
  0x202f, 0x205f, 0x3000]);
const pyBlank = (t) => [...t].every((ch) => PY_SPACE.has(ch.codePointAt(0)));
const PLACEHOLDER = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

/** The set of `{placeholder}` names in a text, sorted (matches the admin API's rule). */
export function placeholderSet(text) {
  return [...new Set(Array.from(String(text ?? "").matchAll(PLACEHOLDER), (m) => m[1]))].sort();
}

/**
 * The exact placeholder set of an agent.* text, by key, as the API's
 * owlbank.agent_prompt.ALLOWED_PLACEHOLDERS. Any other agent.* key (every agent.system.* text, the
 * web-activity blocks, the unverified frame, a new key) allows none.
 */
export const AGENT_ALLOWED_PLACEHOLDERS = Object.freeze({
  "agent.caller_context": Object.freeze(["firstName", "memory", "profile"]),
});

/** The sorted allowed set for an agent.* key, or null for a key outside agent (as the API). */
export function allowedPlaceholders(key) {
  if (!key.startsWith("agent.")) return null;
  return Object.hasOwn(AGENT_ALLOWED_PLACEHOLDERS, key) ? AGENT_ALLOWED_PLACEHOLDERS[key] : [];
}

/** Why `text` breaks the allow-list for `key`, with the API's message, or null. */
export function placeholderProblems(key, text) {
  const allowed = allowedPlaceholders(key);
  if (allowed === null) return null;
  if (/[{}]/.test(String(text).replace(PLACEHOLDER, ""))) return "has a '{' or '}' that isn't a {placeholder}";
  if (placeholderSet(text).join() !== allowed.join()) {
    return allowed.length ? `must have exactly ${allowed.map((n) => `{${n}}`).join(", ")}` : "must have no placeholders";
  }
  return null;
}

/**
 * A prompt's texts: every language needs non-empty text of at most 600 characters with no `<` or
 * `>`; an agent.* text (read by the model, not spoken) may have up to 16,000 and may use `<`
 * and `>`. Length counts code points, as the API's Python does. arg.* words have no braces. `current` (the saved texts, by language) turns on the
 * placeholder rule: no stray brace, and each language keeps exactly the placeholder set of its
 * current text, or of the first language's when it has none (as the API does). mask.* phrases
 * are exempt from the same-set part: they may use any placeholder their tool
 * allows, which the editor's preview checks. An agent.* text instead has exactly its key's
 * allowed set (placeholderProblems), with or without `current`, so an edit, a restore of a
 * revision from before agent prompt caching and a new key are checked alike. The server is the
 * authority.
 */
export function checkPromptTexts(key, texts, languages, current = null) {
  const errors = [];
  const saved = current || {};
  const fallback = [languages[0], ...languages].map((l) => saved[l]).find((t) => typeof t === "string" && t);
  const agent = key.startsWith("agent.");
  const limit = agent ? MAX_AGENT_PROMPT_CHARS : MAX_PROMPT_CHARS;
  for (const lang of languages) {
    const field = `texts[${lang}]`;
    const t = texts[lang];
    if (typeof t !== "string" || pyBlank(t)) errors.push({ field, message: "every language needs its text" });
    else if (t.length > limit && [...t].length > limit) errors.push({ field, message: `must be at most ${limit} characters` });
    else if (!agent && (t.includes("<") || t.includes(">"))) errors.push({ field, message: "must not contain '<' or '>'" });
    else if (key.startsWith("arg.") && (t.includes("{") || t.includes("}"))) {
      errors.push({ field, message: "a word can't contain { or }" });
    } else if (agent) {
      const problem = placeholderProblems(key, t);
      if (problem) errors.push({ field, message: problem });
    } else if (current && fallback !== undefined) {
      if (/[{}]/.test(t.replace(PLACEHOLDER, ""))) {
        errors.push({ field, message: "has a '{' or '}' that isn't a {placeholder}" });
        continue;
      }
      if (key.startsWith("mask.")) continue;
      const want = placeholderSet(saved[lang] || fallback);
      const got = placeholderSet(t);
      if (want.join() !== got.join()) {
        errors.push({ field, message: `must keep exactly the placeholders of the current text (${want.map((n) => `{${n}}`).join(", ") || "none"})` });
      }
    }
  }
  return errors;
}

const EMAIL = /^[^@\s{}<>]{1,64}@[A-Za-z0-9.-]{1,189}\.[A-Za-z]{2,63}$/;

export function checkNewUser({ email, group }) {
  const errors = [];
  const e = String(email ?? "").trim();
  if (e.length > 254 || !EMAIL.test(e)) errors.push({ field: "email", message: "must be an email address" });
  if (!GROUPS.includes(group)) errors.push({ field: "group", message: `must be one of ${GROUPS.join(", ")}` });
  return errors;
}

/** The audit item keys the API accepts (policy#tool#x, policy#masking, catalog#key, user#name). */
export function checkAuditItem(item) {
  return /^(policy|catalog|user)#[A-Za-z0-9_.#@+:-]{1,200}$/.test(String(item ?? ""))
    ? null
    : { field: "item", message: "choose an item such as policy#tool#get_balance" };
}
