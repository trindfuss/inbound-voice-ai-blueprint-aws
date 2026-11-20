// Phrase preview and placeholder insertion. Pure functions.
//
// The rules mirror owlbank.masking_config.render: `{name}` placeholders, each replaced by the
// catalog word for an enum value of the tool's argument `name` in the same language; a phrase
// renders only in full (an unknown placeholder, a missing or braced word, or a stray brace is an
// error). The server validates every save with the bridge's own code; this is the preview.

export const PLACEHOLDER = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;
/** Most preview rows shown for one phrase (the product of its placeholders' values x languages). */
export const MAX_PREVIEW_ROWS = 400;

/** The `{name}` placeholders in `text`, in order of first use. */
export function placeholderNames(text) {
  const seen = [];
  for (const m of String(text).matchAll(PLACEHOLDER)) if (!seen.includes(m[1])) seen.push(m[1]);
  return seen;
}

/** A `{` or `}` that isn't part of a `{name}` placeholder. */
export function hasStrayBrace(text) {
  const rest = String(text).replace(PLACEHOLDER, "");
  return rest.includes("{") || rest.includes("}");
}

function badWord(word) {
  return typeof word !== "string" || !word.trim() || word.includes("{") || word.includes("}");
}

/**
 * Render `text` for one combination of argument values in one language.
 * `placeholders` is GET /placeholders/{tool}'s list: [{name, values: [{value, key, words: {lang: word}}]}].
 * Returns {text} or {error}. One pass: a word is never expanded.
 */
export function renderPhrase(text, args, placeholders, lang) {
  if (typeof text !== "string" || !text.trim()) return { error: "The text is empty." };
  if (hasStrayBrace(text)) return { error: "A { or } that isn't part of a {placeholder}." };
  const byName = new Map((placeholders || []).map((p) => [p.name, p]));
  for (const name of placeholderNames(text)) {
    if (!byName.has(name)) return { error: `{${name}} isn't one of this tool's placeholders.` };
  }
  let error = null;
  const out = text.replace(PLACEHOLDER, (whole, name) => {
    const value = args[name];
    const entry = byName.get(name).values.find((v) => v.value === value);
    const word = entry && entry.words ? entry.words[lang] : undefined;
    if (!entry) {
      error ??= `No value given for {${name}}.`;
      return whole;
    }
    if (badWord(word)) {
      error ??= `Missing word for ${entry.key} in ${lang}.`;
      return whole;
    }
    return word;
  });
  return error ? { error } : { text: out };
}

/** Every combination of the values of `names` (cartesian product), as [{name: value}]. */
export function combinations(names, placeholders) {
  const byName = new Map((placeholders || []).map((p) => [p.name, p]));
  let combos = [{}];
  for (const name of names) {
    const entry = byName.get(name);
    if (!entry) continue;
    const next = [];
    for (const c of combos) for (const v of entry.values) next.push({ ...c, [name]: v.value });
    combos = next;
  }
  return combos;
}

/**
 * The preview of one phrase: a row for every combination of its placeholders' values in every
 * language. `texts` maps language to that language's text.
 * Returns {rows: [{lang, args, text?, error?}], truncated, ok}.
 */
export function previewMatrix(texts, placeholders, languages) {
  const rows = [];
  let truncated = false;
  for (const lang of languages) {
    const text = texts[lang] ?? "";
    const names = placeholderNames(text);
    const combos = combinations(names, placeholders);
    for (const args of combos) {
      if (rows.length >= MAX_PREVIEW_ROWS) {
        truncated = true;
        break;
      }
      rows.push({ lang, args, ...renderPhrase(text, args, placeholders, lang) });
    }
  }
  return { rows, truncated, ok: rows.length > 0 && rows.every((r) => !r.error) };
}

/**
 * Insert `{name}` into `text`, replacing the selection [start, end). Returns the new text and
 * the caret position just after the placeholder.
 */
export function insertPlaceholder(text, start, end, name) {
  const s = Math.max(0, Math.min(start ?? text.length, text.length));
  const e = Math.max(s, Math.min(end ?? s, text.length));
  const token = `{${name}}`;
  return { text: text.slice(0, s) + token + text.slice(e), caret: s + token.length };
}

/**
 * The acknowledgement-and-phrase combinations as a caller hears them: each acknowledgement followed by each of a
 * tool's phrases, in one language. `ackPool` and `toolPool` are [{key, texts: {lang: text}}].
 */
export function ackCombinations(ackPool, toolPool, lang) {
  const out = [];
  for (const ack of ackPool || []) {
    for (const phrase of toolPool || []) {
      const a = ack.texts ? ack.texts[lang] : null;
      const p = phrase.texts ? phrase.texts[lang] : null;
      out.push({
        ackKey: ack.key,
        phraseKey: phrase.key,
        text: a && p ? `${a} ${p}` : null,
        missing: [!a ? `${ack.key}[${lang}]` : null, !p ? `${phrase.key}[${lang}]` : null].filter(Boolean),
      });
    }
  }
  return out;
}
