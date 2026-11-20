// Prompts view: mask.* phrases and arg.* words per language, a
// placeholder picker, a live preview of every phrase for every enum value and language, and the
// acknowledgement x tool phrase combinations as a caller hears them. Every other catalog key is
// listed below, grouped by its prefix; the protected ones (confirmation read-backs and
// outcomes) are read-only and show the API's reason. A mask.* phrase may gain or lose a
// placeholder its tool allows (the preview checks it); other keys keep their placeholder set.
// Layout: a page header, three cards, and the prompt editor as a card whose
// language fields sit side by side in .lang-pair. Fields, controls, validation messages
// and API calls are as before.

import { badge, card, emptyState, langBadge, langName, pageHeader, panel, skeleton, tableWrap } from "../lib/components.js";
import { clear, field, h, markInvalid } from "../lib/dom.js";
import { ackCombinations, insertPlaceholder, placeholderNames, previewMatrix } from "../lib/preview.js";
import { canEdit, errorPanel, load, save, takesEffect } from "../lib/ui.js";
import { checkPromptTexts } from "../lib/validate.js";

/**
 * The catalog's other key prefixes, one GET each (the API lists by prefix). A prefix the API
 * doesn't know, or one with no keys, shows no group.
 */
export const OTHER_PREFIXES = [
  "account.", "agent.", "balance.", "clarify.", "command.", "common.", "confirm.", "error.", "escalate.",
  "escalation.", "factor.", "greeting.", "keypad.", "label.", "language.", "login.", "offer.", "option.",
  "prompt.", "slot.", "stepup.", "tool.", "verify.",
];

function argsLabel(args) {
  const parts = Object.entries(args).map(([k, v]) => `${k}=${v}`);
  return parts.length ? parts.join(", ") : "(no placeholders)";
}

function previewTable(texts, placeholders, languages) {
  const { rows, truncated } = previewMatrix(texts, placeholders, languages);
  return h("div", {},
    tableWrap(h("table.preview", {},
      h("caption", {}, "Preview: every value in every language"),
      h("thead", {}, h("tr", {}, ["Language", "Values", "Caller hears"].map((c) => h("th", { scope: "col" }, c)))),
      h("tbody", {}, rows.map((r) => h("tr", { class: r.error ? "is-invalid" : null },
        h("td", {}, r.lang), h("td", {}, argsLabel(r.args)),
        h("td", {}, r.error ? [badge("invalid", "warning"), " ", r.error] : r.text)))))),
    truncated ? h("p.note", {}, "Only the first rows are shown.") : null);
}

/** A language field's label: the language badge, then the name (read once, from the badge). */
const langLabel = (lang) => [langBadge(lang), h("span", { "aria-hidden": "true" }, langName(lang))];

/**
 * An editor for one prompt key, as a card. `placeholders` is the owning tool's
 * GET /placeholders list (or []), for mask.* phrases with placeholders.
 */
export function promptEditor(ctx, prompt, allLanguages, { placeholders = [], onSaved } = {}) {
  // The key's own languages from GET /prompts (one for a single-language key such as
  // language.select, which the API refuses in any other); every language when absent.
  const languages = Array.isArray(prompt.languages) && prompt.languages.length ? prompt.languages : allLanguages;
  const protectedKey = prompt.editable === false;
  const editable = canEdit(ctx.user) && !protectedKey;
  const isWord = prompt.key.startsWith("arg.");
  // Only mask.* phrases are rendered with a tool's argument words; other keys have their own
  // runtime values, so there is no preview (they keep their placeholder set; checkPromptTexts).
  const hasPreview = prompt.key.startsWith("mask.");
  const areas = {};
  let focused = null;
  const preview = h("div", { "aria-live": "polite" });
  const status = h("div");

  function texts() {
    return Object.fromEntries(languages.map((l) => [l, areas[l].value]));
  }
  function refresh() {
    if (hasPreview) clear(preview, previewTable(texts(), placeholders, languages));
  }

  const fields = languages.map((lang) => {
    const ta = h(isWord ? "input" : "textarea", {
      rows: isWord ? null : "2", type: isWord ? "text" : null, value: prompt.texts[lang] ?? "",
      "data-field": `texts[${lang}]`, disabled: !editable, spellcheck: "true",
    });
    ta.addEventListener("input", refresh);
    ta.addEventListener("focus", () => {
      focused = ta;
    });
    areas[lang] = ta;
    return field(langLabel(lang), ta).wrap;
  });
  focused = areas[languages[0]];

  const picker = !isWord && placeholders.length ? h("div", { role: "group", "aria-label": "Insert a placeholder" },
    h("span", {}, "Insert: "),
    placeholders.map((p) => h("button.sm", {
      type: "button", disabled: !editable, title: `Values: ${p.values.map((v) => v.value).join(", ")}`,
      on: {
        click: () => {
          const ta = focused;
          const { text, caret } = insertPlaceholder(ta.value, ta.selectionStart, ta.selectionEnd, p.name);
          ta.value = text;
          ta.focus();
          ta.setSelectionRange(caret, caret);
          refresh();
        },
      },
    }, h("code", {}, p.placeholder)))) : null;
  const missing = placeholders.flatMap((p) => p.missing || []);

  const form = h("form", { novalidate: true },
    h("div.lang-pair", {}, fields),
    picker,
    protectedKey ? panel({ tone: "info" }, h("p", {}, `Read-only: ${prompt.reason ?? "protected"}.`)) : null,
    missing.length ? panel({ tone: "warning" }, h("p", {}, "Missing words (fill these in under arg.*): ", missing.join(", "))) : null,
    hasPreview ? preview : null,
    status,
    editable ? h("button.primary", { type: "submit" }, "Save") : null);
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const t = texts();
    const errors = checkPromptTexts(prompt.key, t, languages, prompt.texts);
    if (hasPreview) {
      const { rows } = previewMatrix(t, placeholders, languages);
      const bad = rows.find((r) => r.error);
      if (bad) errors.push({ field: `texts[${bad.lang}]`, message: bad.error });
    }
    if (errors.length) {
      clear(status, errorPanel({ title: "Not saved", message: errors.map((e) => `${e.field}: ${e.message}`).join("; ") }));
      for (const e of errors) {
        const el = form.querySelector(`[data-field="${CSS.escape(e.field)}"]`);
        if (el) markInvalid(el, e.message);
      }
      return;
    }
    const yours = { rev: prompt.rev, texts: t };
    const res = await save(status, () => ctx.api.put(["prompts", prompt.key], yours),
      { yours, form, button: ev.submitter, onReload: onSaved });
    if (res && res.prompt) {
      prompt.rev = res.prompt.rev;
      prompt.texts = res.prompt.texts;
    }
  });
  refresh();
  return card({ title: h("code", {}, prompt.key), description: prompt.owner_tool || null }, form);
}

async function placeholdersFor(ctx, tool, cache) {
  if (!tool) return [];
  if (!cache.has(tool)) cache.set(tool, ctx.api.get(["placeholders", tool]).then((d) => d.placeholders).catch(() => []));
  return cache.get(tool);
}

/** A text cell: the text, or a "missing" badge. */
const textCell = (t, ...extra) => h("td", {}, t ?? badge("missing", "warning"), extra);
const actionsHeader = () => h("th", { scope: "col" }, h("span.visually-hidden", {}, "Actions"));

export async function promptsView(main, ctx, { params = new URLSearchParams() } = {}) {
  const head = { title: "Prompts", description: "Phrases and placeholder words callers hear, in each language." };
  const data = await load(main, () => Promise.all([
    ctx.api.get(["prompts"], { prefix: "mask." }),
    ctx.api.get(["prompts"], { prefix: "arg." }),
    ctx.api.get(["tools"]),
    ctx.api.get(["masking"]),
    Promise.all(OTHER_PREFIXES.map((prefix) => ctx.api.get(["prompts"], { prefix }).catch(() => null))),
  ]), head);
  if (!data) return;
  const [masks, args, tools, masking, otherLists] = data;
  const groups = otherLists.filter((g) => g && g.prompts && g.prompts.length);
  const languages = masks.languages;
  const only = params.get("tool");
  const cache = new Map();
  const ackKeys = new Set(masking.ack_pool.map((a) => a.key));
  const detail = h("div#prompt-editor");
  const editLabel = canEdit(ctx.user) ? "Edit" : "View";

  async function open(prompt) {
    clear(detail, skeleton("form"));
    const placeholders = await placeholdersFor(ctx, prompt.owner_tool, cache);
    clear(detail, promptEditor(ctx, prompt, languages, { placeholders, onSaved: () => promptsView(main, ctx, { params }) }));
    detail.querySelector("textarea, input")?.focus();
  }
  const openButton = (p) => h("td", {}, h("button.sm", { type: "button", on: { click: () => open(p) } }, editLabel));

  const phraseRows = masks.prompts
    .filter((p) => !only || p.owner_tool === only)
    .map((p) => h("tr", {},
      h("th", { scope: "row" }, h("code", {}, p.key)),
      h("td", {}, p.owner_tool ?? (ackKeys.has(p.key) ? "acknowledgement pool" : "-")),
      languages.map((l) => textCell(p.texts[l],
        placeholderNames(p.texts[l] ?? "").length ? [" ", badge("placeholders", "neutral")] : null)),
      openButton(p)));
  const wordRows = args.prompts.map((p) => h("tr", {},
    h("th", { scope: "row" }, h("code", {}, p.key)),
    languages.map((l) => textCell(p.texts[l])),
    openButton(p)));

  const groupCard = (g) => card({
    title: `Other prompts (${g.prefix}*)`,
    description: g.prompts.some((p) => p.editable === false) ? "Prompts marked read-only can't be edited here." : null,
    flush: true,
  },
  tableWrap(h("table", { "aria-label": `Prompts (${g.prefix}*)` },
    h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Key"),
      languages.map((l) => h("th", { scope: "col" }, langBadge(l))), actionsHeader())),
    h("tbody", {}, g.prompts.map((p) => h("tr", {},
      h("th", { scope: "row" }, h("code", {}, p.key),
        p.editable === false ? [" ", badge("read-only", "neutral"), h("span.note", {}, ` ${p.reason ?? ""}`)] : null),
      languages.map((l) => textCell(p.texts[l],
        placeholderNames(p.texts[l] ?? "").length ? [" ", badge("placeholders", "neutral")] : null)),
      h("td", {}, h("button.sm", { type: "button", on: { click: () => open(p) } },
        p.editable === false ? "View" : editLabel))))))));

  // Each acknowledgement followed by each of a tool's phrases.
  const combos = tools.tools.filter((t) => t.mask && (!only || t.name === only)).map((t) => h("details", {},
    h("summary", {}, `${t.name}: ${masking.ack_pool.length} acknowledgements x ${t.mask.pool.length} phrases`),
    languages.map((lang) => h("div", {}, h("h3", {}, langBadge(lang)), h("ul", {},
      ackCombinations(masking.ack_pool, t.mask.pool, lang).map((c) => h("li", {},
        c.text ?? [badge("missing", "warning"), " ", c.missing.join(", ")])))))));

  clear(main,
    pageHeader(head),
    only ? h("p", {}, `Showing ${only}'s phrases. `, h("a", { href: "#/prompts" }, "Show all")) : null,
    card({ title: "Phrases (mask.*)", flush: true },
      tableWrap(h("table", { "aria-label": "Phrases (mask.*)" },
        h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Key"), h("th", { scope: "col" }, "Used by"),
          languages.map((l) => h("th", { scope: "col" }, langBadge(l))), actionsHeader())),
        h("tbody", {}, phraseRows)))),
    detail,
    card({
      title: "Words for placeholder values (arg.*)",
      description: "A phrase's {placeholder} is replaced by the word for the call's value, in the caller's language.",
      flush: true,
    },
    tableWrap(h("table", { "aria-label": "Words for placeholder values (arg.*)" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Key"),
        languages.map((l) => h("th", { scope: "col" }, langBadge(l))), actionsHeader())),
      h("tbody", {}, wordRows)))),
    card({
      title: "As a caller hears them",
      description: "Each acknowledgement followed by each tool phrase (placeholders shown as {name}; each phrase's preview shows the values).",
    }, combos.length ? combos : emptyState({ icon: "prompts", title: "No combinations", message: "No tool masks its wait with a phrase." })),
    groups.map(groupCard),
    takesEffect(masks.takes_effect_s));
}
