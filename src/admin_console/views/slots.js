// Slots view (phases 1 and 2).
//
// Phase 1: a page header; the read-only acceptSpokenFactors ceiling as an info panel; a flush card
// with one table row per slot: its type, the sensitive, keypad_only and barge_in switches,
// maxRetries as text and an "effective keypad-only" badge when the deployment forces it. Each
// switch saves on its own (PUT /slots/<name> with the one field and the row's rev). Turning
// sensitive or keypad_only off asks first (danger), since it is a security change.
//
// Phase 2 adds: the read-only mode ceiling (an info panel); per slot a mode select, the configured
// and effective mode in each flow as badges (with where the configured mode came from, and why the
// effective one is lower: "capped by deployment" and the like), and a confirm select (for a choice
// slot with the hint: "never" skips only a keypad press or an exact name); the
// slot_resolution card (default_mode, flow overrides, resolver_budget_ms, and generative_enabled
// for admins only, with its warning, shown as having no effect when the ceiling isn't generative);
// and for each choice slot its options, with spoken names per language (PUT /prompts/option.*) and
// aliases per language (PUT /slots/<name> with the whole options list). A mode change (a slot's
// mode, default_mode, a flow override, generative_enabled) first asks POST /slots/preview which
// flows and slots would then call a model, and shows that in the confirmation; the danger variant
// is used when the preview warns of a move away from deterministic, or generative is turned on.
//
// The warning, the effective keypad-only, the effective-mode table and every warning come
// from the API (the front door's own rules); the view never recomputes them.

import { badge, card, emptyState, langBadge, langName, pageHeader, panel, tableWrap, toast } from "../lib/components.js";
import { clear, field, h, icon, markInvalid } from "../lib/dom.js";
import { canEdit, confirmDialog, errorPanel, isAdmin, load, save, takesEffect } from "../lib/ui.js";
import { parseIntField } from "../lib/validate.js";

/** The badge text for a slot that the deployment makes keypad-only. */
export const EFFECTIVE_KEYPAD_ONLY = "Effective: keypad only";

/** The modes, least to most model use, and the flows (slot_policy.FLOWS). */
export const MODES = ["deterministic", "assisted", "generative"];
export const FLOWS = ["login", "stepup", "confirm"];
export const CONFIRM_POLICIES = ["when_inferred", "always", "never"];
/**
 * The rule next to a choice slot's confirm select: "never" skips the
 * confirmation only for a keypad press or an exact spoken name or alias; a model-resolved option
 * is always confirmed (dialog.decide). A digit slot's "never" keeps its meaning, so no hint there.
 */
export const CHOICE_NEVER_HINT = "Never: skipped only for a keypad press or an exact name; a model match is always confirmed";

/** Where a configured mode came from (ModeDecision.source). */
export const SOURCE_TEXT = {
  slot: "set on the slot",
  flow: "the flow's setting",
  default: "default_mode",
  invalid: "invalid setting, loaded as deterministic",
};

/** The texts of the effective-mode badges, from one ModeDecision as GET /slots sends it. */
export function configuredText(d) {
  return `Configured: ${d.configured} (${SOURCE_TEXT[d.source] ?? d.source})`;
}
export function effectiveText(d) {
  return `Effective: ${d.effective}`;
}
export function capText(d) {
  if (d.capped === "ceiling") return `Capped by deployment (ceiling ${d.ceiling})`;
  if (d.capped === "generative_disabled") return "Capped: generative isn't enabled";
  if (d.capped === "type") return "Capped: generative needs a choice slot";
  if (d.capped === "sensitive") return "Capped: a sensitive slot";
  return `Capped: ${d.capped}`;
}

/** Shown next to the generative_enabled control. */
export const GENERATIVE_WARNING = "Generative mode lets the model word some follow-up questions (confirmations, "
  + "disambiguations, not-available questions) on non-sensitive choice slots. That wording is checked "
  + "structurally but is no longer validated catalog text.";
/** Shown when the ceiling isn't generative. */
export const GENERATIVE_NO_EFFECT = "No effect in this deployment";

const LABELS = { sensitive: "sensitive", keypad_only: "keypad only", barge_in: "barge-in" };
const COLUMNS = ["Slot", "Type", "Sensitive", "Keypad only", "Barge-in", "Mode", "Effective mode", "Confirm",
  "Max retries", "History"];
const INHERIT = "Inherit";
const USE_DEFAULT = "Use default_mode";

const onOff = (v) => (v === true ? "On" : "Off");
const dash = (v) => (v === null || v === undefined || v === "" ? "-" : String(v));
const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const list = (v) => (Array.isArray(v) ? v : []);
const modeName = (m) => (m === null || m === undefined || m === "" ? "inherit" : String(m));

/** The confirmation for turning a security setting off. */
const OFF_CONFIRM = {
  sensitive: (s) => ({
    title: "Turn sensitive off?",
    message: `If ${s.name} isn't sensitive, its value may be written to logs and the call journal, read back to `
      + "the caller and passed to the AI agent. The audit log marks this as a security change."
      + (s.is_factor ? " It is a verification factor: production deployments should keep every factor sensitive." : ""),
    details: ["sensitive: true -> false"],
    confirmLabel: "Turn sensitive off",
    danger: true,
  }),
  keypad_only: (s, acceptSpokenFactors) => ({
    title: "Allow spoken answers?",
    message: `Callers may then speak ${s.name}'s answer, so people nearby can hear it and the words pass through `
      + "Twilio speech recognition. The audit log marks this as a security change."
      + (s.is_factor && acceptSpokenFactors === false
        ? " acceptSpokenFactors is off in this deployment, so this factor stays keypad-only until it is turned on." : ""),
    details: ["keypad_only: true -> false"],
    confirmLabel: "Allow spoken answers",
    danger: true,
  }),
};

/**
 * The confirmation for a mode change, from the API's preview: which slots and flows would
 * call a model, the effective modes that change, and the preview's warnings. Danger when the
 * preview warns of a move away from deterministic, or generative is being turned on.
 */
export function previewConfirm(preview, { title, details = [], generativeOn = false }) {
  const warnings = list(preview && preview.warnings);
  const calls = list(preview && preview.calls_model);
  const changes = list(preview && preview.changes);
  const away = warnings.some((w) => w && w.code === "away_from_deterministic");
  const ceiling = preview && preview.ceiling;
  const callsText = calls.length
    ? `After this change, these would call a model on Amazon Bedrock (the deployment's ceiling is ${ceiling}): `
      + `${calls.map((c) => `${c.slot} in ${c.flow} (${c.effective})`).join("; ")}.`
    : `After this change, no slot would call a model (the deployment's ceiling is ${ceiling}).`;
  const message = [
    callsText,
    changes.length ? null : "No effective mode changes.",
    ...warnings.map((w) => w && w.message),
    generativeOn ? GENERATIVE_WARNING : null,
    away || generativeOn ? "The audit log marks this as a security change." : null,
  ].filter(Boolean).join(" ");
  return {
    title,
    message,
    details: [...details, ...changes.map((c) => `${c.slot} (${c.flow}) effective: ${c.from} -> ${c.to}`)],
    confirmLabel: "Save",
    danger: away || generativeOn,
  };
}

function ceilingPanel(acceptSpokenFactors) {
  const on = acceptSpokenFactors === true;
  const box = panel({ tone: "info", title: on ? "Spoken factors allowed" : "Spoken factors off" },
    h("p", {}, h("code", {}, "acceptSpokenFactors"), on
      ? " is on in this deployment: callers may speak a verification factor unless its slot is keypad-only."
      : " is off in this deployment: every verification factor is keypad-only, whatever its keypad_only setting."),
    h("p", {}, "It is set when the stack deploys and can't be changed here."));
  box.setAttribute("data-ceiling", "acceptSpokenFactors");
  return box;
}

function modeCeilingPanel(ceiling) {
  const box = panel({ tone: "info", title: `Mode ceiling: ${dash(ceiling)}` },
    h("p", {}, "No slot runs above the deployment's mode ceiling (", h("code", {}, "slotModeCeiling"),
      "), whatever its configured mode; a capped slot shows \"Capped by deployment\"."),
    h("p", {}, "It is set when the stack deploys and can't be changed here."));
  box.setAttribute("data-ceiling", "mode");
  return box;
}

const WARNING_TITLES = {
  verification_not_sensitive: "Verification factor isn't sensitive",
  generative_no_effect: "Generative has no effect here",
  away_from_deterministic: "Moving away from deterministic",
  generative_unvalidated_wording: "Generative wording isn't validated text",
};
const warningTitle = (code) => WARNING_TITLES[code] ?? "Warning";

/** A language field's label: the language badge, then the name (read once, from the badge). */
const langLabel = (lang) => [langBadge(lang), h("span", { "aria-hidden": "true" }, langName(lang))];

/** A select with `choices` ([value, label]) and `current` selected; an unknown current is kept as an option. */
function selectOf(choices, current, attrs) {
  const all = choices.some(([v]) => v === current) || current === undefined
    ? choices : [...choices, [current, String(current)]];
  return h("select", attrs, all.map(([v, label]) => h("option", { value: v, selected: v === current }, label)));
}

/** A select inside a table cell: the select-wrap and chevron that field() gives a labelled one. */
function cellSelect(select) {
  const chevron = icon("chevron-down");
  chevron.classList.add("select-chevron");
  return h("span.select-wrap", {}, select, chevron);
}

const splitAliases = (text) => String(text ?? "").split(",").map((a) => a.trim()).filter(Boolean);
const sameList = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

export async function slotsView(main, ctx) {
  const head = {
    title: "Slots",
    description: "Each slot's input settings, how its answers are resolved (its mode and confirmation), and a choice "
      + "slot's options.",
  };
  const data = await load(main, () => ctx.api.get(["slots"]), head);
  if (!data) return;
  const editable = canEdit(ctx.user);
  const admin = isAdmin(ctx.user);
  const slots = list(data.slots);
  const header = pageHeader(head);
  const resolutionData = isObj(data.slot_resolution) ? data.slot_resolution : null;
  const problems = [
    ...slots.flatMap((s) => list(s.problems).map((p) => `${s.name}: ${p}`)),
    ...list(resolutionData && resolutionData.problems).map((p) => `slot_resolution: ${p}`),
  ];
  const problemsPanel = problems.length
    ? panel({ tone: "warning", title: "Problems" }, h("ul", {}, problems.map((p) => h("li", {}, p))))
    : null;
  if (problemsPanel) problemsPanel.setAttribute("data-problems", "slots");
  const phase2 = Object.hasOwn(data, "ceiling");
  const modePanel = phase2 ? modeCeilingPanel(data.ceiling) : null;

  if (!slots.length) {
    clear(main, header, ceilingPanel(data.acceptSpokenFactors), modePanel,
      card({ label: "Slots" }, emptyState({ icon: "slots", title: "No slots", message: "Seed the policy table to list slots." })));
    return;
  }

  // The option spoken names are catalog prompts: their revs come from GET /prompts, read now so
  // a save is checked against what was shown.
  const choiceSlots = slots.filter((s) => s.loaded !== false && s.type === "choice" && Array.isArray(s.options));
  const promptRevs = new Map(); // option.<slot>.<value> -> rev
  const promptsRead = new Set(); // slot names whose prompt revs were read
  await Promise.all(choiceSlots.map(async (s) => {
    try {
      const found = await ctx.api.get(["prompts"], { prefix: `option.${s.name}.` });
      if (!isObj(found) || !Array.isArray(found.prompts)) return;
      for (const p of found.prompts) if (isObj(p) && typeof p.key === "string") promptRevs.set(p.key, p.rev);
      promptsRead.add(s.name);
    } catch {
      // Spoken names stay read-only for this slot; the card says so.
    }
  }));

  const languages = list(data.languages).filter((l) => typeof l === "string" && l.trim() !== "");
  const status = h("div", { "aria-live": "polite" });
  const tbody = h("tbody");
  const table = h("table", {},
    h("caption.visually-hidden", {}, "Slots, their input settings and modes"),
    h("thead", {}, h("tr", {}, COLUMNS.map((c) => h("th", { scope: "col", class: c === "Max retries" ? "num" : null }, c)))),
    tbody);
  const rowNodes = new Map(); // slot name -> its rendered rows (the slot's row, then its warning row)
  const current = new Map(); // slot name -> the slot as last read (GET or a save's response)
  const optionHolders = new Map(); // choice slot name -> the element holding its options card
  const saving = new Set(); // slot names with a confirmation or save in progress

  function showSaveWarnings(res, key, into) {
    const warnings = list(res && res.warnings);
    if (!warnings.length) return;
    const box = panel({ tone: "warning", title: res.security_change ? "Saved: a security change" : "Saved, with warnings" },
      h("ul", {}, warnings.map((w) => h("li", {}, `${w.slot}: ${w.message}`))));
    box.setAttribute("data-save-warnings", key);
    into.append(box);
  }

  /** POST /slots/preview; on an error, a toast and the field marked (or the error panel). */
  async function runPreview(body, form, into) {
    try {
      return await ctx.api.post(["slots", "preview"], body);
    } catch (err) {
      toast({ tone: "error", title: err.title || "Error", message: err.message || String(err) });
      const el = err.kind === "invalid" && err.field && form
        ? form.querySelector(`[data-field="${CSS.escape(err.field)}"]`)
        : null;
      if (el) {
        clear(into);
        markInvalid(el, err.message);
        el.focus();
      } else {
        clear(into, errorPanel(err));
      }
      return null;
    }
  }

  // ---------------------------------------------------------------- table rows

  function switchCell(s, f) {
    const input = h("input.switch", {
      type: "checkbox", role: "switch", checked: s[f] === true, disabled: !editable,
      "data-field": `slot#${s.name}.${f}`, "aria-label": `${s.name}: ${LABELS[f]}`,
    });
    const stateText = h("span", { "aria-hidden": "true" }, onOff(s[f]));
    input.addEventListener("change", () => onToggle(s, f, input, stateText));
    return h("label.inline", {}, input, stateText);
  }

  function modeCell(s) {
    const select = selectOf([["", INHERIT], ...MODES.map((m) => [m, m])], s.mode ?? "", {
      disabled: !editable, "data-field": `slot#${s.name}.mode`, "aria-label": `${s.name}: mode`,
    });
    select.addEventListener("change", () => onMode(s, select));
    return cellSelect(select);
  }

  function confirmCell(s) {
    const select = selectOf(CONFIRM_POLICIES.map((c) => [c, c]), s.confirm, {
      disabled: !editable, "data-field": `slot#${s.name}.confirm`, "aria-label": `${s.name}: confirm`,
    });
    select.addEventListener("change", () => onConfirm(s, select));
    if (s.type !== "choice") return cellSelect(select);
    const hint = h("p.hint", { id: `confirm-hint-${encodeURIComponent(s.name)}`, "data-confirm-hint": s.name }, CHOICE_NEVER_HINT);
    select.setAttribute("aria-describedby", hint.id);
    return [cellSelect(select), hint];
  }

  /** The effective-mode table, exactly as GET /slots sends it: one line per flow. */
  function effectiveCell(s) {
    const modes = isObj(s.modes) ? Object.entries(s.modes).filter(([, d]) => isObj(d)) : [];
    if (!modes.length) return "-";
    return modes.map(([flow, d]) => h("div", {
      "data-flow": flow, "data-configured": dash(d.configured), "data-source": dash(d.source),
      "data-effective": dash(d.effective), "data-capped": dash(d.capped),
    },
    h("code", {}, flow), " ",
    badge(configuredText(d), "neutral"), " ",
    badge(effectiveText(d), d.effective === "deterministic" ? "neutral" : "info"),
    d.capped ? [" ", badge(capText(d), "warning")] : null));
  }

  function rowsFor(s) {
    const loaded = s.loaded !== false;
    const n = list(s.problems).length;
    const settings = ["sensitive", "keypad_only", "barge_in"].map((f) => {
      if (!loaded) return h("td", {}, "-");
      const forced = f === "keypad_only" && s.effective_keypad_only === true && s.keypad_only === false;
      return h("td", {}, switchCell(s, f), forced ? badge(EFFECTIVE_KEYPAD_ONLY, "info") : null);
    });
    const phase2Row = loaded && (Object.hasOwn(s, "mode") || Object.hasOwn(s, "modes"));
    const row = h("tr", { "data-slot": s.name },
      h("th", { scope: "row" }, h("code", {}, s.name),
        s.is_factor === true ? [" ", badge("factor", "brand")] : null,
        n ? [" ", badge(`${n} problem(s)`, "warning")] : null),
      h("td", {}, dash(s.type)),
      settings,
      h("td", {}, phase2Row ? modeCell(s) : "-"),
      h("td", { "data-effective-for": s.name }, loaded ? effectiveCell(s) : "-"),
      h("td", {}, phase2Row && typeof s.confirm === "string" ? confirmCell(s) : "-"),
      h("td.num", {}, dash(s.maxRetries)),
      h("td", {}, h("a", { href: `#/audit?item=${encodeURIComponent(`policy#slot#${s.name}`)}` },
        "History", h("span.visually-hidden", {}, ` of ${s.name}`))));
    const warnings = list(s.warnings);
    const warningRow = warnings.length
      ? h("tr", { "data-warning-for": s.name }, h("td", { colspan: String(COLUMNS.length) },
        warnings.map((w) => panel({ tone: "warning", title: warningTitle(w.code) }, h("p", {}, w.message)))))
      : null;
    return [row, warningRow].filter(Boolean);
  }

  function place(s) {
    const before = current.get(s.name);
    current.set(s.name, s);
    const old = rowNodes.get(s.name);
    const fresh = rowsFor(s);
    if (old) {
      for (const node of fresh) tbody.insertBefore(node, old[0]);
      for (const node of old) node.remove();
    } else {
      tbody.append(...fresh);
    }
    rowNodes.set(s.name, fresh);
    const holder = optionHolders.get(s.name);
    if (holder && before && JSON.stringify(before.options) !== JSON.stringify(s.options)) {
      clear(holder, Array.isArray(s.options) ? optionsCard(s) : null);
    }
  }

  async function onToggle(s, f, input, stateText) {
    const want = input.checked;
    const reset = () => {
      input.checked = s[f] === true;
      stateText.textContent = onOff(s[f]);
    };
    if (saving.has(s.name)) { // one change per slot at a time: its rev is in flight
      reset();
      return;
    }
    stateText.textContent = onOff(want);
    saving.add(s.name);
    try {
      if (!want && s[f] === true && OFF_CONFIRM[f]) {
        const ok = await confirmDialog(OFF_CONFIRM[f](s, data.acceptSpokenFactors));
        if (!ok) {
          reset();
          return;
        }
      }
      const yours = { rev: s.rev, [f]: want };
      const res = await save(status, () => ctx.api.put(["slots", s.name], yours),
        { yours, form: table, button: input, onReload: () => slotsView(main, ctx) });
      if (!res) {
        reset();
        return;
      }
      if (res.slot) place(res.slot);
      showSaveWarnings(res, s.name, status);
    } finally {
      saving.delete(s.name);
    }
  }

  /** A slot's mode: the preview in the confirmation first, then PUT {rev, mode}. */
  async function onMode(s, select) {
    const was = s.mode ?? "";
    const want = select.value === "" ? null : select.value;
    const reset = () => { select.value = was; };
    if (saving.has(s.name) || (want ?? "") === was) {
      reset();
      return;
    }
    saving.add(s.name);
    try {
      const preview = await runPreview({ slots: { [s.name]: { mode: want } } }, table, status);
      if (!preview) {
        reset();
        return;
      }
      const ok = await confirmDialog(previewConfirm(preview, {
        title: `Change ${s.name}'s mode?`,
        details: [`${s.name}.mode: ${modeName(s.mode)} -> ${modeName(want)}`],
        generativeOn: want === "generative",
      }));
      if (!ok) {
        reset();
        return;
      }
      const yours = { rev: s.rev, mode: want };
      const res = await save(status, () => ctx.api.put(["slots", s.name], yours),
        { yours, form: table, button: select, onReload: () => slotsView(main, ctx) });
      if (!res) {
        reset();
        return;
      }
      if (res.slot) place(res.slot);
      showSaveWarnings(res, s.name, status);
    } finally {
      saving.delete(s.name);
    }
  }

  /** A slot's confirm policy saves at once, like the switches. */
  async function onConfirm(s, select) {
    const was = s.confirm;
    const reset = () => { select.value = was; };
    if (saving.has(s.name) || select.value === was) {
      reset();
      return;
    }
    saving.add(s.name);
    try {
      const yours = { rev: s.rev, confirm: select.value };
      const res = await save(status, () => ctx.api.put(["slots", s.name], yours),
        { yours, form: table, button: select, onReload: () => slotsView(main, ctx) });
      if (!res) {
        reset();
        return;
      }
      if (res.slot) place(res.slot);
      showSaveWarnings(res, s.name, status);
    } finally {
      saving.delete(s.name);
    }
  }

  // ---------------------------------------------------------------- options (choice slots)

  function langsOf(s) {
    if (languages.length) return languages;
    const seen = new Set();
    for (const o of list(s.options)) {
      for (const l of [...Object.keys(isObj(o.spoken_names) ? o.spoken_names : {}),
        ...Object.keys(isObj(o.aliases) ? o.aliases : {})]) if (l.trim() !== "") seen.add(l);
    }
    return [...seen];
  }

  /** One option's spoken names, per language: PUT /prompts/option.<slot>.<value> {rev, texts}. */
  function spokenForm(s, o, langs) {
    const key = `option.${s.name}.${o.value}`;
    const canSave = editable && promptsRead.has(s.name);
    const names = isObj(o.spoken_names) ? o.spoken_names : {};
    const inputs = {};
    const fields = langs.map((lang) => {
      const input = h("input", { type: "text", value: names[lang] ?? "", "data-field": `texts[${lang}]`, disabled: !canSave });
      inputs[lang] = input;
      return field(langLabel(lang), input).wrap;
    });
    const formStatus = h("div", { "aria-live": "polite" });
    const form = h("form", { "data-spoken": key, novalidate: true },
      h("fieldset", {},
        h("legend", {}, h("code", {}, o.value), o.dtmf ? ` (key ${o.dtmf})` : null, ": spoken names"),
        h("div.lang-pair", {}, fields)),
      formStatus,
      canSave ? h("button.sm", { type: "submit" }, "Save spoken names") : null);
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const opt = list((current.get(s.name) || s).options).find((x) => x && x.value === o.value) || o;
      const shown = isObj(opt.spoken_names) ? opt.spoken_names : {};
      const texts = Object.fromEntries(langs.map((l) => [l, inputs[l].value]));
      if (langs.every((l) => texts[l] === (shown[l] ?? ""))) {
        clear(formStatus, panel({ tone: "info" }, "Nothing to save: no spoken name changed."));
        return;
      }
      const yours = { rev: promptRevs.get(key) ?? 0, texts };
      const res = await save(formStatus, () => ctx.api.put(["prompts", key], yours),
        { yours, form, button: ev.submitter, onReload: () => slotsView(main, ctx) });
      if (res && isObj(res.prompt)) {
        promptRevs.set(key, res.prompt.rev);
        if (isObj(res.prompt.texts)) opt.spoken_names = { ...res.prompt.texts };
      }
    });
    return form;
  }

  /** Every option's aliases, per language: PUT /slots/<name> {rev, options}. A conflict's 400 names a language. */
  function aliasesForm(s, langs) {
    const inputs = new Map(); // `${index}|${lang}` -> input
    const groups = langs.map((lang) => h("fieldset", { "data-field": `slot#${s.name}.options.aliases.${lang}`, tabindex: "-1" },
      h("legend", {}, langLabel(lang), " aliases"),
      list(s.options).map((o, i) => {
        const aliases = isObj(o.aliases) ? list(o.aliases[lang]) : [];
        const input = h("input", { type: "text", value: aliases.join(", "), disabled: !editable,
          "data-field": `slot#${s.name}.options[${i}].aliases.${lang}` });
        inputs.set(`${i}|${lang}`, input);
        return field(h("code", {}, o.value), input).wrap;
      })));
    const formStatus = h("div", { "aria-live": "polite" });
    const form = h("form", { "data-aliases": s.name, novalidate: true },
      h("p.hint", {}, "Other words a caller may say for an option, separated by commas. No word may belong to two options "
        + "in one language, and no alias may equal another option's spoken name."),
      h("div.lang-pair", {}, groups),
      formStatus,
      editable ? h("button.sm", { type: "submit" }, "Save aliases") : null);
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const cur = current.get(s.name) || s;
      if (saving.has(s.name)) return;
      const curOptions = list(cur.options).filter(isObj);
      const options = curOptions.map((o, i) => {
        const was = isObj(o.aliases) ? o.aliases : {};
        const aliases = Object.fromEntries(Object.entries(was).map(([l, a]) => [l, list(a)]));
        for (const lang of langs) {
          const input = inputs.get(`${i}|${lang}`);
          if (!input) continue;
          const typed = splitAliases(input.value);
          if (typed.length || Object.hasOwn(aliases, lang)) aliases[lang] = typed;
        }
        const out = { value: o.value, aliases };
        if (o.dtmf !== null && o.dtmf !== undefined) out.dtmf = o.dtmf;
        return out;
      });
      const changed = options.some((o, i) => {
        const was = isObj(curOptions[i].aliases) ? curOptions[i].aliases : {};
        const langsHere = new Set([...Object.keys(was), ...Object.keys(o.aliases)]);
        return [...langsHere].some((l) => !sameList(list(was[l]), list(o.aliases[l])));
      });
      if (!changed) {
        clear(formStatus, panel({ tone: "info" }, "Nothing to save: no alias changed."));
        return;
      }
      saving.add(s.name);
      try {
        const yours = { rev: cur.rev, options };
        const res = await save(formStatus, () => ctx.api.put(["slots", s.name], yours),
          { yours, form, button: ev.submitter, onReload: () => slotsView(main, ctx) });
        if (!res) return;
        showSaveWarnings(res, s.name, formStatus);
        if (res.slot) place(res.slot);
      } finally {
        saving.delete(s.name);
      }
    });
    return form;
  }

  function optionsCard(s) {
    const langs = langsOf(s);
    const opts = list(s.options).filter(isObj);
    const body = opts.length
      ? [
        promptsRead.has(s.name) || !editable ? null
          : panel({ tone: "warning" }, h("p", {}, "The spoken names couldn't be read for editing; reload to change them.")),
        opts.map((o) => spokenForm(s, o, langs)),
        aliasesForm({ ...s, options: opts }, langs),
      ]
      : emptyState({ icon: "slots", title: "No options", message: "This choice slot has no options." });
    const el = card({
      title: `Options: ${s.name}`,
      description: "Each option's spoken name per language (the catalog's option texts) and the aliases callers may "
        + "also say. Spoken names and aliases save separately.",
    }, body);
    el.setAttribute("data-options-for", s.name);
    return el;
  }

  // ---------------------------------------------------------------- slot_resolution

  let resolution = resolutionData;
  const resolutionHolder = h("div");
  const resStatus = h("div", { "aria-live": "polite" });
  let resolutionBusy = false;

  async function refreshSlots() {
    try {
      const fresh = await ctx.api.get(["slots"]);
      for (const s of list(fresh && fresh.slots)) if (rowNodes.has(s.name)) place(s);
    } catch {
      resStatus.append(panel({ tone: "warning" },
        h("p", {}, "Saved, but the slots couldn't be read again: reload the page to see the new effective modes.")));
    }
  }

  function lastSetText(set) {
    if (!isObj(set)) return "The audit log has no change to generative_enabled.";
    const who = (isObj(set.by) && (set.by.email || set.by.sub)) || "unknown";
    return `Last set to ${onOff(set.value === true).toLowerCase()} by ${who} at ${dash(set.at)}.`;
  }

  function resolutionCard(r) {
    const flows = isObj(r.flows) ? r.flows : {};
    const def = selectOf(MODES.map((m) => [m, m]), r.default_mode ?? undefined,
      { disabled: !editable, "data-field": "slot_resolution.default_mode" });
    const flowSelects = Object.fromEntries(FLOWS.map((f) => [f, selectOf([["", USE_DEFAULT], ...MODES.map((m) => [m, m])],
      flows[f] ?? "", { disabled: !editable, "data-field": `slot_resolution.flows.${f}.mode` })]));
    const budget = h("input", { type: "number", min: "500", max: "4000", step: "100", value: dash(r.resolver_budget_ms),
      disabled: !editable, "data-field": "slot_resolution.resolver_budget_ms" });
    const noEffect = data.ceiling !== "generative";
    const noEffectNote = noEffect
      ? h("p", { "data-generative-no-effect": "" }, badge(GENERATIVE_NO_EFFECT, "neutral"),
        ` The deployment's ceiling is ${dash(data.ceiling)}, so no slot runs generative whatever this says.`)
      : null;
    let genInput = null;
    let generative;
    if (admin) {
      genInput = h("input", { type: "checkbox", checked: r.generative_enabled === true, disabled: !editable,
        "data-field": "slot_resolution.generative_enabled" });
      generative = h("fieldset", { "data-generative": "admin" },
        h("legend", {}, "Generative mode (admins only)"),
        h("label.inline", {}, genInput, " Allow generative mode"),
        panel({ tone: "warning", title: "The wording is no longer validated text" }, h("p", {}, GENERATIVE_WARNING)),
        noEffectNote,
        h("p.note", {}, lastSetText(r.generative_enabled_set)));
    } else {
      generative = h("div", { "data-generative": "read-only" },
        h("p", {}, "Generative mode: ", h("strong", {}, onOff(r.generative_enabled === true)),
          ". Only admins can change it."),
        noEffectNote,
        h("p.note", {}, lastSetText(r.generative_enabled_set)));
    }
    const warnings = list(r.warnings).map((w) => panel({ tone: "warning", title: warningTitle(w.code) }, h("p", {}, w.message)));
    const form = h("form", { "data-form": "slot_resolution", novalidate: true },
      warnings,
      field("Default mode", def, "The mode of every flow and slot that doesn't set its own.").wrap,
      h("fieldset", {},
        h("legend", {}, "Flow overrides"),
        h("div.field-grid", {}, FLOWS.map((f) => field([h("code", {}, f), " flow"], flowSelects[f]).wrap))),
      field("Resolver budget (ms)", budget, "How long the model may take for one answer, 500 to 4000 ms.").wrap,
      generative,
      resStatus,
      editable ? h("button.primary", { type: "submit" }, "Save") : null);

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (resolutionBusy) return;
      const was = resolution;
      const wasFlows = isObj(was.flows) ? was.flows : {};
      const changes = {};
      const details = [];
      if (def.value !== was.default_mode) {
        changes.default_mode = def.value;
        details.push(`default_mode: ${modeName(was.default_mode)} -> ${def.value}`);
      }
      const nextFlows = {};
      for (const f of FLOWS) if (flowSelects[f].value) nextFlows[f] = { mode: flowSelects[f].value };
      const flowChanged = FLOWS.filter((f) => (wasFlows[f] ?? "") !== flowSelects[f].value);
      if (flowChanged.length) {
        changes.flows = nextFlows;
        for (const f of flowChanged) details.push(`flows.${f}: ${modeName(wasFlows[f])} -> ${modeName(flowSelects[f].value)}`);
      }
      const n = parseIntField(budget.value);
      if (n === null) {
        markInvalid(budget, "must be a whole number of milliseconds");
        budget.focus();
        return;
      }
      if (n !== was.resolver_budget_ms) {
        changes.resolver_budget_ms = n;
        details.push(`resolver_budget_ms: ${dash(was.resolver_budget_ms)} -> ${n}`);
      }
      if (genInput && genInput.checked !== (was.generative_enabled === true)) {
        changes.generative_enabled = genInput.checked;
        details.push(`generative_enabled: ${was.generative_enabled === true} -> ${genInput.checked}`);
      }
      if (!Object.keys(changes).length) {
        clear(resStatus, panel({ tone: "info" }, "Nothing to save: no setting changed."));
        return;
      }
      resolutionBusy = true;
      try {
        const modeChange = Object.fromEntries(["default_mode", "flows", "generative_enabled"]
          .filter((k) => Object.hasOwn(changes, k)).map((k) => [k, changes[k]]));
        if (Object.keys(modeChange).length) {
          const preview = await runPreview({ slot_resolution: modeChange }, form, resStatus);
          if (!preview) return;
          const ok = await confirmDialog(previewConfirm(preview, {
            title: "Save the slot resolution settings?", details, generativeOn: changes.generative_enabled === true,
          }));
          if (!ok) return;
        }
        const yours = { rev: was.rev, ...changes };
        const res = await save(resStatus, () => ctx.api.put(["slots", "resolution"], yours),
          { yours, form, button: ev.submitter, onReload: () => slotsView(main, ctx) });
        if (!res) return;
        if (isObj(res.slot_resolution)) {
          resolution = res.slot_resolution;
          clear(resolutionHolder, resolutionCard(resolution));
        }
        showSaveWarnings(res, "slot_resolution", resStatus);
        await refreshSlots();
      } finally {
        resolutionBusy = false;
      }
    });

    const el = card({
      title: "Slot resolution",
      description: "How answers are resolved when a slot doesn't set its own mode. A mode change shows which slots "
        + "would then call a model, and asks first.",
    }, form);
    el.setAttribute("data-resolution", "");
    return el;
  }

  // ---------------------------------------------------------------- the page

  for (const s of slots) place(s);
  if (resolution) clear(resolutionHolder, resolutionCard(resolution));
  for (const s of choiceSlots) {
    const holder = h("div", { "data-options-holder": s.name }, optionsCard(s));
    optionHolders.set(s.name, holder);
  }
  clear(main,
    header,
    ceilingPanel(data.acceptSpokenFactors),
    modePanel,
    problemsPanel,
    status,
    card({
      description: "Each switch and the confirm select save at once. Turning sensitive or keypad_only off is a "
        + "security change and asks first; a mode change shows its effect and asks first.",
      flush: true,
      label: "Slots",
    }, tableWrap(table)),
    resolution ? resolutionHolder : null,
    [...optionHolders.values()],
    editable ? null : h("p.note", {}, "Your group can view only."),
    takesEffect(data.takes_effect_s));
}
