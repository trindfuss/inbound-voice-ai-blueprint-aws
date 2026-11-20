// Redaction view.
// Admins only: app.js's SECTIONS entry has when: isAdmin, so the nav hides it and the route shows
// the denied empty state for anyone else; the controls here are also disabled for a non-admin.
//
// GET /redaction gives the stored item, the config the components run with (`effective`: the
// defaults when the item is missing or invalid, with the field named in `problems`), the entity
// types Comprehend offers and the on_failure values. The form shows `effective`: the enabled
// switch, the entity types as checkboxes grouped by type, each group under a heading checkbox that
// selects or clears the group (indeterminate, with a visible "n of m" count, when mixed), and
// min_confidence, mask and on_failure. Save PUTs the whole edited config with the rev read.
//
// The switch and an empty selection. The server refuses enabled: true with no
// entity types. No types means off: the switch shows `enabled AND at least one type`, so a stored
// (false, []) loads with the switch off and the state line "Redaction is off (no entity types
// selected)"; (false, types) is the switch off with its types kept for when it is switched back on.
// The form never holds the switch on with nothing ticked: clearing the last type turns the switch
// off, and switching on with nothing ticked ticks the default types (or, with none, leaves it off).
// So every state the form shows is the one it saves, and reads back the same after the save; the
// PUT's enabled is still `switch on AND at least one type` (never a body the server refuses), and a
// save with nothing ticked sends enabled: false, entity_types: [].
//
// Weakening edits: switching off (enabled going from true to false, which includes
// clearing every type), removing a type, raising min_confidence, and setting on_failure to allow
// each show a warning as soon as the edit is made, and a danger confirmDialog before the save;
// Cancel sends nothing and leaves the edit in the form.

import { badge, card, pageHeader, panel } from "../lib/components.js";
import { clear, field, h, icon, markInvalid } from "../lib/dom.js";
import { confirmDialog, isAdmin, load, save, takesEffect } from "../lib/ui.js";

/**
 * The entity-type groups, in display order. Every type in pii.COMPREHEND_PII_ENTITY_TYPES is
 * in one; a type the API offers that isn't listed here goes under "Other".
 */
export const ENTITY_GROUPS = [
  { name: "Financial", types: ["CREDIT_DEBIT_NUMBER", "CREDIT_DEBIT_CVV", "CREDIT_DEBIT_EXPIRY", "BANK_ACCOUNT_NUMBER",
    "BANK_ROUTING", "INTERNATIONAL_BANK_ACCOUNT_NUMBER", "SWIFT_CODE"] },
  { name: "Credentials", types: ["PIN", "PASSWORD", "USERNAME", "AWS_ACCESS_KEY", "AWS_SECRET_KEY"] },
  { name: "Identity", types: ["SSN", "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "PASSPORT_NUMBER", "DRIVER_ID",
    "CA_SOCIAL_INSURANCE_NUMBER", "CA_HEALTH_NUMBER", "UK_NATIONAL_INSURANCE_NUMBER", "UK_NATIONAL_HEALTH_SERVICE_NUMBER",
    "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER", "IN_AADHAAR", "IN_PERMANENT_ACCOUNT_NUMBER", "IN_VOTER_NUMBER", "IN_NREGA"] },
  { name: "Contact", types: ["NAME", "ADDRESS", "PHONE", "EMAIL"] },
  { name: "Personal", types: ["AGE", "DATE_TIME"] },
  { name: "Vehicle", types: ["LICENSE_PLATE", "VEHICLE_IDENTIFICATION_NUMBER"] },
  { name: "Network", types: ["IP_ADDRESS", "MAC_ADDRESS", "URL"] },
];
const OTHER = "Other";

/**
 * The groups to show for the types the API offers plus any configured type it doesn't list:
 * [{name, types}] in ENTITY_GROUPS order, empty groups left out, unknown types under "Other".
 */
export function groupTypes(available, configured = []) {
  const all = [...new Set([...available, ...configured].filter((t) => typeof t === "string" && t !== ""))];
  const known = new Set(ENTITY_GROUPS.flatMap((g) => g.types));
  const groups = ENTITY_GROUPS.map((g) => ({ name: g.name, types: g.types.filter((t) => all.includes(t)) }));
  groups.push({ name: OTHER, types: all.filter((t) => !known.has(t)).sort() });
  return groups.filter((g) => g.types.length);
}

/** Shown before saving an edit that switches redaction off. */
export const OFF_WARNING = "Switching transcript redaction off is a security change: the AI turns in the call log, the "
  + "escalation summary and the screen-pop timeline are then stored and shown without PII masking. A sensitive "
  + "slot's value is still masked.";
/** Shown before saving an edit that removes an entity type. */
export const REMOVED_WARNING = "Removing an entity type is a security change: text of that type is then no longer "
  + "masked in the call log, the escalation summary or the screen-pop timeline.";
/** Shown before saving an edit that raises min_confidence. */
export const CONFIDENCE_WARNING = "Raising the minimum confidence is a security change: text Comprehend is less sure "
  + "about is then no longer masked in the call log, the escalation summary or the screen-pop timeline.";
/** Shown before saving an edit that sets on_failure to allow. */
export const ALLOW_WARNING = "Setting \"If the check fails\" to allow is a security change: when Comprehend can't be "
  + "reached, the text is stored and shown unmasked instead of being withheld.";
/** The state line when no type is selected (no types means off). */
export const OFF_NO_TYPES = "Redaction is off (no entity types selected)";

const ON_FAILURE_TEXT = {
  withhold: "withhold: keep the text out until it can be checked",
  allow: "allow: keep the text unmasked",
};

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const list = (v) => (Array.isArray(v) ? v : []);
const strings = (v) => list(v).filter((x) => typeof x === "string");

/**
 * The config the form starts from (GET's `effective`, normalized for the controls). enabled is
 * true only with at least one type, so the start is always a state the form can show.
 */
function startOf(view) {
  const e = isObj(view.effective) ? view.effective : {};
  const entity_types = strings(e.entity_types);
  return {
    enabled: e.enabled === true && entity_types.length > 0,
    entity_types,
    min_confidence: e.min_confidence === undefined || e.min_confidence === null ? "" : String(e.min_confidence),
    mask: typeof e.mask === "string" ? e.mask : "",
    on_failure: typeof e.on_failure === "string" ? e.on_failure : "",
  };
}

/** Whether the edited config differs from the one shown (min_confidence compared as a number). */
function differs(a, b) {
  return a.enabled !== b.enabled
    || a.entity_types.length !== b.entity_types.length || a.entity_types.some((t, i) => t !== b.entity_types[i])
    || Number(a.min_confidence) !== Number(b.min_confidence)
    || a.mask !== b.mask || a.on_failure !== b.on_failure;
}

/**
 * The weakening parts of an edit: switching off, the removed types (sorted), a raised
 * min_confidence (compared as numbers) and on_failure newly set to allow.
 */
function weakening(start, next) {
  const from = Number(start.min_confidence);
  const to = Number(next.min_confidence);
  const w = {
    off: start.enabled && !next.enabled,
    removed: start.entity_types.filter((t) => !next.entity_types.includes(t)).sort(),
    raised: next.min_confidence !== "" && start.min_confidence !== "" && Number.isFinite(from)
      && Number.isFinite(to) && to > from,
    allow: next.on_failure === "allow" && start.on_failure !== "allow",
  };
  w.any = w.off || w.removed.length > 0 || w.raised || w.allow;
  return w;
}

export async function redactionView(main, ctx) {
  const head = {
    title: "Redaction",
    description: "Which kinds of personal data are masked in the call log, the escalation summary and the "
      + "screen-pop timeline.",
  };
  const data = await load(main, () => ctx.api.get(["redaction"]), head);
  if (!data) return;
  const editable = isAdmin(ctx.user);
  const history = h("a.button.quiet", { href: `#/audit?item=${encodeURIComponent("policy#redaction")}` }, icon("history"), "History");
  const savedHolder = h("div", { "aria-live": "polite" });
  const body = h("div");
  clear(main, pageHeader({ ...head, actions: [history] }), savedHolder, body, takesEffect(data.takes_effect_s));
  paint(data);

  function paint(view) {
    const start = startOf(view);
    const groups = groupTypes(strings(view.entity_types_available), start.entity_types);
    const boxes = new Map(); // type -> checkbox
    const heads = new Map(); // group name -> heading checkbox
    const counts = new Map(); // group name -> its "n of m" text

    // ---- the problems the API reports, and its warnings about the stored config
    const problems = list(view.problems).filter(isObj);
    const problemsPanel = problems.length
      ? panel({ tone: "warning", title: "Problems" },
        h("p", {}, "The stored setting can't be read, so the components run the built-in defaults shown below."),
        h("ul", {}, problems.map((p) => h("li", {}, String(p.message ?? p.field)))))
      : null;
    if (problemsPanel) problemsPanel.setAttribute("data-problems", "redaction");
    const serverWarnings = list(view.warnings).filter(isObj);
    const serverPanel = serverWarnings.length
      ? panel({ tone: "warning", title: "Redaction is off" }, serverWarnings.map((w) => h("p", {}, String(w.message))))
      : null;
    if (serverPanel) serverPanel.setAttribute("data-server-warnings", "");

    // ---- the switch
    const sw = h("input.switch", {
      type: "checkbox", role: "switch", checked: start.enabled,
      disabled: !editable, "data-field": "redaction.enabled",
    });
    const swText = h("span", { "aria-hidden": "true" }, sw.checked ? "On" : "Off");
    const stateLine = h("p.note", { "data-redaction-state": "" });

    // ---- the entity types
    const groupSets = groups.map((g) => {
      const headBox = h("input", { type: "checkbox", disabled: !editable, "data-entity-group": g.name });
      const count = h("span.note", {});
      heads.set(g.name, headBox);
      counts.set(g.name, count);
      headBox.addEventListener("change", () => {
        for (const t of g.types) boxes.get(t).checked = headBox.checked;
        sync();
      });
      return h("fieldset", { "data-group": g.name },
        h("legend", {}, h("label.inline", {}, headBox, h("span", {}, g.name), count)),
        g.types.map((t) => {
          const box = h("input", { type: "checkbox", checked: start.entity_types.includes(t), disabled: !editable,
            "data-entity-type": t });
          boxes.set(t, box);
          box.addEventListener("change", sync);
          return h("label.inline", {}, box, h("code", {}, t));
        }));
    });
    const typesSet = h("fieldset", { "data-field": "redaction.entity_types", tabindex: "-1" },
      h("legend", {}, "Entity types to mask"),
      h("p.hint", {}, "Tick a group's heading to select or clear the whole group. With none selected, nothing is "
        + "redacted."),
      h("p.note", { "data-selected-count": "" }),
      h("div.field-grid", {}, groupSets));

    // ---- detection settings
    const conf = h("input", { type: "number", min: "0", max: "1", step: "0.05", value: start.min_confidence,
      disabled: !editable, "data-field": "redaction.min_confidence" });
    const mask = h("input", { type: "text", value: start.mask, disabled: !editable, "data-field": "redaction.mask" });
    const values = [...new Set([...strings(view.on_failure_values), start.on_failure].filter(Boolean))];
    const onFailure = h("select", { disabled: !editable, "data-field": "redaction.on_failure" },
      values.map((v) => h("option", { value: v, selected: v === start.on_failure }, ON_FAILURE_TEXT[v] ?? v)));

    const pending = h("div", { "data-pending-warnings": "" });
    const status = h("div", { "aria-live": "polite" });
    const submitButton = editable ? h("button.primary", { type: "submit" }, "Save") : null;
    const form = h("form", { "data-form": "redaction", novalidate: true },
      card({ title: "Transcript redaction", description: "Detected with Amazon Comprehend before the text is stored." },
        h("label.inline", {}, sw, h("span", {}, "Redact transcripts"), swText),
        stateLine),
      card({ title: "Entity types" }, typesSet),
      card({ title: "Detection" }, h("div.field-grid", {},
        field("Minimum confidence", conf, "0 to 1: how sure Comprehend must be before text is masked.").wrap,
        field("Mask", mask, "The text that replaces a match; {type} is the entity type, the only placeholder allowed.").wrap,
        field("If the check fails", onFailure, "What happens to the text when Comprehend can't be reached.").wrap)),
      pending,
      status,
      submitButton,
      editable ? null : h("p.note", {}, "Only admins can change redaction."));

    /** The config the form now holds, as the PUT sends it (without rev). */
    function edited() {
      const ticked = new Set([...boxes].filter(([, b]) => b.checked).map(([t]) => t));
      const entity_types = [...start.entity_types.filter((t) => ticked.has(t)),
        ...[...boxes.keys()].filter((t) => ticked.has(t) && !start.entity_types.includes(t))];
      return {
        enabled: sw.checked && entity_types.length > 0, // never (true, []): the server refuses it
        entity_types,
        min_confidence: conf.value.trim(),
        mask: mask.value,
        on_failure: onFailure.value,
      };
    }

    const anyTicked = () => [...boxes.values()].some((b) => b.checked);

    function sync() {
      if (!anyTicked()) sw.checked = false; // No types means off, so the switch is never on alone
      swText.textContent = sw.checked ? "On" : "Off";
      for (const g of groups) {
        const n = g.types.filter((t) => boxes.get(t).checked).length;
        const box = heads.get(g.name);
        box.checked = n === g.types.length;
        box.indeterminate = n > 0 && n < g.types.length;
        counts.get(g.name).textContent = `${n} of ${g.types.length}`;
      }
      const next = edited();
      const state = next.enabled ? "on" : "off";
      stateLine.setAttribute("data-redaction-state", state);
      stateLine.textContent = next.enabled
        ? `On: ${next.entity_types.length} entity type(s) are masked.`
        : next.entity_types.length
          ? "Off: nothing is redacted. The selected types are kept for when it is switched back on."
          : OFF_NO_TYPES;
      typesSet.querySelector("[data-selected-count]").textContent = `${next.entity_types.length} selected.`;
      const w = weakening(start, next);
      clear(pending,
        w.off ? panel({ tone: "warning", title: "Redaction will be off" },
          h("p", {}, OFF_WARNING), next.entity_types.length ? null : h("p", {}, OFF_NO_TYPES)) : null,
        w.removed.length ? panel({ tone: "warning", title: "Types will no longer be masked" },
          h("p", {}, REMOVED_WARNING), h("p", {}, "Removed: ", w.removed.map((t, i) => [i ? ", " : null, h("code", {}, t)]))) : null,
        w.raised ? panel({ tone: "warning", title: "Less will be masked" }, h("p", {}, CONFIDENCE_WARNING),
          h("p", {}, `Minimum confidence: ${start.min_confidence} -> ${next.min_confidence}`)) : null,
        w.allow ? panel({ tone: "warning", title: "Unchecked text will be kept" }, h("p", {}, ALLOW_WARNING)) : null);
    }
    sw.addEventListener("change", () => {
      // Switching on with nothing ticked ticks the default types, so the switch is never on alone.
      if (sw.checked && !anyTicked()) {
        const defaults = isObj(view.defaults) ? strings(view.defaults.entity_types) : [];
        for (const t of defaults) if (boxes.has(t)) boxes.get(t).checked = true;
      }
      sync();
    });
    for (const el of [conf, mask, onFailure]) el.addEventListener("change", sync);

    let busy = false;
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (busy || !editable) return;
      const next = edited();
      const n = Number(next.min_confidence);
      if (next.min_confidence === "" || !Number.isFinite(n) || n < 0 || n > 1) {
        markInvalid(conf, "must be a number from 0 to 1");
        conf.focus();
        return;
      }
      if (next.mask === "") {
        markInvalid(mask, "must not be empty");
        mask.focus();
        return;
      }
      if (!differs(next, start)) {
        clear(status, panel({ tone: "info" }, "Nothing to save: no setting changed."));
        return;
      }
      busy = true;
      try {
        const w = weakening(start, next);
        if (w.any) {
          const ok = await confirmDialog({
            title: w.off ? "Switch transcript redaction off?"
              : w.removed.length ? "Stop masking these entity types?" : "Weaken transcript redaction?",
            message: [w.off ? OFF_WARNING : null, w.off && !next.entity_types.length ? `${OFF_NO_TYPES}.` : null,
              w.removed.length ? REMOVED_WARNING : null, w.raised ? CONFIDENCE_WARNING : null,
              w.allow ? ALLOW_WARNING : null, "The audit log marks this as a security change."]
              .filter(Boolean).join(" "),
            details: [w.off ? "enabled: true -> false" : null,
              w.removed.length ? `entity_types: ${w.removed.join(", ")} removed` : null,
              w.raised ? `min_confidence: ${start.min_confidence} -> ${next.min_confidence}` : null,
              w.allow ? `on_failure: ${start.on_failure} -> allow` : null].filter(Boolean),
            confirmLabel: "Save",
            danger: true,
          });
          if (!ok) return;
        }
        const yours = { rev: view.rev, ...next };
        const res = await save(status, async () => {
          try {
            return await ctx.api.put(["redaction"], yours);
          } catch (err) {
            // Compare the config the components run with, not the whole view, next to yours.
            if (err && err.kind === "conflict" && isObj(err.current) && isObj(err.current.effective)) {
              err.current = { rev: err.current.rev, ...err.current.effective };
            }
            throw err;
          }
        }, { yours, form, button: ev.submitter, onReload: () => redactionView(main, ctx) });
        if (!res) return;
        if (isObj(res.effective)) paint(res);
        showSaved(res);
      } finally {
        busy = false;
      }
    });

    clear(body, problemsPanel, serverPanel, form);
    sync();
  }

  function showSaved(res) {
    const warnings = list(res && res.warnings).filter(isObj);
    if (!warnings.length) {
      clear(savedHolder);
      return;
    }
    const box = panel({ tone: "warning", title: res.security_change ? "Saved: a security change" : "Saved, with warnings" },
      res.security_change ? h("p", {}, badge("security change", "danger")) : null,
      h("ul", {}, warnings.map((w) => h("li", {}, String(w.message),
        strings(w.entity_types).length ? [" (", strings(w.entity_types).join(", "), ")"] : null))));
    box.setAttribute("data-save-warnings", "");
    clear(savedHolder, box);
  }
}
