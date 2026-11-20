// Tools view: every tool's settings, and a detail page to edit one.
// Layout: a page header, the list in a flush card, the detail's Permissions and
// Masking cards. Fields, controls, validation messages and API calls are as before.

import { badge, card, emptyState, langBadge, pageHeader, panel, tableWrap } from "../lib/components.js";
import { clear, field, h, icon, markInvalid } from "../lib/dom.js";
import { canEdit, confirmDialog, errorPanel, load, save, sourceMark, takesEffect } from "../lib/ui.js";
import { checkToolForm, isSecurityChange } from "../lib/validate.js";

function delayCell(d) {
  return d ? [String(d.value), " ", sourceMark(d.source)] : "-";
}

async function toolsList(main, ctx) {
  const head = { title: "Tools", description: "Each tool's verification level, confirmation and masking." };
  const data = await load(main, () => ctx.api.get(["tools"]), head);
  if (!data) return;
  const header = pageHeader(head);
  if (!data.tools.length) {
    clear(main, header,
      card({ label: "Tools" }, emptyState({ icon: "tools", title: "No tools", message: "Seed the policy table to list tools." })));
    return;
  }
  const rows = data.tools.map((t) => h("tr", {},
    h("th", { scope: "row" }, h("a", { href: `#/tools/${encodeURIComponent(t.name)}` }, t.name)),
    h("td", {}, t.kind ?? "-"),
    h("td.num", {}, String(t.min_level ?? "-")),
    h("td", {}, t.confirm ?? "-", " ", sourceMark(t.confirm_source)),
    h("td", {}, t.mask ? badge("on", "success") : badge("off", "neutral")),
    h("td", {}, t.mask ? h("code", {}, t.mask.phrase_key) : "-"),
    h("td.num", {}, delayCell(t.mask && t.mask.after_ms)),
    h("td.num", {}, delayCell(t.mask && t.mask.handoff_wait_ms)),
    h("td", {}, t.problems.length ? badge(`${t.problems.length} problem(s)`, "warning") : "none")));
  const NUM = new Set(["Min level", "after_ms", "handoff_wait_ms"]);
  clear(main,
    header,
    card({ description: "\"own\" is the tool's value; \"default\" is the policy-wide value.", flush: true, label: "Tools" },
      tableWrap(h("table", {},
        h("caption.visually-hidden", {}, "Tools and their settings"),
        h("thead", {}, h("tr", {}, ["Tool", "Kind", "Min level", "Confirmation", "Masking", "Phrase set",
          "after_ms", "handoff_wait_ms", "Problems"].map((c) => h("th", { scope: "col", class: NUM.has(c) ? "num" : null }, c)))),
        h("tbody", {}, rows)))),
    takesEffect(data.takes_effect_s));
}

/** Mark each control a client-side error names; `toField` maps the error's field to data-field. */
function markErrors(form, errors, toField) {
  for (const e of errors) {
    const el = form.querySelector(`[data-field="${CSS.escape(toField(e.field))}"]`);
    if (el) markInvalid(el, e.message);
  }
}

async function toolDetail(main, ctx, name) {
  const back = { href: "#/tools", label: "All tools" };
  const data = await load(main, () => ctx.api.get(["tools", name]), { title: name, back });
  if (!data) return;
  const tool = data.tool;
  const editable = canEdit(ctx.user);
  const mask = tool.mask;

  const minLevel = field("Minimum verification level", h("input", {
    type: "number", min: "0", step: "1", value: tool.min_level ?? "", "data-field": `tools.${name}.min_level`, required: true, disabled: !editable,
  }), "The caller must have reached this level before the tool runs.");
  const confirmSel = h("select", { "data-field": "confirm", disabled: !editable },
    h("option", { value: "" }, `Default (${tool.confirm_source === "default" ? tool.confirm : "policy default"})`),
    h("option", { value: "none" }, "none"),
    h("option", { value: "deterministic" }, "deterministic (scripted yes/no)"));
  confirmSel.value = tool.confirm_source === "tool" ? (tool.confirm ?? "") : "";
  const confirm = field("Confirmation", confirmSel, "Whether the caller confirms before the tool runs.");

  const masked = h("input.switch", { type: "checkbox", role: "switch", checked: Boolean(mask), disabled: !editable });
  const maskedState = h("span", { "aria-hidden": "true" }, mask ? "On" : "Off");
  masked.addEventListener("change", () => { maskedState.textContent = masked.checked ? "On" : "Off"; });
  const phraseKey = field("Phrase key", h("input", {
    type: "text", value: mask ? mask.phrase_key : `mask.${name}`, "data-field": `tools.${name}.mask.phrase_key`, disabled: !editable,
  }), "The catalog key of the tool's phrase pool (mask.<tool>, then .1, .2 ...).");

  function delayControl(key, label, d) {
    const own = d && d.source === "tool";
    const useDefault = h("input", { type: "checkbox", checked: !own, disabled: !editable });
    const input = h("input", { type: "number", min: "0", max: "10000", step: "50", value: d ? d.value : "",
      "data-field": `tools.${name}.mask.${key}`, disabled: !editable || !own });
    useDefault.addEventListener("change", () => {
      input.disabled = useDefault.checked || !editable;
    });
    const f = field(label, input, "0 to 10000 ms.");
    f.wrap.append(h("label.inline", {}, useDefault, " Use the policy-wide default"));
    return { wrap: f.wrap, value: () => (useDefault.checked ? undefined : input.value) };
  }
  const after = delayControl("after_ms", "Phrase after (after_ms)", mask && mask.after_ms);
  const handoff = delayControl("handoff_wait_ms", "Hand-off wait (handoff_wait_ms)", mask && mask.handoff_wait_ms);

  const status = h("div", { "aria-live": "polite" });
  const form = h("form#tool-form", { novalidate: true },
    status,
    card({ title: "Permissions" }, minLevel.wrap, confirm.wrap),
    card({ title: "Masking" },
      h("div.field", {}, h("label.inline", {}, masked, " Mask this tool's wait with a phrase"), maskedState),
      phraseKey.wrap, after.wrap, handoff.wrap),
    editable ? null : h("p.note", {}, "Your group can view only."));

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const { errors, body } = checkToolForm({
      min_level: minLevel.input.value, confirm: confirmSel.value, masked: masked.checked,
      phrase_key: phraseKey.input.value, after_ms: after.value(), handoff_wait_ms: handoff.value(),
    });
    if (errors.length) {
      clear(status, errorPanel({ title: "Check the form", message: errors.map((e) => `${e.field}: ${e.message}`).join("; ") }));
      markErrors(form, errors, (f) => (f === "confirm" ? f : `tools.${name}.${f}`));
      return;
    }
    if (isSecurityChange(tool, body)) {
      const ok = await confirmDialog({
        title: "This changes security behaviour",
        message: `You are changing ${name}'s verification level or confirmation. Callers will be checked differently from the next turn. The audit log marks this as a security change.`,
        details: [`Minimum level: ${tool.min_level} -> ${body.min_level}`,
          `Confirmation: ${tool.confirm_source === "tool" ? tool.confirm : "default"} -> ${body.confirm ?? "default"}`],
        confirmLabel: "Save security change", danger: true,
      });
      if (!ok) return;
    }
    const yours = { rev: tool.rev, ...body };
    const res = await save(status, () => ctx.api.put(["tools", name], yours),
      { yours, form, button: ev.submitter, onReload: () => toolDetail(main, ctx, name) });
    if (res && res.tool) Object.assign(tool, res.tool); // the new rev, level and confirmation
  });

  const pool = mask ? card({ title: "Phrase pool" },
    mask.pool.length ? h("ul", {}, mask.pool.map((p) => h("li", {}, h("code", {}, p.key),
      Object.entries(p.texts).map(([lang, t]) => h("div", {}, langBadge(lang), " ", t ?? badge("missing", "warning"))))))
      : panel({ tone: "warning" }, h("p", {}, "No phrases in the catalog for this key.")),
    h("p", {}, h("a", { href: `#/prompts?tool=${encodeURIComponent(name)}` }, "Edit the phrases and preview them"))) : null;

  const placeholders = (tool.placeholders || []).length ? card({ title: "Placeholders" },
    h("p", {}, "A phrase may use: ", tool.placeholders.map((p) => [h("code", {}, p.placeholder), " "]))) : null;

  const history = h("a.button.quiet", { href: `#/audit?item=${encodeURIComponent(`policy#tool#${name}`)}` }, icon("history"), "History");
  const saveButton = editable ? h("button.primary", { type: "submit", form: "tool-form" }, "Save") : null;
  clear(main,
    pageHeader({
      back,
      title: name,
      description: `Kind: ${tool.kind ?? "-"}`,
      actions: [history, saveButton].filter(Boolean),
    }),
    tool.problems.length ? panel({ tone: "warning", title: "Problems" }, h("ul", {}, tool.problems.map((p) => h("li", {}, p)))) : null,
    form, pool, placeholders, takesEffect(data.takes_effect_s));
}

/** The Tools section's render: the detail page for #/tools/<name>, else the list. */
export const toolsView = (main, ctx, { arg } = {}) => (arg ? toolDetail(main, ctx, arg) : toolsList(main, ctx));
