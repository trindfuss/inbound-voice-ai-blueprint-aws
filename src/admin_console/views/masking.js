// Masking view: the policy-wide masking values and the acknowledgement pool.
// Layout: a page header with History and Save, the "Policy-wide values" card
// with a two-column field grid, and the acknowledgement pool card. Fields, controls, validation
// messages and API calls are as before.

import { badge, card, emptyState, langBadge, pageHeader, panel, tableWrap } from "../lib/components.js";
import { clear, field, h, icon, markInvalid } from "../lib/dom.js";
import { canEdit, errorPanel, load, save, takesEffect } from "../lib/ui.js";
import { checkMaskingForm, MASKING_FIELDS } from "../lib/validate.js";
import { promptEditor } from "./prompts.js";

export async function maskingView(main, ctx) {
  const head = { title: "Masking", description: "Policy-wide masking values and the acknowledgement pool." };
  const data = await load(main, () => ctx.api.get(["masking"]), head);
  if (!data) return;
  const editable = canEdit(ctx.user);
  const own = data.masking || {};
  const controls = {};

  const fields = Object.entries(MASKING_FIELDS).map(([name, spec]) => {
    const isSet = name in own;
    const value = isSet ? own[name] : data.effective[name];
    const input = spec.type === "enum"
      ? h("select", { "data-field": `masking.${name}`, disabled: !editable || !isSet },
        spec.values.map((v) => h("option", { value: v }, v)))
      : h("input", {
        type: "number", step: spec.type === "number" ? "0.5" : "1",
        min: String(spec.range[0]), max: String(spec.range[1]),
        "data-field": `masking.${name}`, disabled: !editable || !isSet,
      });
    input.value = String(value);
    const useDefault = h("input", { type: "checkbox", checked: !isSet, disabled: !editable });
    useDefault.addEventListener("change", () => {
      input.disabled = useDefault.checked || !editable;
      if (useDefault.checked) input.value = String(data.defaults[name]);
    });
    controls[name] = { input, useDefault };
    const range = spec.type === "enum" ? spec.values.join(" or ") : `${spec.range[0]} to ${spec.range[1]}`;
    const f = field(spec.label, input, `${name}: ${range}. Default ${data.defaults[name]}.`);
    f.wrap.append(h("label.inline", {}, useDefault, " Use the default"));
    return f.wrap;
  });

  const status = h("div");
  const form = h("form#masking-form", { novalidate: true },
    status,
    card({ title: "Policy-wide values" }, h("div.field-grid", {}, fields)),
    editable ? null : h("p.note", {}, "Your group can view only."));
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const values = Object.fromEntries(Object.entries(controls).map(([n, c]) => [n, c.useDefault.checked ? undefined : c.input.value]));
    const { errors, masking } = checkMaskingForm(values);
    if (errors.length) {
      clear(status, errorPanel({ title: "Check the form", message: errors.map((e) => `${e.field}: ${e.message}`).join("; ") }));
      for (const e of errors) {
        const el = form.querySelector(`[data-field="${CSS.escape(e.field)}"]`);
        if (el) markInvalid(el, e.message);
      }
      return;
    }
    const yours = { rev: data.rev, masking };
    const res = await save(status, () => ctx.api.put(["masking"], yours),
      { yours, form, button: ev.submitter, onReload: () => maskingView(main, ctx) });
    if (res && typeof res.rev === "number") data.rev = res.rev;
  });

  const editor = h("div");
  const langs = data.ack_pool.length ? Object.keys(data.ack_pool[0].texts) : [];
  const poolTable = h("table", { "aria-label": "Acknowledgement pool" },
    h("thead", {}, h("tr", {},
      h("th", { scope: "col" }, "Key"),
      langs.map((l) => h("th", { scope: "col" }, langBadge(l))),
      h("th", { scope: "col" }, h("span.visually-hidden", {}, "Actions")))),
    h("tbody", {}, data.ack_pool.map((a) => h("tr", {},
      h("th", { scope: "row" }, h("code", {}, a.key)),
      langs.map((l) => h("td", {}, a.texts[l] ?? badge("missing", "warning"))),
      h("td", {}, h("button.sm", {
        type: "button",
        on: {
          click: async () => {
            const found = await load(editor, () => ctx.api.get(["prompts"], { prefix: a.key }), { kind: "form" });
            const prompt = found && found.prompts.find((p) => p.key === a.key);
            if (prompt) clear(editor, promptEditor(ctx, prompt, found.languages, { onSaved: () => maskingView(main, ctx) }));
            else if (found) clear(editor, panel({ tone: "warning" }, h("p", {}, `${a.key} isn't in the catalog yet.`)));
          },
        },
      }, editable ? "Edit" : "View"))))));
  const pool = card({
    title: "Acknowledgement pool",
    description: "Said when a wait is long enough, before a tool's own phrase.",
    flush: data.ack_pool.length > 0,
  }, data.ack_pool.length
    ? tableWrap(poolTable)
    : emptyState({ icon: "masking", title: "No acknowledgements", message: "The acknowledgement pool is empty." }));

  const history = h("a.button.quiet", { href: `#/audit?item=${encodeURIComponent("policy#masking")}` }, icon("history"), "History");
  const saveButton = editable ? h("button.primary", { type: "submit", form: "masking-form" }, "Save") : null;
  clear(main,
    pageHeader({
      ...head,
      actions: [history, saveButton].filter(Boolean),
    }),
    data.problems.length ? panel({ tone: "warning", title: "Problems" }, h("ul", {}, data.problems.map((p) => h("li", {}, p)))) : null,
    form, pool, editor, takesEffect(data.takes_effect_s));
}
