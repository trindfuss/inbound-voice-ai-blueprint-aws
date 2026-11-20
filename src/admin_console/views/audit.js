// Audit view: an item's changes, newest first, with Restore.
// Layout: a page header, the search card (field and button on one row), empty
// states, and the changes table with security changes marked. The search field, the
// Restore flow and the API calls are as before.

import { badge, card, emptyState, pageHeader, tableWrap } from "../lib/components.js";
import { clear, field, h, markInvalid, show } from "../lib/dom.js";
import { canEdit, confirmDialog, errorPanel, load, save } from "../lib/ui.js";
import { checkAuditItem } from "../lib/validate.js";

/** The item's current rev, which a restore must send (optimistic concurrency). */
async function currentRev(ctx, target) {
  if (target.kind === "tool") return (await ctx.api.get(["tools", target.id])).tool.rev;
  if (target.kind === "masking") return (await ctx.api.get(["masking"])).rev;
  if (target.kind === "prompt") {
    const found = await ctx.api.get(["prompts"], { prefix: target.id });
    const p = found.prompts.find((x) => x.key === target.id);
    if (!p) throw Object.assign(new Error("The prompt no longer exists."), { title: "Not found" });
    return p.rev;
  }
  throw Object.assign(new Error("Only tool, masking and prompt changes can be restored."), { title: "Not restorable" });
}

async function knownItems(ctx) {
  try {
    const [tools, masks, args] = await Promise.all([
      ctx.api.get(["tools"]), ctx.api.get(["prompts"], { prefix: "mask." }), ctx.api.get(["prompts"], { prefix: "arg." })]);
    return ["policy#masking", ...tools.tools.map((t) => `policy#tool#${t.name}`),
      ...masks.prompts.map((p) => `catalog#${p.key}`), ...args.prompts.map((p) => `catalog#${p.key}`)];
  } catch {
    return ["policy#masking"];
  }
}

/** One changed field: dt code(field), dd with the labelled Before and After values. */
function changeRows(changes) {
  const part = (cls, label, value) => h(`div.${cls}`, {},
    h("span.change-label", {}, label), h("span.change-value", {}, show(value)));
  return h("dl.changes", {}, (changes || []).map((c) => [
    h("dt", {}, h("code", {}, c.field)),
    h("dd", {}, part("change-before", "Before", c.old), part("change-after", "After", c.new)),
  ]));
}

export async function auditView(main, ctx, { params = new URLSearchParams() } = {}) {
  const item = params.get("item") || "";
  const list = h("datalist#audit-items");
  const input = h("input", { type: "text", value: item, list: "audit-items", "data-field": "item", required: true, size: "40" });
  const picker = field("Item", input, "For example policy#tool#get_balance, policy#masking, catalog#mask.get_balance or user#<email>.",
    { requiredNote: false });
  const results = h("div");
  const form = h("form.search-row", { role: "search" }, picker.wrap, list, h("button.primary", { type: "submit" }, "Show changes"));
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const err = checkAuditItem(input.value.trim());
    if (err) {
      clear(results, errorPanel({ title: "Check the item", message: err.message }));
      markInvalid(input, err.message);
      return;
    }
    location.hash = `#/audit?item=${encodeURIComponent(input.value.trim())}`;
  });
  clear(main,
    pageHeader({ title: "Audit log", description: "Changes to an item, newest first, with restore." }),
    card({ label: "Find an item" }, form),
    results);
  knownItems(ctx).then((items) => clear(list, items.map((v) => h("option", { value: v }))));
  if (!item) {
    clear(results, card({ label: "Changes" }, emptyState({
      icon: "history", title: "Choose an item", message: "Enter an item above to see its changes, newest first.",
    })));
    return;
  }

  const data = await load(results, () => ctx.api.get(["audit"], { item, limit: 100 }), { kind: "table" });
  if (!data) return;
  if (!data.entries.length) {
    clear(results, card({ label: "Changes" },
      emptyState({ icon: "history", title: "No changes recorded", message: "No changes recorded for this item." })));
    return;
  }
  const status = h("div");
  const rows = data.entries.map((e) => {
    const restorable = canEdit(ctx.user) && ["tool", "masking", "prompt"].includes(e.target && e.target.kind);
    const restore = restorable ? h("button.sm", {
      type: "button",
      on: {
        click: async (ev) => {
          const button = ev.currentTarget;
          const ok = await confirmDialog({
            title: "Restore the previous values?",
            message: `This writes back the old values of this change to ${e.target.id}, through the same checks as an edit. It is recorded as a new change.`,
            details: (e.changes || []).map((c) => `${c.field}: ${show(c.new)} -> ${show(c.old)}`),
            confirmLabel: "Restore",
          });
          if (!ok) return;
          let rev;
          try {
            rev = await currentRev(ctx, e.target);
          } catch (err) {
            clear(status, errorPanel(err));
            return;
          }
          const yours = { audit_pk: e.pk, audit_sk: e.sk, rev };
          const res = await save(status, () => ctx.api.post(["restore"], yours), { yours, button });
          if (res) auditView(main, ctx, { params });
        },
      },
    }, "Restore") : null;
    return h("tr", { class: e.security_change ? "is-security" : null },
      h("td", {}, h("time", { datetime: e.ts }, e.ts)),
      h("td", {}, e.action, e.restored_from ? [" ", badge("restore", "info")] : null),
      h("td", {}, (e.actor && (e.actor.email || e.actor.sub)) || "-"),
      h("td", {}, changeRows(e.changes)),
      h("td", {}, e.security_change ? badge("Security change", "danger", { icon: "shield-alert" }) : ""),
      h("td", {}, restore));
  });
  const hidden = (text) => h("span.visually-hidden", {}, text);
  clear(results,
    card({ flush: true, label: `Changes to ${item}` },
      tableWrap(h("table", {},
        h("caption", {}, `Changes to ${item}, newest first`),
        h("thead", {}, h("tr", {}, ["When (UTC)", "Action", "Who", "Changes", hidden("Security"), hidden("Restore")]
          .map((c) => h("th", { scope: "col" }, c)))),
        h("tbody", {}, rows)))),
    status);
}
