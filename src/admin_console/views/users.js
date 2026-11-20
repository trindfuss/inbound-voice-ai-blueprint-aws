// Users view: admins only. The navigation hides it from other groups, but
// the API is what refuses them (403).
// Layout: a page header with no main action, the users table in a flush card
// with badges for "you", the status and sign-in, and an "Add a user" card with its own primary
// button. Row actions, the create form, messages and API calls are as before.

import { badge, card, pageHeader, tableWrap } from "../lib/components.js";
import { clear, field, h, markInvalid } from "../lib/dom.js";
import { confirmDialog, errorPanel, load, save } from "../lib/ui.js";
import { checkNewUser, GROUPS } from "../lib/validate.js";

const PENDING = "FORCE_CHANGE_PASSWORD";

export async function usersView(main, ctx) {
  const head = { title: "Users", description: "Who can sign in to the console and what their group allows." };
  const data = await load(main, () => ctx.api.get(["users"]), head);
  if (!data) return;
  const groups = data.groups || GROUPS;
  const status = h("div");
  const reload = () => usersView(main, ctx);
  const me = ctx.user;

  async function act(confirm, fn, button) {
    if (!(await confirmDialog(confirm))) return;
    const res = await save(status, fn, { button });
    if (res) reload();
  }
  /** A small row button that confirms, then runs `fn` with itself as the busy button. */
  const rowButton = (spec, label, confirm, fn) => h(spec, {
    type: "button",
    on: { click: (ev) => act(confirm, fn, ev.currentTarget) },
  }, label);

  const rows = data.users.map((u) => {
    const who = u.email || u.username;
    const self = (me.username && me.username === u.username) || (me.email && me.email === u.email);
    const groupSel = h("select", { "aria-label": `Group of ${who}` }, groups.map((g) => h("option", { value: g }, g)));
    groupSel.value = u.group || "";
    groupSel.addEventListener("change", () => {
      const to = groupSel.value;
      groupSel.value = u.group || "";
      act({ title: "Change group?", message: `${who}: ${u.group || "no group"} -> ${to}. This changes what they can do.`, confirmLabel: "Change group", danger: to === "admin" || u.group === "admin" },
        () => ctx.api.patch(["users", u.username], { group: to }));
    });
    const actions = [
      rowButton("button.sm", u.enabled ? "Disable" : "Enable",
        { title: u.enabled ? "Disable user?" : "Enable user?", message: u.enabled ? `${who} won't be able to sign in until enabled again.` : `${who} will be able to sign in again.`, confirmLabel: u.enabled ? "Disable" : "Enable", danger: u.enabled },
        () => ctx.api.patch(["users", u.username], { enabled: !u.enabled })),
      u.status === PENDING ? rowButton("button.sm", "Resend invitation",
        { title: "Resend the invitation?", message: `Cognito emails ${who} a new temporary password.`, confirmLabel: "Resend" },
        () => ctx.api.post(["users", u.username, "resend"])) : null,
      self ? null : rowButton("button.sm.danger", "Delete",
        { title: "Delete user?", message: `${who} is removed from the console for good. This can't be undone.`, confirmLabel: "Delete", danger: true },
        () => ctx.api.del(["users", u.username])),
    ];
    return h("tr", {},
      h("th", { scope: "row" }, who, self ? [" ", badge("you", "brand")] : null),
      h("td", {}, groupSel),
      h("td", {}, u.status === PENDING ? [badge("invited", "warning"), " (hasn't signed in)"] : (u.status || "-")),
      h("td", {}, u.enabled ? badge("enabled", "success") : badge("disabled", "danger")),
      h("td", {}, u.created ? h("time", { datetime: u.created }, u.created) : "-"),
      h("td.row-actions", {}, actions));
  });

  const email = field("Email", h("input", { type: "email", autocomplete: "off", "data-field": "email", required: true }));
  const group = field("Group", h("select", { "data-field": "group" }, groups.map((g) => h("option", { value: g }, g))));
  group.input.value = "admin-viewer";
  const createStatus = h("div");
  const create = h("form", { novalidate: true },
    email.wrap, group.wrap,
    h("button.primary", { type: "submit" }, "Create user"));
  create.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const button = ev.submitter;
    const body = { email: email.input.value.trim(), group: group.input.value };
    const errors = checkNewUser(body);
    if (errors.length) {
      clear(createStatus, errorPanel({ title: "Check the form", message: errors.map((e) => `${e.field}: ${e.message}`).join("; ") }));
      for (const e of errors) {
        const el = create.querySelector(`[data-field="${CSS.escape(e.field)}"]`);
        if (el) markInvalid(el, e.message);
      }
      return;
    }
    if (!(await confirmDialog({ title: "Create user?", message: `${body.email} joins as ${body.group}.`, confirmLabel: "Create", danger: body.group === "admin" }))) return;
    const res = await save(createStatus, () => ctx.api.post(["users"], body), { form: create, button });
    if (res) reload();
  });

  clear(main,
    pageHeader(head),
    card({ description: "The console keeps at least one enabled admin and doesn't let you delete yourself.", flush: true, label: "Console users" },
      tableWrap(h("table", { "aria-label": "Console users" },
        h("thead", {}, h("tr", {}, ["User", "Group", "Status", "Sign-in", "Created", "Actions"].map((c) => h("th", { scope: "col" }, c)))),
        h("tbody", {}, rows)))),
    status,
    card({ title: "Add a user", description: "Cognito emails them a temporary password; they set their own at first sign-in." },
      create, createStatus));
}
