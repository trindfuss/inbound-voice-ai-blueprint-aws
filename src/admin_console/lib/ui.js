// Shared view pieces: the confirmation dialog, error and conflict panels, the "takes effect"
// note and the save flow with its error handling (400, 401, 403, 409). Feedback per
// A toast for every save outcome, the busy state on the submit button, field errors with
// markInvalid, and inline detail (a panel, or the conflict comparison) where it's needed.

import { clear, h, icon, markInvalid } from "./dom.js";
import { badge, compareCards, pageHeader, panel, skeleton, toast } from "./components.js";

let openDialogCleanup = null; // the listeners of the dialog now open, removed when it settles

/**
 * A modal confirmation (<dialog>, so focus is trapped and Esc cancels). Resolves true on
 * confirm. `danger` styles the confirm button as a destructive or security-changing action and
 * adds an alert-triangle to the header. Opens with Cancel focused.
 */
export function confirmDialog({ title, message, details = [], confirmLabel = "Confirm", danger = false }) {
  return new Promise((resolve) => {
    if (openDialogCleanup) openDialogCleanup();
    const dlg = document.getElementById("confirm");
    const ok = h("button", { type: "button", class: danger ? "danger" : "primary" }, confirmLabel);
    const cancel = h("button", { type: "button" }, "Cancel");
    dlg.classList.add("dialog");
    clear(dlg,
      h("div.dialog-header", {}, danger ? icon("alert-triangle") : null, h("h2#confirm-title.dialog-title", {}, title)),
      h("div.dialog-body", {},
        h("p", {}, message),
        details.length ? h("ul.dialog-details", {}, details.map((d) => h("li", {}, h("code", {}, d)))) : null),
      h("div.dialog-footer", {}, cancel, ok));
    dlg.setAttribute("aria-labelledby", "confirm-title");
    let settled = false;
    const onCancel = () => finish(false);
    const cleanup = () => {
      dlg.removeEventListener("cancel", onCancel);
      if (openDialogCleanup === cleanup) openDialogCleanup = null;
    };
    function finish(value) {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    }
    const done = (value) => {
      dlg.close();
      finish(value);
    };
    ok.addEventListener("click", () => done(true));
    cancel.addEventListener("click", () => done(false));
    dlg.addEventListener("cancel", onCancel);
    openDialogCleanup = cleanup;
    dlg.showModal();
    cancel.focus();
  });
}

/**
 * An error panel for an ApiError, any Error, or a plain {title, message}. A conflict adds the
 * Current and Yours comparison (compareCards) and, with `onReload`, a reload button.
 */
export function errorPanel(err, { yours, onReload } = {}) {
  const box = panel({ tone: "error", title: err.title || "Error" }, h("p", {}, err.message || String(err)));
  if (err.kind === "conflict") {
    box.querySelector(".panel-body").append(
      h("p", {}, "Your change was not saved. Here is the current value next to yours; reload, then redo your change."),
      compareCards(err.current, yours),
      onReload ? h("button", { type: "button", on: { click: onReload } }, "Reload current value") : null,
    );
  }
  return box;
}

export function takesEffect(seconds = 60) {
  return panel({ tone: "info" },
    `Saved changes take effect within ${seconds} seconds, on a call's next turn (the bridge's policy cache).`);
}

function busy(button, on) {
  if (!button) return;
  if (on) {
    button.setAttribute("aria-disabled", "true");
    button.setAttribute("aria-busy", "true");
    button.classList.add("is-busy");
  } else {
    button.removeAttribute("aria-disabled");
    button.removeAttribute("aria-busy");
    button.classList.remove("is-busy");
  }
}

/**
 * Run a save. While `fn` is pending, `button` (pass the form's ev.submitter) is busy
 * (aria-disabled, aria-busy and .is-busy) and a second call with it returns null without
 * calling `fn`. Success: a success toast, `status` cleared. Error: an error toast with the title
 * and reason; on a 400 naming a field the form has, that field is marked invalid and focused;
 * otherwise `status` shows the error panel (with the comparison on a 409). Returns the response
 * or null.
 */
export async function save(status, fn, { yours, onReload, form, button } = {}) {
  if (button && button.getAttribute("aria-disabled") === "true") return null;
  busy(button, true);
  clear(status);
  try {
    const res = await fn();
    const changed = res && res.changed === false ? "No change: the values are the same." : "Saved.";
    toast({
      tone: "success",
      title: changed,
      message: res && res.takes_effect_s ? `It takes effect within ${res.takes_effect_s} seconds.` : "",
    });
    clear(status);
    return res;
  } catch (err) {
    toast({ tone: "error", title: err.title || "Error", message: err.message || String(err) });
    const el = err.kind === "invalid" && err.field && form
      ? form.querySelector(`[data-field="${CSS.escape(err.field)}"]`)
      : null;
    if (el) {
      clear(status);
      markInvalid(el, err.message);
      el.focus();
    } else {
      clear(status, errorPanel(err, { yours, onReload }));
    }
    return null;
  } finally {
    busy(button, false);
  }
}

/**
 * Load data for a view: a skeleton of `kind` (page, table, form, inline) and aria-busy on the
 * container while pending, and an error panel on failure (null returned). A view loading into
 * main passes its `title` (and `description`, `back`): on failure the page header comes first,
 * so main's aria-labelledby="page-title" still names an h1.
 */
export async function load(container, fn, { kind = "page", title, description, back } = {}) {
  clear(container, skeleton(kind));
  container.setAttribute("aria-busy", "true");
  try {
    return await fn();
  } catch (err) {
    clear(container, title ? pageHeader({ title, description, back }) : null, errorPanel(err));
    return null;
  } finally {
    container.removeAttribute("aria-busy");
  }
}

export function canEdit(user) {
  return user.groups.includes("admin") || user.groups.includes("admin-editor");
}

export function isAdmin(user) {
  return user.groups.includes("admin");
}

/** "own" (the tool's value) or "default" badge for a value's source. */
export function sourceMark(source) {
  return source === "tool" ? badge("own", "brand") : badge("default", "neutral");
}
