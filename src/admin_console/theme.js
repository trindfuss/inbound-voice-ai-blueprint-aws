// Theme boot: a classic, synchronous script in <head> so data-theme is set before first
// paint (a module would be deferred and could flash). Light unless storage says dark; the system
// setting is never followed. No import or export, so it also loads as a module under node --test.
(function () {
  if (typeof document === "undefined") return;
  var t = null;
  try { t = window.localStorage.getItem("owlbank-admin-theme"); } catch (e) { /* blocked storage: light */ }
  document.documentElement.setAttribute("data-theme", t === "dark" ? "dark" : "light");
})();
