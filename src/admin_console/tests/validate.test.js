import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  checkAuditItem, checkMaskingForm, checkNewUser, checkPromptTexts, checkToolForm, isSecurityChange,
  MAX_AGENT_PROMPT_CHARS, parseIntField,
} from "../lib/validate.js";

test("integers: digits only", () => {
  assert.equal(parseIntField("600"), 600);
  assert.equal(parseIntField(" 0 "), 0);
  for (const bad of ["", "1e3", "2.5", "abc", null, undefined, "6 00"]) assert.equal(parseIntField(bad), null, bad);
});

test("masking form: only set fields are sent, ranges match masking_config", () => {
  const ok = checkMaskingForm({ after_ms: "2000", chars_per_s: "15.5", end_mode: "wait", silence_max_repeats: "5", tool_after_ms: undefined });
  assert.deepEqual(ok.errors, []);
  assert.deepEqual(ok.masking, { after_ms: 2000, chars_per_s: 15.5, end_mode: "wait", silence_max_repeats: 5 });
  const bad = checkMaskingForm({ after_ms: "10001", chars_per_s: "7.5", end_mode: "later", silence_max_repeats: "6", silence_action: "drop" });
  assert.deepEqual(bad.errors.map((e) => e.field).sort(), [
    "masking.after_ms", "masking.chars_per_s", "masking.end_mode", "masking.silence_action", "masking.silence_max_repeats"]);
  assert.deepEqual(bad.masking, {});
  assert.deepEqual(checkMaskingForm({}).masking, {});
});

test("masking delay property: accepted exactly for integers 0..10000", () => {
  for (let v = -50; v <= 10050; v += 25) {
    const { errors } = checkMaskingForm({ handoff_wait_ms: String(v) });
    assert.equal(errors.length === 0, v >= 0 && v <= 10000, String(v));
  }
});

test("tool form: body as PUT /tools/{name} takes it", () => {
  const r = checkToolForm({ min_level: "2", confirm: "", masked: true, phrase_key: "mask.get_balance", after_ms: "800", handoff_wait_ms: undefined });
  assert.deepEqual(r.errors, []);
  assert.deepEqual(r.body, { min_level: 2, confirm: null, mask: { phrase_key: "mask.get_balance", after_ms: 800 } });
  const off = checkToolForm({ min_level: "0", confirm: "deterministic", masked: false });
  assert.deepEqual(off.body, { min_level: 0, confirm: "deterministic", mask: null });
  const bad = checkToolForm({ min_level: "-1", confirm: "maybe", masked: true, phrase_key: "get_balance", after_ms: "abc" });
  assert.deepEqual(bad.errors.map((e) => e.field), ["min_level", "confirm", "mask.phrase_key", "mask.after_ms"]);
});

test("security change: level or confirmation differs", () => {
  const before = { min_level: 2, confirm: "deterministic", confirm_source: "default" };
  assert.equal(isSecurityChange(before, { min_level: 2, confirm: null, mask: null }), false);
  assert.equal(isSecurityChange(before, { min_level: 1, confirm: null }), true);
  assert.equal(isSecurityChange(before, { min_level: 2, confirm: "deterministic" }), true); // now the tool's own
  const own = { min_level: 2, confirm: "none", confirm_source: "tool" };
  assert.equal(isSecurityChange(own, { min_level: 2, confirm: "none" }), false);
  assert.equal(isSecurityChange(own, { min_level: 2, confirm: null }), true);
});

test("agent.* texts have their own limit and may use markup; spoken keys keep theirs", () => {
  const langs = ["en-US", "es-US"];
  const cur = { "en-US": "Hi {firstName}. {memory}", "es-US": "Hola {firstName}. {memory}" };
  const fields = (key, t, c = cur) => checkPromptTexts(key, t, langs, c).map((e) => e.field);
  const pad = (s, n) => s + "a".repeat(n - s.length);
  const sys = { "en-US": "Rules.", "es-US": "Reglas." };
  const frame = { "en-US": "Hi {firstName}. {profile} {memory}", "es-US": "Hola {firstName}. {profile} {memory}" };
  assert.equal(MAX_AGENT_PROMPT_CHARS, 16000);
  assert.deepEqual(fields("agent.system.x", { "en-US": pad(sys["en-US"], 16000), "es-US": sys["es-US"] }, sys), [], "16,000");
  assert.deepEqual(fields("agent.system.x", { "en-US": pad(sys["en-US"], 16001), "es-US": sys["es-US"] }, sys), ["texts[en-US]"], "16,001");
  assert.deepEqual(fields("agent.system.x", { "en-US": "<rules>Hi.</rules>", "es-US": "a > b" }, sys), [], "markup");
  assert.deepEqual(fields("agent.caller_context", { "en-US": "<f>Hi {firstName}. {memory} {profile}</f>", "es-US": frame["es-US"] }, frame), [], "frame markup");
  assert.deepEqual(fields("agent.caller_context", { "en-US": "Hi {firstName}. {memory}", "es-US": frame["es-US"] }, frame), ["texts[en-US]"], "placeholder dropped");
  assert.deepEqual(fields("agent.system.x", { "en-US": sys["en-US"] + " {", "es-US": sys["es-US"] }, sys), ["texts[en-US]"], "stray brace");
  assert.deepEqual(fields("agent.system.x", { "en-US": " ", "es-US": sys["es-US"] }, sys), ["texts[en-US]"], "blank");
  assert.deepEqual(fields("greeting.personal", { "en-US": "Hi <b>{firstName}</b> {memory}", "es-US": cur["es-US"] }), ["texts[en-US]"], "spoken markup");
  assert.deepEqual(fields("greeting.personal", { "en-US": pad(cur["en-US"], 601), "es-US": cur["es-US"] }), ["texts[en-US]"], "spoken 601");
  assert.deepEqual(fields("greeting.personal", { "en-US": pad(cur["en-US"], 600), "es-US": cur["es-US"] }), [], "spoken 600");
});

test("every shipped agent.* text passes as-is", () => {
  const catalog = JSON.parse(readFileSync(new URL("../../../data/prompt_catalog.json", import.meta.url), "utf8"));
  const byKey = new Map();
  for (const e of catalog.entries) if (e.key.startsWith("agent.")) byKey.set(e.key, { ...byKey.get(e.key), [e.lang]: e.text });
  assert.ok([...byKey.keys()].some((k) => k.startsWith("agent.system.")));
  assert.ok([...byKey.values()].some((t) => Object.values(t).some((s) => s.length > 600 && s.includes("<"))));
  for (const [key, texts] of byKey) {
    assert.deepEqual(checkPromptTexts(key, texts, Object.keys(texts), texts), [], key);
    assert.deepEqual(checkPromptTexts(key, texts, Object.keys(texts), null), [], `${key} (no current)`);
  }
});

test("the placeholder allow-list, with the API's messages", () => {
  const catalog = JSON.parse(readFileSync(new URL("../../../data/prompt_catalog.json", import.meta.url), "utf8"));
  const shipped = (key) => Object.fromEntries(catalog.entries.filter((e) => e.key === key).map((e) => [e.lang, e.text]));
  const none = "must have no placeholders";
  const callerSet = "must have exactly {firstName}, {memory}, {profile}";
  const errs = (key, texts, current) => checkPromptTexts(key, texts, Object.keys(texts), current);
  const edit = (texts, f) => Object.fromEntries(Object.entries(texts).map(([l, t]) => [l, f(t)]));

  // A placeholder added to agent.system.haiku-5-5.
  const sys = shipped("agent.system.haiku-5-5");
  assert.deepEqual(errs("agent.system.haiku-5-5", { ...sys, "en-US": sys["en-US"] + "\nCaller: {firstName}." }, sys),
    [{ field: "texts[en-US]", message: none }]);
  // {profile} removed from agent.caller_context.
  const frame = shipped("agent.caller_context");
  assert.ok(Object.values(frame).every((t) => t.includes("{profile}")));
  assert.deepEqual(errs("agent.caller_context", { ...frame, "es-US": frame["es-US"].replace("{profile}", "perfil") }, frame),
    [{ field: "texts[es-US]", message: callerSet }]);
  // A new key agent.system.opus-5-5 with {firstName} (no saved text).
  assert.deepEqual(errs("agent.system.opus-5-5", { "en-US": "Rules.\nCaller: {firstName}.", "es-US": "Reglas." }, null),
    [{ field: "texts[en-US]", message: none }]);
  // A restore of a pre-feature revision: the old texts had the caller line; so did the saved texts.
  const old = edit(sys, (t) => `${t}\nCaller: {firstName}. Verified profile: {profile}. Recent interactions: {memory}.`);
  assert.deepEqual(errs("agent.system.haiku-5-5", old, sys).map((e) => e.message), [none, none]);
  assert.deepEqual(errs("agent.system.haiku-5-5", old, old).map((e) => e.message), [none, none]);
  // Leaving the pre-feature placeholders out is fine, whatever the saved text has.
  assert.deepEqual(errs("agent.system.haiku-5-5", sys, old), []);
  // The other agent.* keys take none; the unverified frame never gets a caller value.
  for (const key of ["agent.caller_context_unverified", "agent.web_activity", "agent.web_activity_context", "agent.call_started"]) {
    const t = shipped(key);
    assert.deepEqual(errs(key, edit(t, (s) => `${s} {memory}`), t).map((e) => e.message), Object.keys(t).map(() => none), key);
  }
  // A stray brace keeps its own message; spoken keys keep the same-set rule.
  assert.deepEqual(errs("agent.caller_context", edit(frame, (t) => `${t} }`), frame).map((e) => e.message),
    Object.keys(frame).map(() => "has a '{' or '}' that isn't a {placeholder}"));
  assert.deepEqual(errs("greeting.personal", { "en-US": "Hi {firstName}." }, { "en-US": "Thanks, {firstName}." }), []);
});

test("the shared accept/refuse table (also asserted by the API's Python tests)", () => {
  const { cases } = JSON.parse(readFileSync(new URL("../../../tests/fixtures/prompt_text_rules.json", import.meta.url), "utf8"));
  assert.ok(cases.length > 20);
  for (const c of cases) {
    let text = c.text;
    if (c.pad_to !== undefined) {
      text += (c.pad_char ?? "a").repeat(c.pad_to - [...text].length);
      assert.equal([...text].length, c.pad_to, c.name);
    }
    const current = c.current === null ? null : { "en-US": c.current };
    const errors = checkPromptTexts(c.key, { "en-US": text }, ["en-US"], current);
    assert.equal(errors.length === 0, c.accept, `${c.name}: ${JSON.stringify(errors)}`);
  }
});

test("prompt texts: every language, no braces in words", () => {
  const langs = ["en-US", "es-US"];
  assert.deepEqual(checkPromptTexts("mask.x", { "en-US": "Hi {account}", "es-US": "Hola" }, langs), []);
  assert.deepEqual(checkPromptTexts("mask.x", { "en-US": "Hi", "es-US": " " }, langs).map((e) => e.field), ["texts[es-US]"]);
  assert.deepEqual(checkPromptTexts("arg.account.savings", { "en-US": "{savings}", "es-US": "ahorros" }, langs).map((e) => e.field), ["texts[en-US]"]);
  assert.equal(checkPromptTexts("mask.x", { "en-US": "Hi" }, langs).length, 1);
});

test("new user: email and group", () => {
  assert.deepEqual(checkNewUser({ email: "ops@example.com", group: "admin-editor" }), []);
  assert.deepEqual(checkNewUser({ email: "not-an-email", group: "root" }).map((e) => e.field), ["email", "group"]);
  assert.deepEqual(checkNewUser({ email: "a@b", group: "admin" }).map((e) => e.field), ["email"]);
});

test("audit items", () => {
  for (const ok of ["policy#tool#get_balance", "policy#masking", "catalog#mask.get_balance.1", "user#ops@example.com"]) {
    assert.equal(checkAuditItem(ok), null, ok);
  }
  for (const bad of ["", "tool#x", "policy#", "policy#a b"]) assert.ok(checkAuditItem(bad), bad);
});

test("greeting silence timeout: integers 0..30000", () => {
  for (const [v, ok] of [["0", true], ["7000", true], ["30000", true], ["30001", false], ["-1", false], ["7.5", false]]) {
    const { errors, masking } = checkMaskingForm({ greeting_silence_timeout_ms: v });
    assert.equal(errors.length === 0, ok, v);
    if (ok) assert.deepEqual(masking, { greeting_silence_timeout_ms: Number(v) });
    else assert.deepEqual(errors.map((e) => e.field), ["masking.greeting_silence_timeout_ms"]);
  }
});
