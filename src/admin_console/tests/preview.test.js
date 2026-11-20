import assert from "node:assert/strict";
import { test } from "node:test";

import {
  ackCombinations, combinations, hasStrayBrace, insertPlaceholder, MAX_PREVIEW_ROWS, placeholderNames,
  previewMatrix, renderPhrase,
} from "../lib/preview.js";

const LANGS = ["en-US", "es-US"];
const PLACEHOLDERS = [
  {
    name: "account", placeholder: "{account}", missing: [],
    values: [
      { value: "checking", key: "arg.account.checking", words: { "en-US": "checking", "es-US": "corriente" } },
      { value: "savings", key: "arg.account.savings", words: { "en-US": "savings", "es-US": "de ahorros" } },
    ],
  },
  {
    name: "period", placeholder: "{period}", missing: ["arg.period.year[es-US]"],
    values: [
      { value: "month", key: "arg.period.month", words: { "en-US": "month", "es-US": "mes" } },
      { value: "year", key: "arg.period.year", words: { "en-US": "year", "es-US": null } },
    ],
  },
];

test("placeholder names in order of first use, stray braces", () => {
  assert.deepEqual(placeholderNames("{a} {b} {a}"), ["a", "b"]);
  assert.deepEqual(placeholderNames("no placeholders"), []);
  assert.equal(hasStrayBrace("Checking {account} now"), false);
  assert.equal(hasStrayBrace("Checking {account now"), true);
  assert.equal(hasStrayBrace("}"), true);
  assert.equal(hasStrayBrace("{1x}"), true);
});

test("render replaces each placeholder with the value's word in the language", () => {
  assert.deepEqual(renderPhrase("Checking your {account} balance now.", { account: "savings" }, PLACEHOLDERS, "en-US"),
    { text: "Checking your savings balance now." });
  assert.deepEqual(renderPhrase("Revisando su cuenta {account}.", { account: "checking" }, PLACEHOLDERS, "es-US"),
    { text: "Revisando su cuenta corriente." });
  assert.deepEqual(renderPhrase("One moment.", {}, PLACEHOLDERS, "en-US"), { text: "One moment." });
});

test("render refuses what the bridge can't say in full", () => {
  assert.match(renderPhrase("Your {colour}", {}, PLACEHOLDERS, "en-US").error, /isn't one of this tool's placeholders/);
  assert.match(renderPhrase("Last {period}", { period: "year" }, PLACEHOLDERS, "es-US").error, /Missing word for arg.period.year in es-US/);
  assert.match(renderPhrase("Oops {account", { account: "savings" }, PLACEHOLDERS, "en-US").error, /\{ or \}/);
  assert.match(renderPhrase("   ", {}, PLACEHOLDERS, "en-US").error, /empty/);
  assert.match(renderPhrase("{account}", {}, PLACEHOLDERS, "en-US").error, /No value/);
});

test("a word is never expanded (one pass)", () => {
  const sneaky = [{ name: "a", values: [{ value: "x", key: "arg.a.x", words: { "en-US": "{a}" } }] }];
  assert.ok(renderPhrase("{a}", { a: "x" }, sneaky, "en-US").error);
});

test("combinations: the product of the used placeholders' values", () => {
  assert.deepEqual(combinations([], PLACEHOLDERS), [{}]);
  assert.equal(combinations(["account"], PLACEHOLDERS).length, 2);
  assert.deepEqual(combinations(["account", "period"], PLACEHOLDERS), [
    { account: "checking", period: "month" }, { account: "checking", period: "year" },
    { account: "savings", period: "month" }, { account: "savings", period: "year" },
  ]);
});

test("preview: every value x every language, errors where a word is missing", () => {
  const texts = { "en-US": "Your {account} for the {period}.", "es-US": "Su cuenta {account} del {period}." };
  const { rows, ok, truncated } = previewMatrix(texts, PLACEHOLDERS, LANGS);
  assert.equal(rows.length, 8);
  assert.equal(truncated, false);
  assert.equal(ok, false);
  const bad = rows.filter((r) => r.error);
  assert.deepEqual(bad.map((r) => [r.lang, r.args.period]), [["es-US", "year"], ["es-US", "year"]]);
  assert.equal(rows[0].text, "Your checking for the month.");
});

test("preview property: rows = sum over languages of the product of used value counts", () => {
  const texts = ["", "x", "{account}", "{period}", "{account} {period}", "{period}{account}{period}"];
  for (const en of texts) {
    for (const es of texts) {
      const { rows } = previewMatrix({ "en-US": en, "es-US": es }, PLACEHOLDERS, LANGS);
      const count = (t) => placeholderNames(t).reduce((n, p) => n * 2, 1);
      assert.equal(rows.length, count(en) + count(es));
      for (const r of rows) assert.ok(("text" in r) !== ("error" in r));
    }
  }
});

test("preview is capped", () => {
  const many = Array.from({ length: 5 }, (_, i) => ({
    name: `p${i}`, values: Array.from({ length: 6 }, (_, j) => ({ value: `v${j}`, key: `arg.p${i}.v${j}`, words: { "en-US": "w" } })),
  }));
  const text = many.map((p) => `{${p.name}}`).join(" ");
  const { rows, truncated } = previewMatrix({ "en-US": text }, many, ["en-US"]);
  assert.equal(rows.length, MAX_PREVIEW_ROWS);
  assert.equal(truncated, true);
});

test("insert a placeholder at the caret or over the selection", () => {
  assert.deepEqual(insertPlaceholder("Checking your  balance", 14, 14, "account"),
    { text: "Checking your {account} balance", caret: 23 });
  assert.deepEqual(insertPlaceholder("Checking your XXX balance", 14, 17, "account"),
    { text: "Checking your {account} balance", caret: 23 });
  assert.deepEqual(insertPlaceholder("abc", undefined, undefined, "x"), { text: "abc{x}", caret: 6 });
  assert.deepEqual(insertPlaceholder("abc", 99, 2, "x"), { text: "abc{x}", caret: 6 });
});

test("insert property: the result holds the placeholder and keeps the text around it", () => {
  const base = "Un momento, por favor.";
  for (let s = 0; s <= base.length; s++) {
    for (let e = s; e <= base.length; e += 3) {
      const { text, caret } = insertPlaceholder(base, s, e, "account");
      assert.equal(text, base.slice(0, s) + "{account}" + base.slice(e));
      assert.equal(text.slice(caret - 9, caret), "{account}");
      assert.equal(hasStrayBrace(text), false);
    }
  }
});

test("acknowledgement x tool phrase combinations, with missing texts named", () => {
  const ack = [{ key: "mask.ack.1", texts: { "en-US": "Sure.", "es-US": "Claro." } },
    { key: "mask.ack.2", texts: { "en-US": "Okay.", "es-US": null } }];
  const pool = [{ key: "mask.get_balance", texts: { "en-US": "Checking your {account} balance.", "es-US": "Revisando." } }];
  const en = ackCombinations(ack, pool, "en-US");
  assert.deepEqual(en.map((c) => c.text), ["Sure. Checking your {account} balance.", "Okay. Checking your {account} balance."]);
  const es = ackCombinations(ack, pool, "es-US");
  assert.equal(es[1].text, null);
  assert.deepEqual(es[1].missing, ["mask.ack.2[es-US]"]);
  assert.deepEqual(ackCombinations([], pool, "en-US"), []);
});
