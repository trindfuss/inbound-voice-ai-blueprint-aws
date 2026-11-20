"""Consistency checks over the seed files in data/."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from scripts.hash_tool import hash_factor, main as hash_main, verify

DATA = Path(__file__).resolve().parents[2] / "data"
LANGS = ("en-US", "es-US")
FICTIONAL = re.compile(r"^\+1\d{3}55501\d{2}$")  # NANP 555-0100..0199
# "+1 202 555 0101", "+12025550101", "(202) 555-0101" and similar dialable forms.
MASK_GENERIC_RE = re.compile(r"^mask\.generic\.\d+$")
PHONE_LIKE = re.compile(r"\+\d[\d\s().-]{8,}\d|\(\d{3}\)\s?\d{3}[\s.-]?\d{4}")


def load(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog() -> dict[str, Any]:
    return load("prompt_catalog.json")


@pytest.fixture(scope="module")
def langs_by_key(catalog: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for e in catalog["entries"]:
        out[e["key"]].add(e["lang"])
    return out


@pytest.fixture(scope="module")
def policy() -> dict[str, Any]:
    return load("policy.json")


@pytest.fixture(scope="module")
def fixtures() -> dict[str, Any]:
    return load("fixtures.json")


@pytest.fixture(scope="module")
def test_factors() -> dict[str, Any]:
    return load("fixtures.test-factors.json")


# --------------------------------------------------------------------------- catalog


def test_catalog_languages(catalog: dict[str, Any]) -> None:
    assert catalog["defaultLanguage"] == "en-US"
    assert set(catalog["languages"]) == set(LANGS)
    for lang, cfg in catalog["languages"].items():
        assert cfg["say"]["language"] == lang
        relay = cfg["conversationRelay"]
        assert relay["ttsProvider"] == "ElevenLabs"
        assert relay["transcriptionProvider"] == "Deepgram"
    en, es = catalog["languages"]["en-US"], catalog["languages"]["es-US"]
    assert en["conversationRelay"]["voice"] == "UgBBYS2sOqTuMpoF3BR0-v4_turbo"
    assert es["conversationRelay"]["voice"] == "CaJslL1xziwefCeTNzHv-v4_turbo"
    assert es["conversationRelay"]["transcriptionLanguage"] == "es"
    # Same speaker in <Say> and ConversationRelay.
    for cfg in (en, es):
        assert cfg["conversationRelay"]["voice"].split("-")[0] in cfg["say"]["voice"]
    assert any("audition pending" in n.lower() for n in catalog["_notes"])


def test_catalog_entry_shape_and_uniqueness(catalog: dict[str, Any]) -> None:
    seen = set()
    for e in catalog["entries"]:
        assert set(e) == {"key", "lang", "text", "version", "status", "audioUrl"}
        assert e["lang"] in LANGS and e["text"].strip()
        assert e["status"] == "published" and e["version"] >= 1
        assert (e["key"], e["lang"]) not in seen
        seen.add((e["key"], e["lang"]))


def test_every_key_has_every_language(catalog: dict[str, Any], langs_by_key: dict[str, set[str]]) -> None:
    single = catalog["singleLanguageKeys"]
    assert single == {"language.select": "es-US", "agent.recap.system": "en-US",
                      "agent.briefing.system": "en-US"}
    for key, langs in langs_by_key.items():
        expected = {single[key]} if key in single else set(LANGS)
        assert langs == expected, key


def test_handover_keys_present(langs_by_key: dict[str, set[str]]) -> None:
    required = {
        "language.select", "greeting.generic", "greeting.personal", "factor.ssn4", "factor.zip",
        "stepup.intro", "stepup.success", "tool.make_payment", "tool.transfer_funds",
        "error.lookup_unavailable", "escalate.default", "escalate.fraud", "escalate.verify",
        "escalate.unavailable", "balance.read", "balance.read.pending", "balance.no_account",
        "common.anything_else", "common.one_moment", "command.repeat", "command.agent",
        "command.cancel", "clarify.digits", "clarify.choice", "label.zip", "label.ssn4",
        "account.checking", "account.savings",
        "agent.system.haiku-5-5", "agent.system.haiku-4-5", "agent.system.sonnet-5-5",
        "escalation.verification_risk", "escalation.otp_unavailable",
        "login.phone_number", "login.phone_number_retry", "label.phone",
    }
    assert required <= set(langs_by_key)


def test_placeholders_match_across_languages(catalog: dict[str, Any]) -> None:
    by_key: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for e in catalog["entries"]:
        by_key[e["key"]][e["lang"]] = set(re.findall(r"\{(\w+)\}", e["text"]))
    for key, per_lang in by_key.items():
        assert len({frozenset(v) for v in per_lang.values()}) == 1, key


# --------------------------------------------------------------------------- policy


def test_policy_tools(policy: dict[str, Any]) -> None:
    tools = policy["tools"]
    expected_levels = {
        "get_branch_hours": 0, "faq_lookup": 0, "get_balance": 1, "list_recent_transactions": 1,
        "change_address": 2, "transfer_funds": 3, "make_payment": 3,
    }
    for name, level in expected_levels.items():
        assert tools[name]["min_level"] == level
    assert {"authenticate", "escalate"} <= set(tools)
    kinds = {"level0", "agentic_inline", "deterministic", "handoff"}
    for name, row in tools.items():
        assert row["kind"] in kinds, name
        assert 0 <= row["min_level"] <= policy["max_level"]
        assert set(row) >= {"min_level", "phrase_key", "kind"}
        if row["min_level"] > 0:
            assert row["phrase_key"], f"{name} needs a connecting phrase"
    assert tools["get_balance"]["kind"] == "deterministic"
    assert tools["authenticate"]["kind"] == tools["escalate"]["kind"] == "handoff"


def test_policy_levels_are_cumulative(policy: dict[str, Any]) -> None:
    levels = {int(k): v for k, v in policy["levels"].items()}
    assert policy["max_level"] == max(levels) == 3
    assert levels[1] == ["ani_match", "zip"]
    assert levels[2] == ["ani_match", "zip", "ssn4"]
    assert levels[3] == ["ani_match", "zip", "ssn4", "otp"]
    for n in range(2, policy["max_level"] + 1):
        assert set(levels[n - 1]) < set(levels[n])
    collected = {f for fs in levels.values() for f in fs} - {"ani_match"}
    assert collected == set(policy["factors"])


def test_policy_escalation_reasons(policy: dict[str, Any]) -> None:
    esc = policy["escalation"]
    reasons = esc["reasons"]
    assert set(reasons) == {
        "caller_requested_human", "unable_to_resolve", "step_up_failed", "fraud_suspected",
        "card_dispute", "complaint", "vulnerable_customer", "verification_risk", "otp_unavailable",
        "no_input",
    }
    assert esc["default_reason"] in reasons
    assert {k: v for k, v in reasons["fraud_suspected"].items() if k != "description"} == {
        "queue": "fraud", "priority": 100, "phrase_key": "escalate.fraud"}
    assert reasons["step_up_failed"]["phrase_key"] == "escalate.verify"
    # The untrusted-attempt cap routes to the fraud queue with a neutral phrase.
    assert reasons["verification_risk"] == {"queue": "fraud", "priority": 100,
                                            "phrase_key": "escalation.verification_risk"}
    # No one-time code can go to the calling number -> the default support queue
    # (language-routed by the workflow), a neutral phrase, no extra priority.
    assert reasons["otp_unavailable"] == {"queue": "support", "priority": 0,
                                          "phrase_key": "escalation.otp_unavailable"}
    # A silent caller goes to a human like any unresolved request.
    assert reasons["no_input"] == reasons["unable_to_resolve"]
    assert reasons["vulnerable_customer"]["priority"] > reasons["complaint"]["priority"]
    for row in reasons.values():
        assert set(esc["queues"][row["queue"]]) == set(LANGS)


# --------------------------------------------------------------------------- web-activity traits

# The demo data's merchant, amount, card product, channel and offer words: no reason
# description may name them.
DEMO_WORDS = ["Example Airlines", "airline", "412", "Red Card", "Purple Card", "Rewards Card",
              "OwlBank", "WhatsApp", "chat", "SMS", "HELOC", "home-equity", "auto loan",
              "Thomas", "Doe", "2026-10-05"]


def test_policy_disputes_queue_and_reason(policy: dict[str, Any]) -> None:
    """The disputes queue in both languages and the ``card_dispute`` reason."""
    esc = policy["escalation"]
    assert esc["queues"]["disputes"] == {"en-US": "OwlBank Disputes", "es-US": "OwlBank Disputes"}
    row = esc["reasons"]["card_dispute"]
    assert {k: v for k, v in row.items() if k != "description"} == {
        "queue": "disputes", "priority": 50, "phrase_key": "escalate.dispute"}


def test_policy_reason_descriptions(policy: dict[str, Any]) -> None:
    """``card_dispute`` and ``fraud_suspected`` are described, each valid, none naming
    a demo word; the tools contract's defaults are the seed's."""
    from owlbank.policy import parse_reason_description
    from owlbank.tools_contract import DEFAULT_ESCALATION_REASONS, DEFAULT_REASON_DESCRIPTIONS

    reasons = policy["escalation"]["reasons"]
    described = {n: r["description"] for n, r in reasons.items() if "description" in r}
    assert set(described) >= {"card_dispute", "fraud_suspected"}
    for name, text in described.items():
        assert parse_reason_description(name, text) == text
        low = text.lower()
        for word in DEMO_WORDS:
            assert not re.search(rf"\b{re.escape(word.lower())}\b", low), (name, word)
    assert dict(DEFAULT_REASON_DESCRIPTIONS) == described
    assert "card_dispute" in DEFAULT_ESCALATION_REASONS
    assert set(DEFAULT_ESCALATION_REASONS) <= set(reasons)


def test_policy_web_offers(policy: dict[str, Any], langs_by_key: dict[str, set[str]]) -> None:
    """The demo offers parse, and each has ``offer.<id>`` in every language."""
    from owlbank.policy import WebOffer, parse_web_offers

    assert parse_web_offers(policy["web_offers"]) == (
        WebOffer("heloc", ("HELOC", "home-equity")), WebOffer("auto_loan", ("Auto loan", "auto-loans")))
    for offer in policy["web_offers"]["offers"]:
        assert langs_by_key.get(f"offer.{offer['id']}") == set(LANGS), offer["id"]


def test_policy_model_failover(policy: dict[str, Any]) -> None:
    """The shipped section parses and equals the built-in defaults."""
    from owlbank.failover.settings import Settings
    from owlbank.policy import parse_model_failover

    ids = ["us.anthropic.claude-haiku-5-5-20260301-v1:0", "us.anthropic.claude-sonnet-5-5-20260301-v1:0"]
    assert parse_model_failover(policy["model_failover"]).settings_for(ids) == Settings.for_models(ids)


def test_every_policy_phrase_key_is_in_catalog(policy: dict[str, Any], langs_by_key: dict[str, set[str]]) -> None:
    keys: list[str] = [r["phrase_key"] for r in policy["tools"].values() if r["phrase_key"]]
    keys += [r["template_key"] for r in policy["tools"].values() if r.get("template_key")]
    keys += [r["phrase_key"] for r in policy["escalation"]["reasons"].values()]
    keys += [f[k] for f in policy["factors"].values() for k in ("promptKey", "labelKey")]
    keys += [r["mask"]["phrase_key"] for r in policy["tools"].values() if r.get("mask")]
    for key in keys:
        assert langs_by_key.get(key) == set(LANGS), key


def test_mask_keys_in_every_language(policy: dict[str, Any], langs_by_key: dict[str, set[str]]) -> None:
    """The acknowledgement pool ``mask.generic.1`` .. ``.5`` and every policy
    ``mask.<tool>`` key exist in both languages; the pool has at least four phrases."""
    mask_keys = {k for k in langs_by_key if k.startswith("mask.")}
    generic = sorted(k for k in mask_keys if MASK_GENERIC_RE.match(k))
    assert {f"mask.generic.{n}" for n in range(1, 6)} <= set(generic)
    assert len(generic) >= 4
    assert "mask.generic" not in mask_keys  # the single acknowledgement is gone
    for row in policy["tools"].values():
        if row.get("mask") is not None:
            assert row["mask"]["phrase_key"] in mask_keys
    for key in mask_keys:
        assert langs_by_key[key] == set(LANGS), key
    assert set(policy["masking"]) == {
        "after_ms", "tool_after_ms", "handoff_wait_ms", "end_mode", "chars_per_s",
        "silence_timeout_ms", "silence_max_repeats", "silence_action", "greeting_silence_timeout_ms",
    }  # data model


def test_mask_phrases_are_present_non_empty_and_short(catalog: dict[str, Any], policy: dict[str, Any]) -> None:
    """Every ``mask.generic.<n>`` and every ``mask.<tool>`` (catalog or policy) has a
    non-empty text of at most 80 characters in en-US and es-US. Structural only, no content checks."""
    texts = {(e["key"], e["lang"]): e.get("text") for e in catalog["entries"]}
    keys = {k for k, _ in texts if k.startswith("mask.")}
    keys |= {r["mask"]["phrase_key"] for r in policy["tools"].values() if r.get("mask")}
    keys |= {f"mask.{t}" for t, r in policy["tools"].items() if r.get("mask")}
    assert any(MASK_GENERIC_RE.match(k) for k in keys)
    for key in sorted(keys):
        for lang in LANGS:
            text = texts.get((key, lang))
            assert isinstance(text, str) and text.strip(), (key, lang)
            assert len(text.strip()) <= 80, (key, lang, len(text.strip()))


# --------------------------------------------------------------------------- faq


def test_faq_bilingual() -> None:
    faq = load("faq.json")
    assert faq["branches"] and faq["faqs"]
    for b in faq["branches"]:
        assert set(b["hours"]) == set(LANGS) and set(b["name"]) == set(LANGS)
    for f in faq["faqs"]:
        assert set(f["question"]) == set(LANGS) and set(f["answer"]) == set(LANGS)
    assert set(faq["phoneSupportHours"]) == set(LANGS)


# --------------------------------------------------------------------------- fixtures


def _walk_strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _walk_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _walk_strings(v)]
    return []


def test_fixture_phones_are_fictional(fixtures: dict[str, Any]) -> None:
    for c in fixtures["customers"]:
        assert FICTIONAL.match(c["phone"]), c["phone"]
    for n in fixtures["unseededNumbers"]:
        assert n["phone"] == "anonymous" or FICTIONAL.match(n["phone"])


EMAIL = re.compile(r"[^@\s\"',;<>]+@([^@\s\"',;<>]+)")
RESERVED_TLDS = {"example", "invalid", "test", "localhost"}


def test_every_customer_has_a_unique_memory_email(fixtures: dict[str, Any]) -> None:
    """Verified customers are keyed on Memory's `email` identifier, one per customer."""
    emails = [c.get("memory_email") for c in fixtures["customers"]]
    assert all(isinstance(e, str) and EMAIL.fullmatch(e) for e in emails), emails
    assert len({e.lower() for e in emails if e}) == len(emails), "duplicate memory_email"
    phones = {c["phone"] for c in fixtures["customers"]}
    assert not any(e in phones or re.search(r"\d{7,}", e) for e in emails if e)  # never the ANI
    assert all("memory_user_id" not in c for c in fixtures["customers"])  # 'user_id' is not valid


@pytest.mark.parametrize("name", ["fixtures.json", "fixtures.test-factors.json", "faq.json",
                                  "policy.json", "prompt_catalog.json", "prompt_suite.json"])
def test_no_real_looking_email_addresses_anywhere(name: str) -> None:
    """Every address in the data files is on a reserved (RFC 2606) name: .example, .invalid..."""
    for s in _walk_strings(load(name)):
        for m in EMAIL.finditer(s):
            domain = m.group(1).strip(".").lower()
            assert domain.rsplit(".", 1)[-1] in RESERVED_TLDS or domain in {
                "example.com", "example.net", "example.org"}, f"{name}: {m.group(0)}"


@pytest.mark.parametrize("name", ["fixtures.json", "fixtures.test-factors.json", "faq.json",
                                  "policy.json", "prompt_catalog.json", "prompt_suite.json"])
def test_no_real_looking_phone_numbers_anywhere(name: str) -> None:
    for s in _walk_strings(load(name)):
        for m in PHONE_LIKE.findall(s):
            digits = re.sub(r"\D", "", m)
            if len(digits) == 10:
                digits = "1" + digits
            assert FICTIONAL.match("+" + digits), f"{name}: {m}"


def test_fixtures_cover_lookup_outcomes(fixtures: dict[str, Any]) -> None:
    by_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in fixtures["customers"]:
        by_phone[c["phone"]].append(c)
    thomas = by_phone["+12025550101"]
    assert len(thomas) == 1
    assert {a["type"] for a in thomas[0]["accounts"]} == {"checking", "savings", "credit_card"}
    assert len(by_phone["+12025550102"]) == 2  # Doe household
    maria = by_phone["+12025550103"]
    assert len(maria) == 1 and maria[0]["language"] == "es-US"
    assert "+12025550199" not in by_phone
    placeholders = fixtures["realNumberPlaceholders"]
    assert placeholders["developerPhone"]["cdkParameter"] == "developerPhone"
    assert set(placeholders["testNumbers"]) == {"A", "B", "C", "D"}


def test_fixture_amounts_are_decimal_strings(fixtures: dict[str, Any]) -> None:
    money = re.compile(r"^-?\d+\.\d{2}$")
    for c in fixtures["customers"]:
        for a in c["accounts"]:
            for f in ("available", "current", "pending"):
                assert isinstance(a[f], str) and money.match(a[f]), (c["customer_id"], f)


def test_fixtures_hold_no_plaintext_factors(fixtures: dict[str, Any], test_factors: dict[str, Any]) -> None:
    plaintext = {v for fs in test_factors["customers"].values() for v in fs.values()}
    text = json.dumps(fixtures)
    for value in plaintext:
        assert not re.search(rf"(?<![0-9a-f]){value}(?![0-9a-f])", text), value


def test_hashes_verify_against_test_factors(fixtures: dict[str, Any], test_factors: dict[str, Any]) -> None:
    assert "FAKE TEST DATA" in test_factors["_warning"]
    for c in fixtures["customers"]:
        plain = test_factors["customers"][c["customer_id"]]
        assert set(c["factors"]) == set(plain) == {"zip", "ssn4"}
        for name, value in plain.items():
            assert verify(value, c["factors"][name])
            assert not verify(value + "0", c["factors"][name])
    for p in test_factors["personas"]:
        if p["customer_id"] is None:
            continue
        stored = next(c for c in fixtures["customers"] if c["customer_id"] == p["customer_id"])
        assert stored["phone"] == p["ani"]
        ok = verify(p["zip"], stored["factors"]["zip"]) and verify(p["ssn4"], stored["factors"]["ssn4"])
        assert ok == (p["expect"] != "verification_failed"), p["id"]


# --------------------------------------------------------------------------- hash_tool


def test_hash_tool_salts_and_verifies() -> None:
    a, b = hash_factor("94105"), hash_factor("94105")
    assert a != b  # fresh salt each time
    assert verify("94105", a) and verify("94105", b)
    assert not verify("94106", a)
    assert hash_factor("1111", salt="abc") == hash_factor("1111", salt="abc")
    for bad in ("", "nosep", "$abc", "abc$"):
        assert not verify("1111", bad)
    with pytest.raises(ValueError):
        hash_factor("")
    with pytest.raises(ValueError):
        hash_factor("1", salt="a$b")


def test_hash_tool_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert hash_main(["2222"]) == 0
    stored = capsys.readouterr().out.strip()
    assert verify("2222", stored)
    assert hash_main(["2222", "--verify", stored]) == 0
    assert hash_main(["2223", "--verify", stored]) == 1
