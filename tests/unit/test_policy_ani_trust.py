"""Policy sections: ``ani_trust`` and per-tool ``confirm``.

Both follow the ``redaction`` pattern: validated fail closed, safe defaults and an
ERROR naming the field at run time, refused by :meth:`Policy.to_items` (seeding).
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from owlbank.policy import (
    DEFAULT_ACCEPTED_VERSTAT,
    DEFAULT_ANI_TRUST,
    AniTrustConfig,
    Policy,
    PolicyError,
    PolicySectionError,
    confirm_prompt_keys,
    parse_ani_trust,
    parse_confirm,
)
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
DOC: dict[str, Any] = json.loads((DATA / "policy.json").read_text())
TRUSTED = "TN-Validation-Passed-A"


def doc(**changes: Any) -> dict[str, Any]:
    d = copy.deepcopy(DOC)
    d.update(changes)
    return d


def with_tool(name: str, **row: Any) -> dict[str, Any]:
    d = copy.deepcopy(DOC)
    d["tools"][name] = {**d["tools"][name], **row}
    return d


def without_confirm(d: dict[str, Any], name: str) -> dict[str, Any]:
    d["tools"][name].pop("confirm", None)
    return d


# --------------------------------------------------------------------------- ani_trust


def test_seed_file_ani_trust_is_the_documented_default() -> None:
    p = Policy.from_dict(DOC)
    assert p.ani_trust == AniTrustConfig(("TN-Validation-Passed-A",), "no_ani_match")
    assert DEFAULT_ANI_TRUST.accepted_verstat == DEFAULT_ACCEPTED_VERSTAT


@pytest.mark.parametrize("verstat", [None, "", "TN-Validation-Passed-B", "TN-Validation-Passed-C",
                                     "No-TN-Validation", "TN-Validation-Failed",
                                     "tn-validation-passed-a", " TN-Validation-Passed-A"])
def test_anything_but_an_accepted_value_is_untrusted(verstat: Any) -> None:
    p = Policy.from_dict(DOC)
    assert p.ani_trusted(verstat) is False
    assert p.ani_match_counts(verstat) is False


def test_accepted_value_is_trusted_and_counts() -> None:
    p = Policy.from_dict(DOC)
    assert p.ani_trusted(TRUSTED) and p.ani_match_counts(TRUSTED)


def test_allow_mode_counts_the_ani_match_but_it_stays_untrusted() -> None:
    p = Policy.from_dict(doc(ani_trust={"accepted_verstat": [TRUSTED], "on_untrusted": "allow"}))
    assert p.ani_match_counts(None) is True
    assert p.ani_trusted(None) is False  # the lockout rule still keys on trust


def test_accepted_list_is_configurable() -> None:
    p = Policy.from_dict(doc(ani_trust={"accepted_verstat": [TRUSTED, "TN-Validation-Passed-B"]}))
    assert p.ani_trusted("TN-Validation-Passed-B") and p.ani_trust.on_untrusted == "no_ani_match"


@pytest.mark.parametrize("raw, field", [
    ("allow", "ani_trust"),
    ({"accepted_verstat": "TN-Validation-Passed-A"}, "ani_trust.accepted_verstat"),
    ({"accepted_verstat": [""]}, "ani_trust.accepted_verstat"),
    ({"accepted_verstat": [7]}, "ani_trust.accepted_verstat"),
    ({"accepted_verstat": ["x" * 65]}, "ani_trust.accepted_verstat"),
    ({"accepted_verstat": [TRUSTED, TRUSTED]}, "ani_trust.accepted_verstat"),
    ({"on_untrusted": "trust"}, "ani_trust.on_untrusted"),
    ({"on_untrusted": None}, "ani_trust.on_untrusted"),
    ({"accept": [TRUSTED]}, "ani_trust.accept"),
])
def test_invalid_ani_trust_gives_defaults_and_an_error_naming_the_field(
        raw: Any, field: str, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(PolicySectionError) as exc:
        parse_ani_trust(raw)
    assert exc.value.field == field
    caplog.set_level(logging.ERROR, logger="owlbank.policy")
    p = Policy.from_dict(doc(ani_trust=raw))  # the rest of the policy still loads
    assert p.ani_trust == DEFAULT_ANI_TRUST
    assert any(r.levelno == logging.ERROR and field in r.getMessage() for r in caplog.records)
    with pytest.raises(PolicyError):
        Policy.to_items(doc(ani_trust=raw))


def test_missing_ani_trust_gives_defaults_and_an_error(caplog: pytest.LogCaptureFixture) -> None:
    d = doc()
    del d["ani_trust"]
    caplog.set_level(logging.ERROR, logger="owlbank.policy")
    assert Policy.from_dict(d).ani_trust == DEFAULT_ANI_TRUST
    assert any("ani_trust" in r.getMessage() for r in caplog.records)


def test_empty_accepted_list_trusts_nothing() -> None:
    p = Policy.from_dict(doc(ani_trust={"accepted_verstat": []}))
    assert not p.ani_trusted(TRUSTED)


def test_ani_trust_round_trips_through_the_store() -> None:
    store = MemoryStore()
    cfg = {"accepted_verstat": [TRUSTED, "TN-Validation-Passed-B"], "on_untrusted": "allow"}
    for item in Policy.to_items(doc(ani_trust=cfg)):
        store.put("policy", item)
    # The stored item also carries the untrusted-attempt cap (defaults here).
    assert store.get("policy", {"pk": "ani_trust"}) == {
        "pk": "ani_trust", **cfg, "untrusted_failure_limit": 3, "untrusted_window_s": 3600,
        "ask_phone_on_no_match": True}
    assert Policy.from_store(store).ani_trust == parse_ani_trust(cfg)


def test_effective_factors_replace_the_ani_match_with_the_next_levels_factors() -> None:
    p = Policy.from_dict(DOC)
    assert p.effective_factors(1, ani_counts=True) == ["ani_match", "zip"]
    assert p.effective_factors(1, ani_counts=False) == ["zip", "ssn4"]
    assert p.effective_factors(2, ani_counts=False) == ["zip", "ssn4", "otp"]
    assert p.effective_factors(3, ani_counts=False) == ["zip", "ssn4", "otp"]  # max: no next level
    assert p.effective_factors(0, ani_counts=False) == []


def test_effective_factors_leave_levels_without_an_ani_match_alone() -> None:
    d = doc(levels={"1": ["zip"], "2": ["zip", "ssn4"], "3": ["zip", "ssn4", "otp"]})
    p = Policy.from_dict(d)
    assert p.effective_factors(1, ani_counts=False) == ["zip"]


# --------------------------------------------------------------------------- confirm


def test_seed_file_confirm_defaults() -> None:
    p = Policy.from_dict(DOC)
    assert {t: p.confirm(t) for t in p.tools} == {
        "get_branch_hours": "none", "faq_lookup": "none", "get_balance": "none",
        "list_recent_transactions": "none", "change_address": "none",
        "transfer_funds": "deterministic", "make_payment": "deterministic",
        "authenticate": "none", "escalate": "none", "end_call": "none"}
    assert p.tool_row("make_payment")["confirm"] == "deterministic"


def test_missing_confirm_takes_the_built_in_default_silently(caplog: pytest.LogCaptureFixture) -> None:
    d = without_confirm(without_confirm(doc(), "make_payment"), "change_address")
    caplog.set_level(logging.ERROR, logger="owlbank.policy")
    p = Policy.from_dict(d)
    assert p.confirm("make_payment") == "deterministic" and p.confirm("change_address") == "none"
    assert not [r for r in caplog.records if "confirm" in r.getMessage()]


def test_confirm_is_configurable_per_tool() -> None:
    p = Policy.from_dict(with_tool("change_address", confirm="deterministic"))
    assert p.confirm("change_address") == "deterministic"
    p = Policy.from_dict(with_tool("make_payment", confirm="none"))
    assert p.confirm("make_payment") == "none"


@pytest.mark.parametrize("tool", ["get_branch_hours", "faq_lookup"])
@pytest.mark.parametrize("value", ["deterministic", "none"])
def test_confirm_is_rejected_on_gateway_tools(tool: str, value: str,
                                              caplog: pytest.LogCaptureFixture) -> None:
    d = with_tool(tool, confirm=value)
    with pytest.raises(PolicySectionError) as exc:
        parse_confirm(tool, d["tools"][tool]["kind"], d["tools"][tool])
    assert exc.value.field == f"tools.{tool}.confirm"
    assert "Gateway" in str(exc.value) and "inline tool" in str(exc.value)
    caplog.set_level(logging.ERROR, logger="owlbank.policy")
    assert Policy.from_dict(d).confirm(tool) == "none"
    assert any(f"tools.{tool}.confirm" in r.getMessage() for r in caplog.records)
    with pytest.raises(PolicyError, match="Gateway"):
        Policy.to_items(d)


def test_a_level0_kind_counts_as_a_gateway_tool_whatever_its_name() -> None:
    d = copy.deepcopy(DOC)
    d["tools"]["store_locator"] = {"min_level": 0, "phrase_key": None, "kind": "level0",
                                   "confirm": "deterministic"}
    with pytest.raises(PolicyError, match="Gateway"):
        Policy.to_items(d)
    assert Policy.from_dict(d).confirm("store_locator") == "none"


@pytest.mark.parametrize("value", ["always", "", None, True, 1, "DETERMINISTIC"])
def test_invalid_confirm_falls_back_to_the_tools_default_with_an_error(
        value: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="owlbank.policy")
    p = Policy.from_dict(with_tool("make_payment", confirm=value))
    assert p.confirm("make_payment") == "deterministic"  # the safe default for money movement
    assert any("tools.make_payment.confirm" in r.getMessage() for r in caplog.records)
    assert Policy.from_dict(with_tool("change_address", confirm=value)).confirm("change_address") == "none"
    with pytest.raises(PolicyError, match="tools.make_payment.confirm"):
        Policy.to_items(with_tool("make_payment", confirm=value))


@pytest.mark.parametrize("tool", ["get_balance", "authenticate", "escalate"])
def test_deterministic_confirm_needs_an_inline_tool(tool: str) -> None:
    d = with_tool(tool, confirm="deterministic")
    assert Policy.from_dict(d).confirm(tool) == "none"
    with pytest.raises(PolicyError, match="agentic_inline"):
        Policy.to_items(d)


def test_one_bad_confirm_keeps_the_other_tools_settings() -> None:
    d = with_tool("change_address", confirm="deterministic")
    d["tools"]["make_payment"]["confirm"] = "sometimes"
    p = Policy.from_dict(d)
    assert p.confirm("change_address") == "deterministic" and p.confirm("make_payment") == "deterministic"


def test_unknown_tool_gets_the_built_in_default() -> None:
    p = Policy.from_dict(DOC)
    assert p.confirm("transfer_funds") == "deterministic" and p.confirm("nope") == "none"


def test_confirm_round_trips_through_the_store_and_gateway_rows_carry_none() -> None:
    store = MemoryStore()
    d = without_confirm(with_tool("change_address", confirm="deterministic"), "make_payment")
    for item in Policy.to_items(d):
        store.put("policy", item)
    assert store.get("policy", {"pk": "tool#make_payment"})["confirm"] == "deterministic"
    assert "confirm" not in store.get("policy", {"pk": "tool#faq_lookup"})
    p = Policy.from_store(store)
    assert p.confirm("change_address") == "deterministic" and p.confirm("make_payment") == "deterministic"
    assert p.confirm("faq_lookup") == "none"


def test_deterministic_tools_reference_their_confirm_prompts() -> None:
    p = Policy.from_dict(with_tool("change_address", confirm="deterministic"))
    keys = p.referenced_prompt_keys()
    assert set(confirm_prompt_keys("make_payment")) <= keys
    assert {"confirm.transfer_funds", "confirm.change_address"} <= keys


def test_seed_catalog_has_every_confirm_prompt_in_every_language() -> None:
    from owlbank.catalog import PromptCatalog

    catalog = PromptCatalog.from_dict(json.loads((DATA / "prompt_catalog.json").read_text()))
    for tool in ("make_payment", "transfer_funds"):
        for key in confirm_prompt_keys(tool):
            for lang in catalog.languages:
                assert catalog.has(key, lang), (key, lang)
