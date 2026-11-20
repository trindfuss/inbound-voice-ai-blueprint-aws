"""Unit tests for owlbank.task_attributes."""

from __future__ import annotations

import json

import pytest

from owlbank import segments as sg
from owlbank.callstate import CallState, PendingTool
from owlbank.store import MemoryStore
from owlbank.task_attributes import build_task_attributes, clean_summary, normalize_reason

SID = "CA" + "d" * 32
NOW = 1_760_108_645  # 2025-10-10T15:04:05Z


@pytest.fixture(autouse=True)
def _clear() -> None:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)


def state(**kw) -> CallState:
    base = {"call_sid": SID, "expires_at": NOW + 100, "language": "es-US", "ani": "+15551234567",
            "verified_customer_id": "C-1042", "level": 2, "factors_verified": ("zip", "ssn4")}
    base.update(kw)
    return CallState(**base)


def test_full_attributes() -> None:
    store = MemoryStore()
    store.put("customers", {"customer_id": "C-1042", "name": "Thomas Doe", "phone": "+15551234567"})
    attrs = build_task_attributes(
        state(step_up_target="block_card"), {"toolsCalled": ["get_balance"]},
        "fraud_suspected", "Caller reports two transactions.", now=NOW,
        conversation_id="conv_1", profile_id="mem_p_1", journal_ref="ref-1", store=store)
    assert attrs == {
        "type": "inbound", "name": "Thomas Doe", "from": "+15551234567", "call_sid": SID,
        "language": "es-US", "escalation_reason": "fraud_suspected",
        "ai_summary": "Caller reports two transactions.", "auth_level": 2,
        "customer_id": "C-1042", "factors_verified": ["zip", "ssn4"], "identity_source": "ani",
        "last_tool_attempted": "block_card", "tools_run": ["get_balance"],
        "conversation_id": "conv_1", "profile_id": "mem_p_1", "journal_ref": "ref-1",
        "escalated_at": "2025-10-10T15:04:05Z", "test": False,
    }
    json.dumps(attrs)


def test_unverified_caller_has_no_identity() -> None:
    attrs = build_task_attributes(state(verified_customer_id=None, level=0, factors_verified=()),
                                  {}, "caller_requested_human", "wants a person", now=NOW,
                                  store=MemoryStore())
    assert attrs["customer_id"] is None and attrs["name"] is None and attrs["auth_level"] == 0


def test_last_tool_falls_back_to_pending_then_journal() -> None:
    pt = PendingTool("t1", "make_payment", {})
    assert build_task_attributes(state(pending_tool=pt), {}, "complaint", "x",
                                 now=NOW)["last_tool_attempted"] == "make_payment"
    assert build_task_attributes(state(), {"toolsCalled": ["a", "b"]}, "complaint", "x",
                                 now=NOW)["last_tool_attempted"] == "b"


@pytest.mark.parametrize("reason", ["bogus", None, 3, "", "FRAUD_SUSPECTED"])
def test_unknown_reason_maps_to_unable_to_resolve(reason: object) -> None:
    assert build_task_attributes(state(), {}, reason, "x", now=NOW)["escalation_reason"] == "unable_to_resolve"
    assert normalize_reason(reason) == ("unable_to_resolve", False)


def test_reason_list_is_a_parameter() -> None:
    attrs = build_task_attributes(state(), {}, "billing", "x", now=NOW, reasons=["billing"])
    assert attrs["escalation_reason"] == "billing"
    attrs = build_task_attributes(state(), {}, "fraud_suspected", "x", now=NOW, reasons=["billing"])
    assert attrs["escalation_reason"] == "unable_to_resolve"


def test_summary_capped_and_control_chars_stripped() -> None:
    assert len(build_task_attributes(state(), {}, "complaint", "a" * 5000, now=NOW)["ai_summary"]) == 1000
    s = clean_summary(SID, "line1\nline2\x00\x1b[31m​red‮")
    assert s == "line1 line2[31mred"
    assert clean_summary(SID, None) == ""


def test_summary_factor_values_redacted() -> None:
    sg.register_sensitive(SID, ["94110", "4821"])
    attrs = build_task_attributes(state(), {}, "complaint",
                                  "Caller said zip 9 4 1 1 0 and ssn 48-21.", now=NOW)
    assert "94110" not in json.dumps(attrs).replace(" ", "").replace("-", "")
    assert "4821" not in attrs["ai_summary"].replace(" ", "").replace("-", "")
    assert "[redacted]" in attrs["ai_summary"]


def test_test_flag_must_be_bool() -> None:
    assert build_task_attributes(state(), {}, "complaint", "x", now=NOW, test=True)["test"] is True
    with pytest.raises(TypeError):
        build_task_attributes(state(), {}, "complaint", "x", now=NOW, test="true")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- defence in depth


def test_summary_is_pii_masked_or_withheld_with_a_guard() -> None:
    from owlbank.callstate import CallState
    from owlbank.pii import FakePiiRedactor, PiiGuard
    from owlbank.task_attributes import build_task_attributes

    st = CallState(call_sid="CA" + "e" * 32, expires_at=2_000_000_000, language="es-US")
    raw = "tarjeta 4539 1488 0343 6467 por favor"
    fake = FakePiiRedactor()
    attrs = build_task_attributes(st, {}, "caller_requested_human", raw, now=1_900_000_000,
                                  pii=PiiGuard(fake), withheld_summary="W")
    assert attrs["ai_summary"] == "tarjeta [CREDIT_DEBIT_NUMBER] por favor"
    assert fake.languages == ["es"]
    down = build_task_attributes(st, {}, "caller_requested_human", raw, now=1_900_000_000,
                                 pii=PiiGuard(FakePiiRedactor(fail=True)), withheld_summary="W")
    assert down["ai_summary"] == "W" and down["escalation_reason"] == "caller_requested_human"


def test_summary_pii_check_failure_logs_and_falls_back_to_withheld_text(caplog: pytest.LogCaptureFixture) -> None:
    """This error path used an undefined ``log`` and raised NameError instead of logging."""
    import logging

    from tests.logutil import lines

    class ExplodingGuard:
        def text(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError("pii backend down")

    with caplog.at_level(logging.WARNING, logger="owlbank.task_attributes"):
        attrs = build_task_attributes(state(), {}, "complaint", "some summary", now=NOW,
                                      pii=ExplodingGuard(), withheld_summary="W")  # type: ignore[arg-type]
    assert attrs["ai_summary"] == "W"  # the escalation still goes ahead with the withheld text
    [rec] = [r for r in caplog.records if r.name == "owlbank.task_attributes"]
    assert rec.levelno == logging.WARNING
    [fields] = lines([rec], "owlbank.task_attributes")
    assert fields["event"] == "summary_pii_check_failed"
    assert fields["error_class"] == "RuntimeError"
    assert fields["decision"] == "fallback_withheld_text"
    assert "some summary" not in caplog.text  # the summary text itself is never logged


def test_keyed_phone_identity_source_is_carried_without_a_number() -> None:
    """The agent learns the caller was identified by a keyed-in number; no digits other
    than the calling number (``from``) appear, and the call state has no keyed number to leak."""
    keyed = "+14155550123"
    attrs = build_task_attributes(state(identity_source="keyed_phone"), {}, "complaint",
                                  "caller keyed a number", now=NOW, store=MemoryStore())
    assert attrs["identity_source"] == "keyed_phone"
    assert keyed not in json.dumps(attrs) and keyed[2:] not in json.dumps(attrs)
    assert not any(k for k in attrs if "phone" in k and k != "identity_source")


def test_identity_source_defaults_to_ani() -> None:
    attrs = build_task_attributes(state(), {}, "complaint", "x", now=NOW)
    assert attrs["identity_source"] == "ani"
