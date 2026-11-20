"""Escalation without a Flex workflow: the caller is read the briefing and the call ends.

Both modes: with the workflow configured the escalation still enqueues (the Flex characterization
in ``test_frontdoor_humanleg.py`` pins its details); without it nothing is enqueued, dialled or
transferred, and the response ends in a hang-up. The model is a fake; no AWS call is made.
The old tests of the no-workflow guard (an abort that took the apology path) are replaced by these;
none of them was kept because that guard no longer exists.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

import pytest

from lambdas.frontdoor import handler
from lambdas.frontdoor.app import BRIEFING_MARGIN_MS, REQUEST_TIMEOUT_MS, FrontDoorConfig
from owlbank import briefing, callstate, tokens
from owlbank.briefing import BriefingFacts

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    NOBODY, THOMAS, build, clear_sensitive, fd, sid,
)

CLAUSE = "you asked about a balance and a card"
GOODBYE_EN = "Thank you for calling OwlBank. Goodbye."


class FixedModel:
    def __init__(self, text: str | None) -> None:
        self.text, self.calls = text, 0

    def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
        self.calls += 1
        return self.text


class RaisingModel:
    def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
        raise RuntimeError("model down")


class HangingModel:
    """Blocks until released, like a model that never answers inside the budget."""

    def __init__(self) -> None:
        self.release = threading.Event()

    def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
        self.release.wait(5)
        return CLAUSE


def escalate(h, call_sid: str, reason: str = "caller_requested_human", **handoff: str) -> ET.Element:
    h.login_thomas(call_sid)
    callstate.mutate(h.store, call_sid, lambda s: {"ai_sessions": 1})
    h.set_pending_tool(call_sid, "escalate", {"reason": reason, "summary": "Card lost."})
    return h.handoff(call_sid, {"reason": "escalate", "escalationReason": reason,
                                "journal": "jr-7f3a", **handoff})


def spoken(root: ET.Element) -> list[str]:
    return [e.text or "" for e in root.iter("Say")]


def route_phrase(h, reason: str) -> str:
    route = h.app.d.policy.escalation_route(reason)
    return h.app.d.catalog.text(route.phrase_key, "en-US")


def escalation_events(h, call_sid: str) -> list[dict[str, Any]]:
    return [e for e in h.journal(call_sid) if e["type"] == "escalation"]


@pytest.fixture
def briefed(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    made: list[Any] = []

    def make(model: Any = None, **kw: Any):
        h = build(flex=False, briefing_model=model, **kw)
        made.append(h)
        return h

    yield make
    for h in made:
        for call_sid in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
            clear_sensitive(call_sid)


def test_flex_mode_still_enqueues(fd) -> None:
    root = escalate(fd, sid(1))
    assert root.find("Enqueue") is not None
    assert root.find("Hangup") is None


def test_default_mode_reads_the_briefing_and_hangs_up(briefed) -> None:
    model = FixedModel(CLAUSE)
    h = briefed(model)
    call = sid(2)
    root = escalate(h, call)
    assert root.find("Enqueue") is None and root.find("Dial") is None
    assert [c.tag for c in root][-1] == "Hangup"
    said = spoken(root)
    assert "authentication level was" in said[0]
    assert said[1] == CLAUSE
    assert said[-1] == GOODBYE_EN
    assert route_phrase(h, "caller_requested_human") not in said
    assert model.calls == 1
    prompts = [e for e in h.journal(call) if e["type"] == "prompt"]
    assert route_phrase(h, "caller_requested_human") not in [p.get("rendered_text") for p in prompts]
    assert any(p.get("generated") for p in prompts)
    (event,) = escalation_events(h, call)
    assert event["destination"] == "briefing" and event["journal_ref"] == "jr-7f3a"
    st = h.state(call)
    assert st.escalation["destination"] == "briefing" and st.escalation["journal_ref"] == "jr-7f3a"
    assert st.sealed_token is None


def test_pre_items_come_first_and_the_route_phrase_is_silent(briefed) -> None:
    h = briefed(FixedModel(CLAUSE))
    root = escalate(h, sid(3), "step_up_failed")
    said = spoken(root)
    assert route_phrase(h, "step_up_failed") not in said
    assert said[-1] == GOODBYE_EN


def test_verification_risk_reads_level_zero(briefed) -> None:
    h = briefed(FixedModel(CLAUSE))
    call = sid(4)
    root = escalate(h, call, "verification_risk")
    assert "zero, meaning the caller was not verified" in spoken(root)[0]
    st = h.state(call)
    assert st.level == 0 and st.factors_verified == () and st.verified_customer_id is None


def test_spanish_caller_hears_spanish_labels(briefed) -> None:
    h = briefed(FixedModel(CLAUSE))
    call = sid(5)
    h.start(call, "+12025550100", lang_digit="2")
    callstate.mutate(h.store, call, lambda s: {"ai_sessions": 1})
    h.set_pending_tool(call, "escalate", {"reason": "complaint", "summary": ""})
    root = h.handoff(call, {"reason": "escalate", "escalationReason": "complaint"})
    assert "nivel de autenticación" in spoken(root)[0]


@pytest.mark.parametrize("model", [None, FixedModel(None), RaisingModel(), FixedModel("call 123 now"),
                                   FixedModel("one\ntwo"), FixedModel("<b>x</b>")],
                         ids=["none", "empty", "raises", "digit", "newline", "markup"])
def test_fixed_line_when_there_is_no_usable_clause(briefed, model) -> None:
    h = briefed(model)
    root = escalate(h, sid(6), "fraud_suspected")
    said = spoken(root)
    assert len(said) == 2
    assert "a suspected fraud" in said[0]
    assert said[1] == GOODBYE_EN
    assert [c.tag for c in root][-1] == "Hangup"


def test_slow_model_is_cut_off_at_the_budget(briefed) -> None:
    model = HangingModel()
    h = briefed(model, config_kw={"briefing_budget_ms": 100})
    started = time.monotonic()
    try:
        root = escalate(h, sid(7), "complaint")
    finally:
        model.release.set()
    assert time.monotonic() - started < 3
    assert "a complaint" in spoken(root)[0]
    assert CLAUSE not in spoken(root)


def test_unknown_reason_label_falls_back_to_the_default_label(briefed, caplog) -> None:
    def drop(doc: dict[str, Any]) -> None:
        doc["entries"] = [e for e in doc["entries"]
                          if e["key"] != "escalate.briefing.reason.complaint"]

    h = briefed(None, catalog_edit=drop)
    root = escalate(h, sid(8), "complaint")
    assert "that the request could not be completed" in spoken(root)[0]
    assert any("briefing label missing" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_missing_level_label_in_the_caller_language_uses_the_default_language(briefed) -> None:
    def drop(doc: dict[str, Any]) -> None:
        doc["entries"] = [e for e in doc["entries"]
                          if not (e["key"].startswith("escalate.briefing.level.") and e["lang"] == "es-US")]

    h = briefed(FixedModel(CLAUSE), catalog_edit=drop)
    call = sid(9)
    h.start(call, "+12025550100", lang_digit="2")
    callstate.mutate(h.store, call, lambda s: {"ai_sessions": 1})
    h.set_pending_tool(call, "escalate", {"reason": "complaint", "summary": ""})
    root = h.handoff(call, {"reason": "escalate", "escalationReason": "complaint"})
    assert "the caller was not verified" in spoken(root)[0]


def test_caller_hanging_up_during_the_wait_does_not_raise(briefed) -> None:
    call = sid(10)
    statuses: list[int] = []

    class DropsDuring:
        """The status callback ``completed`` arrives while the briefing is being generated."""

        def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
            statuses.append(h.post("status", call, CallStatus="completed")["statusCode"])
            return CLAUSE

    h = briefed(DropsDuring())
    root = escalate(h, call)
    assert statuses == [200]
    assert [c.tag for c in root][-1] == "Hangup"
    assert h.post("status", call, CallStatus="completed")["statusCode"] == 200  # a repeat too


def test_the_briefing_text_is_never_logged(briefed, caplog) -> None:
    secret = "zebra marmalade clause"
    h = briefed(FixedModel(secret))
    escalate(h, sid(11))
    texts: list[str] = []
    for r in caplog.records:
        texts.append(r.getMessage())
        texts.extend(str(v) for v in vars(r).values())
    assert not any(secret in t for t in texts)
    generated = [r for r in caplog.records if r.getMessage() == "briefing_generated"
                 or getattr(r, "event", None) == "briefing_generated"]
    assert len(generated) == 1
    rec = generated[0]
    assert rec.source == "model" and rec.words == len(secret.split()) and rec.latency_ms >= 0
    escalations = [r for r in caplog.records if getattr(r, "event", None) == "escalation"
                   or r.getMessage() == "escalation"]
    assert any(getattr(r, "destination", None) == "briefing" for r in escalations)


def test_a_failing_escalation_gives_apology_goodbye_and_hangup(briefed, monkeypatch) -> None:
    h = briefed(FixedModel(CLAUSE))

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("store down")

    monkeypatch.setattr(h.app, "_briefing_read_out", boom)
    call = sid(12)
    root = escalate(h, call)
    said = spoken(root)
    assert said == [h.app.d.catalog.text("error.generic", "en-US"), GOODBYE_EN]
    assert [c.tag for c in root][-1] == "Hangup"
    assert root.find("Enqueue") is None


def test_a_repeated_request_revokes_each_time_and_journals_once(briefed, monkeypatch) -> None:
    h = briefed(FixedModel(CLAUSE))
    revoked: list[str] = []
    real = tokens.revoke_call
    monkeypatch.setattr("lambdas.frontdoor.app.tokens.revoke_call",
                        lambda store, call_sid: (revoked.append(call_sid), real(store, call_sid))[1])
    call = sid(13)
    first = escalate(h, call)
    before = h.state(call).escalation
    second = h.handoff(call, {"reason": "escalate", "escalationReason": "caller_requested_human",
                              "journal": "jr-other"})
    assert revoked == [call, call]
    assert len(escalation_events(h, call)) == 1
    after = h.state(call).escalation
    assert after["journal_ref"] == before["journal_ref"] and after["at"] == before["at"]
    assert spoken(first) == spoken(second)  # the read-out is built again


def test_a_crash_between_the_state_save_and_the_revoke_is_recovered_by_a_repeat(briefed, monkeypatch) -> None:
    h = briefed(FixedModel(CLAUSE))
    real = tokens.revoke_call
    down = {"on": True}
    seen: list[str] = []  # the call SID of every revoke attempt
    raised: list[str] = []  # the attempts that failed

    def flaky(store: Any, call_sid: str) -> Any:
        seen.append(call_sid)
        if down["on"]:
            raised.append(call_sid)
            raise RuntimeError("token store down")
        return real(store, call_sid)

    monkeypatch.setattr("lambdas.frontdoor.app.tokens.revoke_call", flaky)
    call = sid(16)
    first = escalate(h, call)
    # The caller still gets the fail-safe: the apology, the goodbye and a hang-up.
    assert spoken(first) == [h.app.d.catalog.text("error.generic", "en-US"), GOODBYE_EN]
    assert [c.tag for c in first][-1] == "Hangup"
    saved = h.state(call).escalation  # the escalation state was saved before the crash
    assert saved["destination"] == "briefing"
    assert escalation_events(h, call) == []  # the revoke came first: nothing journaled yet
    # The escalation's own revoke raised, and so did the fail-safe's best-effort one (suppressed).
    assert raised == [call, call] and seen == [call, call]

    down["on"] = False
    second = h.handoff(call, {"reason": "escalate", "escalationReason": "caller_requested_human",
                              "journal": "jr-other"})
    assert [c.tag for c in second][-1] == "Hangup"
    # The repeat revoked once more, for the same call, and that call succeeded.
    assert seen == [call] * 3 and raised == [call, call]
    said = spoken(second)
    assert "authentication level was" in said[0] and said[-1] == GOODBYE_EN  # read-out regenerated
    (event,) = escalation_events(h, call)
    after = h.state(call).escalation
    assert event["journal_ref"] == saved["journal_ref"] == after["journal_ref"]
    assert after["at"] == saved["at"]


def test_deterministic_ceiling_builds_no_model_and_no_client(monkeypatch) -> None:
    def no_client(*a: Any, **k: Any) -> Any:
        raise AssertionError("a Bedrock client was asked for")

    monkeypatch.setattr(briefing, "client_for_budget", no_client)
    monkeypatch.setattr(briefing, "_make_client", no_client)
    h = build(flex=False, config_kw={"slot_mode_ceiling": "deterministic"})
    cfg = h.app.cfg
    assert handler.build_briefing_model(cfg, h.app.d.catalog) is None
    root = escalate(h, sid(14), "complaint")
    assert "a complaint" in spoken(root)[0]
    clear_sensitive(sid(14))


def test_model_is_built_only_when_the_briefing_is_on(monkeypatch) -> None:
    catalog = build(flex=False).app.d.catalog
    flex = FrontDoorConfig(relay_ws_url="wss://x/ws", workflow_sid="WW" + "a" * 32)
    assert handler.build_briefing_model(flex, catalog) is None
    on = FrontDoorConfig(relay_ws_url="wss://x/ws", briefing_budget_ms=1500)
    model = handler.build_briefing_model(on, catalog, client=object())
    assert isinstance(model, briefing.BedrockBriefingModel)


def test_briefing_budget_setting(monkeypatch) -> None:
    env = {"OWLBANK_RELAY_WS_URL": "wss://x/ws"}
    assert FrontDoorConfig.from_env(env).briefing_budget_ms == briefing.BRIEFING_BUDGET_MS
    assert FrontDoorConfig.from_env({**env, "OWLBANK_BRIEFING_BUDGET_MS": "2500"}).briefing_budget_ms == 2500
    for bad in ("abcx", "42", "9001"):
        with pytest.raises(ValueError, match="OWLBANK_BRIEFING_BUDGET_MS") as err:
            FrontDoorConfig.from_env({**env, "OWLBANK_BRIEFING_BUDGET_MS": bad})
        assert bad not in str(err.value)


def test_a_clause_that_is_not_valid_xml_is_replaced_by_the_fixed_line(briefed) -> None:
    h = briefed(FixedModel("a clause with a control character \x1b in it"))
    root = escalate(h, sid(15), "complaint")  # parses: the response is well formed
    said = spoken(root)
    assert len(said) == 2 and "a complaint" in said[0]
    assert [c.tag for c in root][-1] == "Hangup"


# --------------------------------------------------------------------------- the request's time left


def _with_remaining(h, monkeypatch, ms: float) -> None:
    real = h.app.handle
    monkeypatch.setattr(h.app, "handle", lambda event, **_kw: real(event, remaining_ms=lambda: ms))


def _record_budgets(monkeypatch) -> list[int]:
    budgets: list[int] = []
    real = briefing.generate

    def spy(model: Any, facts: Any, **kw: Any) -> Any:
        budgets.append(kw["budget_ms"])
        return real(model, facts, **kw)

    monkeypatch.setattr("lambdas.frontdoor.app.briefing.generate", spy)
    return budgets


def test_little_time_left_skips_the_model_and_speaks_the_fixed_line(briefed, monkeypatch, caplog) -> None:
    model = FixedModel(CLAUSE)
    h = briefed(model)
    _with_remaining(h, monkeypatch, BRIEFING_MARGIN_MS + 200)  # under the 300 ms floor
    root = escalate(h, sid(16), "complaint")
    said = spoken(root)
    assert model.calls == 0 and CLAUSE not in said
    assert len(said) == 2 and "a complaint" in said[0] and said[-1] == GOODBYE_EN
    assert [c.tag for c in root][-1] == "Hangup"
    assert any(getattr(r, "event", None) == "briefing_skipped" or r.getMessage() == "briefing_skipped"
               for r in caplog.records)


def test_ample_time_uses_the_configured_budget(briefed, monkeypatch) -> None:
    model = FixedModel(CLAUSE)
    h = briefed(model, config_kw={"briefing_budget_ms": 2000})
    budgets = _record_budgets(monkeypatch)
    _with_remaining(h, monkeypatch, 9000)
    root = escalate(h, sid(17), "complaint")
    assert budgets == [2000] and model.calls == 1 and CLAUSE in spoken(root)


def test_the_budget_is_capped_by_the_time_left_less_the_margin(briefed, monkeypatch) -> None:
    h = briefed(FixedModel(CLAUSE), config_kw={"briefing_budget_ms": 2000})
    budgets = _record_budgets(monkeypatch)
    _with_remaining(h, monkeypatch, BRIEFING_MARGIN_MS + 700)
    escalate(h, sid(18), "complaint")
    assert budgets == [700]


def test_without_a_context_the_elapsed_request_time_counts(briefed) -> None:
    h = briefed(FixedModel(CLAUSE), config_kw={"briefing_budget_ms": 2000})
    h.app._remaining_ms = None  # noqa: SLF001
    h.app._request_started = time.monotonic() - 9.0  # noqa: SLF001  9 s of the 10 s are gone
    assert h.app._briefing_budget_ms() < 300  # noqa: SLF001
    h.app._request_started = time.monotonic()  # noqa: SLF001
    assert h.app._briefing_budget_ms() == 2000  # noqa: SLF001
    assert REQUEST_TIMEOUT_MS == 10_000


# --------------------------------------------------------------------------- other entries to the briefing


def test_a_failed_login_says_the_apology_then_the_briefing_then_goodbye(briefed) -> None:
    h = briefed(FixedModel(CLAUSE), config_kw={"fallbacks": {"verify_failed": "escalate"}})
    call = sid(19)
    h.start(call, THOMAS)
    for z in ("94106", "94107"):
        h.factor(call, z)
    root = h.factor(call, "94108")
    said = spoken(root)
    assert said[0] == h.app.d.catalog.text("verify.failed", "en-US")
    assert "authentication level was" in said[1] and said[2] == CLAUSE
    assert said[-1] == GOODBYE_EN and len(said) == 4
    assert [c.tag for c in root][-1] == "Hangup"
    assert root.find("Enqueue") is None
    assert len(escalation_events(h, call)) == 1


def _no_phone_question(doc: dict[str, Any]) -> None:
    doc["ani_trust"]["ask_phone_on_no_match"] = False


def test_no_match_escalating_gives_the_briefing_and_hangs_up(briefed) -> None:
    h = briefed(FixedModel(CLAUSE), config_kw={"fallbacks": {"no_match": "escalate"}},
                policy_edit=_no_phone_question)
    call = sid(20)
    root = h.start(call, NOBODY)
    said = spoken(root)
    assert "authentication level was" in said[0] and CLAUSE in said and said[-1] == GOODBYE_EN
    assert root.find("Enqueue") is None and [c.tag for c in root][-1] == "Hangup"
    assert len(escalation_events(h, call)) == 1


def test_lookup_error_escalating_gives_the_briefing_and_hangs_up(briefed) -> None:
    h = briefed(FixedModel(CLAUSE), config_kw={"fallbacks": {"lookup_error": "escalate"}})
    h.lam.fail_ops.add("ani_lookup")
    call = sid(21)
    root = h.start(call, THOMAS)
    said = spoken(root)
    assert said[0] == h.app.d.catalog.text("error.lookup_unavailable", "en-US")
    assert "authentication level was" in said[1] and said[-1] == GOODBYE_EN
    assert root.find("Enqueue") is None and [c.tag for c in root][-1] == "Hangup"
    assert len(escalation_events(h, call)) == 1


def test_a_handoff_with_the_bridges_escalation_record_keeps_the_bridges_ref(briefed, monkeypatch) -> None:
    h = briefed(FixedModel(CLAUSE))
    revoked: list[str] = []
    real = tokens.revoke_call
    monkeypatch.setattr("lambdas.frontdoor.app.tokens.revoke_call",
                        lambda store, call_sid: (revoked.append(call_sid), real(store, call_sid))[1])
    call = sid(22)
    h.login_thomas(call)
    summary = "agent summary that must stay unspoken"
    callstate.mutate(h.store, call, lambda s: {
        "ai_sessions": 1,
        "escalation": {"reason": "caller_requested_human", "summary": summary, "at": 1_700_000_000,
                       "source": "bridge", "journal_ref": "jr-bridge1"}})
    h.set_pending_tool(call, "escalate", {"reason": "caller_requested_human", "summary": summary})
    root = h.handoff(call, {"reason": "escalate", "escalationReason": "caller_requested_human",
                            "journal": "jr-handoff"})
    (event,) = escalation_events(h, call)
    assert event["journal_ref"] == "jr-bridge1" and event["destination"] == "briefing"
    st = h.state(call)
    assert st.escalation["journal_ref"] == "jr-bridge1" and st.escalation["destination"] == "briefing"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", st.escalation["at"])
    assert st.sealed_token is None and revoked == [call]
    assert not any(summary in t for t in spoken(root))
    assert [c.tag for c in root][-1] == "Hangup"


def test_a_control_character_clause_that_passes_the_check_is_caught_at_the_parse(
        briefed, monkeypatch, caplog) -> None:
    monkeypatch.setattr("owlbank.briefing.check_output", lambda raw: raw)  # as if the check missed it
    h = briefed(FixedModel("a clause with a control character \x1b in it"))
    root = escalate(h, sid(23), "complaint")
    said = spoken(root)
    assert len(said) == 2 and "a complaint" in said[0] and said[-1] == GOODBYE_EN
    assert [c.tag for c in root][-1] == "Hangup"
    assert any(getattr(r, "event", None) == "briefing_unparseable" or r.getMessage() == "briefing_unparseable"
               for r in caplog.records)
