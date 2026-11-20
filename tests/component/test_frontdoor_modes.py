"""The front door's slot modes.

* with every effective mode ``deterministic``, by the console (``default_mode``)
  or by the deploy ceiling, the resolver is never called on ``login``, ``stepup`` or ``confirm``.
* 2.3 (rewritten behaviour): in ``deterministic`` an off-script request is re-asked, not an
  intent switch.
* a policy edit mid-flow changes nothing until the next flow; a lower ceiling
  restricts the frozen mode at once.
* ``slot_resolution``: the value for a non-sensitive slot only, never the words.
* A messy SSN last four is accepted unconfirmed (``confirm: never``) and never read back.
* An intent switch with no value reaches the agent unchanged (ZIP isn't sensitive).
* A keypad-only slot never calls the resolver.
* ``slot_modes`` is logged once per cold start.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

import pytest

from lambdas.frontdoor import handler
from lambdas.frontdoor.app import FrontDoorConfig
from owlbank.policy import Policy
from owlbank.slots import FakeResolver, ResolverOutput
from tests.component.test_frontdoor_caller_id_trust import S, Z, confirm_handoff, to_level3
from tests.component.test_frontdoor_support import (
    DATA,
    THOMAS,
    Harness,
    build,
    relay,
    says,
    sid,
)

ZIP_CLARIFY = "Please enter the 5 digits of your ZIP code on your keypad."
SSN_PROMPT = "Please say or enter the last four digits of your Social Security number. If you can't, press pound."


class Spy(FakeResolver):
    """A resolver that records every call (and answers ``none``)."""


def _policy(**resolution: Any) -> Policy:
    raw = json.loads((DATA / "policy.json").read_text())
    raw["slot_resolution"].update(resolution)
    return Policy.from_dict(raw)


def _deterministic_by(how: str) -> Harness:
    if how == "ceiling":
        return build(resolver=Spy(), config_kw={"slot_mode_ceiling": "deterministic"})
    h = build(resolver=Spy())
    h.app.d.policy = _policy(default_mode="deterministic")
    return h


def _speak(h: Harness, call: str, text: str):
    return h.xml("factor", call, SpeechResult=text, Confidence="0.9")


def resolutions(h: Harness, call: str) -> list[dict[str, Any]]:
    return [e for e in h.journal(call) if e["type"] == "slot_resolution"]




@pytest.mark.parametrize("how", ["ceiling", "console"])
def test_deterministic_never_calls_the_resolver_on_any_flow(how: str) -> None:
    h = _deterministic_by(how)
    call = sid(1)
    h.start(call, THOMAS)
    assert says(_speak(h, call, "actually, what are your branch hours?")) == [ZIP_CLARIFY]
    h.factor(call, Z)
    h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
    h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    _speak(h, call, "eh, one, hold on... two, three and four")
    h.factor(call, S)
    from tests.component.test_frontdoor_support import OTP
    h.factor(call, OTP)
    assert h.state(call).level == 3
    confirm_handoff(h, call)
    _speak(h, call, "go ahead and do it I guess")
    assert h.resolver.calls == []
    modes = {e["effective_mode"] for e in resolutions(h, call)}
    assert modes == {"deterministic"}


def test_deterministic_ceiling_builds_no_resolver_and_no_client() -> None:
    made: list[str] = []
    cfg = FrontDoorConfig(relay_ws_url="wss://x.example/ws", slot_mode_ceiling="deterministic")
    assert handler.build_resolver(cfg, lambda *a, **k: made.append("client")) is None
    resolver = handler.build_resolver(replace(cfg, slot_mode_ceiling="assisted"),
                                      lambda *a, **k: made.append("client"))
    assert resolver is not None and made == []  # built on first use only (LazyClient)


def test_an_off_script_request_in_assisted_reaches_the_resolver() -> None:
    h = build(resolver=FakeResolver([ResolverOutput(kind="intent_switch")]))
    call = sid(2)
    h.start(call, THOMAS)
    said = "actually, what are your branch hours?"
    root = _speak(h, call, said)
    assert relay(root) is not None
    env = h.pending_result(call)
    assert env.status == "interrupted" and env.callerUtterance == said  # unchanged
    assert h.resolver.calls[0]["req"].utterance == said and h.resolver.calls[0]["flow"] == "login"




def test_a_policy_edit_mid_flow_waits_for_the_next_flow() -> None:
    h = build(resolver=Spy())
    call = sid(3)
    h.start(call, THOMAS)                                 # login frozen in assisted
    h.app.d.policy = _policy(default_mode="deterministic")
    _speak(h, call, "hmm I am not sure, maybe")
    assert len(h.resolver.calls) == 1                     # still assisted
    h.factor(call, Z)
    h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    h.handoff(call, {"reason": "step-up", "tool": "change_address"})
    _speak(h, call, "hmm I am not sure, maybe")
    assert len(h.resolver.calls) == 1                     # the step-up froze deterministic
    assert [e["effective_mode"] for e in resolutions(h, call)] == ["assisted", "assisted",
                                                                    "deterministic"]


def test_a_lower_ceiling_restricts_a_frozen_flow_at_once() -> None:
    h = build(resolver=Spy())
    call = sid(4)
    h.start(call, THOMAS)
    h.app.cfg = replace(h.app.cfg, slot_mode_ceiling="deterministic")   # a new deploy, same call
    _speak(h, call, "hmm I am not sure, maybe")
    assert h.resolver.calls == []
    (e,) = resolutions(h, call)
    assert (e["configured_mode"], e["effective_mode"], e["ceiling"]) == ("assisted", "deterministic",
                                                                         "deterministic")


# --------------------------------------------------------------------------- journal


def test_slot_resolution_carries_the_value_for_zip_only() -> None:
    h = build()
    call = sid(5)
    to_level3(h, call)
    events = resolutions(h, call)
    assert [(e["slot"], e["match"], e["source"]) for e in events] == [
        ("zip", "exact", "dtmf"), ("ssn4", "exact", "dtmf"), ("otp", "exact", "dtmf")]
    zip_event, ssn_event, otp_event = events
    assert zip_event["value"] == Z
    assert "value" not in ssn_event and "value" not in otp_event
    for e in events:
        assert e["flow"] in ("login", "stepup") and e["asked"] == "none" and e["answer"] == "none"
        assert e["correction"] is False and e["mode_source"] == "default"
    assert S not in json.dumps(events) and "246813" not in json.dumps(events)


def test_factor_answer_log_has_the_modes_and_no_sensitive_value(caplog) -> None:
    caplog.set_level(logging.INFO)
    h = build()
    call = sid(6)
    to_level3(h, call)
    recs = [r for r in caplog.records if getattr(r, "event", None) == "factor_answer"]
    assert [r.factor for r in recs] == ["zip", "ssn4", "otp"]
    for r in recs:
        assert r.effective_mode == "assisted" and r.configured_mode == "assisted"
        assert r.match == "exact" and r.source == "dtmf" and r.model == "not_called"
    assert not any(hasattr(r, "value") for r in recs)  # login factors: never in the log line


# --------------------------------------------------------------------------- SSN last four


def test_a_messy_ssn4_is_accepted_unconfirmed_and_never_read_back() -> None:
    said = f"eh, {' '.join(S[:2])}, hold on... {' and '.join(S[2:])}"
    h = build(resolver=FakeResolver([ResolverOutput(kind="inferred", value=S)]))
    call = sid(7)
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    assert SSN_PROMPT in says(h.handoff(call, {"reason": "step-up", "tool": "change_address"}))
    root = _speak(h, call, said)
    assert relay(root) is not None and h.state(call).level == 2   # verified, no read-back
    prompts = [e["prompt_key"] for e in h.journal(call) if e["type"] == "prompt"]
    assert "slot.confirm_digits" not in prompts
    (e,) = [e for e in resolutions(h, call) if e["slot"] == "ssn4"]
    assert (e["match"], e["source"], e["model"]) == ("inferred", "model", "ok") and "value" not in e


def test_a_sensitive_value_the_resolver_cant_ground_is_reasked() -> None:
    h = build(resolver=FakeResolver([ResolverOutput(kind="exact", value="9876")]))
    call = sid(8)
    h.login_thomas(call)
    h.set_pending_tool(call, "authenticate", {"tool": "change_address"})
    h.handoff(call, {"reason": "step-up", "tool": "change_address"})
    verifies = h.lam.ops().count("verify")
    root = _speak(h, call, "it's the usual, I always forget it")
    assert h.lam.ops().count("verify") == verifies
    assert says(root) == ["Please enter the 4 digits of your Social Security number on your keypad."]
    (e,) = [e for e in resolutions(h, call) if e["slot"] == "ssn4"]
    assert (e["match"], e["model"]) == ("none", "ungrounded")


# --------------------------------------------------------------------------- keypad-only


def test_a_keypad_only_slot_never_calls_the_resolver() -> None:
    h = build(resolver=Spy(), config_kw={"accept_spoken_factors": False})
    call = sid(9)
    h.start(call, THOMAS)
    _speak(h, call, "hmm I am not sure, maybe")
    assert h.resolver.calls == []




def test_slot_modes_is_logged_once_per_cold_start(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO)
    h = build()
    monkeypatch.setattr(handler, "build_app", lambda env=None: h.app)
    handler.new_app()
    (line,) = [r for r in caplog.records if getattr(r, "event", None) == "slot_modes"]
    assert line.ceiling == "assisted" and line.default_mode == "assisted"
    assert line.slots["zip"]["login"]["effective"] == "assisted"
    assert set(line.slots) == set(h.app.d.policy.slots())
    assert "94105" not in json.dumps(line.slots)
