"""Segment-traits: the front door starts ``prepare_welcome`` and uses the prepared welcome.

When the last login factor is asked for an identified customer and ``WebActivity`` is
listed, the front door ``Event``-invokes the auth service's ``prepare_welcome`` once per call.
``_login_complete`` speaks ``greeting.offer`` only for a ``matched`` record made for
the verified customer and the call's language whose phrase renders; every other path gives
``greeting.personal`` exactly, and ``welcome_offer`` is set only with the offer (in the write
of ``last_spoken``). Once set it stays for the call (later connects keep it); call end and the
silence hangup clear it.

Signed fake webhooks into the real front door and the real auth service
(``test_frontdoor_support.build``); the auth service's ``prepare_welcome`` is stood in for by
:func:`install`, which writes the record the way the auth service does (``callstate.seal_welcome``).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from lambdas.frontdoor.app import FrontDoorConfig
from lambdas.frontdoor.backend import LambdaBackendClient
from lambdas.frontdoor.fakes import FakeLambdaClient
from owlbank import callstate, last_spoken, welcome
from tests.component.test_frontdoor_support import (
    CLOCK,
    LEES,
    NOBODY,
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    all_fixture_factor_values,
    assert_no_leak,
    build,
    greeting,
    sid,
)

EN, ES = "en-US", "es-US"
GROUPS = ("Contact", "WebActivity")
OFFER_EN = "Hi Thomas. I am the OwlBank virtual agent. Would you like help with our home equity line of credit?"
OFFER_ES = "Hola, Thomas. Soy el agente virtual de OwlBank. ¿Le puedo ayudar con nuestra línea de crédito sobre el valor de su vivienda?"
PERSONAL_EN = "Hi Thomas. I am the OwlBank virtual agent. How can I help you today?"
PERSONAL_ES = "Hola, Thomas. Soy el agente virtual de OwlBank. ¿En qué puedo ayudarle hoy?"
ZIP = TEST_FACTORS["C-1001"]["zip"]
SSN = TEST_FACTORS["C-1001"]["ssn4"]
WRONG_ZIP = "98101"
OTHER_CALL = "CA" + "f" * 32


@dataclass
class Prep:
    """What the stand-in ``prepare_welcome`` writes. ``kind``: ``matched``, another reason of
    :data:`owlbank.welcome.REASONS`, ``none`` (writes nothing), or ``other_call`` (a matched record
    sealed for another call). ``customer_id``/``language``: override the requested ones."""

    kind: str = "matched"
    offer_id: str = "heloc"
    customer_id: str | None = None
    language: str | None = None


def install(h: Harness, prep: Prep) -> None:
    real = h.lam.functions["owlbank-auth"]

    def fn(event: dict[str, Any]) -> Any:
        if event.get("op") != "prepare_welcome":
            return real(event)
        assert set(event) == {"op", "call_sid", "customer_id", "language"}
        if prep.kind == "none":
            return {"result": "ok"}
        reason = welcome.MATCHED if prep.kind in ("matched", "other_call") else prep.kind
        rec = callstate.seal_welcome(
            h.sealer, prep.offer_id if reason == welcome.MATCHED else None,
            call_sid=OTHER_CALL if prep.kind == "other_call" else event["call_sid"],
            customer_id=prep.customer_id or event["customer_id"],
            language=prep.language or event["language"], reason=reason, now=CLOCK)
        callstate.mutate(h.store, event["call_sid"], lambda s: {callstate.F_WELCOME_PREPARED: rec})
        return {"result": "ok"}

    h.lam.functions["owlbank-auth"] = fn  # type: ignore[index]


def harness(prep: Prep | None = None, *, groups: tuple[str, ...] = GROUPS) -> Harness:
    h = build(config_kw={"memory_trait_groups": groups})
    install(h, prep or Prep())
    return h


@pytest.fixture
def leakcheck(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    made: list[Harness] = []
    yield made
    for h in made:
        assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)


def used(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [{"outcome": r.outcome, "reason": getattr(r, "reason", None)}
            for r in caplog.records if r.getMessage() == "welcome_used"]


def spoken(h: Harness, call: str) -> str | None:
    return last_spoken.unseal(h.sealer, h.state(call).last_spoken, call_sid=call, now=CLOCK)


def untrusted_start(h: Harness, call: str) -> None:
    """Thomas with no StirVerstat: the ANI match doesn't count, so login asks ZIP then SSN-4."""
    h.xml("incoming", call, From=THOMAS, To="+12025550100", CallStatus="ringing")
    h.xml("language", call, Digits="")


# --------------------------------------------------------------------------- the trigger


def test_trigger_once_for_the_identified_customer_when_the_last_factor_is_asked(
        leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    h.start(sid(1), THOMAS)  # trusted: ZIP is the only (so the last) factor
    assert h.lam.welcome_requests() == [{"call_sid": sid(1), "customer_id": "C-1001", "language": EN}]
    assert h.lam.event_ops() == ["prepare_welcome"]
    assert h.session(sid(1)).welcome_requested is True
    h.factor(sid(1), WRONG_ZIP)  # retry: ZIP asked again, no second invoke
    h.factor(sid(1), ZIP)
    assert len(h.lam.welcome_requests()) == 1


def test_untrusted_caller_triggers_on_the_last_factor_not_the_first(leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    untrusted_start(h, sid(2))
    assert h.session(sid(2)).flow["factor"] == "zip"
    assert h.lam.welcome_requests() == []  # SSN-4 is still to come
    h.factor(sid(2), ZIP)
    assert h.session(sid(2)).flow["factor"] == "ssn4"
    assert len(h.lam.welcome_requests()) == 1
    root = h.factor(sid(2), SSN)
    assert greeting(root) == OFFER_EN


@pytest.mark.parametrize("ani", [LEES, NOBODY], ids=["several_candidates", "unknown_ani"])
def test_no_trigger_without_an_identified_customer(ani: str, leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    h.start(sid(3), ani)
    assert h.lam.welcome_requests() == []


def test_no_trigger_when_web_activity_is_not_listed(leakcheck: list[Harness],
                                                     caplog: pytest.LogCaptureFixture) -> None:
    h = harness(groups=("Contact",))
    leakcheck.append(h)
    root = h.login_thomas(sid(4))
    assert h.lam.welcome_requests() == []
    assert greeting(root) == PERSONAL_EN
    assert used(caplog) == []  # the feature is off: no line


def test_an_invoke_that_isnt_queued_gives_todays_welcome(leakcheck: list[Harness],
                                                         caplog: pytest.LogCaptureFixture) -> None:
    h = harness()
    leakcheck.append(h)
    h.lam.fail_event_ops.add("prepare_welcome")
    root = h.login_thomas(sid(5))
    assert greeting(root) == PERSONAL_EN
    (req,) = [r for r in caplog.records if r.getMessage() == "welcome_requested"]
    assert req.queued is False and req.levelno == logging.WARNING
    assert used(caplog) == [{"outcome": "not_ready", "reason": None}]


# --------------------------------------------------------------------------- _login_complete


def test_ready_offer_is_the_welcome_and_reaches_the_last_spoken_write(
        leakcheck: list[Harness], caplog: pytest.LogCaptureFixture) -> None:
    h = harness()
    leakcheck.append(h)
    root = h.login_thomas(sid(6))
    assert greeting(root) == OFFER_EN
    st = h.state(sid(6))
    assert st.welcome_offer == callstate.WELCOME_OFFER_MADE
    assert st.welcome_prepared is None
    assert spoken(h, sid(6)) == OFFER_EN
    assert used(caplog) == [{"outcome": "made", "reason": None}]
    prompts = [e for e in h.journal(sid(6)) if e["type"] == "prompt"]
    assert prompts[-1]["prompt_key"] == welcome.GREETING_KEY


def test_welcome_offer_is_written_in_the_same_update_as_last_spoken(leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    writes: list[dict[str, Any]] = []
    real_put = h.store.put

    def put(table: str, item: dict[str, Any], **kw: Any) -> None:
        if table == "calls" and item.get("call_sid") == sid(7):
            writes.append(dict(item))
        real_put(table, item, **kw)

    h.store.put = put  # type: ignore[method-assign]
    h.login_thomas(sid(7))
    made = [w for w in writes if w.get(callstate.F_WELCOME_OFFER) == callstate.WELCOME_OFFER_MADE]
    first = made[0]
    assert first["last_spoken"] is not None and first["ws_nonce_hash"]
    before = writes[: writes.index(first)]
    assert all(w.get("last_spoken") is None for w in before)


def test_spanish_call_gets_the_spanish_offer(leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    h.start(sid(8), THOMAS, lang_digit="2")
    assert h.lam.welcome_requests()[0]["language"] == ES
    root = h.factor(sid(8), ZIP)
    assert greeting(root) == OFFER_ES
    assert h.state(sid(8)).welcome_offer == callstate.WELCOME_OFFER_MADE


@pytest.mark.parametrize(("prep", "outcome", "reason"), [
    (Prep(kind="none"), "not_ready", None),
    (Prep(customer_id="C-1002"), "mismatch", None),
    (Prep(language=ES), "mismatch", None),
    (Prep(offer_id="mortgage"), "no_phrase", None),
    (Prep(kind=welcome.NO_MATCH), "fallback", welcome.NO_MATCH),
    (Prep(kind=welcome.NO_WEB_ACTIVITY), "fallback", welcome.NO_WEB_ACTIVITY),
    (Prep(kind=welcome.MEMORY_TIMEOUT), "fallback", welcome.MEMORY_TIMEOUT),
    (Prep(kind="other_call"), "fallback", "unusable"),
], ids=["not_ready", "other_customer", "other_language", "missing_phrase", "no_match",
        "no_web_activity", "memory_timeout", "other_call"])
def test_every_other_path_gives_greeting_personal_exactly(
        prep: Prep, outcome: str, reason: str | None, leakcheck: list[Harness],
        caplog: pytest.LogCaptureFixture) -> None:
    h = harness(prep)
    leakcheck.append(h)
    root = h.login_thomas(sid(9))
    assert greeting(root) == PERSONAL_EN
    st = h.state(sid(9))
    assert st.welcome_offer is None and st.welcome_prepared is None
    assert spoken(h, sid(9)) == PERSONAL_EN
    assert used(caplog) == [{"outcome": outcome, "reason": reason}]


def test_a_welcome_prepared_after_login_is_never_used(leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    h.lam.defer_events = True
    root = h.login_thomas(sid(10))
    assert greeting(root) == PERSONAL_EN
    assert h.lam.run_deferred() == 1  # the late record lands after the connect
    assert h.state(sid(10)).welcome_offer is None


def test_failed_verification_discards_the_prepared_welcome(leakcheck: list[Harness],
                                                           caplog: pytest.LogCaptureFixture) -> None:
    h = harness()
    leakcheck.append(h)
    h.start(sid(11), THOMAS)
    assert h.state(sid(11)).welcome_prepared is not None
    root = None
    for _ in range(3):
        root = h.factor(sid(11), WRONG_ZIP)
    assert root is not None
    st = h.state(sid(11))
    assert st.level == 0 and st.welcome_prepared is None and st.welcome_offer is None
    assert root.find(".//Enqueue") is not None or root.find(".//ConversationRelay") is not None
    assert all("home equity" not in body for body in h.bodies)
    assert used(caplog) == []  # login never completed


def test_welcome_offer_stays_made_through_later_connects_until_call_end(leakcheck: list[Harness]) -> None:
    """Once spoken, the offer is in the agent's history, so later connects keep
    ``welcome_offer`` (the bridge keeps ``agent.web_activity``); call end clears it."""
    h = harness()
    leakcheck.append(h)
    h.login_thomas(sid(12))
    assert h.state(sid(12)).welcome_offer == callstate.WELCOME_OFFER_MADE
    h.set_pending_tool(sid(12), "authenticate", {"tool": "make_payment"})
    h.handoff(sid(12), {"reason": "step-up", "tool": "make_payment", "journal": "jr-1"})
    h.factor(sid(12), SSN)  # step-up: SSN-4, then the OTP via the redirect path
    root = h.factor(sid(12), OTP)
    assert h.state(sid(12)).level == 3
    assert greeting(root) and greeting(root) != OFFER_EN
    st = h.state(sid(12))
    assert st.welcome_offer == callstate.WELCOME_OFFER_MADE  # a later connect keeps it
    assert spoken(h, sid(12)) == greeting(root)  # last_spoken is the new greeting
    assert len(h.lam.welcome_requests()) == 1  # a step-up never asks again
    h.xml("status", sid(12), CallStatus="completed", CallDuration="30")
    st = h.state(sid(12))
    assert st.welcome_offer is None and st.last_spoken is None


def test_reconnect_after_a_relay_drop_keeps_welcome_offer(leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    h.login_thomas(sid(13))
    h.xml("connect-action", sid(13), CallStatus="in-progress", SessionStatus="failed",
          SessionId="VX" + "c" * 32, ErrorCode="64107")
    st = h.state(sid(13))
    assert st.welcome_offer == callstate.WELCOME_OFFER_MADE
    assert spoken(h, sid(13)) != OFFER_EN  # the reconnect's own greeting


def test_a_connect_without_the_offer_never_sets_welcome_offer(leakcheck: list[Harness]) -> None:
    h = harness(Prep(kind=welcome.NO_MATCH))
    leakcheck.append(h)
    h.login_thomas(sid(14))
    assert h.state(sid(14)).welcome_offer is None
    h.xml("connect-action", sid(14), CallStatus="in-progress", SessionStatus="failed",
          SessionId="VX" + "c" * 32, ErrorCode="64107")
    assert h.state(sid(14)).welcome_offer is None


def test_the_silence_hangup_clears_welcome_offer(leakcheck: list[Harness]) -> None:
    h = harness()
    leakcheck.append(h)
    h.login_thomas(sid(15))
    assert h.state(sid(15)).welcome_offer == callstate.WELCOME_OFFER_MADE
    root = h.handoff(sid(15), {"reason": "no_input_hangup", "journal": "jr-2"})
    assert [c.tag for c in root] == ["Say", "Hangup"]
    st = h.state(sid(15))
    assert st.welcome_offer is None and st.last_spoken is None and st.welcome_prepared is None


# --------------------------------------------------------------------------- backend and config


def test_prepare_welcome_async_is_an_event_invoke_and_reports_failures() -> None:
    lam = FakeLambdaClient({"auth": lambda e: {"result": "ok"}, "bank": lambda e: {}})
    client = LambdaBackendClient(lam, auth_function="auth", bank_function="bank")
    kw = {"call_sid": sid(1), "customer_id": "C-1001", "language": EN}
    assert client.prepare_welcome_async(**kw) is True
    assert lam.invocation_types == ["Event"]
    assert lam.payloads == [{"op": "prepare_welcome", **kw}]
    lam.fail_event_ops.add("prepare_welcome")
    assert client.prepare_welcome_async(**kw) is False

    class NotAccepted(FakeLambdaClient):
        def invoke(self, **_: Any) -> dict[str, Any]:
            return {"StatusCode": 500, "Payload": io.BytesIO(b"")}

    bad = LambdaBackendClient(NotAccepted({}), auth_function="auth", bank_function="bank")
    assert bad.prepare_welcome_async(**kw) is False


@pytest.mark.parametrize(("raw", "groups", "on"), [
    (None, (), False),
    ("", (), False),
    ("Contact", ("Contact",), False),
    ("Contact,WebActivity", ("Contact", "WebActivity"), True),
    (" WebActivity , Contact,WebActivity", ("WebActivity", "Contact"), True),
])
def test_memory_trait_groups_from_env(raw: str | None, groups: tuple[str, ...], on: bool) -> None:
    env = {"OWLBANK_RELAY_WS_URL": "wss://bridge.example/ws"}
    if raw is not None:
        env["OWLBANK_MEMORY_TRAIT_GROUPS"] = raw
    cfg = FrontDoorConfig.from_env(env)
    assert cfg.memory_trait_groups == groups
    assert cfg.welcome_offers is on
