"""Slot modes end to end.

With every effective mode ``deterministic``, by the deploy ceiling or by the console
(``default_mode``), over every flow (``login``, ``stepup``, ``confirm``: the card question, its
follow-ups and the read-back) and any Gather inputs, the resolver is never called and the Bedrock
client factory is never called:

* by the ceiling, :func:`lambdas.frontdoor.handler.build_resolver` gives no resolver at all, and a
  spy resolver injected anyway is never called;
* by the console, the real :class:`~owlbank.slots.BedrockResolver` over a spy client factory is
  built (the ceiling allows the model) and the factory is never called.

Every result kind: the spy answers every resolver output kind (and every failure outcome) in turn,
so if the front door ever reached it, any kind could come back. A positive control runs the same
inputs in ``assisted`` and shows they do reach the resolver (the driver isn't vacuous), with each
result kind handled on each flow (a 200 and TwiML that waits for the caller or leaves the call).

The template half (no ``bedrock:InvokeModel`` with ``slotModeCeiling=deterministic``)
is in ``infra/tests``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import handler
from lambdas.frontdoor.app import FrontDoorConfig
from owlbank import sensitive
from owlbank.policy import Policy
from owlbank.segments import clear_sensitive
from owlbank.slots import FakeResolver, ResolverOutput
from owlbank.slots.resolver import BedrockResolver, client_for_budget, failed
from tests.component.test_frontdoor_card import propose, to_level3
from tests.component.test_frontdoor_support import (
    DATA,
    OTP,
    TEST_FACTORS,
    THOMAS,
    Harness,
    build,
    sid,
)

SEED = json.loads((DATA / "policy.json").read_text())
Z, S = TEST_FACTORS["C-1001"]["zip"], TEST_FACTORS["C-1001"]["ssn4"]
FLOWS = ("login", "stepup", "card", "readback")

#: Every result kind the resolver can return, and every failure outcome.
KINDS: tuple[ResolverOutput, ...] = (
    ResolverOutput(kind="exact", value="purple"), ResolverOutput(kind="exact", value=Z),
    ResolverOutput(kind="inferred", value="red"), ResolverOutput(kind="inferred", value=S),
    ResolverOutput(kind="ambiguous", candidates=("red", "purple")),
    ResolverOutput(kind="ambiguous", candidates=(Z, "98101")),
    ResolverOutput(kind="none"), ResolverOutput(kind="none", reply="yes"),
    ResolverOutput(kind="correction", slot="zip", value=Z),
    ResolverOutput(kind="correction", slot="ssn4", value=S),
    ResolverOutput(kind="correction", slot="card", value="red"),
    ResolverOutput(kind="intent_switch"), ResolverOutput(kind="intent_switch", value=S),
    ResolverOutput(kind="repeat"), ResolverOutput(kind="escalate"),
    ResolverOutput(kind="inferred", value="red", say="The {option:red}?", say_lang="en-US"),
    failed("timeout"), failed("error"), failed("refusal"), failed("malformed"),
)

#: Caller inputs a Gather can get: keys, clean and messy speech, corrections, requests, commands.
INPUTS: tuple[dict[str, str], ...] = (
    {"Digits": Z}, {"Digits": S}, {"Digits": OTP}, {"Digits": "1"}, {"Digits": "2"},
    {"Digits": "9"}, {"Digits": "12"}, {},
    *({"SpeechResult": t, "Confidence": "0.9"} for t in (
        "nine four one oh five", "um, nine four one, wait, oh five I think", "one one one one",
        "eh, one, hold on... one, one and one", "yes", "no", "yeah sure", "the purple card",
        "the one that's purple", "the violet one", "the OwlBank card", "no, I meant the red one",
        "no, the red one", "actually the ZIP should be 94105", "what's my balance?",
        "actually, can I get my recent transactions", "say that again", "a person please",
        "23456, oh, no actually it is 12345", "yes my last four are 1111, and what's my balance?",
        "hmm", "cancel")),
    {"SpeechResult": "nine four one oh five", "Confidence": "0.1"},
)


class Spy(FakeResolver):
    """Answers every kind in :data:`KINDS`, in turn, and records each call."""

    def __init__(self) -> None:
        super().__init__(list(KINDS) * 20)


class SpyFactory:
    """A ``boto3.client`` stand-in: records each call and returns a client whose ``converse``
    fails (a real call never happens)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, *a: Any, **k: Any) -> Any:
        self.calls.append(a)

        class _Client:
            def converse(self, **_: Any) -> Any:
                raise RuntimeError("no network in tests")

        return _Client()


@pytest.fixture(autouse=True)
def _forget_calls() -> Any:
    yield
    for n in range(40):
        clear_sensitive(sid(400 + n))
        sensitive.forget(sid(400 + n))


def _policy(default_mode: str) -> Policy:
    raw = json.loads(json.dumps(SEED))
    raw["slot_resolution"]["default_mode"] = default_mode
    return Policy.from_dict(raw)


def harness(how: str) -> tuple[Harness, SpyFactory]:
    """``how``: ``ceiling`` or ``console`` (both deterministic), or ``assisted`` (the control,
    with a spy resolver) or ``assisted_bedrock`` (the control over the spy factory)."""
    factory = SpyFactory()
    if how == "ceiling":
        cfg = FrontDoorConfig(relay_ws_url="wss://x.example/ws", slot_mode_ceiling="deterministic")
        assert handler.build_resolver(cfg, factory) is None
        h = build(resolver=Spy(), config_kw={"slot_mode_ceiling": "deterministic"})
    elif how in ("console", "assisted_bedrock"):
        h = build(resolver=Spy())
        bedrock = BedrockResolver(client_for_budget(factory))
        h.app.d = replace(h.app.d, resolver=bedrock)
        h.resolver = bedrock  # type: ignore[assignment]
        if how == "console":
            h.app.d.policy = _policy("deterministic")
    else:
        h = build(resolver=Spy())
    return h, factory


def _waiting(root: ET.Element) -> bool:
    return root.find(".//Gather") is not None


def _to_flow(h: Harness, call: str, flow: str) -> ET.Element:
    """The call at ``flow``'s first question."""
    if flow == "login":
        return h.start(call, THOMAS)
    if flow == "stepup":
        h.start(call, THOMAS)
        h.factor(call, Z)
        h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
        return h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
    to_level3(h, call)
    if flow == "card":
        return propose(h, call, absent=True)
    return propose(h, call, "purple")


def _drive(h: Harness, call: str, flow: str, inputs: list[dict[str, str]]) -> list[ET.Element]:
    roots = [_to_flow(h, call, flow)]
    for params in inputs:
        if not _waiting(roots[-1]):
            break
        roots.append(h.xml("factor", call, **params))
    return roots


def _modes(h: Harness, call: str) -> set[str]:
    return {e["effective_mode"] for e in h.journal(call) if e["type"] == "slot_resolution"}


def _spy_calls(h: Harness) -> int:
    return len(getattr(h.resolver, "calls", ()))


# --------------------------------------------------------------------------- (property)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(how=st.sampled_from(("ceiling", "console")), flow=st.sampled_from(FLOWS),
       inputs=st.lists(st.sampled_from(INPUTS), max_size=8))
def test_deterministic_never_calls_the_resolver_or_the_client_factory(
        how: str, flow: str, inputs: list[dict[str, str]]) -> None:
    h, factory = harness(how)
    call = sid(401)
    try:
        roots = _drive(h, call, flow, inputs)
        assert all(r.tag == "Response" for r in roots)
        assert _spy_calls(h) == 0
        assert factory.calls == []
        assert _modes(h, call) <= {"deterministic"}
    finally:
        clear_sensitive(call)
        sensitive.forget(call)


# --------------------------------------------------------------------------- every flow, every kind


@pytest.mark.parametrize("flow", FLOWS)
@pytest.mark.parametrize("kind", range(len(KINDS)), ids=lambda i: f"{KINDS[i].kind}-{i}")
def test_every_flow_and_result_kind_never_reaches_the_model_in_deterministic(flow: str,
                                                                            kind: int) -> None:
    """The same messy turn in each flow, with the spy set to answer ``KINDS[kind]``: in
    ``deterministic`` (ceiling and console) no call; in ``assisted`` the call happens and the
    front door handles that kind (a 200, then a question, the AI, the queue or a hang-up)."""
    turn = {"SpeechResult": "um, I am not sure, maybe the other one, hold on", "Confidence": "0.9"}
    for how in ("ceiling", "console", "assisted"):
        h, factory = harness(how)
        if how == "assisted":
            h.resolver = FakeResolver([KINDS[kind]])
            h.app.d = replace(h.app.d, resolver=h.resolver)
        call = sid(402)
        try:
            roots = _drive(h, call, flow, [turn])
            assert len(roots) == 2, (how, flow)          # the turn was answered
            last = roots[-1]
            assert last.tag == "Response" and len(last) > 0
            if how == "assisted":
                assert _spy_calls(h) == 1, flow
            else:
                assert _spy_calls(h) == 0 and factory.calls == [], (how, flow)
                assert _modes(h, call) == {"deterministic"}
        finally:
            clear_sensitive(call)
            sensitive.forget(call)


def test_the_control_reaches_the_client_factory_in_assisted() -> None:
    """The positive control for the console half: the same Bedrock resolver and spy factory, in
    ``assisted``, does build a client on an unresolved answer (and fails closed without one)."""
    h, factory = harness("assisted_bedrock")
    call = sid(403)
    roots = _drive(h, call, "login",
                   [{"SpeechResult": "um, I am not sure, hold on", "Confidence": "0.9"}])
    assert len(roots) == 2 and _waiting(roots[-1])       # the error is a none: asked again
    assert factory.calls and factory.calls[0][0] == "bedrock-runtime"
    clear_sensitive(call)


def test_the_ceiling_builds_no_resolver_and_never_calls_the_factory() -> None:
    made: list[Any] = []
    record: Callable[..., Any] = lambda *a, **k: made.append(a)  # noqa: E731
    cfg = FrontDoorConfig(relay_ws_url="wss://x.example/ws", slot_mode_ceiling="deterministic")
    assert handler.build_resolver(cfg, record) is None
    assert handler.build_resolver(replace(cfg, slot_mode_ceiling="assisted"), record) is not None
    assert made == []                                    # built on first use only (LazyClient)
