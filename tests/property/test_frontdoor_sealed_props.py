"""Properties of the front door's sealed values across instances, and one answering-turn text

Two front-door Lambda instances are simulated in one process: each has its own app object and its
own registry (``segments`` registered values and ``sensitive`` process state), over one shared
store and sealer. For any interleaving of requests for one call across the two:

* after each request (each begins with a call-state load), the instance's registry holds every
  sensitive value of 4 or more characters accepted so far on either instance, and no shorter one
;
* every intent switch's answering-turn text is one string on every surface: the engine's
  ``utterance_for_ai``, the envelope's ``callerUtterance``, call state's
  ``interrupted_utterance``, and the screen pop's ``ai_summary`` when an agent hands the turn back
  as its summary (on the other instance).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import HealthCheck, event, example, given, settings
from hypothesis import strategies as st

import lambdas.frontdoor.slotflow as fd_slotflow
from lambdas.frontdoor import state as sessions
from lambdas.frontdoor.app import FrontDoor
from owlbank import segments, sensitive
from owlbank.policy import Policy
from owlbank.segments import MIN_SENSITIVE_LEN
from owlbank.slots import FakeResolver, ResolverOutput
from tests.component.test_frontdoor_support import DATA, THOMAS, Harness, build, sid

CALL = sid(40)
INTENT_SWITCH = ResolverOutput(kind="intent_switch")
PROPS = settings(max_examples=150, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow,
                                        HealthCheck.function_scoped_fixture])


def _policy() -> Policy:
    """The seed with ZIP flagged sensitive and 3 digits long: one short sensitive slot
    next to SSN last four (4 digits)."""
    raw = json.loads((DATA / "policy.json").read_text())
    raw["slots"]["zip"]["sensitive"] = True
    raw["factors"]["zip"]["constraints"] = {"length": 3}
    return Policy.from_dict(raw)


POLICY = _policy()
FLOWS = {
    "zip": {"kind": "login", "target_level": 1, "factor": "zip"},
    "ssn4": {"kind": "stepup", "tool": "change_address", "target_level": 2, "factor": "ssn4"},
}


class _Instance:
    def __init__(self, h: Harness) -> None:
        self.h = replace(h, app=FrontDoor(h.app.d, h.app.cfg), bodies=[])
        self.registered: dict[str, set[str]] = {}
        self.state = sensitive._ProcessState()

    @contextlib.contextmanager
    def active(self) -> Iterator[Harness]:
        saved = segments._SENSITIVE, sensitive._state
        segments._SENSITIVE, sensitive._state = self.registered, self.state  # type: ignore[assignment]
        try:
            yield self.h
        finally:
            segments._SENSITIVE, sensitive._state = saved  # type: ignore[assignment]


def _set_flow(h: Harness, slot: str) -> None:
    sess = sessions.load(h.store, CALL)
    assert sess is not None
    flow = dict(FLOWS[slot])
    fd_slotflow.start_flow(h.app, sess, flow.pop("kind"), **flow)  # modes frozen
    sessions.save(h.store, sess)


WORDS = ["what's", "my", "balance", "uh", "I", "think", "it", "ends", "in", "please", "the",
         "one", "two", "seven", "nine", "uno", "dos", "cuatro", "5", "38"]
step = st.one_of(
    st.tuples(st.sampled_from([0, 1]), st.just("answer"), st.sampled_from(["zip", "ssn4"]),
              st.text(alphabet="0123456789", min_size=3, max_size=4)),
    st.tuples(st.sampled_from([0, 1]), st.just("load"), st.just(""), st.just("")),
    st.tuples(st.sampled_from([0, 1]), st.just("switch"), st.just("ssn4"),
              st.lists(st.sampled_from(WORDS), min_size=1, max_size=8).map(" ".join)),
)


@PROPS
@given(steps=st.lists(step, min_size=1, max_size=7))
# An SSN last four equal to the test clock's year ("2025-10-09T..."): the guard scanned
# escalation.at, so the escalation and every later call-state write raised RedactionError
# (fixed by adding "at" to segments.ID_KEYS). One example per surface it broke:
# the screen pop (no Enqueue), the registry on a later load (the sealed value was never saved),
# the envelope (still the earlier turn's text) and the pending result (never written).
@example(steps=[(0, "answer", "ssn4", "2025"), (0, "switch", "ssn4", "what's")])
@example(steps=[(0, "switch", "ssn4", "what's"), (1, "answer", "ssn4", "2025"),
                (0, "load", "", "")])
@example(steps=[(0, "switch", "ssn4", "what's what's"), (0, "answer", "ssn4", "2025"),
                (0, "switch", "ssn4", "what's")])
@example(steps=[(0, "answer", "zip", "000")] * 3
         + [(1, "answer", "ssn4", "2025"), (1, "switch", "ssn4", "what's")])
def test_registry_after_each_load_and_one_answering_turn(steps: list[tuple[int, str, str, str]],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    h = build(resolver=FakeResolver(default=INTENT_SWITCH))
    h.app.d.policy = POLICY
    instances = [_Instance(h), _Instance(h)]
    results: list[Any] = []
    real_resolve = fd_slotflow.resolve

    def spy(slot: Any, **kw: Any) -> Any:
        res = real_resolve(slot, **kw)
        results.append((slot, res))
        return res

    monkeypatch.setattr(fd_slotflow, "resolve", spy)
    with instances[0].active() as h0:
        h0.start(CALL, THOMAS)
    accepted: set[str] = set()
    for who, action, slot, payload in steps:
        inst, other = instances[who], instances[1 - who]
        results.clear()
        with inst.active() as hx:
            if action == "load":
                hx.app._load(CALL)
            else:
                _set_flow(hx, slot)
                params = {"Digits": payload} if action == "answer" else \
                    {"SpeechResult": payload, "Confidence": "0.9"}
                resp = hx.post("factor", CALL, **params)
                assert resp["statusCode"] == 200
            accepted |= {res.value for s, res in results
                         if s.sensitive and res.value is not None}
            have = segments.sensitive_values(CALL)
            long = {segments._norm_text(v) for v in accepted if len(v) >= MIN_SENSITIVE_LEN}
            short = {segments._norm_text(v) for v in accepted if len(v) < MIN_SENSITIVE_LEN}
            assert long <= have
            if who == 1 and long:
                event("instance B holds a value accepted earlier")
            assert not (short - long) & have
            assert not sensitive.is_unverifiable(CALL)
        switched = [res for _, res in results
                    if res.match == "intent_switch" and res.utterance_for_ai]
        if action != "switch" or not switched:
            continue
        turn = switched[-1].utterance_for_ai
        event("answering turn checked" + (" (placeholder)" if "[ssn4]" in turn else ""))
        env = h.pending_result(CALL)
        assert env.callerUtterance == turn
        assert h.state(CALL).interrupted_utterance == turn
        assert all(v not in turn for v in accepted if len(v) >= MIN_SENSITIVE_LEN)
        with other.active() as ho:
            ho.set_pending_tool(CALL, "escalate", {"reason": "complaint", "summary": turn})
            root = ho.xml("connect-action", CALL, CallStatus="in-progress",
                          SessionStatus="ended", SessionId="VX" + "c" * 32,
                          SessionDuration="12",
                          HandoffData=json.dumps({"reason": "escalate",
                                                  "escalationReason": "complaint"}))
        task = root.find("Enqueue/Task")
        assert task is not None
        assert json.loads(task.text or "{}")["ai_summary"] == " ".join(turn.split())
