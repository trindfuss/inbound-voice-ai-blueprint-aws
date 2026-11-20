"""AgentCore hosting properties: the 60-minute cap's silent reconnect.

* ``classify_close`` is ``stream_limit`` iff the hosting is ``agentcore``, the session ``failed``
  with ErrorCode 64105, there is no ``HandoffData``, and at least 3,540 s have passed since
  ``relay_minted_at``; on ``fargate`` it is always ``other`` (generated statuses, codes,
  ``HandoffData`` and elapsed times clustered around the threshold).
* Over any sequence of ``connect-action`` failures on ``agentcore``: a ``stream_limit`` close
  reconnects with no apology and an empty greeting, leaves ``relay_failures`` unchanged, and
  happens at most ``MAX_STREAM_RECONNECTS`` times per call; every other close takes today's path
  (apology and reconnect while ``relay_failures`` allows, then a human).
* On ``fargate`` the same sequence gives what today's front door gives whatever the elapsed time:
  the answers equal those of the same sequence with the clock standing still.

The real :class:`RuntimeUrlSigner` signs with fixed fake credentials (no AWS call).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.app import (
    MAX_STREAM_RECONNECTS,
    STREAM_LIMIT_MIN_AGE_S,
    FrontDoor,
)
from owlbank.callstate import CallState
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_agentcore import agentcore, signer
from tests.component.test_frontdoor_support import CLOCK, Harness, build, says

SID = "CA" + "e" * 30 + "ab"

_FARGATE = build().app
_AGENTCORE = agentcore(build(), signer()).app
APOLOGY = _FARGATE.d.catalog.text("error.generic", "en-US")

#: Elapsed seconds since the URL was minted, most of them near the 3,540 s threshold.
elapsed = st.one_of(st.integers(min_value=STREAM_LIMIT_MIN_AGE_S - 5, max_value=STREAM_LIMIT_MIN_AGE_S + 5),
                    st.integers(min_value=-60, max_value=20_000),
                    st.floats(min_value=STREAM_LIMIT_MIN_AGE_S - 1.0, max_value=STREAM_LIMIT_MIN_AGE_S + 1.0,
                              allow_nan=False))
statuses = st.one_of(st.sampled_from(["failed", "ended", "completed", "in-progress", "Failed", ""]),
                     st.text(max_size=12))
codes = st.one_of(st.sampled_from(["64105", " 64105 ", "64107", "64102", "64101", "", "6410"]),
                  st.text(alphabet="0123456789 ", max_size=6))
handoffs = st.one_of(st.none(), st.just(""), st.just("  "),
                     st.just(json.dumps({"reason": "tool", "tool": "get_balance"})),
                     st.text(max_size=20))


def _expected(hosting: str, params: dict[str, str], minted: int | None, now: float) -> str:
    return "stream_limit" if (
        hosting == "agentcore" and minted is not None
        and params.get("SessionStatus") == "failed"
        and not (params.get("HandoffData") or "").strip()
        and (params.get("ErrorCode") or "").strip() == "64105"
        and now - minted >= STREAM_LIMIT_MIN_AGE_S) else "other"


@settings(max_examples=500, deadline=None)
@given(status=statuses, code=st.one_of(st.none(), codes), handoff=handoffs, age=elapsed,
       minted=st.one_of(st.none(), st.just(int(CLOCK))), hosting=st.sampled_from(["agentcore", "fargate"]))
def test_classify_close_is_the_duration_rule(status: str, code: str | None, handoff: str | None,
                                                   age: float, minted: int | None, hosting: str) -> None:
    params = {"CallStatus": "in-progress", "SessionStatus": status}
    if code is not None:
        params["ErrorCode"] = code
    if handoff is not None:
        params["HandoffData"] = handoff
    state = CallState(call_sid=SID, expires_at=int(CLOCK) + 86_400, relay_minted_at=minted)
    app = _AGENTCORE if hosting == "agentcore" else _FARGATE
    now = CLOCK + age
    got = app.classify_close(params, state, now)
    assert got == _expected(hosting, params, minted, now)
    if hosting == "fargate":
        assert got == "other"


def test_threshold_is_inclusive() -> None:
    state = CallState(call_sid=SID, expires_at=int(CLOCK) + 86_400, relay_minted_at=int(CLOCK))
    params = {"SessionStatus": "failed", "ErrorCode": "64105"}
    assert _AGENTCORE.classify_close(params, state, CLOCK + STREAM_LIMIT_MIN_AGE_S) == "stream_limit"
    assert _AGENTCORE.classify_close(params, state, CLOCK + STREAM_LIMIT_MIN_AGE_S - 1) == "other"


#: One ``connect-action`` failure: seconds since the previous answer, and the ErrorCode.
failures = st.lists(st.tuples(st.one_of(st.integers(0, 120), st.integers(3530, 3700)),
                              st.sampled_from(["64105", "64105", "64105", "64107"])),
                    min_size=1, max_size=9)


def _call(hosting: str, max_reconnects: int, steps: list[tuple[int, str]], *,
          freeze: bool = False) -> tuple[Harness, list[ET.Element]]:
    h = build(config_kw={"max_relay_reconnects": max_reconnects})
    if hosting == "agentcore":
        agentcore(h, signer())
    now = [CLOCK]
    h.app = FrontDoor(replace(h.app.d, clock=lambda: now[0]), h.app.cfg)
    h.login_thomas(SID)
    roots = []
    for tick, code in steps:
        if not freeze:
            now[0] += tick
        roots.append(h.xml("connect-action", SID, CallStatus="in-progress", SessionStatus="failed",
                           ErrorCode=code))
        if roots[-1].find("./Connect") is None:
            break  # escalated: the call left the AI stage
    return h, roots


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(steps=failures, max_reconnects=st.integers(min_value=0, max_value=6))
def test_silent_reconnects_never_apologize_and_are_capped(steps: list[tuple[int, str]],
                                                                max_reconnects: int) -> None:
    try:
        h, roots = _call("agentcore", max_reconnects, steps)
        silent = failures_seen = 0
        for (tick, code), root in zip(steps, roots, strict=False):
            # Each answer before an escalation minted a URL, so ``tick`` is the session's age.
            cap = code == "64105" and tick >= STREAM_LIMIT_MIN_AGE_S
            if cap and silent < MAX_STREAM_RECONNECTS:
                silent += 1
                cr = root.find("./Connect/ConversationRelay")
                assert says(root) == [] and cr is not None and cr.get("welcomeGreeting") == ""
            else:
                failures_seen += 1
                if failures_seen <= max_reconnects:  # today's drop path: apology, reconnect
                    assert says(root)[0] == APOLOGY and root.find("./Connect") is not None
                else:  # then a human
                    assert root.find("./Connect") is None and root.find("Enqueue") is not None
        sess = h.session(SID)  # after the whole sequence
        assert sess.stream_reconnects == silent <= MAX_STREAM_RECONNECTS
        assert sess.relay_failures == failures_seen  # silent reconnects never count
    finally:
        clear_sensitive(SID)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(steps=failures)
def test_fargate_answers_do_not_depend_on_elapsed_time(steps: list[tuple[int, str]]) -> None:
    def shape(root: ET.Element) -> tuple[Any, ...]:
        cr = root.find("./Connect/ConversationRelay")
        return (tuple(says(root)), cr is not None and cr.get("welcomeGreeting"),
                root.find("Enqueue") is not None)

    try:
        h, timed = _call("fargate", 1, steps)
        assert h.session(SID).stream_reconnects == 0
        clear_sensitive(SID)
        _, frozen = _call("fargate", 1, steps, freeze=True)
        assert [shape(r) for r in timed] == [shape(r) for r in frozen]
    finally:
        clear_sensitive(SID)
