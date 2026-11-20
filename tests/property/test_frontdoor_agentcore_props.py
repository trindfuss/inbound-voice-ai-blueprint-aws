"""AgentCore hosting properties of the front door's ``agentcore`` wiring.

* Over any sequence of connects for one call (first connect,
  dropped sockets, re-entry after a scripted step), every nonce is distinct, travels only as
  ``<Parameter name="n">`` (never in the URL), and the stored hash is that of the last nonce.
  Two URLs minted in the same second can match, so only the nonce must differ.
* The warmer is dispatched only after ``callstate.create`` and only for a
  ``CALL_SID_RE`` ID, at most once per call; URLs are minted only from a loaded call state.
* For any sequence of status callbacks, including repeats and
  out-of-order ones, each call in state that reaches an end status gets exactly one
  ``StopRuntimeSession`` attempt (none on ``fargate``), and the status answer never changes.

The real :class:`RuntimeUrlSigner` signs with fixed fake credentials (no AWS call).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.app import END_STATUSES, FrontDoor
from lambdas.frontdoor.relay import CALL_SID_RE
from owlbank import twiml, ws_nonce
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_agentcore import (
    CheckingSigner,
    RecordingStopper,
    RecordingWarmer,
    agentcore,
    query,
    signer,
)
from tests.component.test_frontdoor_support import CLOCK, TEST_FACTORS, build, relay

EMPTY = twiml.empty()

#: CallSids with no digit run that could look like a factor (the harness's leak rule).
HEX_LETTERS = "abcdef"
call_sids = st.builds(lambda a, b: "CA" + "e" * 30 + a + b,
                      st.sampled_from(HEX_LETTERS), st.sampled_from(HEX_LETTERS))
#: IDs Twilio would never send but the route must survive: none is a CallSid.
not_call_sids = st.one_of(
    st.builds(lambda s: "CA" + s.upper(), st.text(alphabet=HEX_LETTERS, min_size=32, max_size=32)),
    st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-", min_size=1, max_size=40),
)

#: One connect-action outcome after the first connect: a dropped socket (reconnect with an
#: apology), or the bridge handing off for a balance read-out (re-entry as a scripted greeting).
connects = st.lists(st.sampled_from(["drop", "error", "balance"]), min_size=0, max_size=6)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(steps=connects,
       ticks=st.lists(st.integers(min_value=0, max_value=400), min_size=7, max_size=7))
def test_every_connect_gets_a_fresh_nonce(steps: list[str], ticks: list[int]) -> None:
    h = build(config_kw={"max_relay_reconnects": 10})
    now = [CLOCK]
    sid = "CA" + "e" * 30 + "ab"
    try:
        s = CheckingSigner(signer(), h)
        s.inner._clock = lambda: now[0]  # noqa: SLF001 - the credential check follows the clock
        agentcore(h, s)
        h.app.d.clock = lambda: now[0]
        sent = [relay(h.login_thomas(sid))]
        for step, tick in zip(steps, ticks, strict=False):
            now[0] += tick
            if step == "balance":
                h.set_pending_tool(sid, "get_balance", {"account": "checking"})
                root = h.handoff(sid, {"reason": "tool", "tool": "get_balance",
                                        "args": {"account": "checking"}, "journal": "jr-1"})
            else:
                root = h.xml("connect-action", sid, CallStatus="in-progress",
                             SessionStatus="failed" if step == "drop" else "ended")
            sent.append(relay(root))  # every step ends in a fresh <Connect>
        nonces = [n for _, n, _ in s.mints]
        urls = [u for _, _, u in s.mints]
        assert len(nonces) == len(steps) + 1
        assert len(set(nonces)) == len(nonces)
        assert [cr.get("url") for cr in sent] == urls
        assert all("n" not in query(u) and n not in u for u, n in zip(urls, nonces, strict=True))
        assert [{p.get("name"): p.get("value") for p in cr.iter("Parameter")} for cr in sent] == [
            {"callSid": sid, "n": n} for n in nonces]
        st_ = h.state(sid)
        assert st_.ws_nonce_hash == ws_nonce.nonce_hash(nonces[-1])
        assert st_.relay_minted_at == int(now[0])
        assert all(s.state_existed)
    finally:
        clear_sensitive(sid)


#: Front-door requests in any order; ``incoming`` repeats and requests for unknown calls included.
requests = st.lists(
    st.tuples(st.sampled_from(["incoming", "language", "factor", "status"]),
              st.one_of(call_sids, not_call_sids)),
    min_size=1, max_size=12)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(reqs=requests)
def test_warmer_only_after_create_and_only_for_callsids(reqs: list[tuple[str, str]]) -> None:
    h = build()
    warmer = RecordingWarmer(h)
    s = CheckingSigner(signer(), h)
    agentcore(h, s, warmer)
    created: list[str] = []
    zip_code = TEST_FACTORS["C-1001"]["zip"]
    try:
        for route, sid in reqs:
            existed = h.store.get("calls", {"call_sid": sid}) is not None
            params: dict[str, Any] = {}
            if route == "incoming":
                params = {"From": "+12025550101", "To": "+12025550100", "CallStatus": "ringing"}
            elif route == "factor":
                params = {"Digits": zip_code}
                h.sent_factors.add(zip_code)
            elif route == "status":
                params = {"CallStatus": "completed"}
            resp = h.post(route, sid, **params)
            assert resp["statusCode"] == 200
            if route == "incoming" and not existed:
                created.append(sid)
        assert warmer.calls == [x for x in created if CALL_SID_RE.match(x)]
        assert len(set(warmer.calls)) == len(warmer.calls)
        assert all(warmer.state_existed)
        assert all(s.state_existed)  # URLs only from a loaded call state
        for call_sid, _, _ in s.mints:
            assert call_sid in created
    finally:
        for _, sid in reqs:
            clear_sensitive(sid)


def test_strategies_cover_both_kinds() -> None:
    """The ID strategies mean what their names say."""
    assert CALL_SID_RE.match("CA" + "e" * 30 + "ab")
    assert not CALL_SID_RE.match("CA" + "E" * 32)


# ---------------------------------------------------------------------------- Part 2

NOT_END = ("queued", "initiated", "ringing", "in-progress")
#: Webhooks for a few calls in any order: repeated status callbacks (Twilio retries), several end
#: statuses for one call, end statuses before ``incoming``, unknown and non-CallSid IDs.
#: Three calls, so repeats and interleavings are common; plus IDs that are not CallSids.
few_calls = st.sampled_from(["CA" + "e" * 30 + x for x in ("ab", "cd", "fa")])
later_events = st.lists(
    st.tuples(st.one_of(few_calls, few_calls, few_calls, not_call_sids),
              st.one_of(st.just("incoming"), st.sampled_from(sorted(END_STATUSES)),
                        st.sampled_from(sorted(END_STATUSES)), st.sampled_from(NOT_END))),
    min_size=1, max_size=16)
#: Calls that arrive first (state created), then any webhooks; a call left out of ``started``
#: can still arrive later, after its own end status (out of order).
call_events = st.builds(lambda started, later: [(s, "incoming") for s in sorted(started)] + later,
                        st.sets(few_calls), later_events)
#: What the stopper does: each outcome the real one returns, or breaking its contract by raising.
behaviours = st.sampled_from(["ok", "not_found", "error:ThrottlingException", "raise"])


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(events=call_events, behaviour=behaviours, hosting=st.sampled_from(["agentcore", "fargate"]))
def test_one_stop_attempt_per_call_that_ends(events: list[tuple[str, str]], behaviour: str,
                                                   hosting: str) -> None:
    h = build()
    stopper = RecordingStopper(h, outcome=behaviour,
                               error=RuntimeError("stop failed") if behaviour == "raise" else None)
    if hosting == "agentcore":
        agentcore(h, signer(), stopper=stopper)
    else:
        h.app = FrontDoor(replace(h.app.d, runtime_sessions=stopper), h.app.cfg)
    expected: list[str] = []  # each call in state, at its first end status
    try:
        for sid, event in events:
            exists = h.store.get("calls", {"call_sid": sid}) is not None
            if event == "incoming":
                resp = h.post("incoming", sid, From="+12025550101", To="+12025550100",
                              CallStatus="ringing")
            else:
                resp = h.post("status", sid, CallStatus=event)
                if event in END_STATUSES and exists and CALL_SID_RE.match(sid) and sid not in expected:
                    expected.append(sid)
                assert resp["body"] == EMPTY  # a stop, failed or not, never changes the answer
            assert resp["statusCode"] == 200
        assert stopper.calls == (expected if hosting == "agentcore" else [])
        assert all(stopper.state_existed)  # only for a call in state
    finally:
        for sid, _ in events:
            clear_sensitive(sid)
