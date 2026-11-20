"""Properties of the escalation without a Flex workflow.

- Whatever the model returns (text of any shape, nothing, an exception) and whatever the caller's
  language and the escalation reason, nothing the caller is read holds a digit (or any Unicode
  number), the response ends in a hang-up, and nothing is enqueued or dialled.
- However many times the escalation is requested for one call (and with whatever reason or model
  output each time), the journal holds one ``escalation`` event for the call's ref, the token is
  revoked on every request, and the ref and the time of the first request are kept.
"""

from __future__ import annotations

import unicodedata
import xml.etree.ElementTree as ET
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank import callstate, tokens
from owlbank.briefing import BriefingFacts
from tests.component.test_frontdoor_support import MARIA, THOMAS, build, clear_sensitive, sid

PROPS = settings(max_examples=40, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])

DIGITS = st.sampled_from(["1", "٣", "²", "½", "①", "Ⅻ", "9"])
MODEL_TEXT = st.one_of(
    st.none(),
    st.text(max_size=300),
    st.builds(lambda a, d, b: f"{a} {d} {b}", st.text(max_size=40), DIGITS, st.text(max_size=40)),
    st.lists(st.sampled_from(["asked", "about", "a", "balance", "card", "payment"]),
             min_size=1, max_size=40).map(" ".join),
)


class Scripted:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs, self.n = outputs, 0

    def summarise(self, facts: BriefingFacts, *, deadline: float) -> str | None:
        out = self.outputs[min(self.n, len(self.outputs) - 1)]
        self.n += 1
        if isinstance(out, Exception):
            raise out
        return out


MODEL_OUT = st.one_of(MODEL_TEXT, st.just(RuntimeError("model down")))


def _start(h, call: str, spanish: bool) -> None:
    h.start(call, MARIA if spanish else THOMAS, lang_digit="2" if spanish else "")
    callstate.mutate(h.store, call, lambda s: {"ai_sessions": 1})


def _request(h, call: str, reason: str, ref: str = "jr-7f3a") -> ET.Element:
    h.set_pending_tool(call, "escalate", {"reason": reason, "summary": ""})
    return h.handoff(call, {"reason": "escalate", "escalationReason": reason, "journal": ref})


def _reasons() -> list[str]:
    return list(build(flex=False).app.d.policy.reasons)


REASONS = _reasons()


@PROPS
@given(output=MODEL_OUT, spanish=st.booleans(), reason=st.sampled_from(REASONS))
def test_the_read_out_never_has_a_digit(output, spanish, reason):
    h = build(flex=False, briefing_model=Scripted([output]))
    call = sid(1)
    try:
        _start(h, call, spanish)
        root = _request(h, call, reason)
        said = " ".join(e.text or "" for e in root.iter("Say"))
        assert not any(unicodedata.category(c).startswith("N") for c in said), said
        assert [c.tag for c in root][-1] == "Hangup"
        assert root.find("Enqueue") is None and root.find("Dial") is None
        assert said.strip()
    finally:
        clear_sensitive(call)


@PROPS
@given(requests=st.lists(st.tuples(st.sampled_from(REASONS), MODEL_OUT, st.text(
    alphabet="abcdef0123456789-", min_size=1, max_size=20)), min_size=1, max_size=4))
def test_repeats_journal_once_and_revoke_every_time(requests):
    h = build(flex=False, briefing_model=Scripted([r[1] for r in requests]))
    call = sid(2)
    revoked: list[str] = []
    real = tokens.revoke_call

    def spy(store: Any, call_sid: str) -> Any:
        revoked.append(call_sid)
        return real(store, call_sid)

    import lambdas.frontdoor.app as app_module

    original = app_module.tokens.revoke_call
    app_module.tokens.revoke_call = spy
    try:
        _start(h, call, False)
        first: dict[str, Any] | None = None
        for reason, _output, ref in requests:
            _request(h, call, reason, ref)
            esc = dict(h.state(call).escalation or {})
            first = first or esc
            assert esc["journal_ref"] == first["journal_ref"] and esc["at"] == first["at"]
            assert esc["reason"] == first["reason"]
        events = [e for e in h.journal(call) if e["type"] == "escalation"]
        assert len(events) == 1 and events[0]["journal_ref"] == first["journal_ref"]
        assert len(revoked) == len(requests)
    finally:
        app_module.tokens.revoke_call = original
        clear_sensitive(call)
