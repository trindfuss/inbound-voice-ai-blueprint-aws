"""Envelope rules and JSON round-trip."""

from __future__ import annotations

from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from owlbank.envelope import (
    FOLLOW_UP_TEMPLATES,
    NEXT_ACTIONS,
    STATUSES,
    EnvelopeError,
    ResultEnvelope,
    Spoken,
)

template_keys = st.one_of(st.sampled_from(sorted(FOLLOW_UP_TEMPLATES)),
                          st.from_regex(r"[a-z]{1,8}(\.[a-z_]{1,8}){0,2}", fullmatch=True))
spoken_item = st.builds(Spoken, template_keys, st.text(max_size=40))
json_scalar = st.one_of(st.none(), st.booleans(), st.integers(-10**15, 10**15),
                        st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=20))
json_value = st.recursive(json_scalar, lambda inner: st.one_of(
    st.lists(inner, max_size=4), st.dictionaries(st.text(max_size=8), inner, max_size=4)),
    max_leaves=12)
json_obj = st.dictionaries(st.text(max_size=8), json_value, max_size=5)
opt_text = st.one_of(st.none(), st.text(max_size=30))


@st.composite
def raw_fields(draw: st.DrawFn) -> dict[str, Any]:
    """Arbitrary field values, valid or not."""
    return {
        "status": draw(st.one_of(st.sampled_from(sorted(STATUSES)), st.text(max_size=10))),
        "tool": draw(st.one_of(st.sampled_from(["get_balance", "authenticate"]), st.text(max_size=10))),
        "spoken": draw(st.lists(spoken_item, max_size=4)),
        "data": draw(json_obj),
        "next": draw(st.one_of(st.sampled_from(sorted(NEXT_ACTIONS)), st.text(max_size=10))),
        "forTool": draw(st.one_of(st.none(), st.sampled_from(["make_payment", "get_balance"]))),
        "callerUtterance": draw(opt_text),
        "completed": draw(st.one_of(st.none(), st.booleans())),
    }


@st.composite
def valid_envelopes(draw: st.DrawFn) -> ResultEnvelope:
    f = draw(raw_fields())
    f["status"] = draw(st.sampled_from(sorted(STATUSES)))
    f["tool"] = draw(st.sampled_from(["get_balance", "authenticate", "make_payment"]))
    f["next"] = draw(st.sampled_from(sorted(NEXT_ACTIONS)))
    if f["next"] == "await_caller":
        f["spoken"] = [*f["spoken"],
                       Spoken(draw(st.sampled_from(sorted(FOLLOW_UP_TEMPLATES))), draw(st.text()))]
    if f["next"] == "handle_utterance":
        f["callerUtterance"] = draw(st.text(min_size=1, max_size=30))
    return ResultEnvelope(**f)


@given(env=valid_envelopes())
def test_round_trip(env: ResultEnvelope) -> None:
    assert ResultEnvelope.from_json(env.to_json()) == env
    assert ResultEnvelope.from_dict(env.to_dict()) == env


@given(fields=raw_fields())
def test_every_accepted_envelope_satisfies_rules(fields: dict[str, Any]) -> None:
    try:
        env = ResultEnvelope(**fields)
    except EnvelopeError:
        return
    assert env.status in STATUSES and env.next in NEXT_ACTIONS
    if env.next == "await_caller":
        assert env.spoken and env.spoken[-1].template in FOLLOW_UP_TEMPLATES
    if env.next == "handle_utterance":
        assert env.callerUtterance
    assert ResultEnvelope.from_json(env.to_json()) == env


@given(fields=raw_fields())
def test_rule_violations_rejected(fields: dict[str, Any]) -> None:
    breaks = (
        fields["status"] not in STATUSES
        or fields["next"] not in NEXT_ACTIONS
        or not fields["tool"]
        or (fields["next"] == "await_caller"
            and (not fields["spoken"] or fields["spoken"][-1].template not in FOLLOW_UP_TEMPLATES))
        or (fields["next"] == "handle_utterance" and not fields["callerUtterance"])
    )
    try:
        ResultEnvelope(**fields)
        accepted = True
    except EnvelopeError:
        accepted = False
    assert accepted is not breaks
