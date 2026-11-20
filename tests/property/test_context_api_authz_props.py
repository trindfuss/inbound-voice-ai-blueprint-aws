"""Human-leg properties: the Context API's per-task check and the timeline allow-list.

* Over arbitrary reservations x roles x task attributes,
  ``authorize_task`` allows exactly when the task's ``call_sid`` and ``journal_ref`` equal the
  request's AND (the worker holds a pending, accepted or wrapping reservation on the task OR has
  the ``supervisor`` or ``admin`` role). A reader that fails never yields an allow.
* Over arbitrary journal events, every timeline entry's keys
  are a subset of ``TIMELINE_FIELDS`` (plus ``rendered_text`` on prompts) and no transcript-like text
  planted in a non-allow-listed field appears in the output.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.context_api.handler import (
    HELD_RESERVATION_STATUSES,
    PRIVILEGED_ROLES,
    TIMELINE_FIELDS,
    authorize_task,
    build_timeline,
)
from owlbank.taskrouter import TaskRouterUnavailable

SID = "CA" + "a" * 32
OTHER_SID = "CA" + "b" * 32
REF = "jr_props"
TASK = "WT" + "1" * 32
ME = "WK" + "2" * 32
OTHERS = ["WK" + "3" * 32, "WK" + "4" * 32]

STATUSES = ["pending", "accepted", "wrapping", "rejected", "timeout", "canceled", "rescinded",
            "completed", None]
RESERVATIONS = st.lists(st.fixed_dictionaries({
    "worker_sid": st.sampled_from([ME, *OTHERS]), "status": st.sampled_from(STATUSES)}), max_size=6)
ROLES = st.lists(st.sampled_from(["agent", "supervisor", "admin", "Supervisor", "supervisors", ""]),
                 max_size=4)
ATTR_VALUE = st.sampled_from([SID, OTHER_SID, REF, "jr_other", "", None, 5, ["x"]])
ATTRS = st.one_of(
    st.none(),
    st.fixed_dictionaries({}, optional={"call_sid": st.sampled_from([SID, OTHER_SID, "", None, 5]),
                                        "journal_ref": st.sampled_from([REF, "jr_other", "", None, 5]),
                                        "extra": ATTR_VALUE}))


class Reader:
    def __init__(self, attrs, reservations) -> None:
        self.attrs, self.res = attrs, reservations

    def task_attributes(self, task_sid: str, *, timeout_s=None):
        return self.attrs

    def reservations(self, task_sid: str, *, timeout_s=None):
        return self.res


def rule(attrs, reservations, roles) -> bool:
    """The design's rule, written independently of the implementation."""
    if attrs is None or attrs.get("call_sid") != SID or attrs.get("journal_ref") != REF:
        return False
    if set(roles) & {"supervisor", "admin"}:
        return True
    return any(r["worker_sid"] == ME and r["status"] in ("pending", "accepted", "wrapping")
               for r in reservations)


def test_rule_constants_match_the_design() -> None:
    assert {"supervisor", "admin"} == PRIVILEGED_ROLES
    assert {"pending", "accepted", "wrapping"} == HELD_RESERVATION_STATUSES


@settings(max_examples=500)
@given(ATTRS, RESERVATIONS, ROLES)
def test_access_exactly_when_the_rule_holds(attrs, reservations, roles) -> None:
    got = authorize_task(ME, roles, TASK, SID, REF, taskrouter=Reader(attrs, reservations))
    assert got is rule(attrs, reservations, roles)


@settings(max_examples=200)
@given(st.sampled_from([RuntimeError("x"), TimeoutError(), TaskRouterUnavailable("x"), KeyError("k")]),
       st.sampled_from(["task", "reservations"]), ROLES)
def test_a_failing_reader_never_allows(exc, where, roles) -> None:
    class Failing(Reader):
        def task_attributes(self, task_sid, *, timeout_s=None):
            if where == "task":
                raise exc
            return {"call_sid": SID, "journal_ref": REF}

        def reservations(self, task_sid, *, timeout_s=None):
            raise exc

    if where == "reservations" and set(roles) & {"supervisor", "admin"}:
        # privileged roles never need the reservation read
        assert authorize_task(ME, roles, TASK, SID, REF, taskrouter=Failing(None, [])) is True
        return
    with pytest.raises(TaskRouterUnavailable):
        authorize_task(ME, roles, TASK, SID, REF, taskrouter=Failing(None, []))



TRANSCRIPT = "TRANSCRIPT-LINE caller says my card number is stolen"
KEYS = st.sampled_from(["transcript", "utterance", "text", "body", "message", "content", "speech",
                        "caller_text", "args", "data", "summary", "seq", "ts", "type", "outcome",
                        "tool", "factor", "level", "reason", "queue", "answer", "prompt_key",
                        "rendered_text", "token"])
TYPES = st.sampled_from(["call_start", "prompt", "escalation", "handoff", "tool_call", "ai_turn",
                         "human_transcript", "transcript", "conversation_message", "weird", 7, None])
VALUES = st.one_of(st.just(TRANSCRIPT), st.text(max_size=12), st.integers(), st.booleans(),
                   st.lists(st.just(TRANSCRIPT), max_size=2),
                   st.dictionaries(st.text(max_size=4), st.just(TRANSCRIPT), max_size=2))
EVENT = st.builds(lambda t, d: {**d, "type": t}, TYPES, st.dictionaries(KEYS, VALUES, max_size=10))


@settings(max_examples=400)
@given(st.lists(EVENT, max_size=10))
def test_output_keys_are_allow_listed_and_no_transcript_text(events: list[dict[str, Any]]) -> None:
    out = build_timeline(events)
    for entry in out:
        extra = set(entry) - set(TIMELINE_FIELDS)
        assert extra <= {"rendered_text"}
        assert "rendered_text" not in entry or entry["type"] == "prompt"
    # The transcript marker can only come out through a field the allow-list copies as a scalar;
    # none of the transcript-like keys is one, so it must reach the output only via those.
    allowed_sources = set(TIMELINE_FIELDS) | {"rendered_text"}
    planted_in_listed = any(
        isinstance(e.get(k), str) and TRANSCRIPT in e[k]
        for e in events for k in allowed_sources if k in e)
    if not planted_in_listed:
        assert TRANSCRIPT not in json.dumps(out)
