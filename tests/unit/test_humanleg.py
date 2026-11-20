"""Unit tests for owlbank.humanleg."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from owlbank import humanleg


def test_fallback_start_params_are_named_both_tracks_and_labelled() -> None:
    """``human_leg.agent_legs`` off (and the default): a single both-tracks transcription."""
    for params in (humanleg.transcription_start("CH123", enabled=True, language=None, status_url=None),
                   humanleg.transcription_start("CH123", enabled=True, language=None, status_url=None,
                                                agent_legs=False)):
        assert params == {"Name": "humanleg", "Track": "both_tracks", "InboundTrackLabel": "customer",
                          "OutboundTrackLabel": "agent", "ConversationId": "CH123"}


def test_customer_leg_params_with_agent_legs_are_the_inbound_track_only() -> None:
    """The customer's voice only; the agent's comes from the agent leg."""
    params = humanleg.transcription_start("CH123", enabled=True, language="en-US", status_url="https://x/t",
                                          engine="deepgram", speech_model="nova-3", agent_legs=True)
    assert params == {"Name": "humanleg", "Track": "inbound_track", "InboundTrackLabel": "customer",
                      "TranscriptionEngine": "deepgram", "LanguageCode": "en-US", "SpeechModel": "nova-3",
                      "StatusCallbackUrl": "https://x/t", "ConversationId": "CH123"}


CUSTOMER = "CA" + "c" * 32
AGENT1, AGENT2 = ("CA" + c * 32 for c in "12")


def test_agent_leg_params_map_the_callback_back_to_the_customer_call() -> None:
    """Same name, conversation, engine, language and model; label agent; ``?call=``."""
    params = humanleg.agent_transcription_start(
        "CH123", enabled=True, customer_call_sid=CUSTOMER, language="en-US",
        status_url="https://u:p@x/voice/transcription", engine="deepgram", speech_model="nova-3")
    assert params == {"Name": "humanleg", "Track": "inbound_track", "InboundTrackLabel": "agent",
                      "TranscriptionEngine": "deepgram", "LanguageCode": "en-US", "SpeechModel": "nova-3",
                      "StatusCallbackUrl": f"https://u:p@x/voice/transcription?call={CUSTOMER}",
                      "ConversationId": "CH123"}
    with_query = humanleg.agent_transcription_start("CH1", enabled=True, customer_call_sid=CUSTOMER,
                                                    language=None, status_url="https://x/t?a=1")
    assert with_query is not None and with_query["StatusCallbackUrl"] == f"https://x/t?a=1&call={CUSTOMER}"


@pytest.mark.parametrize(("conversation_id", "enabled", "customer"), [
    ("CH1", False, CUSTOMER), (None, True, CUSTOMER), ("", True, CUSTOMER), ("CH1", True, None),
    ("CH1", True, "CAnope"), ("CH1", True, "WK" + "c" * 32)])
def test_agent_leg_params_none(conversation_id, enabled, customer) -> None:
    assert humanleg.agent_transcription_start(conversation_id, enabled=enabled, customer_call_sid=customer,
                                              language="en-US", status_url="https://x") is None


def test_agent_legs_from_conference_participants() -> None:
    attrs = {"conference": {"participants": {"customer": CUSTOMER, "worker": AGENT1}}, "call_sid": CUSTOMER}
    assert humanleg.agent_legs(attrs, CUSTOMER) == [AGENT1]
    import json
    assert humanleg.agent_legs(json.dumps(attrs), CUSTOMER) == [AGENT1]
    transfer = {"conference": {"participants": {"customer": CUSTOMER, "worker": AGENT1, "transfer": AGENT2,
                                                "again": AGENT1}}}
    assert humanleg.agent_legs(transfer, CUSTOMER) == [AGENT1, AGENT2]  # Flex's order, each once
    assert humanleg.agent_legs(transfer, None) == [CUSTOMER, AGENT1, AGENT2]


@pytest.mark.parametrize("attrs", [
    None, "", "not json", "[1]", 5, [], {}, {"conference": None}, {"conference": {"participants": []}},
    {"conference": {"participants": {"customer": CUSTOMER}}},
    {"conference": {"participants": {"worker": "CA123", "w2": 7, "w3": None, "w4": {"sid": AGENT1}}}},
    {"conference": {"participants": {"worker": "WK" + "1" * 32}}},
    '{"conference": {"participants": {"worker": "' + "CA" + "1" * 32 + '"}}' + " " * 70_000])
def test_agent_legs_malformed_or_own_sid_is_empty(attrs) -> None:
    assert humanleg.agent_legs(attrs, CUSTOMER) == []


def test_agent_legs_cap() -> None:
    legs = ["CA" + f"{i:032x}" for i in range(1, 10)]
    attrs = {"conference": {"participants": {f"p{i}": leg for i, leg in enumerate(legs)}}}
    assert humanleg.agent_legs(attrs, CUSTOMER) == legs[:humanleg.MAX_AGENT_LEGS] and humanleg.MAX_AGENT_LEGS == 6


def _legs_attrs(*legs: str) -> dict:
    return {"conference": {"participants": {f"w{i}": leg for i, leg in enumerate(legs)}}}


def _leg(role: str, ended_at: int | None = None) -> dict:
    return {"name": "humanleg", "role": role, "started_at": 1, "ended_at": ended_at}


HANDED_OFF = {"reason": "x", "handoff_at": 1, "conversation_id": "conv"}


@pytest.mark.parametrize(("escalation", "reason"), [
    (None, "not_escalated"), ({}, "not_escalated"), ({"reason": "x"}, "before_handoff"),
    ({**HANDED_OFF, "call_ended_at": 2}, "call_ended"),
    ({**HANDED_OFF, "conversation_closed_at": 2}, "after_close")])
def test_legs_to_start_nothing(escalation, reason) -> None:
    todo = humanleg.legs_to_start(escalation, _legs_attrs(AGENT1), CUSTOMER, with_agents=True)
    assert not todo and todo.reason == reason


def test_legs_to_start_new_legs_only_and_the_switch() -> None:
    todo = humanleg.legs_to_start(HANDED_OFF, _legs_attrs(AGENT1), CUSTOMER, with_agents=True)
    assert (todo.customer, todo.agents, todo.reason) == (True, (AGENT1,), "start")
    off = humanleg.legs_to_start(HANDED_OFF, _legs_attrs(AGENT1), CUSTOMER, with_agents=False)
    assert (off.customer, off.agents) == (True, ())
    recorded = {**HANDED_OFF, "transcriptions": {CUSTOMER: _leg("customer"), AGENT1: _leg("agent", 5)}}
    todo = humanleg.legs_to_start(recorded, _legs_attrs(AGENT1, AGENT2), CUSTOMER, with_agents=True)
    assert (todo.customer, todo.agents) == (False, (AGENT2,))  # an ended leg is not started again
    none_new = humanleg.legs_to_start(recorded, _legs_attrs(AGENT1), CUSTOMER, with_agents=True)
    assert not none_new and none_new.reason == "nothing_new"


def test_legs_to_start_keeps_to_the_cap() -> None:
    legs = ["CA" + f"{i:032x}" for i in range(1, 10)]
    recorded = {**HANDED_OFF, "transcriptions": {leg: _leg("agent") for leg in legs[:4]}}
    todo = humanleg.legs_to_start(recorded, _legs_attrs(*legs[3:]), CUSTOMER, with_agents=True)
    assert todo.agents == tuple(legs[4:6])


def test_start_params_carry_the_callers_settings() -> None:
    params = humanleg.transcription_start(
        "CH1", enabled=True, language="en-US", status_url="https://x/voice/transcription",
        engine="deepgram", speech_model="nova-3")
    assert params is not None
    assert (params["TranscriptionEngine"], params["LanguageCode"], params["SpeechModel"],
            params["StatusCallbackUrl"]) == ("deepgram", "en-US", "nova-3", "https://x/voice/transcription")


def test_start_params_leave_out_empty_values() -> None:
    params = humanleg.transcription_start("CH1", enabled=True, language="", status_url="", engine="",
                                          speech_model=None)
    assert params is not None
    assert not {"LanguageCode", "StatusCallbackUrl", "TranscriptionEngine", "SpeechModel"} & set(params)


def test_start_name_is_the_one_the_close_stops() -> None:
    params = humanleg.transcription_start("CH1", enabled=True, language="en-US", status_url="https://x")
    assert params is not None and params["Name"] == humanleg.TRANSCRIPTION_NAME


class _Policy:
    def __init__(self, value: object) -> None:
        self.human_leg_transcription = value


@pytest.mark.parametrize(("value", "expected"), [(True, True), (False, False), ("true", False), (1, False), (None, False)])
def test_transcription_enabled(value: object, expected: bool) -> None:
    assert humanleg.transcription_enabled(_Policy(value)) is expected  # type: ignore[arg-type]


def test_transcription_enabled_reads_off_when_the_policy_raises() -> None:
    class Broken:
        @property
        def human_leg_transcription(self) -> bool:
            raise RuntimeError("boom")

    assert humanleg.transcription_enabled(Broken()) is False


@pytest.mark.parametrize(("provider", "engine"), [("Deepgram", "deepgram"), ("deepgram", "deepgram"),
                                                  (" Google ", "google"), ("ElevenLabs", None), (None, None), (5, None)])
def test_stt_settings_maps_the_catalog_provider(provider: object, engine: str | None) -> None:
    got = humanleg.stt_settings({"transcriptionProvider": provider, "speechModel": "m", "transcriptionLanguage": "es"})
    assert got.engine == engine
    assert got.speech_model == ("m" if engine else None)
    assert got.language_code == "es"


def test_stt_settings_tolerates_a_missing_section() -> None:
    assert humanleg.stt_settings(None) == humanleg.SttSettings(None, None, None)


def test_stt_settings_match_the_shipped_catalog() -> None:
    import json
    from pathlib import Path

    cat = json.loads((Path(__file__).resolve().parents[2] / "data" / "prompt_catalog.json").read_text())
    for lang, body in cat["languages"].items():
        got = humanleg.stt_settings(body["conversationRelay"])
        assert got.engine in {"deepgram", "google"}, lang
        assert got.speech_model and got.language_code, lang


@pytest.mark.parametrize("conversation_id", ["CH123", None, ""])
def test_none_when_disabled(conversation_id: str | None) -> None:
    assert humanleg.transcription_start(conversation_id, enabled=False, language="en-US",
                                        status_url="https://x") is None


@pytest.mark.parametrize("conversation_id", [None, ""])
def test_none_when_id_missing_or_empty(conversation_id: str | None) -> None:
    assert humanleg.transcription_start(conversation_id, enabled=True, language="en-US",
                                        status_url="https://x") is None


# --------------------------------------------------------------------------- close

NOW = 1_760_000_000
W1, W2, W3 = ("WK" + c * 32 for c in "abc")
ESC = {"reason": "fraud_suspected", "queue": "q", "at": "2025-10-09T08:53:20Z", "journal_ref": "jr"}


@dataclass(frozen=True)
class Res:
    worker_sid: str
    status: str


def leg(role: str = "customer", ended_at: int | None = None) -> dict:
    return {"name": "humanleg", "role": role, "started_at": NOW - 60, "ended_at": ended_at}


def running(**agents: int | None) -> dict:
    """The customer leg running, plus agent legs ``CA..`` (key) with their ``ended_at`` (value)."""
    return {**ESC, "transcriptions": {CUSTOMER: leg(), **{a: leg("agent", e) for a, e in agents.items()}}}


@pytest.mark.parametrize("event", sorted(humanleg.END_EVENTS))
def test_a_call_that_never_escalated_is_left_alone(event) -> None:
    for esc in (None, {}):
        d = humanleg.end_decision(esc, event, NOW)
        assert (d.escalation, d.close, d.reason) == (None, False, "not_escalated")


def test_status_closes_when_no_transcription_runs() -> None:
    d = humanleg.end_decision(ESC, humanleg.CALL_ENDED, NOW)
    assert d.close and d.escalation["call_ended_at"] == NOW and d.escalation["close_claimed_at"] == NOW


def test_status_waits_for_the_transcript_when_one_runs() -> None:
    d = humanleg.end_decision(running(), humanleg.CALL_ENDED, NOW)
    assert not d.close and d.reason == "waiting_for_transcript"
    assert d.escalation["call_ended_at"] == NOW and "close_claimed_at" not in d.escalation


def test_customer_transcription_stopped_closes_and_records_the_call_end() -> None:
    for which in (None, CUSTOMER):
        d = humanleg.end_decision(running(), humanleg.TRANSCRIPTION_STOPPED, NOW, leg=which)
        assert d.close and d.escalation["transcriptions"][CUSTOMER]["ended_at"] == NOW
        assert d.escalation["call_ended_at"] == NOW


def test_an_agent_leg_ending_never_closes_alone() -> None:
    """A transfer: the first agent's leg stops while the call goes on."""
    esc = running(**{AGENT1: None})
    for event in (humanleg.TRANSCRIPTION_STOPPED, humanleg.TRANSCRIPTION_ERROR):
        d = humanleg.end_decision(esc, event, NOW, leg=AGENT1)
        assert not d.close and d.reason == "waiting_for_call_end"
        assert d.escalation["transcriptions"][AGENT1]["ended_at"] == NOW
        assert d.escalation["transcriptions"][CUSTOMER]["ended_at"] is None
        assert "call_ended_at" not in d.escalation


def test_close_waits_for_every_leg_after_the_call_ended() -> None:
    esc = running(**{AGENT1: NOW - 30, AGENT2: None})
    d = humanleg.end_decision(esc, humanleg.TRANSCRIPTION_STOPPED, NOW, leg=CUSTOMER)
    assert not d.close and d.reason == "waiting_for_transcript" and d.escalation["call_ended_at"] == NOW
    last = humanleg.end_decision(d.escalation, humanleg.TRANSCRIPTION_STOPPED, NOW + 1, leg=AGENT2)
    assert last.close and last.escalation["close_claimed_at"] == NOW + 1
    status = humanleg.end_decision(running(**{AGENT1: None}), humanleg.CALL_ENDED, NOW)
    assert not status.close and status.reason == "waiting_for_transcript"


def test_an_unrecorded_leg_is_ignored() -> None:
    d = humanleg.end_decision(running(), humanleg.TRANSCRIPTION_STOPPED, NOW, leg=AGENT2)
    assert (d.escalation, d.close, d.reason) == (None, False, "unknown_leg")
    d = humanleg.end_decision(ESC, humanleg.TRANSCRIPTION_STOPPED, NOW)  # no customer leg recorded
    assert (d.escalation, d.close, d.reason) == (None, False, "unknown_leg")


def test_transcription_error_closes_only_after_the_call_ended() -> None:
    early = humanleg.end_decision(running(), humanleg.TRANSCRIPTION_ERROR, NOW)
    assert not early.close and early.reason == "waiting_for_call_end"
    assert early.escalation["transcriptions"][CUSTOMER]["ended_at"] == NOW
    assert "call_ended_at" not in early.escalation
    later = humanleg.end_decision({**running(), "call_ended_at": NOW - 5}, humanleg.TRANSCRIPTION_ERROR, NOW)
    assert later.close
    # The transcript ended early: the status callback then closes, as with the switch off.
    assert humanleg.end_decision(early.escalation, humanleg.CALL_ENDED, NOW + 1).close


def test_nothing_after_close() -> None:
    closed = {**ESC, "conversation_closed_at": NOW}
    for event in humanleg.END_EVENTS:
        d = humanleg.end_decision(closed, event, NOW + 1)
        assert (d.escalation, d.close, d.reason) == (None, False, "after_close")
    assert humanleg.close_plan(SimpleNamespace(escalation=closed), [Res(W1, "accepted")]) == \
        humanleg.NOTHING_TO_CLOSE


def test_a_fresh_claim_blocks_a_second_close_and_a_stale_one_does_not() -> None:
    claimed = {**ESC, "close_claimed_at": NOW}
    assert humanleg.end_decision(claimed, humanleg.CALL_ENDED, NOW + 1).reason == "claimed_elsewhere"
    stale = humanleg.end_decision(claimed, humanleg.CALL_ENDED, NOW + humanleg.CLOSE_CLAIM_TTL_S)
    assert stale.close and stale.escalation["close_claimed_at"] == NOW + humanleg.CLOSE_CLAIM_TTL_S


def test_a_repeated_signal_writes_nothing_new() -> None:
    d = humanleg.end_decision({**running(), "call_ended_at": NOW}, humanleg.CALL_ENDED, NOW + 9)
    assert d.escalation is None and not d.close  # call_ended_at kept, nothing to write


def test_unknown_event_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        humanleg.end_decision(ESC, "hangup", NOW)


def test_close_plan_stops_every_running_leg_only() -> None:
    plan = humanleg.close_plan(SimpleNamespace(escalation=running(**{AGENT1: NOW, AGENT2: None})), [])
    assert plan.stop_transcriptions == ((CUSTOMER, "humanleg"), (AGENT2, "humanleg"))
    ended = humanleg.end_decision(running(), humanleg.TRANSCRIPTION_ERROR, NOW).escalation
    assert humanleg.close_plan(SimpleNamespace(escalation=ended), []).stop_transcriptions == ()
    assert humanleg.close_plan(SimpleNamespace(escalation=ESC), []) == humanleg.ClosePlan((), (), True)


def test_close_plan_repairs_joined_workers_not_recorded_once_each() -> None:
    esc = {**ESC, "participants": [{"worker_sid": W1, "added_at": NOW}]}
    plan = humanleg.close_plan(SimpleNamespace(escalation=esc), [
        Res(W1, "accepted"), Res(W2, "rejected"), Res(W3, "completed"), Res(W3, "wrapping"),
        Res(W2, "accepted")])
    assert plan.add_workers == (W3, W2) and plan.close


@pytest.mark.parametrize("raw,expected", [
    ("transcription-stopped", humanleg.TRANSCRIPTION_STOPPED),
    ("Transcription-Error", humanleg.TRANSCRIPTION_ERROR),
    ("transcription-content", None), ("transcription-started", None), ("", None), (None, None)])
def test_transcription_end_event(raw, expected) -> None:
    assert humanleg.transcription_end_event(raw) == expected


def test_conversation_closed_reads_state_or_map() -> None:
    assert humanleg.conversation_closed(SimpleNamespace(escalation={"conversation_closed_at": NOW}))
    assert humanleg.conversation_closed({"conversation_closed_at": NOW})
    assert not humanleg.conversation_closed(SimpleNamespace(escalation=None))
    assert not humanleg.conversation_closed(None)


# --------------------------------------------------------------------------- accepts

import json  # noqa: E402

W1, W2 = "WK" + "a" * 32, "WK" + "b" * 32
TASK = "WT" + "d" * 32
CONV = "conv_" + "a" * 26


def _accepted(worker: str = W1, event_sid: str | None = "EA" + "1" * 32, **extra: str) -> dict[str, str]:
    event = {"EventType": "reservation.accepted", "WorkerSid": worker, "WorkerName": "Alice Agent",
             "TaskSid": TASK, "ReservationSid": "WR" + "f" * 32,
             "TaskAttributes": json.dumps({"call_sid": "CA" + "e" * 32, "conversation_id": CONV}), **extra}
    if event_sid is not None:
        event["EventSid"] = event_sid
    return event


ESCALATED = SimpleNamespace(escalation={"reason": "fraud_suspected", "queue": "q", "journal_ref": "jr"})


def test_first_accept_is_first_and_adds_the_worker() -> None:
    action = humanleg.worker_event_to_action(_accepted(), ESCALATED)
    assert action == humanleg.ParticipantAction(
        event_key="EA" + "1" * 32, worker_sid=W1, identity="Alice Agent", task_sid=TASK,
        conversation_id=CONV, first=True, add=True)


@pytest.mark.parametrize("event, state", [
    (_accepted(EventType="reservation.created"), ESCALATED),
    (_accepted(worker="not-a-worker"), ESCALATED),
    (_accepted(), SimpleNamespace(escalation=None)),
    (_accepted(), SimpleNamespace(escalation={})),
    (_accepted(), None),
    (_accepted(), SimpleNamespace(escalation={**ESCALATED.escalation, "conversation_closed_at": 1})),
    (_accepted(), SimpleNamespace(escalation={**ESCALATED.escalation,
                                              "taskrouter_events": ["EA" + "1" * 32]})),
])
def test_nothing_to_do(event, state) -> None:
    assert humanleg.worker_event_to_action(event, state) is None


def test_event_key_falls_back_to_sid_then_the_reservation() -> None:
    assert humanleg.accept_event_key({"EventSid": "EA1", "Sid": "EA2"}) == "EA1"
    assert humanleg.accept_event_key({"Sid": "EA2"}) == "EA2"
    assert humanleg.accept_event_key({"ReservationSid": "WR9"}) == "reservation.accepted:WR9"
    assert humanleg.accept_event_key({"EventSid": "bad key with spaces"}) is None


def test_a_later_accept_is_not_first_and_a_listed_worker_is_not_added() -> None:
    state = SimpleNamespace(escalation={**ESCALATED.escalation, "handoff_at": 5, "conversation_id": "conv_x",
                                        "participants": [{"worker_sid": W1, "added_at": 5}]})
    same = humanleg.worker_event_to_action(_accepted(event_sid="EA2"), state)
    other = humanleg.worker_event_to_action(_accepted(W2, event_sid="EA3"), state)
    assert same is not None and (same.first, same.add, same.conversation_id) == (False, False, "conv_x")
    assert other is not None and (other.first, other.add) == (False, True)


def test_identity_falls_back_to_the_worker_sid() -> None:
    action = humanleg.worker_event_to_action(_accepted(WorkerName="  "), ESCALATED)
    assert action is not None and action.identity == W1


def test_claim_records_the_event_task_conversation_and_handoff_once() -> None:
    action = humanleg.worker_event_to_action(_accepted(), ESCALATED)
    assert action is not None
    c = humanleg.claim_accept(ESCALATED.escalation, action, 10)
    assert c.action is not None and (c.action.first, c.action.add, c.reason) == (True, True, "claimed")
    esc = c.escalation
    assert esc is not None
    assert (esc["handoff_at"], esc["task_sid"], esc["conversation_id"]) == (10, TASK, CONV)
    assert esc["taskrouter_events"] == ["EA" + "1" * 32] and esc["participant_claims"] == {W1: 10}
    assert humanleg.claim_accept(esc, action, 11) == humanleg.AcceptClaim(None, None, "duplicate")
    # A concurrent delivery of another event for the same worker: neither first nor an add.
    other = humanleg.worker_event_to_action(_accepted(event_sid="EA2"), ESCALATED)
    assert other is not None
    c2 = humanleg.claim_accept(esc, other, 11)
    assert c2.action is not None and (c2.action.first, c2.action.add, c2.reason) == (False, False, "nothing_to_do")
    # A claim older than its lifetime belongs to a dead delivery: the add may be claimed again.
    c3 = humanleg.claim_accept(esc, other, 10 + humanleg.PARTICIPANT_CLAIM_TTL_S)
    assert c3.action is not None and c3.action.add is True


def test_claim_after_close_or_without_escalation_writes_nothing() -> None:
    action = humanleg.worker_event_to_action(_accepted(), ESCALATED)
    assert action is not None
    closed = {**ESCALATED.escalation, "conversation_closed_at": 3}
    assert humanleg.claim_accept(closed, action, 4) == humanleg.AcceptClaim(None, None, "after_close")
    assert humanleg.claim_accept({}, action, 4).reason == "not_escalated"


def test_claim_keeps_the_first_task_and_conversation() -> None:
    action = humanleg.worker_event_to_action(_accepted(), ESCALATED)
    assert action is not None
    esc = {**ESCALATED.escalation, "task_sid": "WT" + "0" * 32, "conversation_id": "conv_first"}
    c = humanleg.claim_accept(esc, action, 4)
    assert c.escalation is not None and c.escalation["task_sid"] == "WT" + "0" * 32
    assert c.action is not None and c.action.conversation_id == "conv_first"


def test_record_lists_an_added_worker_once_and_releases_the_claim() -> None:
    esc = {**ESCALATED.escalation, "participant_claims": {W1: 1, W2: 1}}
    new = humanleg.record_accept(esc, W1, added=True, conversation_id=CONV, now=7)
    assert new is not None
    assert new["participants"] == [{"worker_sid": W1, "added_at": 7}]
    assert new["participant_claims"] == {W2: 1} and new["conversation_id"] == CONV
    again = humanleg.record_accept(new, W1, added=True, conversation_id=CONV, now=8)
    assert again is None  # listed already, no claim left: nothing to write
    failed = humanleg.record_accept(new, W2, added=False, conversation_id=None, now=8)
    assert failed is not None and failed["participant_claims"] == {} and len(failed["participants"]) == 1


def test_record_after_close_writes_nothing() -> None:
    esc = {**ESCALATED.escalation, "conversation_closed_at": 1, "participant_claims": {W1: 1}}
    assert humanleg.record_accept(esc, W1, added=True, conversation_id=CONV, now=2) is None


def claim(esc, sid: str, role: str = "customer", now: int = 5):
    return humanleg.claim_transcription(esc, sid, name="humanleg", role=role, now=now)


def test_transcription_claim_and_release() -> None:
    esc = dict(ESCALATED.escalation)
    claimed = claim(esc, CUSTOMER)
    assert claimed is not None
    assert claimed["transcriptions"] == {CUSTOMER: {"name": "humanleg", "role": "customer", "started_at": 5,
                                                     "ended_at": None}}
    assert claim(claimed, CUSTOMER) is None  # one start per leg
    assert claim(claimed, AGENT1) is None  # one customer leg per call
    assert claim({**esc, "call_ended_at": 1}, CUSTOMER) is None
    assert claim({**esc, "conversation_closed_at": 1}, CUSTOMER) is None
    assert claim({}, CUSTOMER) is None and claim(None, CUSTOMER) is None
    assert claim(esc, "CA123") is None and claim(esc, CUSTOMER, role="boss") is None
    assert humanleg.claim_transcription(esc, CUSTOMER, name="", role="customer", now=1) is None
    released = humanleg.release_transcription(claimed, CUSTOMER)
    assert released == esc
    assert humanleg.release_transcription(esc, CUSTOMER) is None
    assert humanleg.release_transcription({**claimed, "conversation_closed_at": 1}, CUSTOMER) is None


def test_agent_leg_claims_per_leg_up_to_the_cap() -> None:
    esc = claim(dict(ESCALATED.escalation), CUSTOMER)
    legs = ["CA" + f"{i:032x}" for i in range(1, 9)]
    for i, sid in enumerate(legs):
        nxt = claim(esc, sid, role="agent", now=10 + i)
        if i < humanleg.MAX_AGENT_LEGS:
            assert nxt is not None and nxt["transcriptions"][sid]["role"] == "agent"
            assert claim(nxt, sid, role="agent") is None  # already claimed
            esc = nxt
        else:
            assert nxt is None  # cap reached
    assert len(humanleg.transcriptions(esc)) == 1 + humanleg.MAX_AGENT_LEGS
    # Releasing one agent leg after its failed start leaves the others, and frees a slot.
    released = humanleg.release_transcription(esc, legs[0])
    assert released is not None and legs[0] not in released["transcriptions"]
    assert claim(released, legs[7], role="agent") is not None


def test_end_unstarted_transcription_stops_the_close_waiting() -> None:
    """The fallback when the release cannot be written: the end is recorded instead."""
    esc = dict(ESCALATED.escalation)
    claimed = claim(esc, CUSTOMER)
    assert claimed is not None and humanleg.transcription_running(claimed)
    ended = humanleg.end_unstarted_transcription(claimed, CUSTOMER, 7)
    assert ended is not None and ended["transcriptions"][CUSTOMER]["ended_at"] == 7
    assert not humanleg.transcription_running(ended)
    decision = humanleg.end_decision({**ended, "call_ended_at": 7}, humanleg.CALL_ENDED, 8)
    assert decision.close  # nothing to wait for
    assert humanleg.end_unstarted_transcription(ended, CUSTOMER, 9) is None
    assert humanleg.end_unstarted_transcription(esc, CUSTOMER, 9) is None
    assert humanleg.end_unstarted_transcription({**claimed, "conversation_closed_at": 1}, CUSTOMER, 9) is None


def test_handoff_metadata() -> None:
    assert humanleg.handoff_metadata(1760000000) == {"handoff_at": "1760000000", "handoff_to": "human"}
