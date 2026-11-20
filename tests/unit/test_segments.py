"""Unit tests for owlbank.segments (the call journal and redaction guard)."""

from __future__ import annotations

import pytest

from owlbank import segments as sg
from owlbank.store import ConditionFailed, MemoryStore

SID = "CA" + "a" * 32


@pytest.fixture(autouse=True)
def _clear() -> None:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)


def make_log(store: MemoryStore | None = None) -> sg.SegmentLog:
    return sg.SegmentLog(store or MemoryStore(), now=lambda: 1_000, ttl_s=60)


def test_append_allocates_monotonic_sequence_and_read_orders() -> None:
    log = make_log()
    seqs = [log.append(SID, {"type": "prompt", "prompt_key": f"k{i}"}) for i in range(3)]
    assert seqs == [1, 2, 3]
    events = log.read(SID)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["ts"] == 1_000 and events[0]["expires_at"] == 1_060
    assert [e["seq"] for e in log.read(SID, since_seq=1)] == [2, 3]


def test_two_writers_never_overwrite() -> None:
    store = MemoryStore()
    a, b = make_log(store), make_log(store)
    a.append(SID, {"type": "prompt", "prompt_key": "a1"})
    b.append(SID, {"type": "prompt", "prompt_key": "b1"})  # b has no hint; queries
    a.append(SID, {"type": "prompt", "prompt_key": "a2"})  # a's hint is stale -> conflict, retry
    keys = [e["prompt_key"] for e in a.read(SID)]
    assert keys == ["a1", "b1", "a2"]


def test_retry_gives_up_with_append_conflict() -> None:
    class AlwaysConflict(MemoryStore):
        def put(self, table, item, *, condition=None):  # type: ignore[override]
            raise ConditionFailed("busy")

    log = sg.SegmentLog(AlwaysConflict(), max_retries=3)
    with pytest.raises(sg.AppendConflict):
        log.append(SID, {"type": "call_start"})


@pytest.mark.parametrize("bad", [{"type": "nope"}, {}, {"type": "prompt", "seq": 9},
                                 {"type": "prompt", "call_sid": "x"}])
def test_rejects_unknown_type_and_reserved_fields(bad: dict) -> None:
    with pytest.raises(ValueError):
        make_log().append(SID, bad)


@pytest.mark.parametrize("key", ["digits", "Digits", "SpeechResult", "speech_result", "factor_value",
                                 "ssn", "ssn4", "zip", "zip_code", "otp", "otp_code", "token",
                                 "token_hash"])
def test_deny_list_keys_rejected_at_any_depth(key: str) -> None:
    log = make_log()
    with pytest.raises(sg.RedactionError):
        log.append(SID, {"type": "verification", key: "x"})
    with pytest.raises(sg.RedactionError):
        log.append(SID, {"type": "tool_call", "args": {"nested": [{key: 1}]}})
    assert log.read(SID) == []


def test_registered_value_rejected_in_any_string_field_including_spaced_forms() -> None:
    log = make_log()
    sg.register_sensitive(SID, ["94110", "4821"])
    for text in ["my zip is 94110", "9 4 1 1 0", "nine 94-110", "last four 4821"]:
        with pytest.raises(sg.RedactionError) as err:
            log.append(SID, {"type": "ai_turn", "role": "user", "text": text})
        assert "94110" not in str(err.value) and "4821" not in str(err.value)
    with pytest.raises(sg.RedactionError):
        log.append(SID, {"type": "tool_result", "data": {"n": 94110}})
    assert log.read(SID) == []


def test_clean_events_accepted_and_factor_names_allowed() -> None:
    log = make_log()
    sg.register_sensitive(SID, ["94110"])
    log.append(SID, {"type": "factor_asked", "factor": "zip"})
    log.append(SID, {"type": "verification", "factor": "zip", "outcome": "matched", "level": 1})
    log.append(SID, {"type": "ai_turn", "role": "user", "text": "what's my balance"})
    assert len(log.read(SID)) == 3


def test_registry_is_per_call_and_clearable() -> None:
    other = "CA" + "b" * 32
    sg.register_sensitive(SID, ["123456"])
    assert sg.contains_sensitive(SID, "code 123456")
    assert not sg.contains_sensitive(other, "code 123456")
    sg.clear_sensitive(SID)
    assert not sg.contains_sensitive(SID, "code 123456")


def test_register_refuses_short_or_non_string_values() -> None:
    with pytest.raises(ValueError):
        sg.register_sensitive(SID, ["12"])
    with pytest.raises(TypeError):
        sg.register_sensitive(SID, [1234])  # type: ignore[list-item]


def test_registered_values_never_stored() -> None:
    store = MemoryStore()
    log = make_log(store)
    sg.register_sensitive(SID, ["94110"])
    log.append(SID, {"type": "call_start", "ani": "+15551234567"})
    assert "94110" not in repr(store._tables)


def test_summary_for_resume() -> None:
    log = make_log()
    log.append(SID, {"type": "ai_turn", "role": "user", "text": "pay my card"})
    since = log.append(SID, {"type": "tool_call", "tool": "authenticate", "tool_use_id": "t1"})
    log.append(SID, {"type": "handoff", "reason": "authenticate", "tool": "make_payment"})
    log.append(SID, {"type": "prompt", "prompt_key": "stepup.connecting.make_payment"})
    log.append(SID, {"type": "factor_asked", "factor": "otp"})
    log.append(SID, {"type": "prompt", "prompt_key": "verify.otp"})
    log.append(SID, {"type": "verification", "factor": "otp", "outcome": "failed"})
    log.append(SID, {"type": "verification", "factor": "otp", "outcome": "verified", "level": 3})
    s = log.summary_for_resume(SID, since)
    assert s["promptsPlayed"] == ["stepup.connecting.make_payment", "verify.otp"]
    assert s["factorsAsked"] == ["otp"]
    assert s["attempts"] == 2
    assert s["outcome"] == "verified"
    assert s["levelReached"] == 3
    assert s["toolsCalled"] == [] and s["lastTool"] is None
    assert s["lastSeq"] == 8
    full = log.summary_for_resume(SID)
    assert full["toolsCalled"] == ["authenticate"]


def test_summary_for_empty_range() -> None:
    s = make_log().summary_for_resume(SID, 5)
    assert s["attempts"] == 0 and s["outcome"] is None and s["lastSeq"] == 5


# --------------------------------------------------------------------------- word boundaries, values only

def test_registered_value_matches_on_word_boundaries_only() -> None:
    sg.register_sensitive(SID, ["lang", "1234"])
    assert sg.contains_sensitive(SID, "my lang is set")
    assert sg.contains_sensitive(SID, "LANG.")
    assert sg.contains_sensitive(SID, "it's 1 2-3 4, thanks")
    assert sg.contains_sensitive(SID, "ssn_1234")  # "_" is a boundary
    for text in ["language", "slang", "12345", "01234"]:
        assert not sg.contains_sensitive(SID, text), text
        assert sg.redact_text(SID, text) == text
    for text in ["x1234", "1234th", "acct1234"]:  # digits next to letters match
        assert sg.contains_sensitive(SID, text), text
    assert sg.redact_text(SID, "Lang, language and 1234") == "[redacted], language and [redacted]"


def test_a_value_with_a_non_word_edge_needs_no_boundary_on_that_side() -> None:
    sg.register_sensitive(SID, ["#4821"])
    assert sg.contains_sensitive(SID, "code x#4821")
    assert not sg.contains_sensitive(SID, "code #48210")


def test_keys_are_never_matched_only_values() -> None:
    log = make_log()
    sg.register_sensitive(SID, ["lang"])
    log.append(SID, {"type": "call_start", "language": "en-US", "detail": {"lang_hint": "es"}})
    with pytest.raises(sg.RedactionError) as err:
        log.append(SID, {"type": "ai_turn", "role": "user", "text": "my lang is es"})
    assert "lang" not in str(err.value).replace("$.text", "")


def test_a_registered_lang_does_not_block_a_call_state_write_with_the_key_language() -> None:
    from owlbank import callstate as cs

    store = MemoryStore()
    sg.register_sensitive(SID, ["lang"])
    v1 = cs.create(store, cs.CallState(call_sid=SID, expires_at=10_000, language="es-US"))
    v2 = cs.update(store, v1, now=lambda: 1, language="en-US")
    assert cs.load(store, SID) == v2
    assert "language" in v2.to_item()


@pytest.mark.parametrize("value", ["2025", "1009"])
def test_a_factor_equal_to_part_of_the_escalation_time_does_not_block_call_state_writes(
        value: str) -> None:
    """Regression (stored example): the front door writes ``escalation.at`` as an ISO time;
    an SSN last four equal to its year or month and day matched it, so the escalation write and
    every later call-state write raised RedactionError and the caller was hung up on."""
    from owlbank import callstate as cs

    store = MemoryStore()
    sg.register_sensitive(SID, [value])
    v1 = cs.create(store, cs.CallState(call_sid=SID, expires_at=10_000))
    escalation = {"reason": "complaint", "queue": "Q", "at": "2025-10-09T08:53:20Z",
                  "journal_ref": "ref"}
    v2 = cs.update(store, v1, now=lambda: 1, escalation=escalation)
    v3 = cs.update(store, v2, now=lambda: 2, interrupted_utterance="what's")
    assert cs.load(store, SID) == v3
    # Only the server's time is skipped: the same text under any other key is still refused.
    with pytest.raises(sg.RedactionError):
        sg.assert_clean(SID, {"escalation": {"summary": "2025-10-09T08:53:20Z"}})


EMAIL = "jane.roe@example.org"  # fictional


def test_a_registered_email_is_scrubbed_from_a_log_line() -> None:
    import io
    import json
    import logging

    from owlbank import obs

    sg.register_sensitive(SID, [EMAIL])
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(obs.JsonFormatter())
    handler.addFilter(obs.ScrubFilter())
    log = logging.getLogger("tests.segments.email")
    log.propagate = False
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    try:
        with obs.call_context(SID):
            log.warning("caller said %s", "send it to Jane.Roe@Example.org, please",
                        extra={"note": {"to": [EMAIL]}})
    finally:
        log.handlers.clear()
    line = stream.getvalue()
    assert EMAIL not in line.lower()
    doc = json.loads(line)
    assert "[redacted]" in doc["event"] and doc["note"] == {"to": ["[redacted]"]}


def test_a_registered_email_is_scrubbed_from_a_journal_text() -> None:
    store = MemoryStore()
    log = make_log(store)
    sg.register_sensitive(SID, [EMAIL])
    text = sg.redact_text(SID, f"my email is {EMAIL.upper()}.")
    assert text == "my email is [redacted]."
    log.append(SID, {"type": "ai_turn", "role": "user", "text": text})
    assert EMAIL not in repr(store._tables).lower()
    with pytest.raises(sg.RedactionError):
        log.append(SID, {"type": "ai_turn", "role": "user", "text": f"to {EMAIL}"})


def test_value_helpers_for_an_unregistered_value() -> None:
    # Used by the placeholder (rule 4): same matching, no registry and no minimum length.
    assert sg.value_spans("No, I said nothing; no.", "no") == [(0, 2), (20, 22)]
    assert sg.contains_value("say 1 2-3 4", "1234")
    assert not sg.contains_value("12345", "1234")
    assert sg.value_spans("anything", "") == []


def test_all_digit_values_match_next_to_letters_but_not_inside_longer_numbers():
    """An all-digit registered value matches wherever no other digit
    touches it, so a value glued to letters is still caught; a value with letters stays
    whole-word."""
    from owlbank import segments
    assert segments.contains_value("acct1234 ok", "1234")
    assert segments.contains_value("ssn-1234", "1234")
    assert segments.contains_value("my last four are 1 2 3 4.", "1234")
    assert not segments.contains_value("12345", "1234")
    assert not segments.contains_value("01234", "1234")
    assert not segments.contains_value("language", "lang")
    assert segments.contains_value("lang is set", "lang")
