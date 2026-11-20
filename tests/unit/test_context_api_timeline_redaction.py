"""Model-assisted slots: the screen-pop
timeline's prompt texts are redacted at read time with one batched ``PiiGuard.text`` call.

``FakePiiRedactor`` (offline) stands in for Comprehend. Its built-in shapes don't include a NAME, so
:class:`NameRedactor` adds one fixed name, still through the fake's masking code.
"""

from __future__ import annotations

import json
import logging

import pytest

from lambdas.context_api import handler as ctx
from lambdas.context_api.handler import (
    ALLOWED_ORIGIN,
    build_timeline,
    get_context,
    join_texts,
    redact_texts,
    redaction_guard,
    split_texts,
)
from owlbank import callstate
from owlbank.callstate import CallState
from owlbank.pii import WITHHELD_TURN, Entity, FakePiiRedactor, PiiGuard, RedactionConfig
from owlbank.segments import SegmentLog
from owlbank.store import MemoryStore
from tests.context_api_fakes import TASK_SID, PermissiveTaskRouter

SID = "CA" + "a" * 32
REF = "jr_p219"
NOW = 1_760_108_645
FAR = 4_000_000_000  # the handler test uses the real clock
WORKER = "WK" + "2" * 32
NAME = "Jane Doe"
WITHHELD_ES = "Resumen retenido: la verificación de datos personales no está disponible."
ON = RedactionConfig(entity_types=("NAME", "ADDRESS", "CREDIT_DEBIT_NUMBER", "SSN"))
OFF = RedactionConfig(enabled=False, entity_types=("NAME",))


class NameRedactor(FakePiiRedactor):
    """The offline fake plus one NAME: ``Jane Doe`` wherever it appears."""

    def detect(self, text: str) -> list[Entity]:
        found = super().detect(text)
        start = text.find(NAME)
        while start >= 0:
            found.append(Entity(start, start + len(NAME), "NAME"))
            start = text.find(NAME, start + 1)
        return found


class SpyRedactor(FakePiiRedactor):
    """Records each text it is given (to check the batch)."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def redact(self, text, *, language, config=None):  # type: ignore[no-untyped-def]
        self.texts.append(text)
        return super().redact(text, language=language, config=config)


def prompt(seq: int, text: str, key: str = "k") -> dict:
    return {"seq": seq, "ts": NOW, "type": "prompt", "prompt_key": key, "rendered_text": text}


EVENTS = [
    {"seq": 1, "ts": NOW, "type": "call_start"},
    prompt(2, f"Hello {NAME}, welcome to OwlBank.", "greet"),
    {"seq": 3, "ts": NOW, "type": "factor_asked", "factor": "zip"},
    prompt(4, "Please enter your ZIP code.", "verify.zip"),
    {"seq": 5, "ts": NOW, "type": "ai_turn", "text": f"I am {NAME}"},
    prompt(6, f"Thanks, {NAME}.\nOne moment.", "thanks"),
    {"seq": 7, "ts": NOW, "type": "escalation", "reason": "complaint", "queue": "General"},
]


def texts_of(timeline: list[dict]) -> list[str]:
    return [e["rendered_text"] for e in timeline if "rendered_text" in e]


# ------------------------------------------------------------------------------- join and split


def test_join_and_split_keep_newlines_inside_texts() -> None:
    texts = ["a", "b\nc", "", "\n", "d"]
    assert split_texts(join_texts(texts), texts) == texts


def test_split_refuses_a_changed_line_count() -> None:
    texts = ["a", "b"]
    assert split_texts("a b", texts) is None
    assert split_texts("a\nb\nc", texts) is None


# ------------------------------------------------------------------------------- redact_texts


def test_a_name_in_a_prompt_is_masked() -> None:
    redactor = NameRedactor()
    guard = PiiGuard(redactor, ON)
    out = redact_texts([f"Hello {NAME}.", "Next.", f"Bye {NAME}"], guard, language="en-US",
                       withheld="W")
    assert out == ["Hello [NAME].", "Next.", "Bye [NAME]"]
    assert redactor.calls == 1


def test_one_call_for_all_texts_in_order() -> None:
    spy = SpyRedactor()
    texts = ["one", "two\nlines", "three"]
    assert redact_texts(texts, PiiGuard(spy, ON), language="es-US", withheld="W") == texts
    assert spy.texts == ["one\ntwo\nlines\nthree"]
    assert spy.languages == ["es"]


def test_a_failure_withholds_every_text() -> None:
    redactor = NameRedactor(fail=True)
    out = redact_texts(["a", f"b {NAME}", "c"], PiiGuard(redactor, ON), language="en-US",
                       withheld=WITHHELD_ES)
    assert out == [WITHHELD_ES] * 3


def test_disabled_makes_no_call_and_passes_the_texts() -> None:
    redactor = NameRedactor()
    texts = [f"Hello {NAME}", "x"]
    assert redact_texts(texts, PiiGuard(redactor, OFF), language="en-US", withheld="W") == texts
    assert redactor.calls == 0


def test_no_texts_no_call() -> None:
    redactor = NameRedactor()
    assert redact_texts([], PiiGuard(redactor, ON), language="en-US", withheld="W") == []
    assert redactor.calls == 0


class MergingRedactor(FakePiiRedactor):
    """A redactor whose mask swallows the separator (a span across two prompts)."""

    def redact(self, text, *, language, config=None):  # type: ignore[no-untyped-def]
        result = super().redact(text, language=language, config=config)
        return type(result)(result.text.replace("\n", " ", 1), ("NAME",))


def test_a_mask_across_the_separator_withholds(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    out = redact_texts(["a", "b", "c"], PiiGuard(MergingRedactor(), ON), language="en-US",
                       withheld="W")
    assert out == ["W", "W", "W"]
    assert any(getattr(r, "decision", None) == "withhold" for r in caplog.records)


# ------------------------------------------------------------------------------- build_timeline


def test_timeline_masks_names_and_keeps_order_and_count() -> None:
    redactor = NameRedactor()
    timeline = build_timeline(EVENTS, pii=PiiGuard(redactor, ON), language="en-US", withheld="W")
    assert [e["seq"] for e in timeline] == [1, 2, 3, 4, 6, 7]
    assert texts_of(timeline) == ["Hello [NAME], welcome to OwlBank.",
                                  "Please enter your ZIP code.",
                                  "Thanks, [NAME].\nOne moment."]
    assert redactor.calls == 1
    assert NAME not in json.dumps(timeline)


def test_timeline_digit_runs_are_masked_before_comprehend() -> None:
    spy = SpyRedactor()
    build_timeline([prompt(1, "You entered 9 4 1 0 7.")], pii=PiiGuard(spy, ON), language="en-US",
                   withheld="W")
    assert spy.texts == ["You entered [redacted]."]


def test_timeline_failure_withholds_each_prompt_only() -> None:
    timeline = build_timeline(EVENTS, pii=PiiGuard(NameRedactor(fail=True), ON), language="es-US",
                              withheld=WITHHELD_ES)
    assert texts_of(timeline) == [WITHHELD_ES] * 3
    assert [e["type"] for e in timeline] == ["call_start", "prompt", "factor_asked", "prompt",
                                             "prompt", "escalation"]


def test_timeline_disabled_makes_zero_calls() -> None:
    redactor = NameRedactor()
    timeline = build_timeline(EVENTS, pii=PiiGuard(redactor, OFF), language="en-US", withheld="W")
    assert redactor.calls == 0
    assert texts_of(timeline) == texts_of(build_timeline(EVENTS))


def test_timeline_without_a_guard_is_unchanged() -> None:
    assert texts_of(build_timeline(EVENTS))[0] == f"Hello {NAME}, welcome to OwlBank."


# ------------------------------------------------------------------------------- the policy item


def _seed_redaction(store: MemoryStore, doc: dict | None) -> None:
    if doc is not None:
        store.put("policy", {"pk": "redaction", **doc})


def test_guard_reads_the_policy_redaction_item() -> None:
    store = MemoryStore()
    _seed_redaction(store, {**ON.to_doc(), "rev": 3})
    guard = redaction_guard(store, NameRedactor(), ttl_s=60)
    assert guard.config() == ON


def test_guard_with_enabled_false_from_the_store_makes_no_call() -> None:
    store = MemoryStore()
    _seed_redaction(store, OFF.to_doc())
    redactor = NameRedactor()
    guard = redaction_guard(store, redactor, ttl_s=60)
    assert redact_texts([f"Hi {NAME}"], guard, language="en-US", withheld="W") == [f"Hi {NAME}"]
    assert redactor.calls == 0


def test_guard_missing_or_invalid_item_uses_the_defaults(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)
    store = MemoryStore()
    assert redaction_guard(store, NameRedactor(), ttl_s=60).config().enabled is True
    _seed_redaction(store, {"enabled": "maybe"})
    assert redaction_guard(store, NameRedactor(), ttl_s=60).config() == RedactionConfig()


class BrokenStore(MemoryStore):
    def get(self, table, key):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")


def test_guard_store_error_fails_closed_to_the_defaults() -> None:
    assert redaction_guard(BrokenStore(), NameRedactor(), ttl_s=60).config() == RedactionConfig()


def test_guard_caches_within_the_ttl() -> None:
    store = MemoryStore()
    _seed_redaction(store, ON.to_doc())
    t = [0.0]
    guard = redaction_guard(store, NameRedactor(), ttl_s=60, clock=lambda: t[0])
    assert guard.config() == ON
    _seed_redaction(store, OFF.to_doc())
    t[0] = 30.0
    assert guard.config() == ON
    t[0] = 61.0
    assert guard.config() == OFF


# ------------------------------------------------------------------------------- get_context


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore()
    callstate.create(s, CallState(call_sid=SID, expires_at=FAR, language="es-US",
                                  escalation={"reason": "complaint", "journal_ref": REF}))
    log = SegmentLog(s, now=lambda: NOW)
    for ev in EVENTS:
        log.append(SID, {k: v for k, v in ev.items() if k not in ("seq", "ts")})
    return s


def _request() -> dict:
    return {"httpMethod": "GET", "headers": {"Origin": ALLOWED_ORIGIN},
            "pathParameters": {"journal_ref": REF}, "queryStringParameters": {"call_sid": SID, "task_sid": TASK_SID},
            "requestContext": {"authorizer": {"worker_sid": WORKER, "roles": "supervisor"}}}


def test_get_context_redacts_in_the_call_language(store: MemoryStore) -> None:
    redactor = NameRedactor()
    resp = get_context(_request(), store, now=lambda: NOW, pii=PiiGuard(redactor, ON),
                       withheld=lambda lang: f"withheld:{lang}", taskrouter=PermissiveTaskRouter(SID, REF))
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200
    assert NAME not in resp["body"]
    assert texts_of(body["timeline"])[0] == "Hello [NAME], welcome to OwlBank."
    assert redactor.calls == 1 and redactor.languages == ["es"]


def test_get_context_failure_uses_the_withheld_text_for_the_language(store: MemoryStore) -> None:
    resp = get_context(_request(), store, now=lambda: NOW, pii=PiiGuard(NameRedactor(fail=True), ON),
                       withheld=lambda lang: f"withheld:{lang}", taskrouter=PermissiveTaskRouter(SID, REF))
    assert texts_of(json.loads(resp["body"])["timeline"]) == ["withheld:es-US"] * 3


def test_get_context_withheld_text_error_falls_back(store: MemoryStore) -> None:
    def broken(lang: str) -> str:
        raise KeyError(lang)

    resp = get_context(_request(), store, now=lambda: NOW, pii=PiiGuard(NameRedactor(fail=True), ON),
                       withheld=broken, taskrouter=PermissiveTaskRouter(SID, REF))
    assert resp["statusCode"] == 200
    assert texts_of(json.loads(resp["body"])["timeline"]) == [WITHHELD_TURN] * 3


def test_handler_builds_the_guard_from_the_store(store: MemoryStore,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_redaction(store, OFF.to_doc())
    redactor = NameRedactor()
    monkeypatch.setattr(ctx, "_store", store)
    monkeypatch.setattr(ctx, "_pii", None)
    monkeypatch.setattr(ctx, "_redactor", lambda: redactor)

    class _TaskRouterOk:  # the per-task check has its own tests (lambdas/test_context_api.py)
        def task_attributes(self, task_sid, *, timeout_s=None):
            return {"call_sid": SID, "journal_ref": REF}

        def reservations(self, task_sid, *, timeout_s=None):
            return []

    monkeypatch.setattr(ctx, "_default_taskrouter", _TaskRouterOk)
    req = _request()
    req["queryStringParameters"]["task_sid"] = "WT" + "4" * 32
    req["requestContext"]["authorizer"]["roles"] = "supervisor"
    resp = ctx.handler(req, None)
    assert resp["statusCode"] == 200
    assert redactor.calls == 0  # enabled: false in the policy item
    assert NAME in resp["body"]


# ------------------------------------------------------------------------------- transcript


def test_events_with_transcript_like_fields_are_dropped_or_stripped() -> None:
    line = "caller: my name is Jane Doe and my card is stolen"
    events = [
        # A denied-or-unlisted type is dropped whole.
        {"seq": 1, "ts": NOW, "type": "human_transcript", "text": line},
        {"seq": 2, "ts": NOW, "type": "ai_turn", "text": line, "transcript": line},
        # An allowed type keeps only allow-listed fields: the transcript-like ones never leave.
        {"seq": 3, "ts": NOW, "type": "escalation", "reason": "complaint", "transcript": line,
         "utterance": line, "text": line, "body": line},
        {"seq": 4, "ts": NOW, "type": "handoff", "outcome": "ok", "transcript": [{"body": line}]},
    ]
    out = build_timeline(events)
    assert [e["seq"] for e in out] == [3, 4]
    assert out[0] == {"seq": 3, "ts": NOW, "type": "escalation", "reason": "complaint"}
    assert "Jane" not in json.dumps(out) and "stolen" not in json.dumps(out)


# ------------------------------------------------------------------------------- close event


def test_conversation_closed_shows_closed_by_only() -> None:
    line = "agent: your card ending 4242 is cancelled, Jane Doe"
    events = [
        {"seq": 1, "ts": NOW, "type": "conversation_closed", "conversation_id": "conv_" + "a" * 26,
         "closed_by": "sweep", "transcript": line, "text": line, "worker_sid": WORKER},
        # A value outside the three is not shown (the event still is).
        {"seq": 2, "ts": NOW, "type": "conversation_closed", "closed_by": line},
    ]
    out = build_timeline(events)
    assert out == [{"seq": 1, "ts": NOW, "type": "conversation_closed", "closed_by": "sweep"},
                   {"seq": 2, "ts": NOW, "type": "conversation_closed"}]
    text = json.dumps(out)
    assert "conv_" not in text and "Jane" not in text and "4242" not in text and WORKER not in text


def test_closed_by_values_match_the_closer() -> None:
    from owlbank.humanleg_close import CLOSED_BY

    assert frozenset(CLOSED_BY.values()) == ctx.CLOSED_BY_VALUES


def test_the_close_event_adds_nothing_to_other_events() -> None:
    """``closed_by`` (or a conversation ID) on any other event type is not shown: the only new thing
    in the timeline is the ``conversation_closed`` entry."""
    base = [{"seq": i, "ts": NOW, "type": t, "outcome": "ok"}
            for i, t in enumerate(sorted(ctx.TIMELINE_TYPES - {"conversation_closed"}), 1)]
    tagged = [{**e, "closed_by": "status", "conversation_id": "conv_" + "b" * 26} for e in base]
    assert build_timeline(tagged) == build_timeline(base)


def test_get_context_shows_the_close_the_closer_journalled(store: MemoryStore) -> None:
    """The closer's real event through the journal and the Context API: one new entry, last."""
    from tests.humanleg_close_fakes import FakeClock, SlowTwilio, closer_over, fake_monotonic

    before = json.loads(get_context(_request(), store, now=lambda: NOW,
                                    taskrouter=PermissiveTaskRouter(SID, REF))["body"])["timeline"]
    callstate.mutate(store, SID, lambda s: {"ai_sessions": 1, "escalation": {
        **(s.escalation or {}), "task_sid": "WT" + "d" * 32}})
    clock = FakeClock()
    with fake_monotonic(clock):
        closer = closer_over(store, SlowTwilio(clock), journal=SegmentLog(store, now=lambda: NOW))
        assert closer.end(SID, "call_ended") is True
    after = json.loads(get_context(_request(), store, now=lambda: NOW,
                                   taskrouter=PermissiveTaskRouter(SID, REF))["body"])["timeline"]
    assert after[:-1] == before
    assert after[-1] == {"seq": before[-1]["seq"] + 1, "ts": NOW, "type": "conversation_closed",
                         "closed_by": "status"}
