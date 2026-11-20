"""Model-assisted slots: timeline redaction properties.

* **Split round trip:** for any list of texts (newlines inside included), the join, an identity
  redaction and the split return the list unchanged; and so does ``redact_texts`` with an identity
  redactor, in one call.
* With ``enabled: false``, ``build_timeline`` makes zero redactor calls for any
  journal, and its prompt texts equal the unguarded timeline's.
* With a failing redactor, every prompt text is the withheld text and the entries keep their order.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.context_api.handler import build_timeline, join_texts, redact_texts, split_texts
from owlbank.pii import FakePiiRedactor, PiiGuard, RedactionConfig, RedactionResult

TEXT = st.text(alphabet=st.characters(exclude_categories=("Cs",)) | st.sampled_from(
    ["\n", "\r", " ", "-", "1"]), max_size=40)
TEXTS = st.lists(TEXT, max_size=12)
ON = RedactionConfig(entity_types=("NAME", "SSN", "CREDIT_DEBIT_NUMBER"))


class IdentityRedactor:
    def __init__(self) -> None:
        self.calls = 0

    def redact(self, text: str, *, language: str, config: Any = None) -> RedactionResult:
        self.calls += 1
        return RedactionResult(text, ())


@settings(max_examples=300)
@given(TEXTS)
def test_split_round_trip(texts: list[str]) -> None:
    assert split_texts(join_texts(texts), texts) == texts


@settings(max_examples=300)
@given(TEXTS)
def test_identity_redaction_round_trip(texts: list[str]) -> None:
    redactor = IdentityRedactor()
    out = redact_texts(texts, PiiGuard(redactor, ON), language="en-US", withheld="W")
    assert out == texts
    # One call per request, none when there is nothing to check (PiiGuard skips blank text).
    assert redactor.calls == (1 if join_texts(texts).strip() else 0)


EVENT = st.one_of(
    st.fixed_dictionaries({"type": st.just("prompt"), "prompt_key": st.text(max_size=8),
                           "rendered_text": TEXT}),
    st.fixed_dictionaries({"type": st.sampled_from(["call_start", "verification", "ai_turn",
                                                    "escalation", "tool_call"]),
                           "outcome": st.text(max_size=8)}),
    st.dictionaries(st.text(max_size=6), st.text(max_size=6) | st.integers(), max_size=4),
)
EVENTS = st.lists(EVENT, max_size=15).map(
    lambda evs: [{"seq": i, **e} for i, e in enumerate(evs)])


@settings(max_examples=300)
@given(EVENTS, st.sampled_from(["en-US", "es-US", "fr-FR"]),
       st.lists(st.sampled_from(sorted(RedactionConfig().entity_types) + ["NAME"]), max_size=4))
def test_disabled_timeline_makes_zero_calls(events: list[dict], language: str,
                                                   types: list[str]) -> None:
    redactor = FakePiiRedactor()
    off = RedactionConfig(enabled=False, entity_types=tuple(dict.fromkeys(types)))
    guarded = build_timeline(events, pii=PiiGuard(redactor, off), language=language, withheld="W")
    assert redactor.calls == 0
    assert guarded == build_timeline(events)


@settings(max_examples=200)
@given(EVENTS)
def test_failure_withholds_every_prompt_and_keeps_order(events: list[dict]) -> None:
    plain = build_timeline(events)
    failed = build_timeline(events, pii=PiiGuard(FakePiiRedactor(fail=True), ON), language="en-US",
                            withheld="W")
    assert [e["seq"] for e in failed] == [e["seq"] for e in plain]
    for p, f in zip(plain, failed, strict=True):
        if "rendered_text" in p:
            # A blank batch is never sent, so it can't fail: it passes through.
            batch_blank = not join_texts([e["rendered_text"] for e in plain
                                          if "rendered_text" in e]).strip()
            assert f["rendered_text"] == (p["rendered_text"] if batch_blank else "W")
        else:
            assert f == p
