"""Recap writer properties end to end.

- Once** as a state machine against a call-state store and a fake Memory: duplicate
  triggers, crashes at every step between the claim and the outcome (a ``BaseException`` that no
  handler catches, as when Lambda is killed), time jumps past the 90 s stale claim (and the run's
  deadline), and a slow writer overtaken by a second one in the middle of its run. Whatever the
  sequence, at most one summary is written per call.
- Scrub** end to end: factor values sealed on the call and spoken in the transcript,
  plain, spaced, hyphenated, with commas or as English and Spanish number words (mixed with digits),
  with fillers (uh, um, este, o sea, ...) and punctuation (``;``, ````) between the words or
  between digit groups ("94, uh, 105"), and echoed or invented by the model, never reach a Memory
  write, and neither does any run of number words reading as 4+ digits. The oracle reads spoken
  digits with :mod:`tests.recap_oracle`, not with ``number_runs``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from lambdas.recap_writer import handler as rw
from owlbank import callstate, recap, sensitive
from owlbank.callstate import CallState
from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard, RedactionConfig
from owlbank.segments import MIN_SENSITIVE_LEN, contains_value, norm_text
from owlbank.store import MemoryStore
from owlbank.token_seal import FakeSealer
from tests.recap_oracle import spoken_factor, spoken_long

SID = "CA" + "d" * 32
CONV = "conv_" + "e" * 26
EMAIL = "pat.doe@owlbank-demo.example"
START = 1_793_873_730
SEALER = FakeSealer()
CATALOG = PromptCatalog.from_dict({
    "defaultLanguage": "en-US", "languages": {"en-US": {}}, "singleLanguageKeys": {},
    "entries": [{"key": "agent.recap.system", "lang": "en-US", "text": "Recap the call.", "version": 1,
                 "status": "published", "audioUrl": None}]})


class Crash(BaseException):
    """The process dies here: no ``except Exception`` catches it, so no outcome is recorded."""


def _sealed(values: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    return tuple(SEALER.seal(json.dumps({"v": 1, "slot": s, "value": v}), call_sid=SID) for s, v in values)


def _store(sealed: tuple[str, ...] = ()) -> MemoryStore:
    store = MemoryStore()
    callstate.create(store, CallState(call_sid=SID, expires_at=START + 10**7, ani="+14155550123",
                                      verified_customer_id="C-1", level=1, ai_sessions=1,
                                      sealed_sensitive=sealed))
    store.put("customers", {"customer_id": "C-1", "memory_email": EMAIL})
    return store


@dataclass
class Memory:
    summaries: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    crash_at: str | None = None

    def _maybe(self, step: str) -> None:
        if self.crash_at == step:
            self.crash_at = None
            raise Crash(step)

    def lookup(self, id_type: str, value: str) -> list[str]:
        self._maybe("lookup")
        return ["mem_profile_1"]

    def create(self, id_type: str, value: str) -> str:
        raise AssertionError("never called: the profile exists")

    def write_summary(self, profile_id: str, item: Mapping[str, Any]) -> None:
        self._maybe("before_summary")
        self.summaries.append(dict(item))
        self._maybe("after_summary")

    def write_observations(self, profile_id: str, items: Sequence[Mapping[str, Any]]) -> None:
        self._maybe("before_observations")
        self.observations.extend(dict(i) for i in items)


@dataclass
class Conversations:
    lines: list[str] = field(default_factory=lambda: ["The customer asked about a card."])

    def latest_conversation(self, call_sid: str) -> str | None:
        return CONV

    def communications(self, conversation_id: str, *, more: Any = None) -> list[Mapping[str, Any]]:
        return [{"author": {"type": "CUSTOMER"}, "content": {"text": t}} for t in self.lines]

    def participants(self, conversation_id: str, *, more: Any = None) -> list[Mapping[str, Any]]:
        return []


def _deps(store: Any, memory: Any, model: Any, clock: Any, conversations: Any = None,
          sealer: Any = SEALER) -> rw.RecapDeps:
    return rw.RecapDeps(store=store, policy=SimpleNamespace(memory_write=True, redaction=RedactionConfig()),
                        pii=PiiGuard(FakePiiRedactor(), RedactionConfig()), sealer=sealer, memory=memory,
                        conversations=conversations or Conversations(), model=model, catalog=CATALOG,
                        clock=clock)



STEPS = ["claim", "lookup", "model", "mark", "before_summary", "after_summary", "before_observations",
         "outcome"]


class CrashingStore(MemoryStore):
    """A call-state store that dies on a chosen write: right after the claim, after the mark, or on
    the outcome."""

    crash_at: str | None = None

    def put(self, table: str, item: Mapping[str, Any], *, condition: Any = None) -> None:
        stage = None
        if table == "calls" and isinstance(item.get("recap"), Mapping) and condition == "exists":
            r = item["recap"]
            stage = "outcome" if "outcome" in r else "mark" if "writing_at" in r else "claim"
        if stage == "outcome" and self.crash_at == "outcome":
            self.crash_at = None
            raise Crash(stage)  # before the outcome lands
        super().put(table, item, condition=condition)
        if stage in ("claim", "mark") and self.crash_at == stage:
            self.crash_at = None
            raise Crash(stage)  # right after the claim or the mark landed


class OnceMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.now = START
        self.store = CrashingStore()
        callstate.create(self.store, CallState(call_sid=SID, expires_at=START + 10**7, ani="+14155550123",
                                               verified_customer_id="C-1", level=1, ai_sessions=1))
        self.store.put("customers", {"customer_id": "C-1", "memory_email": EMAIL})
        self.memory = Memory()
        self.nested: tuple[int, str | None] | None = None
        self.model_crash = False
        self.deps = _deps(self.store, self.memory, self, lambda: float(self.now))

    # The model: may crash, or let time jump and a second writer run to the end mid-call.
    def recap(self, system: str, user: str) -> Mapping[str, Any]:
        if self.model_crash:
            self.model_crash = False
            raise Crash("model")
        if self.nested is not None:
            jump, crash = self.nested
            self.nested = None
            self.now += jump
            self._run(crash)
        return {"summary": "Asked about a card.", "observations": ["Card question"]}

    def _arm(self, crash: str | None) -> None:
        self.store.crash_at = crash if crash in ("claim", "mark", "outcome") else None
        self.memory.crash_at = crash if crash in ("lookup", "before_summary", "after_summary",
                                                  "before_observations") else None
        self.model_crash = crash == "model"

    def _run(self, crash: str | None) -> None:
        self._arm(crash)
        try:
            rw.write_recap(SID, self.deps)
        except Crash:
            pass
        finally:
            self._arm(None)

    @rule(crash=st.one_of(st.none(), st.sampled_from(STEPS)))
    def trigger(self, crash: str | None) -> None:
        self._run(crash)

    @rule(jump=st.integers(0, 1500), crash=st.one_of(st.none(), st.sampled_from(STEPS)))
    def slow_writer_overtaken(self, jump: int, crash: str | None) -> None:
        self.nested = (jump, crash)
        self._run(None)
        self.nested = None

    @rule(seconds=st.integers(0, 1500))
    def time_passes(self, seconds: int) -> None:
        self.now += seconds

    @invariant()
    def at_most_one_summary(self) -> None:
        assert len(self.memory.summaries) <= 1
        assert len(self.memory.observations) <= recap.MAX_OBSERVATIONS

    @invariant()
    def a_recorded_outcome_is_final(self) -> None:
        state = callstate.load(self.store, SID)
        assert state is not None
        r = state.recap or {}
        if r.get("outcome") == "written":
            assert len(self.memory.summaries) == 1

    def teardown(self) -> None:
        sensitive.forget(SID)


TestP2Once = OnceMachine.TestCase
TestP2Once.settings = settings(max_examples=150, stateful_step_count=25, deadline=None,
                               suppress_health_check=[HealthCheck.too_slow])


def test_a_crash_free_run_writes_once() -> None:
    m = OnceMachine()
    m.trigger(None)
    m.trigger(None)
    m.time_passes(5000)
    m.trigger(None)
    assert len(m.memory.summaries) == 1



FACTOR = st.one_of(
    st.text(alphabet="0123456789", min_size=4, max_size=9),
    st.text(alphabet="abcdefghij", min_size=4, max_size=8),
)


_EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ES_DIGITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]


@st.composite
def spoken(draw: st.DrawFn, value: str) -> str:
    """``value`` as a caller or a model might write it."""
    form = draw(st.sampled_from(["plain", "spaced", "hyphen", "commas", "groups", "upper", "words",
                                 "filled"]))
    if form == "words" and value.isdigit():
        # Real-time transcription: each digit a word of either language, "oh" for 0, or the digit,
        # with separators, fillers or punctuation between, sometimes in parentheses.
        words = [draw(st.sampled_from([_EN_DIGITS[int(d)], _ES_DIGITS[int(d)], d.replace("0", "oh"), d]))
                 for d in value]
        words = [f"({w})" if draw(st.integers(0, 5)) == 0 else w for w in words]
        return "".join((draw(st.sampled_from(SEPS)) if i else "") + w for i, w in enumerate(words))
    if form == "filled" and value.isdigit():
        # Digit groups with a filler or punctuation between ("94, uh, 105").
        cut = draw(st.integers(1, len(value) - 1))
        return value[:cut] + draw(st.sampled_from(SEPS[3:])) + value[cut:]
    if form == "spaced":
        return " ".join(value)
    if form == "hyphen":
        return "-".join(value)
    if form == "commas":
        cuts = sorted(draw(st.sets(st.integers(1, len(value) - 1), max_size=3)))
        parts, prev = [], 0
        for c in cuts:
            parts.append(value[prev:c])
            prev = c
        return ",".join([*parts, value[prev:]])
    if form == "groups" and value.isdigit():
        head = len(value) % 3 or 3
        return ",".join([value[:head]] + [value[i:i + 3] for i in range(head, len(value), 3)])
    if form == "upper":
        return value.upper()
    return value


#: Between spoken digits: separators (the first three), then fillers and other punctuation.
SEPS = [" ", "-", ", ", ", uh, ", " um ", "; ", "... ", " (like) ", ", este, ", " o sea ", " pues ",
        " mm ", ": "]
FILLER = st.sampled_from(["my number is ", "it's ", " and ", "on 2026-11-08 ", "$12.50 ", "the card ",
                          ". ", ", ", "account ", "zip "])


@st.composite
def leaky_text(draw: st.DrawFn, values: Sequence[str]) -> str:
    parts = []
    for _ in range(draw(st.integers(1, 6))):
        parts.append(draw(FILLER))
        if values and draw(st.booleans()):
            parts.append(draw(spoken(draw(st.sampled_from(values)))))
    return "".join(parts) or "hello"


def _leaks(text: str, values: Sequence[str]) -> bool:
    """An independent oracle: a factor value in ``text`` with any spaces, hyphens or commas between
    its characters (digits need a non-digit around them, words a word boundary, the registry's rule),
    the registry's own match, and (a digit value) inside the digits of a spoken window
    (:mod:`tests.recap_oracle`: digit words from its own table and digits, fillers and punctuation
    skipped, within a line)."""
    folded = text.casefold()
    squeezed = re.sub(r"[\s\-,]", "", folded)
    kept_out = re.sub(r"\d{4}-\d{2}-\d{2}|\$\s?[\d,]+(?:\.\d{2})?", " KEPT ", text)
    if spoken_long(kept_out):
        return True  # a 4+ digit spoken run is never written
    for v in values:
        n = norm_text(v)
        if len(n) < MIN_SENSITIVE_LEN:
            continue
        if n.isdigit():
            if re.search(rf"(?<!\d){re.escape(n)}(?!\d)", squeezed):
                return True
            if spoken_factor(text, n):
                return True
        elif re.search(r"(?<![^\W_])" + r"[\s\-,]*".join(map(re.escape, n)) + r"(?![^\W_])", folded):
            return True
        if contains_value(text, n):
            return True
    return False


@dataclass
class EchoModel:
    """A model that copies the transcript it was given and adds factor values of its own."""

    extra_summary: str
    extra_observations: list[str]

    def recap(self, system: str, user: str) -> Mapping[str, Any]:
        body = user.split("<transcript>\n", 1)[-1].split("\n</transcript>", 1)[0]
        return {"summary": f"{body} {self.extra_summary}",
                "observations": [*self.extra_observations, body[:200]]}


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.data(), st.lists(FACTOR, min_size=1, max_size=3, unique_by=norm_text))
def test_factor_values_never_reach_a_write(data: st.DataObject, values: list[str]) -> None:
    lines = [data.draw(leaky_text(values)) for _ in range(data.draw(st.integers(1, 4)))]
    model = EchoModel(data.draw(leaky_text(values)),
                      [data.draw(leaky_text(values)) for _ in range(data.draw(st.integers(0, 5)))])
    store = _store(_sealed([(f"slot{i}", v) for i, v in enumerate(values)]))
    memory = Memory()
    deps = _deps(store, memory, model, lambda: float(START), Conversations(lines))
    try:
        out = rw.write_recap(SID, deps)
    finally:
        sensitive.forget(SID)
    written = [s["content"] for s in memory.summaries] + [o["content"] for o in memory.observations]
    for text in written:
        assert not _leaks(text, values), (text, values)
        assert not recap.has_factor_value(text, values)
    if out.outcome == rw.WRITTEN:
        assert len(memory.summaries) == 1
