"""Property tests for the resolver.

No AWS calls: clients are fakes, and a fake pool plus a fake clock model a call that blocks
(``FakeFuture.result`` advances the clock by the block, or by the timeout and raises
``TimeoutError``, as a real future would). Real-thread timing is covered in
``tests/unit/test_resolver.py``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

import pytest
from botocore.exceptions import ClientError
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from owlbank.slot_policy import SlotSettings
from owlbank.slots.resolver import (
    GENERATIVE_TOOL_NAME,
    MAX_CANDIDATES,
    MAX_VALUE_LEN,
    RESOLVER_KINDS,
    TOOL_NAME,
    BedrockResolver,
    ResolverOutput,
    ResolverRequest,
    build_request,
)
from owlbank.slots.types import (
    PENDING_KINDS,
    SLOT_TYPES,
    CollectedSlot,
    OptionView,
    Pending,
    ResolveContext,
    SlotDecl,
)
from tests.logutil import text_of

EN = "en-US"

# Catalog words, names and the caller's words come from lowercase letters; every secret carries
# an upper-case marker that no generated catalog text can contain.
words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=16)
names = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)
secrets_ = st.builds(lambda n, tag: f"{tag}{n}", st.integers(0, 10**9),
                     st.sampled_from(["SNTLV", "SNTLR", "SNTLT"]))




@st.composite
def option_lists(draw: st.DrawFn, min_size: int = 0) -> tuple[OptionView, ...]:
    values = draw(st.lists(names, unique=True, min_size=min_size, max_size=3))
    return tuple(OptionView(v, draw(words), tuple(draw(st.lists(words, max_size=2))))
                 for v in values)


@st.composite
def collected_slots(draw: st.DrawFn, name: str) -> CollectedSlot:
    """A collected slot as the front door builds it: a choice slot (sometimes) with its options
    and a subset of them as the caller's; no value anywhere."""
    t = draw(st.sampled_from(SLOT_TYPES))
    options: tuple[OptionView, ...] = ()
    caller: tuple[str, ...] = ()
    if t == "choice":
        options = draw(option_lists())
        caller = tuple(v for v in (o.value for o in options) if draw(st.booleans()))
    return CollectedSlot(name, t, draw(words), draw(st.sampled_from(["current", "earlier"])),
                         draw(st.booleans()), options=options, caller_options=caller)


@st.composite
def sessions(draw: st.DrawFn) -> dict[str, Any]:
    """A call's session as the front door holds it: collected values, the customer record and
    the token (all secrets), plus what the resolver may see (names, labels, options)."""
    collected_names = draw(st.lists(names, unique=True, max_size=4))
    return {
        "values": {n: draw(secrets_) for n in collected_names},
        "record": {k: draw(secrets_) for k in ("name", "account_number", "balance", "zip", "ssn4")},
        "token": draw(secrets_),
        "collected": tuple(draw(collected_slots(n)) for n in collected_names),
        "options": tuple(OptionView(draw(names), draw(words), tuple(draw(st.lists(words, max_size=2))))
                         for _ in range(draw(st.integers(0, 3)))),
        "pending_kind": draw(st.none() | st.sampled_from(PENDING_KINDS)),
        "question": draw(words),
        "slot_type": draw(st.sampled_from(SLOT_TYPES)),
        "length": draw(st.integers(1, 10)),
    }


def context_of(s: dict[str, Any]) -> ResolveContext:
    pending = None
    if s["pending_kind"] is not None:
        sealed = s["values"].get(next(iter(s["values"]), ""), s["token"])
        pending = Pending(s["pending_kind"], "zip",
                          option=s["options"][0].value if s["options"] else None,
                          candidates=tuple(o.value for o in s["options"]),
                          sealed_value=sealed)            # a secret, as a sealed blob would be
    return ResolveContext(
        question_text=s["question"], labels={c.name: c.label for c in s["collected"]},
        commands={"agent": ("agent",)}, known_options=s["options"],
        caller_options=tuple(o.value for o in s["options"]), collected=s["collected"],
        pending=pending, budget_ms=1500)


def test_request_fields_are_exactly_what_3_3_allows() -> None:
    assert [f.name for f in dataclasses.fields(ResolverRequest)] == [
        "language", "question", "slot", "known_options", "caller_options", "collected", "pending",
        "utterance"]


@settings(max_examples=200, deadline=None)
@given(s=sessions(), utterance=st.builds(lambda w: f"UTTERANCE {w}", words),
       generative=st.booleans())
def test_request_holds_no_session_secret(s: dict[str, Any], utterance: str, generative: bool) -> None:
    slot = SlotSettings(name="current", decl=SlotDecl(s["slot_type"], {"length": s["length"]}),
                        is_factor=False, sensitive=True)
    req = build_request(slot, context_of(s), lang=EN, utterance=utterance)
    kwargs = BedrockResolver(lambda _b: None).request(req, generative=generative)
    blob = json.dumps(kwargs, ensure_ascii=False)
    secrets = [*s["values"].values(), *s["record"].values(), s["token"]]
    assert not [x for x in secrets if x in blob]
    # Only the utterance carries caller words, and only inside the reply delimiters.
    text = kwargs["messages"][0]["content"][0]["text"]
    head, _, reply = text.partition("\n\nCaller reply:\n")
    assert "UTTERANCE" not in head and "UTTERANCE" not in kwargs["system"][0]["text"]
    assert reply == f"<reply>{utterance}</reply>"
    # Collected slots: names, types, labels and scope; a choice slot also its options and the
    # caller's options, exactly as collected (never which one was chosen, 6.7). Option names,
    # aliases and caller option values appear nowhere else for a non-choice entry.
    payload = json.loads(head.removeprefix("Request:\n"))
    assert len(payload["collected"]) == len(s["collected"])
    for entry, c in zip(payload["collected"], s["collected"], strict=True):
        base = {"name": c.name, "type": c.slot_type, "label": c.label, "scope": c.scope}
        if c.slot_type != "choice":
            assert entry == base
            continue
        assert entry == {**base,
                         "options": [{"value": o.value, "name": o.name, "aliases": list(o.aliases)}
                                     for o in c.options],
                         "caller_options": list(c.caller_options)}
        assert set(entry["caller_options"]) <= {o["value"] for o in entry["options"]}


def test_collected_slot_fields_hold_no_value() -> None:
    """A collected slot can't carry a value; its options come from the policy."""
    assert [f.name for f in dataclasses.fields(CollectedSlot)] == [
        "name", "slot_type", "label", "scope", "verified", "options", "caller_options"]


@pytest.mark.parametrize("kw", [
    {"slot_type": "digits", "options": (OptionView("a", "A"),)},
    {"slot_type": "yes_no", "caller_options": ("yes",)},
    {"slot_type": "choice", "options": (OptionView("a", "A"),), "caller_options": ("b",)},
    {"slot_type": "choice", "options": (OptionView("a", "A"), OptionView("a", "B"))},
    {"slot_type": "choice", "options": [OptionView("a", "A")]},
])
def test_collected_slot_options_are_validated(kw: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        CollectedSlot(**{"name": "card", "label": "card", "scope": "current", "verified": False,
                         **kw})


# ------------------------------------------------------- Cached prefix

#: Every per-call text below carries this marker; the fixed prompt and tool spec never do.
PER_CALL = "PERCALL"


def marked(s: dict[str, Any]) -> dict[str, Any]:
    """``s`` with the marker on every per-call text field the request may carry: the question,
    option names and aliases, and collected-slot labels."""
    return {
        **s,
        "question": f"{PER_CALL}Q {s['question']}",
        "options": tuple(OptionView(o.value, f"{PER_CALL}N {o.name}",
                                    tuple(f"{PER_CALL}A {a}" for a in o.aliases))
                         for o in s["options"]),
        "collected": tuple(dataclasses.replace(
            c, label=f"{PER_CALL}L {c.label}",
            options=tuple(OptionView(o.value, f"{PER_CALL}N {o.name}",
                                     tuple(f"{PER_CALL}A {a}" for a in o.aliases))
                          for o in c.options))
            for c in s["collected"]),
    }


CACHE_POINT = {"cachePoint": {"type": "default"}}


def _reference_request(generative: bool) -> dict[str, Any]:
    """A fixed, unrelated request: the cached segments of every request must equal its own."""
    slot = SlotSettings(name="zip", decl=SlotDecl("digits", {"length": 5}), is_factor=True,
                        sensitive=False)
    ctx = ResolveContext(question_text="What is your ZIP code?", labels={}, commands={},
                         known_options=(), caller_options=(), collected=(), pending=None,
                         budget_ms=2500)
    return BedrockResolver(lambda _b: None).request(
        build_request(slot, ctx, lang="es-US", utterance="nueve cuatro"), generative=generative)


@settings(max_examples=200, deadline=None)
@given(s=sessions().map(marked), utterance=st.builds(lambda w: f"UTTERANCE {w}", words),
       slot_name=names, generative=st.booleans())
def test_nothing_per_call_reaches_a_cached_segment(s: dict[str, Any], utterance: str,
                                                         slot_name: str, generative: bool) -> None:
    """The cacheable prefix (``toolConfig``, then ``system``, up to and including their cache
    points) is the same for every call of a mode, so it can hold no utterance, collected value,
    record field, token or request field; all of that is in ``messages``, which has no cache
    point. Bedrock reuses a cache entry only on an exact prefix match, so a hit can only ever
    reuse these fixed segments."""
    slot = SlotSettings(name=slot_name, decl=SlotDecl(s["slot_type"], {"length": s["length"]}),
                        is_factor=False, sensitive=True)
    req = build_request(slot, context_of(s), lang=EN, utterance=utterance)
    kwargs = BedrockResolver(lambda _b: None).request(req, generative=generative)
    ref = _reference_request(generative)

    # The cached segments are a function of the mode only.
    assert kwargs["system"] == ref["system"]
    assert kwargs["toolConfig"] == ref["toolConfig"]
    # Exactly one cache point closes each segment; nothing follows it there.
    assert kwargs["system"][-1] == CACHE_POINT and kwargs["toolConfig"]["tools"][-1] == CACHE_POINT
    assert json.dumps(kwargs).count('"cachePoint"') == 2
    # No marker of per-call text, and no secret, in a cached segment.
    cached = json.dumps([kwargs["toolConfig"], kwargs["system"]], ensure_ascii=False)
    secrets = [*s["values"].values(), *s["record"].values(), s["token"]]
    assert PER_CALL not in cached and "UTTERANCE" not in cached
    assert not [x for x in secrets if x in cached]
    # Everything per call is in the one user message, which holds no cache point.
    assert "cachePoint" not in json.dumps(kwargs["messages"])
    [message] = kwargs["messages"]
    [block] = message["content"]
    assert set(block) == {"text"}
    text = block["text"]
    assert utterance in text and s["question"] in text
    assert all(o.name in text for o in s["options"])
    assert all(c.label in text for c in s["collected"])
    assert all(o.name in text and all(a in text for a in o.aliases)
               for c in s["collected"] for o in c.options)
    event(f"options={len(s['options'])} collected={len(s['collected'])}")


# ------------------------------------------------------------------------------ fakes


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeFuture:
    def __init__(self, pool: FakePool, fn: Any, args: tuple[Any, ...]) -> None:
        self.pool, self.fn, self.args = pool, fn, args

    def result(self, timeout: float | None = None) -> Any:
        try:
            value, exc = self.fn(*self.args), None
        except BaseException as e:  # noqa: BLE001 - re-raised below, as a future does
            value, exc = None, e
        block = self.pool.block_s
        if timeout is not None and block > timeout:
            self.pool.clock.now += timeout
            raise TimeoutError
        self.pool.clock.now += block
        if exc is not None:
            raise exc
        return value

    def cancel(self) -> bool:
        return False


class FakePool:
    def __init__(self, clock: FakeClock, block_s: float = 0.0) -> None:
        self.clock, self.block_s = clock, block_s

    def submit(self, fn: Any, *args: Any) -> FakeFuture:
        return FakeFuture(self, fn, args)


class ScriptedClient:
    def __init__(self, behaviour: tuple[str, Any]) -> None:
        self.behaviour = behaviour
        self.calls = 0

    def converse(self, **_kw: Any) -> Any:
        self.calls += 1
        what, arg = self.behaviour
        if what == "raise":
            raise arg
        return arg



SENTINEL = "SNTLX0042Q"

json_leaf = st.none() | st.booleans() | st.integers() | st.just(SENTINEL) | st.text(max_size=8)
any_json = st.recursive(json_leaf, lambda c: st.lists(c, max_size=4)
                        | st.dictionaries(st.text(max_size=8) | st.just(SENTINEL), c, max_size=4),
                        max_leaves=12)
maybe_sentinel = st.one_of(st.just(SENTINEL), st.text(max_size=70), st.none(), st.integers())


@st.composite
def tool_inputs(draw: st.DrawFn) -> Any:
    """Near-valid tool inputs: each field valid or not, often carrying the sentinel."""
    d: dict[str, Any] = {
        "kind": draw(st.sampled_from([*RESOLVER_KINDS, SENTINEL, "slot_value"]) | st.none()),
        "value": draw(maybe_sentinel),
        "candidates": draw(st.lists(maybe_sentinel, max_size=7) | maybe_sentinel),
        "slot": draw(maybe_sentinel),
        "reply": draw(st.sampled_from(["yes", "no", None, SENTINEL])),
    }
    if draw(st.booleans()):
        d["say"], d["say_lang"] = draw(maybe_sentinel), draw(maybe_sentinel)
    if draw(st.booleans()):
        d[draw(st.sampled_from(["extra", SENTINEL]))] = draw(any_json)
    if draw(st.booleans()):
        d.pop(draw(st.sampled_from(sorted(d))))
    return json.dumps(d) if draw(st.booleans()) else d


@st.composite
def responses(draw: st.DrawFn) -> Any:
    name = draw(st.sampled_from([TOOL_NAME, GENERATIVE_TOOL_NAME, SENTINEL]))
    blocks = [{"toolUse": {"toolUseId": SENTINEL, "name": name, "input": draw(tool_inputs())}}
              for _ in range(draw(st.integers(0, 2)))]
    if draw(st.booleans()):
        blocks.append({"text": SENTINEL})
    return {"output": {"message": {"role": "assistant", "content": blocks}},
            "stopReason": draw(st.sampled_from(["tool_use", "tool_use", "end_turn", SENTINEL])),
            "usage": draw(st.fixed_dictionaries({"inputTokens": st.integers(0, 9999) | st.just(SENTINEL),
                                                 "outputTokens": st.integers(0, 999)}) | any_json)}


class SentinelError(Exception):
    pass


exceptions = st.one_of(
    st.builds(lambda: RuntimeError(SENTINEL)),
    st.builds(lambda: ValueError(f"bad input {SENTINEL}", SENTINEL)),
    st.builds(lambda: KeyError(SENTINEL)),
    st.builds(lambda: OSError(5, SENTINEL)),
    st.builds(lambda: TimeoutError(SENTINEL)),
    st.builds(lambda: SentinelError(SENTINEL)),
    st.builds(lambda code: ClientError({"Error": {"Code": code, "Message": f"echo {SENTINEL}"},
                                        "ResponseMetadata": {"HTTPStatusCode": 400}}, "Converse"),
              st.sampled_from(["ThrottlingException", "ValidationException", "AccessDeniedException"])),
)

valid_inputs = st.fixed_dictionaries({
    "kind": st.sampled_from(RESOLVER_KINDS),
    "value": st.none() | st.just(SENTINEL) | st.text(max_size=MAX_VALUE_LEN),
    "candidates": st.lists(st.just(SENTINEL) | st.text(max_size=MAX_VALUE_LEN), max_size=MAX_CANDIDATES),
    "slot": st.none() | st.just(SENTINEL) | st.text(max_size=MAX_VALUE_LEN),
    "reply": st.sampled_from(["yes", "no", None]),
    "say": st.none() | st.just(SENTINEL) | st.text(max_size=40),
    "say_lang": st.none() | st.just(EN) | st.just(SENTINEL),
})

behaviours = st.one_of(
    st.tuples(st.just("raise"), exceptions),
    st.tuples(st.just("valid"), valid_inputs),
    st.tuples(st.just("return"), responses()),
    st.tuples(st.just("return"), any_json),
)


def assert_schema_valid(out: ResolverOutput, generative: bool) -> None:
    assert out.kind in RESOLVER_KINDS
    assert out.value is None or (isinstance(out.value, str) and len(out.value) <= MAX_VALUE_LEN)
    assert out.slot is None or (isinstance(out.slot, str) and len(out.slot) <= MAX_VALUE_LEN)
    assert len(out.candidates) <= MAX_CANDIDATES
    assert all(isinstance(c, str) and len(c) <= MAX_VALUE_LEN for c in out.candidates)
    assert out.reply in ("yes", "no", None)
    if not generative:
        assert out.say is None and out.say_lang is None


CARD = SlotSettings(name="card", decl=SlotDecl("choice", {}), is_factor=False, sensitive=False)


def card_request(utterance: str) -> ResolverRequest:
    ctx = ResolveContext(question_text="Which card?", labels={}, commands={},
                         known_options=(OptionView("red", "OwlBank Red Card"),),
                         caller_options=("red",), collected=(), pending=None, budget_ms=1500)
    return build_request(CARD, ctx, lang=EN, utterance=utterance)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(behaviour=behaviours, generative=st.booleans(), block_s=st.floats(0, 3), budget=st.integers(500, 4000))
def test_any_client_behaviour_is_none_or_valid_and_never_logged(
        caplog: pytest.LogCaptureFixture, behaviour: tuple[str, Any], generative: bool,
        block_s: float, budget: int) -> None:
    caplog.clear()
    clock = FakeClock()
    if behaviour[0] == "valid":                # schema-valid for the tool this call uses
        inp = dict(behaviour[1])
        if not generative:
            del inp["say"], inp["say_lang"]
        behaviour = ("return", {"stopReason": "tool_use", "usage": {"inputTokens": 3},
                                "output": {"message": {"content": [{"toolUse": {
                                    "toolUseId": SENTINEL, "input": inp,
                                    "name": GENERATIVE_TOOL_NAME if generative else TOOL_NAME}}]}}})
    client = ScriptedClient(behaviour)
    res = BedrockResolver(lambda _b: client, clock=clock, pool=FakePool(clock, block_s))  # type: ignore[arg-type]
    with caplog.at_level(logging.DEBUG):
        out = res.resolve(card_request(f"my card {SENTINEL}"), budget_ms=budget,
                          generative=generative, flow="confirm")
    assert isinstance(out, ResolverOutput)
    event(f"outcome={out.outcome}")
    if out.outcome != "ok":
        assert out.kind == "none"
    else:
        assert behaviour[0] == "return" and block_s <= budget / 1000
        assert_schema_valid(out, generative)
    if block_s > budget / 1000:
        assert out.outcome == "timeout"
    assert SENTINEL not in text_of(caplog.records)



OK_RESPONSE = {"output": {"message": {"role": "assistant", "content": [
    {"toolUse": {"toolUseId": "t", "name": TOOL_NAME, "input": {
        "kind": "exact", "value": "red", "candidates": [], "slot": None, "reply": None}}}]}},
    "stopReason": "tool_use", "usage": {"inputTokens": 1, "outputTokens": 1}}

EPS = 1e-6


@settings(max_examples=300, deadline=None)
@given(budget=st.integers(500, 4000), block_s=st.floats(0, 20, allow_nan=False),
       setup_frac=st.floats(0, 0.99))
def test_returns_within_budget_after_one_attempt(budget: int, block_s: float,
                                                       setup_frac: float) -> None:
    clock = FakeClock()
    client = ScriptedClient(("return", OK_RESPONSE))
    setup_s = budget / 1000 * setup_frac

    def client_for(b: int) -> Any:
        assert b == budget
        clock.now += setup_s           # building the client costs part of the budget
        return client

    res = BedrockResolver(client_for, clock=clock, pool=FakePool(clock, block_s))  # type: ignore[arg-type]
    t0 = clock.now
    out = res.resolve(card_request("the red one"), budget_ms=budget, generative=False)
    assert clock.now - t0 <= budget / 1000 + EPS
    assert client.calls == 1
    if setup_s + block_s > budget / 1000 + EPS:
        assert out.outcome == "timeout"
    elif setup_s + block_s < budget / 1000 - EPS:
        assert out.outcome == "ok" and out.value == "red"
