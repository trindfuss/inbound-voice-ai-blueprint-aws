"""Slot model bench: latency, tokens and cost per 1,000 calls of the model a slot answer reaches.

Fixed, fictional utterance sets per slot type (``digits``, ``yes_no``, ``choice``, ``amount``,
``date``), in en-US and es-US, in five categories: ``clean``, ``messy``, ``self_corrected``,
``choice`` (the choice slot's answers) and ``off_script`` (repeat, a human, another request).

A model is reached through an **adapter** (:class:`Adapter`, registered in :data:`ADAPTERS`).

* ``resolver`` (the default) runs each case through the slot engine
  (:func:`owlbank.slots.engine.resolve`) with :class:`~owlbank.slots.resolver.BedrockResolver`,
  once per **mode**: ``deterministic`` (the engine is given no resolver, so no model call is made
  and none is reported), ``assisted`` and ``generative``. An answer reaches the model only when the
  deterministic stages leave it open, as on a call. The generative mode is the effective one
  (:func:`owlbank.slots.modes.mode_decision` with ``generative_enabled``): ``generative`` on a
  non-sensitive choice slot, ``assisted`` on every other slot. Its wording (``say``) is
  checked with :func:`owlbank.slots.generative.fallback_reason` whenever the grounded answer needs
  a follow-up question (an inferred choice to confirm, an ambiguous choice): the fallback rate by
  reason.

What the model sees is what it sees on a call: the caller's words, unredacted, for a
sensitive slot too (``ssn4`` here: fictional digits).

What is reported, per mode, overall and by slot type, language and
category:

* ``answers`` and ``model_calls`` (0 in ``deterministic``);
* ``latency_p50_ms`` / ``latency_p90_ms`` over the answers that called the model: the time the
  caller waits for the answer, measured around the engine call (deterministic stages, request
  build, Converse and parsing, under the 1.5 s budget). ``server_latency_p50_ms`` is Bedrock's
  ``metrics.latencyMs``. Percentiles are nearest rank, so each is one of the samples;
* ``input_tokens_mean`` / ``output_tokens_mean`` per call, from Converse ``usage``;
* ``cost_per_1000_usd``: 1000 x the mean per-call cost, ``in x p_in + out x p_out``, from
  :data:`PRICES` (the Bedrock on-demand per-token prices, with their source and date);
* ``model_outcomes`` (``ok``, ``timeout``, ``error``, ``refusal``, ``malformed``, ``ungrounded``);
* ``fallback`` and ``fallback_rate`` (generative): ``accepted`` or the check's reason;
* ``kinds`` and ``correct_rate`` against each case's expected kind (and value, where one is set).

**Stop rule.** If any mode's p90 is over 1.0 s, the report says ``STOP``, the result
file records it (``latency_stop.stop``) and the run exits 3, so the numbers are
reported before going on.

The result file (``<label>-<UTC>.json`` in the results directory) holds case IDs, kinds, outcomes, token counts
and latencies, **never the words**: no utterance, question, label, wording or value.

**Invocation logging guard.** Before any model call the bench reads
``GetModelInvocationLoggingConfiguration`` in ``--region``, the source Region whose configuration
applies to every call through the US profile, and refuses to run (exit 2) if it delivers
text, or doesn't say. ``--accept-text-logging`` runs anyway with a warning: you accept, in a
trusted dev sandbox, that the synthetic utterances are logged.

Usage (repository root; approve every run yourself; AWS credentials for the sandbox)::

    uv run python scripts/bench/slot_model_bench.py --profile <profile> --repeats 3
    uv run python scripts/bench/slot_model_bench.py --list                     # the cases, offline
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from owlbank.catalog import PromptCatalog  # noqa: E402
from owlbank.slot_policy import (  # noqa: E402
    DEFAULT_RESOLVER_BUDGET_MS,
    MODES,
    Mode,
    OptionDef,
    ResolutionSettings,
    SlotSettings,
)
from owlbank.slots import engine, generative  # noqa: E402
from owlbank.slots.modes import mode_decision  # noqa: E402
from owlbank.slots.resolver import (  # noqa: E402
    DEFAULT_MODEL_ID,
    BedrockResolver,
    Resolver,
    ResolverOutput,
    ResolverRequest,
    client_for_budget,
)
from owlbank.slots.types import OptionView, Resolution, ResolveContext, SlotDecl, SlotType  # noqa: E402

LANGS: tuple[str, ...] = ("en-US", "es-US")
CATEGORIES: tuple[str, ...] = ("clean", "messy", "self_corrected", "choice", "off_script")
#: Result kinds: the engine's match, with ``exact`` and ``inferred`` as ``value`` and ``none`` as
#: ``unclear``.
KINDS: tuple[str, ...] = ("value", "unclear", "ambiguous", "correction", "repeat", "escalate",
                          "intent_switch", "cancel")
DEFAULT_REGION = "us-east-1"
DEFAULT_LABEL = "slot-resolver"
#: A p90 over this is a stop.
LATENCY_STOP_P90_MS = 1000.0
#: The date the cases are resolved on, so a run is repeatable (the date cases' window holds it).
TODAY = date(2026, 10, 20)
CATALOG_PATH = ROOT / "data" / "prompt_catalog.json"


# ----------------------------------------------------------------------------- prices


@dataclass(frozen=True, slots=True)
class Price:
    """On-demand per-token prices of one model, per million tokens."""

    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    currency: str
    source: str
    effective: str
    looked_up: str

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_per_mtok / 1e6 + output_tokens * self.output_per_mtok / 1e6

    def record(self) -> dict[str, Any]:
        return asdict(self)


#: Looked up 2026-10-19 with the AWS Price List API (``pricing:GetProducts``, service
#: ``AmazonBedrockFoundationModels``, ``regionCode=us-east-1``, model "Claude Haiku 5.5", on-demand).
#: The ``us.`` geographic profile is billed at the Standard (in-Region) rates; the global profile's
#: are lower ($0.10 and $0.50), and long-context (> 200K tokens) rates don't apply to a slot call.
PRICES: dict[str, Price] = {
    "us.anthropic.claude-haiku-5-5": Price(
        model_id="us.anthropic.claude-haiku-5-5",
        input_per_mtok=0.11,
        output_per_mtok=0.55,
        currency="USD",
        source=("AWS Price List API GetProducts, AmazonBedrockFoundationModels, us-east-1, Claude Haiku 5.5 "
                "on-demand: USE1-MP:USE1_input_tokens_standard-Units (Million Input Tokens Standard) and "
                "USE1-MP:USE1_output_tokens_standard-Units (Million Response Tokens Standard)"),
        effective="2026-10-01",
        looked_up="2026-10-19",
    ),
}


def price_for(model_id: str) -> Price:
    """The price of ``model_id``; a model with no price is an error, never a zero cost."""
    return PRICES[model_id]


# ----------------------------------------------------------------------------- cases


@dataclass(frozen=True, slots=True)
class Case:
    """One fixed input. ``slot`` is the slot it answers (the seed's names where the seed has the
    slot). ``options`` is ``(value, label)`` pairs in the case's language (choice slots).
    ``expect_kind`` ``None`` means the case isn't scored."""

    id: str
    slot_type: SlotType
    lang: str
    category: str
    question: str
    utterance: str
    slot: str = ""
    constraints: Mapping[str, Any] = field(default_factory=dict)
    options: tuple[tuple[str, str], ...] = ()
    label: str | None = None
    expect_kind: str | None = "value"
    expect_value: str | None = None
    sensitive: bool = False

    def decl(self) -> SlotDecl:
        constraints = dict(self.constraints)
        if self.options:
            constraints["options"] = [{"value": v, "labelKey": f"bench.{v}"} for v, _ in self.options]
        return SlotDecl(slot_type=self.slot_type, constraints=constraints)

    def option_labels(self) -> list[dict[str, str]] | None:
        return [{"value": v, "label": lab} for v, lab in self.options] if self.options else None


#: The seed's card aliases (``data/policy.json``), by option and language.
CARD_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "red": {"en-US": ("red",), "es-US": ("roja",)},
    "purple": {"en-US": ("purple",), "es-US": ("morada",)},
}


def _cases() -> tuple[Case, ...]:
    zip5 = {"length": 5}
    ssn4 = {"length": 4}
    amount = {"min": "1.00", "max": "5000.00", "currency": "USD"}
    when = {"min": "2026-10-01", "max": "2027-12-31"}
    cards_en = (("red", "OwlBank Red Card"), ("purple", "OwlBank Purple Card"))
    cards_es = (("red", "Tarjeta Roja OwlBank"), ("purple", "Tarjeta Morada OwlBank"))
    q = {
        "en-US": {"zip": "What is the ZIP code on your account?", "ssn4": "What are the last four digits "
                  "of your Social Security number?", "yes_no": "Would you like to make this payment now?",
                  "card": "Which card is this about: the OwlBank Red Card or the OwlBank Purple Card?",
                  "amount": "How much would you like to pay?", "date": "On what date should the payment go out?",
                  "phone": "What is the phone number associated with your account?"},
        "es-US": {"zip": "¿Cuál es el código postal de su cuenta?", "ssn4": "¿Cuáles son los últimos cuatro "
                  "dígitos de su número de Seguro Social?", "yes_no": "¿Desea hacer este pago ahora?",
                  "card": "¿De qué tarjeta se trata: la Tarjeta Roja OwlBank o la Tarjeta Morada OwlBank?",
                  "amount": "¿Cuánto desea pagar?", "date": "¿En qué fecha debe salir el pago?",
                  "phone": "¿Cuál es el número de teléfono asociado a su cuenta?"},
    }
    lab = {"en-US": {"zip": "ZIP code", "ssn4": "last four of your Social Security number",
                     "phone": "phone number"},
           "es-US": {"zip": "código postal", "ssn4": "últimos cuatro de su Seguro Social",
                     "phone": "número de teléfono"}}
    out: list[Case] = []

    def add(lang: str, cid: str, slot_type: SlotType, category: str, slot: str, utterance: str,
            **kw: Any) -> None:
        out.append(Case(id=f"{cid}-{lang[:2]}", slot_type=slot_type, lang=lang, category=category,
                        question=q[lang][slot], utterance=utterance, slot=slot, **kw))

    for lang, cards, w in (("en-US", cards_en, "en"), ("es-US", cards_es, "es")):
        en = w == "en"
        z = {"constraints": zip5, "label": lab[lang]["zip"]}
        # ssn4 is sensitive in the seed: its words still reach the resolver unredacted.
        # Fictional digits only; the result file never holds them.
        s = {"constraints": ssn4, "label": lab[lang]["ssn4"], "sensitive": True}
        a = {"constraints": amount}
        d = {"constraints": when}
        c = {"options": cards}
        # digits: ZIP (not sensitive in the seed) and SSN last-4 (sensitive)
        add(lang, "zip-clean", "digits", "clean", "zip",
            "my zip is nine four one zero five" if en else "mi código postal es nueve cuatro uno cero cinco",
            expect_value="94105", **z)
        add(lang, "ssn4-clean", "digits", "clean", "ssn4",
            "the last four are four eight two one" if en else "los últimos cuatro son cuatro ocho dos uno",
            expect_value="4821", **s)
        add(lang, "zip-messy", "digits", "messy", "zip",
            "uh it's like nine four uh one oh five I think" if en
            else "eh es como nueve cuatro eh uno cero cinco creo", expect_value="94105", **z)
        add(lang, "zip-corrected", "digits", "self_corrected", "zip",
            "nine four one zero six, no sorry, nine four one zero five" if en
            else "nueve cuatro uno cero seis, no perdón, nueve cuatro uno cero cinco",
            expect_value="94105", **z)
        add(lang, "ssn4-corrected", "digits", "self_corrected", "ssn4",
            "it's four eight two two, wait, no, four eight two one" if en
            else "es cuatro ocho dos dos, espere, no, cuatro ocho dos uno", expect_value="4821", **s)
        # Digits given in parts (as one caller said it: "the first two are one one and then the
        # last two after that are one one"); fictional values.
        add(lang, "ssn4-parts", "digits", "messy", "ssn4",
            "the first two are four eight and then the last two after that are two one" if en
            else "los primeros dos son cuatro ocho y luego los últimos dos son dos uno",
            expect_value="4821", **s)
        add(lang, "ssn4-pairs", "digits", "messy", "ssn4",
            "forty eight and twenty one" if en else "cuarenta y ocho y veintiuno",
            expect_value="4821", **s)
        # A phone number in groups with thinking aloud, and with one group corrected
        # fictional values, as words (sensitive).
        ph = {"constraints": {"length": 10}, "label": lab[lang]["phone"], "sensitive": True}
        add(lang, "phone-groups", "digits", "messy", "phone",
            "yeah the number is four oh seven, eh, five four four, what was it, five five five five"
            if en else "sí, el número es cuatro cero siete, eh, cinco cuatro cuatro, cómo era, "
            "cinco cinco cinco cinco", expect_value="4075445555", **ph)
        add(lang, "phone-corrected", "digits", "self_corrected", "phone",
            "four oh seven, five five five... no, five four four, five five five five" if en
            else "cuatro cero siete, cinco cinco cinco... no, cinco cuatro cuatro, cinco cinco cinco cinco",
            expect_value="4075445555", **ph)
        add(lang, "zip-offscript", "digits", "off_script", "zip",
            "I don't have that handy, can I just talk to someone" if en
            else "no lo tengo a la mano, ¿puedo hablar con alguien?", expect_kind="escalate", **z)
        # yes_no
        add(lang, "yesno-clean", "yes_no", "clean", "yes_no",
            "yes please go ahead" if en else "sí por favor adelante", expect_value="yes")
        add(lang, "yesno-messy", "yes_no", "messy", "yes_no",
            "yeah I mean sure why not" if en else "pues sí claro por qué no", expect_value="yes")
        add(lang, "yesno-corrected", "yes_no", "self_corrected", "yes_no",
            "no, actually wait, yes, do it" if en else "no, espere, mejor sí, hágalo", expect_value="yes")
        add(lang, "yesno-offscript", "yes_no", "off_script", "yes_no",
            "sorry, could you say that again" if en else "perdón, ¿puede repetir eso?", expect_kind="repeat")
        # choice: every case of the choice slot is in the choice category
        add(lang, "card-clean", "choice", "choice", "card",
            "the purple card" if en else "la tarjeta morada", expect_value="purple", **c)
        add(lang, "card-messy", "choice", "choice", "card",
            "um the red one I guess, the red card" if en else "eh la roja creo, la tarjeta roja",
            expect_value="red", **c)
        add(lang, "card-corrected", "choice", "choice", "card",
            "the red one, no, the purple one" if en else "la roja, no, la morada", expect_value="purple", **c)
        add(lang, "card-offscript", "choice", "choice", "card",
            "actually I want to check my balance first" if en else "en realidad primero quiero ver mi saldo",
            expect_kind="intent_switch", **c)
        # amount
        add(lang, "amount-clean", "amount", "clean", "amount",
            "two hundred fifty dollars" if en else "doscientos cincuenta dólares", expect_value="250.00", **a)
        add(lang, "amount-messy", "amount", "messy", "amount",
            "uh let's do like a hundred and twenty bucks" if en else "eh pongamos unos ciento veinte dólares",
            expect_value="120.00", **a)
        add(lang, "amount-corrected", "amount", "self_corrected", "amount",
            "three hundred, no, make it two hundred" if en else "trescientos, no, mejor doscientos",
            expect_value="200.00", **a)
        add(lang, "amount-offscript", "amount", "off_script", "amount",
            "I'd like to speak with a representative" if en else "quisiera hablar con un representante",
            expect_kind="escalate", **a)
        # date
        add(lang, "date-clean", "date", "clean", "date",
            "November fifteenth twenty twenty six" if en else "el quince de noviembre de dos mil veintiséis",
            expect_value="2026-11-15", **d)
        add(lang, "date-messy", "date", "messy", "date",
            "um the fifteenth of November I suppose" if en else "eh el quince de noviembre supongo", **d)
        add(lang, "date-corrected", "date", "self_corrected", "date",
            "November tenth, sorry, November fifteenth, twenty twenty six" if en
            else "el diez de noviembre, perdón, el quince de noviembre de dos mil veintiséis",
            expect_value="2026-11-15", **d)
        add(lang, "date-offscript", "date", "off_script", "date",
            "wait, what's my current balance" if en else "espere, ¿cuál es mi saldo actual?",
            expect_kind="intent_switch", **d)
    return tuple(out)


CASES: tuple[Case, ...] = _cases()
CASE_BY_ID: dict[str, Case] = {c.id: c for c in CASES}


def select(cases: Sequence[Case], *, slot_types: Sequence[str] | None, langs: Sequence[str] | None,
           categories: Sequence[str] | None) -> list[Case]:
    picked = [c for c in cases
              if (not slot_types or c.slot_type in slot_types)
              and (not langs or c.lang in langs)
              and (not categories or c.category in categories)]
    if not picked:
        raise ValueError("no case matches the filters")
    return picked


# ----------------------------------------------------------------------------- samples and adapters


@dataclass(frozen=True, slots=True)
class Sample:
    """One answer. ``model_calls`` is how many model calls it made (0 or 1); tokens are ``None``
    when no response arrived. ``model`` is the engine's model outcome, ``fallback`` the generative
    check's verdict (``accepted`` or a reason) when the answer needed wording. ``error`` is a bench
    failure (the adapter raised), never a model outcome."""

    case_id: str
    slot_type: str
    lang: str
    category: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    server_latency_ms: int | None
    kind: str | None
    correct: bool | None
    error: str | None
    model_calls: int = 1
    model: str | None = None
    fallback: str | None = None
    mode: str | None = None
    effective_mode: str | None = None

    @classmethod
    def of(cls, case: Case, *, latency_ms: float, input_tokens: int | None, output_tokens: int | None,
           server_latency_ms: int | None, kind: str | None, value: str | None,
           error: str | None = None, model_calls: int = 1, model: str | None = None,
           fallback: str | None = None, mode: str | None = None,
           effective_mode: str | None = None) -> Sample:
        """A sample from an adapter's outcome; ``value`` is only compared, never kept."""
        correct = None
        if error is None and case.expect_kind is not None and kind is not None:
            correct = kind == case.expect_kind and (case.expect_value is None or value == case.expect_value)
        return cls(case.id, case.slot_type, case.lang, case.category, round(latency_ms, 1), input_tokens,
                   output_tokens, server_latency_ms, kind, correct, error, model_calls, model, fallback,
                   mode, effective_mode)

    @classmethod
    def failed(cls, case: Case, *, latency_ms: float, error: str, mode: str | None = None) -> Sample:
        return cls.of(case, latency_ms=latency_ms, input_tokens=None, output_tokens=None,
                      server_latency_ms=None, kind=None, value=None, error=error, mode=mode)


class Adapter(Protocol):
    name: str

    def describe(self) -> dict[str, Any]:
        """What the result file records about the model (no credentials)."""
        ...

    def call(self, case: Case) -> Sample: ...


def kind_of(match: str) -> str:
    """The engine's match as a bench kind."""
    if match in ("exact", "inferred"):
        return "value"
    return "unclear" if match == "none" else match


def slot_settings(case: Case) -> SlotSettings:
    """The slot ``case`` answers, as the seed declares it where the seed has it (``zip``, ``ssn4``,
    ``yes_no``, ``card``); ``amount`` and ``date`` are bench slots with the same shape rules."""
    options: tuple[OptionDef, ...] = ()
    if case.slot_type == "choice":
        options = tuple(OptionDef(value=v, aliases=dict(CARD_ALIASES.get(v, {}))) for v, _ in case.options)
    decl = SlotDecl(slot_type=case.slot_type, constraints=dict(case.constraints),
                    label_key=f"label.{case.slot}")
    return SlotSettings(name=case.slot, decl=decl, is_factor=case.slot in ("zip", "ssn4"),
                        sensitive=case.sensitive, confirm="never" if case.sensitive else "when_inferred",
                        options=options)


class _Capture:
    """What one engine call did with the model: the calls, the output and Converse's usage."""

    __slots__ = ("calls", "output", "input_tokens", "output_tokens", "server_latency_ms")

    def __init__(self) -> None:
        self.calls = 0
        self.output: ResolverOutput | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.server_latency_ms: int | None = None


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


class _RecordingClient:
    """A ``bedrock-runtime`` client that writes each Converse response's usage and server latency
    into the capture of the answer that made the call (an abandoned call writes into its own)."""

    def __init__(self, inner: Any, capture: _Capture) -> None:
        self._inner = inner
        self._capture = capture

    def converse(self, **kwargs: Any) -> Any:
        response = self._inner.converse(**kwargs)
        if isinstance(response, Mapping):
            usage = response.get("usage")
            metrics = response.get("metrics")
            if isinstance(usage, Mapping):
                self._capture.input_tokens = _int(usage.get("inputTokens"))
                self._capture.output_tokens = _int(usage.get("outputTokens"))
            if isinstance(metrics, Mapping):
                self._capture.server_latency_ms = _int(metrics.get("latencyMs"))
        return response


class _Tap:
    """A :class:`Resolver` that counts the calls and keeps the output (compared, never written)."""

    def __init__(self, inner: Resolver, capture: _Capture) -> None:
        self._inner = inner
        self._capture = capture

    def resolve(self, req: ResolverRequest, *, budget_ms: int, generative: bool,
                flow: str | None = None) -> ResolverOutput:
        self._capture.calls += 1
        out = self._inner.resolve(req, budget_ms=budget_ms, generative=generative, flow=flow)
        self._capture.output = out
        return out


def load_catalog(path: Path = CATALOG_PATH) -> PromptCatalog:
    return PromptCatalog.from_dict(json.loads(path.read_text()))


class ResolverAdapter:
    """The slot engine with :class:`BedrockResolver`, in one mode.

    ``client_for(budget_ms)`` gives the ``bedrock-runtime`` client, as on a call
    (:func:`~owlbank.slots.resolver.client_for_budget`). In ``deterministic`` mode no resolver is
    built and ``client_for`` is never called. ``clock`` is in seconds (monotonic).
    """

    name = "resolver"

    def __init__(self, client_for: Callable[[int], Any], *, model_id: str, mode: Mode,
                 budget_ms: int = DEFAULT_RESOLVER_BUDGET_MS, catalog: PromptCatalog | None = None,
                 clock: Callable[[], float] = time.perf_counter, today: date = TODAY) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode: Mode = mode
        self.model_id = model_id
        self.budget_ms = budget_ms
        self._catalog = catalog or load_catalog()
        self._clock = clock
        self._today = today
        self._client_for = client_for
        self._capture = _Capture()
        self._resolver: BedrockResolver | None = None
        if mode != "deterministic":
            self._resolver = BedrockResolver(self._recording_client, model_id=model_id)
        self._resolution = ResolutionSettings(default_mode="assisted", generative_enabled=True,
                                              resolver_budget_ms=budget_ms)

    def _recording_client(self, budget_ms: int) -> Any:
        return _RecordingClient(self._client_for(budget_ms), self._capture)

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name, "mode": self.mode, "model_id": self.model_id,
                "budget_ms": self.budget_ms,
                "max_tokens": self._resolver.max_tokens if self._resolver is not None else None}

    def effective_mode(self, slot: SlotSettings) -> Mode:
        """The mode the slot runs in with this run's mode as the configured one and the ceiling
        (generative only on a non-sensitive choice slot)."""
        return mode_decision(slot, self._resolution, "bench", self.mode, raw_mode=self.mode,
                             raw_flow_mode=None, raw_default="assisted").effective

    def context(self, case: Case, slot: SlotSettings) -> ResolveContext:
        cat = self._catalog
        lang = case.lang
        key = slot.decl.label_key or ""
        labels: dict[str, str] = {}
        if case.label:
            labels[key] = case.label
        elif cat.has(key, lang):
            labels[key] = cat.text(key, lang)
        known = []
        for v, lab in case.options:
            okey = generative.option_key(slot.name, v)
            known.append(OptionView(value=v, name=cat.text(okey, lang) if cat.has(okey, lang) else lab,
                                    aliases=CARD_ALIASES.get(v, {}).get(lang, ())))
        return engine.make_context(question_text=case.question, labels=labels,
                                   commands=cat.commands(lang), known_options=known,
                                   budget_ms=self.budget_ms)

    def _fallback(self, case: Case, slot: SlotSettings, res: Resolution, out: ResolverOutput | None
                  ) -> str | None:
        """The generative check's verdict when the answer needs a follow-up question (a choice to
        confirm or to disambiguate), else ``None``."""
        if res.source != "model" or slot.decl.slot_type != "choice":
            return None
        if res.match == "inferred" and res.value is not None:
            allowed: tuple[str, ...] = (res.value,)
        elif res.match == "ambiguous" and res.candidates:
            allowed = tuple(res.candidates)
        else:
            return None
        say, say_lang = (out.say, out.say_lang) if out is not None else (None, None)
        reason = generative.fallback_reason(say, case.lang, say_lang, slot=slot.name, allowed=allowed,
                                            catalog_has=self._catalog.has)
        return reason or "accepted"

    def call(self, case: Case) -> Sample:
        slot = slot_settings(case)
        eff = self.effective_mode(slot)
        ctx = self.context(case, slot)
        cap = self._capture = _Capture()
        resolver = _Tap(self._resolver, cap) if self._resolver is not None else None
        started = self._clock()
        try:
            res = engine.resolve(slot, digits=None, speech=case.utterance, confidence=None,
                                 lang=case.lang, mode=eff, ctx=ctx, resolver=resolver,
                                 today=self._today, flow="bench")
        except Exception as exc:  # noqa: BLE001 - a bench failure is recorded, never fatal
            return Sample.failed(case, latency_ms=(self._clock() - started) * 1000,
                                 error=type(exc).__name__, mode=self.mode)
        latency_ms = (self._clock() - started) * 1000
        fallback = self._fallback(case, slot, res, cap.output) if eff == "generative" else None
        return Sample.of(case, latency_ms=latency_ms, input_tokens=cap.input_tokens,
                         output_tokens=cap.output_tokens, server_latency_ms=cap.server_latency_ms,
                         kind=kind_of(res.match), value=res.value, model_calls=cap.calls,
                         model=res.model, fallback=fallback, mode=self.mode, effective_mode=eff)


#: The adapters by name: each takes ``client_for(budget_ms)`` (a ``bedrock-runtime`` client per
#: budget), ``model_id`` and ``mode``. The classifier's went with the classifier.
ADAPTERS: dict[str, Callable[..., Adapter]] = {"resolver": ResolverAdapter}


def run(adapter: Adapter, cases: Sequence[Case], *, repeats: int,
        progress: Callable[[int, int, Sample], None] | None = None) -> list[Sample]:
    """Every case once per repeat, in order (repeat 1 of all cases, then repeat 2, ...)."""
    total = repeats * len(cases)
    samples: list[Sample] = []
    for _ in range(repeats):
        for c in cases:
            samples.append(adapter.call(c))
            if progress is not None:
                progress(len(samples), total, samples[-1])
    return samples


def warm(adapter: Adapter, cases: Sequence[Case], n: int) -> int:
    """Unrecorded answers until ``n`` of them called the model (at most one pass over ``cases``
    per call wanted), to open the connection. Returns the model calls made."""
    made = 0
    for i in range(n * len(cases)):
        if made >= n:
            break
        made += adapter.call(cases[i % len(cases)]).model_calls
    return made


# ----------------------------------------------------------------------------- statistics


def pct(xs: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest sample with at least ``p`` of the samples at or below it."""
    if not xs:
        return None
    ys = sorted(xs)
    return ys[max(1, math.ceil(p * len(ys))) - 1]


def cost_per_1000(usage: Iterable[tuple[int, int]], price: Price) -> float | None:
    """1000 x the mean per-call cost over ``(input_tokens, output_tokens)`` pairs."""
    costs = [price.cost(i, o) for i, o in usage]
    return 1000 * statistics.fmean(costs) if costs else None


def _mean(xs: Sequence[float]) -> float | None:
    return round(statistics.fmean(xs), 1) if xs else None


def _group(samples: Sequence[Sample], price: Price) -> dict[str, Any]:
    called = [s for s in samples if s.model_calls > 0 and s.error is None]
    answered = [s for s in called if s.input_tokens is not None and s.output_tokens is not None]
    lat = [s.latency_ms for s in called]
    server = [float(s.server_latency_ms) for s in called if s.server_latency_ms is not None]
    scored = [s.correct for s in samples if s.correct is not None]
    cost = cost_per_1000([(s.input_tokens or 0, s.output_tokens or 0) for s in answered], price)
    fallbacks = collections.Counter(s.fallback for s in samples if s.fallback is not None)
    needed = sum(fallbacks.values())
    return {
        "answers": len(samples),
        "model_calls": sum(s.model_calls for s in samples),
        "errors": sum(1 for s in samples if s.error),
        "latency_p50_ms": pct(lat, 0.5),
        "latency_p90_ms": pct(lat, 0.9),
        "latency_max_ms": max(lat) if lat else None,
        "server_latency_p50_ms": pct(server, 0.5),
        "server_latency_p90_ms": pct(server, 0.9),
        "input_tokens_mean": _mean([float(s.input_tokens or 0) for s in answered]),
        "output_tokens_mean": _mean([float(s.output_tokens or 0) for s in answered]),
        "cost_per_1000_usd": round(cost, 6) if cost is not None else None,
        "model_outcomes": dict(collections.Counter(s.model for s in called if s.model is not None)),
        "fallback": dict(fallbacks),
        "fallback_rate": round((needed - fallbacks["accepted"]) / needed, 3) if needed else None,
        "kinds": dict(collections.Counter(s.kind for s in samples if s.kind is not None)),
        "correct_rate": round(sum(scored) / len(scored), 3) if scored else None,
    }


def summarize(samples: Sequence[Sample], price: Price) -> dict[str, Any]:
    def by(key: str) -> dict[str, Any]:
        groups: dict[str, list[Sample]] = {}
        for s in samples:
            groups.setdefault(getattr(s, key), []).append(s)
        return {k: _group(v, price) for k, v in sorted(groups.items())}

    return {"overall": _group(samples, price), "by_slot_type": by("slot_type"), "by_lang": by("lang"),
            "by_category": by("category")}


def latency_stop(summaries: Mapping[str, Mapping[str, Any]], threshold_ms: float = LATENCY_STOP_P90_MS) -> dict[str, Any]:
    """The latency stop flag: whether any mode's p90 is over the stop threshold, and the message to show."""
    p90 = {m: s["overall"]["latency_p90_ms"] for m, s in summaries.items()}
    over = [m for m, v in p90.items() if v is not None and v > threshold_ms]
    msg = None
    if over:
        msg = ("STOP: resolver p90 over 1.0 s in "
               + ", ".join(f"{m} ({p90[m]} ms)" for m in over)
               + ". The resolver runs with no strict mode or effort, a prompt-cache point, a 2.5 s"
                 " budget and no filler phrase: report the numbers before going on.")
    return {"threshold_ms": threshold_ms, "p90_ms": p90, "over": over, "stop": bool(over),
            "message": msg}


# ----------------------------------------------------------------------------- invocation logging guard


def logging_verdict(response: Mapping[str, Any]) -> str:
    """``off`` (no configuration), ``on_no_text`` (text delivery explicitly disabled) or ``on_text``."""
    cfg = response.get("loggingConfig")
    if not cfg:
        return "off" if cfg is None else "on_text"
    return "on_no_text" if cfg.get("textDataDeliveryEnabled") is False else "on_text"


def logging_check(read: Callable[[str], Mapping[str, Any]], regions: Iterable[str]) -> dict[str, str]:
    return {r: logging_verdict(read(r)) for r in regions}


def logging_blocks(verdicts: Mapping[str, str]) -> bool:
    return any(v == "on_text" for v in verdicts.values())


def _session(profile: str | None, region: str) -> Any:
    import boto3

    return boto3.Session(profile_name=profile, region_name=region)


def read_logging_config(profile: str | None, region: str) -> Mapping[str, Any]:
    resp: Mapping[str, Any] = _session(profile, region).client("bedrock").get_model_invocation_logging_configuration()
    return resp


def make_client_for(profile: str | None, region: str) -> Callable[[int], Any]:
    """``client_for(budget_ms)`` as the front door builds it (``connect_timeout = read_timeout =
    budget``, one attempt), from the profile's session. No client is built before the first call."""
    return client_for_budget(_session(profile, region).client, region_name=region)


# ----------------------------------------------------------------------------- the result file


def write_result(out: Path, *, label: str, adapters: Mapping[str, Adapter], price: Price,
                 runs: Mapping[str, Sequence[Sample]], cases: Sequence[Case], repeats: int, warmup: int,
                 logging_check: Mapping[str, str],
                 region: str = DEFAULT_REGION) -> Path:
    """One file for the run: per mode, the summary and the samples; the stop verdict."""
    now = datetime.now(UTC)
    summaries = {m: summarize(s, price) for m, s in runs.items()}
    result = {
        "label": label,
        "at": now.isoformat(timespec="seconds"),
        "region": region,
        "price": price.record(),
        "logging_check": dict(logging_check),
        "cases": len(cases),
        "repeats": repeats,
        "warmup": warmup,
        "latency_stop": latency_stop(summaries),
        "modes": {m: {"adapter": adapters[m].describe(), "summary": summaries[m],
                      "samples": [asdict(s) for s in runs[m]]} for m in runs},
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}-{now:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return path


def _fmt(x: Any) -> str:
    return "-" if x is None else f"{x}"


def print_summary(summary: Mapping[str, Any], title: str) -> None:
    print(f"\n{title}")
    print(f"{'group':<28}{'answers':>8}{'calls':>6}{'err':>5}{'p50 ms':>9}{'p90 ms':>9}{'in tok':>8}"
          f"{'out tok':>8}{'$/1000':>10}{'correct':>9}")
    rows = [("overall", summary["overall"])]
    for key in ("by_slot_type", "by_lang", "by_category"):
        rows += [(f"{key[3:]}={k}", v) for k, v in summary[key].items()]
    for name, g in rows:
        print(f"{name:<28}{g['answers']:>8}{g['model_calls']:>6}{g['errors']:>5}{_fmt(g['latency_p50_ms']):>9}"
              f"{_fmt(g['latency_p90_ms']):>9}{_fmt(g['input_tokens_mean']):>8}{_fmt(g['output_tokens_mean']):>8}"
              f"{_fmt(g['cost_per_1000_usd']):>10}{_fmt(g['correct_rate']):>9}")
    o = summary["overall"]
    if o["model_outcomes"]:
        print("model outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(o["model_outcomes"].items())))
    if o["fallback"]:
        print(f"generative fallback rate {o['fallback_rate']}: "
              + ", ".join(f"{k} {v}" for k, v in sorted(o["fallback"].items())))


def print_comparison(data: Mapping[str, Any]) -> None:
    """The modes side by side, then the stop verdict."""
    print(f"\n{'':<24}{'calls':>7}{'p50 ms':>9}{'p90 ms':>9}{'in tok':>8}{'out tok':>8}{'$/1000':>10}"
          f"{'correct':>9}")
    rows = []
    rows += [(m, v["summary"]["overall"]) for m, v in data["modes"].items()]
    for name, g in rows:
        print(f"{name:<24}{_fmt(g.get('model_calls')):>7}{_fmt(g.get('latency_p50_ms')):>9}"
              f"{_fmt(g.get('latency_p90_ms')):>9}{_fmt(g.get('input_tokens_mean')):>8}"
              f"{_fmt(g.get('output_tokens_mean')):>8}{_fmt(g.get('cost_per_1000_usd')):>10}"
              f"{_fmt(g.get('correct_rate')):>9}")
    verdict = data["latency_stop"]
    if verdict["stop"]:
        print(f"\n{verdict['message']}")
        print(verdict["message"], file=sys.stderr)
    else:
        print(f"\np90 at or under {verdict['threshold_ms']:.0f} ms in every mode: no stop.")


# ----------------------------------------------------------------------------- CLI


def _csv(s: str | None) -> list[str] | None:
    return [x for x in s.split(",") if x] if s else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--adapter", default="resolver", help=f"one of: {', '.join(sorted(ADAPTERS))}")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated (default: all three)")
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--profile", default=None, help="AWS profile (default: the environment's)")
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--repeats", type=int, default=3, help="answers per case and mode (default 3)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="unrecorded model calls first, per mode that calls the model")
    ap.add_argument("--slot-types", default=None, help="comma-separated filter")
    ap.add_argument("--langs", default=None, help="comma-separated filter")
    ap.add_argument("--categories", default=None, help="comma-separated filter")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results"))
    ap.add_argument("--list", action="store_true", help="list the cases (IDs and categories) and exit")
    ap.add_argument("--accept-text-logging", action="store_true",
                    help="run even if Bedrock invocation logging delivers text in --region (trusted dev "
                         "sandbox only: the synthetic utterances will be logged)")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        cases = select(CASES, slot_types=_csv(args.slot_types), langs=_csv(args.langs),
                       categories=_csv(args.categories))
        price = price_for(args.model_id)
    except (ValueError, KeyError) as exc:
        ap.error(f"{exc}")
    if args.list:
        for c in cases:
            print(f"{c.id:<24}{c.slot_type:<9}{c.lang:<7}{c.category:<16}{_fmt(c.expect_kind)}")
        print(f"{len(cases)} cases")
        return
    modes = _csv(args.modes) or []
    if not modes or any(m not in MODES for m in modes) or len(set(modes)) != len(modes):
        ap.error(f"--modes must be distinct values of: {', '.join(MODES)}")
    if args.repeats < 1 or args.warmup < 0:
        ap.error("--repeats must be at least 1 and --warmup at least 0")

    verdicts = logging_check(lambda r: read_logging_config(args.profile, r), (args.region,))
    print("invocation logging: " + ", ".join(f"{r} {v}" for r, v in verdicts.items()), flush=True)
    if logging_blocks(verdicts) and args.accept_text_logging:
        print("WARNING: Bedrock invocation logging delivers text in " + args.region + "; running anyway "
              "(--accept-text-logging): the synthetic utterances will be logged", file=sys.stderr)
    elif logging_blocks(verdicts):
        print("refusing to run: Bedrock invocation logging delivers text in "
              + ", ".join(r for r, v in verdicts.items() if v == "on_text"), file=sys.stderr)
        raise SystemExit(2)

    if args.adapter not in ADAPTERS:
        ap.error(f"--adapter must be one of: {', '.join(sorted(ADAPTERS))}")
    client_for = make_client_for(args.profile, args.region)

    def progress(n: int, total: int, s: Sample) -> None:
        print(f"{s.mode or ''} {n}/{total} {s.case_id}: {s.latency_ms} ms calls={s.model_calls} "
              f"{s.kind or ''} {s.model or ''} {s.error or ''}".rstrip(), flush=True)

    adapters: dict[str, Adapter] = {}
    runs: dict[str, list[Sample]] = {}
    for mode in modes:
        adapter = adapters[mode] = ADAPTERS[args.adapter](client_for, model_id=args.model_id, mode=mode)
        if mode != "deterministic":
            warm(adapter, cases, args.warmup)
        runs[mode] = run(adapter, cases, repeats=args.repeats, progress=progress)
    out = Path(args.out)
    path = write_result(out, label=args.label, adapters=adapters, price=price, runs=runs, cases=cases,
                        repeats=args.repeats, warmup=args.warmup, logging_check=verdicts,
                        region=args.region)
    data = json.loads(path.read_text())
    for mode, v in data["modes"].items():
        o = v["summary"]["overall"]
        print_summary(v["summary"], f"{args.label} {mode}: {o['answers']} answers, {o['model_calls']} model "
                                    f"calls, {o['errors']} errors")
    print_comparison(data)
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(f"\nsaved {shown}")
    if data["latency_stop"]["stop"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
