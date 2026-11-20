"""The slot resolution engine.

Order for every ``<Gather input="dtmf speech">`` result:

1. **DTMF:** parse and validate ``Digits``. Done if valid; never reaches the resolver. For a follow-up
question answered yes or no (a read-back, an option confirmation, a
   not-available question with one option on file) keys 1 and 2 are yes and no. Before it,
   on a ``declinable`` slot, ``#`` alone (``FinishedOnKey`` ``#``, no other key, no speech) is
   ``decline`` in every mode. ``*`` is no decline: alone it is an
   invalid entry, and it keeps its meaning elsewhere (an amount's decimal point).
2. **Speech, deterministic** (unchanged stages): the slot type's parser over the normalized
   words. Stage 2b (a value inside a longer sentence) refuses a sentence that names another
   collected slot's label. A choice slot with policy options also matches their spoken
   names and aliases as ``exact``, then, when the question just listed options
   (``pending.listed``), an ordinal reference to them as ``exact``, source ``ordinal``.
   A follow-up question's yes or no comes from the word lists (``reply_deterministic``).
3. **2c:** a digit answer cut short is re-asked at once, with no model call.
4. **Universal commands:** whole-utterance match of ``command.*`` phrases.
5. **Resolver** (``assisted`` and ``generative`` only; never touched in ``deterministic``): one strict
tool call on the caller's words, unredacted. Its output is never
   trusted: a digit value must be grounded (exact length, ordered subsequence of the words), a choice
   must be a known option and a model ``exact`` is demoted to ``inferred`` unless
   the option was named, candidates must be caller options. Anything else is ``none``
   with ``model="ungrounded"``. **Answer first:** a value that grounds for the slot being
   answered is that answer, whatever ``kind`` the resolver gave it.

The engine never decides the next question: :func:`owlbank.slots.dialog.decide` does, from the
:class:`Resolution`. It never interprets which new intent the caller wants either; that is the
agent's job, from ``utterance_for_ai``: the caller's words unchanged for a non-sensitive slot, the
``[<slot name>]`` placeholder text for a sensitive one. With an answer that has a
value, that text is the placeholder text for the value.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

# Modules, not names: ``owlbank.sensitive`` and ``owlbank.slot_policy`` import from this package.
from owlbank import sensitive as _sensitive
from owlbank import slot_policy as _slot_policy
from owlbank.obs import log_caught
from owlbank.tools_contract import PHONE_SLOT

from . import grounding
from .commands import match_command
from .normalize import normalize
from .resolver import Resolver, ResolverOutput, build_request
from .types import (
    CollectedSlot,
    Match,
    ModelOutcome,
    OptionView,
    Pending,
    Resolution,
    ResolveContext,
    SlotDecl,
    Source,
)
from .validators import (
    YES_NO_DTMF,
    option_where,
    ordinal_where,
    parse_dtmf,
    parse_speech_located,
    short_digit_answer,
    validate,
    yes_no_speech,
)

log = logging.getLogger(__name__)

#: Follow-up questions answered with yes or no: their keypad menu is 1 = yes, 2 = no.
YES_NO_PENDING: frozenset[str] = frozenset({"confirm_digits", "confirm_option", "not_available_one"})
#: The finish key that, pressed alone, declines a ``declinable`` slot:
#: Twilio's ``FinishedOnKey`` with empty ``Digits``.
DECLINE_FINISH_KEY = "#"


def utterance_for_ai(slot: _slot_policy.SlotSettings, speech: str | None, value: str | None, *,
                     value_runs: Sequence[int] | None = None,
                     span: tuple[int, int] | None = None) -> str | None:
    """The caller's words as they may reach the agent and the journal: unchanged
    for a non-sensitive slot, :func:`owlbank.sensitive.placeholder_text` for a sensitive one.
    ``value_runs`` and ``span`` locate the value in the words (the deterministic parse, or
    grounding for a resolver value); ``None`` when there was no speech."""
    if speech is None or not speech.strip():
        return None
    if not slot.sensitive:
        return speech
    return _sensitive.placeholder_text(speech, slot, value, value_runs=value_runs, span=span)


def option_labels(slot: SlotDecl, labels: Mapping[str, str]) -> list[dict[str, str]] | None:
    """``[{"value", "label"}]`` for a ``choice`` slot declared with ``constraints.options``,
    labels from the catalog text."""
    if slot.slot_type != "choice":
        return None
    return [
        {"value": o["value"], "label": labels.get(o["label_key"], o["value"])}
        for o in slot.options()
    ]


def hints(slot: SlotDecl, labels: Mapping[str, str]) -> list[str]:
    """Speech-recognition hints derived from the slot (choice option labels)."""
    return [o["label"] for o in option_labels(slot, labels) or []]


def make_context(*, question_text: str, labels: Mapping[str, str],
                 commands: Mapping[str, Sequence[str]],
                 known_options: Sequence[OptionView] = (),
                 caller_options: Sequence[str] | None = None,
                 collected: Sequence[CollectedSlot] = (), pending: Pending | None = None,
                 budget_ms: int | None = None) -> ResolveContext:
    """A :class:`ResolveContext`. ``caller_options`` defaults to every known option (no subset,
    5.1); ``budget_ms`` to ``slot_resolution``'s default."""
    if budget_ms is None:
        budget_ms = _slot_policy.DEFAULT_RESOLVER_BUDGET_MS
    caller = tuple(o.value for o in known_options) if caller_options is None else tuple(caller_options)
    return ResolveContext(question_text=question_text, labels=dict(labels), commands=dict(commands),
                          known_options=tuple(known_options), caller_options=caller,
                          collected=tuple(collected), pending=pending, budget_ms=budget_ms)


def slot_options(slot: _slot_policy.SlotSettings, ctx: ResolveContext
                 ) -> tuple[tuple[OptionView, ...], tuple[str, ...]]:
    """The known options and the caller's options of a choice slot: the policy's options as the
    context carries them (spoken names and aliases in the call's language), or, for a slot
    declared with ``constraints.options``, those with their catalog labels as spoken names (every
    option is the caller's). Empty for any other slot."""
    if slot.decl.slot_type != "choice":
        return (), ()
    if slot.options:
        return tuple(ctx.known_options), tuple(ctx.caller_options)
    views = tuple(OptionView(value=o["value"], name=o["label"])
                  for o in option_labels(slot.decl, ctx.labels) or [])
    return views, tuple(v.value for v in views)


def _dtmf_value(slot: _slot_policy.SlotSettings, digits: str, caller: tuple[str, ...],
                today: date) -> str | None:
    decl = slot.decl
    if decl.slot_type == "choice" and slot.options:
        key = digits.strip().rstrip("#")
        for o in slot.options:                      # an explicit key wins
            if o.dtmf is not None and o.dtmf == key:
                return o.value if o.value in caller else None
        explicit = {o.value for o in slot.options if o.dtmf is not None}
        free = [v for v in caller if v not in explicit]
        if key.isdigit() and 1 <= int(key) <= len(free):
            return free[int(key) - 1]             # default: position among the caller's options
        return None
    return validate(decl, parse_dtmf(decl, digits), today=today)


def _other_labels(slot: _slot_policy.SlotSettings, ctx: ResolveContext) -> tuple[str, ...]:
    return tuple(c.label for c in ctx.collected if c.name != slot.name and c.label)


def _ground(slot: _slot_policy.SlotSettings, value: object, speech: str, lang: str,
            known: tuple[OptionView, ...]) -> str | None:
    """A resolver value for ``slot`` in canonical form if it passes the validator and grounding
    (3.4: 4.2 for digits, 5.2 for choices), else ``None``. A yes/no needs only the validator; an
    amount or a date is never taken from the model (fail closed)."""
    if not isinstance(value, str):
        return None
    t = slot.decl.slot_type
    if t == "digits":
        return grounding.ground_digits(slot, value, speech, lang)
    if t == "choice":
        return value if value in {o.value for o in known} else None
    if t == "yes_no":
        return validate(slot.decl, value)
    return None


def _ground_target(target: _slot_policy.SlotSettings | None, value: object, speech: str,
                   lang: str) -> str | None:
    """A correction's value with the target's rules: digits grounded; a choice kept as said
    (:func:`~owlbank.slots.dialog.decide` checks it is one of the caller's options); a yes/no
    validated. ``None`` when it fails or the target is unknown."""
    if target is None or not isinstance(value, str):
        return None
    t = target.decl.slot_type
    if t == "digits":
        return grounding.ground_digits(target, value, speech, lang)
    if t == "choice":
        return value
    if t == "yes_no":
        return validate(target.decl, value)
    return None


def resolve(
    slot: _slot_policy.SlotSettings,
    *,
    digits: str | None,
    speech: str | None,
    confidence: float | None,
    lang: str,
    mode: _slot_policy.Mode,
    ctx: ResolveContext,
    resolver: Resolver | None,
    today: date | None = None,
    min_speech_confidence: float = 0.0,
    flow: str | None = None,
    slots: Mapping[str, _slot_policy.SlotSettings] | None = None,
    finished_on_key: str | None = None,
) -> Resolution:
    """Resolve one Gather result against ``slot``.

    ``ctx.labels`` maps catalog keys (option ``label_key``s and the slot's ``label_key``) to their
    text in ``lang``; ``ctx.commands`` the ``command.*`` keys to their phrases. ``mode`` is the
    effective mode now (frozen at flow start and restricted by the running ceiling); in
    ``deterministic`` mode, or with ``resolver=None``, the resolver is never touched. ``slots``
    gives the settings of the collected slots, for grounding a correction with its target's rules. ``flow`` is
    for the resolver's logs only. A deterministic speech parse is accepted only
    when ``confidence`` is unknown or at least ``min_speech_confidence``. ``finished_on_key`` is
    Twilio's ``FinishedOnKey``: ``#`` with no other key and no speech declines a ``declinable``
    slot in every mode; an empty or absent one (a timeout) never does.
    Never raises for a resolver failure."""
    today = today or date.today()
    decl = slot.decl
    said = speech if speech is not None and speech.strip() else None
    pending = ctx.pending
    yes_no_pending = pending is not None and pending.kind in YES_NO_PENDING
    known, caller = slot_options(slot, ctx)

    out = _Out(slot, said)

    # 0. ``#`` alone declines, but not at a yes/no follow-up such as a read-back, where
    # callers press ``#`` out of habit (asked again). A lone ``*`` is no decline.
    if (slot.declinable and not yes_no_pending and finished_on_key == DECLINE_FINISH_KEY
            and said is None and not (digits or "").strip().rstrip("#").strip()):
        return out("decline", "dtmf")

    # 1. DTMF
    if digits:
        key = digits.strip().rstrip("#")
        if yes_no_pending and key in YES_NO_DTMF:
            return out("exact", "dtmf", reply=_yes_no(YES_NO_DTMF[key]), reply_deterministic=True)
        value = _dtmf_value(slot, digits, caller, today)
        if value is not None:
            return out("exact", "dtmf", value)
        if said is None:
            return Resolution(match="none", source="dtmf")
    if said is None:
        return Resolution(match="none", source="none")

    # 2. Speech, deterministic
    norm = normalize(said, lang)
    label = ctx.labels.get(decl.label_key or "")
    if confidence is None or confidence >= min_speech_confidence:
        if yes_no_pending:
            said_yes_no = yes_no_speech(norm, lang)
            if said_yes_no is not None:
                return out("exact", "speech", reply=_yes_no(said_yes_no), reply_deterministic=True)
        if decl.slot_type == "choice" and slot.options:
            opt, source, where = option_where(known, said, lang, label=label)
            if opt is not None and source is not None:
                src: Source = "alias" if source == "alias" else "speech"
                return out("exact", src, opt.value, span=where.span)
        else:
            # The phone slot reads a North American number (a leading 1 is the country code;
            # a ten-digit reading starting with 0 or 1, or followed by another number, is not
            # a value). Every other digits slot reads its digits as they are.
            parsed = parse_speech_located(decl, said, lang, labels=ctx.labels, today=today,
                                          other_labels=_other_labels(slot, ctx),
                                          phone=slot.name == PHONE_SLOT)
            value = validate(decl, parsed.value, today=today)
            if value is not None:
                return out("exact", "speech", value, runs=parsed.value_runs, span=parsed.span)
        # An ordinal reference to the options the question just listed ("the second one").
        if decl.slot_type == "choice" and pending is not None and pending.listed:
            hit = ordinal_where(said, lang, len(pending.listed), label=label)
            if hit is not None:
                return out("exact", "ordinal", pending.listed[hit[0]], span=hit[1])

    # 2c. A digit answer cut short ("one" when four are needed): asked again at once, no model.
    if decl.slot_type == "digits" and short_digit_answer(decl, norm, lang, label):
        return out("none", "short_digits")

    # 3. Universal commands (whole utterance only)
    command = match_command(said, lang, ctx.commands)
    if command is not None:
        return out(command, "command")

    # 4. The resolver: never in deterministic mode
    if mode == "deterministic" or resolver is None:
        return out("none", "none")
    try:
        seen = replace(ctx, known_options=known, caller_options=caller)  # The slot's options
        req = build_request(slot, seen, lang=lang, utterance=said, label=label)
        o = resolver.resolve(req, budget_ms=ctx.budget_ms, generative=mode == "generative",
                             flow=flow)
    except Exception as exc:  # a resolver failure is never a call failure
        log_caught(log, "resolver raised", exc, operation="resolve", decision="fallback_none")
        return out("none", "model", model="error")
    return _from_model(slot, o, said=said, norm=norm, lang=lang, ctx=ctx, known=known,
                       caller=caller, yes_no_pending=yes_no_pending, slots=slots or {}, out=out)


def _yes_no(v: str) -> Literal["yes", "no"]:
    return "yes" if v == "yes" else "no"


@dataclass(frozen=True, slots=True)
class _Out:
    """Builds a :class:`Resolution` with this turn's ``utterance_for_ai``."""

    slot: _slot_policy.SlotSettings
    said: str | None

    def __call__(self, match: Match, source: Source, value: str | None = None, *,
                 runs: Sequence[int] | None = None, span: tuple[int, int] | None = None,
                 model: ModelOutcome = "not_called", reply: Literal["yes", "no"] | None = None,
                 reply_deterministic: bool = False, candidates: tuple[str, ...] = (),
                 target: str | None = None) -> Resolution:
        return Resolution(match=match, source=source, value=value, model=model, reply=reply,
                          reply_deterministic=reply_deterministic, candidates=candidates,
                          target=target,
                          utterance_for_ai=utterance_for_ai(self.slot, self.said, value,
                                                            value_runs=runs, span=span))


def _from_model(slot: _slot_policy.SlotSettings, o: ResolverOutput, *, said: str, norm: str,
                lang: str, ctx: ResolveContext, known: tuple[OptionView, ...],
                caller: tuple[str, ...], yes_no_pending: bool,
                slots: Mapping[str, _slot_policy.SlotSettings], out: _Out) -> Resolution:
    """The resolver's output, grounded and with answer first."""
    if not isinstance(o, ResolverOutput):
        return out("none", "model", model="malformed")
    if o.outcome != "ok":
        return out("none", "model", model=o.outcome)
    kind = o.kind
    t = slot.decl.slot_type
    collected = {c.name for c in ctx.collected}
    reply: Literal["yes", "no"] | None = o.reply if yes_no_pending else None
    if yes_no_pending and reply is None and t != "yes_no" and o.value in ("yes", "no"):
        reply = "yes" if o.value == "yes" else "no"   # a follow-up's yes/no given as the value
    names_collected = kind == "correction" and o.slot in collected and o.slot != slot.name

    # Answer first: a value that grounds for this slot is the answer, whatever the kind.
    value = None if names_collected else _ground(slot, o.value, said, lang, known)
    if value is not None:
        match: Match = "exact" if kind == "exact" else "inferred"
        span = None
        if t == "choice":
            view = next(v for v in known if v.value == value)
            if match == "exact" and not grounding.named_in(norm, view, lang):
                match = "inferred"                       # The option wasn't named
            span = option_where((view,), said, lang)[2].span
        runs = grounding.grounded_runs(value, said) if t == "digits" else None
        return out(match, "model", value, runs=runs, span=span, model="ok", reply=reply)
    if reply is not None:
        # A model's yes or no to a follow-up: never deterministic.
        return out("inferred", "model", model="ok", reply=reply)
    if kind == "correction":
        return _correction(o, said=said, lang=lang, slots=slots, out=out,
                           target=o.slot if names_collected else None)
    if kind in ("exact", "inferred"):
        return out("none", "model", model="ungrounded" if o.value is not None else "ok")
    if kind == "ambiguous":
        if t == "choice":
            cands = grounding.ground_candidates(list(o.candidates), caller=caller)
            if cands is None:
                return out("none", "model", model="ungrounded")
            return out("ambiguous", "model", model="ok", candidates=cands)
        if t == "digits":
            return out("ambiguous", "model", model="ok")      # never listed
        return out("none", "model", model="ungrounded")
    if kind in ("intent_switch", "repeat", "escalate", "decline"):
        return out(kind, "model", model="ok")     # decline on a non-declinable slot: dialog
    return out("none", "model", model="ok")


def _correction(o: ResolverOutput, *, said: str, lang: str,
                slots: Mapping[str, _slot_policy.SlotSettings], out: _Out,
                target: str | None) -> Resolution:
    """A correction (6.1-6.4). Of a collected slot: its value grounded with the target's rules
    (``None`` when it fails, so the target is re-asked). Of an unasked or unknown slot: no
    target, which the dialog treats as ``none``. When the target is sensitive, the AI text
    gets its placeholder too: the words hold its value."""
    ts = slots.get(target) if target is not None else None
    value = _ground_target(ts, o.value, said, lang)
    res = out("correction", "model", model="ok", target=target)
    res = replace(res, value=value)
    if ts is not None and ts.sensitive and res.utterance_for_ai is not None:
        unchanged = res.utterance_for_ai == said
        runs = (grounding.grounded_runs(value, said)
                if value and unchanged and ts.decl.slot_type == "digits" else None)
        res = replace(res, utterance_for_ai=_sensitive.placeholder_text(
            res.utterance_for_ai, ts, value, value_runs=runs))
    return res
