"""Slot answers in the front door: resolve, decide, execute.

For every answer of ``login``, ``stepup`` and the confirm flow's ``yes_no``:

1. **The mode** is the one frozen at flow start (``flow["modes"]``) restricted by the
   running ceiling (``FrontDoorConfig.slot_mode_ceiling``). ``deterministic`` passes no
   resolver to the engine, so Bedrock is never touched; neither does a keypad-only
   slot, whose follow-ups are keypad menus.
2. **The context** (:class:`~owlbank.slots.ResolveContext`) is built from the session: the question
   as the caller heard it, the names, types and labels of the slots collected on the call (never
   their values), the pending follow-up question and the budget.
3. :func:`~owlbank.slots.resolve` grounds whatever the resolver proposes; a digit value is grounded
   before :func:`~owlbank.slots.decide` sees it. Keypad reminders stay here, before
   ``decide``.
4. :func:`~owlbank.slots.decide` gives the action and the next dialog state (tries, the pending
   question with its value sealed, the earlier-flow strikes), saved in the session.
5. :meth:`SlotFlow.execute` runs the action over the front door's ``_verify``, ``_reprompt``,
   ``_retry``, ``_off_script`` and ``_give_up``. Every follow-up Gather uses the asked slot's
   settings (``keypad_only``, ``barge_in`` with Path B).

**Corrections.** ``decide`` gives
:class:`ApplyCorrection` only for a slot collected in this flow, not yet sent to the auth service,
and with no API call in the flow since it was collected: every API call the front door makes
in a flow (``precheck``, ``verify``, ``send_otp``, ``make_payment`` and the others) goes through
:meth:`SlotFlow.api_call`, which counts it in ``flow["api_calls"]``; each collected slot records the
count at its first collection in the flow (``at_api_call``), and a slot whose count is behind is
handed to ``decide`` as ``verified``, so it is handled as a finished flow's slot. The front
door never re-runs an API call because an earlier value changed. In ``login`` and ``stepup``
such a value lives in one place, ``flow["held"]`` (``_verify`` holds it, sealed), and the
correction replaces it there, sealed again. A sensitive value goes through
``sensitive.remember``; the old one stays registered. The caller then hears ``slot.corrected``
(the digits) or ``slot.corrected_sensitive`` (the label, never a value) and the interrupted
slot's question. A correction whose value fails is re-asked once for the corrected slot
(``correction_reask``: the corrected slot's ``clarify.*`` and Gather), and then the
interrupted slot resumes. A correction of a slot already sent to the auth service, or of an
earlier flow's slot, is re-asked and twice is an intent switch: it never calls
``verify`` and changes no level, attempt, step-up or risk counter. No value is ever stored
for a slot that wasn't asked, and no try is ever given back. Anything the design
doesn't cover fails safe: the interrupted slot is asked again and nothing is stored.

**The card in the confirm flow.** The caller's cards
come from the verified customer's record and are frozen in ``flow["cards"]``; they are the card
slot's caller's options. A ``to`` the AI proposed that isn't one of them is never mapped to
another card: :meth:`SlotFlow.ask_card` asks ``prompt.card`` first, and its answer goes
through ``decide`` (confirm, disambiguate, not available, "no, I meant Y"). A card is selected only
when the caller named it, pressed its key, or said a deterministic yes to a question naming it;
it then replaces ``pending_tool.args.to`` and the payment is read back. During the
read-back a card correction replaces it again, says ``slot.corrected`` and reads back again; a
model's card the caller didn't name is not applied (the card is re-asked once), and a change
to anything else the AI proposed is an intent switch. Follow-up Gathers use ``card``'s
settings: a keypad-only card hears its keypad menu (``slot.keypad_choice``).

**Generative wording.** Only on a slot whose mode for this answer is
``generative`` (a non-sensitive choice slot with the ceiling and ``generative_enabled`` allowing it;
checked again here, fail closed) the resolver is called with ``generative=True``, and only
for :class:`ConfirmOption`, :class:`Disambiguate` and :class:`NotAvailable` is its ``say`` spoken:
rendered (:func:`~owlbank.slots.generative.render`) when
:func:`~owlbank.slots.generative.check` accepts it with ``allowed`` = the confirmed option, the
candidates, or the named option and the caller's options. Otherwise the catalog template is
spoken, with no second model call, and ``generative_fallback`` is logged with the ``reason`` (one of
the check's, which judges the healed text, 18.5; ``budget`` when the resolver ran out of time),
never the text. The response holding an accepted text is parsed as XML once before it is
returned (:meth:`FrontDoor._reprompt`); if it doesn't parse, the template is spoken instead, with
``generative_fallback`` ``reason: twiml_invalid`` at ERROR. The ``prompt``
journal event of a generated text has ``generated: true``. Repeats of a pending question,
digit read-backs, re-asks and every other sentence are catalog templates.

**The card question**. ``prompt.card`` names the caller's cards (``{options}``: each
spoken name with "your", joined like ``slot.disambiguate``), in policy order. Every question that
lists options records the order it read them, for ordinals ("the second one"): a pending
follow-up in ``Pending.listed`` (a disambiguation's candidates, not available with several cards),
and a question with no follow-up pending (``prompt.card``, the clarify that lists the cards) in
``flow["listed"]``, which the next answer reads once. A generated question's order isn't known, so
it lists nothing (no ordinal; fail safe).

**Early finish**. A choice question's partial result that :func:`early_choice` finds
complete is claimed by the front door; the redirect carries the option value as
:data:`EARLY_VALUE`, and the answer is ``match: exact, source: early`` with no parse or model call.

Logs and journal: one ``factor_answer`` log line and one ``slot_resolution`` journal event per
answer: names, modes, match, source, the model outcome, the try count and, for a
non-sensitive slot only, the value (in the log line, only for a slot that isn't a login factor
either). Never the caller's words or the resolver's text. The
``factor_answer`` line is written when the request ends, with its ``timings_ms``. An applied
correction writes its own ``slot_resolution`` for the slot it changed (``correction: true``,
``correction_of`` = the slot being asked), so the original answer's event and the correction's are
both in the journal.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from owlbank import obs
from owlbank.pii import ON_FAILURE_WITHHOLD, SURFACE_SPEECH_LOG
from owlbank.sensitive import placeholder_text
from owlbank.slot_policy import Mode, SlotSettings, gather_input, is_mode
from owlbank.slots import (
    CollectedSlot,
    OptionView,
    Pending,
    Resolution,
    ResolveContext,
    Resolver,
    ResolverOutput,
    ResolverRequest,
    generative,
    grounding,
    make_context,
    normalize,
    resolve,
    slot_options,
)
from owlbank.slots.dialog import (
    Accept,
    Action,
    ApplyCorrection,
    Command,
    ConfirmOption,
    Disambiguate,
    GiveUp,
    IntentSwitch,
    NotAvailable,
    ReadBackDigits,
    Reask,
    Turn,
    decide,
)
from owlbank.slots.modes import freeze, restrict, slot_flows
from owlbank.slots.normalize import lang_family
from owlbank.telemetry import frontdoor as telemetry
from owlbank.token_seal import UnsealError
from owlbank.tools_contract import CARD_SLOT, PHONE_SLOT

from .speech import digits_words

if TYPE_CHECKING:
    from owlbank.callstate import CallState
    from owlbank.envelope import Spoken

    from . import http
    from .app import FrontDoor
    from .state import Session

log = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Follow-up questions answered with yes or no; their Gather is a yes/no Gather of the asked slot.
_YES_NO_FOLLOW_UPS = frozenset({"confirm_digits", "confirm_option", "not_available_one"})
#: Follow-up questions that list options: their order is ``Pending.listed``, and their
#: Gather may finish early.
LISTING_KINDS = frozenset({"disambiguate", "not_available_many"})
#: The early redirect's option value: set only by the front door from the sealed claim;
#: a webhook parameter of this name is dropped before any handler sees it.
EARLY_VALUE = "OwlBankEarlyValue"
#: The possessive before each of the caller's cards in ``prompt.card``, per language family;
#: another language reads the names alone.
_YOUR = {"en": "your", "es": "su"}
#: ``slot_resolution.asked`` per pending kind.
_ASKED = {"confirm_digits": "confirm_digits", "confirm_option": "confirm_option",
          "disambiguate": "disambiguate", "not_available_one": "not_available",
          "not_available_many": "not_available"}


# ------------------------------------------------------------------------- flow state


def start_flow(fd: FrontDoor, sess: Session, kind: str, **fields: Any) -> dict[str, Any]:
    """A new flow of ``kind`` (``login``, ``stepup``, ``confirm``) in ``sess.flow``, with its slots'
    modes frozen now: every slot that can be asked in the flow. The caller saves."""
    sess.flow_seq = int(sess.flow_seq) + 1
    policy = fd.d.policy
    names = [n for n in policy.slots() if kind in slot_flows(policy.slot(n))]
    frozen = freeze(policy, kind, names, fd.cfg.slot_mode_ceiling)
    flow: dict[str, Any] = {"kind": kind, "factor": None, "tries": {}, "repeats": 0, "spoken": [],
                            "modes": {n: d.to_doc() for n, d in frozen.items()},
                            "pending": None, "earlier_strikes": 0, "seq": sess.flow_seq,
                            "api_calls": 0}
    flow.update(fields)
    sess.flow = flow
    return flow


#: Stands in for a placeholder already in a login factor's text while its number runs are replaced.
_KEPT = "\ue000"


def hide_interrupted(res: Resolution, *, asked: SlotSettings,
                     answered: SlotSettings) -> Resolution:
    """``res`` with its ``utterance_for_ai`` also placeholdered for ``asked`` when the caller is
    answering another slot (a correction's re-ask or follow-up) while a sensitive slot is
    asked (9.2(a)). The engine builds the text for the slot it resolves, so a
    non-sensitive re-ask keeps the caller's words as transcribed; but the interrupted sensitive
    slot is still being asked, its value isn't known yet (no exact-value scrub, 9.2(b)), and the
    caller may well say it ("yes my last four are 1234"). Every number run that could hold it is
    replaced, single digits too, whatever the resolver labelled the turn (fail closed): this text is what the speech
    log, the agent and its call log get. A turn that answers ``asked`` itself is unchanged."""
    if (not asked.sensitive or answered.name == asked.name
            or res.utterance_for_ai is None):
        return res
    return replace(res, utterance_for_ai=placeholder_text(res.utterance_for_ai, asked, None, every_run=True))


def tries_of(flow: Mapping[str, Any], name: str) -> int:
    raw = flow.get("tries")
    n = raw.get(name, 0) if isinstance(raw, Mapping) else 0
    return int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) else 0


def set_tries(flow: dict[str, Any], name: str, n: int) -> None:
    raw = flow.get("tries")
    tries = {k: int(v) for k, v in raw.items()} if isinstance(raw, Mapping) else {}
    tries[name] = n
    flow["tries"] = tries


def mode_doc(flow: Mapping[str, Any], name: str) -> dict[str, Any]:
    """The slot's frozen decision. A slot the flow didn't freeze (it can't be asked in this
    flow) runs ``deterministic`` (fail closed)."""
    modes = flow.get("modes")
    doc = modes.get(name) if isinstance(modes, Mapping) else None
    if isinstance(doc, Mapping) and is_mode(doc.get("effective")):
        return dict(doc)
    obs.emit(log, logging.WARNING, "slot_mode_not_frozen", slot=name, decision="deterministic")
    return {"configured": "deterministic", "source": "invalid", "ceiling": "deterministic",
            "effective": "deterministic", "capped": None}


def mode_now(fd: FrontDoor, flow: Mapping[str, Any], name: str) -> tuple[Mode, dict[str, Any]]:
    """The mode for this answer: the frozen effective mode restricted by the running ceiling
    ."""
    doc = mode_doc(flow, name)
    return restrict(doc["effective"], fd.cfg.slot_mode_ceiling), doc


def api_calls(flow: Mapping[str, Any] | None) -> int:
    """``flow["api_calls"]``: the API calls the front door has made in the flow."""
    n = (flow or {}).get("api_calls", 0)
    return int(n) if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else 0


def _at_api_call(row: Mapping[str, Any]) -> int | None:
    n = row.get("at_api_call")
    return int(n) if isinstance(n, int) and not isinstance(n, bool) else None


def collect(sess: Session, name: str, slot_type: str, *, verified: bool) -> None:
    """Record that ``name`` was collected in the current flow (``verified``: sent to the auth
    service), with the flow's API-call count (``at_api_call``). A slot collected
    again in the same flow keeps its first count and its ``verified`` (fail safe: collecting it
    again never makes it correctable again). Names only."""
    seq = int((sess.flow or {}).get("seq", sess.flow_seq))
    rows = [r for r in sess.collected if r.get("name") != name]
    old = next((r for r in sess.collected if r.get("name") == name), None)
    same = old is not None and int(old.get("flow_seq", -1)) == seq
    was = bool(old.get("verified")) if old is not None and same else False
    at = _at_api_call(old) if old is not None and same else None
    rows.append({"name": name, "slot_type": slot_type, "flow_seq": seq, "verified": verified or was,
                 "at_api_call": api_calls(sess.flow) if at is None else min(at, api_calls(sess.flow))})
    sess.collected = rows


def fresh(sess: Session, name: str) -> bool:
    """18.6: ``name`` was collected in the current flow and no API call has happened in the flow
    since (its ``at_api_call`` equals ``flow["api_calls"]``). A row without a count isn't fresh
    (fail safe)."""
    if sess.flow is None:
        return False
    seq = int(sess.flow.get("seq", sess.flow_seq))
    row = next((r for r in sess.collected if r.get("name") == name), None)
    return (row is not None and int(row.get("flow_seq", -1)) == seq
            and _at_api_call(row) == api_calls(sess.flow))


def _prompt_key(slot: SlotSettings, keypad_only: bool, catalog: Any, lang: str,
                base: str | None = None) -> str:
    """The key of the slot's own question: its ``.keypad`` twin when the slot is effectively
    keypad-only (``gather_input``) and the twin exists in ``lang``, else the base key (the slot's
    ``prompt_key``, ``factor.<name>``, or ``base``)."""
    key = base or slot.decl.prompt_key or f"factor.{slot.name}"
    twin = f"{key}.keypad"
    return twin if keypad_only and catalog.has(twin, lang) else key


class _Tap:
    """The front door's resolver for one answer: keeps the output, so the generative wording
    (``say``, ``say_lang``) reaches :meth:`SlotFlow.wording` (the engine's resolution doesn't
    carry it). Exactly one call per answer; nothing is called again on a fallback."""

    def __init__(self, inner: Resolver) -> None:
        self._inner = inner
        self.output: ResolverOutput | None = None

    def resolve(self, req: ResolverRequest, *, budget_ms: int, generative: bool,
                flow: str | None = None) -> ResolverOutput:
        out = self._inner.resolve(req, budget_ms=budget_ms, generative=generative, flow=flow)
        self.output = out
        return out


def _confidence(params: Mapping[str, str]) -> float | None:
    try:
        return float(params["Confidence"]) if params.get("Confidence") else None
    except ValueError:
        log.debug("speech confidence is not a number", extra={
            "decision": "ignored_because_unparseable_confidence"})
        return None


# ------------------------------------------------------------------------------ SlotFlow


#: The ``speech`` of a turn whose redaction failed (never the unredacted text).
SPEECH_WITHHELD = "[withheld]"


class SlotFlow:
    """Resolves, decides and executes slot answers for :class:`~lambdas.frontdoor.app.FrontDoor`."""

    def __init__(self, fd: FrontDoor) -> None:
        self.fd = fd
        #: ``factor_answer`` lines of the request in progress, written by :meth:`flush_answers`.
        self._answers: list[dict[str, Any]] = []

    def api_call(self, sess: Session, op: Callable[[], _T]) -> _T:
        """Run ``op``, an API call the front door makes (auth, bank tools or any other), counting it
        in the current flow's ``flow["api_calls"]`` first: from then on no slot collected
        before it is correctable. Counted before the call, so a call that fails or raises still
        counts (fail safe). Outside a flow nothing is counted. The caller saves the session."""
        if sess.flow is not None:
            sess.flow["api_calls"] = api_calls(sess.flow) + 1
        return op()

    def flush_answers(self, timings_ms: Mapping[str, int]) -> None:
        """Write the request's ``factor_answer`` lines with its ``timings_ms`` (18.12; names and
        whole milliseconds only)."""
        lines, self._answers = self._answers, []
        for fields in lines:
            obs.emit(log, logging.INFO, "factor_answer", **fields, timings_ms=dict(timings_ms))
            if telemetry.installed():  # the same fields on the webhook span
                span_fields = {k: v for k, v in fields.items() if k not in ("value", "speech", "call_sid")}
                span_fields.update({f"timing.{name}_ms": ms for name, ms in timings_ms.items()})
                telemetry.add_fields(**span_fields)   # never the value or speech

    # -- context

    def collected(self, sess: Session, lang: str) -> tuple[CollectedSlot, ...]:
        """The collected slots, ``current`` in this flow, else ``earlier`` (no values). A
        current slot collected before the flow's last API call is given as ``verified``, so
        ``decide`` handles a correction of it as for a finished flow's slot; the resolver
        never sees ``verified``. A choice slot carries its options in ``lang`` and the caller's
        options among them (:meth:`caller_views`: the card in the confirm flow lists the cards on
        file), so a correction of it can be resolved; never the option chosen."""
        fd = self.fd
        seq = int((sess.flow or {}).get("seq", sess.flow_seq))
        out: list[CollectedSlot] = []
        for row in sess.collected:
            name = row.get("name")
            if not isinstance(name, str):
                continue
            try:
                s = fd.d.policy.slot(name)
            except KeyError:
                continue
            key = s.decl.label_key
            label = fd.d.catalog.text(key, lang) if key and key in fd.d.catalog.keys else name
            scope: Literal["current", "earlier"] = (
                "current" if int(row.get("flow_seq", -1)) == seq else "earlier")
            locked = scope == "current" and not fresh(sess, name)
            options: tuple[OptionView, ...] = ()
            caller: tuple[str, ...] = ()
            if s.decl.slot_type == "choice" and s.options:
                options = self.option_views(s, lang)
                caller = tuple(v.value for v in self.caller_views(sess, s, lang))
            out.append(CollectedSlot(name=name, slot_type=s.decl.slot_type, label=label,
                                     scope=scope, verified=bool(row.get("verified")) or locked,
                                     options=options, caller_options=caller))
        return tuple(out)

    def option_views(self, slot: SlotSettings, lang: str) -> tuple[OptionView, ...]:
        """A choice slot's policy options in ``lang``: catalog spoken name, aliases."""
        cat = self.fd.d.catalog
        out = []
        for o in slot.options:
            key = f"option.{slot.name}.{o.value}"
            name = cat.text(key, lang) if key in cat.keys else o.value
            aliases = o.aliases.get(lang) or o.aliases.get(lang.split("-")[0]) or ()
            out.append(OptionView(value=o.value, name=name, aliases=tuple(aliases)))
        return tuple(out)

    def caller_views(self, sess: Session, slot: SlotSettings, lang: str) -> tuple[OptionView, ...]:
        """The caller's options as views (the card in the confirm flow: the cards on file)."""
        views = self.option_views(slot, lang)
        listed = self.listed(sess, slot)
        return views if listed is None else tuple(v for v in views if v.value in listed)

    @staticmethod
    def listed_now(flow: Mapping[str, Any], slot: SlotSettings) -> tuple[str, ...]:
        """The order the question being answered read ``slot``'s options in: the pending
        follow-up's ``listed``, else ``flow["listed"]`` (a question with nothing pending). Empty
        when it listed nothing, or the record is unreadable (fail safe: no ordinal)."""
        raw = flow.get("pending")
        if isinstance(raw, Mapping):
            return _valid_listed(raw.get("listed") if raw.get("slot") == slot.name else None)
        return _valid_listed(_record_values(flow, slot))

    def context(self, slot: SlotSettings, lang: str, *, question: str, pending: Pending | None,
                collected: tuple[CollectedSlot, ...],
                caller: Sequence[str] | None = None) -> ResolveContext:
        """``caller``: a choice slot's caller's options (the card: the cards on file); ``None``
        means every known option."""
        fd = self.fd
        decl = slot.decl
        keys = [k for k in [decl.label_key, *(o["label_key"] for o in decl.options())] if k]
        labels = {k: fd.d.catalog.text(k, lang) for k in keys if k in fd.d.catalog.keys}
        known = self.option_views(slot, lang)
        return make_context(question_text=question, labels=labels, commands=fd.d.catalog.commands(lang),
                            known_options=known, caller_options=caller, collected=collected,
                            pending=pending,
                            budget_ms=fd.d.policy.slot_resolution.resolver_budget_ms)

    def settings(self, collected: tuple[CollectedSlot, ...]) -> dict[str, SlotSettings]:
        out = {}
        for c in collected:
            try:
                out[c.name] = self.fd.d.policy.slot(c.name)
            except KeyError:
                continue
        return out

    # -- pending (sealed value)

    def pending(self, st: CallState, flow: Mapping[str, Any], slot: SlotSettings,
                collected: tuple[CollectedSlot, ...] = ()) -> Pending | None:
        raw = flow.get("pending")
        if not isinstance(raw, Mapping):
            return None
        try:
            sealed = raw.get("sealed_value")
            value = (self.fd.d.sealer.unseal(sealed, call_sid=st.call_sid)
                     if isinstance(sealed, str) else None)
            p = Pending(kind=raw["kind"], slot=str(raw["slot"]), option=raw.get("option"),
                        candidates=tuple(raw.get("candidates") or ()), sealed_value=value,
                        listed=tuple(raw.get("listed") or ()))
        except (KeyError, TypeError, ValueError, UnsealError) as exc:
            obs.log_caught(log, "pending question unreadable", exc, operation="load_pending",
                           decision="asked_again")
            return None
        if p.slot != slot.name:
            # The corrected slot's re-ask or a follow-up of it (its read-back, an option
            # question): only for another slot still correctable in this flow; anything
            # else is dropped and the slot being asked is answered (fail safe).
            row = next((c for c in collected if c.name == p.slot), None)
            ok = row is not None and row.scope == "current" and not row.verified
            return p if ok else None
        if p.kind == "correction_reask":
            return None
        return p

    def store_pending(self, st: CallState, p: Pending | None) -> dict[str, Any] | None:
        if p is None:
            return None
        sealed = (self.fd.d.sealer.seal(p.sealed_value, call_sid=st.call_sid)
                  if p.sealed_value is not None else None)
        # A question listing options reads them in its candidates' order (the dialog builds
        # its follow-ups without the order; it is copied here).
        listed = p.listed or (p.candidates if p.kind in LISTING_KINDS else ())
        return {"kind": p.kind, "slot": p.slot, "option": p.option,
                "candidates": list(p.candidates), "sealed_value": sealed, "listed": list(listed)}

    # -- one answer

    def _resolve(self, req: http.TwilioRequest, st: CallState, sess: Session, slot: SlotSettings,
                 *, question: str, pending: Pending | None, mode: Mode, keypad_only: bool,
                 caller: Sequence[str] | None = None,
                 ) -> tuple[Resolution, ResolveContext, tuple[CollectedSlot, ...],
                            ResolverOutput | None]:
        """The answer resolved, its context, the collected slots and the resolver's output
        (``None`` when it wasn't called). A slot that may not be generative now is resolved as
        ``assisted`` (fail closed), so the resolver never gets ``generative=True`` for it."""
        with self.fd._span("resolve"):
            return self._resolve_timed(req, st, sess, slot, question=question, pending=pending,
                                       mode=mode, keypad_only=keypad_only, caller=caller)

    def _resolve_timed(self, req: http.TwilioRequest, st: CallState, sess: Session,
                       slot: SlotSettings, *, question: str, pending: Pending | None, mode: Mode,
                       keypad_only: bool, caller: Sequence[str] | None = None,
                       ) -> tuple[Resolution, ResolveContext, tuple[CollectedSlot, ...],
                                  ResolverOutput | None]:
        fd = self.fd
        p = req.params
        lang = st.language
        flow = sess.flow if sess.flow is not None else {}
        # The order a question with nothing pending listed the options in, read once.
        listed = (_valid_listed(_record_values(flow, slot))
                  if pending is None or pending.kind == "correction_reask" else ())
        flow.pop("listed", None)
        if mode == "generative" and not _may_generate(slot):
            mode = "assisted"
        coll = self.collected(sess, lang)
        ctx = self.context(slot, lang, question=question, pending=pending, collected=coll,
                           caller=caller)
        flow_kind = flow.get("kind")
        today = date.fromtimestamp(fd._now())
        settings = self.settings(coll)

        def run(c: ResolveContext, m: Mode, r: Resolver | None) -> Resolution:
            return resolve(slot, digits=p.get("Digits") or None,
                           speech=p.get("SpeechResult") or None, confidence=_confidence(p),
                           lang=lang, mode=m, ctx=c, resolver=r, today=today,
                           flow=flow_kind if isinstance(flow_kind, str) else None, slots=settings,
                           finished_on_key=p.get("FinishedOnKey"))

        res: Resolution | None = None
        if EARLY_VALUE in p:
            # The early redirect. The partial was an exact deterministic match of this
            # option (early_choice); nothing is parsed again and no model is called.
            early = p.get(EARLY_VALUE)
            known = {o.value for o in slot.options}
            res = (Resolution(match="exact", source="speech", value=early)
                   if slot.decl.slot_type == "choice" and early in known
                   else Resolution(match="none", source="none"))
        elif listed and slot.decl.slot_type == "choice":
            # The ordinal stage needs the listed order; a question with nothing pending has no
            # follow-up to carry it, so the deterministic parse runs once with it (no model). Only
            # an ordinal is taken from it: names, keys and the rest are the normal parse's.
            probe_pending = Pending("not_available_many", slot.name, candidates=listed,
                                    listed=listed)
            probe_ctx = self.context(slot, lang, question=question, pending=probe_pending,
                                     collected=coll, caller=caller)
            probe = run(probe_ctx, "deterministic", None)
            if probe.source == "ordinal" and probe.value is not None:
                res = probe
        out: ResolverOutput | None = None
        if res is None:
            no_model = mode == "deterministic" or keypad_only
            tap = None if no_model or fd.d.resolver is None else _Tap(fd.d.resolver)
            res = run(ctx, mode, tap)
            out = tap.output if tap is not None else None
        if mode == "generative" and out is not None and out.outcome == "timeout":
            # Out of time; whatever is asked next is the catalog's (no second call).
            self._fallback(st, flow_kind, slot, "budget")
        if res.value is not None and res.match != "correction" and slot.sensitive:
            fd._remember(st.call_sid, slot, res.value)
        if res.match == "correction" and res.value is not None and res.target:
            target = self.settings(coll).get(res.target)
            if target is not None and target.sensitive:
                fd._remember(st.call_sid, target, res.value)
        return res, ctx, coll, out

    def _turn(self, slot: SlotSettings, mode: Mode, flow: Mapping[str, Any], pending: Pending | None,
              coll: tuple[CollectedSlot, ...], owned: frozenset[str],
              caller_options: Mapping[str, tuple[str, ...]] | None = None,
              named: frozenset[str] = frozenset()) -> Turn:
        raw = flow.get("tries")
        tries = ({k: int(v) for k, v in raw.items() if isinstance(k, str)}
                 if isinstance(raw, Mapping) else {})
        strikes = flow.get("earlier_strikes", 0)
        settings = self.settings(coll)
        settings.pop(slot.name, None)
        return Turn(slot=slot, mode=mode, tries=tries, pending=pending, collected=coll,
                    earlier_strikes=int(strikes) if isinstance(strikes, (int, float)) else 0,
                    owned=owned, settings=settings, caller_options=dict(caller_options or {}),
                    named=named)

    def _decide(self, st: CallState, flow: dict[str, Any], turn: Turn, res: Resolution,
                ctx: ResolveContext) -> tuple[Action, Turn]:
        known, caller = slot_options(turn.answered(), ctx)
        action, nxt = decide(turn, res, known=known, caller=caller)
        flow["tries"] = _no_decrease(flow, nxt.tries)
        flow["pending"] = self.store_pending(st, nxt.pending)
        flow["earlier_strikes"] = nxt.earlier_strikes
        return action, nxt

    def _log(self, req: http.TwilioRequest, st: CallState, flow: Mapping[str, Any],
             slot: SlotSettings, res: Resolution, doc: Mapping[str, Any], mode: Mode, *,
             tries: int, pending: Pending | None, action: Action | None,
             gather_ms: int | None = None) -> None:
        """``factor_answer`` (log; written with the request's ``timings_ms``
        when it ends) and ``slot_resolution`` (journal). Shape, names and modes; the
        value only for a non-sensitive slot, and in the log line only when the
        slot isn't a login factor either. An early finish is ``source: early``."""
        p = req.params
        logged_source = "early" if EARLY_VALUE in p else res.source
        source = "dtmf" if res.source == "dtmf" else (
            "speech" if res.source in ("speech", "alias") else "none")
        if res.value is not None or res.reply is not None:
            kind = "value"
        elif res.match in ("none", "ambiguous", "inferred", "exact", "correction"):
            kind = "unclear"
        else:
            kind = res.match
        flow_kind = flow.get("kind")
        modes = {"configured_mode": doc.get("configured"), "effective_mode": mode,
                 "mode_source": doc.get("source"), "ceiling": self.fd.cfg.slot_mode_ceiling,
                 "capped": doc.get("capped")}
        value = {"value": res.value} if res.value is not None and not slot.sensitive \
            and res.match != "correction" else {}
        logged_value = {} if slot.is_factor else value  # a login factor's never reaches the log
        fields: dict[str, Any] = {
            "factor": slot.name, "input_source": source,
            "digit_count": len((p.get("Digits") or "").strip()),
            "speech_present": bool((p.get("SpeechResult") or "").strip()),
            "result_kind": kind, "tries": tries,
        }
        extra: dict[str, Any] = {"gather_ms": gather_ms} if gather_ms is not None else {}
        if res.source == "short_digits":
            extra["short_digits"] = True  # cut short; asked again without the resolver
        line: dict[str, Any] = {"call_sid": st.call_sid, "flow": flow_kind, "match": res.match,
                                "source": logged_source, "model": res.model, **modes, **fields,
                                **extra, **logged_value}
        speech = self._speech(st, slot, res, p)
        if speech is not None:
            line["speech"] = speech      # This line only, never the journal event below
        if self.fd._timings is not None:
            self._answers.append(line)        # written when the request ends (flush_answers)
        else:
            obs.emit(log, logging.INFO, "factor_answer", **line)
        if pending is not None:
            asked, answer = _ASKED.get(pending.kind, "none"), res.reply or "none"
        else:
            asked = _asked_by(action)
            answer = "none"
        # This answer's event; an applied correction writes its own as well (_log_correction).
        self.fd._log_event(st.call_sid, {
            "type": "slot_resolution", "flow": flow_kind, "slot": slot.name, **modes,
            "match": res.match, "source": logged_source, "model": res.model, "asked": asked,
            "answer": answer, "correction": False, "correction_of": None,
            "tries": tries, **value})

    def _speech(self, st: CallState, slot: SlotSettings, res: Resolution,
                p: Mapping[str, str]) -> str | None:
        """``factor_answer.speech``: ``None`` with ``speechLog`` off or no ``SpeechResult``
        (keypad input is not speech). Otherwise the text the agent would get (9.2(a)): the
        turn's ``utterance_for_ai``, built by the engine from the same resolution, so a value found
        by the model is a placeholder too, and so is a sensitive correction target's and, in
        another slot's re-ask, the interrupted sensitive slot's (:func:`hide_interrupted`). Without
        one, a sensitive slot's words go through :func:`placeholder_text` with no value, which
        replaces every number run that could hold it (fail closed). A login factor that isn't
        marked sensitive (the seed's ZIP) is never logged as said either, whatever the settings:
        its text goes through :func:`placeholder_text` with the value and every number run
        replaced, since the engine leaves its words unchanged. With ``speechLogRedaction`` on,
        the text then passes the transcript redaction for the call's language; any failure gives
        :data:`SPEECH_WITHHELD`, whatever ``on_failure`` says. The exact-value scrub of the log
        handler still applies to the line."""
        cfg = self.fd.cfg
        said = p.get("SpeechResult") or ""
        if not cfg.speech_log or not said.strip():
            return None
        text = res.utterance_for_ai
        if text is None:
            text = placeholder_text(said, slot, None, every_run=True) if slot.sensitive else said
        if slot.is_factor and not slot.sensitive:
            text = self._factor_text(text, slot, res.value)
        if not cfg.speech_log_redaction:
            return text
        with self.fd._span("speech_redact"):
            try:
                return self.fd.d.pii.text(SURFACE_SPEECH_LOG, text, language=st.language,
                                          withheld=SPEECH_WITHHELD, on_failure=ON_FAILURE_WITHHOLD)
            except Exception as exc:  # a guard bug is a failure too: never the unredacted text
                obs.log_caught(log, "speech log redaction failed", exc, operation="speech_redact",
                               decision="withhold")
                return SPEECH_WITHHELD

    def _factor_text(self, text: str, slot: SlotSettings, value: str | None) -> str:
        """``text`` with every number run and ``value`` replaced by ``slot``'s placeholder, for a
        login factor whose words the engine leaves unchanged. Placeholders already in the text
        (another slot's, :func:`hide_interrupted`) are kept as they are; if one goes missing, the
        whole text is the placeholder (fail closed)."""
        names = "|".join(re.escape(n) for n in {*self.fd.d.policy.slots(), slot.name})
        token_re = re.compile(rf"\[(?:{names})\]")
        kept = token_re.findall(text)
        out = placeholder_text(token_re.sub(_KEPT, text), slot, value, every_run=True)
        if out.count(_KEPT) != len(kept):
            return f"[{slot.name}]"
        restored = iter(kept)
        return re.sub(_KEPT, lambda _m: next(restored), out)

    # -- the factor flows (login, stepup)

    def factor(self, req: http.TwilioRequest, st: CallState, sess: Session, name: str) -> str:
        fd = self.fd
        flow = sess.flow or {}
        slot = fd._slot(name)
        lang = st.language
        pending = self.pending(st, flow, slot, self.collected(sess, lang))
        # The slot the caller is answering: the one asked, or the corrected slot while its re-ask
        # or a follow-up of it is pending. Its mode, input settings and rules apply.
        answered = (fd._slot(pending.slot) if pending is not None
                    and pending.slot != slot.name else slot)
        mode, doc = mode_now(fd, flow, answered.name)
        keypad_only = gather_input(answered, fd.cfg.accept_spoken_factors).keypad_only
        question = self.question_text(answered, pending, lang, named=_named_of(sess), sess=sess)
        res, ctx, coll, out = self._resolve(req, st, sess, answered, question=question,
                                            pending=pending, mode=mode, keypad_only=keypad_only)
        res = hide_interrupted(res, asked=slot, answered=answered)
        tries = tries_of(flow, answered.name)
        gather_ms = fd._gather_ms(flow)
        if res.value is not None and res.source != "dtmf" and keypad_only and pending is None:
            # A spoken answer to a keypad-only slot is reminded, before the dialog.
            self._log(req, st, flow, slot, res, doc, mode, tries=tries, pending=None, action=None,
                      gather_ms=gather_ms)
            return fd._keypad_reminder(req, st, sess, slot)
        owned = frozenset(n for n in fd.d.policy.slots() if fd.d.policy.slot(n).is_factor)
        turn = self._turn(slot, mode, flow, pending, coll, owned)
        action, nxt = self._decide(st, flow, turn, res, ctx)
        self._log(req, st, flow, answered, res, doc, mode, tries=tries, pending=pending,
                  action=action, gather_ms=gather_ms)
        # Generated wording only for the slot asked and answered, in its own mode.
        said = Said(mode, out, flow.get("kind")) if answered.name == slot.name else None
        return self.execute(req, st, sess, slot, res, action, nxt, said=said)

    def execute(self, req: http.TwilioRequest, st: CallState, sess: Session, slot: SlotSettings,
                res: Resolution, action: Action, nxt: Turn, *, said: Said | None = None) -> str:
        """Run ``action`` for a factor flow (``login``, ``stepup``). ``said``: this answer's mode
        and resolver output, for the generative wording of an option question."""
        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        if isinstance(action, Accept):
            if action.slot != slot.name:   # never a value for another slot
                return self._reask(req, st, sess, slot, Reask(_clarify_key(slot), slot.name), nxt)
            flow["pending"] = None
            if slot.name == PHONE_SLOT:   # keyed-in phone login: a claim, looked up
                return fd._claim_phone(req, st, sess, action.value)
            return fd._verify(req, st, sess, slot.name, action.value)
        if isinstance(action, (ReadBackDigits, ConfirmOption, Disambiguate, NotAvailable)) \
                and action.slot != slot.name:
            # A follow-up about the corrected slot (its read-back or option question): its
            # own Gather, catalog wording only (generated text is for the slot asked).
            asked = self._correctable(sess, action.slot, lang, nxt)
            if asked is None:
                return self._reask(req, st, sess, slot, Reask(_clarify_key(slot), slot.name),
                                   self._drop_pending(flow, nxt))
            if isinstance(action, ReadBackDigits):
                item = fd._say(action.key, lang, digits=digits_words(action.value, lang))
            else:
                if isinstance(action, NotAvailable):
                    flow["named"] = action.named
                item = self._option_question(asked, action, lang)
            return self._follow_up(req, st, sess, asked, [item], nxt.pending)
        if isinstance(action, ReadBackDigits):
            items = [fd._say(action.key, lang, digits=digits_words(action.value, lang))]
            return self._follow_up(req, st, sess, slot, items, nxt.pending)
        if isinstance(action, (ConfirmOption, Disambiguate, NotAvailable)):
            item = self.wording(st, slot, action, lang, said)
            _unlist_if_generated(flow, item)
            return self._follow_up(req, st, sess, slot, [item], nxt.pending)
        if isinstance(action, ApplyCorrection):
            return self._apply_correction(req, st, sess, slot, action, nxt)
        if isinstance(action, Reask):
            if action.slot != slot.name:
                # The corrected slot's re-ask: its own clarify and Gather.
                asked = self._correctable(sess, action.slot, lang, nxt)
                if asked is None:
                    return self._reask(req, st, sess, slot, Reask(_clarify_key(slot), slot.name),
                                       self._drop_pending(flow, nxt))
                return self._reask(req, st, sess, asked, action, nxt)
            return self._reask(req, st, sess, slot, action, nxt)
        if isinstance(action, Command):
            if action.kind == "repeat":
                return self._repeat(req, st, sess, slot, nxt)
            return fd._off_script(req, st, sess, action.kind, res.utterance_for_ai)
        if isinstance(action, IntentSwitch):
            return fd._off_script(req, st, sess, "intent_switch", res.utterance_for_ai)
        if isinstance(action, GiveUp):
            return fd._give_up(req, st, sess, "unclear")
        raise RuntimeError(f"unhandled slot action {type(action).__name__}")

    # -- corrections

    def _correctable(self, sess: Session, name: str, lang: str, nxt: Turn) -> SlotSettings | None:
        """``name``'s settings if a correction may still change it: collected in this flow, not
        sent to the auth service, no API call in the flow since it was collected
         and held by ``_verify`` (the one place a factor's value waits, sealed). ``None``
        otherwise (fail safe: nothing is stored)."""
        row = next((c for c in self.collected(sess, lang) if c.name == name), None)
        held = (sess.flow or {}).get("held")
        if (row is None or row.scope != "current" or row.verified or not fresh(sess, name)
                or name not in nxt.owned or not isinstance(held, Mapping) or name not in held):
            return None
        try:
            return self.fd.d.policy.slot(name)
        except KeyError:
            return None

    @staticmethod
    def _drop_pending(flow: dict[str, Any], nxt: Turn) -> Turn:
        flow["pending"] = None
        return Turn(slot=nxt.slot, mode=nxt.mode, tries=nxt.tries, pending=None,
                    collected=nxt.collected, earlier_strikes=nxt.earlier_strikes, owned=nxt.owned,
                    settings=nxt.settings, caller_options=nxt.caller_options)

    def _apply_correction(self, req: http.TwilioRequest, st: CallState, sess: Session,
                          slot: SlotSettings, action: ApplyCorrection, nxt: Turn) -> str:
        """Replace the held value of ``action.target`` (sealed), remember it if sensitive
        (the old value stays registered), journal the correction, say
        ``slot.corrected`` or ``slot.corrected_sensitive`` and ask ``slot`` again
        (``action.then``; its try was counted by ``decide``). Never a ``verify`` call."""
        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        sid = st.call_sid
        ts = self._correctable(sess, action.target, lang, nxt)
        if ts is None:
            # Not a value this flow still holds: nothing is stored; the slot is asked again.
            return self._reask(req, st, sess, slot, Reask(_clarify_key(slot), slot.name),
                               self._drop_pending(flow, nxt))
        if ts.sensitive:
            fd._remember(sid, ts, action.value)
        held = dict(flow.get("held") or {})
        held[ts.name] = fd.d.sealer.seal(action.value, call_sid=sid)
        flow["held"] = held
        flow["pending"] = None
        self._log_correction(st, flow, ts, of=slot.name, value=action.value,
                             tries=int(nxt.tries.get(ts.name, 0)))
        if ts.sensitive:
            label = fd.d.catalog.text(ts.decl.label_key, lang) if ts.decl.label_key else ""
            said = fd._say("slot.corrected_sensitive", lang, label=label)
        elif ts.decl.slot_type == "digits":
            said = fd._say("slot.corrected", lang, value=digits_words(action.value, lang))
        else:
            said = fd._say("slot.corrected", lang, value=self._name(ts, action.value, lang))
        return self._reask(req, st, sess, slot, action.then, self._drop_pending(flow, nxt),
                           pre=[said])

    def _log_correction(self, st: CallState, flow: Mapping[str, Any], ts: SlotSettings, *,
                        of: str, value: str, tries: int) -> None:
        """The correction's own ``slot_resolution``: ``slot`` is the slot it
        changed, ``correction_of`` the slot being asked; the value only for a non-sensitive slot
        ."""
        mode, doc = mode_now(self.fd, flow, ts.name)
        extra = {} if ts.sensitive else {"value": value}
        self.fd._log_event(st.call_sid, {
            "type": "slot_resolution", "flow": flow.get("kind"), "slot": ts.name,
            "configured_mode": doc.get("configured"), "effective_mode": mode,
            "mode_source": doc.get("source"), "ceiling": self.fd.cfg.slot_mode_ceiling,
            "capped": doc.get("capped"), "match": "correction", "source": "model",
            "model": "ok", "asked": "none", "answer": "none", "correction": True,
            "correction_of": of, "tries": tries, **extra})

    # -- questions again

    def _reask(self, req: http.TwilioRequest, st: CallState, sess: Session, slot: SlotSettings,
               action: Reask, nxt: Turn, *, pre: list[tuple[Any, Spoken]] | None = None) -> str:
        """The question again (the try was counted by ``decide``): the slot's prompt, a
        ``clarify.*`` template, or for a yes/no follow-up ``clarify.yes_no`` and the question.
        ``pre`` is spoken first (a correction's confirmation)."""
        fd = self.fd
        lang = st.language
        decl = slot.decl
        if action.key == "clarify.digits":
            c = decl.constraints
            length = c.get("length") or c.get("maxLength") or ""
            label = fd.d.catalog.text(decl.label_key, lang) if decl.label_key else ""
            items = [fd._say("clarify.digits", lang, length=length, label=label)]
        elif action.key == "prompt":
            items = [self.prompt(sess, slot, lang)]
        elif action.key == "clarify.choice":
            # The caller's options (the card: the cards on file), as today; the prompt that
            # follows lists them in the same order.
            if self._prompt_lists(slot, lang):
                # The question names the options itself: ask it again, without a
                # "Please say …" that would list them twice in a row.
                items = [self.prompt(sess, slot, lang)]
            else:
                listed = self._names(slot, self.listed(sess, slot), lang)
                items = [fd._say("clarify.choice", lang, options=listed),
                         self.prompt(sess, slot, lang)]
        else:
            items = [fd._say("clarify.yes_no", lang)]
            p = nxt.pending
            if p is not None:
                again = self._pending_question(slot, p, lang, named=_named_of(sess))
                if again is not None:
                    items.append(again)
        items = [*(pre or []), *items]
        if nxt.pending is not None:
            return self._follow_up(req, st, sess, slot, items, nxt.pending)
        return fd._reprompt(req, st, sess, slot, self._keypad_menu(sess, slot, items, None, lang))

    def _repeat(self, req: http.TwilioRequest, st: CallState, sess: Session, slot: SlotSettings,
                nxt: Turn) -> str:
        """``repeat``: the question again without a try, up to ``MAX_REPEATS``; then each one is an
        unclear answer."""
        from .app import MAX_REPEATS

        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        flow["repeats"] = int(flow.get("repeats", 0)) + 1
        p = nxt.pending
        if flow["repeats"] > MAX_REPEATS:
            n = tries_of(flow, slot.name) + 1
            set_tries(flow, slot.name, n)
            if n > slot.decl.max_retries:
                return fd._give_up(req, st, sess, "unclear")
            if p is not None and p.slot != slot.name:
                nxt = self._drop_pending(flow, nxt)   # the correction is dropped; resume
            return self._reask(req, st, sess, slot, Reask(_clarify_key(slot), slot.name), nxt)
        if p is not None and p.slot != slot.name:
            asked = self._correctable(sess, p.slot, lang, nxt)
            if asked is not None and p.kind == "correction_reask":
                # The corrected slot's re-ask again, no try.
                return self._reask(req, st, sess, asked, Reask(_clarify_key(asked), asked.name), nxt)
            again = (self._pending_question(asked, p, lang, named=_named_of(sess))
                     if asked is not None else None)
            if asked is not None and again is not None:
                return self._follow_up(req, st, sess, asked, [again], p)   # its follow-up again
            nxt = self._drop_pending(flow, nxt)
            p = None
        if p is not None:
            again = self._pending_question(slot, p, lang, named=_named_of(sess))
            if again is not None:
                return self._follow_up(req, st, sess, slot, [again], p)
        return fd._reprompt(req, st, sess, slot, [self.prompt(sess, slot, lang)])

    def _follow_up(self, req: http.TwilioRequest, st: CallState, sess: Session, slot: SlotSettings,
                   items: list[tuple[Any, Spoken]], p: Pending | None) -> str:
        """A follow-up question ``p`` about ``slot``: its Gather follows the slot's settings
        (``keypad_only``, ``barge_in``, Path B); a yes/no follow-up listens for a yes or no (no
        digit count, the confirmation's speech timeout), and a keypad-only slot hears its keypad
        menu (a choice question: the options with their keys)."""
        fd = self.fd
        lang = st.language
        yes_no = p is not None and p.kind in _YES_NO_FOLLOW_UPS
        if gather_input(slot, fd.cfg.accept_spoken_factors).keypad_only and yes_no:
            items = [*items, fd._say("slot.keypad_yes_no", lang)]
        elif not yes_no:
            items = self._keypad_menu(sess, slot, items,
                                      p.candidates if p is not None and p.candidates else None, lang)
        return fd._reprompt(req, st, sess, slot, items, follow_up=yes_no)

    def _keypad_menu(self, sess: Session, slot: SlotSettings, items: list[tuple[Any, Spoken]],
                     values: Sequence[str] | None, lang: str) -> list[tuple[Any, Spoken]]:
        """``items`` and, for a keypad-only choice slot, its keypad menu: ``slot.keypad_choice``
        over ``slot.keypad_option`` per option (``values``, else the caller's options), each with
        the key the keypad parse gives it (an explicit ``dtmf``, else its position among the
        caller's options). Any other slot: ``items`` unchanged."""
        fd = self.fd
        if (slot.decl.slot_type != "choice" or not slot.options
                or not gather_input(slot, fd.cfg.accept_spoken_factors).keypad_only):
            return items
        caller = self.listed(sess, slot) or tuple(o.value for o in slot.options)
        keys = _keypad_keys(slot, caller)
        shown = [v for v in (values or caller) if v in keys]
        if not shown:
            return items
        cat = fd.d.catalog
        menu = ", ".join(cat.text("slot.keypad_option", lang, digit=keys[v],
                                  option=self._name(slot, v, lang)) for v in shown)
        return [*items, fd._say("slot.keypad_choice", lang, menu=menu)]

    # -- question texts (catalog only)

    def question_text(self, slot: SlotSettings, pending: Pending | None, lang: str, *,
                      named: str | None = None, sess: Session | None = None) -> str:
        """The question for the resolver. A digit read-back holds the value it reads back, so
        the resolver gets the slot's prompt with the pending kind instead. With ``sess``, a
        choice slot's prompt names its options as the caller heard them."""
        fd = self.fd
        if pending is not None and pending.kind != "confirm_digits":
            again = self._pending_question(slot, pending, lang, named=named)
            if again is not None:
                return again[1].text
        key = self._prompt_key_for(slot, lang)
        if sess is not None and slot.decl.slot_type == "choice" and slot.options:
            values = self._prompt_values(sess, slot)
            return fd.d.catalog.text(key, lang,
                                     options=self._prompt_options(sess, slot, values, lang))
        return fd.d.catalog.text(key, lang)

    def _prompt_key_for(self, slot: SlotSettings, lang: str, base: str | None = None) -> str:
        fd = self.fd
        keypad_only = gather_input(slot, fd.cfg.accept_spoken_factors).keypad_only
        return _prompt_key(slot, keypad_only, fd.d.catalog, lang, base)

    def prompt(self, sess: Session, slot: SlotSettings, lang: str) -> tuple[Any, Spoken]:
        """The slot's own question (its ``prompt_key``). A choice slot's names its options
        (``{options}``): in the confirm flow the card's are the caller's cards, each with "your",
        in policy order; that order is recorded for ordinals."""
        fd = self.fd
        key = self._prompt_key_for(slot, lang)
        if slot.decl.slot_type != "choice" or not slot.options:
            return fd._say(key, lang)
        values = self._prompt_values(sess, slot)
        _set_listed(sess, slot, values)
        return fd._say(key, lang, options=self._prompt_options(sess, slot, values, lang))

    def _prompt_lists(self, slot: SlotSettings, lang: str) -> bool:
        """Whether the slot's own question names its options (``{options}`` in its template)."""
        if slot.decl.slot_type != "choice" or not slot.options:
            return False
        key = self._prompt_key_for(slot, lang)
        try:
            return "{options}" in self.fd.d.catalog.template(key, lang)
        except Exception:  # noqa: BLE001 - a missing template: keep the clarify (fail safe)
            return False

    def _prompt_values(self, sess: Session, slot: SlotSettings) -> tuple[str, ...]:
        listed = self.listed(sess, slot)
        return listed if listed is not None else tuple(o.value for o in slot.options)

    def _prompt_options(self, sess: Session, slot: SlotSettings, values: tuple[str, ...],
                        lang: str) -> str:
        yours = self.listed(sess, slot) is not None      # the caller's own (the cards on file)
        return self._names(slot, values, lang,
                           possessive=_YOUR.get(lang_family(lang)) if yours else None)

    def _pending_question(self, slot: SlotSettings, p: Pending, lang: str, *,
                          named: str | None = None) -> tuple[Any, Spoken] | None:
        """The pending question again. ``named``: the option a not-available question named (the
        flow's ``named``); without it that question isn't repeated (``clarify.yes_no`` alone)."""
        fd = self.fd
        if p.kind == "confirm_digits":
            if p.sealed_value is None:
                return None
            return fd._say("slot.confirm_digits", lang, digits=digits_words(p.sealed_value, lang))
        if p.kind == "confirm_option" and p.option:
            return self._option_question(slot, ConfirmOption(slot.name, p.option), lang)
        if p.kind == "disambiguate":
            return self._option_question(slot, Disambiguate(slot.name, p.candidates), lang)
        if p.kind == "not_available_one" and p.option and named:
            return self._option_question(slot, NotAvailable(slot.name, named, (p.option,)), lang)
        return None

    def _name(self, slot: SlotSettings, value: str, lang: str) -> str:
        cat = self.fd.d.catalog
        key = f"option.{slot.name}.{value}"
        return cat.text(key, lang) if key in cat.keys else value

    def _names(self, slot: SlotSettings, values: tuple[str, ...] | None, lang: str, *,
               possessive: str | None = None) -> str:
        """The options' spoken names joined with ``label.or`` (``slot.disambiguate``'s list);
        ``possessive`` ("your") before each one."""
        vals = values if values is not None else tuple(o.value for o in slot.options)
        names = [self._name(slot, v, lang) for v in vals]
        if possessive:
            names = [f"{possessive} {n}" for n in names]
        if len(names) <= 1:
            return "".join(names)
        word = self.fd.d.catalog.text("label.or", lang) if "label.or" in self.fd.d.catalog.keys else "or"
        return f"{', '.join(names[:-1])} {word} {names[-1]}"

    def _option_question(self, slot: SlotSettings, action: Action, lang: str) -> tuple[Any, Spoken]:
        fd = self.fd
        if isinstance(action, ConfirmOption):
            return fd._say(action.key, lang, option=self._name(slot, action.option, lang))
        if isinstance(action, Disambiguate):
            return fd._say(action.key, lang, options=self._names(slot, action.candidates, lang))
        if isinstance(action, NotAvailable):
            return fd._say(action.key, lang, named=self._name(slot, action.named, lang),
                           option=self._name(slot, action.on_file[0], lang) if action.on_file else "",
                           options=self._names(slot, action.on_file, lang))
        raise TypeError(type(action).__name__)

    # -- generative wording

    def wording(self, st: CallState, slot: SlotSettings,
                action: ConfirmOption | Disambiguate | NotAvailable, lang: str,
                said: Said | None) -> tuple[Any, Spoken]:
        """The option question for ``action``: the model's ``say``, rendered, when this answer ran
        in ``generative`` on a slot that may be generative and the check accepts it;
        else the catalog template. A fallback logs ``generative_fallback`` once, with the reason
        and never the text; a timeout was logged as ``budget`` already. No second call."""
        fd = self.fd
        template = self._option_question(slot, action, lang)
        if (said is None or said.mode != "generative" or said.output is None
                or action.slot != slot.name or not _may_generate(slot)):
            return template      # not generative, or no resolver call on this answer
        out = said.output
        if out.outcome == "timeout":
            return template      # logged as ``budget`` when it happened
        allowed = _allowed(action)
        reason: str | None = generative.fallback_reason(
            out.say, lang, out.say_lang, slot=slot.name, allowed=allowed,
            catalog_has=fd.d.catalog.has)
        if reason is not None or out.say is None:   # an accepted text is present (``missing``)
            self._fallback(st, said.flow, slot, reason or "missing", asked=_asked_by(action))
            return template
        text = generative.render(out.say, slot.name, lang, fd.d.catalog)
        # The template stays with it, spoken instead if the response doesn't parse.
        return fd._say_generated(action.key, lang, text, template=template)

    def _fallback(self, st: CallState, flow: object, slot: SlotSettings, reason: str, *,
                  asked: str | None = None) -> None:
        """``generative_fallback``: names and the reason only, never the text."""
        extra = {"asked": asked} if asked is not None else {}
        obs.emit(log, logging.INFO, "generative_fallback", call_sid=st.call_sid,
                 flow=flow if isinstance(flow, str) else None, slot=slot.name, reason=reason,
                 decision="template", **extra)

    # -- the confirm flow: the card and the read-back

    def caller_cards(self, sess: Session) -> tuple[str, ...]:
        """The caller's cards, as the confirm flow froze them at its start (``flow["cards"]``, from
        the verified customer's record), restricted to the ``card`` slot's known options."""
        raw = (sess.flow or {}).get("cards")
        try:
            known = {o.value for o in self.fd.d.policy.slot(CARD_SLOT).options}
        except KeyError:
            return ()
        return tuple(v for v in raw if isinstance(v, str) and v in known) \
            if isinstance(raw, list) else ()

    def listed(self, sess: Session, slot: SlotSettings) -> tuple[str, ...] | None:
        """The options a choice question lists and keys: the caller's cards in the confirm flow;
        ``None`` (every option) otherwise."""
        if slot.name == CARD_SLOT and (sess.flow or {}).get("kind") == "confirm":
            return self.caller_cards(sess)
        return None

    def ask_card(self, req: http.TwilioRequest, st: CallState, sess: Session, *,
                 pre: list[tuple[Any, Spoken]]) -> str:
        """``prompt.card`` before the read-back: the AI gave no card, or a value that
        isn't a known card, and the caller holds two or more (never mapped to another). The
        question names the caller's cards in policy order (that order is the ordinals'). A
        choice Gather with the cards' spoken names and aliases as hints, in ``card``'s input
        settings, which may finish early."""
        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        slot = fd._slot(CARD_SLOT)
        flow.update(factor=CARD_SLOT, pending=None, repeats=0)
        sess.flow = flow
        fd._log_event(st.call_sid, {"type": "factor_asked", "factor": CARD_SLOT})
        items = [*pre, self.prompt(sess, slot, lang)]
        return fd._reprompt(req, st, sess, slot, self._keypad_menu(sess, slot, items, None, lang))

    def ask_not_available(self, req: http.TwilioRequest, st: CallState, sess: Session,
                          named: str, *, pre: list[tuple[Any, Spoken]]) -> str:
        """The AI named a known card the caller doesn't hold. The not-available
        question about it (``slot.not_available_one`` with one card on file, a yes/no follow-up;
        ``slot.not_available_many`` with several, listing them), never another card in its
        place. The AI's naming uses no try of the caller's."""
        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        slot = fd._slot(CARD_SLOT)
        cards = self.caller_cards(sess)
        if len(cards) == 1:
            p = Pending("not_available_one", slot.name, option=cards[0])
        else:
            p = Pending("not_available_many", slot.name, candidates=cards, listed=cards)
        flow.update(factor=CARD_SLOT, repeats=0, named=named, pending=self.store_pending(st, p))
        sess.flow = flow
        fd._log_event(st.call_sid, {"type": "factor_asked", "factor": CARD_SLOT})
        item = self._option_question(slot, NotAvailable(slot.name, named, cards), lang)
        return self._follow_up(req, st, sess, slot, [*pre, item], p)

    def card(self, req: http.TwilioRequest, st: CallState, sess: Session, tool: str) -> str:
        """An answer to the card question or one of its follow-ups, through ``decide``
        with the caller's cards as the caller's options. A card is selected only when the caller
        named it, pressed its key, or said a deterministic yes to a question naming it; then it replaces ``pending_tool.args.to`` and the payment is read back."""
        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        slot = fd._slot(CARD_SLOT)
        cards = self.caller_cards(sess)
        pending = self.pending(st, flow, slot, self.collected(sess, lang))
        if pending is not None and pending.slot != slot.name:
            pending = None     # nothing else of this flow is correctable here (fail safe)
        mode, doc = mode_now(fd, flow, slot.name)
        keypad_only = gather_input(slot, fd.cfg.accept_spoken_factors).keypad_only
        question = self.question_text(slot, pending, lang, named=_named_of(sess), sess=sess)
        res, ctx, coll, out = self._resolve(req, st, sess, slot, question=question,
                                            pending=pending, mode=mode, keypad_only=keypad_only,
                                            caller=cards)
        if keypad_only and res.source != "dtmf" and res.match != "none":
            res = Resolution(match="none", source="none")   # Keys only (fail safe)
        tries = tries_of(flow, slot.name)
        turn = self._turn(slot, mode, flow, pending, coll, frozenset({CARD_SLOT}),
                          caller_options={CARD_SLOT: cards})
        action, nxt = self._decide(st, flow, turn, res, ctx)
        self._log(req, st, flow, slot, res, doc, mode, tries=tries, pending=pending, action=action)
        if isinstance(action, Accept) and action.slot == slot.name and action.value in cards:
            flow["pending"] = None
            st = fd._set_card(st, tool, action.value)
            collect(sess, CARD_SLOT, "choice", verified=False)     # correctable in the read-back
            return fd._confirm_readback(req, st, sess, tool)
        if isinstance(action, (ConfirmOption, Disambiguate, NotAvailable)):
            if isinstance(action, NotAvailable):
                flow["named"] = action.named
            item = self.wording(st, slot, action, lang, Said(mode, out, flow.get("kind")))
            _unlist_if_generated(flow, item)
            return self._follow_up(req, st, sess, slot, [item], nxt.pending)
        if isinstance(action, Reask) and action.slot == slot.name:
            return self._reask(req, st, sess, slot, action, nxt)
        if isinstance(action, Command) and action.kind == "repeat":
            from .app import MAX_REPEATS

            flow["repeats"] = int(flow.get("repeats", 0)) + 1
            if flow["repeats"] > MAX_REPEATS:
                return fd._confirm_cancel(req, st, sess, tool, "no_answer")
            p = nxt.pending
            again = self._pending_question(slot, p, lang, named=_named_of(sess)) if p else None
            if p is not None and again is not None:
                return self._follow_up(req, st, sess, slot, [again], p)
            items = [self.prompt(sess, slot, lang)]
            return fd._reprompt(req, st, sess, slot,
                                self._keypad_menu(sess, slot, items, None, lang))
        off = self._confirm_off_script(req, st, sess, tool, res, action)
        if off is not None:
            return off
        # Anything else (an option not on file, a correction, a digit read-back): asked again,
        # nothing stored (fail safe).
        return self._reask(req, st, sess, slot, Reask("clarify.choice", slot.name),
                           self._drop_pending(flow, nxt))

    def _confirm_off_script(self, req: http.TwilioRequest, st: CallState, sess: Session,
                            tool: str, res: Resolution, action: Action) -> str | None:
        """Give up, cancel, escalate or an intent switch in the confirm flow; ``None`` for any
        other action. The tool doesn't run."""
        fd = self.fd
        if isinstance(action, GiveUp):
            return fd._confirm_cancel(req, st, sess, tool, "no_answer")
        if isinstance(action, Command) and action.kind == "cancel":
            return fd._confirm_cancel(req, st, sess, tool, "cancel")
        if isinstance(action, (Command, IntentSwitch)) and not (
                isinstance(action, Command) and action.kind == "repeat"):
            turn_text = res.utterance_for_ai or None
            if not turn_text:
                return fd._confirm_cancel(req, st, sess, tool, "cancel")
            return fd._confirm_interrupted(req, st, sess, tool, turn_text)
        return None

    def _card_named(self, card: SlotSettings, res: Resolution, speech: str | None,
                    lang: str, coll: tuple[CollectedSlot, ...] = ()) -> Resolution:
        """A card the model proposes as a correction: kept only if the caller named
        it (its spoken name or an alias in the words), checked against the target's options as
        the resolver saw them (the collected card's ``options``, else the policy's); otherwise its
        value is dropped, so the dialog re-asks the card and nothing is selected."""
        if res.value is None or res.source != "model":
            return res
        row = next((c for c in coll if c.name == card.name and c.options), None)
        views = row.options if row is not None else self.option_views(card, lang)
        view = next((v for v in views if v.value == res.value), None)
        if view is not None and speech and grounding.named_in(normalize(speech, lang), view, lang):
            return res
        return replace(res, value=None)

    def confirm(self, req: http.TwilioRequest, st: CallState, sess: Session, tool: str) -> str:
        """An answer to the read-back (``slot#yes_no``), or to the card's re-ask after a
        correction whose card failed (``correction_reask``: the card's mode, rules and
        Gather). A deterministic yes runs the tool with the card read back. A card
        correction replaces ``pending_tool.args.to``, says ``slot.corrected`` and reads back again
        . A correction of anything else the AI proposed (the amount, the from-account) is an
        intent switch back to the AI."""
        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        slot = fd._slot("yes_no")   # The confirmation's settings are ``slot#yes_no``
        cards = self.caller_cards(sess)
        pending = self.pending(st, flow, slot, self.collected(sess, lang))
        if pending is not None and pending.slot != CARD_SLOT:
            pending = None
        answered = fd._slot(pending.slot) if pending is not None else slot
        mode, doc = mode_now(fd, flow, answered.name)
        keypad_only = gather_input(answered, fd.cfg.accept_spoken_factors).keypad_only
        question = (fd._readback(st, tool)[1].text if pending is None
                    else self.question_text(answered, pending, lang, named=_named_of(sess),
                                            sess=sess))
        res, ctx, coll, _ = self._resolve(req, st, sess, answered, question=question,
                                          pending=pending, mode=mode, keypad_only=keypad_only,
                                          caller=cards if pending is not None else None)
        res = hide_interrupted(res, asked=slot, answered=answered)
        speech = req.params.get("SpeechResult") or None
        card = fd._slot(CARD_SLOT) if CARD_SLOT in fd.d.policy.slots() else None
        named: frozenset[str] = frozenset()
        if card is not None and res.match == "correction" and res.target == CARD_SLOT:
            res = self._card_named(card, res, speech, lang, coll)
            if res.value is not None:
                named = frozenset({CARD_SLOT})    # the caller named it
        elif card is not None and pending is not None and res.match == "inferred":
            res = replace(res, value=None)    # the card's re-ask: named or keyed only
        tries = tries_of(flow, answered.name)
        if pending is None and res.match == "correction" and res.target is None \
                and mode != "deterministic":
            # A change to an argument the AI proposed (no slot this call collected): back to
            # the AI, which proposes again; a new read-back confirms it.
            flow["pending"] = None
            action: Action = IntentSwitch("not_owned")
            self._log(req, st, flow, slot, res, doc, mode, tries=tries, pending=None, action=action)
            return fd._confirm_interrupted(req, st, sess, tool, res.utterance_for_ai or "") \
                if res.utterance_for_ai else fd._confirm_cancel(req, st, sess, tool, "cancel")
        turn = self._turn(slot, mode, flow, pending, coll, frozenset({CARD_SLOT}),
                          caller_options={CARD_SLOT: cards}, named=named)
        action, nxt = self._decide(st, flow, turn, res, ctx)
        self._log(req, st, flow, answered, res, doc, mode, tries=tries, pending=pending,
                  action=action)
        if (isinstance(action, (ConfirmOption, Disambiguate, NotAvailable)) and card is not None
                and action.slot == CARD_SLOT and nxt.pending is not None):
            # A follow-up about the corrected card: in the card's Gather, from
            # the catalog. Only a deterministic yes or the card named selects it.
            if isinstance(action, NotAvailable):
                flow["named"] = action.named
            return self._follow_up(req, st, sess, card, [self._option_question(card, action, lang)],
                                   nxt.pending)
        if isinstance(action, Accept) and action.slot == slot.name:
            flow["pending"] = None
            if action.value == "yes":
                return fd._confirm_execute(req, st, sess, tool)
            return fd._confirm_cancel(req, st, sess, tool, "no")
        if isinstance(action, ApplyCorrection):
            return self._correct_card(req, st, sess, tool, action, cards)
        if isinstance(action, Command) and action.kind == "repeat":
            from .app import MAX_REPEATS

            flow["repeats"] = int(flow.get("repeats", 0)) + 1
            if flow["repeats"] > MAX_REPEATS:
                return fd._confirm_cancel(req, st, sess, tool, "no_answer")
            p = nxt.pending
            if p is not None and card is not None:    # the card's question, again (no try)
                again = (self._pending_question(card, p, lang, named=_named_of(sess))
                         if p.kind != "correction_reask" else None)
                items = [again or self.prompt(sess, card, lang)]
                return self._follow_up(req, st, sess, card, items, p)
            return fd._confirm_ask(req, st, sess, [fd._readback(st, tool)])
        off = self._confirm_off_script(req, st, sess, tool, res, action)
        if off is not None:
            return off
        if (isinstance(action, Reask) and card is not None and action.slot == CARD_SLOT
                and nxt.pending is not None):
            # The corrected card's re-ask, in the card's Gather.
            return self._reask(req, st, sess, card, action, nxt)
        flow["pending"] = None
        if isinstance(action, Reask) and not action.consume:
            return fd._confirm_ask(req, st, sess, [fd._readback(st, tool)])   # resumes
        # Reask (an unclear answer, an earlier-flow correction) or anything else: ask again.
        return fd._confirm_ask(req, st, sess, [fd._say("clarify.yes_no", lang),
                                               fd._readback(st, tool)])

    def _correct_card(self, req: http.TwilioRequest, st: CallState, sess: Session, tool: str,
                      action: ApplyCorrection, cards: tuple[str, ...]) -> str:
        """The card correction. ``pending_tool.args.to`` is replaced (``callstate.mutate``),
        the correction journaled, ``slot.corrected`` said with the card's spoken name and
        the payment read back again. A target other than one of the caller's cards changes
        nothing and is asked again (fail safe; ``decide`` only corrects owned slots)."""
        fd = self.fd
        flow = sess.flow or {}
        lang = st.language
        flow["pending"] = None
        if action.target != CARD_SLOT or action.value not in cards or not fresh(sess, CARD_SLOT):
            return fd._confirm_ask(req, st, sess, [fd._say("clarify.yes_no", lang),
                                                   fd._readback(st, tool)])
        card = fd._slot(CARD_SLOT)
        st = fd._set_card(st, tool, action.value)
        collect(sess, CARD_SLOT, "choice", verified=False)
        self._log_correction(st, flow, card, of="yes_no", value=action.value,
                             tries=tries_of(flow, CARD_SLOT))
        said = fd._say("slot.corrected", lang, value=self._name(card, action.value, lang))
        return fd._confirm_readback(req, st, sess, tool, pre=[said])

    def save(self, sess: Session) -> None:
        self.fd._save(sess)


@dataclass(frozen=True, slots=True)
class Said:
    """One answer's mode and resolver output (``None``: not called), and the flow it was in: what
    :meth:`SlotFlow.wording` needs for an option question's generative wording."""

    mode: Mode
    output: ResolverOutput | None
    flow: object = None


def _record_values(flow: Mapping[str, Any], slot: SlotSettings) -> Any:
    """``flow["listed"]``'s values for ``slot`` (a question with nothing pending)."""
    rec = flow.get("listed")
    return rec.get("values") if isinstance(rec, Mapping) and rec.get("slot") == slot.name else None


def _valid_listed(vals: Any) -> tuple[str, ...]:
    """A stored listed order, or ``()`` when it isn't distinct non-empty strings (fail safe)."""
    if not isinstance(vals, list) or not all(isinstance(v, str) and v for v in vals) \
            or len(set(vals)) != len(vals):
        return ()
    return tuple(vals)


def _set_listed(sess: Session, slot: SlotSettings, values: Sequence[str]) -> None:
    """Record the order a question with nothing pending read ``slot``'s options in."""
    if sess.flow is not None:
        sess.flow["listed"] = {"slot": slot.name, "values": list(dict.fromkeys(values))}


def _unlist_if_generated(flow: Mapping[str, Any], item: tuple[Any, Spoken]) -> None:
    """A generated question's order isn't the candidates' (the model words it): nothing is listed,
    so no ordinal answers it (fail safe)."""
    from .app import GeneratedSpoken

    pending = flow.get("pending")
    if isinstance(item[1], GeneratedSpoken) and isinstance(pending, dict):
        pending["listed"] = []


def _may_generate(slot: SlotSettings) -> bool:
    """Generative wording only on a non-sensitive choice slot (the frozen mode already says
    so; checked again with the running policy, fail closed)."""
    return slot.decl.slot_type == "choice" and not slot.sensitive


def _allowed(action: ConfirmOption | Disambiguate | NotAvailable) -> tuple[str, ...]:
    """The options a generated question may name: a confirmation's option, a
    disambiguation's candidates, a not-available question's named option and the caller's
    options."""
    if isinstance(action, ConfirmOption):
        return (action.option,)
    if isinstance(action, Disambiguate):
        return tuple(action.candidates)
    return (action.named, *action.on_file)


def _no_decrease(flow: Mapping[str, Any], tries: Mapping[str, int]) -> dict[str, int]:
    """The next ``flow["tries"]``: ``decide``'s counts, never below the stored ones."""
    old = flow.get("tries")
    out = {k: int(v) for k, v in old.items()} if isinstance(old, Mapping) else {}
    for k, v in tries.items():
        out[k] = max(out.get(k, 0), int(v))
    return out


def _clarify_key(slot: SlotSettings) -> Any:
    t = slot.decl.slot_type
    return {"digits": "clarify.digits", "choice": "clarify.choice",
            "yes_no": "clarify.yes_no"}.get(t, "prompt")


def _asked_by(action: Action | None) -> str:
    if isinstance(action, ReadBackDigits):
        return "confirm_digits"
    if isinstance(action, ConfirmOption):
        return "confirm_option"
    if isinstance(action, Disambiguate):
        return "disambiguate"
    if isinstance(action, NotAvailable):
        return "not_available"
    return "none"


def _named_of(sess: Session) -> str | None:
    """The option the last not-available question named (``flow["named"]``), if any."""
    named = (sess.flow or {}).get("named")
    return named if isinstance(named, str) else None


def _keypad_keys(slot: SlotSettings, caller: Sequence[str]) -> dict[str, str]:
    """Each caller option's keypad key, as the engine's keypad parse reads it: an explicit
    ``dtmf``, else its position among the caller's options without one."""
    explicit = {o.value: o.dtmf for o in slot.options if o.dtmf is not None}
    free = [v for v in caller if v not in explicit]
    return {v: explicit[v] if v in explicit else str(free.index(v) + 1) for v in caller}
