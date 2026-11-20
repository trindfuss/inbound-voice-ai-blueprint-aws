"""Dialog policy: the next action for one resolved answer (task).

:func:`decide` is pure: no I/O, no model call, no logging. It turns a :class:`Resolution` (already
grounded by the engine) and the flow's dialog state (:class:`Turn`) into one :class:`Action` and the
next state. The front door (``slotflow.py``) executes the action. Every rule comes from the
requirements:

* **Answer first.** A resolver output whose ``value`` grounds for the slot being answered is
  that value, as ``inferred``, whatever its kind (``intent_switch``, a command, a correction that
  names no collected slot). "Actually, can I get my recent transactions" carries no value and stays
  an intent switch.
* **Accept or confirm.** ``exact`` accepts, except a model ``exact`` with
  ``confirm: always``; ``inferred`` follows ``confirm``. A digit value is read back
  (``slot.confirm_digits``) only when the slot is not sensitive; a sensitive one is re-asked with
  ``clarify.digits``. An inferred choice is always confirmed: an option the caller didn't
  name is never selected without a yes, so ``confirm: never`` doesn't skip it.
* **Ambiguous.** Digits re-ask with ``clarify.digits`` and never list candidates; a
  choice asks ``slot.disambiguate`` with the candidates only (>= 2 caller options).
* **Not available.** A known option that isn't one of the caller's gives
  ``slot.not_available_one`` (answered as yes/no; only a deterministic yes selects the option on
  file) or ``slot.not_available_many`` (answered as a fresh answer).
* **Explicit yes.** A yes selects the option its question names only if it is
  deterministic (keypad or the lists). A model yes is unclear (``clarify.yes_no``, one
  try). The same holds for a read-back of digits and for a yes on a ``yes_no`` slot.
* **"No, I meant Y".** In reply to a question that names an option, Y is selected directly
  only if it is an ``exact`` caller option; inferred, it is confirmed; several give a
  disambiguation; a known option not on file gives not available.
* ``none`` re-asks (``clarify.*``) and never selects the nearest option.
* **Corrections (6.1-6.7).** Of a ``current``, unverified slot the flow owns: grounded with the
  target's rules, then :class:`ApplyCorrection` with the interrupted slot re-asked; a value that
  fails is re-asked once for the target (``correction_reask``), then the interrupted slot resumes. A choice
  correction is applied only if the caller named the option (``Turn.named``); otherwise it is
  confirmed first (``slot.confirm_option`` about the target). The
  re-ask's reply follows the rules of a normal answer to the target: an inferred value follows its
  ``confirm`` (a read-back, an option confirmation), and only a value a normal answer would accept
  (a deterministic yes to that question included) is applied; anything else (unclear, a no, the
  target out of tries) drops the correction and the interrupted slot resumes. These follow-ups
  are pending questions about the target (``pending.slot`` is not the slot asked). Of an unasked
  or unknown slot: ``none``. Of an ``earlier`` or ``verified`` slot:
  re-asked, ``earlier_strikes + 1``; a second in a row is an intent switch. Of a
  current slot outside ``owned`` (the confirm flow's AI arguments): an intent switch.
* **Counters.** Every question action uses one try of the slot it asks; a correction
  uses one try of its target and one of the interrupted slot (its re-ask). ``tries[s] >
  max_retries`` gives :class:`GiveUp`. Nothing decrements ``tries``. ``earlier_strikes`` counts
  *consecutive* earlier-flow corrections and is reset by any other answer.
* In ``deterministic`` mode the engine can't produce ``correction``, ``intent_switch`` or a model
  answer; if one arrives anyway it is ``none`` (fail closed).
* **Decline.** ``decline`` (from the resolver, or ``#``
  alone on the keypad) ends the flow with ``Command("decline")``, like ``cancel``, but only on a
  slot with ``declinable`` set; on any other slot it is ``none``. In ``deterministic`` mode
  only the keypad's ``#`` alone can give it (a model answer is ``none`` there, above).

Follow-up questions are about their own slot: the engine resolves a pending question's reply
against ``pending.slot``, so ``known``/``caller`` are that slot's options.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from .grounding import ground_candidates, ground_choice
from .types import CollectedSlot, OptionView, Pending, Resolution, SlotType
from .validators import validate

if TYPE_CHECKING:  # slot_policy imports the slots package; avoid the cycle at run time
    from owlbank.slot_policy import Mode, SlotSettings

ReaskKey = Literal["prompt", "clarify.digits", "clarify.choice", "clarify.yes_no"]
IntentReason = Literal["intent_switch", "earlier_correction", "not_owned"]
CommandKind = Literal["repeat", "escalate", "cancel", "decline"]

#: Sources whose answer is the caller's own words or keys, read by deterministic code.
DETERMINISTIC_SOURCES: frozenset[str] = frozenset({"dtmf", "speech", "alias", "ordinal",
                                                    "command"})

#: Two consecutive earlier-flow corrections are an intent switch.
EARLIER_STRIKES_LIMIT = 2


# ----------------------------------------------------------------------------- state


@dataclass(frozen=True, slots=True)
class Turn:
    """The dialog state of the slot being asked, as the front door holds it.

    ``settings`` and ``caller_options`` describe the collected slots a correction may name: their
    rules and ``max_retries``, and for a choice slot the caller's options. A
    correction of a slot missing from ``settings`` is ``none`` (fail closed). ``owned`` holds the
    slots this flow may change: every slot of ``login`` and ``stepup``, the card in ``confirm``."""

    slot: SlotSettings
    mode: Mode
    tries: Mapping[str, int]                  # per slot name, this flow
    pending: Pending | None
    collected: tuple[CollectedSlot, ...]
    earlier_strikes: int                      # Consecutive earlier-flow corrections
    owned: frozenset[str]                     # slots this flow may change
    settings: Mapping[str, SlotSettings] = field(default_factory=dict)
    caller_options: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The collected choice slots whose option the caller named in this answer (its spoken name or
    #: an alias), as the caller of :func:`decide` checked the words. A choice correction of
    #: any other slot is confirmed before it is applied.
    named: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.earlier_strikes < 0:
            raise ValueError("earlier_strikes must be >= 0")
        if any(v < 0 for v in self.tries.values()):
            raise ValueError("tries must be >= 0")
        p = self.pending
        if p is None:
            return
        if p.kind == "correction_reask" and p.slot == self.slot.name:
            raise ValueError("a correction_reask names a collected slot with settings")
        if p.slot != self.slot.name and p.slot not in self.settings:
            # A follow-up about another slot is a correction's: a collected slot with rules.
            raise ValueError("a follow-up question is about the slot asked or a collected slot")

    def correcting(self) -> bool:
        """True while the pending question is about a corrected slot, not the one asked."""
        return self.pending is not None and self.pending.slot != self.slot.name

    def answered(self) -> SlotSettings:
        """The slot the caller is answering: the pending question's, else the one asked."""
        p = self.pending
        if p is not None and p.slot != self.slot.name:
            return self.settings[p.slot]
        return self.slot

    def max_retries(self, name: str) -> int:
        if name == self.slot.name:
            return self.slot.decl.max_retries
        s = self.settings.get(name)
        return s.decl.max_retries if s is not None else 0


# ---------------------------------------------------------------------------- actions


@dataclass(frozen=True, slots=True)
class Accept:
    """Store ``value`` for ``slot`` (and verify it, for a factor). For a confirmed read-back the value
    is the pending's ``sealed_value`` as the front door supplied it."""

    slot: str
    value: str


@dataclass(frozen=True, slots=True)
class ReadBackDigits:
    """``slot.confirm_digits``. Never for a sensitive slot."""

    slot: str
    value: str
    key: str = "slot.confirm_digits"


@dataclass(frozen=True, slots=True)
class ConfirmOption:
    """``slot.confirm_option``: "Did you mean {option}?"."""

    slot: str
    option: str
    key: str = "slot.confirm_option"


@dataclass(frozen=True, slots=True)
class Disambiguate:
    """``slot.disambiguate`` over the candidates only."""

    slot: str
    candidates: tuple[str, ...]
    key: str = "slot.disambiguate"


@dataclass(frozen=True, slots=True)
class NotAvailable:
    """``slot.not_available_one|many``. ``named`` is a known option (spoken from the catalog,
    never the caller's words); ``on_file`` are the caller's options."""

    slot: str
    named: str
    on_file: tuple[str, ...]

    @property
    def key(self) -> str:
        return "slot.not_available_one" if len(self.on_file) == 1 else "slot.not_available_many"


@dataclass(frozen=True, slots=True)
class Reask:
    """Ask ``slot`` again: its own prompt (``prompt``) or a ``clarify.*`` template. ``consume`` is
    False only when the try was already counted (the interrupted slot resuming after a correction
    re-ask)."""

    key: ReaskKey
    slot: str
    consume: bool = True


@dataclass(frozen=True, slots=True)
class ApplyCorrection:
    """Replace ``target``'s value, say ``slot.corrected`` (``slot.corrected_sensitive`` for a
    sensitive target, no value), then ``then``."""

    target: str
    value: str
    sensitive: bool
    then: Reask

    @property
    def key(self) -> str:
        return "slot.corrected_sensitive" if self.sensitive else "slot.corrected"


@dataclass(frozen=True, slots=True)
class IntentSwitch:
    """End the flow and pass the utterance to the AI."""

    reason: IntentReason = "intent_switch"


@dataclass(frozen=True, slots=True)
class Command:
    """``repeat`` (the front door counts repeats, ``MAX_REPEATS``), ``escalate``, ``cancel`` or
    ``decline`` (the caller can't or won't give a declinable slot's value)."""

    kind: CommandKind


@dataclass(frozen=True, slots=True)
class GiveUp:
    """``tries[slot] > max_retries``: the flow's failure path."""

    slot: str


Action = (Accept | ReadBackDigits | ConfirmOption | Disambiguate | NotAvailable | ApplyCorrection
          | Reask | IntentSwitch | Command | GiveUp)

#: The actions that ask the caller a question about a slot.
QUESTION_ACTIONS: tuple[type, ...] = (ReadBackDigits, ConfirmOption, Disambiguate, NotAvailable,
                                      Reask)


# --------------------------------------------------------------------------- helpers


def _clarify(t: SlotType) -> ReaskKey:
    if t == "digits":
        return "clarify.digits"
    if t == "choice":
        return "clarify.choice"
    if t == "yes_no":
        return "clarify.yes_no"
    return "prompt"          # amount, date: no clarify template


def _canon(s: SlotSettings, value: object, source: str, options: Sequence[str]) -> str | None:
    """``value`` in ``s``'s canonical form if it is valid for ``s``, else ``None``. A choice is valid
    only as one of ``options``. A model digit value needs the exact ``constraints.length`` (as
    :func:`ground_digits`; fail closed)."""
    if not isinstance(value, str):
        return None
    t = s.decl.slot_type
    if t == "choice":
        return value if value in options else None
    canon = validate(s.decl, value)
    if canon is None:
        return None
    if t == "digits" and source not in DETERMINISTIC_SOURCES:
        length = s.decl.constraints.get("length")
        if isinstance(length, bool) or not isinstance(length, (int, str)):
            return None
        try:
            want = int(length)
        except ValueError:
            return None
        if len(canon) != want:
            return None
    return canon


def _yes_no(res: Resolution, known: tuple[str, ...]) -> tuple[Literal["yes", "no"] | None, bool]:
    """A yes/no reply and whether it is deterministic: ``reply`` if set, else a ``yes`` or
    ``no`` value that isn't an option (``exact`` from keypad 1/2 or the word lists is
    deterministic). The engine sends a reply with ``match`` ``exact`` (deterministic) or
    ``inferred`` (model); a reply on a ``none`` is ignored (fail closed)."""
    if res.match == "none":
        return None, False         # a none never selects, whatever its reply
    if res.reply is not None:
        return res.reply, res.reply_deterministic
    if (res.match in ("exact", "inferred") and res.value in ("yes", "no")
            and res.value not in known):
        det = res.match == "exact" and res.source in DETERMINISTIC_SOURCES
        return ("yes" if res.value == "yes" else "no"), det
    return None, False


class _Step:
    """Builds the result for one turn: counts tries and applies the give-up rule."""

    __slots__ = ("turn", "tries")

    def __init__(self, turn: Turn) -> None:
        self.turn = turn
        self.tries = dict(turn.tries)

    def bump(self, *names: str) -> str | None:
        """One try of each name; the first over its ``max_retries``, if any."""
        over = None
        for n in names:
            self.tries[n] = self.tries.get(n, 0) + 1
            if over is None and self.tries[n] > self.turn.max_retries(n):
                over = n
        return over

    def done(self, action: Action, *, pending: Pending | None = None,
             strikes: int = 0) -> tuple[Action, Turn]:
        return action, replace(self.turn, tries=self.tries, pending=pending,
                               earlier_strikes=strikes)

    def ask(self, action: Action, *names: str, pending: Pending | None = None,
            strikes: int = 0) -> tuple[Action, Turn]:
        """A question: one try of each of ``names``, or :class:`GiveUp` for the first over."""
        over = self.bump(*names)
        if over is not None:
            return self.done(GiveUp(over), strikes=strikes)
        return self.done(action, pending=pending, strikes=strikes)

    def reask(self, key: ReaskKey, slot: str, *, pending: Pending | None = None,
              strikes: int = 0) -> tuple[Action, Turn]:
        return self.ask(Reask(key, slot), slot, pending=pending, strikes=strikes)


# ----------------------------------------------------------------------------- decide


def decide(turn: Turn, res: Resolution, *, known: Sequence[OptionView],
           caller: Sequence[str]) -> tuple[Action, Turn]:
    """The next action for ``res`` and the next state. ``known`` and ``caller`` are the
    options of the slot being answered (:meth:`Turn.answered`); empty for a non-choice slot."""
    known_values = tuple(o.value for o in known)
    caller_t = tuple(caller)
    ans = turn.answered()
    res = _normalize(turn, ans, res, known_values)
    step = _Step(turn)
    m = res.match

    if m == "repeat":
        return step.done(Command("repeat"), pending=turn.pending, strikes=turn.earlier_strikes)
    if m in ("escalate", "cancel", "decline"):
        kind: CommandKind = "escalate" if m == "escalate" else "cancel" if m == "cancel" else "decline"
        return step.done(Command(kind))
    if m == "intent_switch":
        return step.done(IntentSwitch())
    p = turn.pending
    if m == "correction" and not (turn.correcting() and p is not None and res.target == p.slot):
        return _correction(step, res, known_values, caller_t)
    if turn.correcting():
        return _correction_follow_up(step, ans, res, known_values, caller_t)
    return _reply(step, ans, turn.pending, res, known_values, caller_t)


def _reply(step: _Step, ans: SlotSettings, p: Pending | None, res: Resolution,
           known: tuple[str, ...], caller: tuple[str, ...]) -> tuple[Action, Turn]:
    """An answer to ``ans``: a fresh answer (``p`` is ``None``), or the reply to the pending
    follow-up ``p`` about ``ans``."""
    if p is None or p.kind in ("not_available_many", "correction_reask"):
        return _answer(step, ans, res, known, caller)
    if p.kind == "confirm_digits":
        return _confirm_digits(step, ans, p, res, known, caller)
    if p.kind in ("confirm_option", "not_available_one"):
        return _names_option(step, ans, p, res, known, caller)
    return _disambiguate(step, ans, p, res, known, caller)


def _normalize(turn: Turn, ans: SlotSettings, res: Resolution,
               known_values: tuple[str, ...]) -> Resolution:
    """Fail-closed guards for ``deterministic`` mode, a ``decline`` on a slot that isn't
    declinable (``none``), then answer first."""
    if turn.mode == "deterministic" and (res.match in ("correction", "intent_switch")
                                         or res.source == "model"
                                         or (res.match == "decline" and res.source != "dtmf")):
        return Resolution(match="none", source="none")      # ′: only ``#`` alone declines here
    if res.match == "decline" and not ans.declinable:
        return replace(res, match="none")
    if res.value is None or res.match not in ("intent_switch", "repeat", "escalate", "cancel",
                                              "correction"):
        return res
    if res.match == "correction" and any(c.name == res.target for c in turn.collected):
        return res     # the value was grounded with the named target's rules
    options = known_values if ans.decl.slot_type == "choice" else ()
    if _canon(ans, res.value, res.source, options) is None:
        return res
    return replace(res, match="inferred", target=None)


def _answer(step: _Step, s: SlotSettings, res: Resolution, known: tuple[str, ...],
            caller: tuple[str, ...]) -> tuple[Action, Turn]:
    """A fresh answer to ``s`` (no follow-up pending)."""
    t = s.decl.slot_type
    m = res.match
    if m == "ambiguous":
        if t == "choice":
            cands = ground_candidates(list(res.candidates), caller=caller)
            if cands is not None:
                return step.ask(Disambiguate(s.name, cands), s.name,
                                pending=Pending("disambiguate", s.name, candidates=cands))
        return step.reask(_clarify(t), s.name)       # digits: never the candidates
    if m not in ("exact", "inferred") or res.value is None:
        return step.reask(_clarify(t), s.name)       # none
    if t == "choice":
        return _choice_value(step, s, res, known, caller)
    v = _canon(s, res.value, res.source, ())
    if v is None:
        return step.reask(_clarify(t), s.name)
    if t == "yes_no":
        if v == "yes" and res.source not in DETERMINISTIC_SOURCES:
            return step.reask("clarify.yes_no", s.name)    # No model-inferred yes
        return step.done(Accept(s.name, v))
    if not _needs_confirm(s, res):
        return step.done(Accept(s.name, v))
    if t == "digits":
        if s.sensitive:
            return step.reask("clarify.digits", s.name)
        return step.ask(ReadBackDigits(s.name, v), s.name,
                        pending=Pending("confirm_digits", s.name, sealed_value=v))
    return step.reask("prompt", s.name)    # amount, date: no read-back template (fail safe)


def _needs_confirm(s: SlotSettings, res: Resolution) -> bool:
    if res.match == "inferred":
        return s.confirm != "never"
    return res.source == "model" and s.confirm == "always"


def _choice_value(step: _Step, s: SlotSettings, res: Resolution, known: tuple[str, ...],
                  caller: tuple[str, ...]) -> tuple[Action, Turn]:
    g = ground_choice(res.value, known=known, caller=caller)
    if g == "none" or res.value is None:
        return step.reask("clarify.choice", s.name)
    v = res.value
    if g == "known_other":
        return _not_available(step, s, v, caller)
    if res.match == "inferred" or (res.source == "model" and s.confirm == "always"):
        # An inferred choice is confirmed even with confirm: never.
        return step.ask(ConfirmOption(s.name, v), s.name,
                        pending=Pending("confirm_option", s.name, option=v))
    return step.done(Accept(s.name, v))


def _not_available(step: _Step, s: SlotSettings, named: str,
                   caller: tuple[str, ...]) -> tuple[Action, Turn]:
    if not caller:
        return step.reask("clarify.choice", s.name)
    if len(caller) == 1:
        pending = Pending("not_available_one", s.name, option=caller[0])
    else:
        pending = Pending("not_available_many", s.name, candidates=caller)
    return step.ask(NotAvailable(s.name, named, caller), s.name, pending=pending)


def _confirm_digits(step: _Step, s: SlotSettings, p: Pending, res: Resolution,
                    known: tuple[str, ...], caller: tuple[str, ...]) -> tuple[Action, Turn]:
    """Reply to ``slot.confirm_digits``: a deterministic yes accepts; a new value is a fresh
    answer; no re-asks."""
    yn, det = _yes_no(res, ())
    if yn == "yes":
        if det and p.sealed_value is not None:
            return step.done(Accept(s.name, p.sealed_value))
        return step.reask("clarify.yes_no", s.name, pending=p)
    if res.match == "ambiguous" or (res.match in ("exact", "inferred")
                                    and res.value not in (None, "yes", "no")):
        return _answer(step, s, res, known, caller)                # a fresh answer
    if yn == "no":
        return step.reask("prompt", s.name)
    return step.reask("clarify.yes_no", s.name, pending=p)


def _names_option(step: _Step, s: SlotSettings, p: Pending, res: Resolution,
                  known: tuple[str, ...], caller: tuple[str, ...]) -> tuple[Action, Turn]:
    """Reply to a yes/no question that names ``p.option``: ``slot.confirm_option`` or
    ``slot.not_available_one``."""
    x = p.option
    yn, det = _yes_no(res, known)
    v = res.value if res.value not in ("yes", "no") or res.value in known else None
    if res.match in ("exact", "inferred") and v is not None and v != x:
        return _names_other(step, s, res, v, known, caller)        # "no, I meant Y"
    if res.match == "exact" and v is not None and v == x and v in caller:
        return step.done(Accept(s.name, v))                        # the caller named it
    if yn == "yes":
        if det and x is not None and x in caller:
            return step.done(Accept(s.name, x))
        return step.reask("clarify.yes_no", s.name, pending=p)
    if res.match == "ambiguous":
        cands = ground_candidates(list(res.candidates), caller=caller)
        if cands is not None:
            return step.ask(Disambiguate(s.name, cands), s.name,
                            pending=Pending("disambiguate", s.name, candidates=cands))
        return step.reask("clarify.choice", s.name)
    if yn == "no":
        return step.reask("prompt", s.name)
    return step.reask("clarify.yes_no", s.name, pending=p)


def _names_other(step: _Step, s: SlotSettings, res: Resolution, y: str, known: tuple[str, ...],
                 caller: tuple[str, ...]) -> tuple[Action, Turn]:
    """A reply naming another option Y. Each such reply uses one try."""
    g = ground_choice(y, known=known, caller=caller)
    if g == "caller":
        if res.match == "exact":
            step.bump(s.name)
            return step.done(Accept(s.name, y))
        return step.ask(ConfirmOption(s.name, y), s.name,
                        pending=Pending("confirm_option", s.name, option=y))
    if g == "known_other":
        return _not_available(step, s, y, caller)
    return step.reask("clarify.choice", s.name)


def _disambiguate(step: _Step, s: SlotSettings, p: Pending, res: Resolution,
                  known: tuple[str, ...], caller: tuple[str, ...]) -> tuple[Action, Turn]:
    """Reply to ``slot.disambiguate``: resolved against the candidates."""
    v = res.value
    if res.match in ("exact", "inferred") and v is not None:
        g = ground_choice(v, known=known, caller=caller)
        if g == "caller" and res.match == "exact":
            return step.done(Accept(s.name, v))                    # named exactly
        if g == "caller" and v in p.candidates:
            return step.ask(ConfirmOption(s.name, v), s.name,
                            pending=Pending("confirm_option", s.name, option=v))
        if g == "known_other":
            return _not_available(step, s, v, caller)
        return step.reask("clarify.choice", s.name)
    if res.match == "ambiguous":
        cands = ground_candidates(list(res.candidates), caller=caller)
        if cands is not None:
            return step.ask(Disambiguate(s.name, cands), s.name,
                            pending=Pending("disambiguate", s.name, candidates=cands))
    return step.reask("clarify.choice", s.name)


# ------------------------------------------------------------------------ corrections


def _correction(step: _Step, res: Resolution, known: tuple[str, ...],
                caller: tuple[str, ...]) -> tuple[Action, Turn]:
    turn = step.turn
    cur = turn.slot
    target = next((c for c in turn.collected if c.name == res.target), None)
    if target is None or target.name == cur.name:
        # An unasked or unknown slot is `none` for the slot being answered.
        if turn.correcting():
            return _correction_follow_up(step, turn.answered(),
                                         Resolution(match="none", source=res.source), known, caller)
        return _reply(step, turn.answered(), turn.pending,
                      Resolution(match="none", source=res.source), known, caller)
    if target.scope == "earlier" or target.verified:
        strikes = turn.earlier_strikes + 1
        if strikes >= EARLIER_STRIKES_LIMIT:
            return step.done(IntentSwitch("earlier_correction"), strikes=strikes)
        return step.reask(_clarify(cur.decl.slot_type), cur.name, strikes=strikes)
    if target.name not in turn.owned:
        return step.done(IntentSwitch("not_owned"))                 # 6.7
    ts = turn.settings.get(target.name)
    if ts is None:
        return step.reask(_clarify(cur.decl.slot_type), cur.name)   # no rules to ground with
    v = _canon(ts, res.value, res.source, turn.caller_options.get(target.name, ()))
    if v is None:
        # Re-asked once for the corrected slot. The interrupted slot's try is used now.
        return step.ask(Reask(_clarify(ts.decl.slot_type), ts.name), ts.name, cur.name,
                        pending=Pending("correction_reask", ts.name))
    if ts.decl.slot_type == "choice" and ts.name not in turn.named:
        # An option the caller didn't name is confirmed before it is applied. The
        # interrupted slot's try is used now, as for the re-ask.
        return step.ask(ConfirmOption(ts.name, v), ts.name, cur.name,
                        pending=Pending("confirm_option", ts.name, option=v))
    return step.ask(ApplyCorrection(ts.name, v, ts.sensitive, then=Reask("prompt", cur.name)),
                    ts.name, cur.name)


def _correction_follow_up(step: _Step, ts: SlotSettings, res: Resolution, known: tuple[str, ...],
                          caller: tuple[str, ...]) -> tuple[Action, Turn]:
    """Reply to a question about the corrected slot ``ts``: its re-ask or a follow-up of it
    (a read-back, an option confirmation, a disambiguation, a not-available question). The reply
    follows the rules of a normal answer to ``ts``: a value a normal answer would accept is
    applied (:class:`ApplyCorrection`) and the interrupted slot resumes; a value that needs a
    question first (``confirm``) gets that question about ``ts``, with one try of
    ``ts``; anything else (unclear, a no, ``ts`` out of tries) drops the correction and the
    interrupted slot resumes. The interrupted slot's try was used on the correction turn."""
    turn = step.turn
    p = turn.pending
    resume = Reask("prompt", turn.slot.name, consume=False)
    if res.match == "correction" and res.target == ts.name:
        res = replace(res, match="inferred", target=None)     # a model's value for ts
    inner = _Step(turn)
    follow_up = None if p is None or p.kind == "correction_reask" else p
    action, nxt = _reply(inner, ts, follow_up, res, known, caller)
    if isinstance(action, Accept) and action.slot == ts.name:
        return step.done(ApplyCorrection(ts.name, action.value, ts.sensitive, then=resume))
    if (isinstance(action, (ReadBackDigits, ConfirmOption, Disambiguate, NotAvailable))
            and action.slot == ts.name and nxt.pending is not None):
        return action, replace(nxt, earlier_strikes=0)
    return step.done(resume)


__all__ = [
    "DETERMINISTIC_SOURCES", "EARLIER_STRIKES_LIMIT", "QUESTION_ACTIONS", "Accept", "Action",
    "ApplyCorrection", "Command", "ConfirmOption", "Disambiguate", "GiveUp", "IntentSwitch",
    "NotAvailable", "ReadBackDigits", "Reask", "Turn", "decide",
]
