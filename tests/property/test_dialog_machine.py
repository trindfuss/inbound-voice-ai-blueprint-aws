"""Property tests for the dialog policy.

A ``RuleBasedStateMachine`` drives:func:`decide` with arbitrary resolutions (adversarial values
included: options that aren't known or on file, digit strings of any length, replies on any match,
targets that aren't slots) over generated flows (slot type, sensitivity, ``confirm``,
``max_retries``, mode, collected slots in every scope and verification state, ``owned``):

* Every selected option is named in the answer that selected it (an ``exact`` match of
  that option, or a correction the front door found named: ``Turn.named``) or is the option of the
  pending question answered with a deterministic yes; a ``none`` never selects.
  This covers corrected choices (``ApplyCorrection``) too, and the corrected slot's re-ask and its
  follow-ups: they follow the target's rules, so an inferred digit value of a slot whose
  ``confirm`` asks is applied only after a deterministic yes to its read-back.
* Questions per slot <= ``max_retries + 1`` (repeats and keypad reminders are the front
  door's and never questions here); no action decreases ``tries``. The corrector clause
  is a separate ``@given`` test.
* A correction of a ``verified`` slot never yields ``Accept`` or
  ``ApplyCorrection`` and changes no ``tries`` but the asked slot's.
* No action stores a value for a slot that isn't the answered one or a collected one; two
  consecutive earlier-flow corrections give ``IntentSwitch``.
* An accepted or corrected choice is a caller option; candidates are >= 2
  distinct caller options; a not-available question names a known option that isn't on file. Also 9.3 (no read-back of a sensitive slot) and 4.4 (no digit candidates).
* A ``decline`` on a declinable slot is
  ``Command("decline")`` with no try used, from the keypad in any mode and from the model outside
  ``deterministic``; on any other slot it gives exactly what ``none`` gives; in ``deterministic``
  mode only a keypad decline ends the flow.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from owlbank.slot_policy import CONFIRM_POLICIES, MODES, OptionDef, SlotSettings
from owlbank.slots.dialog import (
    DETERMINISTIC_SOURCES,
    Accept,
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
from owlbank.slots.validators import validate
from owlbank.slots.types import (
    MATCHES,
    SOURCES,
    CollectedSlot,
    OptionView,
    Resolution,
    SlotDecl,
)

KNOWN = (OptionView("red", "OwlBank Red Card", ("red",)),
         OptionView("purple", "OwlBank Purple Card", ("purple",)),
         OptionView("rewards", "OwlBank Rewards Card", ("rewards",)),
         OptionView("gold", "OwlBank Gold Card", ("gold",)))
KNOWN_VALUES = tuple(o.value for o in KNOWN)
#: Slot names the flows draw from; "amount" and "ghost" are never collected settings.
NAMES = ("zip", "dob", "ssn4", "card", "confirm_payment")
TARGETS = (*NAMES, "amount", "ghost", "", None)

_SHAPES: dict[str, tuple[str, int | None]] = {
    "zip": ("digits", 5), "dob": ("digits", 8), "ssn4": ("digits", 4), "card": ("choice", None),
    "confirm_payment": ("yes_no", None),
}


def _settings(name: str, sensitive: bool, confirm: str, max_retries: int) -> SlotSettings:
    t, length = _SHAPES[name]
    if t == "digits":
        decl = SlotDecl("digits", {"length": length}, max_retries=max_retries)
        return SlotSettings(name=name, decl=decl, is_factor=True, sensitive=sensitive,
                            confirm=confirm)  # type: ignore[arg-type]
    if t == "choice":
        decl = SlotDecl("choice", {"options": [{"value": v} for v in KNOWN_VALUES]},
                        max_retries=max_retries)
        return SlotSettings(name=name, decl=decl, is_factor=False, sensitive=False,
                            confirm=confirm,  # type: ignore[arg-type]
                            options=tuple(OptionDef(v) for v in KNOWN_VALUES))
    return SlotSettings(name=name, decl=SlotDecl("yes_no", max_retries=max_retries),
                        is_factor=False, sensitive=False)


@dataclass
class Flow:
    turn: Turn
    caller: dict[str, tuple[str, ...]]        # caller options per choice slot (current included)


@st.composite
def flows(draw: st.DrawFn) -> Flow:
    current = draw(st.sampled_from(NAMES))
    def slot(name: str) -> SlotSettings:
        return _settings(name, draw(st.booleans()), draw(st.sampled_from(CONFIRM_POLICIES)),
                         draw(st.integers(0, 4)))
    cur = slot(current)
    others = [n for n in NAMES if n != current]
    picked = draw(st.lists(st.sampled_from(others), unique=True, max_size=len(others)))
    collected = tuple(CollectedSlot(name=n, slot_type=_SHAPES[n][0],  # type: ignore[arg-type]
                                    label=n, scope=draw(st.sampled_from(("current", "earlier"))),
                                    verified=draw(st.sampled_from((False, False, True))))
                      for n in picked)
    settings_ = {c.name: slot(c.name) for c in collected}
    # An adversarial flow may collect a slot whose settings the front door didn't pass.
    drop: bool = draw(st.booleans())
    if collected and drop:
        settings_.pop(draw(st.sampled_from([c.name for c in collected])))
    caller = {"card": tuple(draw(st.lists(st.sampled_from(KNOWN_VALUES), unique=True, max_size=4)))}
    tries = {current: draw(st.integers(0, min(1, cur.decl.max_retries)))}
    for n, s in settings_.items():
        tries[n] = draw(st.integers(0, s.decl.max_retries))
    owned = frozenset(draw(st.one_of(st.just(list(NAMES)),
                                     st.lists(st.sampled_from([*NAMES, "amount"]), unique=True))))
    named = frozenset(n for n in settings_ if _SHAPES[n][0] == "choice" and draw(st.booleans()))
    turn = Turn(slot=cur, mode=draw(st.sampled_from(MODES)), tries=tries, pending=None,
                collected=collected, earlier_strikes=0, owned=owned, settings=settings_,
                caller_options={"card": caller["card"]}, named=named)
    return Flow(turn, caller)


_values = st.one_of(st.none(), st.sampled_from([*KNOWN_VALUES, "platinum", "yes", "no", ""]),
                    st.text("0123456789", min_size=1, max_size=9), st.text(max_size=6))


@st.composite
def resolutions(draw: st.DrawFn) -> Resolution:
    return Resolution(
        match=draw(st.sampled_from(MATCHES)),
        source=draw(st.sampled_from(SOURCES)),
        value=draw(_values),
        candidates=tuple(draw(st.lists(st.sampled_from([*KNOWN_VALUES, "platinum", "12345"]),
                                       max_size=4))),
        target=draw(st.sampled_from(TARGETS)),
        reply=draw(st.sampled_from((None, "yes", "no"))),
        reply_deterministic=draw(st.booleans()),
    )


@st.composite
def plausible(draw: st.DrawFn, turn: Turn, caller: tuple[str, ...]) -> Resolution:
    """A resolution shaped for the slot being answered, so the machine reaches every follow-up."""
    ans = turn.answered()
    t = ans.decl.slot_type
    if t == "digits":
        n = int(ans.decl.constraints["length"])
        value: st.SearchStrategy[str | None] = st.one_of(
            st.text("0123456789", min_size=n, max_size=n), st.sampled_from(("yes", "no")),
            st.none())
    elif t == "choice":
        value = st.sampled_from((*KNOWN_VALUES, "platinum", None))
    else:
        value = st.sampled_from(("yes", "no", None))
    names = [c.name for c in turn.collected]
    match = draw(st.sampled_from(("exact", "exact", "inferred", "inferred", "ambiguous", "none",
                                  "correction", "intent_switch", "repeat")))
    target = draw(st.sampled_from(names)) if match == "correction" and names else None
    if target is not None:
        tt = _SHAPES[target][0]
        value = (st.sampled_from((*KNOWN_VALUES, None)) if tt == "choice"
                 else st.sampled_from(("94105", "12345678", "1234", "yes", "12")))
    return Resolution(
        match=match,  # type: ignore[arg-type]
        source=draw(st.sampled_from(("dtmf", "speech", "alias", "model", "model"))),
        value=draw(value),
        candidates=tuple(draw(st.lists(st.sampled_from(caller or KNOWN_VALUES), max_size=3))),
        target=target,
        reply=draw(st.sampled_from((None, "yes", "no"))),
        reply_deterministic=draw(st.booleans()),
    )


def _det_yes(r: Resolution) -> bool:
    if r.match == "none":
        return False
    if r.reply is not None:
        return r.reply == "yes" and r.reply_deterministic
    return r.match == "exact" and r.value == "yes" and r.source in DETERMINISTIC_SOURCES


SELECTS = (Accept, ApplyCorrection)


class DialogMachine(RuleBasedStateMachine):
    """One flow at a time; a new flow starts when one ends."""

    def __init__(self) -> None:
        super().__init__()
        self.flow: Flow | None = None
        self.done = True
        self.questions: dict[str, int] = {}
        self.last_earlier = False

    def _start(self, flow: Flow) -> None:
        self.flow = flow
        self.done = False
        self.last_earlier = False
        t = flow.turn
        # Each slot was asked once, plus once per try already used.
        self.questions = {n: v + 1 for n, v in t.tries.items()}

    @initialize(flow=flows())
    def first(self, flow: Flow) -> None:
        self._start(flow)

    @precondition(lambda self: self.done)
    @rule(flow=flows())
    def next_flow(self, flow: Flow) -> None:
        self._start(flow)

    @precondition(lambda self: not self.done)
    @rule(r=resolutions())
    def adversarial(self, r: Resolution) -> None:
        self._answer(r)

    @precondition(lambda self: not self.done)
    @rule(data=st.data())
    def answer(self, data: st.DataObject) -> None:
        assert self.flow is not None
        turn = self.flow.turn
        caller = self.flow.caller.get(turn.answered().name, ())
        self._answer(data.draw(plausible(turn, caller)))

    @precondition(lambda self: not self.done and self.flow is not None
                  and self.flow.turn.pending is not None)
    @rule(match=st.sampled_from(("exact", "inferred", "none")), with_option=st.booleans(),
          reply=st.sampled_from(("yes", "no", None)), det=st.booleans(),
          source=st.sampled_from(("dtmf", "speech", "model")))
    def reply(self, match: str, with_option: bool, reply: str | None, det: bool,
              source: str) -> None:
        """A yes/no reply to the pending follow-up, naming its option or not."""
        assert self.flow is not None
        p = self.flow.turn.pending
        assert p is not None
        value = p.option if with_option else (reply if reply is not None and not det else None)
        self._answer(Resolution(match=match, source=source,  # type: ignore[arg-type]
                                value=value, reply=reply,  # type: ignore[arg-type]
                                reply_deterministic=det))

    @precondition(lambda self: not self.done and self.flow is not None
                  and bool(self.flow.turn.collected))
    @rule(data=st.data(), ok=st.booleans())
    def correct(self, data: st.DataObject, ok: bool) -> None:
        """A correction of a collected slot, with a value valid for it or not."""
        assert self.flow is not None
        c = data.draw(st.sampled_from(self.flow.turn.collected))
        good = {"zip": "94105", "dob": "19800101", "ssn4": "1234", "confirm_payment": "no",
                "card": data.draw(st.sampled_from(KNOWN_VALUES))}[c.name]
        self._answer(Resolution(match="correction", source="model", model="ok", target=c.name,
                                value=good if ok else "1"))

    @precondition(lambda self: not self.done and self.flow is not None
                  and self.flow.turn.correcting())
    @rule(data=st.data(), match=st.sampled_from(("exact", "inferred", "correction")),
          source=st.sampled_from(("speech", "alias", "model")))
    def reask_reply(self, data: st.DataObject, match: str, source: str) -> None:
        """A value valid for the corrected slot, in reply to its re-ask or a follow-up."""
        assert self.flow is not None
        turn = self.flow.turn
        ts = turn.answered()
        value = {"zip": "94105", "dob": "19800101", "ssn4": "1234", "confirm_payment": "no",
                 "card": data.draw(st.sampled_from(KNOWN_VALUES))}[ts.name]
        self._answer(Resolution(match=match, source=source,  # type: ignore[arg-type]
                                value=value, target=ts.name if match == "correction" else None))

    def _answer(self, r: Resolution) -> None:
        assert self.flow is not None
        prev = self.flow.turn
        ans = prev.answered()
        is_choice = ans.decl.slot_type == "choice"
        caller = self.flow.caller.get(ans.name, ()) if is_choice else ()
        known = KNOWN if is_choice else ()
        action, nxt = decide(prev, r, known=known, caller=caller)
        model_ok = prev.mode != "deterministic"
        collected = {c.name: c for c in prev.collected}

        # Nothing decrements a try.
        for k, v in prev.tries.items():
            assert nxt.tries.get(k, 0) >= v, (k, prev.tries, nxt.tries)

        # A selection is named or confirmed with a deterministic yes; a none never selects.
        if r.match == "none":
            assert not isinstance(action, SELECTS), action
        p = prev.pending
        # Routing as decide() does it: a correction naming another slot than the one being re-asked
        # is a fresh correction, even mid re-ask; anything else while correcting is a reply
        # to the re-ask or its follow-up.
        fresh = r.match == "correction" and not (prev.correcting() and p is not None
                                                   and r.target == p.slot)
        follow_up = prev.correcting() and not fresh
        if isinstance(action, Accept) and is_choice:
            named = r.match == "exact" and r.value == action.value
            confirmed = (p is not None and p.option == action.value and _det_yes(r))
            assert named or confirmed, (action, r, p)
        if isinstance(action, ApplyCorrection) and collected[action.target].slot_type == "choice":
            # A corrected choice: named in a correction (``named``), named exactly in the reply to
            # the target's re-ask or follow-up, or the option a deterministic yes confirmed.
            named = (fresh and r.target == action.target and action.target in prev.named)
            said = (follow_up and r.match == "exact" and r.value == action.value)
            confirmed = (p is not None and p.slot == action.target and p.option == action.value
                         and _det_yes(r))
            assert named or said or confirmed, (action, r, p, prev.named)
        if isinstance(action, ApplyCorrection) and follow_up:
            # The re-ask follows the target's rules: a model-inferred digit value of a slot
            # whose confirm asks is applied only from its read-back, with a deterministic yes.
            ts = prev.settings[action.target]
            inferred = r.match in ("inferred", "correction") or (
                r.source == "model" and ts.confirm == "always")
            if ts.decl.slot_type == "digits" and inferred and ts.confirm != "never":
                assert (p is not None and p.kind == "confirm_digits"
                        and p.sealed_value == action.value and _det_yes(r)), (action, r, p)
            assert p is not None and p.slot == action.target
        # Read-backs and yes_no slots.
        if isinstance(action, Accept) and ans.decl.slot_type == "yes_no" and action.value == "yes":
            assert r.source in DETERMINISTIC_SOURCES
        p = prev.pending
        if (isinstance(action, Accept) and p is not None and p.kind == "confirm_digits"
                and action.value == p.sealed_value and r.value != p.sealed_value):
            assert _det_yes(r), r

        # An option question is about its own slot: a corrected card's
        # confirmation is asked at another slot's prompt, with the card's options.
        if isinstance(action, (ConfirmOption, Disambiguate, NotAvailable)):
            caller = self.flow.caller.get(action.slot, ())
            is_choice = (prev.settings.get(action.slot, prev.slot).decl.slot_type == "choice")
        if isinstance(action, Accept) and is_choice:
            assert action.value in caller
        if isinstance(action, ApplyCorrection) and collected[action.target].slot_type == "choice":
            assert action.value in prev.caller_options.get(action.target, ())
        if isinstance(action, Disambiguate):
            assert is_choice                                         # 4.4: never digits
            assert len(set(action.candidates)) == len(action.candidates) >= 2
            assert set(action.candidates) <= set(caller)
        if isinstance(action, NotAvailable):
            assert action.named in KNOWN_VALUES and action.named not in caller
            assert action.on_file == caller and caller
        if isinstance(action, ConfirmOption):
            assert action.option in caller
        if isinstance(action, ReadBackDigits):
            assert not prev.settings.get(action.slot, prev.slot).sensitive    # 9.3

        # A stored or read-back digit value fits its slot exactly.
        stored = None
        if isinstance(action, (Accept, ReadBackDigits)):
            stored = (prev.settings.get(action.slot, prev.slot), action.value)
        elif isinstance(action, ApplyCorrection):
            stored = (prev.settings[action.target], action.value)
        if stored is not None and stored[0].decl.slot_type == "digits":
            decl, value = stored[0].decl, stored[1]
            assert validate(decl, value) == value
            assert len(value) == int(decl.constraints["length"])

        # Values only for the answered slot or a collected, current, owned, unverified one.
        if isinstance(action, Accept):
            assert action.slot == ans.name
        if isinstance(action, ApplyCorrection):
            c = collected[action.target]
            assert c.scope == "current" and not c.verified and c.name in prev.owned
            assert action.target in prev.settings

        is_correction = model_ok and r.match == "correction" and r.target in collected
        earlier = is_correction and (collected[str(r.target)].scope == "earlier"
                                     or collected[str(r.target)].verified)
        # A correction of a verified slot neither accepts nor verifies.
        if is_correction and collected[str(r.target)].verified:
            assert not isinstance(action, SELECTS), action
            for k in set(prev.tries) | set(nxt.tries):
                if k != prev.slot.name:
                    assert nxt.tries.get(k, 0) == prev.tries.get(k, 0), k
        # Two consecutive earlier-flow corrections are an intent switch.
        if earlier and self.last_earlier:
            assert action == IntentSwitch("earlier_correction"), action
        if isinstance(action, Command) and action.kind == "repeat":
            pass                                                     # not an answer
        else:
            self.last_earlier = earlier and isinstance(action, Reask)

        # Questions per slot <= max_retries + 1.
        asked: str | None = None
        if isinstance(action, (ReadBackDigits, ConfirmOption, Disambiguate, NotAvailable, Reask)):
            asked = action.slot
        elif isinstance(action, ApplyCorrection):
            asked = action.then.slot
        if asked is not None:
            self.questions[asked] = self.questions.get(asked, 1) + 1
            assert self.questions[asked] <= prev.max_retries(asked) + 1, (asked, self.questions)
            assert nxt.tries.get(asked, 0) <= prev.max_retries(asked)

        self.flow = Flow(nxt, self.flow.caller)
        if isinstance(action, (GiveUp, IntentSwitch)) or (
                isinstance(action, Command) and action.kind != "repeat") or (
                isinstance(action, Accept) and action.slot == prev.slot.name):
            self.done = True

    @invariant()
    def strikes_bounded(self) -> None:
        if self.flow is not None:
            assert 0 <= self.flow.turn.earlier_strikes <= 2


DialogMachine.TestCase.settings = settings(max_examples=300, stateful_step_count=25,
                                           deadline=None)
TestDialogMachine = DialogMachine.TestCase


# ------------------------------------------------- The corrector gives up no sooner


def _run(turn: Turn, answer: Any, limit: int = 200) -> tuple[int, int]:
    """Caller turns and questions of the asked slot until the flow ends."""
    turns = 0
    questions = 1
    cur = turn.slot.name
    while turns < limit:
        turns += 1
        action, turn = decide(turn, answer(turn), known=(), caller=())
        if isinstance(action, GiveUp | IntentSwitch | Accept):
            return turns, questions
        if isinstance(action, Reask) and action.slot == cur:
            questions += 1
        if isinstance(action, ApplyCorrection) and action.then.slot == cur:
            questions += 1
    raise AssertionError("the flow never ended")


@settings(max_examples=300)
@given(cur_retries=st.integers(0, 4), target_retries=st.integers(0, 10),
       valid=st.lists(st.booleans(), min_size=1), mode=st.sampled_from(("assisted", "generative")))
def test_corrector_gives_up_no_later_than_the_wrong_answerer(
        cur_retries: int, target_retries: int, valid: list[bool], mode: str) -> None:
    cur = _settings("ssn4", True, "never", cur_retries)
    zip_ = _settings("zip", False, "when_inferred", target_retries)
    base = Turn(slot=cur, mode=mode, tries={}, pending=None,  # type: ignore[arg-type]
                collected=(CollectedSlot("zip", "digits", "ZIP", "current", False),),
                earlier_strikes=0, owned=frozenset({"ssn4", "zip"}), settings={"zip": zip_})
    wrong = _run(base, lambda t: Resolution(match="none", source="none"))
    # The wrong answerer types a wrong value each time: unclear, re-asked, until it gives up.
    assert wrong == (cur_retries + 1, cur_retries + 1)
    i = iter(valid * 50)

    def corrector(t: Turn) -> Resolution:
        ok = next(i)
        if t.pending is not None:      # the corrected slot's re-ask
            return Resolution(match="exact", source="speech", value="94105" if ok else "1")
        return Resolution(match="correction", source="model", model="ok", target="zip",
                          value="94105" if ok else "123")
    turns, questions = _run(base, corrector)
    assert questions <= wrong[1]
    if all(valid):
        assert turns <= wrong[0]


# ------------------------- A correction and its re-ask follow the target's rules


def _replies(values: tuple[str | None, ...], target: str) -> st.SearchStrategy[Resolution]:
    """Replies shaped for the corrected slot: mostly a value valid for it, or a yes/no."""
    return st.builds(
        Resolution,
        match=st.sampled_from(("exact", "exact", "inferred", "inferred", "ambiguous", "none",
                               "correction")),
        source=st.sampled_from(("dtmf", "speech", "alias", "model", "model")),
        value=st.sampled_from(values),
        candidates=st.lists(st.sampled_from(KNOWN_VALUES), max_size=3).map(tuple),
        target=st.sampled_from((target, target, None)),
        reply=st.sampled_from((None, None, "yes", "no")),
        reply_deterministic=st.booleans())


@settings(max_examples=400, deadline=None)
@given(named=st.booleans(), on_file=st.lists(st.sampled_from(KNOWN_VALUES), unique=True,
                                              min_size=1, max_size=4).map(tuple),
       first=st.sampled_from([*KNOWN_VALUES, "platinum", None]), retries=st.integers(1, 4),
       replies=st.lists(_replies((*KNOWN_VALUES, "platinum", "yes", None), "card"), max_size=6))
def test_a_corrected_card_is_named_or_confirmed(named: bool, on_file: tuple[str, ...],
                                                first: str | None, retries: int,
                                                replies: list[Resolution]) -> None:
    """At the read-back, the model corrects the card. Whatever follows, a card is applied only if
    the correction named it, a reply named it exactly, or a deterministic yes confirmed the option
    the last question named; the read-back's slot never moves (no Accept of it)."""
    card = _settings("card", False, "when_inferred", retries)
    yn = _settings("confirm_payment", False, "when_inferred", 2)
    t = Turn(slot=yn, mode="assisted", tries={}, pending=None,
             collected=(CollectedSlot("card", "choice", "card", "current", False),),
             earlier_strikes=0, owned=frozenset({"card"}), settings={"card": card},
             caller_options={"card": on_file}, named=frozenset({"card"}) if named else frozenset())
    rs = [Resolution(match="correction", source="model", model="ok", target="card", value=first),
          *replies]
    for r in rs:
        is_card = t.answered().name == "card"
        action, nxt = decide(t, r, known=KNOWN if is_card else (),
                             caller=on_file if is_card else ())
        if isinstance(action, ApplyCorrection):
            p = t.pending
            by_name = (r.match == "correction" and named and not t.correcting()
                       and r.value == action.value)
            said = t.correcting() and r.match == "exact" and r.value == action.value
            confirmed = p is not None and p.option == action.value and _det_yes(r)
            assert by_name or said or confirmed, (action, r, p)
            assert action.value in on_file
        if isinstance(action, Accept):
            assert action.slot == "confirm_payment"
        if isinstance(action, (GiveUp, IntentSwitch, Command, Accept)):
            return
        t = replace(nxt, named=t.named)


@settings(max_examples=300, deadline=None)
@given(confirm=st.sampled_from(CONFIRM_POLICIES), sensitive=st.booleans(),
       retries=st.integers(1, 4),
       replies=st.lists(_replies(("94105", "12345", "1", "yes", None), "zip"), min_size=1,
                        max_size=6))
def test_a_corrected_digit_value_follows_the_targets_confirm(
        confirm: str, sensitive: bool, retries: int, replies: list[Resolution]) -> None:
    """After a ZIP correction that failed, the ZIP's re-ask follows the ZIP's rules: a model value
    is applied unread only with ``confirm: never``; otherwise only the value of a read-back the
    caller said a deterministic yes to; a sensitive ZIP is never read back."""
    zip_ = _settings("zip", sensitive, confirm, retries)
    ssn = _settings("ssn4", True, "never", 2)
    t = Turn(slot=ssn, mode="assisted", tries={}, pending=None,
             collected=(CollectedSlot("zip", "digits", "ZIP", "current", False),),
             earlier_strikes=0, owned=frozenset({"ssn4", "zip"}), settings={"zip": zip_})
    action, t = decide(t, Resolution(match="correction", source="model", model="ok",
                                     target="zip", value="1"), known=(), caller=())
    if not isinstance(action, Reask):
        return                                                       # out of tries
    for r in replies:
        prev = t
        action, t = decide(prev, r, known=(), caller=())
        if isinstance(action, ReadBackDigits):
            assert not sensitive and action.slot == "zip"
        if isinstance(action, ApplyCorrection) and prev.correcting():
            p = prev.pending
            assert p is not None
            inferred = r.match in ("inferred", "correction")
            if (inferred and confirm != "never") or (r.source == "model" and confirm == "always"):
                assert (p.kind == "confirm_digits" and p.sealed_value == action.value
                        and _det_yes(r)), (action, r, p)
        if not prev.correcting() or isinstance(action, (GiveUp, IntentSwitch, Command)):
            return


# ------------------------------------------------------------------ (decline)


def _with_declinable(flow: Flow, on: bool) -> Turn:
    return replace(flow.turn, slot=replace(flow.turn.slot, declinable=on))


def _decide_flow(turn: Turn, res: Resolution, caller: dict[str, tuple[str, ...]]):
    choice = turn.answered().decl.slot_type == "choice"
    return decide(turn, res, known=KNOWN if choice else (),
                  caller=caller.get(turn.answered().name, ()) if choice else ())


@settings(max_examples=400, deadline=None)
@given(flow=flows(), res=resolutions(), source=st.sampled_from(SOURCES))
def test_a_decline_ends_a_declinable_slot(flow: Flow, res: Resolution, source: str) -> None:
    turn = _with_declinable(flow, True)
    res = replace(res, match="decline", source=source, value=None)  # type: ignore[arg-type]
    action, nxt = _decide_flow(turn, res, flow.caller)
    if turn.mode == "deterministic" and source != "dtmf":
        assert action != Command("decline")
        return
    assert action == Command("decline")
    assert dict(nxt.tries) == dict(turn.tries)


@settings(max_examples=400, deadline=None)
@given(flow=flows(), res=resolutions())
def test_a_decline_is_none_on_a_slot_that_isnt_declinable(flow: Flow, res: Resolution) -> None:
    turn = _with_declinable(flow, False)
    declined = _decide_flow(turn, replace(res, match="decline"), flow.caller)
    none = _decide_flow(turn, replace(res, match="none"), flow.caller)
    assert declined == none
    assert declined[0] != Command("decline")
