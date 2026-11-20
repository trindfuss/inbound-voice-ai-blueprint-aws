"""Grounding: a model-proposed value counts only if the caller actually said it.

Pure functions, no I/O. The resolver's word is never trusted:

* digits: the value has exactly ``constraints.length`` digits and is an ordered
  subsequence of the utterance's digit stream. There is no bound on how many digits the words
  hold, so "23456, oh, no actually it is 12345" grounds 12345;
* choices: the value is a known option, and is accepted only if it is one of the
  caller's; ``candidates`` are 2 or more caller options; :func:`named_in` says whether an option's
  spoken name or an alias was said, so a model ``exact`` can be demoted to ``inferred``;
* :func:`grounded_runs` locates the number runs that hold the value, for
  :func:`owlbank.sensitive.placeholder_text`.

A run of number tokens that doesn't read as a whole is split at the language's joiners ("y",
"and"; :func:`read_at_joiners`) and each piece is read, so Spanish "uno, dos, tres y cuatro"
grounds 1234. A run that reads as a whole ("treinta y uno", 31) is never split.
The split only adds digits the caller said: every digit comes from reading the caller's own words.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from .normalize import (
    _JOINERS,
    _TOKEN_RE,
    NumberRun,
    _read_digits,
    _token_key,
    lang_family,
    normalize,
    number_runs,
    spoken_digits,
)
from .types import OptionView
from .validators import _is_number_token, validate

if TYPE_CHECKING:  # slot_policy imports the slots package; avoid the cycle at run time
    from owlbank.slot_policy import SlotSettings

ChoiceGround = Literal["caller", "known_other", "none"]

#: The languages of :func:`owlbank.slots.normalize.number_runs` for located runs (as in
#: :func:`owlbank.sensitive.placeholder_text`).
RUN_LANGS: tuple[str, str] = ("en", "es")


#: Every joiner of :func:`owlbank.slots.normalize.number_runs` in :data:`RUN_LANGS`.
_RUN_JOINERS: frozenset[str] = frozenset().union(*(_JOINERS[f] for f in RUN_LANGS))


# ---------------------------------------------------------------------------- digits


def read_at_joiners(n: int, is_joiner: Callable[[int], bool],
                    read: Callable[[int, int], str | None]) -> str | None:
    """Read a run of ``n`` tokens: ``read(0, n)`` if it reads as a whole; else the run is split at
    its joiner tokens and read piece by piece, left to right.

    From each start, the longest piece that ends before a joiner (or at the run's end) and reads
    is taken, so "treinta y uno y dos" is 31 then 2, not 30, 1, 2. A piece that doesn't read is
    skipped up to the next joiner (it adds no digits); joiners themselves add none. ``read(lo,
    hi)`` reads tokens ``[lo, hi)``. ``None`` if nothing reads."""
    whole = read(0, n) if n else None
    if whole:
        return whole
    joiners = [k for k in range(n) if is_joiner(k)]
    if not joiners:
        return None
    out: list[str] = []
    lo = 0
    while lo < n:
        if is_joiner(lo):
            lo += 1
            continue
        ends = [k for k in joiners if k > lo] + [n]
        for hi in reversed(ends):
            got = read(lo, hi)
            if got:
                out.append(got)
                lo = hi + 1
                break
        else:
            lo = ends[0] + 1
    return "".join(out) or None


def digit_stream(normalized: str, lang: str) -> str:
    """The concatenated digits of every maximal run of number tokens in ``normalized``
    (:func:`validators._is_number_token`), each run read with :func:`normalize.spoken_digits`.
    A run that doesn't read is split at the language's joiners and each piece read
    (:func:`read_at_joiners`); what still doesn't read is skipped (it adds no digits)."""
    out: list[str] = []
    run: list[str] = []
    for tok in [*normalized.split(), ""]:   # "" ends the last run
        if tok and _is_number_token(tok, lang):
            run.append(tok)
            continue
        if run:
            digits = _read_token_run(run, lang)
            if digits:
                out.append(digits)
            run = []
    return "".join(out)


def _read_token_run(toks: list[str], lang: str) -> str | None:
    """One run of :func:`digit_stream`, split at ``lang``'s joiners if it doesn't read whole."""
    joiners = _JOINERS[lang_family(lang)]
    return read_at_joiners(len(toks), lambda k: toks[k] in joiners,
                           lambda lo, hi: spoken_digits(" ".join(toks[lo:hi]), lang))


def run_digits(text: str, run: NumberRun) -> str | None:
    """A :func:`number_runs` run's digits (``text`` is the text the runs were found in):
    ``run.digits`` if it reads as a whole, else the run split at its joiners ("and", "y",
    "con") with each piece read in the first of :data:`RUN_LANGS` that reads it
    (:func:`read_at_joiners`). ``None`` if nothing reads."""
    if run.digits is not None:
        return run.digits
    seg = text[run.start:run.end]
    toks = list(_TOKEN_RE.finditer(seg))
    return read_at_joiners(
        len(toks), lambda k: _token_key(toks[k].group()) in _RUN_JOINERS,
        lambda lo, hi: _read_digits(seg[toks[lo].start():toks[hi - 1].end()], RUN_LANGS))


def is_ordered_subsequence(value: str, stream: str) -> bool:
    """``value``'s characters occur in ``stream`` in order (not necessarily adjacent)."""
    it = iter(stream)
    return all(c in it for c in value)


def ground_digits(slot: SlotSettings, value: object, speech: str | None, lang: str) -> str | None:
    """The canonical digit value, or ``None``.

    Accepted only if ``value`` is a string that :func:`validate` accepts for a ``digits`` slot
    with an exact ``constraints.length``, has that many digits, and is an ordered subsequence of
    ``digit_stream(normalize(speech))``. A slot with no exact length can't be grounded (fail
    closed). Any other JSON value is refused: the resolver's schema sends a string."""
    decl = slot.decl
    if decl.slot_type != "digits" or not isinstance(value, str) or not speech:
        return None
    length = decl.constraints.get("length")
    if isinstance(length, bool) or not isinstance(length, (int, str)):
        return None
    try:
        want = int(length)
    except ValueError:
        return None
    canon = validate(decl, value)
    if canon is None or len(canon) != want:
        return None
    stream = digit_stream(normalize(speech, lang), lang)
    return canon if is_ordered_subsequence(canon, stream) else None


def grounded_runs(value: str, speech: str | None) -> tuple[int, ...] | None:
    """Indices into ``number_runs(NFC(speech), ("en", "es"))`` of the runs that hold the rightmost
    match of ``value`` as an ordered subsequence of their digits (:func:`run_digits`: a run that
    doesn't read as a whole is split at its joiners, and its index is reported when one of its
    pieces supplied a digit), ascending. Each run in it supplied at least one digit. ``None`` if ``value`` isn't a digit string or the runs don't hold
    it (``placeholder_text`` then falls back).

    The rightmost match takes each digit of ``value``, from its last, from the latest run that
    still has it: the caller's last attempt is the one the value came from."""
    if not value or not (value.isascii() and value.isdigit()) or not speech:
        return None
    text = unicodedata.normalize("NFC", speech)
    runs = number_runs(text, RUN_LANGS)
    need = len(value) - 1
    used: list[int] = []
    for i in range(len(runs) - 1, -1, -1):
        digits = run_digits(text, runs[i])
        if digits is None:
            continue
        took = False
        for c in reversed(digits):
            if need >= 0 and c == value[need]:
                need -= 1
                took = True
        if took:
            used.append(i)
        if need < 0:
            return tuple(reversed(used))
    return None


# --------------------------------------------------------------------------- choices


def ground_choice(value: object, *, known: Sequence[str], caller: Sequence[str]) -> ChoiceGround:
    """``caller`` for one of the caller's options (that is also a known option), ``known_other``
    for a known option that isn't the caller's, else ``none``. Exact values only."""
    if not isinstance(value, str) or value not in known:
        return "none"
    return "caller" if value in caller else "known_other"


def ground_candidates(values: object, *, caller: Sequence[str]) -> tuple[str, ...] | None:
    """The distinct candidates in order, if ``values`` is a list of the caller's options with at
    least 2 distinct ones; else ``None``. One entry that isn't a caller option refuses the
    whole list: candidates are never filtered into shape."""
    if not isinstance(values, (list, tuple)):
        return None
    out: list[str] = []
    for v in values:
        if not isinstance(v, str) or v not in caller:
            return None
        if v not in out:
            out.append(v)
    return tuple(out) if len(out) >= 2 else None


def named_in(normalized: str, option: OptionView, lang: str) -> bool:
    """The option's spoken name or one of its aliases occurs in ``normalized`` (output of
    :func:`normalize.normalize`) as a contiguous run of whole tokens. A word that merely
    contains an alias ("reddish" for "red") is no match."""
    toks = normalized.split()
    for phrase in (option.name, *option.aliases):
        want = normalize(phrase, lang).split()
        n = len(want)
        if n and any(toks[i:i + n] == want for i in range(len(toks) - n + 1)):
            return True
    return False
