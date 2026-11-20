"""The call recap written to Memory: the pure rules.

No I/O here; ``lambdas.recap_writer`` does the reads and writes around these functions.

- :func:`recap_target`: which Memory profile a call's recap goes to. A verified caller
  (``verified_customer_id`` set and ``level >= 1``) gets the profile keyed by the customer record's
  ``memory_email`` (looked up, created if missing); an unverified caller with a real E.164 caller ID
  that passed the STIR/SHAKEN check (``ani_trusted``) gets a phone lookup that never creates; an
  untrusted one, or a caller who keyed in a phone number (option A, whatever STIR/SHAKEN said),
  is skipped (``ani_untrusted``), and so is anything else. Never the phone for a verified
  caller, never ``Anonymous``, empty or withheld.
- :func:`scrub`: factor values removed, digit runs masked (number words count
  as digits in both, and fillers and punctuation between them don't split a run), then the PII check,
  fail closed (``None`` = don't write). Applied to the transcript before the model and to the
  model's output after.
- :func:`transcript_lines`: the Orchestrator conversation as ``"<who>: <text>"`` lines.
- :func:`recap_claim` and :func:`recap_mark`: the claim that admits one writer per
  call, and the mark that admits one Memory write.
- :func:`bounded`: the model's tool output cut to Memory-safe sizes.

**The digit rule (step 2).** A digit run is digits with any spaces, dots or dashes between them
(``4111 1111``, ``123-45-6789``, ``1.2.3.4``). A run of 4 or more digits becomes :data:`DIGITS_MASK`,
except:

- an ISO date ``YYYY-MM-DD`` (year 1900-2099, month 01-12, day 01-31),
- a clock time ``H:MM`` or ``HH:MM`` (hour 0-23, minute 00-59), and
- a money amount: ``$`` (one optional space) before, or ``dollar``/``dollars``/``USD``/``dólares``
  after, a number of at most 7 integer digits, either plain (``1234``) or grouped with commas
  (``1,234,567``), with optional cents (``.56``).

No exception applies when the date, time or amount touches another digit, directly or across one
separator (``$50 1234``, ``2026-11-08 1234``), or is joined to other digits or number words as a run
below (``$12.50; 3``, ``2026-11-08 one``, ``9:15, um, 9:20``): the whole run is masked. Factor
values are removed first, so a factor value can't stay as a time (``10:30`` for ``1030``). Anything
else with 4+ digits is masked, bare years (``2025``), bare grouped numbers (``1,234``), ``11/08``, invalid times (``25:99``)
and Spanish-style amounts (``1.234,56``) included; masking too much is the safe side.

**Number words, fillers and punctuation.** Real-time transcription may write
digits as words, so English and Spanish number words count as digits, read as the placeholder rule
reads them (:func:`owlbank.slots.normalize.number_runs` over both languages, whatever the call's
language). Here only (the placeholder rule is unchanged), a run also continues across fillers
(English um, umm, uhm, uh, uhh, er, erm, ah, eh, hm, hmm, mhm, like, so, ok, okay; Spanish eh, ehh,
este, pues, o sea, mm, mmm; and the hesitation sounds drawn out: ummm, uhhh, hmmm) and any punctuation
between its parts, within one line: "nine four, uh, one oh five", "four; one; one; one", "four
(eight) two one", "94, uh, 105". Fillers and punctuation are dropped before the run is read. A run
("nine four one oh five", "nine 4 one oh five", "cuatro-uno-uno-uno", "four, one, one, one") is:

- removed as a factor value (:data:`_FACTOR_MASK`) when the digits it reads as contain one (a
  run that only reads as digits through a joiner such as "and" is masked whole instead, below);
- masked whole (:data:`DIGITS_MASK`) when it reads as 4+ digits, or when it can't be read as digits
  and could carry 4 or more however it is read (each word's most digits added up: "one thousand
  two hundred", "dos mil veinticinco", and also "two hundred and fifty", whose words could carry
  five). A run number_runs alone joins is read only as number_runs reads it ("one and twenty nine"
  can't be read, could carry 4, and is masked); a joiner in a run that fillers or punctuation split
  ("four and, uh, eight two one") is dropped if the run can't be read with it.

Spoken dates and amounts get no exception: "twenty twenty six", "nineteen ninety nine" and "one
thousand two hundred dollars" are masked. "five hundred dollars", "the fifteenth", "two accounts"
and "the first one" are kept: they read as fewer than 4 digits, or can't be read and could carry at
most 3. A date or amount in digits that a number word touches joins the word's run and is masked with
it ("2026-11-08 one"). Words inside other words ("someone", "tone") are not number words.

The limit: a run never crosses a line break except where number_runs alone joins it (whitespace,
``,``, ``.``, ``-``, ``/``, "and"/"y"/"con"), and a transcript line starts with its speaker's label,
so a number split across two turns ("Customer: nine four" / "Customer: one oh five") is two runs.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

from owlbank.callstate import CallState
from owlbank.pii import ON_FAILURE_WITHHOLD, PiiGuard
from owlbank.segments import MIN_SENSITIVE_LEN, contains_value, norm_text, redact_values
from owlbank.slots.normalize import (
    NUMBER_WORDS,
    ORDINALS,
    REPEATERS,
    TEENS,
    TENS,
    UNITS,
    NumberRun,
    number_runs,
    strip_accents,
)

__all__ = [
    "ANI_UNTRUSTED",
    "CLAIM_STALE_S",
    "DIGITS_MASK",
    "E164",
    "MAX_OBSERVATIONS",
    "MAX_OBSERVATION_CHARS",
    "MAX_SUMMARY_CHARS",
    "NOT_VERIFIED",
    "NO_IDENTIFIER",
    "Recap",
    "SURFACE_MEMORY_RECAP",
    "Skip",
    "Target",
    "bounded",
    "has_factor_value",
    "mask_digit_runs",
    "recap_claim",
    "recap_mark",
    "recap_target",
    "scrub",
    "transcript_lines",
]

#: Skip reasons (as logged).
NOT_VERIFIED: Final = "not_verified"
NO_IDENTIFIER: Final = "no_identifier"
ANI_UNTRUSTED: Final = "ani_untrusted"

#: A real phone number as caller ID. ``Anonymous``, ``Restricted``, empty and SIP URIs fail it.
E164: Final = re.compile(r"\+[1-9]\d{7,14}")

#: A claim with no outcome is taken over after this long (a writer that crashed). More than
#: the writer's 60 s run, so a live writer is never taken over. Lambda's first async retry comes
#: about a minute after the failed run ends, so the retry of a run killed at its 60 s timeout finds
#: the claim stale and takes the call over (review). A run that dies early
#: (before about 30 s) is retried while its claim is still live: the retry stops at the claim and
#: that call's recap is lost, as it was with the earlier 600 s.
CLAIM_STALE_S: Final = 90

#:  (Memory allows 4,096 characters per observation and 10 per batch).
MAX_SUMMARY_CHARS: Final = 1000
MAX_OBSERVATIONS: Final = 5
MAX_OBSERVATION_CHARS: Final = 300

#: The PII check's surface label for the logs (the guard has no surface registry).
SURFACE_MEMORY_RECAP: Final = "memory_recap"
#: What a masked digit run becomes.
DIGITS_MASK: Final = "[number]"
#: What a removed factor value becomes (the journal's replacement).
_FACTOR_MASK: Final = "[redacted]"

_AUTHOR_LABELS: Final[Mapping[str, str]] = {
    "CUSTOMER": "Customer",
    "AI_AGENT": "AI agent",
    "AGENT": "AI agent",
    "HUMAN_AGENT": "Agent",
}
_OTHER_LABEL: Final = "Other"


# --------------------------------------------------------------------------- target


@dataclass(frozen=True, slots=True)
class Target:
    """The profile to write: looked up by ``id_type``/``value``, created only if ``create``."""

    id_type: Literal["email", "phone"]
    value: str = field(repr=False)  # an identifier of the caller: kept out of reprs and logs
    create: bool


@dataclass(frozen=True, slots=True)
class Skip:
    """No write, and why (:data:`NOT_VERIFIED`, :data:`NO_IDENTIFIER` or :data:`ANI_UNTRUSTED`)."""

    reason: str


def recap_target(state: CallState, customer: Mapping[str, Any] | None) -> Target | Skip:
    """Where the recap of ``state``'s call goes. ``customer`` is the ``customers``
    item of ``state.verified_customer_id`` (None if absent; ignored for an unverified call)."""
    if state.verified_customer_id and state.level >= 1:
        # The auth service's rule (prepare_welcome): an alias record has no profile of its own.
        if customer is None or "alias_of" in customer:
            return Skip(NO_IDENTIFIER)
        cid = customer.get("customer_id")
        if cid is not None and cid != state.verified_customer_id:  # a mismatched read: never guess
            return Skip(NO_IDENTIFIER)
        email = customer.get("memory_email")
        if not isinstance(email, str) or not email.strip():
            return Skip(NO_IDENTIFIER)
        return Target("email", email, create=True)
    ani = state.ani
    if isinstance(ani, str) and E164.fullmatch(ani):
        # A spoofed caller ID never adds text to someone's profile, and a caller who keyed in
        # a number (the front door clears ani_trusted then) is untrusted whatever STIR/SHAKEN said.
        if not state.ani_trusted or state.identity_source == "keyed_phone":
            return Skip(ANI_UNTRUSTED)
        return Target("phone", ani, create=False)  # An existing profile only
    return Skip(NOT_VERIFIED)


# --------------------------------------------------------------------------- scrub

_RUN: Final = r"\d(?:[\s.\-]*\d)*"
_ISO_DATE: Final = r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
_AMOUNT: Final = r"(?:\d{1,3}(?:,\d{3})+|\d{1,7})(?:\.\d{2})?"
_TIME: Final = r"(?:[01]?\d|2[0-3]):[0-5]\d"  # H:MM or HH:MM, 0:00 to 23:59
# Not preceded or followed by a digit, directly or across one separator (see the module docstring).
_NO_DIGIT_BEFORE: Final = r"(?<!\d)(?<!\d[\s.,\-])"
_NO_DIGIT_AFTER: Final = r"(?!\d|[\s.,\-]\d)"
_KEEP: Final = (
    rf"{_NO_DIGIT_BEFORE}{_ISO_DATE}{_NO_DIGIT_AFTER}"
    rf"|{_NO_DIGIT_BEFORE}{_TIME}{_NO_DIGIT_AFTER}"
    rf"|\$\s?{_AMOUNT}{_NO_DIGIT_AFTER}"
    rf"|{_NO_DIGIT_BEFORE}{_AMOUNT}\s?(?:dollars?|USD|d[oó]lares)\b"
)
# Leftmost match wins, and at one position a date, time or amount is tried before a run, so a run
# that starts earlier swallows a date, time or amount inside it (masked whole).
_DIGITS: Final = re.compile(rf"(?P<keep>{_KEEP})|(?P<run>{_RUN})", re.IGNORECASE)


def _mask_run(m: re.Match[str]) -> str:
    text = m.group(0)
    if m.group("keep") is not None or sum(ch.isdigit() for ch in text) < 4:
        return text
    return DIGITS_MASK


#: Number words of both languages, whatever the call's language (the placeholder rule's choice).
_WORD_LANGS: Final = ("en", "es")
_TOKEN: Final = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")  # number_runs' token
#: The most digits a scale word adds ("hundred" two zeros, "mil" three, ...); "point" counts one so
#: that no run carries more number tokens than this bound.
_SCALE_DIGITS: Final[Mapping[str, int]] = {
    "hundred": 2, "thousand": 3, "million": 6, "mil": 3, "millon": 6, "millones": 6,
    "point": 1, "punto": 1,
}


def _word_digits(key: str) -> int:
    """At most how many digits the number word ``key`` (accent-, case- and apostrophe-free) adds."""
    for fam in _WORD_LANGS:
        if key in UNITS[fam]:
            return 1
        if key in TEENS[fam] or key in TENS[fam] or key in ORDINALS[fam] or key in REPEATERS[fam]:
            return 3 if key == "triple" else 2
    if key in _SCALE_DIGITS:
        return _SCALE_DIGITS[key]
    return 3 if any(key in NUMBER_WORDS[fam] for fam in _WORD_LANGS) else 0  # "doscientos": 3


def _max_digits(run_text: str) -> int:
    """An upper bound on the digits a run of number words can stand for, however it is read
    (each word's most, added up; joiners add none)."""
    total = 0
    for tok in _TOKEN.findall(run_text):
        # As number_runs counts a token's numeric characters: "¼" decomposes to "1⁄4", two.
        numeric = max(sum(c.isnumeric() for c in tok), sum(c.isnumeric() for c in strip_accents(tok)))
        key = strip_accents(tok).lower().replace("'", "").replace("’", "")
        total += numeric if numeric else _word_digits(key)
    return total


#: Fillers that may sit between the parts of a spoken number (review), as token keys (accent-,
#: case- and apostrophe-free). "o sea" is a filler of two words, recognised before "o" (a number
#: word, zero) is.
_FILLERS: Final = frozenset({
    "um", "umm", "uhm", "uh", "uhh", "er", "erm", "ah", "eh", "hm", "hmm", "mhm", "like", "so",
    "ok", "okay",  # English
    "este", "pues", "mm", "mmm", "ehh",  # Spanish ("eh" above)
})
#: Hesitation sounds however the transcriber spells them ("ummm", "uhhh", "hmmm", "mhmm", "errm",
#: "ahh"): letters drawn out are still a filler. Never a number word (those are checked first).
_FILLER_SOUND: Final = re.compile(r"u+[hm]+|h+m+|m+h?m+|e+r+m*|e+h+|a+h+")
_FILLER_PAIRS: Final = frozenset({("o", "sea")})
#: number_runs' joiners and separators: what it joins across on its own.
_JOINERS: Final = frozenset({"and", "y", "con"})
_PLAIN_GAP: Final = re.compile(r"[\s,.\-/]*")
_LINE_BREAKS: Final = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
_NUMBER_KEYS: Final = frozenset().union(*(NUMBER_WORDS[fam] for fam in _WORD_LANGS))
_NUM, _FILL, _JOIN = "num", "fill", "join"


def _key(tok: str) -> str:
    return strip_accents(tok).lower().replace("'", "").replace("’", "")


def _filler(key: str) -> bool:
    return key in _FILLERS or _FILLER_SOUND.fullmatch(key) is not None


def _kinds(text: str, toks: list[re.Match[str]]) -> list[str | None]:
    """Each token's part in a run: a number (a numeric character or a number word of either
    language), a joiner, a filler, or nothing (it ends a run)."""
    keys = [_key(m.group()) for m in toks]
    kinds: list[str | None] = []
    for m, key in zip(toks, keys, strict=True):
        tok = m.group()
        if any(c.isnumeric() for c in tok) or any(c.isnumeric() for c in strip_accents(tok)) \
                or key in _NUMBER_KEYS:
            kinds.append(_NUM)
        else:
            kinds.append(_JOIN if key in _JOINERS else _FILL if _filler(key) else None)
    for i in range(len(toks) - 1):
        if (keys[i], keys[i + 1]) in _FILLER_PAIRS and text[toks[i].end():toks[i + 1].start()].isspace():
            kinds[i] = kinds[i + 1] = _FILL
    return kinds


def _gap_char(ch: str) -> bool:
    return ch not in _LINE_BREAKS and (ch.isspace() or unicodedata.category(ch).startswith("P"))


def _join(text: str, toks: list[re.Match[str]], kinds: list[str | None], a: int, b: int) -> tuple[bool, bool]:
    """Whether number tokens ``a`` and ``b`` (only fillers and joiners between them) are one run, and
    whether number_runs alone joins them ("plain")."""
    chunks = [text[toks[i].end():toks[i + 1].start()] for i in range(a, b)]
    middle = kinds[a + 1:b]
    if all(_PLAIN_GAP.fullmatch(c) for c in chunks) and (not middle or middle == [_JOIN]):
        return True, True
    return all(_gap_char(ch) for c in chunks for ch in c), False


def _read(run_text: str) -> str | None:
    """``run_text`` read as digits as number_runs reads one run, or None."""
    runs = number_runs(run_text, _WORD_LANGS)
    if len(runs) == 1 and runs[0].start == 0 and runs[0].end == len(run_text):
        return runs[0].digits
    return None


def _spoken_runs(text: str) -> list[tuple[NumberRun, bool]]:
    """The runs of digits and number words in ``text``: number_runs' runs (en and es), also joined
    across fillers (:data:`_FILLERS`, "o sea") and any punctuation between their parts, within one
    line (review: "nine four, uh, one oh five", "four; one; one; one", "four (eight) two one",
    "94, uh, 105"). Fillers and punctuation are dropped before reading the digits. A run never
    spans a line break other than as number_runs joins it, so a number split across two transcript
    lines (two turns) is two runs. Each run comes with whether number_runs alone would join it
    ("plain")."""
    toks = list(_TOKEN.finditer(text))
    kinds = _kinds(text, toks)
    runs: list[tuple[NumberRun, bool]] = []
    i = 0
    while i < len(toks):
        if kinds[i] != _NUM:
            i += 1
            continue
        first = last = i
        plain = True
        while True:
            nxt = last + 1
            while nxt < len(toks) and kinds[nxt] in (_FILL, _JOIN):
                nxt += 1
            if nxt >= len(toks) or kinds[nxt] != _NUM:
                break
            joined, simple = _join(text, toks, kinds, last, nxt)
            if not joined:
                break
            plain = plain and simple
            last = nxt
        start, end = toks[first].start(), toks[last].end()
        nums = [t for t in range(first, last + 1) if kinds[t] == _NUM]
        span = range(first, last + 1)
        digits = _read(text[start:end] if plain else
                       " ".join(toks[t].group() for t in span if kinds[t] in (_NUM, _JOIN)))
        # "four and, uh, eight two one": a joiner that number_runs can't read once fillers split
        # the run. Never for a plain run: number_runs already read it as it stands, and "one and
        # twenty nine" unreadable is masked when it could carry 4+ digits (fail closed).
        if digits is None and not plain and any(kinds[t] == _JOIN for t in span):
            digits = _read(" ".join(toks[t].group() for t in nums))
        runs.append((NumberRun(start, end, len(nums), digits), plain))
        i = last + 1
    return runs


def _keep_spans(text: str) -> list[tuple[int, int]]:
    """The ISO dates, clock times and money amounts the digit rule keeps in ``text``."""
    return [m.span() for m in _DIGITS.finditer(text) if m.group("keep") is not None]


def _maskable_runs(text: str) -> list[NumberRun]:
    """The runs :func:`mask_digit_runs` decides on: all but a date, time or amount the digit rule
    kept (one with nothing joined to it: "10:30" alone, not "9:15, um, 9:20")."""
    keeps = _keep_spans(text)
    return [r for r, _ in _spoken_runs(text)
            if _has_letter(text, r) or not any(ks <= r.start and r.end <= ke for ks, ke in keeps)]


def _has_letter(text: str, run: NumberRun) -> bool:
    return any(ch.isalpha() for ch in text[run.start:run.end])


def _factor_runs(text: str) -> list[NumberRun]:
    """The runs the factor-value check reads as digits: those with a word (a number word, filler or
    joiner) or punctuation number_runs doesn't join across. A plain digit run is the registry's
    (:func:`owlbank.segments.redact_values` and the comma pattern, with their digit boundaries)."""
    return [r for r, plain in _spoken_runs(text) if not plain or _has_letter(text, r)]


def _too_long(text: str, run: NumberRun) -> bool:
    """4+ digits, read; or unreadable and able to carry 4+ (fail closed)."""
    if run.digits is not None:
        return len(run.digits) >= 4
    return _max_digits(text[run.start:run.end]) >= 4


def _replace(text: str, spans: Iterable[tuple[int, int]], token: str) -> str:
    for start, end in sorted(spans, reverse=True):  # number runs never overlap
        text = text[:start] + token + text[end:]
    return text


def mask_digit_runs(text: str) -> str:
    """``text`` (NFC) with every run of 4+ digits masked, ISO dates, clock times and money amounts
    kept, then every run of number words that reads as, or could carry, 4+ digits masked (module
    docstring). Idempotent."""
    text = _DIGITS.sub(_mask_run, unicodedata.normalize("NFC", text))
    return _replace(text, [(r.start, r.end) for r in _maskable_runs(text) if _too_long(text, r)], DIGITS_MASK)


def _digit_norms(norms: Iterable[str]) -> list[str]:
    return [n for n in norms if n.isascii() and n.isdigit()]


def _redact_word_forms(norms: Iterable[str], text: str) -> str:
    """``text`` with each run (:func:`_spoken_runs`: number words, digits, fillers and punctuation
    between them) whose digits contain an all-digit factor value replaced by :data:`_FACTOR_MASK`
    ("nine four one oh five", "nine four, uh, one oh five", "94; 105" for ``94105``). An unreadable
    run is left to :func:`mask_digit_runs`."""
    digit_norms = _digit_norms(norms)
    if not digit_norms:
        return text
    text = unicodedata.normalize("NFC", text)
    hits = [(r.start, r.end) for r in _factor_runs(text)
            if r.digits is not None and any(n in r.digits for n in digit_norms)]
    return _replace(text, hits, _FACTOR_MASK)


def _word_found(text: str, norms: Iterable[str]) -> bool:
    digit_norms = _digit_norms(norms)
    return bool(digit_norms) and any(
        r.digits is not None and any(n in r.digits for n in digit_norms)
        for r in _factor_runs(unicodedata.normalize("NFC", text)))


def _factor_norms(values: Iterable[str]) -> list[str]:
    # The registry's rule (segments.register_sensitive): normalized, and too-short values skipped
    # because they would match almost anything. Already-normalized input normalizes to itself.
    # A value is also kept without its commas ("1,234" is the factor "1234").
    norms: set[str] = set()
    for v in values:
        if isinstance(v, str):
            n = norm_text(v)
            norms.update((n, n.replace(",", "")))
    return sorted(n for n in norms if len(n) >= MIN_SENSITIVE_LEN)


def _comma_pattern(norm: str) -> re.Pattern[str]:
    """The registry's pattern of ``norm`` (:func:`owlbank.segments.value_spans`) with commas also
    allowed between its characters, so ``1,234`` and ``12,34`` match the factor ``1234``. The same
    boundaries: an all-digit value needs a non-digit on each side, any other a word boundary on a
    side whose character is a letter or digit."""
    body = r"[\s\-,]*".join(re.escape(ch) for ch in norm)
    if norm.isdigit():
        return re.compile(r"(?<!\d)" + body + r"(?!\d)", re.IGNORECASE)
    before = r"(?<![^\W_])" if norm[0].isalnum() else ""
    after = r"(?![^\W_])" if norm[-1].isalnum() else ""
    return re.compile(before + body + after, re.IGNORECASE)


def _redact_comma_forms(norms: Iterable[str], text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for n in sorted(norms, key=len, reverse=True):
        text = _comma_pattern(n).sub(_FACTOR_MASK, text)
    return text


def _found(text: str, norms: Iterable[str]) -> bool:
    norms = list(norms)
    folded = unicodedata.normalize("NFKC", text).casefold()
    return any(contains_value(text, n) or _comma_pattern(n).search(folded) for n in norms) \
        or _word_found(folded, norms)


def has_factor_value(text: str, factor_values: Iterable[str]) -> bool:
    """True if any of ``factor_values`` (registry rule: normalized, at least
    :data:`~owlbank.segments.MIN_SENSITIVE_LEN` characters) occurs in ``text`` as the registry
    matches it, with commas between its characters, or (an all-digit value) inside the digits a run
    of digits and number words reads as (fillers and punctuation between them dropped). The writer's last check on what it is about to write (after
    :func:`bounded` cut it)."""
    return _found(text, _factor_norms(factor_values))


def scrub(text: str, *, factor_values: Iterable[str], pii: PiiGuard, language: str) -> str | None:
    """``text`` safe to send to the model or write to Memory, or None when it can't be made safe
    (the caller skips the write with ``redaction_failed``).

    1. ``factor_values`` (the call's registered values, raw or normalized) removed as the journal's
       scrub removes them (:func:`owlbank.segments.redact_values`), then again with commas allowed
       between their characters (``1,234`` for ``1234``; a value given with commas is also matched
       without them), then in runs of digits and number words, fillers and punctuation between
       them (a run whose digits contain an all-digit value);
    2. digit runs and number-word runs masked (:func:`mask_digit_runs`);
    3. the PII check (:meth:`PiiGuard.text`, surface :data:`SURFACE_MEMORY_RECAP`, ``on_failure``
       withhold whatever the policy says).

    A factor value still present at the end (a substitution that didn't hold, or a mask that spells
    one) gives None. With redaction turned off in policy, step 3 passes the text through, as on
    every other surface.
    """
    if not isinstance(text, str):
        return None
    norms = _factor_norms(factor_values)
    withheld = cast(str, object())  # an identity sentinel: no text can be it
    out = redact_values(norms, text, replacement=_FACTOR_MASK, withheld=withheld)
    if out is withheld:
        return None
    out = mask_digit_runs(_redact_word_forms(norms, _redact_comma_forms(norms, out)))
    try:
        checked = pii.text(SURFACE_MEMORY_RECAP, out, language=language, withheld=withheld,
                           on_failure=ON_FAILURE_WITHHOLD)
    except Exception:  # the guard catches its own failures; anything else is an outage too
        return None
    if checked is withheld or not isinstance(checked, str):
        return None
    if _found(checked, norms):
        return None
    return checked


# --------------------------------------------------------------------------- transcript


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _addresses(participant: Mapping[str, Any]) -> list[str]:
    rows = participant.get("addresses")
    if not isinstance(rows, list):
        return []
    return [a["address"] for a in rows if isinstance(a, Mapping) and isinstance(a.get("address"), str)]


def transcript_lines(communications: Iterable[Any], participants: Iterable[Any]) -> list[str]:
    """``"Customer: ..."``, ``"AI agent: ..."``, ``"Agent: ..."`` (``"Other: ..."`` for any other or
    unknown author) per communication with text, in the order given (the API's). The author's type
    comes from the participant named by ``author.participantId``, else ``author.type``, else the
    participant holding ``author.address``. Whitespace in a text is collapsed to single spaces, so a
    caller's words can't start a line of their own (a forged ``"Agent: ..."``)."""
    types_by_id: dict[str, str] = {}
    types_by_address: dict[str, str] = {}
    for p in participants or ():
        if not isinstance(p, Mapping) or not isinstance(p.get("type"), str):
            continue
        if isinstance(p.get("id"), str):
            types_by_id[p["id"]] = p["type"]
        for address in _addresses(p):
            types_by_address.setdefault(address, p["type"])
    lines: list[str] = []
    for c in communications or ():
        if not isinstance(c, Mapping):
            continue
        content = _mapping(c.get("content"))
        text = content.get("text") if isinstance(content.get("text"), str) else c.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        author = _mapping(c.get("author"))
        pid = author.get("participantId")
        kind = types_by_id.get(pid) if isinstance(pid, str) else None
        if kind is None and isinstance(author.get("type"), str):
            kind = author["type"]
        if kind is None and isinstance(author.get("address"), str):
            kind = types_by_address.get(author["address"])
        label = _AUTHOR_LABELS.get(kind.upper() if isinstance(kind, str) else "", _OTHER_LABEL)
        lines.append(f"{label}: {' '.join(text.split())}")
    return lines


# --------------------------------------------------------------------------- claim


def recap_claim(recap: Any, now: int) -> dict[str, Any] | None:
    """The call state ``recap`` value to write, conditionally, to claim the call's recap, or None
    when another writer has it: a new claim when there is none (or it is malformed), or when
    the claim is more than :data:`CLAIM_STALE_S` seconds old with no outcome. A recorded outcome is
    final, and so is a started write (``writing_at``, :func:`recap_mark`): a writer that crashed
    after it may have written the summary, so the call is never taken over (at most one summary
    write). The caller's conditional write must expect the ``recap`` value it read."""
    if not isinstance(recap, Mapping):
        return {"claimed_at": int(now)}
    if recap.get("outcome") is not None or recap.get("writing_at") is not None:
        return None
    claimed_at = recap.get("claimed_at")
    if not isinstance(claimed_at, int) or isinstance(claimed_at, bool):
        return {"claimed_at": int(now)}
    if now - claimed_at > CLAIM_STALE_S:
        return {"claimed_at": int(now)}
    return None


def recap_mark(recap: Any, claimed_at: int, now: int) -> dict[str, Any] | None:
    """The ``recap`` value that records the start of the Memory write (``writing_at``), or None when
    the writer that claimed at ``claimed_at`` no longer holds the call (taken over after
    :data:`CLAIM_STALE_S`, an outcome recorded, or a write already started). Written conditionally
    just before the summary, so of two writers that both passed a claim only one ever writes."""
    if not isinstance(recap, Mapping) or recap.get("claimed_at") != claimed_at:
        return None
    if recap.get("outcome") is not None or recap.get("writing_at") is not None:
        return None
    return {**dict(recap), "writing_at": int(now)}


# --------------------------------------------------------------------------- bounds


@dataclass(frozen=True, slots=True)
class Recap:
    """What is written: one summary and at most :data:`MAX_OBSERVATIONS` observations."""

    summary: str
    observations: tuple[str, ...] = ()


def _cut(text: str, limit: int) -> str:
    return text.strip()[:limit].rstrip()


def bounded(raw: Any) -> Recap | None:
    """The model's ``write_recap`` input within the limits, or None if it has no
    usable summary. Longer texts are cut; non-string and empty observations are dropped; only the
    first :data:`MAX_OBSERVATIONS` are kept."""
    if not isinstance(raw, Mapping):
        return None
    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    items = raw.get("observations")
    observations: list[str] = []
    if isinstance(items, list | tuple):
        for item in items:
            if len(observations) == MAX_OBSERVATIONS:
                break
            if isinstance(item, str) and item.strip():
                observations.append(_cut(item, MAX_OBSERVATION_CHARS))
    return Recap(_cut(summary, MAX_SUMMARY_CHARS), tuple(observations))
