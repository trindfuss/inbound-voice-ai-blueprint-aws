"""owlbank.recap properties.

- Target: a verified state targets the record's ``memory_email`` (create) or skips with
  ``no_identifier``, never the phone; an unverified state targets the phone only for a real E.164 ANI
  that is trusted (``ani_trusted``) on a call where no number was keyed in (``identity_source``
  ``ani``), lookup only; an untrusted ANI or a keyed-in call never targets (``ani_untrusted``);
  ``Anonymous``, empty, withheld or non-E.164 never targets unless verified.
- Over any sequence of claims, outcomes and crashes at any times, a claim is
  granted only when no other writer is live (no claim within 90 s, no outcome); after an outcome
  nothing is granted.
- Scrub: no registered factor value (as digits or as number words), no run of 4+ digits or of
  number words reading as 4+ digits outside an ISO date or a money amount, and None whenever the PII
  check fails. Text is generated with English and Spanish number words, alone or mixed with digits,
  with fillers (uh, um, like, este, o sea, ...) and punctuation (``;``, ````, ``...``) between
  them, and with factor values spoken as words or split digit groups ("94, uh, 105"). Two oracles:
  ``number_runs`` (the placeholder rule's reading) and :mod:`tests.recap_oracle`, which reads digit
  words from its own table and skips fillers and punctuation without ``number_runs``.
- Bounds: whatever the model returns, one summary of at most 1,000 characters and at most 5
  observations of at most 300 characters, or nothing.
"""

from __future__ import annotations

import re
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import recap as rc
from owlbank.callstate import CallState
from owlbank.pii import FakePiiRedactor, PiiGuard, RedactionConfig
from owlbank.segments import MIN_SENSITIVE_LEN, contains_value, norm_text
from owlbank.slots.normalize import number_runs
from tests.recap_oracle import spoken_factor, spoken_long

SID = "CA" + "0" * 32
REAL_ANI = st.from_regex(r"\A\+[1-9]\d{7,14}\Z")
OTHER_ANI = st.one_of(
    st.none(),
    st.sampled_from(["", "Anonymous", "anonymous", "Restricted", "Unknown", "+", "+0123456789",
                     "sip:a@b.example", "client:agent", "+1202555010", "+1234567"]),
    st.text(max_size=20),
    st.from_regex(r"\A\+?\d{1,20}\s?\Z"),
).filter(lambda a: a is None or not re.fullmatch(r"\+[1-9]\d{7,14}", a))
ANI = st.one_of(REAL_ANI, OTHER_ANI)
CUSTOMER = st.one_of(
    st.none(),
    st.fixed_dictionaries(
        {},
        optional={"customer_id": st.sampled_from(["C-1001", "C-1002"]),
                  "memory_email": st.one_of(st.none(), st.integers(), st.sampled_from(["", " "]),
                                            st.emails()),
                  "alias_of": st.just("C-1003")}),
)


@settings(max_examples=300, deadline=None)
@given(ANI, st.sampled_from([None, "", "C-1001", "C-1002"]), st.integers(0, 3), CUSTOMER,
       st.sampled_from(["ani", "keyed_phone"]), st.booleans())
def test_recap_target(ani: str | None, vid: str | None, level: int, customer: dict[str, Any] | None,
                   source: str, trusted: bool) -> None:
    s = CallState(call_sid=SID, expires_at=1, ani=ani, verified_customer_id=vid, level=level,
                  identity_source=source, ani_trusted=trusted)  # type: ignore[arg-type]
    got = rc.recap_target(s, customer)
    if vid and level >= 1:
        email = (customer or {}).get("memory_email")
        usable = (customer is not None and "alias_of" not in customer
                  and customer.get("customer_id", vid) == vid and isinstance(email, str) and email.strip())
        assert got == (rc.Target("email", email, create=True) if usable else rc.Skip(rc.NO_IDENTIFIER))
        assert not (isinstance(got, rc.Target) and got.id_type == "phone")
    elif isinstance(ani, str) and re.fullmatch(r"\+[1-9]\d{7,14}", ani):
        usable = trusted and source == "ani"
        assert got == (rc.Target("phone", ani, create=False) if usable else rc.Skip(rc.ANI_UNTRUSTED))
    else:
        assert got == rc.Skip(rc.NOT_VERIFIED)
    if not (vid and level >= 1) and (not trusted or source == "keyed_phone"):
        assert not isinstance(got, rc.Target), "an untrusted or keyed-in caller never targets"


# --------------------------------------------------------------------------- (claim rule)

EVENT = st.tuples(st.sampled_from(["claim", "finish", "crash"]), st.integers(0, 1500))


@settings(max_examples=400, deadline=None)
@given(st.lists(EVENT, max_size=30), st.sampled_from([None, {}, {"claimed_at": "bad"}]))
def test_claim_admits_one_live_writer(events: list[tuple[str, int]], start: Any) -> None:
    recap: Any = start
    now, granted, holder, finished = 0, [], None, False
    for kind, step in events:
        now += step  # time never goes backwards
        if kind == "claim":
            new = rc.recap_claim(recap, now)
            if new is not None:
                assert not finished, "a claim after an outcome"
                assert all(now - t > rc.CLAIM_STALE_S for t in granted), "two live writers"
                granted.append(now)
                recap, holder = new, now  # the conditional write succeeds (single-threaded model)
        elif kind == "finish" and holder is not None:
            # The holder records its outcome only while the stored claim is still its own.
            if isinstance(recap, dict) and recap.get("claimed_at") == holder:
                recap = {**recap, "outcome": "written", "at": now}
                finished = True
            holder = None
        elif kind == "crash":
            holder = None  # the claim stays, with no outcome, until it goes stale


# --------------------------------------------------------------------------- Scrub

TOKENS = st.sampled_from([
    " ", "-", ".", ",", "$", "$ ", " dollars", " USD", " dólares", "\n", "the ", "zip ", "card ",
    "2026-11-08", "1999-01-31", "2026-13-01", "$1,234.56", "1,234,567", "4111 1111 1111 1111",
    "123-45-6789", "[number]", "redacted", "10:30", "9:15", "0:00", "23:59", "24:00", "25:99",
    "1:2345", ":", " at ", " am",
])
#: English and Spanish number words, joiners and look-alikes ("someone", "tone" hold no number word).
WORDS = st.sampled_from([
    "one", "two", "four", "five", "nine", "oh", "zero", "Zero", "twenty", "fifteen", "first",
    "hundred", "thousand", "double", "point", "and", "o", "cero", "uno", "cuatro", "nueve", "dos",
    "veintiuno", "dieciséis", "doscientos", "mil", "y", "someone", "tone", "won", "once",
])
_EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ES_DIGITS = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]


#: What a transcript may put between spoken digits: separators, fillers and other punctuation.
SEPS = [" ", "-", ", ", " , ", "  ", ", uh, ", " um ", " uh ", "; ", "... ", ": ", " - like - ",
        ", este, ", " o sea ", " pues ", " mm ", " hmm, ", "! ", " / ", " umm ", " uhm ", ", uhh, ",
        " hm ", " mhm ", " erm ", ", ok, ", " okay ", " Umm ", " mmm ", ", ehh, ", ":"]


@st.composite
def in_words(draw: st.DrawFn, digits: str) -> str:
    """``digits`` spoken: each digit a word of either language or the digit itself ("oh" for 0),
    sometimes in parentheses, with a separator, filler or punctuation before each."""
    out = []
    for i, d in enumerate(digits):
        form = draw(st.sampled_from(["en", "es", "digit", "oh"]))
        if form == "oh" and d == "0":
            word = draw(st.sampled_from(["oh", "Oh", "o"]))
        elif form == "es":
            word = _ES_DIGITS[int(d)]
        elif form == "digit":
            word = d
        else:
            word = draw(st.sampled_from([_EN_DIGITS[int(d)], _EN_DIGITS[int(d)].upper()]))
        if draw(st.integers(0, 5)) == 0:
            word = f"({word})"
        out.append((draw(st.sampled_from(SEPS)) if i else "") + word)
    return "".join(out)


@st.composite
def in_groups(draw: st.DrawFn, digits: str) -> str:
    """``digits`` as digit groups with a filler or punctuation between ("94, uh, 105")."""
    if len(digits) < 2:
        return digits
    cut = draw(st.integers(1, len(digits) - 1))
    sep = draw(st.sampled_from([s for s in SEPS if not re.fullmatch(r"[\s\-,]*", s)]))
    return digits[:cut] + sep + digits[cut:]


SPOKEN = st.one_of(st.text(alphabet="0123456789", min_size=1, max_size=8).flatmap(in_words),
                   st.text(alphabet="0123456789", min_size=2, max_size=8).flatmap(in_groups))
TEXT = st.lists(st.one_of(TOKENS, WORDS, SPOKEN, st.text(alphabet="0123456789", min_size=1, max_size=6),
                          st.text(max_size=6)), max_size=25).map("".join)
FACTOR = st.one_of(st.text(alphabet="0123456789", min_size=1, max_size=9),
                   st.text(alphabet="abcdefghij -", min_size=1, max_size=10), st.text(max_size=10))
_LENIENT_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
#: A valid clock time with no digit before or after it across separators (colons included).
_TIME = re.compile(r"(?<!\d)(?<!\d[\s.\-:])(?:[01]?\d|2[0-3]):[0-5]\d(?![\s.\-:]*\d)")
_LENIENT_KEEP = re.compile(r"\$\s?[\d,]+(?:\.\d{2})?|[\d,]+(?:\.\d{2})?\s?(?:dollars?|USD|d[oó]lares)\b", re.IGNORECASE)
_RUN = re.compile(r"\d(?:[\s.\-]*\d)*")


_O_SEA = re.compile(r"\bo(?=\s+sea\b)", re.IGNORECASE)


def _word_runs(text: str) -> list[Any]:
    """Runs of digits and number words, en and es, that hold a word. The "o" of the filler "o sea"
    is not the number word "o" (zero): it is blanked first, offsets kept."""
    blanked = _O_SEA.sub("_", text)
    return [r for r in number_runs(blanked, ("en", "es")) if re.search(r"[^\W\d_]", blanked[r.start:r.end])]


def _long_runs(text: str) -> list[str]:
    """Runs of 4+ digits outside anything shaped like a date or an amount (lenient on purpose:
    the unit tests pin what is kept; this checks nothing else survives), a valid clock time read as
    one run (``H:MM`` or ``HH:MM``), and runs of number words that read as 4+ digits, or can't be
    read and hold 4+ number tokens (no exception for words)."""
    rest = _LENIENT_KEEP.sub("§", _LENIENT_DATE.sub("§", text))  # dates first: "$2026-11-08" is a date
    # A time kept alone; any other colon joins digits like a dash ("10:30:45", "10:30 45").
    rest = _TIME.sub("§", rest).replace(":", "-")
    found = [m.group(0) for m in _RUN.finditer(rest) if sum(c.isdigit() for c in m.group(0)) >= 4]
    found += [text[r.start:r.end] for r in _word_runs(text)
              if (len(r.digits) >= 4 if r.digits is not None else r.tokens >= 4)]
    return found


def _kept_out(text: str) -> str:
    """``text`` with anything shaped like a date or an amount (lenient) replaced by a word, so the
    independent oracle doesn't read a kept date or amount as digits."""
    return _LENIENT_KEEP.sub(" KEPT ", _LENIENT_DATE.sub(" KEPT ", text))


def _spoken_factor(text: str, value: str) -> bool:
    """An all-digit factor value inside the digits a run of number words reads as."""
    n = norm_text(value)
    return n.isascii() and n.isdigit() and any(r.digits is not None and n in r.digits
                                               for r in _word_runs(text))


@st.composite
def text_with_spoken_factors(draw: st.DrawFn, factors: list[str]) -> str:
    """TEXT with the digit factor values spoken in words here and there."""
    parts = [draw(TEXT)]
    for value in factors:
        if value.isascii() and value.isdigit() and draw(st.booleans()):  # "²".isdigit() is True
            spoken = draw(st.one_of(in_words(value), in_groups(value)))
            parts += [draw(st.sampled_from([" ", " zip ", ", ", " y ", ", uh, ", "; "])), spoken, draw(TEXT)]
    return "".join(parts)


@settings(max_examples=400, deadline=None)
@given(st.data(), st.lists(FACTOR, max_size=4), st.booleans())
def test_recap_scrub(data: st.DataObject, factors: list[str], redaction_on: bool) -> None:
    text = data.draw(text_with_spoken_factors(factors))
    guard = PiiGuard(FakePiiRedactor(), RedactionConfig(enabled=redaction_on))
    out = rc.scrub(text, factor_values=factors, pii=guard, language="en-US")
    if out is None:
        return
    for value in factors:
        if len(norm_text(value)) >= MIN_SENSITIVE_LEN:
            assert not contains_value(out, value)
            assert not _spoken_factor(out, value), (out, value)
            assert not spoken_factor(out, norm_text(value)), (out, value)
    assert _long_runs(out) == []
    assert spoken_long(_kept_out(out)) == [], out


@settings(max_examples=200, deadline=None)
@given(TEXT.filter(str.strip), st.lists(FACTOR, max_size=3), st.sampled_from(["allow", "withhold"]))
def test_pii_failure_gives_none(text: str, factors: list[str], on_failure: str) -> None:
    guard = PiiGuard(FakePiiRedactor(fail=True), RedactionConfig(on_failure=on_failure))
    assert rc.scrub(text, factor_values=factors, pii=guard, language="en-US") is None


@settings(max_examples=300, deadline=None)
@given(TEXT)
def test_digit_masking_is_idempotent(text: str) -> None:
    once = rc.mask_digit_runs(text)
    assert rc.mask_digit_runs(once) == once
    assert _long_runs(once) == []
    assert spoken_long(_kept_out(once)) == [], once


# --------------------------------------------------------------------------- Bounds

JUNK = st.one_of(st.none(), st.integers(), st.floats(allow_nan=False), st.booleans(),
                 st.lists(st.text(max_size=3), max_size=2), st.dictionaries(st.text(max_size=3), st.text(max_size=3)))
OBS = st.one_of(JUNK, st.lists(st.one_of(st.text(max_size=400), JUNK), max_size=12),
                st.tuples(st.text(max_size=400), st.text(max_size=400)))
RAW = st.one_of(JUNK, st.text(max_size=5),
                st.fixed_dictionaries({}, optional={"summary": st.one_of(st.text(max_size=1500), JUNK),
                                                    "observations": OBS}))


@settings(max_examples=400, deadline=None)
@given(RAW)
def test_recap_bounds(raw: Any) -> None:
    got = rc.bounded(raw)
    if got is None:
        return
    assert isinstance(got.summary, str) and got.summary.strip()
    assert len(got.summary) <= rc.MAX_SUMMARY_CHARS
    assert len(got.observations) <= rc.MAX_OBSERVATIONS
    assert all(isinstance(o, str) and o.strip() and len(o) <= rc.MAX_OBSERVATION_CHARS
               for o in got.observations)
    summary = raw["summary"]
    assert isinstance(summary, str) and summary.strip().startswith(got.summary)
