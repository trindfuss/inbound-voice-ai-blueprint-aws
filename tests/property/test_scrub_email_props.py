"""The log scrubber and email addresses: for any text with email addresses and phone numbers
planted in it, none survives; and scrubbing is idempotent."""

from __future__ import annotations

import re
import string

from hypothesis import example, given, settings
from hypothesis import strategies as st

from owlbank.obs import scrub

# Surrounding prose with no '@', '+' or digits, so a survivor can only come from a planted value.
FILLER = st.text(alphabet=string.ascii_letters + " .,;!'()-_", max_size=30)

_LOCAL = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._%+-]{0,40}", fullmatch=True)
_LABEL = st.from_regex(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,20}[A-Za-z0-9])?", fullmatch=True)
emails = st.builds(lambda local, labels, tld: f"{local}@{'.'.join(labels)}.{tld}",
                   _LOCAL, st.lists(_LABEL, min_size=1, max_size=4),
                   st.from_regex(r"[A-Za-z]{2,10}", fullmatch=True))
e164 = st.from_regex(r"\+[1-9][0-9]{7,14}", fullmatch=True)
nanp = st.builds(lambda a, b, c, fmt: fmt.format(a, b, c),
                 st.from_regex(r"[2-9][0-9]{2}", fullmatch=True),
                 st.from_regex(r"[0-9]{3}", fullmatch=True),
                 st.from_regex(r"[0-9]{4}", fullmatch=True),
                 st.sampled_from(["({}) {}-{}", "{}-{}-{}", "{}.{}.{}", "+1 {} {} {}", "1-{}-{}-{}"]))
planted = st.one_of(emails.map(lambda v: ("email", v)), e164.map(lambda v: ("phone", v)),
                    nanp.map(lambda v: ("phone", v)))
documents = st.lists(st.tuples(FILLER, planted), min_size=1, max_size=6)

CASES = settings(max_examples=300, deadline=None)


def _join(parts: list[tuple[str, tuple[str, str]]]) -> str:
    return " ".join(f"{filler} {value}" for filler, (_, value) in parts)


@CASES
@given(FILLER, emails, FILLER)
def test_no_email_survives(a: str, email: str, b: str) -> None:
    out = scrub(f"{a} {email} {b}")
    assert email not in out and "@" not in out
    assert "[email]" in out


@CASES
@given(documents)
def test_no_planted_email_or_phone_survives(parts: list[tuple[str, tuple[str, str]]]) -> None:
    out = scrub(_join(parts))
    for _, (_, value) in parts:
        assert value not in out
    assert "@" not in out
    # At most the four kept digits of a masked E.164 number remain together.
    assert not re.search(r"\d{5,}", out), out
    assert out.count("[email]") == sum(kind == "email" for _, (kind, _) in parts)


@CASES
@given(documents)
def test_scrub_is_idempotent_with_planted_values(parts: list[tuple[str, tuple[str, str]]]) -> None:
    once = scrub(_join(parts))
    assert scrub(once) == once


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(["@", ".", "com", "a", "Z", "0", "9", "+", "-", "_", "%", " ", "[email]",
                                 "mail.example", "x" * 25, "+1555", "\n", "'"]),
                max_size=60).map("".join))
@example(".@mail.example0" + "x" * 25)  # an address that would end inside a word
@example(".@" + "com.com_" + "x" * 25)  # the mask exposes a token-like run
def test_idempotent_on_email_fragments(text: str) -> None:
    once = scrub(text)
    assert scrub(once) == once


def test_email_mask_is_readable() -> None:
    assert scrub("lookup for jane.doe+bank@mail.example.co.uk failed") == "lookup for [email] failed"
    assert scrub("{'idType': 'email', 'value': 'a@b.io'}") == "{'idType': 'email', 'value': '[email]'}"
