"""Whole-value matching for the factor leak scans.

A factor value (ZIP, SSN last four, OTP) leaks when it appears in text as its own value: on
its own, inside JSON or after a key (``"ssn": "1111"``, ``ssn=1111``), or spelled out with
spaces or hyphens between its characters (``1 1 1 1``, ``11-11``). It does not leak when it is
only a slice of a longer run of digits the server generated, such as an 8-digit confirmation
number, a timestamp or an id: ``"04131111"`` does not contain the value ``1111``.

So a value matches with its separators optional, and only where the characters on either side
are not of the same kind as its own first and last character (a digit next to a digit, a letter
next to a letter). A letter next to a digit is still a boundary, so ``ssn1111`` is a leak.
A digit value also doesn't match as the integer or fractional part of a decimal number: a
measured ``"duration_ms": 2035.6`` doesn't contain the value ``2035`` (factor values are never
written with a decimal fraction), while ``SSN 2035.`` at the end of a sentence is still a leak.
An ISO 8601 date and time the server wrote (``2025-10-09T08:53:20Z``) is left out before the
scan: its hyphens would otherwise read ``10-09`` as the value ``1009``.

A random token the server minted during the run (the relay's one-time nonce, an opaque journal
reference) is left out too, by its exact value: ``secrets.token_urlsafe(24)`` is 32 characters
of ``[A-Za-z0-9_-]``, so about one in 100,000 holds a four-digit value between letters
(``A1111le``) and would read as a leak. ``record_minted`` is fed by a session fixture in
``tests/conftest.py``; a fixture value that is not a minted token is never left out, and one
glued to a minted token is still found.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SEP = r"[\s\-]*"
#: A generated date and time (casefolded): a date, ``t``, a time, an optional zone.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:z|[+-]\d{2}:?\d{2})?")


#: Random tokens minted so far in this process (``record_minted``), and the lengths they come in.
_MINTED: set[str] = set()
_MINTED_LENGTHS: list[int] = []
#: A run long enough to hold a minted token (``token_urlsafe(16)`` is 22 characters, ``token_hex(8)`` 16).
_RUN = re.compile(r"[A-Za-z0-9_-]{16,}")
_MIN_TOKEN = 16


def record_minted(token: str) -> None:
    """Note a random token the server minted, so ``leaks`` leaves exactly that value out."""
    if len(token) >= _MIN_TOKEN and token not in _MINTED:
        _MINTED.add(token)
        if len(token) not in _MINTED_LENGTHS:
            _MINTED_LENGTHS.append(len(token))


def scrub_minted(text: str) -> str:
    """``text`` with every recorded minted token replaced by ``<token>`` (any other text as it was)."""
    if not _MINTED:
        return text

    def sub(m: re.Match[str]) -> str:
        run = m.group(0)
        if run in _MINTED:
            return "<token>"
        out: list[str] = []
        i = 0
        while i < len(run):
            for n in _MINTED_LENGTHS:
                if run[i:i + n] in _MINTED:
                    out.append("<token>")
                    i += n
                    break
            else:
                out.append(run[i])
                i += 1
        return "".join(out)

    return _RUN.sub(sub, text)


def _edge(ch: str) -> str:
    """The character class that must not touch a value at an end that is ``ch``."""
    if ch.isdigit():
        return r"\d"
    if ch.isalpha():
        return r"[^\W\d_]"
    return r"(?!)"  # punctuation: no boundary to enforce


def value_pattern(value: str) -> re.Pattern[str]:
    """A pattern for ``value`` as a whole value, separators between its characters allowed."""
    chars = re.sub(r"[\s\-]+", "", value).casefold()
    if not chars:
        raise ValueError("empty value")
    body = _SEP.join(re.escape(c) for c in chars)
    before = rf"(?<!{_edge(chars[0])})" + (r"(?<!\d\.)" if chars[0].isdigit() else "")
    after = rf"(?!{_edge(chars[-1])})" + (r"(?!\.\d)" if chars[-1].isdigit() else "")
    return re.compile(rf"{before}{body}{after}")


def compile_values(values: Iterable[str]) -> list[re.Pattern[str]]:
    return [value_pattern(v) for v in values]


def leaks(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    """Whether any of the compiled values appears in ``text`` as a whole value (timestamps and minted tokens left out)."""
    folded = _TIMESTAMP.sub("<ts>", scrub_minted(text).casefold())
    return any(p.search(folded) for p in patterns)
