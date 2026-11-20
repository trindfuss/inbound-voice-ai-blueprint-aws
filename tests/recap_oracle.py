"""An oracle for the recap scrub's spoken-number rule, independent of ``owlbank.slots.normalize``.

It reads text the way a listener would: within one line, each digit token and each single-digit word
(a small table here, English and Spanish, "oh"/"o" for 0) gives its digits, and fillers (um, umm, uhm,
uh, uhh, er, erm, ah, eh, hm, hmm, mhm, like, so, ok, okay; eh, ehh, este, pues, o sea, mm, mmm) and
any whitespace or punctuation between them are skipped. Anything else (another word, a symbol such as ``$``, a line break) ends the window. Scale and
tens words ("twenty", "hundred") also end it: the unit tests and ``number_runs`` cover those.

A window made only of digit tokens with ``[\\s.,\\-/]`` between them is left out (``spoken=False``):
that is the digit rule's and the registry's ground (ISO dates, amounts, digit boundaries), which the
property tests check with their own regexes. A window that is exactly a clock time (``H:MM`` or
``HH:MM``, 0:00 to 23:59) is kept too: only standing on its own, so a time joined
to other digits by a filler or punctuation is part of a longer window and is not kept.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_ES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"]
DIGIT_WORDS: dict[str, str] = {**{w: str(i) for i, w in enumerate(_EN)},
                               **{w: str(i) for i, w in enumerate(_ES)}, "oh": "0", "o": "0"}
FILLERS = frozenset({"um", "umm", "uhm", "uh", "uhh", "er", "erm", "ah", "eh", "hm", "hmm", "mhm", "like",
                     "so", "ok", "okay", "este", "pues", "mm", "mmm", "ehh"})
_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")  # a word with its apostrophes: "four's" is no digit
_PLAIN = re.compile(r"[\s.,\-/]*")
_TIME = re.compile(r"(?:[01]?[0-9]|2[0-3]):[0-5][0-9]")


@dataclass(frozen=True)
class Window:
    text: str
    digits: str
    spoken: bool  # a word or a filler in it, or punctuation beyond [\s.,\-/] between its parts


def _skippable(gap: str) -> bool:
    one_line = len(f"x{gap}x".splitlines()) == 1
    return one_line and all(ch.isspace() or unicodedata.category(ch).startswith("P") for ch in gap)


def windows(text: str) -> list[Window]:
    """The digit windows of ``text`` (module docstring)."""
    toks = list(_TOKEN.finditer(text))
    words = [m.group().casefold() for m in toks]

    def pair(i: int) -> bool:  # "o sea" with only spaces between
        return i + 1 < len(words) and words[i] == "o" and words[i + 1] == "sea" \
            and text[toks[i].end():toks[i + 1].start()].isspace()

    def o_sea(i: int) -> bool:
        return pair(i) or (i > 0 and words[i] == "sea" and pair(i - 1))

    def is_digit(i: int) -> bool:
        return not o_sea(i) and ((words[i].isascii() and words[i].isdigit()) or words[i] in DIGIT_WORDS)

    def is_filler(i: int) -> bool:
        return words[i] in FILLERS or o_sea(i)

    out: list[Window] = []
    cur: list[int] = []  # token indexes of the current window: digits, and fillers after a digit

    def close() -> None:
        while cur and not is_digit(cur[-1]):
            cur.pop()  # trailing fillers aren't part of it
        if cur:
            gaps = [text[toks[a].end():toks[b].start()] for a, b in zip(cur, cur[1:], strict=False)]
            spoken = any(not words[i].isdigit() for i in cur) or any(not _PLAIN.fullmatch(g) for g in gaps)
            digits = "".join(words[i] if words[i].isdigit() else DIGIT_WORDS[words[i]]
                             for i in cur if is_digit(i))
            out.append(Window(text[toks[cur[0]].start():toks[cur[-1]].end()], digits, spoken))
        cur.clear()

    for i in range(len(words)):
        digit, filler = is_digit(i), is_filler(i)
        if not digit and not filler:
            close()
            continue
        if cur and not _skippable(text[toks[cur[-1]].end():toks[i].start()]):
            close()
        if cur or digit:  # a window starts at a digit
            cur.append(i)
    close()
    return out


def spoken_long(text: str) -> list[str]:
    """Spoken windows that read as 4 or more digits, but for a window that is a clock time."""
    return [w.text for w in windows(text) if w.spoken and len(w.digits) >= 4 and not _TIME.fullmatch(w.text)]


def spoken_factor(text: str, value: str) -> bool:
    """An all-digit ``value`` inside the digits of a spoken window."""
    return value.isascii() and value.isdigit() and any(w.spoken and value in w.digits for w in windows(text))
