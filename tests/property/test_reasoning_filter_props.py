"""Model failover: :class:`ReasoningFilter` over any chunk split.

The rule: no ``<`` or ``>`` reaches text-to-speech, but exactly ``<silent/>``
passed on to the silent filter. For any text and any split into chunks, what one message releases
(each feed, then the flush) is the same as for the whole text, and it is the text with

* every reasoning block removed: from an opening tag named ``thinking`` (any case, spacing,
  attributes, any length) to ``</thinking>`` (any case, spaces before ``>``), an unclosed one to the
  end; a self-closing ``<thinking/>`` is just a tag;
* the silent marker in any case or spacing passed on as exactly ``<silent/>``;
* every other tag (a name of letters, digits, ``_`` or ``-``, ``key="value"`` attributes only)
  removed whole, its text kept;
* every other ``<`` or ``>`` dropped and the words kept (a space after a dropped bracket that
  follows a space dropped too);
* a tag that outgrows ``MAX_TAG`` in its attributes dropped to its ``>``, the next ``<`` or the end of
  the message (counted in ``skips`` when it ran to the end); a ``<thinking`` held at the end of a
  message dropped as a block;
* whitespace after removed markup dropped while nothing else was released, one space where markup
  was removed between two non-space characters, and a space after markup removed after a space
  dropped (no double space).

The reference below builds the expected output from the pieces a text is made of, independently of
the filter's parsing. Noisy text (``<`` and ``>`` anywhere, tag pieces) is checked against the
invariants.
"""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from tac_bridge.agent_client import MAX_TAG, SILENT_MARK, ReasoningFilter, strip_reasoning

SECRET = "§"  # only ever inside a reasoning block
ATTR = "¤"  # only ever in a tag's attributes


def stream(text: str, cuts: list[int]) -> list[str]:
    """``text`` split at ``cuts`` (positions, any order, repeats allowed: empty chunks too)."""
    points = sorted({0, len(text), *(c % (len(text) + 1) for c in cuts)})
    return [text[a:b] for a, b in zip(points, points[1:], strict=False)] or [""]


def released(chunks: list[str], f: ReasoningFilter | None = None) -> str:
    """What the bridge sends for one message's chunks: each feed, then the flush."""
    f = f or ReasoningFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def speakable(out: str) -> bool:
    """The rule: no ``<`` or ``>`` but in exactly ``<silent/>``."""
    return "<" not in out.replace(SILENT_MARK, "") and ">" not in out.replace(SILENT_MARK, "")


# --------------------------------------------------------------------------- the reference


def cased(word: str) -> st.SearchStrategy[str]:
    return st.lists(st.booleans(), min_size=len(word), max_size=len(word)).map(
        lambda up: "".join(c.upper() if u else c for c, u in zip(word, up, strict=True)))


ws = st.text(alphabet=" \t\n", max_size=3)
ws1 = st.text(alphabet=" \t\n", min_size=1, max_size=3)
word = st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="<>" + SECRET + ATTR),
               max_size=15)
#: Anything after the name of an opening thinking tag, any length, not ending in "/" (self-closing);
#: a quote right after "=" (and spaces) opens a value, and a quoted ">" does not end it; any other
#: quote is an ordinary character.
_head_plain = st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="<>\"'"),
                      max_size=60)
_head_quoted = st.tuples(ws, st.sampled_from("\"'"), st.text(alphabet=st.sampled_from(list("ab />=")), max_size=8)).map(
    lambda t: f" k={t[0]}{t[1]}{t[2]}{t[1]}")
_head_stray = st.sampled_from([" don't", " it's", ' say "hi', " a'b", ' x"/'])
head = st.lists(st.one_of(_head_plain, _head_quoted, _head_stray), max_size=4).map("".join).filter(
    lambda t: not t.rstrip().endswith("/"))
opening = st.one_of(
    cased("thinking").map(lambda n: f"<{n}>"),
    st.tuples(cased("thinking"), ws1, head).map(lambda t: f"<{t[0]}{t[1]}{t[2]}>"),
    st.tuples(cased("thinking"), st.sampled_from([' type="internal-reasoning-about-the-caller-account-and-next-'
                                                  'step"', " a b", " x='<y'"])).map(lambda t: f"<{t[0]}{t[1]}>"),
)
closing = st.tuples(cased("thinking"), ws).map(lambda t: f"</{t[0]}{t[1]}>")
#: What a block holds: anything (tags, brackets) but its close, always with the secret in it.
content = st.text(alphabet=st.sampled_from(list("ab <>/=\"'\nTthinkg") + [SECRET]), max_size=40).map(
    lambda t: t + SECRET).filter(lambda t: not re.search(r"</thinking\s*>", t, re.IGNORECASE))
block = st.tuples(opening, content, closing).map(lambda t: ("block", "".join(t)))

name = st.from_regex(r"[A-Za-z0-9_-]{1,10}", fullmatch=True).filter(lambda n: n.lower() not in ("thinking", "silent"))
quoted_attr = st.tuples(st.from_regex(r"[A-Za-z_][A-Za-z0-9_.:-]{0,6}", fullmatch=True), ws, ws, st.sampled_from("\"'"),
                        st.text(alphabet=st.sampled_from([*"ab 1/=-", ATTR]), max_size=8)).map(
    lambda t: f"{t[0]}{t[1]}={t[2]}{t[3]}{t[4]}{t[3]}")
#: ``key=value`` with no quotes and no spaces in the value.
bare_attr = st.tuples(st.from_regex(r"[A-Za-z_][A-Za-z0-9_.:-]{0,6}", fullmatch=True), ws, ws,
                      st.text(alphabet=st.sampled_from([*"ab1/.:-", ATTR]), min_size=1, max_size=8)).map(
    lambda t: f"{t[0]}{t[1]}={t[2]}{t[3]}")
attr = st.one_of(quoted_attr, bare_attr)
attrs = st.lists(st.tuples(ws1, attr).map("".join), max_size=3).map("".join)
other_tag = st.one_of(
    st.tuples(name, attrs, ws).map(lambda t: f"<{t[0]}{t[1]}{t[2]}>"),  # opening
    st.tuples(name, attrs, ws).map(lambda t: f"<{t[0]}{t[1]}{t[2]}/>"),  # self-closing
    st.tuples(name, ws).map(lambda t: f"</{t[0]}{t[1]}>"),  # closing
    st.tuples(cased("thinking"), st.sampled_from(["", " ", ' type="x" '])).map(lambda t: f"<{t[0]}{t[1]}/>"),
    st.tuples(cased("thinking"), ws1, head).map(lambda t: f"<{t[0]}{t[1]}{t[2]}/>"),  # stray quotes too
    closing,  # a close with no block open: just a tag
).filter(lambda t: len(t) <= MAX_TAG).map(lambda t: ("tag", t))
#: The silent marker in any case or spacing: after the "<" (the silent marker only), between the
#: name and the "/", and between the "/" and the ">".
silent = st.tuples(ws, cased("silent"), ws, ws).map(lambda t: ("silent", f"<{t[0]}{t[1]}{t[2]}/{t[3]}>"))
#: Angle brackets that are no tag, each resolved by its own end: the words are spoken.
not_a_tag = st.sampled_from(["<Vacation fund>", "a <b and c > d", "<b id=>", "<b id=1=2>", "x < 5",
                             "x > y", "<message Id>", "<thin king>", "<>", "5 <= 6", "<5 dollars>", "> ",
                             "<< .", "<a b='1'c>"]).map(lambda t: ("loose", t))
#: An opening tag that outgrows MAX_TAG in its attributes, with no end yet: a long quoted value, or
#: many unquoted ones.
_filler = st.text(alphabet=st.sampled_from([*"ab 1-", ATTR]), min_size=MAX_TAG, max_size=MAX_TAG + 40)
long_open = st.one_of(
    st.tuples(name, ws1, _filler).map(lambda t: f'<{t[0]}{t[1]}k="{ATTR}{t[2]}'),
    st.tuples(name, st.integers(min_value=MAX_TAG // 5 + 1, max_value=MAX_TAG)).map(
        lambda t: f"<{t[0]}" + f" k={ATTR}1" * t[1]),
)
#: Dropped to its ">" (anything but brackets on the way, quotes included).
long_tag = st.tuples(long_open, st.text(alphabet=st.sampled_from([*"ab \"'=/", ATTR]), max_size=10)).map(
    lambda t: ("skip", t[0] + t[1] + ">"))
piece = st.one_of(word.map(lambda w: ("text", w)), block, other_tag, silent, not_a_tag, long_tag)
#: Dropped to the next "<", which starts what follows.
cut_short = st.tuples(long_open, st.one_of(block, other_tag, silent)).map(lambda t: [("skip", t[0]), t[1]])
pieces = st.lists(st.one_of(piece.map(lambda p: [p]), cut_short), max_size=8).map(lambda ps: [p for g in ps for p in g])
#: How a message may end: an unclosed block, an over-long tag still unended, a held "<thinking".
ending = st.one_of(
    st.none(),
    st.tuples(opening, content).map(lambda t: ("block", "".join(t))),
    long_open.map(lambda t: ("skip_end", t)),
    cased("thinking").map(lambda n: ("held", "<" + n)),
)
message = st.tuples(pieces, ending)


class Reference:
    """The expected output, built piece by piece."""

    def __init__(self) -> None:
        self.out: list[str] = []
        self.spoke = self.cut = self.gap = self.squeeze = False

    def removed(self) -> None:
        self.cut = True
        if self.out and self.out[-1] == " ":
            self.squeeze = True  # between two spaces: one is left
        self.gap = self.gap or bool(self.out) and not self.out[-1].isspace()

    def char(self, c: str) -> None:
        if c in "<>":
            self.squeeze = self.squeeze or bool(self.out) and self.out[-1] == " "
            return
        squeeze, self.squeeze = self.squeeze, False
        if squeeze and c == " ":
            return
        self.emit(c)

    def emit(self, c: str) -> None:
        if not self.spoke and c.isspace() and self.cut:
            return
        self.spoke = self.spoke or not c.isspace()
        if self.gap and not c.isspace():
            self.out.append(" ")
        self.gap = False
        self.out.append(c)


def build(pieces: list[tuple[str, str]], end: tuple[str, str] | None) -> tuple[str, str, int, int, int]:
    """The text, the expected output, the blocks, the other tags and the skips to the end in it."""
    text = "".join(p for _, p in pieces) + ("" if end is None else end[1])
    ref = Reference()
    blocks = tags = 0
    for kind, p in pieces:
        if kind in ("text", "loose"):
            for c in p:
                ref.char(c)
        elif kind == "silent":
            ref.squeeze = False
            for c in SILENT_MARK:
                ref.emit(c)
        else:
            ref.removed()
            blocks += kind == "block"
            tags += kind in ("tag", "skip")
    skips = 0
    if end is not None:
        blocks += end[0] in ("block", "held")
        tags += end[0] == "skip_end"
        skips += end[0] == "skip_end"
    return text, "".join(ref.out), blocks, tags, skips


cuts = st.lists(st.integers(min_value=0, max_value=600), max_size=12)


@given(message, cuts)
def test_any_split_releases_the_reference(msg: tuple[list[tuple[str, str]], tuple[str, str] | None],
                                              at: list[int]) -> None:
    text, want, blocks, tags, skips = build(*msg)
    f = ReasoningFilter()
    out = released(stream(text, at), f)
    assert out == want
    assert out == strip_reasoning(text)
    assert (f.blocks, f.tags, f.skips) == (blocks, tags, skips)
    assert SECRET not in out  # nothing of a reasoning block is ever spoken
    assert ATTR not in out  # nor any attribute text
    assert speakable(out)


@given(message)
def test_single_character_chunks_release_the_reference(msg: tuple[list[tuple[str, str]],
                                                                      tuple[str, str] | None]) -> None:
    text, want, *_ = build(*msg)
    assert released(list(text)) == want


# --------------------------------------------------------------------------- noisy text

noise_piece = st.one_of(
    st.text(alphabet=st.sampled_from(list("<>/ ab\n=\"'")), max_size=10),
    st.sampled_from(["<thinking>", "</thinking>", "<THINKING>", "<thin", "king>", "</think", "<message>",
                     "</message>", "<br/>", SILENT_MARK, "<sil", "ent/>", "<", ">", "less than <5", "a <b c",
                     "<thinking ", 'type="x">', "</thinking >", "<Vacation fund>", ' id="1"', "<message",
                     "<Silent />", "<thinking/>", "/>", '<a b="' + "y" * MAX_TAG, " c=1" * (MAX_TAG // 4 + 1),
                     "y" * MAX_TAG]),
)
noisy = st.lists(noise_piece, max_size=12).map("".join)


@given(noisy, cuts)
def test_boundaries_never_matter_and_no_bracket_is_released(text: str, at: list[int]) -> None:
    f = ReasoningFilter()
    out = "".join(f.feed(c) for c in stream(text, at))
    assert len(f.held) <= MAX_TAG
    out += f.flush()
    assert out == released([text]) == strip_reasoning(text)
    assert speakable(out)


bracket_free = st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="<>"), max_size=60)


@given(bracket_free, cuts)
def test_text_without_brackets_passes_unchanged_at_once(text: str, at: list[int]) -> None:
    f = ReasoningFilter()
    for c in stream(text, at):
        assert f.feed(c) == c
    assert f.flush() == ""
    assert (f.blocks, f.tags, f.skips) == (0, 0, 0)


# --------------------------------------------------------------------------- examples


def test_the_live_example_speaks_only_the_answer() -> None:
    live = "<thinking>\nEl cliente pregunta...\n</thinking>\n\nSetenta y cinco dólares."
    f = ReasoningFilter(keep=True)
    assert released([live[:5], live[5:40], live[40:]], f) == "Setenta y cinco dólares."
    assert (f.blocks, f.chars, f.tags) == (1, len("<thinking>\nEl cliente pregunta...\n</thinking>"), 0)
    assert f.removed == "<thinking>\nEl cliente pregunta...\n</thinking>"


def test_thinking_tags_of_any_length_case_spacing_or_attributes_open_a_block() -> None:
    long_tag = '<thinking type="internal-reasoning-about-the-caller-account-and-next-step">'
    assert len(long_tag) > 64
    assert strip_reasoning(long_tag + "secret plan</thinking>Hi") == "Hi"
    f = ReasoningFilter(keep=True)
    assert released([long_tag[:30], long_tag[30:] + "secret", " plan</thinking", " >", "Hi"], f) == "Hi"
    assert (f.blocks, f.tags, f.removed) == (1, 0, long_tag + "secret plan</thinking >")
    assert strip_reasoning("<thinking >x</thinking>Hi") == "Hi"
    assert strip_reasoning('<thinking type="x">x</thinking >Hi') == "Hi"
    assert strip_reasoning("<Thinking\n>x</THINKING\n>Hi") == "Hi"
    assert strip_reasoning("<thinking x='<y'>secret</thinking>Hi") == "Hi"  # a "<" in the tag
    assert strip_reasoning("<thinking a b c>secret</thinking>Hi") == "Hi"  # any attributes


def test_a_self_closing_thinking_tag_opens_nothing() -> None:
    for tag in ("<thinking/>", "<Thinking/>", "<THINKING/>", '<THINKING type="x" />', "<thinking />"):
        f = ReasoningFilter()
        assert released([tag + "Hi"], f) == "Hi"
        assert (f.blocks, f.tags) == (0, 1)


def test_the_silent_marker_in_any_case_or_spacing_is_passed_on_exactly() -> None:
    for mark in ("<silent/>", "<silent />", "<Silent/>", "<SILENT  />", "<silent/ >", "<silent / >",
                 "< silent/>", "<  Silent \t/\n>", "<silent x=1/>", "<silent x=1 />", '<silent a="b"/>'):
        f = ReasoningFilter()
        assert released(["Okay.", mark[:3], mark[3:]], f) == "Okay." + SILENT_MARK
        assert (f.blocks, f.tags) == (0, 0)


def test_other_tags_go_and_their_text_stays() -> None:
    assert strip_reasoning("<final_answer>Hi</final_answer>") == "Hi"
    assert strip_reasoning("<h1>Hi</h1>") == "Hi"
    assert strip_reasoning("<Answer>Hi</Answer>") == "Hi"
    assert strip_reasoning("<message>Your balance is $5.</message>") == "Your balance is $5."
    assert strip_reasoning("<response>\n<answer>Yes.</answer>\n</response>") == "Yes.\n"
    assert strip_reasoning('<message id="1">Hi</message>') == "Hi"
    assert strip_reasoning("</thinking>Hi") == "Hi"  # a close with no block: a tag
    f = ReasoningFilter()
    assert released(["<final_", "answer>Hi</final", "_answer>"], f) == "Hi"
    assert f.tags == 2


def test_other_angle_brackets_are_dropped_and_the_words_kept() -> None:
    assert strip_reasoning("Your <Vacation fund> account") == "Your Vacation fund account"
    assert strip_reasoning("Your nickname is <Vacation fund>.") == "Your nickname is Vacation fund."
    assert strip_reasoning("a <b and c > d") == "a b and c d"
    assert strip_reasoning("x < 5") == "x 5"
    assert strip_reasoning("x > y") == "x y"
    assert strip_reasoning("less than <5, a<b") == "less than 5, ab"
    assert strip_reasoning("<message id=1>Hi</message>") == "Hi"  # an unquoted value is still a tag
    assert strip_reasoning("<message id=1 lang=en>Hi</message>") == "Hi"
    assert strip_reasoning("<b id=>x") == "b id=x"  # no value: not a tag
    assert strip_reasoning("Your nickname is <Vacation>.") == "Your nickname is ."  # one word: a tag (accepted)
    f = ReasoningFilter()
    assert f.feed("a <b and") == "a "
    assert f.held == "<b and"  # could still be <b and="...">
    assert f.feed(" c > d") == "b and c d"
    assert f.flush() == ""
    assert f.tags == 0


def test_markup_between_words_leaves_one_space() -> None:
    assert strip_reasoning("Hello<thinking>x</thinking>world") == "Hello world"
    assert strip_reasoning("Hello <thinking>x</thinking> world") == "Hello world"  # no double space
    assert strip_reasoning("x <thinking>y</thinking> z") == "x z"
    assert strip_reasoning("Press <b>1</b> now") == "Press 1 now"
    assert strip_reasoning("a <b> <i> c") == "a c"
    assert strip_reasoning("a <b>\tc") == "a \tc"  # only a second space goes
    assert strip_reasoning("a\t<b> c") == "a\t c"
    assert strip_reasoning("Yes.<br/>No.") == "Yes. No."
    assert strip_reasoning("Done.<thinking>x</thinking>") == "Done."  # nothing after: no space
    f = ReasoningFilter()
    assert f.feed("Hello<thin") + f.feed("king>x</thinking>") + f.feed("world") + f.flush() == "Hello world"


def test_an_unclosed_block_is_dropped_to_the_end_and_counted() -> None:
    f = ReasoningFilter(keep=True)
    assert f.feed("Hi. <thinking>the caller") == "Hi. "
    assert f.feed(" wants</thin") == ""
    assert f.flush() == ""
    assert (f.blocks, f.chars) == (1, len("<thinking>the caller wants</thin"))
    assert f.feed("Next message.") == "Next message."  # a flush starts a new message
    assert strip_reasoning('Hi. <thinking type="never closed') == "Hi. "  # its opening tag unfinished too


def test_held_text_is_bounded() -> None:
    f = ReasoningFilter()
    out = "".join(f.feed("<") for _ in range(1000))
    assert f.held == "<"
    assert out + f.flush() == ""
    f = ReasoningFilter()
    out = f.feed('x <a b="' + "y" * 1000)
    assert len(f.held) <= MAX_TAG
    assert out + f.flush() == "x "  # attribute text is never spoken
    assert f.tags == 1


def test_an_over_long_tag_in_its_attributes_is_dropped_to_its_end() -> None:
    long_value = "y" * (2 * MAX_TAG)
    f = ReasoningFilter()
    assert released(['Hi <a b="', long_value, '">there'], f) == "Hi there"
    assert (f.tags, f.held) == (1, "")
    assert strip_reasoning(f"Hi <a b={long_value}>there") == "Hi there"
    assert strip_reasoning("<a " + " " * MAX_TAG + "word") == ""  # past the name: to the end of the message
    # a "<" ends the skip: a reasoning block after it is still removed
    assert strip_reasoning(f'<a b="{long_value}<thinking>secret</thinking>Hi') == "Hi"
    # before the name ends, an over-long run is text as before
    assert strip_reasoning("<" + "n" * (2 * MAX_TAG)) == "n" * (2 * MAX_TAG)


def test_spaces_in_self_closing_tags() -> None:
    assert strip_reasoning("a<br / >b") == "a b"
    assert strip_reasoning("a<br/ >b") == "a b"
    assert strip_reasoning("x < silent >") == "x silent "  # spaces after "<" only for the marker
    assert strip_reasoning("x < silently") == "x silently"
    assert strip_reasoning("x < 5") == "x 5"
    assert strip_reasoning("< br/>x") == " br/x"
    assert strip_reasoning("< silent a=1/>x") == " silent a=1/x"  # the marker with a leading space: no attributes


def test_a_slash_in_an_unquoted_value_before_the_end_closes_the_tag() -> None:
    assert strip_reasoning("<silent x=1/>") == SILENT_MARK
    assert strip_reasoning("<silent x=1/ >") == SILENT_MARK
    f = ReasoningFilter()
    assert released(["a<br a=b/>c"], f) == "a c"
    assert f.tags == 1
    assert strip_reasoning("<a href=/x/y>link</a>") == "link"  # "/" inside a value stays a value
    assert strip_reasoning("<a href=/x/ y=2>link</a>") == "link"


def test_a_quoted_bracket_does_not_end_a_thinking_opening_tag() -> None:
    assert strip_reasoning('<thinking note="a>b"/>Hello') == "Hello"  # self-closing: opens nothing
    assert strip_reasoning("<thinking note='a>b'>secret</thinking>Hello") == "Hello"
    assert strip_reasoning('<thinking a="/">secret</thinking>Hello') == "Hello"  # a quoted "/" is no self-close
    f = ReasoningFilter()
    assert released(['<thinking note="a>', 'b"/>Hello'], f) == "Hello"
    assert (f.blocks, f.tags) == (0, 1)


def test_a_held_thinking_name_at_the_end_of_a_message_is_dropped_as_a_block() -> None:
    f = ReasoningFilter(keep=True)
    assert f.feed("Hello <thinking") == "Hello "
    assert f.flush() == ""
    assert (f.blocks, f.chars, f.removed) == (1, len("<thinking"), "<thinking")
    assert strip_reasoning("Hello <THINKING") == "Hello "
    assert strip_reasoning("Hello <thinkin") == "Hello thinkin"  # not the whole name: text


def test_reasoning_text_is_counted_but_not_kept_without_keep() -> None:
    f = ReasoningFilter()
    assert f.feed("<thinking>abc" + "x" * 1000) == ""
    assert f._block == []  # noqa: SLF001 - nothing collected while in the block
    assert f.feed("</thinking>Hi") + f.flush() == "Hi"
    assert (f.blocks, f.chars, f.removed) == (1, len("<thinking>abc</thinking>") + 1000, "")


def test_behaviours_kept_as_recorded() -> None:
    assert strip_reasoning("Hmm <thinking about it, yes") == "Hmm "  # opens a block (fail closed)
    assert strip_reasoning("<thin<b>king>") == "thin king"  # not a thinking block
    assert strip_reasoning("a <5> b") == "a b"  # a tag


def test_a_stray_quote_in_a_thinking_opening_tag_is_ordinary() -> None:
    # Only a quote right after "=" opens a value.
    assert strip_reasoning("<thinking don't/>Hello world") == "Hello world"  # self-closing: opens nothing
    assert strip_reasoning("<thinking don't>secret</thinking>Hello") == "Hello"
    assert strip_reasoning('<thinking say "hi>secret</thinking>Hello') == "Hello"
    assert strip_reasoning('<thinking a = "x>y" >secret</thinking>Hello') == "Hello"  # spaces before the quote
    f = ReasoningFilter()
    assert released(["<thinking don", "'t/", ">Hello world"], f) == "Hello world"
    assert (f.blocks, f.tags) == (0, 1)


def test_an_unquoted_value_of_only_a_slash_self_closes() -> None:
    assert strip_reasoning("<silent x=/>") == SILENT_MARK
    assert strip_reasoning("<silent x= / >") == SILENT_MARK
    assert strip_reasoning("a<br x=/>b") == "a b"
    assert strip_reasoning("<a href=/>link</a>") == "link"


def test_a_skip_to_the_end_of_a_message_is_counted() -> None:
    f = ReasoningFilter()
    long_attr = ' k="' + "y" * MAX_TAG
    assert released(["Hi <a", long_attr, " the answer"], f) == "Hi "
    assert (f.tags, f.skips) == (1, 1)
    assert released(["<a" + long_attr + '">there'], f) == "there"  # ended by its ">": no skip to the end
    assert released(["<a" + long_attr + "<b>there"], f) == "there"  # ended by a "<"
    assert (f.tags, f.skips) == (4, 1)
