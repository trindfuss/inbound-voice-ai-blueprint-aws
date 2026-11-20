"""``owlbank.masking_config``: lenient loading never raises and stays in
range; strict accepts exactly what lenient uses without a problem; a rendered phrase
has no braces and uses only catalog words for its own tool's enum arguments)."""

from __future__ import annotations

import math
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from owlbank.masking_config import (
    CHARS_PER_S_RANGE,
    DELAY_MS_RANGE,
    END_MODES,
    MASK_PREFIX,
    MASKING_FIELDS,
    SILENCE_ACTIONS,
    SILENCE_MAX_REPEATS_RANGE,
    TOOL_MASK_FIELDS,
    MaskingSettings,
    ToolMask,
    parse_masking,
    parse_tool_mask,
    placeholder_names,
    placeholders_for,
    render,
    word_key,
)
from owlbank.policy import PolicySectionError
from owlbank.tools_contract import GATEWAY_LEVEL0_TOOLS, INLINE_TOOLS


scalars = st.one_of(
    st.none(), st.booleans(), st.integers(-10**6, 10**6), st.integers(), st.integers(0, 10001),
    st.integers(-1, 6), st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=12),
    st.sampled_from(["wait", "immediate", "escalate", "hangup", "15", "8.0", "nan", "inf",
                     "1e999", "-0", "mask.get_balance", "mask.", "mask.x{y}", "tool.x"]),
    st.decimals(allow_nan=True, allow_infinity=True),
)
json_like = st.recursive(scalars, lambda c: st.one_of(
    st.lists(c, max_size=3), st.dictionaries(st.text(max_size=8), c, max_size=3)), max_leaves=6)
_delay = st.one_of(st.integers(-1, 10001), scalars)
near_valid_masking = st.fixed_dictionaries({}, optional={
    "after_ms": _delay, "tool_after_ms": _delay, "handoff_wait_ms": _delay,
    "silence_timeout_ms": _delay, "silence_max_repeats": st.one_of(st.integers(-1, 6), scalars),
    "end_mode": st.one_of(st.sampled_from(sorted(END_MODES)), scalars),
    "silence_action": st.one_of(st.sampled_from(sorted(SILENCE_ACTIONS)), scalars),
    "chars_per_s": st.one_of(st.floats(7, 31), st.integers(7, 31), scalars)})
near_valid_mask = st.fixed_dictionaries(
    {"phrase_key": st.one_of(st.from_regex(r"mask\.[a-z_]{1,10}(\.[0-9])?", fullmatch=True), scalars)},
    optional={"after_ms": _delay, "handoff_wait_ms": _delay})
masking_docs = st.one_of(near_valid_masking, json_like, st.dictionaries(
    st.one_of(st.sampled_from(MASKING_FIELDS), st.text(max_size=10)), json_like, max_size=9))
mask_docs = st.one_of(near_valid_mask, json_like, st.dictionaries(
    st.one_of(st.sampled_from(TOOL_MASK_FIELDS), st.text(max_size=10)), json_like, max_size=4))


def _delay_ok(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and DELAY_MS_RANGE[0] <= v <= DELAY_MS_RANGE[1]


def _settings_in_range(s: MaskingSettings) -> None:
    for name in ("after_ms", "tool_after_ms", "handoff_wait_ms", "silence_timeout_ms"):
        assert _delay_ok(getattr(s, name)), name
    lo, hi = SILENCE_MAX_REPEATS_RANGE
    assert isinstance(s.silence_max_repeats, int) and not isinstance(s.silence_max_repeats, bool)
    assert lo <= s.silence_max_repeats <= hi
    assert s.end_mode in END_MODES and s.silence_action in SILENCE_ACTIONS
    assert isinstance(s.chars_per_s, float) and math.isfinite(s.chars_per_s)
    assert CHARS_PER_S_RANGE[0] <= s.chars_per_s <= CHARS_PER_S_RANGE[1]


@given(raw=masking_docs)
def test_masking_lenient_never_raises_and_strict_agrees(raw: Any) -> None:
    value, problems = parse_masking(raw, strict=False)
    _settings_in_range(value)
    assert parse_masking(value.to_doc(), strict=True) == value  # the stored shape re-parses
    try:
        strict = parse_masking(raw, strict=True)
    except PolicySectionError as exc:
        assert problems and exc.field == problems[0].field
    else:
        assert not problems and strict == value


@given(name=st.sampled_from([*INLINE_TOOLS, *GATEWAY_LEVEL0_TOOLS, "x"]), raw=mask_docs)
def test_tool_mask_lenient_never_raises_and_strict_agrees(name: str, raw: Any) -> None:
    value, problems = parse_tool_mask(name, raw, strict=False)
    if value is not None:
        assert isinstance(value, ToolMask)
        assert isinstance(value.phrase_key, str) and value.phrase_key.startswith(MASK_PREFIX)
        assert len(value.phrase_key) > len(MASK_PREFIX)
        for d in (value.after_ms, value.handoff_wait_ms):
            assert d is None or _delay_ok(d)
        assert parse_tool_mask(name, value.to_doc(), strict=True) == value
    assert all(p.field.startswith(f"tools.{name}.mask") for p in problems)
    try:
        strict = parse_tool_mask(name, raw, strict=True)
    except PolicySectionError as exc:
        assert problems and exc.field == problems[0].field
    else:
        assert not problems and strict == value



TOOLS = [*INLINE_TOOLS, *GATEWAY_LEVEL0_TOOLS, "no_such_tool"]
ARG_NAMES = sorted({a for t in INLINE_TOOLS for a in placeholders_for(t)}
                   | {"amount", "summary", "limit", "street", "question", "branch", "lang"})
ENUM_VALUES = sorted({v for t in INLINE_TOOLS for vs in placeholders_for(t).values() for v in vs})

names = st.one_of(st.sampled_from(ARG_NAMES), st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,6}", fullmatch=True))
literal = st.one_of(st.text(alphabet=st.characters(blacklist_characters="{}"), max_size=8),
                    st.text(max_size=4))  # sometimes a stray brace
segments = st.lists(st.one_of(literal, names.map(lambda n: "{" + n + "}")), max_size=6)
values = st.one_of(st.sampled_from(ENUM_VALUES), st.text(max_size=6), st.integers(), st.none())
words_text = st.one_of(st.text(alphabet=st.characters(blacklist_characters="{}"), min_size=1,
                               max_size=8), st.text(max_size=6))


TEMPLATED = [t for t in INLINE_TOOLS if placeholders_for(t)]


@st.composite
def renderable_cases(draw: st.DrawFn) -> tuple[str, str, Any, dict[str, str]]:
    """Mostly renderable: the tool's own enum args, valid values, words for most of them."""
    tool = draw(st.sampled_from(TEMPLATED))
    allowed = placeholders_for(tool)
    parts = draw(st.lists(st.one_of(literal, st.sampled_from(sorted(allowed)).map(
        lambda n: "{" + n + "}")), min_size=1, max_size=5))
    args: dict[str, Any] = {n: draw(st.one_of(st.sampled_from(vs), values)) if draw(
        st.integers(0, 5)) == 0 else draw(st.sampled_from(vs)) for n, vs in allowed.items()}
    words = {word_key(n, v): draw(words_text) for n, vs in allowed.items() for v in vs
             if draw(st.integers(0, 9)) > 0}
    return tool, "".join(parts), args, words


@st.composite
def random_cases(draw: st.DrawFn) -> tuple[str, str, Any, dict[str, str]]:
    tool = draw(st.sampled_from(TOOLS))
    text = "".join(draw(segments))
    used = placeholder_names(text)
    arg_names = draw(st.lists(st.one_of(st.sampled_from(used) if used else names, names), max_size=5))
    args: Any = {n: draw(values) for n in arg_names}
    if draw(st.integers(0, 9)) == 0:
        args = draw(json_like)
    word_keys = {word_key(n, v) for n in [*used, *ARG_NAMES] for v in ENUM_VALUES}
    keys = draw(st.lists(st.sampled_from(sorted(word_keys)), max_size=12))
    keys += [word_key(draw(names), draw(st.text(max_size=5))) for _ in range(draw(st.integers(0, 2)))]
    words = {k: draw(words_text) for k in keys}
    return tool, text, args, words


cases = st.one_of(renderable_cases(), random_cases())


@given(case=cases)
def test_render_has_no_braces_and_only_own_enum_words(case: tuple[str, str, Any, dict[str, str]]) -> None:
    tool, text, args, words = case
    allowed = placeholders_for(tool)
    used = placeholder_names(text)
    out = render(tool, text, args, words)
    if any(n not in allowed for n in used):
        assert out is None  # an unknown or non-enumerated placeholder never renders
    if out is None:
        return
    assert "{" not in out and "}" not in out
    # Every placeholder was an enum argument of this tool with an allowed value, filled with
    # exactly its catalog word: the output is the template with those words, nothing else.
    expected = text
    for n in used:
        assert n in allowed and args[n] in allowed[n]
        word = words[word_key(n, args[n])]
        assert "{" not in word and "}" not in word and word.strip()
        expected = expected.replace("{" + n + "}", word)
    assert out == expected


@given(tool=st.sampled_from(TOOLS), name=names, text=segments.map("".join), args=st.dictionaries(
    names, values, max_size=3))
def test_non_enum_placeholder_never_renders(tool: str, name: str, text: str,
                                                 args: dict[str, Any]) -> None:
    if name in placeholders_for(tool):
        return
    words = {word_key(name, str(v)): "w" for v in args.values()}
    assert render(tool, f"{text} {{{name}}}", {**args, name: "checking"}, words) is None
