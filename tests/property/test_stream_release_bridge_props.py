"""Stream-agent-text properties of the bridge.

``<silent/>`` is never released by the bridge's:class:`SilentFilter`, whatever the
chunk boundaries. In stream mode the agent client yields each model chunk as it comes, so the
filter sees arbitrary splits, a marker split across any number of chunks included. The filter
holds at most:data:`MAX_HELD` characters of marker prefixes, so the generated
text keeps every run of marker characters within that bound; a longer run is released, not held
(a unit test below). Past the bound a weaker guarantee holds (a property below).

Every hand-off hold (:func:`_handoff_hold_s`, after a hand-off phrase
or released agent text) is never negative, never more than the tool's ``handoff_wait_ms`` (its own,
else the policy's) from the hand-off, 0 with ``end_mode=immediate``, equal to the latest of its
estimates (:func:`_hold_estimates`: the session's speech, the hop's released text, the phrase) when
that is under the cap, and monotone in each estimate. A phrase alone holds as before.
"""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from owlbank.masking_config import ToolMask
from tac_bridge.agent_client import MAX_HELD, SILENT_MARK, SilentFilter, strip_silent
from tac_bridge.bridge import HOLD_BASES, _handoff_hold_s, _hold_estimates, _Mask
from tac_bridge.masking import MaskPlanner, speech_end

MARK = SILENT_MARK



def stream(text: str, cuts: list[int]) -> list[str]:
    """``text`` split at ``cuts`` (positions, any order, repeats allowed: empty chunks too)."""
    points = sorted({0, len(text), *(c % (len(text) + 1) for c in cuts)})
    return [text[a:b] for a, b in zip(points, points[1:], strict=False)] or [""]


def released(chunks: list[str]) -> str:
    """What the bridge sends for a hop's chunks: each feed, then the hop's flush."""
    f = SilentFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


# Answer words as a model writes them: any characters, "<" and pieces of the marker included (a
# "<" or a marker piece next to a removed marker can build another one around it).
plain = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=20)
piece = st.one_of(plain, st.just(MARK), st.sampled_from([MARK[:n] for n in range(1, len(MARK))]),
                  st.sampled_from([MARK[n:] for n in range(1, len(MARK))]))


def within_bound(text: str) -> bool:
    """No run of marker characters longer than :data:`MAX_HELD`: every run of marker prefixes the
    filter could hold (after removed markers join the text around them) fits in the bound."""
    return max((len(r) for r in re.findall(f"[{re.escape(MARK)}]+", text)), default=0) <= MAX_HELD


answer = st.lists(piece, max_size=8).map("".join).filter(within_bound)
cuts = st.lists(st.integers(min_value=0, max_value=400), max_size=12)


@given(answer, cuts)
def test_silent_marker_is_never_released_over_any_split(text: str, at: list[int]) -> None:
    out = released(stream(text, at))
    assert MARK not in out
    assert out == strip_silent(text)  # nothing else is lost, duplicated or reordered


def test_a_marker_built_around_a_removed_one_is_dropped() -> None:
    assert released(["<sil<silent/>ent/>"]) == ""
    assert released(["<", "<silent/>", "silent/>"]) == ""
    assert released(["a<", "<sil", "ent/>sil", "ent/>b"]) == "ab"
    assert released(["<<sil"]) == "<<sil"  # held, then flushed as written: it never became a marker


def test_a_long_run_of_marker_prefixes_is_released_not_held() -> None:
    # Adversarial text ("<" after "<") is held only up to MAX_HELD characters; the
    # rest goes out as it comes, and what is held stays bounded however long the run.
    f = SilentFilter()
    out = [f.feed("<") for _ in range(10_000)]
    assert "".join(out) == "<" * (10_000 - MAX_HELD)
    assert all(MARK not in o for o in out)
    assert f.flush() == "<" * MAX_HELD
    f = SilentFilter()
    assert f.feed("<sil" * 1000) == ("<sil" * 1000)[:-MAX_HELD]  # one long chunk: the same bound
    assert f.flush() == ("<sil" * 1000)[-MAX_HELD:]


@given(st.lists(piece, max_size=6).filter(lambda ps: within_bound("".join(ps))), st.data())
def test_a_marker_split_into_single_characters_is_dropped(parts: list[str], data: st.DataObject) -> None:
    # The worst split: every marker one character per chunk, the rest cut at random.
    chunks: list[str] = []
    for p in parts:
        chunks += list(p) if p == MARK else stream(p, data.draw(cuts))
    out = released(chunks)
    assert MARK not in out
    assert out == strip_silent("".join(parts))


# Any text at all, "<" included: stream mode must release exactly what one feed of the whole
# text would (message mode sends a message's chunks in one burst, but they still reach the filter
# one by one; this pins that the split never changes the result).
anything = st.text(alphabet=st.sampled_from(list("<silent/> ab")), max_size=40).filter(within_bound)


@given(anything, cuts)
def test_boundaries_never_matter(text: str, at: list[int]) -> None:
    out = released(stream(text, at))
    assert out == released([text]) == strip_silent(text)
    assert MARK not in out


# Past the bound: adversarial text with long runs of "<", "<s", "<sil" and the
# like. Exact equality with strip_silent can't hold there: a released "<sil" piece can meet a later
# "ent/>" (``["<sil" * 10, "ent/>" * 10]`` releases "<silent/>", below). The documented guarantee is
# weaker: no single release ever holds the marker, what is held stays within MAX_HELD, and
# stripping everything released gives strip_silent of the input (no text lost, added or reordered).
run_piece = st.sampled_from(["<", *(MARK[:n] for n in range(1, len(MARK))), MARK,
                             *(MARK[n:] for n in range(1, len(MARK))), "a", " "])
runs = st.tuples(run_piece, st.integers(min_value=1, max_value=40)).map(lambda pn: pn[0] * pn[1])
adversarial = st.lists(st.one_of(runs, plain, piece), max_size=10).map("".join)
long_cuts = st.lists(st.integers(min_value=0, max_value=2000), max_size=30)


@given(adversarial, long_cuts)
def test_past_the_bound_each_release_is_clean_and_nothing_is_lost(text: str, at: list[int]) -> None:
    f = SilentFilter()
    outs: list[str] = []
    for c in stream(text, at):
        outs.append(f.feed(c))
        assert MARK not in outs[-1]
        assert len(f._buf) <= MAX_HELD  # noqa: SLF001 - the bound on what is held
    outs.append(f.flush())
    assert MARK not in outs[-1]
    assert strip_silent("".join(outs)) == strip_silent(text)
    if within_bound(text):
        assert "".join(outs) == strip_silent(text)  # within the bound: exact


def test_past_the_bound_a_marker_can_form_across_releases() -> None:
    # Why the property above is weaker past the bound: the documented trade-off, pinned.
    chunks = ["<sil" * 10, "ent/>" * 10]
    assert strip_silent("".join(chunks)) == ""
    assert released(chunks) == MARK  # formed across two releases, neither of which held it
    assert strip_silent(released(chunks)) == ""




@st.composite
def planners(draw: st.DrawFn) -> tuple[MaskPlanner, str]:
    tool = draw(st.sampled_from(["escalate", "get_balance", "authenticate"]))
    own = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=10_000)))
    planner = MaskPlanner(
        after_ms=0, handoff_wait_ms=draw(st.integers(min_value=0, max_value=10_000)),
        end_mode=draw(st.sampled_from(["wait", "immediate"])),
        chars_per_s=draw(st.floats(min_value=8.0, max_value=30.0)),
        tool_mask=lambda name: ToolMask(phrase_key=f"mask.{name}", handoff_wait_ms=own) if name == tool else None)
    return planner, tool


phrase_text = st.text(min_size=0, max_size=600)
elapsed = st.floats(min_value=0.0, max_value=60.0, allow_nan=False)
# The estimated end of all speech sent so far, relative to the hand-off: past (negative), or ahead.
ahead = st.one_of(st.just(float("-inf")), st.floats(min_value=-60.0, max_value=60.0, allow_nan=False))
now = st.floats(min_value=0.0, max_value=1e6, allow_nan=False)


# Remaining speech estimates (seconds from the hand-off, past or ahead), by basis; any may be absent.
estimate_value = st.floats(min_value=-60.0, max_value=60.0, allow_nan=False)
estimates_st = st.dictionaries(st.sampled_from(list(HOLD_BASES)), estimate_value)


@given(planners(), estimates_st)
def test_hold_is_bounded(pt: tuple[MaskPlanner, str], est: dict[str, float]) -> None:
    # One rule for every hand-off hold (phrase and released text).
    planner, tool = pt
    hold, basis = _handoff_hold_s(planner, tool, est)
    cap = planner.handoff_wait_ms_for(tool) / 1000.0
    assert 0.0 <= hold <= cap
    known = [b for b in HOLD_BASES if b in est]  # ties go to the first in HOLD_BASES
    assert basis == (max(known, key=est.__getitem__) if known else "none")
    if planner.end_mode == "immediate":
        assert hold == 0.0
    else:
        assert hold <= max([0.0, *est.values()]) + 1e-9  # never past the latest estimate


@given(planners(), estimates_st)
def test_hold_is_the_latest_estimate_under_the_cap(pt: tuple[MaskPlanner, str], est: dict[str, float]) -> None:
    planner, tool = pt
    planner.end_mode = "wait"
    hold, _ = _handoff_hold_s(planner, tool, est)
    latest = max([0.0, *est.values()])
    cap = planner.handoff_wait_ms_for(tool) / 1000.0
    assert hold == (latest if latest <= cap else cap)


@given(planners(), estimates_st, st.sampled_from(list(HOLD_BASES)), estimate_value,
       st.floats(min_value=0.0, max_value=60.0, allow_nan=False))
def test_hold_is_monotone_in_each_estimate(pt: tuple[MaskPlanner, str], est: dict[str, float],
                                                which: str, base: float, more: float) -> None:
    # A later end of any one estimate (or one that was absent) never shortens the hold.
    planner, tool = pt
    lower = {**est, which: base}
    higher = {**est, which: base + more}
    assert _handoff_hold_s(planner, tool, lower)[0] <= _handoff_hold_s(planner, tool, higher)[0]
    if which not in est:  # adding an estimate never shortens the hold either
        assert _handoff_hold_s(planner, tool, est)[0] <= _handoff_hold_s(planner, tool, lower)[0]


def mask_at(text: str, at: float) -> _Mask:
    return _Mask(key="mask.escalate", tool="escalate", text=text, at=at)


@given(planners(), phrase_text, elapsed, now)
def test_phrase_only_hold_is_the_old_mask_hold_under_the_cap(
        pt: tuple[MaskPlanner, str], text: str, since: float, at: float) -> None:
    # A masked hand-off with no other speech holds exactly as before
    # (the phrase's remaining estimate) whenever the phrase's estimate fits the cap.
    planner, tool = pt
    est = _hold_estimates(planner, at, speech_end=float("-inf"), hop_text=None, mask=mask_at(text, at - since))
    hold, basis = _handoff_hold_s(planner, tool, est)
    wait_ms = planner.handoff_wait_ms_for(tool)
    remaining = speech_end(text.strip(), 0.0, planner.chars_per_s) - since
    old = 0.0 if planner.end_mode == "immediate" else min(max(remaining, 0.0), wait_ms / 1000.0, max(0.0, wait_ms / 1000.0 - since))
    if not text.strip():
        assert (est, hold, basis) == ({}, 0.0, "none")
    elif speech_end(text.strip(), 0.0, planner.chars_per_s) <= wait_ms / 1000.0:
        assert abs(hold - old) < 1e-6
        assert basis == "phrase"


@given(planners(), now, ahead, st.one_of(st.none(), st.tuples(elapsed, st.integers(0, 600))),
       st.one_of(st.none(), st.tuples(phrase_text, elapsed)))
def test_hold_estimates_are_each_sources_remaining_speech(
        pt: tuple[MaskPlanner, str], at: float, until_end: float,
        hop: tuple[float, int] | None, phrase: tuple[str, float] | None) -> None:
    planner, _ = pt
    hop_text = None if hop is None else (at - hop[0], hop[1])
    mask = None if phrase is None else mask_at(phrase[0], at - phrase[1])
    est = _hold_estimates(planner, at, speech_end=at + until_end, hop_text=hop_text, mask=mask)
    if until_end == float("-inf"):
        assert "watch" not in est  # no speech session: nothing known to be playing
    else:
        assert abs(est["watch"] - until_end) < 1e-6
    if hop is None or hop[1] == 0:
        assert "hop_text" not in est
    else:
        assert abs(est["hop_text"] - (0.3 + hop[1] / planner.chars_per_s - hop[0])) < 1e-6
    if phrase is None or not phrase[0].strip():
        assert "phrase" not in est
    else:
        assert abs(est["phrase"] - (speech_end(phrase[0].strip(), 0.0, planner.chars_per_s) - phrase[1])) < 1e-6
