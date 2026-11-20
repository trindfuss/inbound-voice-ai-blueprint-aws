"""Property tests for ``tac_bridge.masking.MaskPlanner``.

The planner is pure: time is an argument. A small driver plays a random turn the way the bridge
does (:func:`play`): events in time order, and before each one any scheduled phrase whose
deadline has passed is fired (the bridge's deadline race fires at the deadline, before a later
event).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.masking_config import ToolMask
from owlbank.policy import MASK_GENERIC_KEYS
from tac_bridge.masking import MaskPlanner, ScheduledPhrase, speech_end
from tac_bridge.bridge import _handoff_hold_s, _hold_estimates, _Mask


def _hold(p: MaskPlanner, tool: str | None, text: str | None, elapsed: float) -> float:
    """The bridge's hand-off hold for a phrase sent ``elapsed`` s ago, with no other speech."""
    now = 100.0
    mask = _Mask(key="mask.x", tool=tool or "escalate", text=text or "", at=now - elapsed)
    est = _hold_estimates(p, now, speech_end=float("-inf"), hop_text=None, mask=mask)
    return _handoff_hold_s(p, tool, est)[0]

MASKED = {"list_recent_transactions": "mask.list_recent_transactions", "get_balance": "mask.get_balance",
          "change_address": "mask.change_address", "faq_lookup": "mask.faq_lookup"}
HANDOFF = frozenset({"get_balance", "authenticate", "escalate", "make_payment"})
TOOLS = [*MASKED, "authenticate", "escalate", "make_payment", "get_branch_hours"]
ACK_TEXT = "Give me a moment."


def _planner(after_ms: int = 1500, end_mode: str = "wait", wait_ms: int = 2500, rate: float = 15.0,
             generic_key: str = MASK_GENERIC_KEYS[0], tool_after_ms: int = 600,
             masks: dict[str, ToolMask] | None = None) -> MaskPlanner:
    table = masks if masks is not None else {t: ToolMask(k) for t, k in MASKED.items()}
    return MaskPlanner(after_ms=after_ms, handoff_wait_ms=wait_ms, end_mode=end_mode,  # type: ignore[arg-type]
                       chars_per_s=rate, generic_key=generic_key, tool_after_ms=tool_after_ms,
                       tool_mask=table.get, is_handoff=HANDOFF.__contains__)


# An event in one turn: the acknowledgement timer fires, the turn speaks answer text, or one model
# message calls tools (or a Gateway tool starts). Times are offsets from the previous event.
EVENT = st.one_of(
    st.tuples(st.just("timer"), st.floats(0, 2)),
    st.tuples(st.just("text"), st.floats(0, 2)),
    st.tuples(st.just("tools"), st.floats(0, 2), st.lists(st.sampled_from(TOOLS), min_size=1, max_size=3)),
)

DELAYS = st.one_of(st.none(), st.integers(0, 3000))


@dataclass
class Played:
    spoken: list[tuple[float, ScheduledPhrase]] = field(default_factory=list)  # tool phrases
    spoken_seq: list[int] = field(default_factory=list)  # order on the wire
    acks: list[float] = field(default_factory=list)
    ack_end: float | None = None
    text_at: float | None = None
    text_seq: int | None = None
    seq: int = 0
    scheduled: list[ScheduledPhrase] = field(default_factory=list)


def play(p: MaskPlanner, events: list[tuple], start: float = 0.0) -> Played:
    out = Played()
    now = start

    def fire_due(until: float) -> None:
        if p.pending is not None and p.pending.deadline <= until:
            at = p.pending.deadline
            got = p.on_deadline(at)
            if got is not None:
                out.spoken.append((at, got))
                out.seq += 1
                out.spoken_seq.append(out.seq)

    for ev in events:
        now += ev[1]
        fire_due(now)
        if ev[0] == "timer":
            key = p.on_timer(spoken_anything=out.text_at is not None)
            if key is not None:
                out.acks.append(now)
                p.note_ack(ACK_TEXT, now)
                out.ack_end = speech_end(ACK_TEXT, now, p.chars_per_s)
        elif ev[0] == "text":
            p.on_text()
            out.seq += 1
            if out.text_at is None:
                out.text_at, out.text_seq = now, out.seq
        else:
            got = p.on_tool_calls(ev[2], now)
            assert len(got) <= 1
            out.scheduled += got
            fire_due(now)  # a phrase due at once is spoken before the tools start
    fire_due(now + 100.0)  # the turn keeps waiting (no more text): anything pending falls due
    return out


@settings(max_examples=400)
@given(st.lists(EVENT, max_size=12), st.sampled_from([0, 1500]), st.sampled_from(MASK_GENERIC_KEYS),
       st.integers(0, 2))
def test_one_timer_and_one_tool_phrase_per_turn_never_after_text(
        events, after_ms, generic_key, turns) -> None:
    """Per turn: at most one acknowledgement and at most one tool
    phrase; the acknowledgement only before any tool call and before any answer text, and its key
    is the planner's ``generic_key``; a tool phrase is the first masked tool of its message. A
    phrase of one kind does not block the other. :meth:`reset` starts the rules over."""
    p = _planner(after_ms=after_ms, generic_key=generic_key)
    for turn in range(turns + 1):
        if turn:
            p.reset()
        got = play(p, events)
        assert len(got.acks) <= 1 and len(got.spoken) <= 1
        first_tool_at = None
        t = 0.0
        for ev in events:
            t += ev[1]
            if ev[0] == "tools" and first_tool_at is None:
                first_tool_at = t
        for at in got.acks:
            assert after_ms > 0
            assert got.text_at is None or at <= got.text_at
            assert first_tool_at is None or at <= first_tool_at
        for _, s in got.spoken:
            names = next(ev[2] for ev in events if ev[0] == "tools" and s.tool in ev[2])
            assert s.tool in MASKED and s.tool in names


@settings(max_examples=400)
@given(st.lists(EVENT, max_size=12), st.sampled_from([0, 1500]),
       st.fixed_dictionaries(dict.fromkeys(MASKED, DELAYS)), st.integers(0, 3000),
       st.floats(8, 30))
def test_tool_phrase_timing(events, after_ms, delays, tool_after_ms, rate) -> None:
    """A tool phrase is never spoken after answer text, never earlier than
    its tool call plus its delay (the tool's ``after_ms``, else ``tool_after_ms``), and never before
    the acknowledgement's estimated end (hand-off phrases excepted: at once). Not vacuous:
    a scheduled phrase with no answer text before its deadline is spoken."""
    masks = {t: ToolMask(k, after_ms=delays[t]) for t, k in MASKED.items()}
    p = _planner(after_ms=after_ms, tool_after_ms=tool_after_ms, rate=rate, masks=masks)
    got = play(p, events)
    for (at, s), seq in zip(got.spoken, got.spoken_seq, strict=True):
        assert got.text_seq is None or seq < got.text_seq  # never after answer text
        if s.tool in HANDOFF:
            assert (s.after_ms, s.source) == (0, "handoff") and at == s.called_at
            continue
        want = delays[s.tool] if delays[s.tool] is not None else tool_after_ms
        assert s.after_ms == want and s.source == ("tool" if delays[s.tool] is not None else "default")
        assert at >= s.called_at + want / 1000.0 - 1e-9
        if got.ack_end is not None and got.acks and got.acks[0] <= s.called_at:
            assert at >= got.ack_end - 1e-9
    spoken_ids = {id(s) for _, s in got.spoken}
    for s in got.scheduled:
        if id(s) in spoken_ids:
            continue
        # not spoken: answer text came by its deadline, or a later hand-off phrase replaced it
        later = got.scheduled[got.scheduled.index(s) + 1:]
        replaced = any(o.source == "handoff" for o in later)
        assert replaced or (got.text_at is not None and got.text_at <= s.deadline)


@settings(max_examples=300)
@given(st.text(max_size=200) | st.none(), st.floats(0, 20, allow_nan=False), st.integers(0, 5000),
       st.floats(8, 30, allow_nan=False), st.sampled_from(["wait", "immediate"]))
def test_handoff_wait_is_bounded_and_zero_when_immediate(text, elapsed, wait_ms, rate, mode) -> None:
    w = _hold(_planner(end_mode=mode, wait_ms=wait_ms, rate=rate), None, text, elapsed)
    assert 0.0 <= w <= wait_ms / 1000.0
    if mode == "immediate" or not (text or "").strip():
        assert w == 0.0


@settings(max_examples=300)
@given(st.sampled_from(sorted(HANDOFF | set(MASKED))), DELAYS, st.one_of(st.none(), st.integers(0, 10000)),
       st.integers(0, 10000), st.text(max_size=200), st.floats(0, 20, allow_nan=False),
       st.sampled_from(["wait", "immediate"]))
def test_handoff_phrase_at_once_and_hold_bounded_by_the_tools_wait(
        tool, after_ms, tool_wait, policy_wait, text, elapsed, mode) -> None:
    """A hand-off tool's phrase has delay 0 whatever its ``after_ms``, and the
    hold never exceeds the tool's ``handoff_wait_ms`` when set, else the policy-wide value."""
    masks = {tool: ToolMask(f"mask.{tool}", after_ms=after_ms, handoff_wait_ms=tool_wait)}
    p = _planner(after_ms=0, wait_ms=policy_wait, end_mode=mode, masks=masks)
    (s,) = p.on_tool_calls([tool], now=3.0)
    if tool in HANDOFF:
        assert (s.after_ms, s.source, s.deadline) == (0, "handoff", 3.0)
    cap = p.handoff_wait_ms_for(tool)
    assert cap == (tool_wait if tool_wait is not None else policy_wait)
    w = _hold(p, tool, text, elapsed)
    assert 0.0 <= w <= cap / 1000.0
    if mode == "immediate" or not (text or "").strip():
        assert w == 0.0
