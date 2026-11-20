"""Latency-masking phrases in the bridge.

Wire order through real TAC and the ConversationRelay simulator; concurrency, the timer race and
the hand-off hold by driving :meth:`Bridge.run_turn` directly with the same fakes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from owlbank.catalog import PromptCatalog
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import MASK_GENERIC_KEYS, Policy
from owlbank.segments import SegmentLog
from owlbank.store import MemoryStore
from tac_bridge.agent_client import TextDelta, ToolCall, TurnEnd
from tac_bridge.bridge import Bridge
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    CALL_TOKEN,
    NOW,
    SEALER,
    FakeAgent,
    FakeBackend,
    _inline_io,
    end_frame,
    load_json,
    make_rig,
    new_agent,
    run,
    say,
    seeded_store,
    tool,
    write_call_state,
)
from tests.logutil import lines

BRIDGE_LOG = "tac_bridge.bridge"
_CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))


def en(key: str) -> str:
    """A catalog phrase in en-US, as the bridge sends it (stripped; the bridge adds one space)."""
    return _CATALOG.text(key, "en-US").strip()


TX_PHRASE = en("mask.list_recent_transactions")
BAL_PHRASE = en("mask.get_balance")
ADDR_PHRASE = en("mask.change_address")
GENERIC_KEY = MASK_GENERIC_KEYS[0]  # the first acknowledgement of a call
GENERIC = en(GENERIC_KEY)
TXNS = {"tool": "list_recent_transactions",
        "data": {"accountType": "checking", "last4": "4821", "transactions": []}}
BALANCE = {"tool": "get_balance", "data": {"accountType": "checking", "last4": "4821",
                                           "available": "1234.56", "current": "1250.00",
                                           "pending": "15.44", "currency": "USD", "asOf": None}}


# --------------------------------------------------------------------------- a direct rig


@dataclass
class Direct:
    bridge: Bridge
    agent: FakeAgent
    backend: Any
    store: MemoryStore
    journal: SegmentLog
    order: list[str] = field(default_factory=list)

    def call(self) -> dict[str, Any]:
        return self.store.get("calls", {"call_sid": CALL_SID}) or {}

    def events(self) -> list[dict[str, Any]]:
        return self.journal.read(CALL_SID)

    def ai_turn_text(self) -> str:
        return " ".join(str(e.get("text", "")) for e in self.events() if e["type"] == "ai_turn")


def direct(script: list[Any], *, backend: Any | None = None, policy_doc: dict[str, Any] | None = None,
           catalog_doc: dict[str, Any] | None = None, agent_delay_s: float = 0.0,
           slow_journal_s: float = 0.0, agent: Any | None = None, **state: Any) -> Direct:
    """A bridge without TAC. ``slow_journal_s`` makes every journal append suspend that long and
    records it in ``order`` (with the backend's own records), to show what ran before it."""
    store = seeded_store()
    write_call_state(store, **state)
    # Default: no acknowledgement and tool phrases at once (tool_after_ms=0, the timing that
    # these wire-order and concurrency tests pin; the delayed phrases are tested in
    # test_scheduled_masking.py).
    policy = Policy.from_dict(policy_doc or masked_policy_doc(after_ms=0, tool_after_ms=0))
    journal = SegmentLog(store, now=lambda: NOW)
    agent = agent if agent is not None else new_agent(script, store=store, delay_s=agent_delay_s)
    order: list[str] = []
    backend = backend or FakeBackend({"list_recent_transactions": lambda **_: dict(TXNS),
                                      "get_balance": lambda **_: dict(BALANCE)})

    async def io(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if fn == journal.append:
            order.append(f"journal:{args[1].get('type')}")
            if slow_journal_s:
                await asyncio.sleep(slow_journal_s)
        return await _inline_io(fn, *args, **kwargs)

    bridge = Bridge(store=store, agent=agent, backend=backend, memory=None,
                    catalog=PromptCatalog.from_dict(catalog_doc or load_json("prompt_catalog.json")),
                    policy=policy, pii=PiiGuard.for_policy(FakePiiRedactor(), policy), journal=journal,
                    sealer=SEALER, now=lambda: NOW, io=io)
    d = Direct(bridge, agent, backend, store, journal, order)
    if isinstance(backend, RecordingBackend):
        backend.order = order
    return d


def context() -> SimpleNamespace:
    return SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_mask", metadata={},
                           profile_id=None, profile=None, pending_handoff_data=None)


async def collect(d: Direct, message: str | None, ctx: SimpleNamespace | None = None,
                  ) -> tuple[list[str], dict[str, float]]:
    """The turn's chunks, and when (monotonic) each distinct chunk was yielded and the turn ended."""
    ctx = ctx or context()
    chunks: list[str] = []
    times: dict[str, float] = {}
    async for chunk in d.bridge.run_turn(ctx, message):
        d.order.append(f"yield:{chunk.strip()}")
        chunks.append(chunk)
        times.setdefault(chunk.strip(), time.monotonic())
    times["__end__"] = time.monotonic()
    return chunks, times


@dataclass
class RecordingBackend:
    """``bank_tools`` that records start times; ``gate`` makes calls wait for each other."""

    responses: dict[str, Callable[..., Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)
    started: dict[str, float] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    barrier: int = 0  # >0: each call waits until this many calls have started (or 1 s)
    delay_s: dict[str, float] = field(default_factory=dict)
    _arrived: asyncio.Event | None = None

    async def call(self, op: str, *, token: str | None, call_sid: str, args: Any,
                   idempotency_key: str | None = None) -> dict[str, Any]:
        self.calls.append({"op": op, "token": token, "call_sid": call_sid, "args": dict(args)})
        self.started[f"{op}:{len(self.calls)}"] = time.monotonic()
        self.order.append(f"backend:{op}")
        if self.barrier:
            if self._arrived is None:
                self._arrived = asyncio.Event()
            if len(self.calls) >= self.barrier:
                self._arrived.set()
            await asyncio.wait_for(self._arrived.wait(), 1.0)  # sequential execution times out here
        if self.delay_s.get(op):
            await asyncio.sleep(self.delay_s[op])
        r = self.responses[op](args=args)
        if isinstance(r, Exception):
            raise r
        return dict(r)


def mask_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "mask_spoken"]


# --------------------------------------------------------------------------- wire order (through TAC)


def test_masked_inline_tool_phrase_is_on_the_wire_before_the_answer(caplog: pytest.LogCaptureFixture) -> None:
    backend = FakeBackend({"list_recent_transactions": [dict(TXNS)]})
    rig = make_rig(backend=backend, policy_doc=masked_policy_doc(after_ms=0, tool_after_ms=0), script=[
        tool("list_recent_transactions", "tu_tx", account="checking"),
        say("You have no recent", " transactions."),
    ])
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("What did I spend recently?")
        await sim.hangup()
        return frames

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        frames = run(go())
    tokens = [f["token"] for f in frames if f.get("type") == "text"]
    assert tokens == [TX_PHRASE + " ", "You have no recent", " transactions.", ""]
    assert frames[-1]["last"] is True
    events = rig.events()
    types = [e["type"] for e in events]
    assert types == ["prompt", "tool_call", "tool_result", "ai_turn"]
    assert events[0]["prompt_key"] == "mask.list_recent_transactions"
    assert events[0]["rendered_text"] == TX_PHRASE
    (ln,) = mask_lines(caplog)
    assert ln["key"] == "mask.list_recent_transactions" and ln["tool"] == "list_recent_transactions"
    assert ln["kind"] == "tool" and ln["ms_since_turn_start"] >= 0
    assert (ln["after_ms"], ln["source"]) == (0, "default")  # the delay and its source
    # the phrase reached the caller only.
    assert TX_PHRASE not in rig.agent.all_inputs_text()
    assert all(TX_PHRASE not in str(e.get("text", "")) for e in events if e["type"] == "ai_turn")


def test_one_tool_phrase_per_turn_and_none_after_spoken_text() -> None:
    """Through the bridge: a second masked hop, or a hop after the answer, gets no tool phrase."""
    d = direct([tool("list_recent_transactions", "tu_1", account="checking"),
                tool("list_recent_transactions", "tu_2", account="savings"),
                say("Done.")])
    chunks, _ = run(collect(d, "transactions please"))
    assert chunks == [TX_PHRASE + " ", "Done."]


@pytest.mark.protocol_fake("a step no real client emits: text spoken, then a tool call, in one turn() "
                           "(a real client suppresses text of a tool-using message)")
def test_no_tool_phrase_after_spoken_text() -> None:
    """No tool phrase once the turn has spoken text."""
    # The agent client hands over text of a tool-using message only as TextSuppressed,
    # so "text, then a tool" means a spoken text-only hop before the tool hop of the same turn.
    d = direct([[TextDelta("Sure. "), ToolCall("tu_1", "list_recent_transactions",
                                               {"account": "checking"}), TurnEnd("tool_use")],
                say("Nothing recent.")])
    chunks, _ = run(collect(d, "transactions please"))
    assert chunks == ["Sure. ", "Nothing recent."]


def test_level_too_low_for_the_tool_gets_no_phrase() -> None:
    """Nothing is fetched for a caller below the tool's level (AUTH_LEVEL_INSUFFICIENT at once)."""
    d = direct([tool("change_address", "tu_ca", street="1 Main St", city="Springfield", state="CA",
                     postal_code="94105"),
                say("I need to verify you first.")], level=1)
    chunks, _ = run(collect(d, "change my address"))
    assert chunks == ["I need to verify you first."]


# --------------------------------------------------------------------------- concurrency, nothing waits


def test_two_inline_tools_in_one_message_run_concurrently() -> None:
    backend = RecordingBackend({"list_recent_transactions": lambda **_: dict(TXNS)}, barrier=2)
    d = direct([[ToolCall("tu_a", "list_recent_transactions", {"account": "checking"}),
                 ToolCall("tu_b", "list_recent_transactions", {"account": "savings"}),
                 TurnEnd("tool_use")],
                say("Both are quiet.")], backend=backend)
    chunks, _ = run(collect(d, "recent on both"))
    assert chunks == [TX_PHRASE + " ", "Both are quiet."]
    # Both results are successes: a sequential run would have timed out at the barrier.
    (results,) = [c["tool_results"] for c in d.agent.calls[1:2]]
    assert [r.tool_use_id for r in results] == ["tu_a", "tu_b"]  # the model's call order
    assert all(r.status == "success" for r in results)
    events = [e for e in d.events() if e["type"] in ("tool_call", "tool_result")]
    assert [(e["type"], e["tool_use_id"]) for e in events] == [
        ("tool_call", "tu_a"), ("tool_result", "tu_a"), ("tool_call", "tu_b"), ("tool_result", "tu_b")]


def test_the_tools_start_before_anything_is_awaited_after_the_phrase() -> None:
    backend = RecordingBackend({"list_recent_transactions": lambda **_: dict(TXNS)})
    d = direct([[ToolCall("tu_a", "list_recent_transactions", {"account": "checking"}),
                 ToolCall("tu_b", "list_recent_transactions", {"account": "savings"}),
                 TurnEnd("tool_use")],
                say("Both are quiet.")], backend=backend, slow_journal_s=0.05)
    run(collect(d, "recent on both"))
    i = d.order.index(f"yield:{TX_PHRASE}")
    # Straight after the phrase: both bank calls, before the first (slow) journal write.
    assert d.order[i + 1:i + 3] == ["backend:list_recent_transactions"] * 2
    assert d.order[i + 3] == "journal:prompt"


def test_unmasked_message_still_runs_its_tools_concurrently() -> None:
    backend = RecordingBackend({"list_recent_transactions": lambda **_: dict(TXNS)}, barrier=2)
    d = direct([[ToolCall("tu_a", "list_recent_transactions", {"account": "checking"}),
                 ToolCall("tu_b", "list_recent_transactions", {"account": "savings"}),
                 TurnEnd("tool_use")],
                say("Both are quiet.")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, tool_after_ms=0,
                                            masks={"list_recent_transactions": None}))
    chunks, _ = run(collect(d, "recent on both"))
    assert chunks == ["Both are quiet."]
    assert all(r.status == "success" for r in d.agent.calls[1]["tool_results"])


def test_semantics_kept_call_alone() -> None:
    """CALL_ALONE through the real client (the unknown-tool half uses the plain fake, below)."""
    d = direct([[ToolCall("tu_b", "get_balance", {"account": "checking"}),
                 ToolCall("tu_a", "list_recent_transactions", {"account": "checking"}),
                 TurnEnd("tool_use")],
                say("OK.")])
    chunks, _ = run(collect(d, "hi"))
    assert chunks == [TX_PHRASE + " ", "OK."]
    # By tool: the Strands client emits a hand-off tool's call (get_balance) after the invocation's
    # Memory flush, so after the message's other calls.
    contents = {r.tool_use_id: r.content for r in d.agent.calls[1]["tool_results"]}
    assert contents["tu_b"] == {"error": "CALL_ALONE", "tool": "get_balance"}
    assert contents["tu_a"]["tool"] == "list_recent_transactions"
    assert d.call().get("prefetched") is None


@pytest.mark.protocol_fake("the Strands client offers the model only the known tools, and Strands "
                           "answers an unknown tool name inside its own loop: the bridge's own defence")
def test_semantics_kept_unknown_tool_and_call_alone() -> None:
    d = direct([[ToolCall("tu_x", "no_such_tool", {}),
                 ToolCall("tu_b", "get_balance", {"account": "checking"}),
                 ToolCall("tu_a", "list_recent_transactions", {"account": "checking"}),
                 TurnEnd("tool_use")],
                say("OK.")])
    chunks, _ = run(collect(d, "hi"))
    # get_balance is refused (CALL_ALONE), so it gets no phrase; the inline tool does.
    assert chunks == [TX_PHRASE + " ", "OK."]
    contents = [r.content for r in d.agent.calls[1]["tool_results"]]
    assert contents[0] == {"error": "UNKNOWN_TOOL", "tool": "no_such_tool"}
    assert contents[1] == {"error": "CALL_ALONE", "tool": "get_balance"}
    assert contents[2]["tool"] == "list_recent_transactions"
    assert d.call().get("prefetched") is None


# --------------------------------------------------------------------------- the timer


def test_timer_phrase_on_a_slow_first_hop(caplog: pytest.LogCaptureFixture) -> None:
    d = direct([say("Here you go.")], agent_delay_s=0.15, policy_doc=masked_policy_doc(after_ms=50))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "hello"))
    assert chunks == [GENERIC + " ", "Here you go."]  # the stream was not cancelled by the timer
    assert d.agent.closed == 1
    (ln,) = mask_lines(caplog)
    assert ln["key"] == GENERIC_KEY and ln["tool"] is None and ln["kind"] == "timer"
    assert ln["ms_since_turn_start"] >= 50
    prompts = [e for e in d.events() if e["type"] == "prompt"]
    assert prompts == [{**prompts[0], "prompt_key": GENERIC_KEY, "rendered_text": GENERIC}]
    assert GENERIC not in d.agent.all_inputs_text() and GENERIC not in d.ai_turn_text()


def test_slow_first_hop_then_masked_tool_speaks_the_acknowledgement_then_the_tool_phrase(
        caplog: pytest.LogCaptureFixture) -> None:
    """One timed acknowledgement and one per-tool phrase in the same turn, in
    that order on the wire, then the answer; each logged as its own ``mask_spoken`` line.

    The tool phrase now waits for the acknowledgement's estimated end
    (0.3 s + chars / chars_per_s), so the tool is slow here (0.9 s) to keep the turn waiting
    past it; a tool that answers sooner gets no phrase (test_scheduled_masking.py)."""
    backend = RecordingBackend({"list_recent_transactions": lambda **_: dict(TXNS)},
                               delay_s={"list_recent_transactions": 0.9})
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")],
               backend=backend, agent_delay_s=0.06,
               policy_doc=masked_policy_doc(after_ms=50, tool_after_ms=0, chars_per_s=30))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, times = run(collect(d, "transactions"))
    assert chunks == [GENERIC + " ", TX_PHRASE + " ", "Nothing recent."]
    yields = [o for o in d.order if o.startswith("yield:")]
    assert yields == [f"yield:{GENERIC}", f"yield:{TX_PHRASE}", "yield:Nothing recent."]
    assert times[TX_PHRASE] - times[GENERIC] >= 0.3 + len(GENERIC) / 30 - 0.01  # after the ack's end
    timer, tool_ln = mask_lines(caplog)
    assert (timer["kind"], timer["key"], timer["tool"]) == ("timer", GENERIC_KEY, None)
    assert (tool_ln["kind"], tool_ln["key"], tool_ln["tool"]) == (
        "tool", "mask.list_recent_transactions", "list_recent_transactions")
    assert timer["ms_since_turn_start"] <= tool_ln["ms_since_turn_start"]
    prompts = [e["prompt_key"] for e in d.events() if e["type"] == "prompt"]
    assert prompts == [GENERIC_KEY, "mask.list_recent_transactions"]
    # neither phrase reached the agent or the ai_turn journal text.
    for phrase in (GENERIC, TX_PHRASE):
        assert phrase not in d.agent.all_inputs_text() and phrase not in d.ai_turn_text()


def test_no_timer_phrase_on_a_fast_hop() -> None:
    d = direct([say("Here you go.")], policy_doc=masked_policy_doc(after_ms=50))
    chunks, _ = run(collect(d, "hello"))
    assert chunks == ["Here you go."]


def test_timer_only_on_the_first_hop() -> None:
    # The first hop's tool call comes fast; the slow second hop gets no acknowledgement.
    agent_script = [tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")]
    d = direct(agent_script, policy_doc=masked_policy_doc(after_ms=50,
                                                          masks={"list_recent_transactions": None}))

    real_turn = d.agent.turn

    def turn(*a: Any, **kw: Any) -> Any:
        d.agent.delay_s = 0.15 if d.agent.calls else 0.0  # slow from the second hop on
        return real_turn(*a, **kw)

    d.agent.turn = turn  # type: ignore[method-assign]
    chunks, _ = run(collect(d, "transactions"))
    assert chunks == ["Nothing recent."]


def test_no_timer_phrase_on_an_await_caller_resume() -> None:
    env = {"status": "completed", "tool": "get_balance", "next": "await_caller", "spoken": []}
    d = direct([say("<silent/>")], agent_delay_s=0.15, policy_doc=masked_policy_doc(after_ms=50),
               pending_tool={"tool_use_id": "tu_bal", "name": "get_balance", "args": {"account": "checking"}},
               pending_result=env)
    chunks, _ = run(collect(d, ""))
    assert "".join(chunks).strip() == ""
    (rec,) = d.agent.calls
    assert rec["tool_results"][0].content["next"] == "await_caller"


def test_no_timer_phrase_on_a_retry_original_tool_resume() -> None:
    # a resume right after the system's own scripted line ("Thanks, you're verified.")
    # does not arm the timed acknowledgement; only the caller's words do.
    env = {"status": "completed", "tool": "authenticate", "next": "retry_original_tool"}
    d = direct([say("Here are your transactions.")], agent_delay_s=0.15,
               policy_doc=masked_policy_doc(after_ms=50),
               pending_tool={"tool_use_id": "tu_auth", "name": "authenticate",
                             "args": {"tool": "list_recent_transactions"}},
               pending_result=env)
    chunks, _ = run(collect(d, None))
    assert chunks == ["Here are your transactions."]


# --------------------------------------------------------------------------- hand-off


def balance_backend(**kw: Any) -> RecordingBackend:
    return RecordingBackend({"get_balance": lambda **_: dict(BALANCE)}, **kw)


def test_get_balance_handoff_prefetches_and_holds_at_most_handoff_wait_ms(
        caplog: pytest.LogCaptureFixture) -> None:
    backend = balance_backend()
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=400, chars_per_s=8, end_mode="wait"))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, times = run(collect(d, "my checking balance", ctx))
    assert chunks == [BAL_PHRASE + " "]
    held = times["__end__"] - times[BAL_PHRASE]
    assert 0.3 <= held <= 0.4 + 0.1  # the phrase is long at 8 chars/s: capped at handoff_wait_ms
    # the prefetch started at once, with the call token, read-only.
    assert backend.calls == [{"op": "get_balance", "token": CALL_TOKEN, "call_sid": CALL_SID,
                              "args": {"account": "checking"}}]
    assert backend.started["get_balance:1"] - times[BAL_PHRASE] < 0.05
    pre = d.call()["prefetched"]
    assert pre == {"tool_use_id": "tu_bal", "tool": "get_balance", "args": {"account": "checking"},
                   "result": BALANCE, "at": NOW}
    assert d.call()["pending_tool"]["tool_use_id"] == "tu_bal"
    assert d.call()["pending_tool"]["args"] == pre["args"]
    assert ctx.pending_handoff_data is not None
    (hold,) = [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "handoff_hold"]
    assert hold["prefetched"] is True and hold["wait_ms"] <= 400
    assert BAL_PHRASE not in d.agent.all_inputs_text()


def test_handoff_hold_ends_with_the_phrase_estimate() -> None:
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=balance_backend(),
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=5000, chars_per_s=30, end_mode="wait"))
    _, times = run(collect(d, "balance"))
    estimate = 0.3 + len(BAL_PHRASE) / 30
    assert estimate - 0.05 <= times["__end__"] - times[BAL_PHRASE] <= estimate + 0.15


def test_end_mode_immediate_does_not_sleep() -> None:
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=balance_backend(),
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=2500, end_mode="immediate"))
    ctx = context()
    _, times = run(collect(d, "balance", ctx))
    assert times["__end__"] - times[BAL_PHRASE] < 0.1
    assert d.call()["prefetched"]["tool_use_id"] == "tu_bal"
    assert ctx.pending_handoff_data is not None


def test_failing_prefetch_still_hands_off(caplog: pytest.LogCaptureFixture) -> None:
    backend = RecordingBackend({"get_balance": lambda **_: RuntimeError("lambda down")})
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=200))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "balance", ctx))
    assert chunks == [BAL_PHRASE + " "]
    assert ctx.pending_handoff_data is not None and d.call()["pending_tool"]["name"] == "get_balance"
    assert d.call().get("prefetched") is None
    warnings = [r for r in caplog.records if r.name == BRIDGE_LOG and r.levelno == logging.WARNING]
    assert any(getattr(r, "decision", None) == "front_door_fetches" for r in warnings)


def test_error_result_is_not_prefetched() -> None:
    backend = RecordingBackend({"get_balance": lambda **_: {"error": "NOT_FOUND", "tool": "get_balance"}})
    d = direct([tool("get_balance", "tu_bal", account="savings")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=200))
    ctx = context()
    run(collect(d, "savings balance", ctx))
    assert ctx.pending_handoff_data is not None and d.call().get("prefetched") is None


def test_late_prefetch_does_not_hold_the_handoff_and_is_stored_when_it_lands(caplog: pytest.LogCaptureFixture) -> None:
    """The hand-off never waits for the work. In ``immediate`` mode ``end`` follows the
    phrase at once; a prefetch still running is stored from the background when it completes."""
    backend = balance_backend(delay_s={"get_balance": 0.3})
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=2500, end_mode="immediate"))
    ctx = context()

    async def scenario() -> tuple[dict[str, float], Any]:
        _, times = await collect(d, "balance", ctx)
        before = d.call().get("prefetched")
        await d.bridge.wait_idle()
        return times, before

    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        times, before = run(scenario())
    assert times["__end__"] - times[BAL_PHRASE] <= 0.1  # no hold at all
    assert ctx.pending_handoff_data is not None
    assert before is None  # not yet when the session ended
    stored = d.call().get("prefetched")
    assert stored and stored["tool_use_id"] == "tu_bal" and stored["args"] == {"account": "checking"}


def test_wait_mode_gives_the_prefetch_no_more_than_the_phrase(caplog: pytest.LogCaptureFixture) -> None:
    """In ``wait`` mode the hold is the phrase's estimated time; a slower prefetch does not extend it."""
    backend = balance_backend(delay_s={"get_balance": 3.0})
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=2500, end_mode="wait", chars_per_s=30))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        _, times = run(collect(d, "balance"))
    phrase_s = 0.3 + len(BAL_PHRASE) / 30
    assert times["__end__"] - times[BAL_PHRASE] <= phrase_s + 0.15


def test_float_in_the_bank_result_is_stored_as_a_string() -> None:
    raw = {"tool": "get_balance", "data": {"available": 12.5, "current": 13.25}}
    backend = RecordingBackend({"get_balance": lambda **_: dict(raw)})
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, end_mode="immediate"))
    run(collect(d, "balance"))
    assert d.call()["prefetched"]["result"] == {"tool": "get_balance",
                                                "data": {"available": "12.5", "current": "13.25"}}


def test_unmasked_get_balance_neither_prefetches_nor_waits() -> None:
    backend = balance_backend()
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=backend,
               policy_doc=masked_policy_doc(after_ms=0, masks={"get_balance": None}))
    ctx = context()
    chunks, times = run(collect(d, "balance", ctx))
    assert chunks == [] and backend.calls == [] and d.call().get("prefetched") is None
    assert ctx.pending_handoff_data is not None


def test_make_payment_handoff_with_a_phrase_is_never_prefetched() -> None:
    backend = RecordingBackend({})
    d = direct([tool("make_payment", "tu_pay", from_account="checking", to="red", amount="25.00")],
               backend=backend, level=3,
               policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=150,
                                            masks={"make_payment": "mask.change_address"}))
    ctx = context()
    chunks, times = run(collect(d, "pay 25 on my card", ctx))
    assert chunks == [ADDR_PHRASE + " "]
    assert backend.calls == []  # state-changing: runs only after the scripted confirmation
    assert d.call().get("prefetched") is None
    assert d.call()["pending_tool"]["name"] == "make_payment" and ctx.pending_handoff_data is not None
    assert times["__end__"] - times[ADDR_PHRASE] <= 0.15 + 0.1


def test_handoff_through_tac_ends_after_the_phrase() -> None:
    backend = FakeBackend({"get_balance": [dict(BALANCE)]})
    rig = make_rig(backend=backend, script=[tool("get_balance", "tu_bal", account="checking")],
                   policy_doc=masked_policy_doc(after_ms=0, handoff_wait_ms=300))
    write_call_state(rig.store)

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("checking balance")
        await sim.hangup()
        return frames

    frames = run(go())
    types = [(f["type"], f.get("token")) for f in frames]
    assert types[0] == ("text", BAL_PHRASE + " ")
    assert types[-1][0] == "end"
    end = end_frame(frames)
    assert end is not None and end["data"]["tool"] == "get_balance"
    assert rig.call()["prefetched"]["tool_use_id"] == "tu_bal"
    assert backend.calls[0]["token"] == CALL_TOKEN


def test_barge_in_during_the_timer_closes_the_stream() -> None:
    """Closing the turn while the agent is still slow cancels the pending read and
    closes the agent stream, as for any interruption."""
    d = direct([say("Late answer.")], agent_delay_s=0.3, policy_doc=masked_policy_doc(after_ms=30))

    async def go() -> list[str]:
        gen = d.bridge.run_turn(context(), "hello")
        first = await anext(gen)
        await gen.aclose()
        return [first]

    assert run(go()) == [GENERIC + " "]
    assert d.agent.closed == 1
