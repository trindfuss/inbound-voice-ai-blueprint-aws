"""Properties of ``StrandsAgentClient``, offline.

Model streams are scripted (``tests/strands_fakes.py``): a ``FakeModel`` for the loop properties,
and a real ``BedrockModel`` over a faked boto client for the request properties, so the request compared is the exact
ConverseStream request, with every cache point Strands adds.
"""

from __future__ import annotations

import asyncio
import copy
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from owlbank.tools_contract import INLINE_TOOLS
from tac_bridge.agent_client import AgentError, TextDelta, TextSuppressed, ToolCall, ToolResult, TurnEnd
from tac_bridge.agent_client_strands import (
    ALLOWED_GATEWAY_TOOLS,
    StrandsAgentClient,
    default_model_factory,
    pending_interrupts,
)
from tests.strands_fakes import (
    FakeBedrockRuntime,
    FakeGatewayProvider,
    FakeModel,
    ScriptError,
    default_gateway,
    gateway_tool,
    make_client,
    text_msg,
    tool_msg,
)
from tests.strands_fakes import settings as client_settings

SID = "CA" + "7" * 32
PROPS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])

# --------------------------------------------------------------------------- scripted streams

words = st.text(alphabet=string.ascii_letters + " ,.", min_size=1, max_size=24)
inline_use = st.tuples(st.just("inline"), st.sampled_from(INLINE_TOOLS))
gateway_use = st.tuples(st.just("gateway"), st.sampled_from(ALLOWED_GATEWAY_TOOLS))
text_spec = st.tuples(st.just("text"), words, st.integers(1, 4), st.booleans())
tool_spec = st.tuples(st.just("tool"), st.lists(st.one_of(inline_use, gateway_use), min_size=1, max_size=3),
                      st.one_of(st.just(""), words), st.booleans())
gateway_only = st.tuples(st.just("tool"), st.lists(gateway_use, min_size=1, max_size=2),
                         st.one_of(st.just(""), words), st.booleans())
error_spec = st.one_of(st.tuples(st.just("error_before")), st.tuples(st.just("error_mid"), words))
message_spec = st.one_of(text_spec, tool_spec, gateway_only)


def render(spec: tuple) -> Any:
    """A spec as the chunks one model call streams (fresh tool-use ids each time)."""
    kind = spec[0]
    if kind == "text":
        _, text, chunks, reasoning = spec
        return text_msg(text, chunks=chunks, reasoning="" if reasoning else None)
    if kind == "tool":
        _, uses, text, reasoning = spec
        return tool_msg(*[(name, {"account": "checking"} if k == "inline" else {"q": "x"}) for k, name in uses],
                        text=text or None, reasoning="" if reasoning else None)
    if kind == "error_before":
        return ScriptError("scripted failure")
    _, text = spec
    return [{"messageStart": {"role": "assistant"}}, {"contentBlockStart": {"start": {}}},
            {"contentBlockDelta": {"delta": {"text": text}}}, ScriptError("scripted failure")]


def run(coro: Any) -> Any:
    return asyncio.run(coro)


async def collect(gen: Any, out: list[Any]) -> None:
    async for ev in gen:
        out.append(ev)


def agent_of(client: StrandsAgentClient) -> Any:
    return client._calls[SID].build.result().agent




def expected_events(script: list[tuple], max_iter: int = 6) -> tuple[list[int], str, list[str], str]:
    """What the event contract says a turn over ``script`` yields: suppressed lengths, spoken text,
    inline tool calls, and the outcome (end, interrupt, error or limit)."""
    suppressed: list[int] = []
    for i in range(max_iter + 1):
        if i == max_iter:
            return suppressed, "", [], "limit"
        spec = script[i] if i < len(script) else ("text", "ok", 1, False)
        if spec[0] in ("error_before", "error_mid"):
            return suppressed, "", [], "error"
        if spec[0] == "text":
            return suppressed, spec[1], [], "end"
        _, uses, text, _ = spec
        if text:
            suppressed.append(len(text))
        inline = [name for k, name in uses if k == "inline"]
        if inline:
            return suppressed, "", inline, "interrupt"
    raise AssertionError("unreachable")


@PROPS
@given(st.lists(st.one_of(message_spec, error_spec), max_size=8))
def test_event_contract(script):
    """Text only from messages without a tool use; ``TextSuppressed`` exactly for a message
    with both; reasoning never spoken; exactly one ``TurnEnd``, last (or an ``AgentError``)."""
    model = FakeModel([render(s) for s in script])
    # This is message mode's contract; stream mode differs.
    client = make_client(model, gateway=default_gateway(), text_release="message")
    events: list[Any] = []
    error: AgentError | None = None
    try:
        run(collect(client.turn(SID, user_text="hello", tool_results=(), system_prompts={"haiku-5-5": "P"}), events))
    except AgentError as exc:
        error = exc
    suppressed, text, calls, outcome = expected_events(script)
    assert [e.chars for e in events if isinstance(e, TextSuppressed)] == suppressed
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == text
    assert Counter(e.name for e in events if isinstance(e, ToolCall)) == Counter(calls)
    ends = [e for e in events if isinstance(e, TurnEnd)]
    if outcome in ("error", "limit"):
        assert error is not None and not ends
        if outcome == "limit":
            assert error.code == "max_iterations_exceeded"
    else:
        assert error is None
        assert len(ends) == 1 and events[-1] is ends[0]
        assert ends[0].stop_reason == ("tool_use" if outcome == "interrupt" else "end_turn")




def check_pairing(messages: list[dict[str, Any]], *, strict: bool) -> None:
    """Every toolUse has exactly one matching toolResult before the next user text message (or
    at most one, while still pending at the end of the history, unless ``strict``); no result
    without an earlier tool use."""
    used: set[str] = set()
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            uses = [c["toolUse"]["toolUseId"] for c in m["content"] if "toolUse" in c]
            used.update(uses)
            if not uses:
                continue
            counts: Counter[str] = Counter()
            closed = False
            for later in messages[i + 1:]:
                if later["role"] != "user":
                    continue
                counts.update(c["toolResult"]["toolUseId"] for c in later["content"] if "toolResult" in c)
                if any("text" in c for c in later["content"]):
                    closed = True
                    break
            for u in uses:
                if closed or strict:
                    assert counts[u] == 1, (u, counts)
                else:
                    assert counts[u] <= 1, (u, counts)
        else:
            for c in m["content"]:
                if "toolResult" in c:
                    assert c["toolResult"]["toolUseId"] in used


def check_request_adjacency(messages: list[dict[str, Any]]) -> None:
    """What Bedrock requires of every request: an assistant message's tool uses are answered,
    each exactly once, by the very next (user) message."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        uses = Counter(c["toolUse"]["toolUseId"] for c in m["content"] if "toolUse" in c)
        if not uses:
            continue
        assert i + 1 < len(messages) and messages[i + 1]["role"] == "user"
        results = Counter(c["toolResult"]["toolUseId"] for c in messages[i + 1]["content"] if "toolResult" in c)
        assert results == uses


@dataclass(frozen=True)
class Step:
    script: list[tuple]
    answer: list[bool]
    with_text: bool
    cut: tuple[str, int] | None


steps = st.builds(
    Step,
    # weighted toward inline calls, so many turns start with calls pending
    script=st.lists(st.one_of(tool_spec, tool_spec, message_spec), max_size=4),
    answer=st.lists(st.booleans(), min_size=3, max_size=3),
    with_text=st.booleans(),
    cut=st.one_of(st.none(), st.tuples(st.just("events"), st.integers(0, 4)),
                  st.tuples(st.just("ticks"), st.integers(0, 60))),
)


async def consume(gen: Any, cut: tuple[str, int] | None) -> list[Any]:
    """Run a turn to the end, or close it (barge-in) after ``n`` events or ``n`` loop ticks."""
    out: list[Any] = []
    if cut is None:
        await collect(gen, out)
        return out
    kind, n = cut
    if kind == "events":
        try:
            if n:
                async for ev in gen:
                    out.append(ev)
                    if len(out) >= n:
                        break
        finally:
            await gen.aclose()
        return out
    task = asyncio.ensure_future(collect(gen, out))
    for _ in range(n):
        if task.done():
            break
        await asyncio.sleep(0)
    if not task.done():
        task.cancel()
    await asyncio.wait({task})
    await gen.aclose()
    if not task.cancelled() and task.exception() is not None:
        raise task.exception()  # type: ignore[misc]
    return out


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(steps, min_size=1, max_size=6))
def test_tool_pairing_with_barge_in(seq):
    """After any sequence of turns, with ``aclose`` at any point, every tool use in the
    history has exactly one matching result before the next user text message; every request
    the model receives pairs each tool use with its result in the next message."""
    model = FakeModel()
    client = make_client(model, gateway=default_gateway())

    async def go() -> None:
        seen: set[str] = set()
        await client.begin_session(SID)
        for step in seq:
            model.script = [render(s) for s in step.script]
            pending = pending_interrupts(agent_of(client))
            known = [tid for tid in pending if tid in seen]
            results = [ToolResult(tid, {"ok": True}) for tid, yes in zip(known, step.answer, strict=False) if yes]
            text = "caller words" if step.with_text or not results else None
            if pending:
                event("turn with calls pending: " + ("results and text" if results and text else
                                                     "results" if results else "barge-in text"))
            event(f"cut: {step.cut[0] if step.cut else 'none'}")
            try:
                events = await consume(client.turn(SID, user_text=text, tool_results=results, system_prompts={"haiku-5-5": "P"}),
                                       step.cut)
            except AgentError:
                events = []
            seen.update(e.tool_use_id for e in events if isinstance(e, ToolCall))
            if step.cut and pending_interrupts(agent_of(client)):
                event("closed early, calls left pending")
            check_pairing(agent_of(client).messages, strict=False)
        model.script = []
        await consume(client.turn(SID, user_text="final words", tool_results=(), system_prompts={"haiku-5-5": "P"}), None)
        check_pairing(agent_of(client).messages, strict=True)

    run(go())
    for call in model.calls:
        check_request_adjacency(call.messages)



tool_names = st.from_regex(r"[a-z][a-z0-9_]{0,20}", fullmatch=True)
listing = st.lists(st.one_of(st.sampled_from(ALLOWED_GATEWAY_TOOLS), st.sampled_from(INLINE_TOOLS),
                             st.sampled_from(["shell", "file_operations", "level0___delete_all", "faq_lookup"]),
                             tool_names, tool_names.map(lambda n: f"level0___{n}")), max_size=6)


@PROPS
@given(listing)
def test_offered_tools_are_inline_plus_allowed(names):
    """The tools offered to the model are exactly the inline tools plus the approved Gateway
    tools the provider listed, whatever else it lists; an impostor never replaces an inline tool."""
    model = FakeModel([text_msg("hi")])
    provider = FakeGatewayProvider([gateway_tool(n) for n in names])
    run(collect(make_client(model, gateway=provider).turn(SID, user_text="x", tool_results=(), system_prompts={"haiku-5-5": "P"}), []))
    offered = model.calls[0].tool_names
    assert len(offered) == len(set(offered))
    assert set(offered) == set(INLINE_TOOLS) | (set(names) & set(ALLOWED_GATEWAY_TOOLS))
    for spec in model.calls[0].tool_specs:
        if spec["name"] in INLINE_TOOLS:
            assert not spec["description"].startswith("fake gateway tool")




@PROPS
@given(n_tools=st.integers(1, 3), answered=st.lists(st.booleans(), min_size=3, max_size=3),
       prior=st.integers(0, 3), mode=st.sampled_from(["results_and_text", "barge_in"]))
def test_results_then_text(n_tools, answered, prior, mode):
    """Results and caller words in one turn, or words while calls are pending
    (barge-in), give the history ``[..., toolResult message, user text message]`` with every
    pending call answered, and the model's first request ends that way too."""
    uses = [(f"p4-{i}", INLINE_TOOLS[i], {"account": "checking"}) for i in range(n_tools)]
    model = FakeModel([text_msg(f"reply {i}") for i in range(prior)] + [tool_msg(*uses), text_msg("after")])
    client = make_client(model)

    async def go() -> None:
        for i in range(prior):
            await collect(client.turn(SID, user_text=f"turn {i}", tool_results=(), system_prompts={"haiku-5-5": "P"}), [])
        await collect(client.turn(SID, user_text="do it", tool_results=(), system_prompts={"haiku-5-5": "P"}), [])
        results = [] if mode == "barge_in" else [
            ToolResult(tid, {"done": True}) for (tid, _, _), yes in zip(uses, answered, strict=False) if yes]
        calls_before = len(model.calls)
        await collect(client.turn(SID, user_text="NEW WORDS", tool_results=results, system_prompts={"haiku-5-5": "P"}), [])
        assert len(model.calls) == calls_before + 1  # no model call between the results and the words

    run(go())
    msgs = agent_of(client).messages
    idx = next(i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"] == [{"text": "NEW WORDS"}])
    results_msg = msgs[idx - 1]
    assert results_msg["role"] == "user"
    assert {c["toolResult"]["toolUseId"] for c in results_msg["content"]} == {tid for tid, _, _ in uses}
    sent = [(m["role"], m["content"]) for m in model.calls[-1].messages[-2:]]
    assert sent == [(results_msg["role"], results_msg["content"]), ("user", [{"text": "NEW WORDS"}])]




def _history(n: int) -> list[dict[str, Any]]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": [{"text": f"m{i}"}]} for i in range(n)]


@PROPS
@given(history_len=st.integers(0, 90), hops=st.integers(0, 9))
def test_iteration_and_window_limits(history_len, hops):
    """At most 6 model calls per turn, and at most 40 messages in any request, for any
    history length and any number of Gateway hops."""
    model = FakeModel([tool_msg(("level0___faq_lookup", {"q": str(i)})) for i in range(hops)] + [text_msg("done")])
    client = make_client(model, gateway=default_gateway())
    error: AgentError | None = None

    async def go() -> None:
        nonlocal error
        await client.begin_session(SID)
        agent_of(client).messages[:] = _history(history_len)
        try:
            await collect(client.turn(SID, user_text="go", tool_results=(), system_prompts={"haiku-5-5": "P"}), [])
        except AgentError as exc:
            error = exc

    run(go())
    assert len(model.calls) <= 6
    assert all(len(c.messages) <= 40 for c in model.calls)
    if hops + 1 > 6:
        assert error is not None and error.code == "max_iterations_exceeded" and not error.retryable
    else:
        assert error is None and len(model.calls) == hops + 1




def strip_cache_points(request: dict[str, Any]) -> dict[str, Any]:
    """The request with every ``cachePoint`` block removed: system, tools and messages."""
    out = copy.deepcopy(request)
    out["system"] = [b for b in out.get("system", []) if "cachePoint" not in b]
    if "toolConfig" in out:
        out["toolConfig"]["tools"] = [t for t in out["toolConfig"]["tools"] if "cachePoint" not in t]
    for m in out.get("messages", []):
        m["content"] = [c for c in m["content"] if "cachePoint" not in c]
    return out


def _count_cache_points(request: dict[str, Any]) -> int:
    n = sum("cachePoint" in b for b in request.get("system", []))
    n += sum("cachePoint" in t for t in request.get("toolConfig", {}).get("tools", []))
    return n + sum("cachePoint" in c for m in request.get("messages", []) for c in m["content"])


turn_plan = st.lists(st.one_of(st.tuples(st.just("text"), words),
                               st.tuples(st.just("gateway"), words),
                               st.tuples(st.just("inline"), st.sampled_from(INLINE_TOOLS))), min_size=1, max_size=4)


async def _run_plan(client: StrandsAgentClient, rt: FakeBedrockRuntime, plan: list[tuple], base: str,
                    suffixes: list[str]) -> None:
    for i, (kind, value) in enumerate(plan):
        suffix = suffixes[i % len(suffixes)]
        tid = f"p7-{i}"
        if kind == "text":
            rt.script = [text_msg(value)]
        elif kind == "gateway":
            rt.script = [tool_msg((f"{tid}-gw", "level0___faq_lookup", {"q": value})), text_msg("answer")]
        else:
            rt.script = [tool_msg((tid, value, {"account": "checking"}))]
        await collect(client.turn(SID, user_text=f"words {i}", tool_results=(), system_prompts={"haiku-5-5": base},
                                  system_suffix=suffix), [])
        if kind == "inline":
            rt.script = [text_msg("done")]
            await collect(client.turn(SID, user_text=None, tool_results=[ToolResult(tid, {"ok": 1})],
                                      system_prompts={"haiku-5-5": base}, system_suffix=suffix), [])


def _bedrock_client(caching: bool, thinking: str) -> tuple[StrandsAgentClient, FakeBedrockRuntime]:
    rt = FakeBedrockRuntime()

    def factory(s: Any, _sid: str) -> Any:
        model = default_model_factory(s)
        model.client = rt
        return model

    return StrandsAgentClient(client_settings(prompt_caching=caching, thinking=thinking), model_factory=factory,
                              gateway_tools=default_gateway()), rt


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(plan=turn_plan, base=words, suffixes=st.lists(words, min_size=1, max_size=3),
       thinking=st.sampled_from(["adaptive", "off", "default"]))
def test_caching_never_changes_what_the_model_sees(plan, base, suffixes, thinking):
    """With caching on, every request equals the caching-off request once all cache points
    are removed (system, tools, and the one the anthropic strategy adds to the last user
    message); the clock suffix always comes after the last system cache point."""
    on, rt_on = _bedrock_client(True, thinking)
    off, rt_off = _bedrock_client(False, thinking)
    run(_run_plan(on, rt_on, plan, base, suffixes))
    run(_run_plan(off, rt_off, plan, base, suffixes))
    assert len(rt_on.requests) == len(rt_off.requests) > 0
    for req_on, req_off in zip(rt_on.requests, rt_off.requests, strict=True):
        assert _count_cache_points(req_off) == 0
        assert strip_cache_points(req_on) == req_off
        system = req_on["system"]
        points = [i for i, b in enumerate(system) if "cachePoint" in b]
        assert points, "the stable system text is cached"
        suffix_at = max(i for i, b in enumerate(system) if "text" in b)
        assert system[suffix_at]["text"] in suffixes and suffix_at > points[-1]
        assert any("cachePoint" in t for t in req_on["toolConfig"]["tools"])
