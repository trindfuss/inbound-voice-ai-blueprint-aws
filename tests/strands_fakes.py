"""Offline stand-ins for the Strands client's dependencies.

* :class:`FakeModel`: a Strands ``Model`` that replays scripted ConverseStream chunks (text, several
  tool uses per message, reasoning blocks, usage with cache tokens, errors) with an optional delay
  per chunk, and records what each call received (messages, tool specs, system blocks).
* :class:`FakeBedrockRuntime`: a stand-in for the boto3 ``bedrock-runtime`` client, put under a
  real ``BedrockModel`` to capture the exact request (cache points included) or raise a botocore
  ``ClientError``.
* Fake Gateway tools (plain Strands tools named like the Gateway's, ``level0___faq_lookup``) and a
  provider that can list unapproved extras or fail.
* :class:`FakeMemoryData`: an in-memory AgentCore Memory data plane (the ``bedrock-agentcore`` boto3
  client: ``create_event``, ``list_events``, ``get_event``, ``delete_event``) with the service's
  ordering (newest first), metadata filters and pagination, so the session manager runs unchanged
  on it, batching layout included. :class:`FakeBotoSession` hands it to the session manager.

Nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from botocore.exceptions import ClientError
from strands import tool
from strands.models.model import Model
from strands.types.tools import ToolContext

from tac_bridge.agent_client import MessageEnd
from tac_bridge.agent_client_strands import (
    MEMORY_ACTOR_ID,
    ModelEntry,
    StrandsAgentClient,
    StrandsSettings,
    memory_session_manager_factory,
)

Chunk = dict[str, Any]

_ids = itertools.count(1)


def new_id() -> str:
    return f"tu-{next(_ids)}"


def usage(inp: int = 10, out: int = 5, cache_read: int | None = None, cache_write: int | None = None) -> Chunk:
    u: dict[str, int] = {"inputTokens": inp, "outputTokens": out, "totalTokens": inp + out}
    if cache_read is not None:
        u["cacheReadInputTokens"] = cache_read
    if cache_write is not None:
        u["cacheWriteInputTokens"] = cache_write
    return {"metadata": {"usage": u, "metrics": {"latencyMs": 1}}}


def _text_block(text: str, chunks: int) -> list[Chunk]:
    out: list[Chunk] = [{"contentBlockStart": {"start": {}}}]
    step = max(1, -(-len(text) // max(1, chunks)))
    for i in range(0, len(text), step):
        out.append({"contentBlockDelta": {"delta": {"text": text[i:i + step]}}})
    out.append({"contentBlockStop": {}})
    return out


def _reasoning_block(text: str = "", signature: str | None = None) -> list[Chunk]:
    signature = f"sig-{text}" if signature is None else signature  # one per text, as a model signs
    out: list[Chunk] = [{"contentBlockStart": {"start": {}}}]
    if text:
        out.append({"contentBlockDelta": {"delta": {"reasoningContent": {"text": text}}}})
    out.append({"contentBlockDelta": {"delta": {"reasoningContent": {"signature": signature}}}})
    out.append({"contentBlockStop": {}})
    return out


def text_msg(text: str, *, chunks: int = 1, reasoning: str | None = None, use: Chunk | None = None) -> list[Chunk]:
    """An assistant message with only text (``end_turn``)."""
    ev: list[Chunk] = [{"messageStart": {"role": "assistant"}}]
    if reasoning is not None:
        ev += _reasoning_block(reasoning)
    if text:
        ev += _text_block(text, chunks)
    ev += [{"messageStop": {"stopReason": "end_turn"}}, use or usage()]
    return ev


def tool_msg(*uses: tuple[str, dict[str, Any]] | tuple[str, str, dict[str, Any]], text: str | None = None,
             chunks: int = 1, reasoning: str | None = None, use: Chunk | None = None) -> list[Chunk]:
    """An assistant message with optional reasoning and text, then one or more tool uses
    (``(name, input)`` with a fresh id, or ``(id, name, input)``), stopping with ``tool_use``."""
    ev: list[Chunk] = [{"messageStart": {"role": "assistant"}}]
    if reasoning is not None:
        ev += _reasoning_block(reasoning)
    if text:
        ev += _text_block(text, chunks)
    for u in uses:
        tid, name, inp = u if len(u) == 3 else (new_id(), *u)  # type: ignore[misc]
        ev += [{"contentBlockStart": {"start": {"toolUse": {"toolUseId": tid, "name": name}}}},
               {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(inp)}}}},
               {"contentBlockStop": {}}]
    ev += [{"messageStop": {"stopReason": "tool_use"}}, use or usage()]
    return ev


def blocks_msg(*blocks: tuple[Any, ...], stop: str | None = None, use: Chunk | None = None) -> list[Chunk]:
    """An assistant message with its content blocks in the given order (text
    after a tool-use block, text between two tool uses). Each block is ``("text", text[, chunks])``,
    ``("reasoning", text)`` or ``("tool", name, input[, id])``. ``stop`` defaults to ``tool_use``
    when there is a tool block, else ``end_turn``."""
    ev: list[Chunk] = [{"messageStart": {"role": "assistant"}}]
    for b in blocks:
        if b[0] == "text":
            ev += _text_block(b[1], b[2] if len(b) > 2 else 1)
        elif b[0] == "reasoning":
            ev += _reasoning_block(b[1])
        else:
            tid = b[3] if len(b) > 3 else new_id()
            ev += [{"contentBlockStart": {"start": {"toolUse": {"toolUseId": tid, "name": b[1]}}}},
                   {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(b[2])}}}},
                   {"contentBlockStop": {}}]
    if stop is None:
        stop = "tool_use" if any(b[0] == "tool" for b in blocks) else "end_turn"
    ev += [{"messageStop": {"stopReason": stop}}, use or usage()]
    return ev


class ScriptError(Exception):
    """Raised by a scripted stream (stands for any provider failure)."""


@dataclass
class Call:
    messages: list[dict[str, Any]]
    tool_specs: list[dict[str, Any]]
    system: list[dict[str, Any]] | None

    @property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tool_specs]


class FakeModel(Model):
    """Replays ``script``: one entry per model call, each a list of chunks (an ``Exception`` in the
    list is raised when reached) or an ``Exception`` (raised before the first chunk). When the
    script runs out, the model answers ``default``. ``delay`` sleeps before each chunk."""

    def __init__(self, script: Iterable[Any] = (), *, delay: float = 0.0, default: str = "ok") -> None:
        self.script = list(script)
        self.calls: list[Call] = []
        self.delay = delay
        self.default = default
        self.config: dict[str, Any] = {"model_id": "fake"}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> Any:
        return self.config

    async def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None,
                                **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.calls.append(Call(copy.deepcopy(list(messages)), copy.deepcopy(list(tool_specs or [])),
                               copy.deepcopy(system_prompt_content)))
        entry = self.script.pop(0) if self.script else text_msg(self.default)
        if isinstance(entry, BaseException):
            raise entry
        for chunk in entry:
            if self.delay:
                await asyncio.sleep(self.delay)
            else:
                await asyncio.sleep(0)
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


# --------------------------------------------------------------------------- a real BedrockModel, faked boto


class _Meta:
    region_name = "us-east-1"


class FakeBedrockRuntime:
    """``converse_stream(**request)``: records the request, then returns the next scripted stream
    or raises the scripted ``ClientError``. Put it under a real ``BedrockModel`` (``model.client``)."""

    meta = _Meta()

    def __init__(self, script: Iterable[Any] = (), default: str = "ok") -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.default = default

    def converse_stream(self, **request: Any) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(request))
        entry = self.script.pop(0) if self.script else text_msg(self.default)
        if isinstance(entry, BaseException):
            raise entry
        return {"stream": iter(entry)}


def client_error(code: str, message: str = "diagnostic", op: str = "ConverseStream") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, op)


# --------------------------------------------------------------------------- Gateway


@dataclass
class GatewayRuns:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def gateway_tool(name: str, runs: GatewayRuns | None = None, *, delay: float = 0.0, ticks: int = 2,
                 answer: str = "gateway answer") -> Any:
    """A plain Strands tool standing in for an MCP Gateway tool (runs inside the loop). It yields
    to the event loop ``ticks`` times (so a test can cancel it mid-run), then sleeps ``delay``."""

    @tool(name=name, description=f"fake gateway tool {name}",
          inputSchema={"json": {"type": "object", "properties": {"q": {"type": "string"}}}}, context=True)
    async def _gw(tool_context: ToolContext) -> str:
        if runs is not None:
            runs.calls.append((name, dict(tool_context.tool_use.get("input") or {})))
        for _ in range(ticks):
            await asyncio.sleep(0)
        if delay:
            await asyncio.sleep(delay)
        return answer

    return _gw


class FakeGatewayProvider:
    """Lists the given tools, or raises when ``fail`` is set. ``closed`` records ``close()``."""

    def __init__(self, tools: Sequence[Any] = (), *, fail: bool = False) -> None:
        self._tools = list(tools)
        self.fail = fail
        self.closed = False

    def tools(self) -> Sequence[Any]:
        if self.fail:
            raise ConnectionError("gateway down")
        return list(self._tools)

    def close(self) -> None:
        self.closed = True


def default_gateway(runs: GatewayRuns | None = None, **kw: Any) -> FakeGatewayProvider:
    return FakeGatewayProvider([gateway_tool("level0___faq_lookup", runs, **kw),
                                gateway_tool("level0___get_branch_hours", runs, **kw)])


# --------------------------------------------------------------------------- client helpers


#: The keyword arguments :func:`settings` puts into the (single) model entry.
_ENTRY_KEYS = {"model_id": "id", "thinking": "thinking", "max_tokens": "max_tokens",
               "text_release": "text_release", "prompt_caching": "prompt_caching"}


def model_entry(**kw: Any) -> ModelEntry:
    """A :class:`ModelEntry` with the rig's defaults (``model_id`` and the other single-model names
    are accepted as before the model list)."""
    base: dict[str, Any] = {"id": "us.anthropic.claude-haiku-5-5", "max_tokens": 1000, "thinking": "default",
                            "prompt_caching": False}
    base.update({_ENTRY_KEYS.get(k, k): v for k, v in kw.items()})
    if base["thinking"] == "off":  # stream mode needs reasoning (ModelEntry): off goes with message
        base.setdefault("text_release", "message")
    return ModelEntry(**base)


def settings(**kw: Any) -> StrandsSettings:
    """Settings with one model entry (today's single-model path); the entry's fields may be given
    by their single-model names (``model_id``, ``thinking``, ...). ``models=`` gives the list."""
    entry_kw = {k: kw.pop(k) for k in list(kw) if k in _ENTRY_KEYS}
    base: dict[str, Any] = {"region": "us-east-1"}
    if "models" not in kw:
        base["models"] = (model_entry(**entry_kw),)
    base.update(kw)
    return StrandsSettings(**base)


def make_client(model: Model, *, gateway: Any = None, **kw: Any) -> StrandsAgentClient:
    return StrandsAgentClient(settings(**kw), model_factory=lambda _s, _sid: model, gateway_tools=gateway)


async def collect(gen: Any, *, message_ends: bool = False) -> list[Any]:
    """The stream's events; without the :class:`MessageEnd` boundaries unless ``message_ends``
    (their own tests check them)."""
    out = []
    async for ev in gen:
        if message_ends or not isinstance(ev, MessageEnd):
            out.append(ev)
    return out


def history_shape(messages: Sequence[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    return [(m["role"], [next(iter(c)) for c in m["content"]]) for m in messages]


# --------------------------------------------------------------------------- AgentCore Memory


def _matches(event: dict[str, Any], expressions: Sequence[dict[str, Any]]) -> bool:
    meta = event.get("metadata") or {}
    for expr in expressions:
        key = expr["left"]["metadataKey"]
        op = expr["operator"]
        if op == "EQUALS_TO":
            if meta.get(key) != expr["right"]["metadataValue"]:
                return False
        elif op == "EXISTS":
            if key not in meta:
                return False
        elif op == "NOT_EXISTS":
            if key in meta:
                return False
        else:  # pragma: no cover - the session manager uses only EQUALS_TO
            raise ValueError(op)
    return True


Failure = BaseException | Callable[[dict[str, Any]], BaseException | None]


class FakeMemoryData:
    """The AgentCore Memory data plane, in memory. Events keep the payload exactly as written (a
    batched flush is one event with many payload items), ``list_events`` returns them newest first
    (by ``eventTimestamp``, ties in write order) with ``nextToken`` pages of at most ``maxResults``.

    ``fail[op]`` (``create_event``, ``list_events``): an exception to raise, or a callable that gets
    the call's keyword arguments and returns one (or ``None`` to let the call through).
    ``ops`` records ``(operation, sessionId)``; ``on_op`` (if set) is called before each."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.ops: list[tuple[str, str]] = []
        self.fail: dict[str, Failure] = {}
        self.on_op: Callable[[str, dict[str, Any]], None] | None = None
        self._seq = itertools.count()

    def _enter(self, op: str, name: str, kw: dict[str, Any]) -> None:
        self.ops.append((name, kw.get("sessionId", "")))
        if self.on_op is not None:
            self.on_op(name, kw)
        f = self.fail.get(op)
        if f is None:
            return
        exc = f if isinstance(f, BaseException) else f(kw)
        if exc is not None:
            raise exc

    @staticmethod
    def _public(ev: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy({k: v for k, v in ev.items() if not k.startswith("_")})

    def create_event(self, **kw: Any) -> dict[str, Any]:
        self._enter("create_event", "CreateEvent", kw)
        ts = kw["eventTimestamp"]
        if not isinstance(ts, datetime):  # pragma: no cover - the session manager always sends one
            raise TypeError("eventTimestamp must be a datetime")
        seq = next(self._seq)
        ev: dict[str, Any] = {
            "memoryId": kw["memoryId"], "actorId": kw["actorId"], "sessionId": kw["sessionId"],
            "eventId": f"{int(ts.timestamp() * 1000):013d}#{seq:08x}", "eventTimestamp": ts,
            "payload": copy.deepcopy(list(kw["payload"])), "branch": {"name": "main"}, "_seq": seq}
        if kw.get("metadata"):
            ev["metadata"] = copy.deepcopy(dict(kw["metadata"]))
        self.events.append(ev)
        return {"event": self._public(ev)}

    def _session(self, memory_id: str, actor_id: str, session_id: str) -> list[dict[str, Any]]:
        evs = [e for e in self.events
               if (e["memoryId"], e["actorId"], e["sessionId"]) == (memory_id, actor_id, session_id)]
        return sorted(evs, key=lambda e: (e["eventTimestamp"], e["_seq"]), reverse=True)

    def list_events(self, **kw: Any) -> dict[str, Any]:
        self._enter("list_events", "ListEvents", kw)
        evs = self._session(kw["memoryId"], kw["actorId"], kw["sessionId"])
        expressions = ((kw.get("filter") or {}).get("eventMetadata")) or []
        evs = [e for e in evs if _matches(e, expressions)]
        size = min(int(kw.get("maxResults", 20)), 100)
        start = int(kw.get("nextToken") or 0)
        page = evs[start:start + size]
        out: dict[str, Any] = {"events": [self._public(e) for e in page]}
        if not kw.get("includePayloads", True):
            for e in out["events"]:
                e.pop("payload", None)
        if start + size < len(evs):
            out["nextToken"] = str(start + size)
        return out

    def get_event(self, **kw: Any) -> dict[str, Any]:
        self._enter("get_event", "GetEvent", kw)
        for e in self.events:
            if e["eventId"] == kw["eventId"] and e["sessionId"] == kw["sessionId"]:
                return {"event": self._public(e)}
        raise client_error("ResourceNotFoundException", op="GetEvent")

    def delete_event(self, **kw: Any) -> dict[str, Any]:
        self._enter("delete_event", "DeleteEvent", kw)
        self.events = [e for e in self.events if e["eventId"] != kw["eventId"]]
        return {"eventId": kw["eventId"]}

    # -- inspection helpers (not part of the boto3 API)

    def count(self, name: str, session_id: str | None = None) -> int:
        return sum(1 for op, sid in self.ops if op == name and (session_id is None or sid == session_id))

    def stored_messages(self, session_id: str, *, memory_id: str = "mem-1",
                        actor_id: str = MEMORY_ACTOR_ID) -> list[dict[str, Any]]:
        """The session's messages as the session manager would restore them (role and content)."""
        from bedrock_agentcore.memory.integrations.strands.bedrock_converter import AgentCoreMemoryConverter

        evs = [self._public(e) for e in self._session(memory_id, actor_id, session_id)]
        return [content_of(m.to_message()) for m in AgentCoreMemoryConverter.events_to_messages(evs)]

    def newest_agent_state(self, session_id: str, *, memory_id: str = "mem-1",
                           actor_id: str = MEMORY_ACTOR_ID) -> dict[str, Any] | None:
        """The newest agent-state snapshot stored (the last blob of the newest AGENT event)."""
        for e in self._session(memory_id, actor_id, session_id):
            if (e.get("metadata") or {}).get("stateType") == {"stringValue": "AGENT"}:
                return json.loads(e["payload"][-1]["blob"])
        return None


def content_of(message: dict[str, Any]) -> dict[str, Any]:
    """A message's role and content only (Strands adds bookkeeping keys such as ``tracking_id``)."""
    return {"role": message["role"], "content": copy.deepcopy(message["content"])}


class _NullClient:
    """The control-plane client the session manager creates but never calls here."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected control-plane call {name}")


class FakeBotoSession:
    """What the session manager takes as ``boto_session``: ``client()`` gives the fake data plane."""

    region_name = "us-east-1"

    def __init__(self, data: FakeMemoryData) -> None:
        self.data = data

    def client(self, service: str, region_name: str | None = None, config: Any = None) -> Any:
        return self.data if service == "bedrock-agentcore" else _NullClient()


def light_memory_client(monkeypatch: Any) -> None:
    """Skip the boto3 clients the library's ``MemoryClient`` builds in ``__init__`` (about 80 ms each
    time; the session manager replaces them with ``boto_session``'s at once). Its methods (the
    paginating ``list_events``) stay the real ones."""
    from bedrock_agentcore.memory.client import MemoryClient
    from bedrock_agentcore.memory.integrations.strands import session_manager as sm

    class _Light(MemoryClient):
        def __init__(self, region_name: str | None = None, **_: Any) -> None:
            self.region_name = region_name or "us-east-1"
            self.integration_source = None

    monkeypatch.setattr(sm, "MemoryClient", _Light)


@dataclass
class Builds:
    """What a recording factory built: ``(session_id, mode)`` and the managers, in order."""

    made: list[tuple[str, str]] = field(default_factory=list)
    managers: list[Any] = field(default_factory=list)


def memory_factory(data: FakeMemoryData, builds: Builds | None = None, *, memory_id: str = "mem-1",
                   cls: Any = None) -> Callable[[str, str], Any]:
    """The client's real Memory session-manager factory on the fake data plane (``cls``: another
    manager class taking the same arguments, e.g. the unpatched library class)."""
    real = memory_session_manager_factory(memory_id, "us-east-1", boto_session=FakeBotoSession(data))

    def factory(session_id: str, mode: str) -> Any:
        if cls is None:
            manager = real(session_id, mode)  # type: ignore[arg-type]
        else:
            from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig

            config = AgentCoreMemoryConfig(memory_id=memory_id, session_id=session_id, actor_id=MEMORY_ACTOR_ID,
                                           batch_size=100, async_mode=True)
            manager = cls(config, region_name="us-east-1", boto_session=FakeBotoSession(data))
        if builds is not None:
            builds.made.append((session_id, mode))
            builds.managers.append(manager)
        return manager

    return factory


def make_memory_client(model: Model, data: FakeMemoryData, *, builds: Builds | None = None,
                       gateway: Any = None, cls: Any = None, **kw: Any) -> StrandsAgentClient:
    """A client with AgentCore Memory history on the fake data plane."""
    kw.setdefault("memory_id", "mem-1")
    return StrandsAgentClient(settings(**kw), model_factory=lambda _s, _sid: model, gateway_tools=gateway,
                              session_manager_factory=memory_factory(data, builds, memory_id=kw["memory_id"],
                                                                     cls=cls))


# --------------------------------------------------------------------------- call observability


@dataclass
class Traced:
    """The bridge's telemetry installed against an in-memory exporter (:func:`traced`)."""

    handle: Any
    exporter: Any
    #: The ``redacted`` mode's redactor passed to :func:`traced` (``None`` otherwise).
    redact: Any = None

    @property
    def tracer(self) -> Any:
        return self.handle.tracer

    def spans(self) -> list[Any]:
        """Every span exported so far (after the content filter), flushed first."""
        self.handle.tracer_provider.force_flush(5000)
        return list(self.exporter.get_finished_spans())

    def enable(self, bridge: Any) -> None:
        """``bridge.enable_tracing`` as ``build_app_from_env`` calls it: with a
        ``TranscriptRedactor``, its ``retain`` and the session end's flush."""
        from owlbank.telemetry.bridge import TranscriptRedactor

        provider = self.handle.tracer_provider
        redactor = self.redact if isinstance(self.redact, TranscriptRedactor) else None
        bridge.enable_tracing(self.tracer,
                              flush=(lambda: provider.force_flush(3000)) if redactor is not None else None,
                              retain=redactor.retain if redactor is not None else None)

    def clear(self) -> None:
        self.handle.tracer_provider.force_flush(5000)
        self.exporter.clear()


@contextlib.contextmanager
def traced(content: str = "off", *, redact: Any = None, botocore: bool = False, sample_rate: float = 1.0,
           exporter: Any = None) -> Iterator[Traced]:
    """``install_bridge`` as the bridge runs it (baggage, evaluation-shape and batch processors over
    ``ExportHealth(ContentFilterExporter(...))``), with the SigV4 exporter replaced by ``exporter``
    (default an ``InMemorySpanExporter``) and no global provider set (OpenTelemetry's globals are
    set-once). Strands' tracer singleton is pointed at the provider for the duration, so Strands'
    spans go through it too; ``botocore`` instruments botocore against it (uninstrumented after)."""
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from strands.telemetry.tracer import get_tracer

    from owlbank.telemetry import bridge as tb
    from owlbank.telemetry.settings import TelemetrySettings

    sink = exporter if exporter is not None else InMemorySpanExporter()
    settings_ = TelemetrySettings(stage="dev", service_name="owlbank-dev-bridge", hosting="fargate",
                                  sample_rate=sample_rate, content=content,  # type: ignore[arg-type]
                                  traces_log_group="/owlbank/traces/bridge", region="us-east-1")
    real = tb.SigV4OtlpSpanExporter
    tb.SigV4OtlpSpanExporter = lambda region, log_group, **_: sink  # type: ignore[assignment,misc]
    try:
        handle = tb.install_bridge(settings_, redact=redact, environ={},
                                   set_tracer_provider=lambda _p: None, set_meter_provider=lambda _p: None,
                                   set_propagator=lambda _p: None, instrument_botocore=botocore)
    finally:
        tb.SigV4OtlpSpanExporter = real  # type: ignore[misc]
    assert handle is not None, "install_bridge refused (an OTEL_EXPORTER_OTLP_*ENDPOINT in the environment?)"
    strands_tracer = get_tracer()
    saved = (strands_tracer.tracer_provider, strands_tracer.tracer)
    strands_tracer.tracer_provider = handle.tracer_provider
    strands_tracer.tracer = handle.tracer_provider.get_tracer(strands_tracer.service_name)
    try:
        yield Traced(handle, sink, redact if content == "redacted" else None)
    finally:
        strands_tracer.tracer_provider, strands_tracer.tracer = saved
        handle.shutdown()


def span_tree(spans: Sequence[Any]) -> dict[int | None, list[Any]]:
    """Exported spans by parent span ID (``None``: roots)."""
    out: dict[int | None, list[Any]] = {}
    for s in sorted(spans, key=lambda s: s.start_time or 0):
        out.setdefault(s.parent.span_id if s.parent is not None else None, []).append(s)
    return out


def descendants(spans: Sequence[Any], root: Any) -> list[Any]:
    """Every exported span under ``root`` (same trace, reachable through parents)."""
    tree = span_tree(spans)
    out: list[Any] = []
    todo = [root]
    while todo:
        node = todo.pop()
        kids = tree.get(node.context.span_id, [])
        out.extend(kids)
        todo.extend(kids)
    return out
