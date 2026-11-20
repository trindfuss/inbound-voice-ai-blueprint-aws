"""Whole calls through the front door and the bridge for model-assisted slots.

:class:`SplitWorld` is the e2e :class:`~tests.e2e.harness.World` (the real front door, auth service,
bank tools and bridge on TAC, over one store and one fake KMS) with the front door and the bridge
run as **two processes**: each has its own sensitive registry (``segments`` registered values and
``sensitive`` process state), as a Lambda instance and the bridge task share only the calls table.
Every front-door webhook is synchronous, so the front door's registry is swapped in for exactly the
duration of each webhook and the bridge's is active the rest of the time (the event loop the
bridge runs on). The bridge therefore knows a sensitive value only from call state's
``sealed_sensitive`` (``sensitive.restore``), never from the front door's memory.

:class:`LogCapture` keeps every log line as the JSON handler writes it: through
``obs.ScrubFilter`` (the exact-value and pattern scrubs) and ``obs.ContextFilter``. Metric
dimensions come from the same lines (the alarm metric filters match logged event names), so the log
scan covers them.

:func:`surfaces` collects every place a sensitive value could reach in a call: log lines, journal
events, prompts and TwiML, relay frames, call state, task attributes, the Context API timeline
(``context_api.build_timeline`` over the journal) and everything the agent was given.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lambdas.admin_api.app import AdminApi
from lambdas.admin_api.app import Deps as AdminDeps
from lambdas.context_api.handler import build_timeline
from owlbank import obs, segments, sensitive
from scripts.hash_tool import hash_factor
from tests.admin_api_fakes import POOL_ID, FakeCognito
from tests.admin_api_fakes import Clock as AdminClock
from tac_bridge.clock import local_now_line, zone
from tests.e2e.harness import T0, World, say, tool
from tests.leakscan import compile_values, leaks

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

#: Call-state attributes that hold sealed blobs, hashes, ids or times, not caller text.
_STATE_SKIP = frozenset({"sealed_token", "sealed_sensitive", "ws_nonce_hash", "agent_owner",
                         "expires_at", "version", "call_sid", "claimed_at", "ended_at", "held",
                         "token_hash", "tool_use_id", "seq", "ts", "salt", "factors"})
_STD = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}


def policy_doc(**changes: Any) -> dict[str, Any]:
    """``data/policy.json`` with ``a__b__c=value`` set (``None`` deletes the key)."""
    doc = json.loads((DATA / "policy.json").read_text())
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = doc
        for p in parents:
            node = node[p]
        if value is None:
            node.pop(leaf, None)
        else:
            node[leaf] = value
    return doc


# --------------------------------------------------------------------------- processes


class Process:
    """One process's sensitive registry (memory only)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.registered: dict[str, set[str]] = {}
        self.state = sensitive._ProcessState()

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        saved = segments._SENSITIVE, sensitive._state
        segments._SENSITIVE, sensitive._state = self.registered, self.state
        try:
            yield
        finally:
            segments._SENSITIVE, sensitive._state = saved

    def values(self, call_sid: str) -> frozenset[str]:
        with self.active():
            return segments.sensitive_values(call_sid)


class SplitWorld:
    """A :class:`World` whose front door and bridge are separate processes (module docstring)."""

    def __init__(self, **world_kw: Any) -> None:
        self.world = World(**world_kw)
        self.fd = Process("frontdoor")
        self.bridge = Process("bridge")
        self._saved = segments._SENSITIVE, sensitive._state
        segments._SENSITIVE, sensitive._state = self.bridge.registered, self.bridge.state
        inner = self.world.frontdoor_handler

        def handler(event: dict[str, Any], context: Any = None) -> Any:
            with self.fd.active():
                return inner(event, context)

        self.world.frontdoor_handler = handler

    def close(self) -> None:
        for proc in (self.fd, self.bridge):
            with proc.active():
                self.world.cleanup()
                for sid in list(proc.state.seen):
                    sensitive.forget(sid)
        segments._SENSITIVE, sensitive._state = self._saved

    # -- set-up helpers

    def set_factor(self, customer_id: str, name: str, value: str) -> None:
        """Make ``value`` the customer's ``name`` factor (its salted hash in the record)."""
        item = self.world.customer(customer_id)
        item["factors"] = {**item.get("factors", {}), name: hash_factor(value)}
        self.world.store.put("customers", item)

    def respond(self, responder: Callable[[dict[str, Any]], list[Any]]) -> None:
        """Answer every agent input with ``responder(rec)`` instead of a queued script."""
        agent = self.world.agent
        agent._next_step = lambda rec: responder  # type: ignore[method-assign]

    def admin(self, *, accept_spoken_factors: bool = True) -> AdminApi:
        """The admin API (``GET /slots``, ``PUT /slots/{name}``) over this world's policy table."""
        clock = AdminClock()
        return AdminApi(AdminDeps(store=self.world.store, cognito=FakeCognito(), user_pool_id=POOL_ID,
                                  retention_days=90, clock=clock, suffix=clock.suffix,
                                  accept_spoken_factors=accept_spoken_factors))


def echo_agent(*, escalate_on: str | None = None,
               blurt: Callable[[], str | None] | None = None) -> Callable[[dict[str, Any]], list[Any]]:
    """A fake agent that echoes the caller's text back (and into its hand-off summary).

    ``balance`` in the caller's text: ``get_balance`` (level 1, read by the front door); ``transactions``: ``list_recent_transactions`` (an inline tool the bridge runs); ``pay``:
    ``make_payment`` (level 3: SSN last four, then the code, then
    the read-back); ``address``: ``change_address`` (level 2; on ``AUTH_LEVEL_INSUFFICIENT`` it calls ``authenticate`` for it);
    ``escalate_on`` in the text:
    ``escalate`` with the previous caller text as the summary. A tool result alone: silent.

    ``blurt``: what a misbehaving agent adds to every reply and summary (a value it must never
    repeat, e.g. one it read from a tool result), or ``None`` for nothing."""
    heard: list[str] = []

    def extra() -> str:
        said = blurt() if blurt is not None else None
        return f" Your last four are {said}." if said else ""

    def respond(rec: dict[str, Any]) -> list[Any]:
        text = rec.get("user_text")
        if not text:
            for res in rec.get("tool_results") or ():
                content = res.content if isinstance(res.content, Mapping) else {}
                if content.get("error") == "AUTH_LEVEL_INSUFFICIENT":
                    return tool("authenticate", "tu_auth", tool=content.get("tool"))
            return say("<silent/>")
        if escalate_on and escalate_on in text:
            summary = f"Caller said: {heard[-1] if heard else text}.{extra()}"
            return tool("escalate", "tu_esc", reason="caller_requested_human", summary=summary)
        heard.append(text)
        if "balance" in text:
            return tool("get_balance", "tu_bal", account="checking")
        if "transactions" in text:
            return tool("list_recent_transactions", "tu_txn", account="checking")
        if "pay" in text:
            return tool("make_payment", "tu_pay", from_account="checking", to="red",
                        amount="50.00")
        if "address" in text:
            return tool("change_address", "tu_addr", street="12 Example Lane", city="Springfield",
                        state="CA", postal_code="60601")
        return say(f"You said: {text}{extra()}")

    return respond


# --------------------------------------------------------------------------- logs


class LogCapture(logging.Handler):
    """Every log line as the JSON handler writes it: scrubbed (``ScrubFilter``) and correlated
    (``ContextFilter``). Each line is kept as a JSON object of its message and every field."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.addFilter(obs.ContextFilter())
        self.addFilter(obs.ScrubFilter())
        self.lines: list[str] = []
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        fields = {k: v for k, v in record.__dict__.items() if k not in _STD and not k.startswith("_")}
        self.records.append({"name": record.name, "message": record.getMessage(), **fields})
        # One JSON object per line, as the JSON handler writes it: field values stay apart (joined
        # with plain spaces, "2505.3", "0", "1", "1" would read as a spaced-out "3011").
        self.lines.append(json.dumps({"message": record.getMessage(), **fields}, default=str,
                                     ensure_ascii=False))

    def events(self, name: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("event") == name]


@contextlib.contextmanager
def capture_logs(level: int = logging.DEBUG) -> Iterator[LogCapture]:
    root = logging.getLogger()
    cap = LogCapture()
    old = root.level
    root.setLevel(level)
    root.addHandler(cap)
    try:
        yield cap
    finally:
        root.removeHandler(cap)
        root.setLevel(old)


# --------------------------------------------------------------------------- surfaces


def _strings(value: Any, skip: frozenset[str] = frozenset()) -> Iterator[str]:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if str(k) not in skip:
                yield from _strings(v, skip)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _strings(v, skip)
    elif value is not None:
        yield str(value)


def surfaces(sw: SplitWorld, logs: LogCapture, sids: Iterable[str]) -> dict[str, list[str]]:
    """Every surface of the calls ``sids`` (module docstring)."""
    w = sw.world
    journal: list[str] = []
    timeline: list[str] = []
    state: list[str] = []
    for sid in sids:
        events = w.journal(sid)
        journal.extend(json.dumps(e, ensure_ascii=False) for e in events)
        timeline.extend(json.dumps(e, ensure_ascii=False) for e in build_timeline(events))
        item = w.store.get("calls", {"call_sid": sid}) or {}
        state.extend(_strings(item, _STATE_SKIP))
    return {
        "log": [_without_measurements(line) for line in logs.lines],
        "journal": journal,
        "timeline": timeline,
        "twiml (prompts)": list(w.twiml),
        "relay frames": [json.dumps(f, ensure_ascii=False) for s in w.relays for f in s.socket.sent],
        "call state": state,
        "task attributes": [json.dumps(t, ensure_ascii=False) for t in w.task_attributes],
        "agent input": [w.agent.inputs_text()],
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_measurement(key: str, value: Any) -> bool:
    """A size or a timing the code computes (``event_bytes``, ``duration_ms``, or a map of them such
    as ``factor_answer``'s ``timings_ms``), never caller content: a generated value equal to one is a
    collision, not a leak (``event_bytes`` 1137 against an SSN4 of 1137; a 123 ms timing under
    load against a 3-digit ZIP of 123)."""
    if not (key.endswith(("_bytes", "_ms")) or key == "bytes"):
        return False
    if isinstance(value, Mapping):
        return bool(value) and all(_is_number(v) for v in value.values())
    return _is_number(value)


def _strip_measurements(value: Any) -> Any:
    """``value`` without its numeric size and timing fields, at any depth."""
    if isinstance(value, dict):
        return {k: _strip_measurements(v) for k, v in value.items() if not _is_measurement(str(k), v)}
    if isinstance(value, list):
        return [_strip_measurements(v) for v in value]
    return value


def _without_measurements(line: str) -> str:
    """A JSON log line without its numeric size and timing fields (nested ones too); any other line
    unchanged."""
    try:
        record = json.loads(line)
    except ValueError:
        return line
    if not isinstance(record, dict):
        return line
    return json.dumps(_strip_measurements(record), ensure_ascii=False)


def leaked(values: Iterable[str], found: Mapping[str, list[str]]) -> list[str]:
    """Where any of ``values`` appears as a whole value (case, spaces and hyphens tolerated,
    ``tests.leakscan``)."""
    patterns = compile_values(values)
    return [where for where, texts in found.items() if any(leaks(patterns, t) for t in texts)]


def leak_report(values: Iterable[str], found: Mapping[str, list[str]]) -> list[tuple[str, str]]:
    """``(surface, text)`` of each text holding one of ``values`` (for a failure message)."""
    patterns = compile_values(values)
    return [(where, t[:600]) for where, texts in found.items() for t in texts if leaks(patterns, t)]


#: Text the fixtures, catalog, policy and the world's clock line put on the surfaces themselves
#: (balances, account endings, amounts, the year): a generated value found in it would be a
#: collision, not a leak. One generated run drew a wrong answer 2025 and found the clock line's year.
FIXTURE_TEXT = "\n".join(
    [(DATA / n).read_text() for n in ("fixtures.json", "prompt_catalog.json", "policy.json")]
    + [local_now_line(datetime.fromtimestamp(T0, UTC), zone(None), lang) for lang in ("en", "es")])


def collides(value: str) -> bool:
    return leaks(compile_values([value]), FIXTURE_TEXT)


def words_for(digits: str) -> str:
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return " ".join(names[int(c)] for c in digits)


def answering_turns(monkeypatch: Any) -> list[Any]:
    """Every :class:`~owlbank.slots.Resolution` the front door's ``resolve`` returns, in order (the
    answering turn's ``utterance_for_ai``)."""
    import lambdas.frontdoor.slotflow as fd_slotflow

    results: list[Any] = []
    real = fd_slotflow.resolve

    def spy(*a: Any, **k: Any) -> Any:
        out = real(*a, **k)
        results.append(out)
        return out

    monkeypatch.setattr(fd_slotflow, "resolve", spy)
    return results


def without_slot_items(sw: SplitWorld, *names: str) -> None:
    """Remove ``slot#<name>`` items and their ``meta.slots`` entries from the policy table (a table
    seeded before the slot items existed)."""
    store = sw.world.store
    for name in names:
        store.delete("policy", {"pk": f"slot#{name}"})
    meta = store.get("policy", {"pk": "meta"})
    if meta is not None and isinstance(meta.get("slots"), list):
        meta = copy.deepcopy(meta)
        meta["slots"] = [n for n in meta["slots"] if n not in names]
        store.put("policy", meta)


def gather_attrs(xml: str) -> list[dict[str, str]]:
    import xml.etree.ElementTree as ET

    return [dict(g.attrib) for g in ET.fromstring(xml.encode()).iter("Gather")]


def verb_tags(xml: str) -> list[str]:
    """The top-level verbs of a TwiML response, in order."""
    import xml.etree.ElementTree as ET

    return [c.tag for c in ET.fromstring(xml.encode())]


__all__ = ["LogCapture", "Process", "SplitWorld", "answering_turns",
           "capture_logs", "collides", "echo_agent", "gather_attrs", "leak_report", "leaked", "policy_doc",
           "surfaces", "verb_tags", "without_slot_items", "words_for"]
