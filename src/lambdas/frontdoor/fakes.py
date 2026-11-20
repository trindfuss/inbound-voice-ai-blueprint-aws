"""Test doubles for the front door (never used by the production wiring in ``handler.py``).

:class:`FakeLambdaClient` stands in for a boto3 Lambda client behind
:class:`~lambdas.frontdoor.backend.LambdaBackendClient`: each function name routes
to an in-process callable (for example the auth service's ``service.handle_event`` over a
``MemoryStore``), so tests exercise the real wire shapes end to end. Failures can
be injected per operation. Recorded calls keep only the operation and function,
never the payload (it carries factor values and tokens).

Asynchronous invokes (``InvocationType="Event"``, ``send_otp_async`` and ``prepare_welcome_async``, listed by :meth:`FakeLambdaClient.welcome_requests`) answer 202 with an
empty payload, like Lambda. By default the function runs at once, inside ``invoke`` (so its side
effects, e.g. the auth service writing ``otp_send``, are visible straight away); with
``defer_events`` it is queued and runs only from :meth:`FakeLambdaClient.run_deferred`, at or
after ``event_delay_s`` past the invoke on ``clock`` (to simulate a slow send). An ``error_ops``
op invoked asynchronously is accepted and then "crashes": it never runs.

Like a botocore client, it has ``meta.events``: handlers registered for
``before-sign.lambda.Invoke`` run on each invoke with a request whose ``headers`` they may set
(call observability's ``X-Amzn-Trace-Id``); the ``X-Amzn-Trace-Id`` each invoke carried (or
``None``) is kept in ``trace_headers``, parallel to ``calls``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

BEFORE_SIGN = "before-sign.lambda.Invoke"


class FakeEvents:
    """The part of botocore's event system a client's ``meta.events`` offers that we use."""

    def __init__(self) -> None:
        self.handlers: dict[str, dict[str, Callable[..., Any]]] = {}

    def register(self, event_name: str, handler: Callable[..., Any], unique_id: str | None = None) -> None:
        self.handlers.setdefault(event_name, {})[unique_id or str(id(handler))] = handler

    def emit(self, event_name: str, **kwargs: Any) -> None:
        for handler in list(self.handlers.get(event_name, {}).values()):
            handler(**kwargs)


@dataclass
class FakeLambdaClient:
    functions: Mapping[str, Callable[[dict[str, Any]], Any]]
    fail_ops: set[str] = field(default_factory=set)      # raise, like a timeout
    error_ops: set[str] = field(default_factory=set)     # FunctionError, like an unhandled crash
    overrides: dict[str, Any] = field(default_factory=dict)  # op -> canned response payload
    calls: list[tuple[str, str]] = field(default_factory=list)  # (function, op)
    payloads: list[dict[str, Any]] = field(default_factory=list, repr=False)
    fail_event_ops: set[str] = field(default_factory=set)  # raise on an Event invoke only
    invocation_types: list[str] = field(default_factory=list)  # parallel to ``calls``
    defer_events: bool = False  # queue Event invokes until run_deferred()
    event_delay_s: float = 0.0  # with defer_events: earliest run, seconds after the invoke on clock
    clock: Callable[[], float] | None = None
    deferred: list[tuple[float, str, dict[str, Any]]] = field(default_factory=list, repr=False)
    trace_headers: list[str | None] = field(default_factory=list)  # parallel to ``calls``
    meta: Any = field(default_factory=lambda: SimpleNamespace(events=FakeEvents()), repr=False)

    def invoke(self, *, FunctionName: str, InvocationType: str, Payload: bytes) -> dict[str, Any]:
        event = json.loads(Payload)
        op = str(event.get("op"))
        request = SimpleNamespace(headers={})
        self.meta.events.emit(BEFORE_SIGN, request=request)
        self.trace_headers.append(request.headers.get("X-Amzn-Trace-Id"))
        self.calls.append((FunctionName, op))
        self.invocation_types.append(InvocationType)
        self.payloads.append(event)
        if op in self.fail_ops:
            raise TimeoutError("simulated timeout")
        if InvocationType == "Event":
            if op in self.fail_event_ops:
                raise TimeoutError("simulated async invoke failure")
            if op not in self.error_ops:
                if self.defer_events:
                    now = self.clock() if self.clock is not None else 0.0
                    self.deferred.append((now + self.event_delay_s, FunctionName, event))
                else:
                    self.functions[FunctionName](event)
            return {"StatusCode": 202, "Payload": io.BytesIO(b"")}
        if op in self.error_ops:
            return {"StatusCode": 200, "FunctionError": "Unhandled",
                    "Payload": io.BytesIO(b'{"errorMessage": "boom"}')}
        result = self.overrides[op] if op in self.overrides else self.functions[FunctionName](event)
        return {"StatusCode": 200, "Payload": io.BytesIO(json.dumps(result).encode())}

    def ops(self) -> list[str]:
        return [op for _, op in self.calls]

    def event_ops(self) -> list[str]:
        """Operations invoked asynchronously (``InvocationType="Event"``)."""
        return [op for (_, op), kind in zip(self.calls, self.invocation_types, strict=True)
                if kind == "Event"]

    def welcome_requests(self) -> list[dict[str, Any]]:
        """The asynchronous ``prepare_welcome`` invokes, in order: each
        payload's ``call_sid``, ``customer_id`` and ``language`` (no factor value or token)."""
        return [{k: event.get(k) for k in ("call_sid", "customer_id", "language")}
                for (_, op), kind, event in zip(self.calls, self.invocation_types, self.payloads, strict=True)
                if op == "prepare_welcome" and kind == "Event"]

    def run_deferred(self, now: float | None = None) -> int:
        """Run the queued Event invokes due at ``now`` (all of them when ``None``); returns how many ran."""
        due = [d for d in self.deferred if now is None or d[0] <= now]
        self.deferred = [d for d in self.deferred if d not in due]
        for _, fn, event in due:
            self.functions[fn](event)
        return len(due)



