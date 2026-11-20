"""The front door's view of the back-end Lambdas (``auth_service``, ``bank_tools``).

The front door never touches customer factor hashes or account data directly: it
calls the back end through :class:`BackendClient`. Production uses
:class:`LambdaBackendClient` (``lambda:InvokeFunction`` over IAM); tests use
:class:`lambdas.frontdoor.fakes.FakeBackend`.

Wire shapes (``auth_service`` / ``bank_tools`` handler docstrings):

* verify: ``{"op": "verify", "call_sid", "ani", "factors": {name: value}, "target_level",
  "stir_verstat", "identity_source"}`` (the call's ``StirVerstat``, or ``null``; the auth service
  applies the caller-ID trust gate; ``identity_source`` is ``"ani"`` or ``"keyed_phone"``,
  and a keyed number is never trusted)
  -> ``{"result": "verified", "level", "customer_id", "token", "factors_verified", "step_up"}``
  or ``{"result": "not_verified", "level", "reason": "mismatch"|"ambiguous"|"no_candidates",
  "attempts_remaining"}`` or ``{"result": "rate_limited"|"invalid_level"|"unavailable"|
  "bad_request", ...}``. Step-up is detected from the call's active token; send only
  the missing factors (plus any factor held for disambiguation).
  ``{"result": "escalate", "reason": "verification_risk"}`` maps to
  ``VerifyResult("escalate")``.
* precheck: ``{"op": "precheck", "call_sid", "ani", "target_level", "stir_verstat",
  "identity_source"}`` ->
  ``{"result": "ok"|"escalate"|...}``: the untrusted-caller gate before any factor is asked.
* send OTP: ``{"op": "send_otp", "call_sid", "ani", "language", "stir_verstat", "identity_source"}``
  -> ``{"result": "sent"|...}``.
  The code goes only to the calling ``ani`` when it is on file for the verified customer, in
  the call's ``language`` (Verify ``Locale``); otherwise ``{"result": "escalate", "reason":
  "otp_unavailable"}`` maps to ``"escalate"``. The same ``ani`` goes in the
  ``verify`` call that carries the code.
* send OTP asynchronously: the same event plus ``"record": true``, invoked with
  ``InvocationType="Event"`` (Lambda answers 202 at once). The auth service writes the outcome
  to the call state (``otp_send``), where the front door polls for it.
* prepare the welcome: ``{"op": "prepare_welcome", "call_sid",
  "customer_id", "language"}``, always an ``Event`` invoke, started once per call when the last
  login factor is asked for an identified customer. The auth service matches the customer's
  ``WebActivity`` traits against the policy's ``web_offers`` and writes the sealed offer ID to the
  call state (``welcome_prepared``); the front door never waits for it.
* bank tools: ``{"op", "token", "call_sid", "args"[, "idempotency_key"]}`` -> ``{"tool", "data"}``
  or ``{"error": CODE, "tool"}``. The front door runs ``get_balance`` and, after a caller's
  yes, a ``deterministic``-confirm tool with the model's ``tool_use_id`` as key. ``ani_lookup`` data: ``{"candidates": [...], "withheld"}``.

Factor values and tokens are never logged here; result dataclasses hide the token
from ``repr``.

Call observability: each invoke is a ``lambda.invoke <op>`` child span of the webhook span
with the function's short name, invocation type and outcome (never the payload, a token or the
result), and a botocore ``before-sign.lambda.Invoke`` handler sets the ``X-Amzn-Trace-Id`` header
of a sampled span (keeping botocore's ``Lineage=``; unsampled: botocore's header is left alone),
so the called function's X-Ray segment joins the webhook's trace. Both are no-ops when off.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from owlbank.obs import log_caught, timed_call
from owlbank.telemetry import frontdoor as telemetry

log = logging.getLogger(__name__)

LookupStatus = Literal["ok", "error"]
VerifyStatus = Literal["verified", "ambiguous", "not_verified", "locked", "escalate", "error"]
PrecheckStatus = Literal["ok", "escalate", "error"]
#: ``escalate`` = no code can go to the calling number (``otp_unavailable``).
OtpStatus = Literal["sent", "escalate", "error"]
ToolStatus = Literal["ok", "not_found", "auth_insufficient", "error"]

VERIFY_STATUSES: frozenset[str] = frozenset(
    {"verified", "ambiguous", "not_verified", "locked", "escalate", "error"}
)


@dataclass(frozen=True, slots=True)
class LookupResult:
    status: LookupStatus
    candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """Outcome of one verification call. ``token`` is the opaque call token (never logged)."""

    status: VerifyStatus
    level: int = 0
    customer_id: str | None = None
    token: str | None = field(default=None, repr=False)
    factors_verified: tuple[str, ...] = ()
    attempts_remaining: int | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    status: ToolStatus
    data: Mapping[str, Any] = field(default_factory=dict)
    code: str | None = None  # the bank's error code, if any (e.g. INSUFFICIENT_FUNDS)


class BackendClient(Protocol):
    """Everything the front door asks of the back end."""

    def ani_lookup(self, *, call_sid: str, ani: str) -> LookupResult:
        """Candidate customer IDs for a caller ID (not identity)."""
        ...

    def verify(self, *, call_sid: str, ani: str | None, factors: Mapping[str, str],
               target_level: int, stir_verstat: str | None = None,
               identity_source: str = "ani") -> VerifyResult:
        """Verify newly collected factors; the auth service combines them with what the
        call already verified, issues or upgrades the call token, and returns it.
        ``stir_verstat`` and ``identity_source`` are passed through for the
        auth service's trust gate."""
        ...

    def precheck(self, *, call_sid: str, ani: str | None, target_level: int,
                 stir_verstat: str | None = None, identity_source: str = "ani") -> PrecheckStatus:
        """The auth service's untrusted-caller gate before any factor is asked:
        ``escalate`` means transfer to a human with reason ``verification_risk``."""
        ...

    def send_otp(self, *, call_sid: str, ani: str | None, language: str | None = None,
                 stir_verstat: str | None = None, identity_source: str = "ani") -> OtpStatus:
        """Send a one-time passcode (Twilio Verify) to the calling number, in ``language``.
        ``escalate`` means no code was sent (ANI withheld or not on file for the customer, or
        Verify couldn't deliver): transfer to a human with reason ``otp_unavailable``."""
        ...

    def send_otp_async(self, *, call_sid: str, ani: str | None, language: str | None = None,
                       stir_verstat: str | None = None, identity_source: str = "ani") -> bool:
        """Start :meth:`send_otp` without waiting for it. The auth service writes the
        outcome to the call state's ``otp_send``. ``False`` when the request couldn't be queued
        (the caller then sends synchronously); ``True`` says nothing about delivery."""
        ...

    def prepare_welcome_async(self, *, call_sid: str, customer_id: str, language: str) -> bool:
        """Start the auth service's ``prepare_welcome`` without waiting for it.
        The outcome lands in the call state's ``welcome_prepared``. ``False`` when the request
        couldn't be queued (the call then gets today's welcome); ``True`` says nothing about it."""
        ...

    def get_balance(self, *, token: str, call_sid: str, account: str) -> ToolResult:
        """Balance for one account of the token's customer."""
        ...

    def run_tool(self, *, token: str, call_sid: str, tool: str, args: Mapping[str, Any],
                 idempotency_key: str | None) -> ToolResult:
        """Run a bank tool for the token's customer (a confirmed ``deterministic`` tool)."""
        ...


# --------------------------------------------------------------------------- parsing


def _ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for v in value:
        if isinstance(v, str) and v:
            out.append(v)
        elif isinstance(v, Mapping) and isinstance(v.get("customer_id"), str):
            out.append(v["customer_id"])
    return tuple(out)


def parse_lookup(payload: Any) -> LookupResult:
    if not isinstance(payload, Mapping) or payload.get("error"):
        return LookupResult("error")
    inner = payload.get("data")
    body = inner if isinstance(inner, Mapping) else payload
    for key in ("candidates", "candidate_ids", "customers"):
        if key in body:
            return LookupResult("ok", _ids(body[key]))
    return LookupResult("error")


def parse_verify(payload: Any) -> VerifyResult:
    if not isinstance(payload, Mapping):
        return VerifyResult("error")
    result = payload.get("result")
    level = payload.get("level", 0)
    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        level = 0
    remaining = payload.get("attempts_remaining")
    remaining = remaining if isinstance(remaining, int) and not isinstance(remaining, bool) else None
    if result == "verified":
        token, customer_id = payload.get("token"), payload.get("customer_id")
        if not (isinstance(token, str) and token and isinstance(customer_id, str)
                and customer_id and level >= 1):
            return VerifyResult("error")
        fv = payload.get("factors_verified")
        return VerifyResult("verified", level=level, customer_id=customer_id, token=token,
                            factors_verified=tuple(f for f in fv if isinstance(f, str))
                            if isinstance(fv, list) else ())
    if result == "not_verified":
        reason = payload.get("reason")
        if reason == "ambiguous":
            return VerifyResult("ambiguous", level=level, attempts_remaining=remaining)
        if reason == "no_candidates" or remaining == 0:
            return VerifyResult("locked", level=level, attempts_remaining=remaining)
        return VerifyResult("not_verified", level=level, attempts_remaining=remaining)
    if result == "rate_limited":
        return VerifyResult("locked", level=level)
    if result == "escalate":  # verification_risk; any reason: fail closed
        return VerifyResult("escalate", level=level)
    return VerifyResult("error", level=level)  # invalid_level, unavailable, bad_request, junk


def parse_precheck(payload: Any) -> PrecheckStatus:
    result = payload.get("result") if isinstance(payload, Mapping) else None
    if result == "ok":
        return "ok"
    if result == "escalate":
        return "escalate"
    return "error"


def parse_send_otp(payload: Any) -> OtpStatus:
    if not isinstance(payload, Mapping):
        return "error"
    if payload.get("result") == "sent":
        return "sent"
    if payload.get("result") == "escalate" and payload.get("reason") == "otp_unavailable":
        return "escalate"
    return "error"  # not_verified, rate_limited, unavailable, bad_request, junk


_NOT_FOUND = {"NOT_FOUND"}
_AUTH = {"AUTH_LEVEL_INSUFFICIENT", "MISSING", "UNKNOWN", "EXPIRED", "REVOKED", "WRONG_CALL"}


def parse_tool(payload: Any) -> ToolResult:
    if not isinstance(payload, Mapping):
        return ToolResult("error")
    err = payload.get("error")
    if isinstance(err, Mapping):
        err = err.get("code")
    if err:
        code = err if isinstance(err, str) else None
        if err in _NOT_FOUND:
            return ToolResult("not_found", code=code)
        if err in _AUTH:
            return ToolResult("auth_insufficient", code=code)
        return ToolResult("error", code=code)
    if payload.get("ok") is False:
        return ToolResult("error")
    inner = payload.get("data")
    source = inner if isinstance(inner, Mapping) else payload
    data = {k: v for k, v in source.items() if k not in ("ok", "status")}
    return ToolResult("ok", data)


# --------------------------------------------------------------------------- Lambda client

TRACE_HEADER = "X-Amzn-Trace-Id"
_TRACE_HANDLER_ID = "owlbank-frontdoor-xray-trace-header"


def short_function_name(function: str) -> str:
    """A function's name without its ARN's account and Region (``...:function:<name>[:alias]``)."""
    _, sep, rest = function.partition(":function:")
    return rest.split(":", 1)[0] if sep else function


def _add_trace_header(request: Any, **_kwargs: Any) -> None:
    """botocore ``before-sign.lambda.Invoke``: the current span's X-Ray header. Never raises."""
    try:
        header = telemetry.lambda_trace_header(request.headers.get(TRACE_HEADER))
        if header is not None:
            request.headers[TRACE_HEADER] = header
    except Exception:  # noqa: BLE001, S110 - the invoke goes on without the header
        pass


def _identity(identity_source: str) -> dict[str, str]:
    """The ``identity_source`` field of an auth event: left out for ``ani``, which the auth
    service assumes when it is absent, so an older auth service still accepts an ANI
    call's events while the stacks update. Any other value is sent (``keyed_phone``; anything
    else is refused there, never read as ``ani``)."""
    return {} if identity_source == "ani" else {"identity_source": identity_source}


def register_trace_header(lambda_client: Any) -> None:
    """Register :func:`_add_trace_header` on ``lambda_client`` once (by ``unique_id``); a client
    without botocore events (a test double without them) is left alone."""
    try:
        events = lambda_client.meta.events
    except AttributeError:
        return
    try:
        events.register("before-sign.lambda.Invoke", _add_trace_header, unique_id=_TRACE_HANDLER_ID)
    except Exception as exc:  # noqa: BLE001 - tracing must not stop the wiring
        log_caught(log, "trace_header_register_failed", exc, decision="no_trace_header")


class LambdaBackendClient:
    """:class:`BackendClient` over ``lambda:InvokeFunction`` (IAM-authorized; synchronous except
    :meth:`send_otp_async` and :meth:`prepare_welcome_async`, ``Event`` invokes).

    ``lambda_client`` is a boto3 Lambda client created by the caller (with short
    connect/read timeouts). No client is created here.
    """

    def __init__(self, lambda_client: Any, *, auth_function: str, bank_function: str) -> None:
        self._client = lambda_client
        self._auth = auth_function
        self._bank = bank_function
        register_trace_header(lambda_client)

    def _invoke(self, function: str, event: Mapping[str, Any]) -> Any:
        op = event.get("op")
        with telemetry.child_span(f"lambda.invoke {op}", **{
                "owlbank.function": short_function_name(function),
                "owlbank.invocation_type": "RequestResponse"}) as span:
            try:
                with timed_call("lambda", f"invoke:{op}"):
                    resp = self._client.invoke(FunctionName=function, InvocationType="RequestResponse",
                                               Payload=json.dumps(event).encode("utf-8"))
            except Exception as exc:  # network, throttling, timeouts
                span.set("owlbank.outcome", "error")
                log_caught(log, "backend invoke failed", exc, op=op, decision="fallback_error_result")
                return None
            if resp.get("FunctionError"):
                span.set("owlbank.outcome", "function_error")
                log.warning("backend function error", extra={
                    "op": op, "function_error": str(resp.get("FunctionError"))[:40],
                    "decision": "fallback_error_result"})
                return None
            try:
                raw = resp["Payload"].read()
                result = json.loads(raw)
            except (KeyError, ValueError, AttributeError, TypeError) as exc:
                span.set("owlbank.outcome", "malformed")
                log_caught(log, "backend returned malformed payload", exc, op=op, decision="fallback_error_result")
                return None
            span.set("owlbank.outcome", "ok")
            return result

    def ani_lookup(self, *, call_sid: str, ani: str) -> LookupResult:
        return parse_lookup(self._invoke(self._bank, {"op": "ani_lookup", "token": None,
                                                      "call_sid": call_sid, "args": {"ani": ani}}))

    def verify(self, *, call_sid: str, ani: str | None, factors: Mapping[str, str],
               target_level: int, stir_verstat: str | None = None,
               identity_source: str = "ani") -> VerifyResult:
        return parse_verify(self._invoke(self._auth, {
            "op": "verify", "call_sid": call_sid, "ani": ani, "factors": dict(factors),
            "target_level": target_level, "stir_verstat": stir_verstat,
            **_identity(identity_source),
        }))

    def precheck(self, *, call_sid: str, ani: str | None, target_level: int,
                 stir_verstat: str | None = None, identity_source: str = "ani") -> PrecheckStatus:
        return parse_precheck(self._invoke(self._auth, {
            "op": "precheck", "call_sid": call_sid, "ani": ani, "target_level": target_level,
            "stir_verstat": stir_verstat, **_identity(identity_source),
        }))

    def send_otp(self, *, call_sid: str, ani: str | None, language: str | None = None,
                 stir_verstat: str | None = None, identity_source: str = "ani") -> OtpStatus:
        payload = self._invoke(self._auth, {"op": "send_otp", "call_sid": call_sid, "ani": ani,
                                            "language": language, "stir_verstat": stir_verstat,
                                            **_identity(identity_source)})
        return parse_send_otp(payload)

    def send_otp_async(self, *, call_sid: str, ani: str | None, language: str | None = None,
                       stir_verstat: str | None = None, identity_source: str = "ani") -> bool:
        event = {"op": "send_otp", "call_sid": call_sid, "ani": ani, "language": language,
                 "stir_verstat": stir_verstat, **_identity(identity_source),
                 "record": True}
        return self._invoke_event(self._auth, event, fallback="fallback_synchronous_send")

    def prepare_welcome_async(self, *, call_sid: str, customer_id: str, language: str) -> bool:
        event = {"op": "prepare_welcome", "call_sid": call_sid, "customer_id": customer_id,
                 "language": language}
        return self._invoke_event(self._auth, event, fallback="fallback_welcome")

    def _invoke_event(self, function: str, event: Mapping[str, Any], *, fallback: str) -> bool:
        """An ``Event`` invoke (Lambda answers 202 at once): ``True`` when it was accepted. Never
        logs the payload; ``fallback`` names what the caller does when it wasn't."""
        op = str(event.get("op"))
        with telemetry.child_span(f"lambda.invoke {op}", **{
                "owlbank.function": short_function_name(function),
                "owlbank.invocation_type": "Event"}) as span:
            try:
                with timed_call("lambda", f"invoke_event:{op}"):
                    resp = self._client.invoke(FunctionName=function, InvocationType="Event",
                                               Payload=json.dumps(event).encode("utf-8"))
            except Exception as exc:  # network, throttling, timeouts
                span.set("owlbank.outcome", "error")
                log_caught(log, "backend async invoke failed", exc, op=op, decision=fallback)
                return False
            status = resp.get("StatusCode") if isinstance(resp, Mapping) else None
            if status != 202:
                span.set("owlbank.outcome", "not_accepted")
                log.warning("backend async invoke not accepted", extra={
                    "op": op, "status_code": status if isinstance(status, int) else None,
                    "decision": fallback})
                return False
            span.set("owlbank.outcome", "accepted")
            return True

    def get_balance(self, *, token: str, call_sid: str, account: str) -> ToolResult:
        return parse_tool(self._invoke(self._bank, {"op": "get_balance", "token": token,
                                                    "call_sid": call_sid,
                                                    "args": {"account": account}}))

    def run_tool(self, *, token: str, call_sid: str, tool: str, args: Mapping[str, Any],
                 idempotency_key: str | None) -> ToolResult:
        event: dict[str, Any] = {"op": tool, "token": token, "call_sid": call_sid,
                                 "args": dict(args)}
        if idempotency_key is not None:
            event["idempotency_key"] = idempotency_key
        return parse_tool(self._invoke(self._bank, event))
