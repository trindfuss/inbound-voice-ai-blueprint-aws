"""WebSocket handshake time on new and on pre-warmed Runtime sessions
(``bridgeHosting=agentcore``; the p50 and p90 are the figures to keep).

Usage::

    # dry run (the default): checks the arguments, no AWS call
    uv run python scripts/bench/handshake_timer.py --cdk-outputs infra/cdk-outputs.json

    # apply: 10 new and 10 warm sessions, signed with the operator's own AWS session
    uv run python scripts/bench/handshake_timer.py --apply --cdk-outputs infra/cdk-outputs.json \\
        --output handshake.json

The Runtime is ``OwlBankBridge``'s (CDK output ``RuntimeArn``) unless ``--runtime-arn`` names one.
Each session has a fresh CallSid-shaped session ID (each one boots a microVM, which stops after the
Runtime's idle timeout, ``agentcoreIdleTimeoutSeconds``). The URL comes from the signer
(:class:`lambdas.frontdoor.relay.RuntimeUrlSigner`) with the operator's credentials and a throw-away
nonce; no call state is written, since no ``setup`` is sent.

* **new** (cold): connect straight away; the handshake includes the microVM's start.
* **warm**: ``InvokeAgentRuntime`` with ``{"action": "prewarm"}`` on the session first (as the
  ``runtime_warmer`` does; its time is reported as ``prewarm``, roughly the cold start to ``/ping``
  healthy), then connect.

The handshake time is from the start of the connection to the server's ``101`` (``websockets``);
the socket is then closed with 1000. Printed and written (``--output``, JSON): count, successes,
p50 and p90 in milliseconds per kind, failures by error class (and HTTP status), and the session
IDs. **Never a URL, a nonce or a credential.**
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a file: the repo root (scripts) and src (lambdas)
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "src")]

from lambdas.frontdoor.relay import RuntimeUrlSigner, SigningCredentials  # noqa: E402
from lambdas.runtime_warmer.handler import PREWARM_PAYLOAD  # noqa: E402
from owlbank import ws_nonce  # noqa: E402
from scripts._common import (  # noqa: E402
    ScriptError,
    add_mode_args,
    find_output,
    load_cdk_outputs,
    resolve_region,
    script_error,
    script_start,
    script_step,
    start_logging,
)

SCRIPT = "bench_handshake_timer"
slog = logging.getLogger("owlbank.bench.handshake_timer")
DEFAULT_COUNT = 10
MAX_COUNT = 50
OPEN_TIMEOUT_S = 30.0
#: CDK output with the Runtime's ARN (``OwlBankBridge`` with ``bridgeHosting=agentcore``).
RUNTIME_ARN_OUTPUTS = ("RuntimeArn",)

#: ``websockets`` logs the opening handshake at DEBUG, request line included: the pre-signed query
#: (nonce, signature, session token). Its loggers stay at WARNING whatever the log level.
logging.getLogger("websockets").setLevel(logging.WARNING)

#: ``connect(url)`` -> an async context manager yielding a websockets-like connection.
Connect = Callable[[str], Any]


def default_connect() -> Connect:
    def connect(url: str) -> Any:
        from websockets.asyncio.client import connect as ws_connect  # noqa: PLC0415

        return ws_connect(url, open_timeout=OPEN_TIMEOUT_S, ping_interval=None, close_timeout=10)

    return connect


def new_session_id() -> str:
    """A CallSid-shaped session ID (``CA`` + 32 hex): the Runtime session ID."""
    return "CA" + secrets.token_hex(16)


def error_code(exc: BaseException) -> str:
    """``ClassName`` or ``ClassName/<AWS code>``: never the message."""
    resp = getattr(exc, "response", None)
    code = resp.get("Error", {}).get("Code") if isinstance(resp, Mapping) else None
    return f"{type(exc).__name__}/{code}" if code else type(exc).__name__


def percentile(xs: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (``p`` in 0..1), ``None`` for no data."""
    if not xs:
        return None
    if len(xs) == 1:
        return round(xs[0], 1)
    q = statistics.quantiles(xs, n=100, method="inclusive")
    idx = round(p * 100)
    return round(min(xs) if idx <= 0 else max(xs) if idx >= 100 else q[idx - 1], 1)


def operator_credential_provider(session: Any, clock: Callable[[], float],
                                 *, static_lifetime_s: float = 3600.0) -> Callable[[], SigningCredentials]:
    """Signing credentials from the operator's own boto3 session (the signer's injectable provider).

    Refreshable (SSO, assumed-role) credentials carry their expiry; static ones get
    ``static_lifetime_s`` from now, which is only used for the signer's margin check."""

    def provide() -> SigningCredentials:
        creds = session.get_credentials()
        if creds is None:
            raise ScriptError("no AWS credentials in this session (refresh your AWS credentials, then retry)")
        frozen = creds.get_frozen_credentials()
        expiry = getattr(creds, "_expiry_time", None)  # botocore RefreshableCredentials
        if isinstance(expiry, datetime) and expiry.tzinfo is not None:
            expires_at = expiry.timestamp()
        else:
            expires_at = clock() + static_lifetime_s
        return SigningCredentials(access_key=frozen.access_key, secret_key=frozen.secret_key,
                                  token=frozen.token, expires_at=expires_at)

    return provide


def no_credentials() -> SigningCredentials:
    """A provider for argument checks only (a dry run never fetches credentials)."""
    raise ScriptError("no credentials in a dry run")


async def time_handshake(url: str, connect: Connect,
                         clock: Callable[[], float] = time.perf_counter) -> tuple[float | None, str | None]:
    """``(milliseconds to open, None)`` or ``(None, "ErrorClass[/HTTP status]")``."""
    started = clock()
    try:
        async with connect(url) as ws:
            ms = round((clock() - started) * 1000, 1)
            await ws.close(1000, "handshake timed")
            return ms, None
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return None, type(exc).__name__ + (f"/{status}" if isinstance(status, int) else "")


def prewarm(client: Any, runtime_arn: str, session_id: str,
            clock: Callable[[], float] = time.perf_counter) -> tuple[float | None, str | None]:
    started = clock()
    try:
        response = client.invoke_agent_runtime(agentRuntimeArn=runtime_arn, runtimeSessionId=session_id,
                                               contentType="application/json", payload=PREWARM_PAYLOAD)
        body = response.get("response") if isinstance(response, dict) else None
        if body is not None and hasattr(body, "read"):
            body.read()
    except Exception as exc:
        return None, error_code(exc)
    return round((clock() - started) * 1000, 1), None


def summary(values: list[float | None], errors: list[str | None]) -> dict[str, Any]:
    ok = [v for v in values if v is not None]
    return {"count": len(values), "ok": len(ok), "p50_ms": percentile(ok, 0.5),
            "p90_ms": percentile(ok, 0.9), "failures": dict(Counter(e for e in errors if e))}


def run_sessions(*, signer: RuntimeUrlSigner, runtime_client: Any, runtime_arn: str, count: int,
                 connect: Connect) -> dict[str, Any]:
    results: dict[str, Any] = {}
    sessions: dict[str, list[str]] = {"new": [], "warm": []}
    for kind in ("new", "warm"):
        times: list[float | None] = []
        errors: list[str | None] = []
        warm_times: list[float | None] = []
        warm_errors: list[str | None] = []
        for _ in range(count):
            sid = new_session_id()
            sessions[kind].append(sid)
            if kind == "warm":
                ms, err = prewarm(runtime_client, runtime_arn, sid)
                warm_times.append(ms)
                warm_errors.append(err)
            url = signer.mint(call_sid=sid, nonce=ws_nonce.new_nonce()[0])
            ms, err = asyncio.run(time_handshake(url, connect))
            del url
            times.append(ms)
            errors.append(err)
        results[kind] = summary(times, errors)
        if kind == "warm":
            results["prewarm"] = summary(warm_times, warm_errors)
    results["session_ids"] = sessions
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Time the Runtime's WebSocket handshake, new and warm "
                                            "sessions (dry run by default).")
    p.add_argument("--cdk-outputs", type=Path, help="cdk deploy --outputs-file JSON")
    p.add_argument("--runtime-arn", help="Runtime ARN (else CDK output RuntimeArn)")
    p.add_argument("--region", help="AWS Region (else the Runtime ARN's)")
    p.add_argument("--count", type=int, default=DEFAULT_COUNT, help="sessions of each kind (default 10)")
    p.add_argument("--output", type=Path, help="write the result here (JSON, no URL)")
    add_mode_args(p)
    return p


def _boto3_session(region: str) -> Any:
    import boto3

    return boto3.Session(region_name=region)


def main(argv: list[str] | None = None, *, session_factory: Callable[[str], Any] = _boto3_session,
         connect: Connect | None = None, clock: Callable[[], float] = time.time) -> int:
    args = build_parser().parse_args(argv)
    script_start(slog, SCRIPT, mode="apply" if args.apply else "dry_run", count=args.count)
    try:
        if not 1 <= args.count <= MAX_COUNT:
            raise ScriptError(f"--count must be 1 to {MAX_COUNT}")
        runtime_arn = args.runtime_arn or find_output(load_cdk_outputs(args.cdk_outputs),
                                                      RUNTIME_ARN_OUTPUTS)
        if not runtime_arn:
            raise ScriptError("need --runtime-arn (or CDK output RuntimeArn)")
        region = resolve_region(args.region, arns=[runtime_arn], logger=slog)
        try:  # validates the ARN and Region; no credentials are fetched here
            RuntimeUrlSigner(runtime_arn=runtime_arn, region=region, clock=clock,
                             credential_provider=no_credentials)
        except ValueError as exc:
            raise ScriptError(f"--runtime-arn/--region: {exc}") from None
    except ScriptError as exc:
        script_error(slog, SCRIPT, exc, step="plan", decision="exit_2")
        print(f"handshake_timer: {exc}", file=sys.stderr)
        return 2
    script_step(slog, SCRIPT, "plan", "ok", region=region)
    runtime_id = runtime_arn.rsplit("/", 1)[-1]

    if not args.apply:
        script_step(slog, SCRIPT, "sessions", "dry_run")
        print(f"DRY RUN: no AWS calls. Would open {args.count} new and {args.count} pre-warmed sessions on "
              f"Runtime {runtime_id} (each boots a microVM), signed with your own AWS session, and time "
              "each WebSocket handshake.")
        return 0
    try:
        session = session_factory(region)
        creds = operator_credential_provider(session, clock)()
        signer = RuntimeUrlSigner(runtime_arn=runtime_arn, region=region, clock=clock,
                                  credential_provider=lambda: creds)
        results = run_sessions(signer=signer, runtime_client=session.client("bedrock-agentcore"),
                               runtime_arn=runtime_arn, count=args.count,
                               connect=connect or default_connect())
    except ScriptError as exc:
        script_error(slog, SCRIPT, exc, step="sessions", decision="exit_3")
        print(f"handshake_timer: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # AWS errors and PresignError: class and code only
        script_error(slog, SCRIPT, exc, step="sessions", decision="exit_1")
        print(f"handshake_timer: {error_code(exc)}", file=sys.stderr)
        return 1
    doc = {"runtime": runtime_id, **results}
    script_step(slog, SCRIPT, "sessions", "ok",
                **{k: results[k] for k in ("new", "warm", "prewarm")})
    print(json.dumps(doc, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    start_logging(SCRIPT)
    sys.exit(main())
