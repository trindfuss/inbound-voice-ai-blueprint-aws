"""scripts/check_resolver.py: the live resolver check at deploy.

One real Converse call per resolver tool (``resolve_answer``, ``resolve_answer_generative``), built by
the resolver's own request code (:meth:`BedrockResolver.request`) from a fixed synthetic request: an
English ``ssn4`` question (digits, length 4) and the reply "seven seven and seven seven". No caller
data. The client is the front door's (:func:`client_for_budget`: ``connect_timeout = read_timeout =
budget``, one attempt), and the reply is parsed by the resolver's own parser (``output_from_tool``).

A call passes only if Bedrock accepted it, the outcome is ``ok``, the value is ``7777`` and the wall
time is within ``--budget-ms``. Unit tests with a stubbed client can't catch a request Bedrock
refuses (a strict tool schema the SDK accepted failed every call on the deployed line, fixed in
2abb7b0); this check does.

Output: one line per tool (wall latency, Bedrock's ``latencyMs``, tokens, PASS or FAIL and why).
Never the utterance or model text beyond the parsed value. Exit 0 if both pass, 1 if any fails, 2 if
refused by the invocation logging guard, which is the bench's (``slot_model_bench``): it refuses
while Bedrock invocation logging delivers text in ``--region``, unless ``--accept-text-logging``
(dev or test only) is passed, which warns and runs. One untimed warm-up call precedes the timed
calls of ``default``, so a cold first call doesn't fail the budget.

``--scenarios`` picks what runs: ``default`` (the two tool calls above), ``phone`` (the keyed-in
phone login's :data:`PHONE_SCENARIOS` through the engine against the real model, one line per
scenario: its name, mode, the engine's match and PASS or FAIL; never the utterance) or ``all``.

Usage (repository root; after every deploy whose ``slotModeCeiling`` isn't ``deterministic``)::

    uv run python scripts/check_resolver.py --profile <profile>
    uv run python scripts/check_resolver.py --profile <profile> --scenarios phone
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):  # run as a file: make `scripts` importable
    sys.path.insert(0, str(ROOT))

from owlbank.policy import Policy  # noqa: E402
from owlbank.slot_policy import DEFAULT_RESOLVER_BUDGET_MS  # noqa: E402
from owlbank.slots import commands_from_catalog, make_context, resolve  # noqa: E402
from owlbank.slots.resolver import (  # noqa: E402
    GENERATIVE_TOOL_NAME,
    TOOL_NAME,
    BedrockResolver,
    ResolverRequest,
    SlotView,
    parse_response,
)
from scripts.bench.slot_model_bench import (  # noqa: E402 - the bench's guard, not a copy of it
    DEFAULT_REGION,
    logging_blocks,
    logging_check,
    make_client_for,
    read_logging_config,
)

EXPECTED_VALUE = "7777"
UTTERANCE = "seven seven and seven seven"
#: The fixed request: synthetic, no caller data.
REQUEST = ResolverRequest(
    language="en-US",
    question="What are the last four digits of your Social Security number?",
    slot=SlotView(name="ssn4", slot_type="digits", label="last four of your Social Security number",
                  constraints={"length": 4}),
    known_options=(),
    caller_options=(),
    collected=(),
    pending=None,
    utterance=UTTERANCE,
)
#: An error message is cut to this many characters.
MAX_MESSAGE = 300


@dataclass(frozen=True, slots=True)
class PhoneScenario:
    """One ``phone`` slot fixture: what the caller said, the
    mode, and what the engine must make of it. ``model_value`` is what the offline resolver fake
    answers (``None``: the model has nothing); live, the model answers for itself and ``value`` is
    the expectation. ``resolver_called`` is whether the engine reaches the resolver at all.
    Made-up numbers only (407 and 544 are kept as spoken)."""

    name: str
    utterance: str
    mode: str                      # "assisted" | "deterministic"
    match: str                     # the engine's match: exact, inferred or none; "a|b": either
    value: str | None
    resolver_called: bool
    model_value: str | None = None


#: A scenario the model resolves: its kind may be exact or inferred.
MODEL_MATCH = "exact|inferred"

PHONE_SCENARIOS: tuple[PhoneScenario, ...] = (
    # A phrase with groups and filler between: the model reads it, the engine grounds it.
    # The model's kind is its own call: exact and inferred are both accepted (confirm: never).
    PhoneScenario("groups_with_filler", "Yeah the number is 407, eh 544, what was it 5555",
                  "assisted", MODEL_MATCH, "4075445555", True, model_value="4075445555"),
    # A self-correction of one group: the corrected group replaces only that group.
    PhoneScenario("self_correction", "407 555... no, 544 5555",
                  "assisted", MODEL_MATCH, "4075445555", True, model_value="4075445555"),
    # A spoken leading "one" (country code) in front of the ten digits, in digits and in words:
    # the parser drops it, no model (for words the model used to read it, and live it dropped
    # the last digit instead).
    PhoneScenario("leading_one_digits", "one 407 544 5555",
                  "assisted", "exact", "4075445555", False),
    PhoneScenario("leading_one_words", "one four oh seven five four four five five five five",
                  "assisted", "exact", "4075445555", False),
    # Deterministic mode never reaches the model: what the parser can't read is re-asked.
    PhoneScenario("deterministic_re_asks", "Yeah the number is 407, eh 544, what was it 5555",
                  "deterministic", "none", None, False),
    PhoneScenario("deterministic_correction_re_asks", "407 555... no, 544 5555",
                  "deterministic", "none", None, False),
)


@dataclass(frozen=True, slots=True)
class Result:
    tool: str
    latency_ms: int
    server_latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reason: str | None = None          # None: passed

    @property
    def passed(self) -> bool:
        return self.reason is None

    def line(self) -> str:
        verdict = "PASS" if self.passed else f"FAIL {self.reason}"
        return (f"{self.tool:<27} latency_ms={self.latency_ms} server_latency_ms={_fmt(self.server_latency_ms)} "
                f"tokens={_fmt(self.input_tokens)}/{_fmt(self.output_tokens)} {verdict}")


def _fmt(v: int | None) -> str:
    return "-" if v is None else str(v)


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _error_reason(exc: Exception) -> str:
    """A Bedrock refusal as its error code and message (the message says what it refused), with the
    utterance masked should it ever be echoed, cut to :data:`MAX_MESSAGE`; anything else (network,
    the client's budget timeouts) as its exception class only."""
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, Mapping) else None
    if not isinstance(error, Mapping):   # network, timeout: no Bedrock answer at all
        return f"error {type(exc).__name__}"
    code = str(error.get("Code") or type(exc).__name__)
    message = str(error.get("Message") or "")
    message = " ".join(message.replace(UTTERANCE, "[utterance]").split())[:MAX_MESSAGE]
    return f"bedrock refused: {code}" + (f": {message}" if message else "")


def check_tool(client: Any, resolver: BedrockResolver, *, generative: bool, budget_ms: int,
               clock: Callable[[], float] = time.monotonic) -> Result:
    """One Converse call for one tool, judged against the expected value and the budget."""
    tool = GENERATIVE_TOOL_NAME if generative else TOOL_NAME
    kwargs = resolver.request(REQUEST, generative=generative)
    started = clock()
    try:
        response = client.converse(**kwargs)
    except Exception as exc:  # noqa: BLE001 - every failure is a FAIL line, never a traceback
        return Result(tool, max(0, round((clock() - started) * 1000)), reason=_error_reason(exc))
    latency_ms = max(0, round((clock() - started) * 1000))
    metrics = response.get("metrics") if isinstance(response, Mapping) else None
    usage = response.get("usage") if isinstance(response, Mapping) else None
    base: dict[str, Any] = {
        "server_latency_ms": _int(metrics.get("latencyMs")) if isinstance(metrics, Mapping) else None,
        "input_tokens": _int(usage.get("inputTokens")) if isinstance(usage, Mapping) else None,
        "output_tokens": _int(usage.get("outputTokens")) if isinstance(usage, Mapping) else None,
    }
    parsed = parse_response(response, generative=generative)
    out = parsed.output
    reason: str | None = None
    if out.outcome != "ok":
        reason = f"outcome {out.outcome}"
        if parsed.stop_reason is not None:
            reason += f" (stop_reason {parsed.stop_reason})"
        elif parsed.tool_calls is not None:
            reason += f" (tool_calls {parsed.tool_calls})"
    elif out.value != EXPECTED_VALUE:
        reason = f"kind {out.kind} value {out.value!r}, expected {EXPECTED_VALUE!r}"
    elif latency_ms > budget_ms:
        reason = f"over budget ({latency_ms} ms > {budget_ms} ms)"
    return Result(tool, latency_ms, reason=reason, **base)


def run_scenario(sc: PhoneScenario, resolver: Any, *,
                 budget_ms: int | None = None) -> tuple[Any, str | None]:
    """Run ``sc`` through the engine's :func:`resolve` for the ``phone`` slot, as the front door does.
    Returns the resolution and a failure reason (``None``: it matched the scenario). ``resolver`` is
    a :class:`FakeResolver` offline or a :class:`BedrockResolver` live (``resolver_called`` is then
    only checked offline, where calls are recorded). ``budget_ms``: the resolver budget (default:
    ``slot_resolution``'s)."""
    data = ROOT / "data"
    catalog = json.loads((data / "prompt_catalog.json").read_text())
    policy = Policy.from_dict(json.loads((data / "policy.json").read_text()))
    labels = {e["key"]: e["text"] for e in catalog["entries"] if e["lang"] == "en-US"}
    ctx = make_context(question_text="What is your phone number?", labels=labels,
                       commands=commands_from_catalog(catalog["entries"], "en-US"), budget_ms=budget_ms)
    r = resolve(policy.slot("phone"), digits=None, speech=sc.utterance, confidence=0.9, lang="en-US",
                mode=sc.mode, ctx=ctx, resolver=resolver)  # type: ignore[arg-type]
    calls = getattr(resolver, "calls", None)
    if r.match not in sc.match.split("|") or r.value != sc.value:
        return r, f"match {r.match} value {r.value!r}, expected {sc.match} {sc.value!r}"
    if calls is not None and bool(calls) != sc.resolver_called:
        return r, f"resolver called {bool(calls)}, expected {sc.resolver_called}"
    return r, None


#: What to do about the refusal: the setting is the account's, so only the flag is offered here (the
#: ceiling is a deploy and preflight option; this script isn't run when the ceiling is deterministic).
LOGGING_REFUSAL_FIX = ("Invocation logging is account-wide and belongs to whoever enabled it; ask them, "
                       "and don't change it from here. For dev or test, rerun with --accept-text-logging "
                       "(the synthetic utterance will be logged; use fictional values).")


def warm_up(client: Any, resolver: BedrockResolver) -> None:
    """One untimed call with the default tool's request, so the first timed call isn't cold (8.5).
    The outcome is ignored: the timed calls report any failure."""
    with contextlib.suppress(Exception):  # never a result of its own
        client.converse(**resolver.request(REQUEST, generative=False))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--profile", default=None, help="AWS profile (default: the environment's)")
    ap.add_argument("--region", default=DEFAULT_REGION, help=f"deployment Region (default {DEFAULT_REGION})")
    ap.add_argument("--model-id", default=None,
                    help="default: the resolver's (OWLBANK_RESOLVER_MODEL_ID, else its default model)")
    ap.add_argument("--budget-ms", type=int, default=DEFAULT_RESOLVER_BUDGET_MS,
                    help=f"the resolver budget each call must meet (default {DEFAULT_RESOLVER_BUDGET_MS})")
    ap.add_argument("--accept-text-logging", action="store_true",
                    help="run even if Bedrock invocation logging delivers text in --region (dev or test "
                         "only: the synthetic utterance will be logged)")
    ap.add_argument("--scenarios", choices=SCENARIO_SETS, default="default",
                    help="default: the two resolver tool calls; phone: the keyed-in phone scenarios "
                         "against the real model; all: both (default: default)")
    return ap


SCENARIO_SETS = ("default", "phone", "all")


def phone_line(sc: PhoneScenario, match: str, why: str | None) -> str:
    """One phone scenario's line: name, mode, the engine's match and the verdict (no utterance)."""
    verdict = "PASS" if why is None else f"FAIL {why}"
    return f"phone:{sc.name:<34} mode={sc.mode} match={match} {verdict}"


def main(argv: list[str] | None = None, *, clock: Callable[[], float] = time.monotonic) -> int:
    args = build_parser().parse_args(argv)
    if args.budget_ms < 1:
        build_parser().error("--budget-ms must be at least 1")

    verdicts = logging_check(lambda r: read_logging_config(args.profile, r), (args.region,))
    print("invocation logging: " + ", ".join(f"{r} {v}" for r, v in verdicts.items()), flush=True)
    if logging_blocks(verdicts) and args.accept_text_logging:
        print("WARNING: Bedrock invocation logging delivers text in " + args.region + "; running anyway "
              "(--accept-text-logging): the synthetic utterance will be logged", file=sys.stderr)
    elif logging_blocks(verdicts):
        print("refusing to run: Bedrock invocation logging delivers text in "
              + ", ".join(r for r, v in verdicts.items() if v == "on_text"), file=sys.stderr)
        print(LOGGING_REFUSAL_FIX, file=sys.stderr)
        return 2

    client_for = make_client_for(args.profile, args.region)
    resolver = BedrockResolver(client_for, model_id=args.model_id)
    print(f"model {resolver.model_id} region {args.region} budget_ms {args.budget_ms}", flush=True)
    passed = True
    if args.scenarios in ("default", "all"):
        client = client_for(args.budget_ms)
        warm_up(client, resolver)
        results = [check_tool(client, resolver, generative=g, budget_ms=args.budget_ms, clock=clock)
                   for g in (False, True)]
        for r in results:
            print(r.line(), flush=True)
        passed = all(r.passed for r in results)
    if args.scenarios in ("phone", "all"):
        for sc in PHONE_SCENARIOS:
            res, why = run_scenario(sc, resolver, budget_ms=args.budget_ms)
            print(phone_line(sc, str(res.match), why), flush=True)
            passed = passed and why is None
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
