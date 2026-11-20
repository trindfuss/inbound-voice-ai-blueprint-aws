#!/usr/bin/env python3
"""Resolve and live-test the agent's model list before a deploy (model failover preflight,
4a.6). Read-only: it lists models and profiles, then sends one tiny
synthetic request (``maxTokens`` 1, no caller data) to each list entry, because the request is the
real test (the control plane's authorization and agreement fields can disagree with it).

    uv run python scripts/resolve_models.py --profile <profile> --region us-east-1 \
        [--model-id us.anthropic.claude-haiku-5-5 ...] [--params '{"<id>": {"thinking": "off"}}']

* Lists ACTIVE Anthropic foundation models and the system-defined inference profiles of the
  deployment Region's geography prefix (``us.``, ``eu.``, ``apac.``). ``global.`` is used only when an
  entry asks for it (WARN: it may be processed in any commercial Region).
* Orders Haiku and Sonnet candidates by version. No ``--model-id``: the default list is the highest
  Haiku, the next-highest Haiku and the highest Sonnet.
* An entry that fails the request (unavailable or a hard error) is replaced by the next lower
  candidate of its series that is not already listed; the replacement is tested too and shown as a
  WARN. FAIL when an entry has no replacement or no entry works. With ``has_prompt`` (preflight,
  prompt validation) a candidate without validated prompts is skipped before its request.
* Refuses another geography's profile (``eu.`` from ``us-east-1``), and a parameter set that is
  invalid for an entry (thinking, text release, output tokens), each naming the entry.
* WARNs for a model AWS has marked LEGACY, with its end-of-life date when published.
* Prints the final list and the ``-c modelIds=[...]`` to deploy with.

Calls: bedrock:ListFoundationModels, bedrock:ListInferenceProfiles, bedrock-runtime:Converse.

The entry defaults and the error classification below are this script's own copies (it stays
standalone); a parity test keeps them in step with ``owlbank.failover``.
Clients are injected (``resolve``), so the tests use fakes and no AWS account.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from botocore.exceptions import BotoCoreError, ClientError

#: Geography prefix of a deployment Region (the system-defined profile prefix).
REGION_GEOS = (("us-gov-", "us-gov"), ("us-", "us"), ("eu-", "eu"), ("ap-", "apac"))
GLOBAL_PREFIX = "global"
SERIES = ("haiku", "sonnet")
#: Default list length: highest Haiku, next-highest Haiku, highest Sonnet.
DEFAULT_SHAPE = (("haiku", 0), ("haiku", 1), ("sonnet", 0))

THINKING_MODES = ("adaptive", "off", "default", "between_tools")  # plus ``enabled:<budget>``
TEXT_RELEASE_MODES = ("stream", "message")
#: Output tokens per call with reasoning on / off (the bridge's defaults).
MAX_TOKENS_REASONING = 2000
MAX_TOKENS_OFF = 1000
MIN_THINKING_BUDGET = 1024
#: The request's reasoning fields per thinking setting, as the bridge sends them
#: (``tac_bridge.agent_client_strands.THINKING_PARAMS``; the parity test keeps them equal). The
#: synthetic request carries them, so a setting the model refuses fails here, not on a call
#: (Sonnet 5.5 refuses ``off`` and Haiku 5.5 ``between_tools``, as checked live).
THINKING_BETA = "thinking-binding-controls-2026-08-01"
THINKING_FIELDS: dict[str, dict[str, Any] | None] = {
    "adaptive": {"anthropic_beta": [THINKING_BETA],
                 "thinking": {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}},
    "off": {"thinking": {"type": "disabled"}},
    "default": None,
    "between_tools": {"thinking": {"type": "between_tools"}},
}


# --------------------------------------------------------------------------- versions


class Version(NamedTuple):
    """The ordering key of a model ID: (family, major, minor, date, revision). Compared within a
    family; a missing minor or date is 0."""

    family: str
    major: int
    minor: int
    date: int
    rev: tuple[int, int]


_TAIL = r"(?:-(?P<date>\d{8}))?(?:-v(?P<v>\d+)(?::(?P<k>\d+))?)?"
#: ``claude-<family>-<major>[-<minor>]...`` (the current style, minor one or two digits so it can't be
#: taken for a date) and ``claude-<major>[-<minor>]-<family>...`` (``claude-3-haiku``, ``claude-3-5-sonnet``).
_NEW = re.compile(r"^claude-(?P<family>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?" + _TAIL + "$")
_OLD = re.compile(r"^claude-(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?-(?P<family>[a-z]+)" + _TAIL + "$")


def split_model_id(model_id: str) -> tuple[str, str]:
    """``(prefix, base)``: ``us.anthropic.claude-x`` -> ``("us", "anthropic.claude-x")``; a bare
    ``anthropic.claude-x`` has the prefix ``""``."""
    head, sep, rest = model_id.partition("anthropic.")
    if not sep:
        return "", model_id
    return head.rstrip("."), "anthropic." + rest


def parse_version(model_id: str) -> Version | None:
    """The version key of an Anthropic model or profile ID, or ``None`` when it is not one."""
    _, base = split_model_id(model_id)
    if not base.startswith("anthropic."):
        return None
    tail = base[len("anthropic."):]
    m = _NEW.match(tail) or _OLD.match(tail)
    if m is None:
        return None
    return Version(m["family"], int(m["major"]), int(m["minor"] or 0), int(m["date"] or 0),
                   (int(m["v"] or 0), int(m["k"] or 0)))


# --------------------------------------------------------------------------- classification

UNAVAILABLE = "unavailable"
REQUEST_ERROR = "request_error"
HARD_ERROR = "hard_error"
INVALID_ID_MESSAGE = "The provided model identifier is invalid."
REQUEST_ERROR_PREFIX = "The model returned the following errors:"
UNAVAILABLE_CODES = frozenset({"ResourceNotFoundException", "AccessDeniedException"})
#: Request errors other than ``ValidationException`` (Strands' wrapper of a too-long input).
REQUEST_ERROR_CODES = frozenset({"ContextWindowOverflowException"})


def classify_error(code: str, message: str) -> str:
    """The error classes, as ``owlbank.failover.classify`` (a parity test keeps them
    equal): ``unavailable`` (an unknown or inaccessible model: ``ResourceNotFoundException``,
    ``AccessDeniedException``, or ``ValidationException`` with exactly the invalid-identifier
    message), ``request_error`` (any other ``ValidationException``, those starting
    :data:`REQUEST_ERROR_PREFIX` among them, and a context-window overflow: the request is wrong,
    which says nothing about the model, 4a.2) or ``hard_error`` (throttles, model errors, connection
    errors and anything not recognised). Event-stream codes in lower camel case count as their
    class. Never raises."""
    code, message = str(code or "").strip(), str(message or "").strip()
    code = code[:1].upper() + code[1:]
    if code in UNAVAILABLE_CODES:
        return UNAVAILABLE
    if code == "ValidationException":
        return UNAVAILABLE if message == INVALID_ID_MESSAGE else REQUEST_ERROR
    if code in REQUEST_ERROR_CODES:
        return REQUEST_ERROR
    return HARD_ERROR


def _error_parts(exc: Exception) -> tuple[str, str]:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        err = resp.get("Error", {})
        return str(err.get("Code", "")), str(err.get("Message", ""))
    return type(exc).__name__, str(exc)


# --------------------------------------------------------------------------- entries


@dataclass(frozen=True)
class Entry:
    """One list entry with the parameters it needs."""

    model_id: str
    thinking: str
    text_release: str
    max_tokens: int

    def describe(self) -> str:
        return f"thinking={self.thinking} text_release={self.text_release} max_tokens={self.max_tokens}"


def default_params(model_id: str) -> dict[str, Any]:
    """The known defaults per model; any other model gets reasoning off with whole
    messages, valid for every model. Kept in step with ``owlbank.failover.settings`` by a parity test."""
    ver = parse_version(model_id)
    key = (ver.family, ver.major, ver.minor) if ver else None
    if key in (("haiku", 5, 5), ("sonnet", 5, 5)):
        return {"thinking": "adaptive", "text_release": "stream", "max_tokens": MAX_TOKENS_REASONING}
    return {"thinking": "off", "text_release": "message", "max_tokens": MAX_TOKENS_OFF}


def entry_params_error(thinking: Any, text_release: Any, max_tokens: Any) -> str | None:
    """Why this parameter set is invalid, or ``None``."""
    if not isinstance(thinking, str):
        return "thinking must be a string"
    budget = 0
    if thinking.startswith("enabled:"):
        try:
            budget = int(thinking.split(":", 1)[1])
        except ValueError:
            return f"thinking {thinking!r}: the budget must be an integer"
        if budget < MIN_THINKING_BUDGET:
            return f"thinking {thinking!r}: the budget must be at least {MIN_THINKING_BUDGET}"
    elif thinking not in THINKING_MODES:
        return f"thinking {thinking!r} must be one of {list(THINKING_MODES)} or enabled:<budget>"
    if text_release not in TEXT_RELEASE_MODES:
        return f"text_release {text_release!r} must be one of {list(TEXT_RELEASE_MODES)}"
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        return "max_tokens must be a positive integer"
    if text_release == "stream" and thinking == "off":
        return ("text_release=stream needs reasoning on: with reasoning off the model narrates before "
                "tool calls; set text_release=message")
    if budget and budget >= max_tokens:
        return f"thinking budget {budget} must be below max_tokens {max_tokens}"
    return None


# --------------------------------------------------------------------------- findings


@dataclass
class Finding:
    """A preflight line: ``level`` is ``PASS``, ``WARN`` or ``FAIL``."""

    level: str
    name: str
    detail: str
    fix: str = ""


@dataclass
class Resolution:
    entries: list[Entry] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    #: ``modelParams`` overrides the deployer gave, echoed for the ``-c`` command.
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.entries) and all(f.level != "FAIL" for f in self.findings)

    @property
    def model_ids(self) -> list[str]:
        return [e.model_id for e in self.entries]

    def deploy_args(self) -> str:
        """``-c modelIds=[...]`` (and ``-c modelParams=...`` when the deployer overrode any), quoted for a shell."""
        args = "-c " + shlex.quote("modelIds=" + json.dumps(self.model_ids, separators=(",", ":")))
        if self.params:
            args += " -c " + shlex.quote("modelParams=" + json.dumps(self.params, separators=(",", ":")))
        return args


def region_geo(region: str) -> str | None:
    """The system-defined profile prefix for a deployment Region (``us-east-1`` -> ``us``)."""
    for start, geo in REGION_GEOS:
        if region.startswith(start):
            return geo
    return None


def _fmt_time(value: Any) -> str:
    if value is None or value == "":
        return "no end-of-life date published"
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)[:10]


# --------------------------------------------------------------------------- control plane


def list_catalog(bedrock: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """``(lifecycle by foundation-model ID, system-defined profile IDs)`` for Anthropic."""
    models: dict[str, dict[str, Any]] = {}
    for s in bedrock.list_foundation_models(byProvider="Anthropic").get("modelSummaries", []):
        models[s["modelId"]] = s.get("modelLifecycle", {}) or {}
    profiles: list[str] = []
    token: str | None = None
    while True:
        kw: dict[str, Any] = {"typeEquals": "SYSTEM_DEFINED"}
        if token:
            kw["nextToken"] = token
        page = bedrock.list_inference_profiles(**kw)
        for p in page.get("inferenceProfileSummaries", []):
            if p.get("status", "ACTIVE") == "ACTIVE" and "anthropic." in p["inferenceProfileId"]:
                profiles.append(p["inferenceProfileId"])
        token = page.get("nextToken")
        if not token:
            return models, profiles


def candidates(models: dict[str, dict[str, Any]], profiles: list[str], geo: str, series: str) -> list[str]:
    """The geography's profiles of ``series`` whose model is ACTIVE, highest version first."""
    out: list[tuple[Version, str]] = []
    for pid in profiles:
        prefix, base = split_model_id(pid)
        ver = parse_version(pid)
        if prefix != geo or ver is None or ver.family != series:
            continue
        if models.get(base, {}).get("status") != "ACTIVE":
            continue
        out.append((ver, pid))
    return [pid for _, pid in sorted(out, reverse=True)]


def default_list(models: dict[str, dict[str, Any]], profiles: list[str], geo: str) -> list[str]:
    ranked = {s: candidates(models, profiles, geo, s) for s in SERIES}
    return [ranked[s][i] for s, i in DEFAULT_SHAPE if i < len(ranked[s])]


# --------------------------------------------------------------------------- the request


def thinking_request_fields(thinking: str) -> dict[str, Any] | None:
    """The ``additionalModelRequestFields`` for a (valid) thinking setting; ``None``: send nothing."""
    if thinking.startswith("enabled:"):
        return {"thinking": {"type": "enabled", "budget_tokens": int(thinking.split(":", 1)[1])}}
    fields = THINKING_FIELDS[thinking]
    return None if fields is None else dict(fields)


def probe(runtime: Any, model_id: str, thinking: str = "default") -> tuple[bool, str, str]:
    """Send the synthetic request with the entry's thinking fields: ``(worked, error class, error
    code)``. ``maxTokens`` is 1, or one above an ``enabled:<budget>`` budget (which must be below it)."""
    kwargs: dict[str, Any] = {"inferenceConfig": {"maxTokens": 1}}
    fields = thinking_request_fields(thinking)
    if fields is not None:
        kwargs["additionalModelRequestFields"] = fields
        budget = fields["thinking"].get("budget_tokens")
        if budget:
            kwargs["inferenceConfig"] = {"maxTokens": budget + 1}
    try:
        runtime.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": "ok"}]}], **kwargs)
    except Exception as exc:  # noqa: BLE001 - every failure is classified, none is fatal here
        code, message = _error_parts(exc)
        return False, classify_error(code, message), code or type(exc).__name__
    return True, "", ""


# --------------------------------------------------------------------------- resolver


def resolve(*, bedrock: Any, runtime: Any, region: str, model_ids: list[str] | None = None,
            params: dict[str, dict[str, Any]] | None = None, geo: str | None = None,
            has_prompt: Callable[[str], bool] | None = None) -> Resolution:
    """Validate (or build) the model list for ``region``; see the module docstring. ``has_prompt``
    says whether a model ID has validated prompts: a replacement candidate without them is
    skipped, never proposed. It doesn't judge the requested entries (preflight's ``check_prompts``
    does)."""
    res = Resolution(params=dict(params or {}))
    geo = geo or region_geo(region)
    if geo is None:
        res.findings.append(Finding("FAIL", "Model list", f"no profile prefix is known for Region {region}",
                                    "pass --geo (us, eu, apac, ...)"))
        return res
    try:
        models, profiles = list_catalog(bedrock)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "ClientError")
        call = exc.operation_name or "the model catalog listing"
        res.findings.append(Finding("FAIL", "Model list", f"Bedrock {call} failed: {code}",
                                    "allow bedrock:ListFoundationModels and bedrock:ListInferenceProfiles, "
                                    "or retry if it was throttled"))
        return res
    except BotoCoreError as exc:
        res.findings.append(Finding("FAIL", "Model list",
                                    f"the Bedrock model catalog listing failed: {type(exc).__name__}",
                                    "check the credentials, Region and network, then retry"))
        return res
    wanted = list(model_ids) if model_ids else default_list(models, profiles, geo)
    if not 1 <= len(wanted) <= 3:
        res.findings.append(Finding(
            "FAIL", "Model list", f"{len(wanted)} entries; the list takes one to three"
            + ("" if wanted else f" (no ACTIVE Haiku or Sonnet profile with the {geo}. prefix)"),
            "pass one to three model IDs"))
        return res

    reserved = set(wanted)  # IDs asked for: a replacement never duplicates one
    tried: set[str] = set()
    for requested in wanted:
        _resolve_entry(res, requested, models, profiles, geo, bedrock_runtime=runtime,
                       reserved=reserved, tried=tried, overrides=(params or {}), has_prompt=has_prompt)
    if not res.entries:
        res.findings.append(Finding("FAIL", "Model list", "no entry answered the request",
                                    "check Bedrock model access in this Region, or pass other --model-id values"))
    elif all(f.level != "FAIL" for f in res.findings):
        res.findings.append(Finding("PASS", "Model list", ", ".join(res.model_ids)))
    return res


def _resolve_entry(res: Resolution, requested: str, models: dict[str, dict[str, Any]], profiles: list[str],
                   geo: str, *, bedrock_runtime: Any, reserved: set[str], tried: set[str],
                   overrides: dict[str, dict[str, Any]],
                   has_prompt: Callable[[str], bool] | None = None) -> None:
    name = f"Model entry {requested}"
    prefix, base = split_model_id(requested)
    ver = parse_version(requested)
    if not base.startswith("anthropic.") or not prefix:
        res.findings.append(Finding("FAIL", name, "not an Anthropic inference profile ID",
                                    f"use {geo}.anthropic.<model> (a model ID alone can't be called in every Region)"))
        return
    if prefix not in (geo, GLOBAL_PREFIX):
        res.findings.append(Finding(
            "FAIL", name, f"profile prefix {prefix}. is another geography; this Region's is {geo}.",
            f"use the {geo}.{base} profile (or {GLOBAL_PREFIX}.{base} to accept processing in any commercial Region)"))
        return
    if ver is None:
        res.findings.append(Finding("FAIL", name, "the model ID is not a version this script can order",
                                    "use a Claude Haiku or Sonnet profile ID"))
        return

    def build(model_id: str) -> Entry | str:
        given = overrides.get(model_id, {})
        p = {**default_params(model_id), **given}
        if "max_tokens" not in given:  # as synth: follows the resulting thinking (1000 off, else 2000)
            p["max_tokens"] = MAX_TOKENS_OFF if p["thinking"] == "off" else MAX_TOKENS_REASONING
        err = entry_params_error(p["thinking"], p["text_release"], p["max_tokens"])
        if err:
            return err
        return Entry(model_id, p["thinking"], p["text_release"], p["max_tokens"])

    entry = build(requested)
    if isinstance(entry, str):
        res.findings.append(Finding("FAIL", name, f"invalid parameters: {entry}",
                                    "fix modelParams for this entry"))
        return
    if prefix == GLOBAL_PREFIX:
        res.findings.append(Finding(
            "WARN", name, "global. profile: requests may be processed in any commercial Region, not only in this geography",
            f"use the {geo}.{base} profile to keep processing in the geography"))
    ok, cls, code = probe(bedrock_runtime, requested, entry.thinking)
    tried.add(requested)
    if ok:
        _accept(res, entry, models, name)
        return
    if cls == REQUEST_ERROR:
        res.findings.append(Finding(
            "FAIL", name, f"the synthetic request with thinking={entry.thinking} was rejected ({code})",
            "check this model takes that thinking setting (modelParams), then run the script again"))
        return
    # Unavailable or a hard error: the next lower candidate of the series that is not listed and,
    # with ``has_prompt``, has validated prompts.
    no_prompt: list[str] = []
    for cand in candidates(models, profiles, geo, ver.family):
        cver = parse_version(cand)
        if cand in reserved or cand in tried or cver is None or not cver < ver:
            continue
        if has_prompt is not None and not has_prompt(cand):
            no_prompt.append(cand)
            continue
        tried.add(cand)
        rep = build(cand)
        if isinstance(rep, str):
            continue
        ok2, _, _ = probe(bedrock_runtime, cand, rep.thinking)
        if not ok2:
            continue
        reserved.add(cand)
        res.findings.append(Finding(
            "WARN", name, f"failed the request ({cls}: {code}); replaced by {cand} (same series, next lower version)",
            "check the replacement is the model you want, or pass your own --model-id list"))
        _accept(res, rep, models, f"Model entry {cand}")
        return
    if no_prompt:
        res.findings.append(Finding(
            "FAIL", name, f"failed the request ({cls}: {code}) and no {ver.family} replacement with validated "
            f"prompts works (skipped, no validated prompt: {', '.join(no_prompt)})",
            "request model access in the Bedrock console for this Region, validate a replacement's prompts "
            "with the prompt suite (scripts/prompt_suite.py), or choose another entry"))
        return
    res.findings.append(Finding("FAIL", name, f"failed the request ({cls}: {code}) and no {ver.family} replacement works",
                                "request model access in the Bedrock console for this Region, or choose another entry"))


def _accept(res: Resolution, entry: Entry, models: dict[str, dict[str, Any]], name: str) -> None:
    res.entries.append(entry)
    life = models.get(split_model_id(entry.model_id)[1], {})
    if life.get("status") == "LEGACY":
        res.findings.append(Finding(
            "WARN", name, f"model is LEGACY; end of life: {_fmt_time(life.get('endOfLifeTime'))}",
            "plan a replacement before it stops working"))
    res.findings.append(Finding("PASS", name, f"answered the one-token request; {entry.describe()}"))


# --------------------------------------------------------------------------- command line


def report(res: Resolution) -> int:
    for f in res.findings:
        print(f"[{f.level}] {f.name}: {f.detail}")
        if f.level != "PASS" and f.fix:
            print(f"       fix (run it yourself): {f.fix}")
    if res.entries:
        print(f"  deploy with: {res.deploy_args()}")
    return 0 if res.ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", help="AWS CLI profile")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--geo", default=None, help="profile prefix (default: from the Region)")
    ap.add_argument("--model-id", action="append", default=None,
                    help="a list entry, in preference order (repeat up to three times; default: resolved)")
    ap.add_argument("--params", default="{}",
                    help='per-entry overrides as JSON: {"<model id>": {"thinking": "off", "text_release": "message", "max_tokens": 1000}}')
    args = ap.parse_args(argv)
    import boto3  # noqa: PLC0415

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    res = resolve(bedrock=session.client("bedrock"), runtime=session.client("bedrock-runtime"),
                  region=args.region, model_ids=args.model_id, params=json.loads(args.params), geo=args.geo)
    return report(res)


if __name__ == "__main__":
    sys.exit(main())
