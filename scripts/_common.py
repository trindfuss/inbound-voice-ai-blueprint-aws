"""Helpers shared by the deploy scripts (seed, populate_secrets, twilio_setup).

Nothing here creates an AWS or Twilio client: :func:`read_secret` uses the Secrets Manager
client its caller injects.

**Secret values never appear in exceptions, logs or output.**
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from owlbank import obs
from owlbank.secret_schema import OPTIONAL_PAIRS, SCHEMAS, validate

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

#: NANP numbers reserved for fictional use: 555-0100 to 555-0199.
FICTIONAL_PHONE = re.compile(r"^\+1\d{3}55501\d{2}$")
#: An E.164 number: "+", country code, up to 15 digits.
E164 = re.compile(r"^\+[1-9]\d{7,14}$")
#: Dialable-looking phone numbers inside free text (same shape as tests/unit/test_data_files.py).
PHONE_LIKE = re.compile(r"\+\d[\d\s().-]{8,}\d|\(\d{3}\)\s?\d{3}[\s.-]?\d{4}")


# --------------------------------------------------------------------------- logging
#
# The scripts print for the person running them and also log JSON lines (stderr) for a record of
# what ran: a start line, one line per step with its outcome, and errors with context. Neither
# ever carries a secret value: step names and identifiers only, and every field passes through
# the owlbank.obs scrubber.


def start_logging(script: str) -> None:
    """Call from ``if __name__ == "__main__"``: JSON log lines on stderr, level from OWLBANK_LOG_LEVEL."""
    obs.configure_logging(script, stream=sys.stderr)


def script_start(log: logging.Logger, script: str, **fields: Any) -> None:
    obs.emit(log, logging.INFO, "script_start", script=script, **fields)


def script_step(log: logging.Logger, script: str, step: str, outcome: str, **fields: Any) -> None:
    """One step line; ``outcome`` is ``ok``, ``changed``, ``skipped``, ``dry_run`` or ``error``."""
    obs.emit(log, logging.WARNING if outcome == "error" else logging.INFO, "script_step",
             script=script, step=step, outcome=outcome, **fields)


def script_error(log: logging.Logger, script: str, exc: BaseException, *, decision: str,
                 step: str | None = None, **fields: Any) -> None:
    """An error line: class, AWS or Twilio code, what the script did about it and the stack (frames and
    class names, never the message: :func:`owlbank.obs.log_caught`). A :class:`ScriptError` message is
    user-facing and secret-free by contract (an operator's instruction, no caller input), so it is the
    one message that is included."""
    if isinstance(exc, ScriptError):
        fields["error_message"] = str(exc)
    obs.log_caught(log, "script_error", exc, level=logging.ERROR, script=script, step=step,
                   decision=decision, **fields)


class ScriptError(Exception):
    """A user-facing error. Messages never include secret values."""


class PlaceholderError(ScriptError):
    """A secret still holds the generated placeholder: it is created but not written yet."""


# --------------------------------------------------------------------------- CLI


def add_mode_args(parser: argparse.ArgumentParser) -> None:
    """``--dry-run`` (the default) or ``--apply``."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="apply", action="store_false",
                      help="print what would happen and make no changes (default)")
    mode.add_argument("--apply", dest="apply", action="store_true",
                      help="perform the changes")
    parser.set_defaults(apply=False)


def parse_pairs(values: Iterable[str] | None, *, what: str) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` arguments; duplicates and blanks are errors."""
    out: dict[str, str] = {}
    for raw in values or ():
        key, sep, value = raw.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise ScriptError(f"{what}: expected KEY=VALUE, got {raw.split('=')[0]!r}=...")
        if key in out:
            raise ScriptError(f"{what}: {key!r} given twice")
        out[key] = value
    return out


# --------------------------------------------------------------------------- AWS region

#: The Region used when nothing else names one (the blueprint deploys to us-east-1).
DEFAULT_REGION = "us-east-1"
#: The Region field of an ARN (``arn:aws:secretsmanager:us-east-1:123456789012:secret:...``).
_ARN_REGION = re.compile(r"^arn:aws[a-z-]*:[a-z0-9-]+:([a-z]{2}(?:-[a-z]+)+-[0-9]):")
_common_log = logging.getLogger("owlbank.scripts")


def region_from_arn(arn: str) -> str | None:
    """The Region in an ARN, or ``None`` for anything else (a plain name, a global ARN with no Region)."""
    m = _ARN_REGION.match(arn) if isinstance(arn, str) else None
    return m.group(1) if m else None


def resolve_region(explicit: str | None = None, *, arns: Iterable[str] = (),
                   env: Mapping[str, str] | None = None, logger: logging.Logger | None = None) -> str:
    """The AWS Region a script's clients use, so none depends on the SSO profile having a default.

    Precedence: (1) ``explicit`` (``--region``); (2) the Region in the ARN of the secret or table the
    script was given (``arns``; non-ARNs are ignored; ARNs in two Regions are an error, since one
    client can't serve both, so pass ``--region``); (3) ``AWS_DEFAULT_REGION``, then ``AWS_REGION``
    (boto3 itself reads only the first); (4) :data:`DEFAULT_REGION`, with an INFO log line saying so.
    ``env`` defaults to ``os.environ``; tests inject it.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    from_arns = {r for r in (region_from_arn(a) for a in arns) if r}
    if len(from_arns) > 1:
        raise ScriptError(f"the ARNs are in different Regions ({', '.join(sorted(from_arns))}); "
                          "pass --region")
    if from_arns:
        return next(iter(from_arns))
    environ = os.environ if env is None else env
    for var in ("AWS_DEFAULT_REGION", "AWS_REGION"):
        value = environ.get(var, "").strip()
        if value:
            return value
    (logger or _common_log).info(
        "no AWS region from --region, an ARN, AWS_DEFAULT_REGION or AWS_REGION; using %s", DEFAULT_REGION,
        extra={"region": DEFAULT_REGION})
    return DEFAULT_REGION


# --------------------------------------------------------------------------- CDK outputs


def load_cdk_outputs(path: str | Path | None) -> dict[str, str]:
    """Flatten a ``cdk deploy --outputs-file`` JSON (``{stack: {key: value}}``) to ``{key: value}``.

    A flat ``{key: value}`` file is accepted too. ``None`` gives ``{}``.
    """
    if path is None:
        return {}
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScriptError(f"can't read CDK outputs {path}: {type(exc).__name__}") from None
    if not isinstance(doc, dict):
        raise ScriptError(f"CDK outputs {path}: expected a JSON object")
    flat: dict[str, str] = {}
    for key, value in doc.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str):
                    flat[k] = v
        elif isinstance(value, str):
            flat[key] = value
    return flat


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_output(outputs: Mapping[str, str], candidates: Iterable[str]) -> str | None:
    """First output whose key matches a candidate, ignoring case and punctuation.

    CDK appends a hash suffix to auto-named outputs (``TableCustomersABC123``), so a
    key that *starts with* a candidate also matches when no exact match exists.
    """
    normed = {_norm(k): v for k, v in outputs.items()}
    cands = [_norm(c) for c in candidates]
    for c in cands:
        if c in normed:
            return normed[c]
    for c in cands:
        hits = [v for k, v in normed.items() if k.startswith(c)]
        if len(hits) == 1:
            return hits[0]
    return None


# --------------------------------------------------------------------------- Secrets Manager
#
# Every script reads a deployed secret the same way: the ARN from ``--secret NAME=ARN`` or the CDK
# outputs file (:func:`secret_arn`), then ``get_secret_value`` through an injected client and the
# shared shape check (:func:`read_secret`). Errors name the secret, the ARN, the field and the IAM
# action; never a value.

#: logical secret name -> CDK output keys that may hold its ARN, in order. The first is what
#: ``data_stack.py`` emits today (``Secret`` + the logical name in CamelCase); the others are
#: older names, still accepted.
SECRET_OUTPUTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "twilio_api_key": ("SecretTwilioApiKey", "TwilioApiKeySecretArn", "TwilioApiKeySecret"),
    "twilio_auth_token": ("SecretTwilioAuthToken", "TwilioAuthTokenSecretArn", "TwilioAuthTokenSecret"),
    "basic_auth": ("SecretBasicAuth", "BasicAuthSecretArn", "BasicAuthSecret"),
})

#: The restricted Twilio key secrets (``data_stack.py`` ``OPTIONAL_SECRETS``): ``context_api_key`` with
#: ``contextApiRestrictedKey`` (on by default). Same JSON shape as ``twilio_api_key``
#: (:data:`RESTRICTED_KEY_SCHEMA`).
RESTRICTED_KEY_OUTPUTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "context_api_key": ("SecretContextApiKey",),
})
#: The schema (``owlbank.secret_schema.SCHEMAS``) a restricted key secret is checked against.
RESTRICTED_KEY_SCHEMA = "twilio_api_key"
_KNOWN_SECRETS: Mapping[str, tuple[str, ...]] = MappingProxyType({**SECRET_OUTPUTS, **RESTRICTED_KEY_OUTPUTS})

#: The IAM action a script needs to read a secret (named in the ``AccessDenied`` error).
GET_SECRET_ACTION = "secretsmanager:GetSecretValue"  # noqa: S105 (an IAM action name, not a secret)
_ACCESS_DENIED = frozenset({"AccessDeniedException", "AccessDenied"})


def add_secret_args(parser: argparse.ArgumentParser) -> None:
    """``--cdk-outputs``, ``--secret NAME=ARN`` (repeatable) and ``--region``, for scripts that read
    secrets. Resolve with :func:`parse_secret_overrides`, :func:`load_cdk_outputs` and
    :func:`secret_arn`."""
    parser.add_argument("--cdk-outputs", metavar="PATH",
                        help="cdk deploy --outputs-file JSON (the Data stack's Secret* outputs)")
    parser.add_argument("--secret", action="append", metavar="NAME=ARN",
                        help=f"secret ARN, overriding the outputs file; NAME in {sorted(_KNOWN_SECRETS)}")
    parser.add_argument("--region", help="AWS region (default: the secrets' ARN, then AWS_DEFAULT_REGION "
                                         "or AWS_REGION, then us-east-1)")


def parse_secret_overrides(values: Iterable[str] | None) -> dict[str, str]:
    """``--secret NAME=ARN`` arguments as ``{logical: arn}``; an unknown NAME is an error."""
    pairs = parse_pairs(values, what="--secret")
    unknown = sorted(set(pairs) - set(_KNOWN_SECRETS))
    if unknown:
        raise ScriptError(f"--secret: unknown secret name(s) {unknown}; known: {sorted(_KNOWN_SECRETS)}")
    return pairs


def secret_arn(logical: str, outputs: Mapping[str, str], explicit: Mapping[str, str]) -> str:
    """The ARN of secret ``logical``: ``explicit[logical]`` (``--secret``) when given, else the first
    output matching :data:`SECRET_OUTPUTS` (:func:`find_output`), else :class:`ScriptError` naming the
    secret and the outputs tried. Makes no AWS call."""
    if logical not in _KNOWN_SECRETS:
        raise ScriptError(f"unknown secret {logical!r}; known: {sorted(_KNOWN_SECRETS)}")
    given = explicit.get(logical)
    if given:
        return given
    candidates = _KNOWN_SECRETS[logical]
    found = find_output(outputs, candidates)
    if found:
        return found
    raise ScriptError(f"no ARN for secret {logical}: none of the CDK outputs {', '.join(candidates)} "
                      f"found; pass --cdk-outputs or --secret {logical}=ARN")


def schema_of(logical: str) -> str:
    """The schema a secret is validated against: a restricted key is shaped like the Twilio API key."""
    return RESTRICTED_KEY_SCHEMA if logical in RESTRICTED_KEY_OUTPUTS else logical


def _aws_error_code(exc: BaseException) -> str:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = resp.get("Error", {}).get("Code")
        if isinstance(code, str) and code:
            return code
    return type(exc).__name__


def read_secret(logical: str, arn: str, client: Any) -> dict[str, str]:
    """:func:`read_secret_value` reduced to the secret's string fields."""
    return {k: v for k, v in read_secret_value(logical, arn, client).items() if isinstance(v, str)}


def read_secret_value(logical: str, arn: str, client: Any) -> dict[str, Any]:
    """Read secret ``logical`` at ``arn`` with ``client.get_secret_value`` and check its shape
    (:func:`owlbank.secret_schema.validate`). Returns the whole parsed value (the optional pair's
    deadline is an int).

    :class:`ScriptError` (never chained, never carrying a value) when the read fails (``AccessDenied``
    names :data:`GET_SECRET_ACTION`), the value isn't a JSON string, or it has problems (the
    placeholder, a missing or malformed field: each named by field and kind).
    """
    schema = schema_of(logical)
    if schema not in SCHEMAS:
        raise ScriptError(f"unknown secret {logical!r}; known: "
                          f"{sorted([*SCHEMAS, *RESTRICTED_KEY_OUTPUTS])}")
    try:
        resp = client.get_secret_value(SecretId=arn)
    except Exception as exc:  # a client may raise anything; its message is never shown
        code = _aws_error_code(exc)
        if code in _ACCESS_DENIED:
            raise ScriptError(f"secret {logical} ({arn}): access denied; the caller needs "
                              f"{GET_SECRET_ACTION} on it (and kms:Decrypt on its key)") from None
        raise ScriptError(f"secret {logical} ({arn}): get_secret_value failed: {code}") from None
    raw = resp.get("SecretString") if isinstance(resp, Mapping) else None
    if not isinstance(raw, str):
        raise ScriptError(f"secret {logical} ({arn}): no SecretString (expected a JSON object)")
    try:
        value = json.loads(raw)
    except ValueError:
        raise ScriptError(f"secret {logical} ({arn}): value is not JSON") from None
    # the optional pair is ignored here, as in owlbank.secrets._check; check_secrets reports it
    optional = OPTIONAL_PAIRS.get(schema, ())
    problems = [replace(p, secret=logical) for p in validate(schema, value) if p.field not in optional]
    if problems:
        hint = (" (still the placeholder: set it after deploy)"
                if any(p.kind == "placeholder" for p in problems) else "")
        error = PlaceholderError if hint else ScriptError
        raise error(f"secret {logical} ({arn}) is not usable{hint}: "
                    + "; ".join(p.describe() for p in problems))
    _common_log.info("secret read", extra={"secret_name": logical})
    return dict(value)


# --------------------------------------------------------------------------- masking


def mask_phone(phone: str) -> str:
    """``+1******1234``: keeps the country code and last four digits."""
    if not isinstance(phone, str) or len(phone) <= 6:
        return "***"
    return phone[:2] + "*" * (len(phone) - 6) + phone[-4:]


def walk(obj: Any, path: str = "$") -> Iterable[tuple[str, str | None, Any]]:
    """Yield ``(path, key, value)`` for every leaf in a JSON-like structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                yield from walk(v, f"{path}.{k}")
            else:
                yield f"{path}.{k}", k, v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                yield from walk(v, f"{path}[{i}]")
            else:
                yield f"{path}[{i}]", None, v
