"""Read-only readiness check of the deployed secrets.

Usage::

    uv run python scripts/check_secrets.py --cdk-outputs cdk-outputs.json
    uv run python scripts/check_secrets.py --secret twilio_api_key=ARN --secret basic_auth=ARN ...

Run it after the secrets are written and after every deploy: it fails at setup, not on the first call.
A restricted key is written later than the others (``twilio_setup.py --phase pre``), so while it still
holds the placeholder it shows ``PENDING`` and the exit stays 0; ``--strict`` makes that a problem.
It checks ``twilio_api_key``, ``twilio_auth_token`` and ``basic_auth``, plus the restricted Twilio key
``context_api_key`` (the Context API's) when its ARN is given (``--secret NAME=ARN`` or the
``SecretContextApiKey`` CDK output). That secret exists only with ``contextApiRestrictedKey`` (on by
default), so an absent ARN means "not deployed", not an error. For each secret it checks that

* the secret is encrypted with a customer-managed KMS key, not the AWS-managed one
  (``describe_secret``, and ``kms:DescribeKey`` when a KMS client is available), and
* the value reads and is well formed (``scripts/_common.read_secret_value``: not the CDK placeholder, a JSON
  object, every field present and matching ``owlbank.secret_schema``).

It prints one table row per secret and field (or per problem) and exits **0** when every secret is
valid (or only a restricted key is pending, without ``--strict``), **1** when any has a problem (placeholder, malformed, AWS-managed key, no access), **2** on a
usage error (bad arguments, no ARN for a required secret). Every secret is checked even after a failure.

Secret ARNs come from ``--secret NAME=ARN`` or the CDK outputs file; Region as in ``resolve_region``.
Nothing is written. **Values are never printed or logged**: only secret names, ARNs, field names and
problem kinds. For ``basic_auth`` it also reports whether a previous password exists and its deadline
(``previous: active until <time>``, or a warning ``expired, clear it``), never the password.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owlbank.secret_schema import OPTIONAL_PAIRS, SCHEMAS, previous_pair, validate
from scripts._common import (
    RESTRICTED_KEY_OUTPUTS,
    SECRET_OUTPUTS,
    PlaceholderError,
    ScriptError,
    _aws_error_code,
    add_secret_args,
    load_cdk_outputs,
    parse_secret_overrides,
    read_secret_value,
    resolve_region,
    schema_of,
    script_error,
    script_start,
    script_step,
    secret_arn,
    start_logging,
)

log = logging.getLogger("owlbank.check_secrets")

Row = tuple[str, str, str, str]  # secret, field, status, detail


def check_kms(arn: str, sm: Any, kms: Any | None) -> str | None:
    """``None`` when the secret uses a customer-managed KMS key, else a value-free problem text."""
    try:
        desc = sm.describe_secret(SecretId=arn)
    except Exception as exc:  # a client may raise anything; only its code is shown
        return f"describe_secret failed: {_aws_error_code(exc)}"
    key_id = desc.get("KmsKeyId") if isinstance(desc, Mapping) else None
    if not key_id or "alias/aws/" in key_id:
        return "uses the AWS-managed key, not a customer-managed KMS key"
    if kms is not None:
        try:
            meta = kms.describe_key(KeyId=key_id)["KeyMetadata"]
        except Exception as exc:
            return f"kms describe_key failed: {_aws_error_code(exc)}"
        if meta.get("KeyManager") != "CUSTOMER":
            return f"KMS key manager is {meta.get('KeyManager')!r}, not CUSTOMER"
    return None


PENDING = "PENDING"
WARN = "WARN"  # not a problem: the exit stays 0
PENDING_DETAIL = "created, not written yet: twilio_setup.py --phase pre --apply writes it (README, first deploy step 7)"


def previous_rows(logical: str, value: Mapping[str, Any] | None, now: float) -> list[Row]:
    """Rows about the optional previous-password pair of ``logical`` (none for a secret without one):
    whether a previous password exists and its deadline, never its value. A malformed pair is a
    problem; an active one is ``ok`` with its deadline; an expired one is a ``WARN`` to clear it."""
    schema = schema_of(logical)
    if schema not in OPTIONAL_PAIRS:
        return []
    field = OPTIONAL_PAIRS[schema][0]
    if not isinstance(value, Mapping):  # the main read failed: it already reported that
        return []
    problems = [p for p in validate(schema, value) if p.field in OPTIONAL_PAIRS[schema]]
    if problems:
        return [(logical, p.field or field, "PROBLEM", p.kind) for p in problems]
    pair = previous_pair(schema, value)
    if pair is None:
        return [(logical, field, "ok", "no previous password")]
    until = datetime.fromtimestamp(pair[1], UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    if now < pair[1]:
        return [(logical, field, "ok", f"previous: active until {until}; apply the new password to "
                 "Twilio (twilio_setup.py --phase post --apply) before then")]
    return [(logical, field, WARN, f"previous: expired at {until}, clear it "
             "(populate_secrets.py --clear-previous-basic-password --apply)")]


def check_one(logical: str, arn: str, sm: Any, kms: Any | None, *, strict: bool = False,
              now: float | None = None) -> list[Row]:
    """Table rows for one secret: one ``ok`` row per field, or one row per problem. A restricted key that
    still holds the placeholder is ``PENDING``, not a problem, unless ``strict``."""
    rows: list[Row] = []
    kms_problem = check_kms(arn, sm, kms)
    if kms_problem:
        rows.append((logical, "(kms key)", "PROBLEM", kms_problem))
    value: dict[str, Any] | None = None
    try:
        value = read_secret_value(logical, arn, sm)
    except PlaceholderError as exc:
        if logical in RESTRICTED_KEY_OUTPUTS and not strict:
            rows.append((logical, "(value)", PENDING, PENDING_DETAIL))
        else:
            rows.append((logical, "(value)", "PROBLEM", str(exc)))
    except ScriptError as exc:
        rows.append((logical, "(value)", "PROBLEM", str(exc)))
    else:
        if not kms_problem:
            rows.extend((logical, field, "ok", "") for field in SCHEMAS[schema_of(logical)])
        rows.extend(previous_rows(logical, value, time.time() if now is None else now))
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only check that the deployed secrets are set, well "
                                            "formed and use a customer-managed KMS key (values are "
                                            "never printed).")
    add_secret_args(p)
    p.add_argument("--strict", action="store_true",
                   help="treat a restricted key that is still the placeholder as a problem (use it after "
                        "twilio_setup.py --phase pre --apply, and after every deploy)")
    return p


def _table(rows: list[Row]) -> str:
    head: Row = ("SECRET", "FIELD", "STATUS", "DETAIL")
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(3)]
    return "\n".join("  ".join(r[i].ljust(widths[i]) for i in range(3)) + ("  " + r[3] if r[3] else "")
                     for r in [head, *rows])


def main(argv: list[str] | None = None, *, sm_client: Any | None = None,
         kms_client: Any | None = None, clock: Callable[[], float] = time.time) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse already printed the usage error
        return exc.code if isinstance(exc.code, int) else 2
    script_start(log, "check_secrets")
    try:
        explicit = parse_secret_overrides(args.secret)
        outputs = load_cdk_outputs(args.cdk_outputs)
        arns = {name: secret_arn(name, outputs, explicit) for name in SECRET_OUTPUTS}
        for name in RESTRICTED_KEY_OUTPUTS:
            try:
                arns[name] = secret_arn(name, outputs, explicit)
            except ScriptError:
                log.info("secret has no ARN: not checked", extra={"secret_name": name})
        if sm_client is None or kms_client is None:
            import boto3

            region = resolve_region(args.region, arns=arns.values(), logger=log)
            sm_client = sm_client or boto3.client("secretsmanager", region_name=region)
            kms_client = kms_client or boto3.client("kms", region_name=region)
    except ScriptError as exc:
        script_error(log, "check_secrets", exc, step="resolve", decision="exit_2")
        print(f"check_secrets: {exc}", file=sys.stderr)
        return 2

    rows: list[Row] = []
    for name, arn in arns.items():
        got = check_one(name, arn, sm_client, kms_client, strict=args.strict, now=clock())
        bad = any(r[2] not in ("ok", PENDING, WARN) for r in got)
        script_step(log, "check_secrets", "check", "problem" if bad else "ok", secret_name=name)
        rows.extend(got)
    print(_table(rows))
    for r in rows:
        if r[2] == WARN:
            print(f"check_secrets: warning: {r[0]}: {r[3]}", file=sys.stderr)
    problems = sorted({r[0] for r in rows if r[2] not in ("ok", PENDING, WARN)})
    if problems:
        print(f"check_secrets: NOT READY: {', '.join(problems)}", file=sys.stderr)
        return 1
    pending = sorted({r[0] for r in rows if r[2] == PENDING})
    if pending:
        print(f"check_secrets: ready so far; still to write: {', '.join(pending)} "
              "(twilio_setup.py --phase pre --apply); re-run with --strict after it")
        return 0
    print("check_secrets: all secrets ready")
    return 0


if __name__ == "__main__":
    start_logging("check_secrets")
    sys.exit(main())
