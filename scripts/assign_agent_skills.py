"""Give the OwlBank routing skills to named Flex agents (never to everyone).

Usage::

    uv run python scripts/assign_agent_skills.py --email agent@example.com [--email other@example.com] \\
        --cdk-outputs infra/cdk-outputs.json            # dry run: reads, changes nothing
    uv run python scripts/assign_agent_skills.py --email agent@example.com --cdk-outputs ... --apply

Finds the Flex TaskRouter workspace and the worker(s) whose ``attributes.email`` matches each ``--email``
(case-insensitive), and adds the missing skills (``en``, ``es``, ``fraud``, ``disputes``) to
``attributes.routing.skills``. Every other attribute and ``routing.levels`` is kept; a worker that
already has the skills is left alone. A Flex user has a worker only after signing in to the agent
desktop once, so "no worker found" usually means: sign in first, then re-run. The skill names must be
defined in Flex first (``twilio_setup.py --phase pre --with-flex``); you can also add skills in Flex admin
(Users, edit the user, Add skill).

Reads the Standard API key the way ``twilio_setup.py`` does (``--cdk-outputs``, ``--secret``,
``--region``); no secret is printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._agent_skills import AGENT_SKILLS, merge_worker_skills
from scripts._common import (
    ScriptError,
    add_mode_args,
    add_secret_args,
    load_cdk_outputs,
    parse_secret_overrides,
    read_secret,
    resolve_region,
    script_error,
    script_start,
    script_step,
    secret_arn,
    start_logging,
)
from scripts.twilio_setup import Applier, SetupConfig, default_client_factory, default_sm_client

slog = logging.getLogger("assign_agent_skills")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--email", action="append", required=True, metavar="ADDRESS",
                   help="the agent's email as in Flex (repeatable; there is no 'all workers' mode)")
    p.add_argument("--account-sid", help="override the account SID kept in the API key secret")
    add_mode_args(p)
    add_secret_args(p)
    return p


def _email_of(worker: Any) -> str | None:
    try:
        attrs = json.loads(worker.attributes or "{}")
    except (ValueError, TypeError):
        return None
    email = attrs.get("email") if isinstance(attrs, dict) else None
    return email.strip().lower() if isinstance(email, str) else None


def assign(client: Any, account_sid: str, emails: list[str], *, apply: bool,
           log: Callable[[str], None] = print) -> int:
    """Report per email; return 0 when every email has a worker, else 1."""
    ws_sid = Applier(client, SetupConfig(phase="pre", account_sid=account_sid, with_flex=True), None)._workspace_sid()
    ws = client.taskrouter.v1.workspaces(ws_sid)
    workers = list(ws.workers.list())
    missing_worker = False
    for email in dict.fromkeys(e.strip().lower() for e in emails):
        matches = [w for w in workers if _email_of(w) == email]
        if not matches:
            missing_worker = True
            log(f"none    {email}: no worker found (a Flex user who has never signed in to the agent "
                "desktop has no worker yet: sign in once, then re-run)")
            continue
        if len(matches) > 1:
            names = ", ".join(str(w.friendly_name) for w in matches)
            log(f"warn    {email}: matches {len(matches)} workers ({names}); updating all of them")
        for w in matches:
            merged = merge_worker_skills(w.attributes, AGENT_SKILLS)
            if merged is None:
                log(f"ok      {email}: worker {w.friendly_name} already has the skills "
                    f"({', '.join(AGENT_SKILLS)})")
            elif apply:
                ws.workers(w.sid).update(attributes=merged)
                log(f"update  {email}: worker {w.friendly_name} ({w.sid}) gets {', '.join(AGENT_SKILLS)}")
            else:
                log(f"would   {email}: worker {w.friendly_name} ({w.sid}) would get the missing skills")
    return 1 if missing_worker else 0


def main(argv: list[str] | None = None, *, sm_client: Any = None,
         client_factory: Callable[[str, str, str], Any] = default_client_factory) -> int:
    args = build_parser().parse_args(argv)
    script_start(slog, "assign_agent_skills", mode="apply" if args.apply else "dry_run",
                 emails=len(args.email))
    try:
        arn = secret_arn("twilio_api_key", load_cdk_outputs(args.cdk_outputs),
                         parse_secret_overrides(args.secret))
        if sm_client is None:
            sm_client = default_sm_client(resolve_region(args.region, arns=[arn], logger=slog))
        key = read_secret("twilio_api_key", arn, sm_client)
        account = args.account_sid or key["account_sid"]
        client = client_factory(key["api_key"], key["api_secret"], account)
        rc = assign(client, account, args.email, apply=args.apply)
    except ScriptError as exc:
        script_error(slog, "assign_agent_skills", exc, step="assign", decision="exit_3")
        print(f"assign_agent_skills: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # Twilio errors: the type and status only
        script_error(slog, "assign_agent_skills", exc, step="assign", decision="exit_1")
        print(f"assign_agent_skills: {type(exc).__name__} (status {getattr(exc, 'status', None)}, "
              f"code {getattr(exc, 'code', None)})", file=sys.stderr)
        return 1
    script_step(slog, "assign_agent_skills", "assign", "ok" if rc == 0 else "no_worker")
    print("dry run: nothing changed; add --apply to write" if not args.apply else "done")
    return rc


if __name__ == "__main__":
    start_logging("assign_agent_skills")
    sys.exit(main())
