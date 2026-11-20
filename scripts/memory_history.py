"""Demo history writer: put a past chat on a demo customer's Memory profile as a conversation summary
and an observation, or take it off.

In production the other channels (a WhatsApp or SMS assistant, a chat widget) write these through
Conversation Orchestrator and Memory; this script stands in for them so the demo and the tests need no
second channel. The bridge's recall (``## Past Conversation Summaries``, ``## Key Observations``) then
shows the chat to the voice agent after verification.

Usage (from the repo root)::

    uv run python scripts/memory_history.py                    # plan for C-1001, nothing written
    uv run python scripts/memory_history.py --apply            # the Example Airlines dispute chat
    uv run python scripts/memory_history.py --channel sms --apply
    uv run python scripts/memory_history.py --summary "WhatsApp chat: ..." --apply   # only that item
    uv run python scripts/memory_history.py --clear --apply    # remove the demo items again

- **Customer:** ``--customer`` (default ``realNumberPlaceholders.developerPhone.fixture`` in
  ``data/fixtures.json``, C-1001): fixtures only. The profile is found by the fixture's
  ``memory_email``, as the bridge does; it must exist (a verified call or ``web_activity.py --apply``
  creates it).
- **Preset:** with neither ``--summary`` nor ``--observation``, both are built at run time from the
  customer's credit card whose transactions include "Example Airlines" (amount, merchant, card name and
  date from ``data/fixtures.json``; the channel name from ``--channel``). A customer without such a
  transaction is refused. With ``--summary`` and/or ``--observation`` only the given items are written.
- **Items:** ``source`` is ``owlbank-demo``; the summary carries a synthetic ``conversationId`` (the
  same one on the observation's ``conversationIds``); ``occurredAt`` is yesterday. An item whose text
  is already on the profile from this source is not written again, so a re-run writes nothing.
- **Store, credentials, stage:** as ``web_activity.py`` (setup output file, Secrets Manager via
  ``--cdk-outputs`` or ``--secret``, ``dev`` or ``test``; ``prod`` refused). No credential is printed or logged.
- ``--apply`` writes, then reads back for up to ``--wait`` seconds until the items show (Memory accepts
  summaries asynchronously) and says how long that took.
- ``--clear`` lists the profile's summaries and observations and deletes only those whose ``source`` is
  ``owlbank-demo``; anything else on the profile is left alone.

Exit codes: 0 done; 1 an unexpected error (its type only); 2 a bad option or missing setup; 3 a Memory
or Secrets Manager error, or the read-back timed out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owlbank import memory_http
from scripts._common import (
    DATA_DIR,
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
from scripts.web_activity import (
    DEFAULT_SETUP_OUTPUT,
    Customer,
    default_client_factory,
    read_store_id,
    resolve_customer,
)

slog = logging.getLogger("owlbank.memory_history_script")

BANNER = "Demo data: a past chat is written to Memory directly (in production another channel writes it)"
SOURCE = "owlbank-demo"
STAGES = ("dev", "test")
CHANNELS = {"whatsapp": "WhatsApp", "sms": "SMS", "rcs": "RCS", "chat": "Web chat"}
MERCHANT = "Example Airlines"
DEFAULT_WAIT_S = 30
READ_POLL_S = 1.0
MAX_TEXT_CHARS = 2000
EXIT_UNEXPECTED, EXIT_USAGE, EXIT_MEMORY = 1, 2, 3
_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"


@dataclass(frozen=True)
class Charge:
    card: str
    merchant: str
    amount: str  # "412.40"
    date: str


# --------------------------------------------------------------------------- options and content


def stage_guard(stage: str) -> None:
    if stage.strip().lower() == "prod":
        raise ScriptError("--stage prod is refused: memory_history.py writes demo data to dev and test only")
    if stage not in STAGES:
        raise ScriptError(f"--stage must be one of {', '.join(STAGES)}")


def find_charge(fixtures: Mapping[str, Any], customer_id: str) -> Charge | None:
    """The first credit-card transaction at :data:`MERCHANT` on the customer's fixture, or ``None``."""
    for c in fixtures.get("customers", []):
        if not (isinstance(c, dict) and c.get("customer_id") == customer_id):
            continue
        for acct in c.get("accounts", []):
            if not (isinstance(acct, dict) and acct.get("type") == "credit_card"):
                continue
            for t in acct.get("transactions", []):
                desc = t.get("description") if isinstance(t, dict) else None
                if not (isinstance(desc, str) and MERCHANT.lower() in desc.lower()):
                    continue
                try:
                    amount = f"{abs(Decimal(str(t.get('amount')))):.2f}"
                except InvalidOperation:
                    continue
                return Charge(str(acct.get("nickname") or "card"), MERCHANT, amount, str(t.get("date")))
    return None


def preset_texts(charge: Charge, channel: str) -> tuple[str, str]:
    """``(summary, observation)`` for the dispute chat; neutral wording (no gendered pronouns)."""
    summary = (f"{CHANNELS[channel]} chat: the customer said they don't recognise a ${charge.amount} charge "
               f"from {charge.merchant} on their {charge.card} dated {charge.date}. The chat assistant "
               "logged it for the disputes team and said they would follow up.")
    observation = (f"Customer has an open question about an unrecognised {charge.merchant} charge on the "
                   f"{charge.card}.")
    return summary, observation


def conversation_id(customer_id: str, channel: str) -> str:
    """A synthetic, stable TTID-shaped conversation ID: ``conv_conversation_`` + a ULID-shaped
    26 characters, 128 bits in Crockford base 32 with the most significant first, so the first is
    0 to 7 (Memory refuses anything else: ``^conv_conversation_[0-7][0-9a-z]{25}$``, seen live)."""
    digest = hashlib.sha256(f"owlbank-demo/{customer_id}/{channel}".encode()).digest()
    n = int.from_bytes(digest[:16], "big")
    chars = []
    for _ in range(26):
        n, r = divmod(n, 32)
        chars.append(_CROCKFORD[r])
    return "conv_conversation_" + "".join(reversed(chars))


def _stamp(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_items(summary: str | None, observation: str | None, *, conv_id: str,
                occurred_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(summaries, observations)`` request items."""
    summaries = ([{"content": summary, "conversationId": conv_id, "occurredAt": occurred_at, "source": SOURCE}]
                 if summary else [])
    observations = ([{"content": observation, "source": SOURCE, "occurredAt": occurred_at,
                      "conversationIds": [conv_id]}] if observation else [])
    return summaries, observations


def _text(value: str, flag: str) -> str:
    value = value.strip()
    if not value or len(value) > MAX_TEXT_CHARS:
        raise ScriptError(f"{flag} must be 1 to {MAX_TEXT_CHARS} characters")
    return value


# --------------------------------------------------------------------------- Memory


def _demo(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [i for i in items if i.get("source") == SOURCE]


class History:
    """The Memory side of one run, over a Twilio client (only its HTTP layer and API key are used)."""

    def __init__(self, client: Any, store_id: str, customer: Customer, *,
                 log: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.client, self.store_id, self.customer = client, store_id, customer
        self.log, self.sleep, self.clock = log, sleep, clock

    def profile_id(self) -> str:
        ids = memory_http.lookup_profile(self.client, self.store_id, self.customer.email)
        if len(ids) > 1:
            raise ScriptError(f"{len(ids)} profiles share {self.customer.email}; merge or delete the "
                              "extra ones in the Console (the script never picks one)")
        if not ids:
            raise ScriptError(f"no profile for {self.customer.email}; place a verified call or run "
                              "web_activity.py --apply first (either creates it)")
        return ids[0]

    def _lists(self, pid: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (memory_http.list_summaries(self.client, self.store_id, pid),
                memory_http.list_observations(self.client, self.store_id, pid))

    def _wait(self, pid: str, done: Callable[[list[dict[str, Any]], list[dict[str, Any]]], bool],
              wait_s: float, what: str) -> float:
        start = self.clock()
        while True:
            summaries, observations = self._lists(pid)
            elapsed = self.clock() - start
            if done(summaries, observations):
                return elapsed
            if elapsed >= wait_s:
                raise ScriptError(f"{what} not visible after {wait_s:g} s (Memory accepts them "
                                  "asynchronously); re-run the same command (it is safe to repeat)")
            self.sleep(READ_POLL_S)

    def apply(self, summaries: list[dict[str, Any]], observations: list[dict[str, Any]], wait_s: float) -> None:
        pid = self.profile_id()
        have_s, have_o = self._lists(pid)
        seen_s = {i.get("content") for i in _demo(have_s)}
        seen_o = {i.get("content") for i in _demo(have_o)}
        new_s = [i for i in summaries if i["content"] not in seen_s]
        new_o = [i for i in observations if i["content"] not in seen_o]
        if not (new_s or new_o):
            self.log(f"ok      profile {pid} already holds these items; nothing written")
            return
        if new_s:
            memory_http.create_summaries(self.client, self.store_id, pid, new_s)
            self.log(f"create  {len(new_s)} conversation summary on profile {pid}")
        if new_o:
            memory_http.create_observations(self.client, self.store_id, pid, new_o)
            self.log(f"create  {len(new_o)} observation on profile {pid}")
        want_s, want_o = {i["content"] for i in new_s}, {i["content"] for i in new_o}
        took = self._wait(pid, lambda s, o: want_s <= {i.get("content") for i in _demo(s)}
                          and want_o <= {i.get("content") for i in _demo(o)}, wait_s, "the history items")
        self.log(f"ok      items visible after {took:.1f} s")

    def clear(self, wait_s: float) -> None:
        pid = self.profile_id()
        summaries, observations = self._lists(pid)
        mine_s, mine_o = _demo(summaries), _demo(observations)
        if not (mine_s or mine_o):
            self.log(f"ok      profile {pid} has no {SOURCE} items; nothing to clear")
            return
        for s in mine_s:
            memory_http.delete_summary(self.client, self.store_id, pid, str(s["id"]))
        for o in mine_o:
            memory_http.delete_observation(self.client, self.store_id, pid, str(o["id"]))
        self.log(f"clear   {len(mine_s)} summary and {len(mine_o)} observation from {SOURCE} on profile {pid}")
        took = self._wait(pid, lambda s, o: not _demo(s) and not _demo(o), wait_s, "the deletions")
        self.log(f"ok      items gone after {took:.1f} s")


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write or clear a demo past-chat summary and observation on a "
                                            "fixture customer's Twilio Memory profile (demo data).")
    add_mode_args(p)
    add_secret_args(p)
    p.add_argument("--clear", action="store_true",
                   help=f"delete the profile's summaries and observations whose source is {SOURCE} (with --apply)")
    p.add_argument("--customer", help="fixture customer ID (default: the developer phone's fixture, C-1001)")
    p.add_argument("--channel", default="whatsapp", choices=sorted(CHANNELS),
                   help="the channel of the past chat (default whatsapp)")
    p.add_argument("--summary", help="summary text (default: the Example Airlines dispute preset)")
    p.add_argument("--observation", help="observation text (default: the Example Airlines dispute preset)")
    p.add_argument("--stage", default="dev", help="dev (default) or test; prod is refused")
    p.add_argument("--wait", type=float, default=DEFAULT_WAIT_S,
                   help="seconds to wait for the items to show (default 30)")
    p.add_argument("--setup-output", type=Path, default=DEFAULT_SETUP_OUTPUT,
                   help="twilio_setup.py output file holding memoryStoreId (default infra/twilio.context.json)")
    return p


def _api_key_arn(args: argparse.Namespace) -> str:
    """The Twilio API key secret's ARN from ``--secret`` or ``--cdk-outputs``. No AWS call; a missing
    ARN or outputs file is missing setup (:class:`ScriptError`, exit 2)."""
    return secret_arn("twilio_api_key", load_cdk_outputs(args.cdk_outputs), parse_secret_overrides(args.secret))


def _twilio_credentials(arn: str, args: argparse.Namespace, sm_client: Any | None) -> dict[str, str]:
    """The Twilio API key secret (``account_sid``, ``api_key``, ``api_secret``) from Secrets Manager.
    Errors name the secret and field, never a value."""
    if sm_client is None:
        import boto3

        sm_client = boto3.client("secretsmanager", region_name=resolve_region(args.region, arns=[arn]))
    return read_secret("twilio_api_key", arn, sm_client)


def main(argv: list[str] | None = None, *, sm_client: Any | None = None,
         client_factory: Callable[[str, str, str | None], Any] = default_client_factory,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
         now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> int:
    args = build_parser().parse_args(argv)
    print(BANNER)
    mode = "clear" if args.clear else "write"
    script_start(slog, "memory_history", mode=mode, apply=bool(args.apply), stage=args.stage)
    summaries: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    try:
        stage_guard(args.stage)
        if args.wait < 0:
            raise ScriptError("--wait must not be negative")
        try:
            fixtures = json.loads((DATA_DIR / "fixtures.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ScriptError(f"can't read data/fixtures.json: {type(exc).__name__}") from None
        customer = resolve_customer(fixtures, args.customer)
        store_id, account_sid = read_store_id(args.setup_output)
        arn = _api_key_arn(args) if args.apply else ""
        if not args.clear:
            if args.summary is None and args.observation is None:
                charge = find_charge(fixtures, customer.customer_id)
                if charge is None:
                    raise ScriptError(f"fixture {customer.customer_id} has no credit-card transaction at "
                                      f"{MERCHANT}, so the preset can't be built; pass --summary and/or "
                                      "--observation, or use C-1001")
                summary, observation = preset_texts(charge, args.channel)
            else:
                summary = _text(args.summary, "--summary") if args.summary is not None else None
                observation = _text(args.observation, "--observation") if args.observation is not None else None
            summaries, observations = build_items(
                summary, observation, conv_id=conversation_id(customer.customer_id, args.channel),
                occurred_at=_stamp(now() - timedelta(days=1)))
    except ScriptError as exc:
        script_error(slog, "memory_history", exc, step="plan", decision="exit_2")
        print(f"memory_history: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(f"stage {args.stage}, store {store_id}")
    print(f"customer {customer.customer_id} ({customer.name}), profile email {customer.email}")
    script_step(slog, "memory_history", "plan", "ok", customer=customer.customer_id)

    if not args.apply:
        script_step(slog, "memory_history", mode, "dry_run")
        print(f"DRY RUN: no Memory calls. Planned ({mode}):")
        if args.clear:
            lines = ["look up the profile by email",
                     "list its conversation summaries and observations",
                     f"delete those whose source is {SOURCE} (any other item is left alone)",
                     f"read back until none is left (up to {args.wait:g} s)"]
        else:
            lines = ["look up the profile by email (it must exist)",
                     "skip any item whose text is already on the profile from this source"]
            lines += [f"summary ({SOURCE}, {s['conversationId']}, {s['occurredAt']}): {s['content']}"
                      for s in summaries]
            lines += [f"observation ({SOURCE}, {o['occurredAt']}): {o['content']}" for o in observations]
            lines.append(f"read back until the items show (up to {args.wait:g} s)")
        for i, line in enumerate(lines, 1):
            print(f"  {i}. {line}")
        print("re-run with --apply to make the change")
        return 0

    try:
        creds = _twilio_credentials(arn, args, sm_client)
        api_key, api_secret = creds["api_key"], creds["api_secret"]
        account_sid = account_sid or creds["account_sid"]
        history = History(client_factory(api_key, api_secret, account_sid), store_id, customer,
                          sleep=sleep, clock=clock)
        if args.clear:
            history.clear(args.wait)
        else:
            history.apply(summaries, observations, args.wait)
    except (ScriptError, memory_http.MemoryApiError) as exc:
        script_error(slog, "memory_history", exc, step=mode, decision="exit_3")
        print(f"memory_history: {exc}", file=sys.stderr)
        return EXIT_MEMORY
    except Exception as exc:  # network and SDK errors: the type only (a message could carry a URL)
        script_error(slog, "memory_history", exc, step=mode, decision="exit_1")
        print(f"memory_history: {type(exc).__name__}", file=sys.stderr)
        return EXIT_UNEXPECTED
    script_step(slog, "memory_history", mode, "changed")
    return 0


if __name__ == "__main__":
    start_logging("memory_history")
    sys.exit(main())
