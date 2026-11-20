"""Demo trait writer: put web activity on a demo customer's Memory profile, check it, take it off.

In production a feed such as Segment writes the ``WebActivity`` traits; this script stands in for it
so the demo and the tests need no Segment workspace. It writes the same group and trait names a
Segment mapping would (``owlbank.web_activity.TRAITS``), so connecting Segment changes the writer
only.

Usage (from the repo root)::

    uv run python scripts/web_activity.py                          # plan for C-1001, nothing written
    uv run python scripts/web_activity.py --apply                  # HELOC, viewed now, 3 views
    uv run python scripts/web_activity.py --product "auto loan" --views 1 --apply
    uv run python scripts/web_activity.py --check                  # what the bridge would retrieve
    uv run python scripts/web_activity.py --clear --apply          # back to "no web activity"

- **Customer:** ``--customer`` (default ``realNumberPlaceholders.developerPhone.fixture`` in
  ``data/fixtures.json``, C-1001): fixtures only. The profile is found by the fixture's
  ``memory_email``, the identifier the bridge uses, so it is the one a verified call reaches,
  whichever of the two created it.
- **Store:** ``memoryStoreId`` in ``twilio_setup.py``'s output file (``--setup-output``).
- **Credentials:** the Twilio API key secret from Secrets Manager (``--cdk-outputs`` or
  ``--secret twilio_api_key=ARN``; ``--region``). No
  credential is printed or logged.
- **Stage:** ``dev`` (default) or ``test``; ``prod`` is refused. The stage is a guard: the store
  comes from the setup output.
- ``--apply`` makes sure the trait group exists, then PATCHes the profile, or upserts
  ``{"Contact": {"email": ...}, "WebActivity": {...}}`` when there is none (the bridge's
  ``create_profile`` call with the traits added). Memory queues trait writes, so the script reads the
  profile back for up to ``--wait`` seconds and says how long the traits took to show. A profile
  that already holds the same values is left alone, so a re-run with the same ``--viewed-at`` writes
  nothing; without it ``lastViewedAt`` is now, so a re-run refreshes it on the same profile.
- ``--clear`` sets every ``WebActivity`` trait to ``null`` (the three and any other a feed added) and
  reads back until the group is gone or empty.
- ``--check`` is read-only: it fetches the profile's traits and runs the bridge's own filter
  (``bound_profile_traits``) and TAC's ``MemoryPromptBuilder``, as ``TACMemoryClient.retrieve_prompt``
  does, with the deployed allow-list and window (``--trait-groups``, ``--max-age-hours``; the bridge's
  defaults unless the deployment overrides them). It prints the profile part of the recall the agent
  would get, or why ``WebActivity`` doesn't come back. Observations and summaries of the full recall
  are not shown.
- A 308 (the profile was merged into another) re-looks the profile up by email and retries once; two
  profiles for one email is an error, never a guess.

Exit codes: 0 done (``--check``: web activity reaches the agent); 1 an unexpected error (its type
only); 2 a bad option or missing setup; 3 a Memory or Secrets Manager error, or the read-back timed out;
4 ``--check``: no web activity would reach the agent (the reason is printed).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owlbank.web_activity import (
    DEFAULT_MAX_AGE_HOURS,
    DEFAULT_TRAIT_GROUPS,
    FRESHNESS_TRAIT,
    FUTURE_SKEW,
    GROUP,
    MAX_AGE_HOURS_RANGE,
    MAX_TRAIT_GROUPS,
    TRAIT_GROUP_NAME_RE,
    TRAITS,
    _parse_time,
    bound_profile_traits,
)
from scripts._common import (
    DATA_DIR,
    REPO_ROOT,
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
from scripts._memory_traits import (
    MemoryApiError,
    ensure_trait_group,
    get_profile_traits,
    lookup_profile,
    patch_traits,
    upsert_profile,
)
from tac_bridge.bridge import MEMORY_ID_TYPE, MEMORY_IDENTIFIER_FIELD, MEMORY_TRAIT_GROUP

slog = logging.getLogger("owlbank.web_activity_script")

#: The first line of every run.
BANNER = ("Demo data: web-activity traits are written to Memory directly "
          "(in production a feed such as Segment writes them)")
STAGES = ("dev", "test")
DEFAULT_PRODUCT = "HELOC"  # demo data only: no code or prompt depends on it
DEFAULT_VIEWS = 3
DEFAULT_WAIT_S = 30
READ_POLL_S = 1.0
DEFAULT_SETUP_OUTPUT = REPO_ROOT / "infra" / "twilio.context.json"
#: ``--product`` longer than this is refused (the bridge would cut it to MAX_VALUE_CHARS anyway).
MAX_PRODUCT_CHARS = 500
CLEARED: dict[str, Any] = dict.fromkeys(TRAITS)

EXIT_UNEXPECTED, EXIT_USAGE, EXIT_MEMORY, EXIT_NOT_RETRIEVED = 1, 2, 3, 4

T = TypeVar("T")


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    email: str


# --------------------------------------------------------------------------- options


def resolve_customer(fixtures: Mapping[str, Any], customer_id: str | None) -> Customer:
    """The fixture customer (default: the developer phone's fixture); anything else is refused."""
    if customer_id is None:
        dev = fixtures.get("realNumberPlaceholders", {}).get("developerPhone", {}).get("fixture")
        if not isinstance(dev, str) or not dev:
            raise ScriptError("no fixture is assigned to the developer phone (realNumberPlaceholders); "
                              "pass --customer")
        customer_id = dev
    for c in fixtures.get("customers", []):
        if isinstance(c, dict) and c.get("customer_id") == customer_id:
            email = c.get(MEMORY_IDENTIFIER_FIELD)
            if not isinstance(email, str) or not email:
                raise ScriptError(f"fixture {customer_id} has no {MEMORY_IDENTIFIER_FIELD}")
            return Customer(customer_id, f"{c.get('first_name', '')} {c.get('last_name', '')}".strip(), email)
    raise ScriptError(f"--customer {customer_id[:40]!r} is not a customer in data/fixtures.json; only "
                      "synthetic fixture customers are written")


def parse_viewed_at(text: str | None, now: datetime) -> str:
    """``lastViewedAt`` as ``YYYY-MM-DDTHH:MM:SSZ``: now, or ``--viewed-at`` read as the bridge reads it
    (an ISO-8601 date and time; no offset means UTC). Old or future values are allowed on purpose:
    ``--check`` then shows the bridge dropping them."""
    if text is None:
        at = now
    else:
        parsed = _parse_time(text.strip())
        if parsed is None:
            raise ScriptError("--viewed-at must be an ISO-8601 date and time, e.g. 2026-10-28T14:05:00Z")
        at = parsed
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def stage_guard(stage: str) -> None:
    if stage.strip().lower() == "prod":
        raise ScriptError("--stage prod is refused: web_activity.py writes demo data to dev and test only")
    if stage not in STAGES:
        raise ScriptError(f"--stage must be one of {', '.join(STAGES)}")


def read_store_id(path: Path) -> tuple[str, str | None]:
    """``(memoryStoreId, accountSid or None)`` from ``twilio_setup.py``'s output file."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScriptError(f"can't read the setup output {path.name}: {type(exc).__name__}; run "
                          "twilio_setup.py --apply first, or pass --setup-output") from None
    store = doc.get("memoryStoreId") if isinstance(doc, dict) else None
    if not isinstance(store, str) or not store:
        raise ScriptError(f"no memoryStoreId in {path.name}; run twilio_setup.py --apply first")
    account = doc.get("accountSid")
    return store, account if isinstance(account, str) and account else None


def trait_groups(text: str) -> list[str]:
    """The ``--trait-groups`` allow-list, refused where the bridge would refuse it at start-up."""
    groups = [g.strip() for g in text.split(",") if g.strip()]
    if (not 1 <= len(groups) <= MAX_TRAIT_GROUPS or len(set(groups)) != len(groups)
            or not all(TRAIT_GROUP_NAME_RE.fullmatch(g) for g in groups)):
        raise ScriptError(f"--trait-groups must list 1 to {MAX_TRAIT_GROUPS} distinct group names matching "
                          f"{TRAIT_GROUP_NAME_RE.pattern}")
    return groups


def web_activity(traits: Mapping[str, Any]) -> dict[str, Any]:
    """The profile's ``WebActivity`` values, without the removed (``null``) ones."""
    group = traits.get(GROUP)
    return {k: v for k, v in group.items() if v is not None} if isinstance(group, Mapping) else {}


def holds(current: Mapping[str, Any], want: Mapping[str, Any]) -> bool:
    """Whether the group already holds ``want`` for the traits the script writes (another trait in
    the group, such as one a feed added, doesn't count)."""
    return {k: current.get(k) for k in want} == dict(want)


def _show(values: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in values.items())


# --------------------------------------------------------------------------- Memory


class Writer:
    """The Memory side of one run, over a Twilio client (only its HTTP layer and API key are used)."""

    def __init__(self, client: Any, store_id: str, customer: Customer, *,
                 log: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.client, self.store_id, self.customer = client, store_id, customer
        self.log, self.sleep, self.clock = log, sleep, clock

    def profile_ids(self) -> list[str]:
        ids = lookup_profile(self.client, self.store_id, self.customer.email)
        if len(ids) > 1:
            raise ScriptError(f"{len(ids)} profiles share {self.customer.email}; merge or delete the "
                              "extra ones in the Console (the script never picks one)")
        return ids

    def on_profile(self, profile_id: str, op: Callable[[str], T]) -> tuple[str, T]:
        """``op(profile_id)``; on a 308 (merged profile) look the profile up again and retry once."""
        try:
            return profile_id, op(profile_id)
        except MemoryApiError as exc:
            if exc.status != 308:
                raise
        ids = self.profile_ids()
        if not ids:
            raise ScriptError(f"profile for {self.customer.email} was merged and the lookup finds none now")
        self.log(f"note    profile was merged; using {ids[0]}")
        return ids[0], op(ids[0])

    def read(self, profile_id: str) -> tuple[str, dict[str, Any]]:
        return self.on_profile(profile_id, lambda pid: get_profile_traits(self.client, self.store_id, pid))

    def wait_until(self, profile_id: str, done: Callable[[dict[str, Any]], bool], wait_s: float,
                   what: str, *, created: bool = False) -> float:
        """Read the profile until ``done(web_activity)``; returns the seconds it took. With
        ``created`` (just upserted), a 404 counts as not visible yet: Memory may queue the create too."""
        start = self.clock()
        while True:
            try:
                profile_id, traits = self.read(profile_id)
            except MemoryApiError as exc:
                if not (created and exc.status == 404):
                    raise
                traits = {}
            elapsed = self.clock() - start
            if done(web_activity(traits)):
                return elapsed
            if elapsed >= wait_s:
                raise ScriptError(f"{what} not visible after {wait_s:g} s (Memory queues trait writes); run "
                                  "--check in a minute, or re-run the same command (it is safe to repeat)")
            self.sleep(READ_POLL_S)

    def apply(self, want: Mapping[str, Any], wait_s: float) -> None:
        ensure_trait_group(self.client, self.store_id, log=self.log, sleep=self.sleep, clock=self.clock)
        ids = self.profile_ids()
        if ids:
            profile_id, current = self.read(ids[0])
            if holds(web_activity(current), want):
                self.log(f"ok      {GROUP} on profile {profile_id} already holds these values; nothing written")
                return
            profile_id, _ = self.on_profile(
                profile_id, lambda pid: patch_traits(self.client, self.store_id, pid, {GROUP: want}))
            self.log(f"update  {GROUP} on profile {profile_id}")
        else:
            profile_id = upsert_profile(self.client, self.store_id,
                                        {MEMORY_TRAIT_GROUP: {MEMORY_ID_TYPE: self.customer.email}, GROUP: want})
            self.log(f"create  profile {profile_id} for {self.customer.email} with {GROUP}")
        took = self.wait_until(profile_id, lambda wa: holds(wa, want), wait_s, f"{GROUP} traits",
                               created=not ids)
        self.log(f"ok      {GROUP} traits visible after {took:.1f} s")

    def clear(self, wait_s: float) -> None:
        ids = self.profile_ids()
        if not ids:
            self.log(f"ok      no profile for {self.customer.email}; nothing to clear")
            return
        profile_id, current = self.read(ids[0])
        present = web_activity(current)
        if not present:
            self.log(f"ok      profile {profile_id} has no {GROUP} traits; nothing to clear")
            return
        # Every trait of the group, not only the three the script writes: "no web activity".
        cleared = {**CLEARED, **dict.fromkeys(present)}
        profile_id, _ = self.on_profile(
            profile_id, lambda pid: patch_traits(self.client, self.store_id, pid, {GROUP: cleared}))
        self.log(f"clear   {GROUP} on profile {profile_id}")
        took = self.wait_until(profile_id, lambda wa: not wa, wait_s, f"the cleared {GROUP}")
        self.log(f"ok      {GROUP} gone after {took:.1f} s")

    def check(self, allow: Sequence[str], max_age_hours: int, now: datetime) -> tuple[str | None, str]:
        """``(recall text, reason)``: the text when ``WebActivity`` comes back, else ``None`` and why."""
        ids = self.profile_ids()
        if not ids:
            return None, (f"no profile for {self.customer.email} (the bridge creates it on the first "
                          "verified call; or run --apply)")
        profile_id, traits = self.read(ids[0])
        raw = web_activity(traits)
        if not raw:
            return None, (f"profile {profile_id} has no {GROUP} traits (not written, cleared, or not visible "
                          "yet: Memory queues writes)")
        if GROUP not in allow:
            return None, f"{GROUP} is not on the trait-group allow-list ({', '.join(allow)})"
        kept = bound_profile_traits(traits, allow=allow, now=now, max_age_hours=max_age_hours)
        if GROUP not in kept:
            return None, _dropped_reason(raw, now, max_age_hours)
        return render_recall(profile_id, kept, allow), f"{GROUP} reaches the agent: {', '.join(kept[GROUP])}"


def _dropped_reason(raw: Mapping[str, Any], now: datetime, max_age_hours: int) -> str:
    """Why the bridge's filter dropped a ``WebActivity`` group that is on the allow-list."""
    at = _parse_time(raw.get(FRESHNESS_TRAIT))
    if at is None:
        return f"{FRESHNESS_TRAIT} is missing or not an ISO-8601 date and time; the bridge drops the group"
    if at > now + FUTURE_SKEW:
        return f"{FRESHNESS_TRAIT} is in the future ({raw.get(FRESHNESS_TRAIT)}); the bridge drops the group"
    if at < now - timedelta(hours=max_age_hours):
        return (f"too old: {FRESHNESS_TRAIT} {raw.get(FRESHNESS_TRAIT)} is more than {max_age_hours} h ago; "
                "re-run --apply on the day of the call")
    return f"no {GROUP} value is usable after the bridge's filter"


def render_recall(profile_id: str, kept: Mapping[str, Any], allow: Sequence[str]) -> str | None:
    """The profile part of the recall, rendered by TAC's builder as ``TACMemoryClient.retrieve_prompt``
    does after the filter (no observations or summaries)."""
    from tac.adapters import AdapterOptions, MemoryPromptBuilder  # noqa: PLC0415
    from tac.models.memory import ProfileResponse  # noqa: PLC0415
    from tac.models.session import ConversationSession  # noqa: PLC0415

    session = ConversationSession(conversation_id="web-activity-check", call_sid=None, channel="VOICE",
                                  profile_id=profile_id, author_info=None, ai_agent_info=None,
                                  profile=ProfileResponse(id=profile_id, createdAt="", traits=dict(kept)))
    return MemoryPromptBuilder.build(None, session, AdapterOptions(profile_traits=list(allow)))


# --------------------------------------------------------------------------- CLI


def default_client_factory(api_key: str, api_secret: str, account_sid: str | None) -> Any:
    from twilio.rest import Client  # noqa: PLC0415

    return Client(api_key, api_secret, account_sid)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write, check or clear demo web-activity traits on a fixture "
                                            "customer's Twilio Memory profile (demo data; Segment's job in "
                                            "production).")
    add_mode_args(p)
    add_secret_args(p)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--clear", action="store_true", help="remove the WebActivity traits (with --apply)")
    action.add_argument("--check", action="store_true",
                        help="read-only: show what the bridge would retrieve, or why nothing comes back")
    p.add_argument("--customer", help="fixture customer ID (default: the developer phone's fixture, C-1001)")
    p.add_argument("--product", default=DEFAULT_PRODUCT, help="lastViewedProduct (default HELOC; any text)")
    p.add_argument("--viewed-at", help="lastViewedAt, ISO-8601 date and time (default now, UTC)")
    p.add_argument("--views", type=int, default=DEFAULT_VIEWS, help="viewCount (default 3)")
    p.add_argument("--stage", default="dev", help="dev (default) or test; prod is refused")
    p.add_argument("--wait", type=float, default=DEFAULT_WAIT_S,
                   help="seconds to wait for the written traits to show (default 30)")
    p.add_argument("--trait-groups", default=",".join(DEFAULT_TRAIT_GROUPS),
                   help="--check: the deployed allow-list (default Contact,WebActivity)")
    p.add_argument("--max-age-hours", type=int, default=DEFAULT_MAX_AGE_HOURS,
                   help="--check: the deployed web-activity window (default 24)")
    p.add_argument("--setup-output", type=Path, default=DEFAULT_SETUP_OUTPUT,
                   help="twilio_setup.py output file holding memoryStoreId (default infra/twilio.context.json)")
    return p


def _plan_lines(args: argparse.Namespace, want: Mapping[str, Any], wait_s: float) -> list[str]:
    if args.clear:
        return ["look up the profile by email",
                f"PATCH every {GROUP} trait to null ({', '.join(TRAITS)} and any other; only if it holds one)",
                f"read back until {GROUP} is gone or empty (up to {wait_s:g} s)"]
    return [f"ensure trait group {GROUP} (create it if missing)",
            "look up the profile by email",
            f"PATCH {GROUP} on it (nothing if it already holds these values), or, with no profile, upsert "
            f"{{{MEMORY_TRAIT_GROUP}: {{email}}, {GROUP}: ...}}",
            f"read back until the traits show (up to {wait_s:g} s)",
            f"values: {_show(want)}"]


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
    mode = "check" if args.check else ("clear" if args.clear else "write")
    script_start(slog, "web_activity", mode=mode, apply=bool(args.apply), stage=args.stage)
    try:
        stage_guard(args.stage)
        if args.check and args.apply:
            raise ScriptError("--check is read-only; run it without --apply")
        if args.wait < 0 or args.views < 0:
            raise ScriptError("--wait and --views must not be negative")
        product = args.product.strip()
        if not product or len(product) > MAX_PRODUCT_CHARS:
            raise ScriptError(f"--product must be 1 to {MAX_PRODUCT_CHARS} characters")
        allow = trait_groups(args.trait_groups)
        low, high = MAX_AGE_HOURS_RANGE
        if not low <= args.max_age_hours <= high:
            raise ScriptError(f"--max-age-hours must be an integer from {low} to {high}")
        try:
            fixtures = json.loads((DATA_DIR / "fixtures.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ScriptError(f"can't read data/fixtures.json: {type(exc).__name__}") from None
        customer = resolve_customer(fixtures, args.customer)
        store_id, account_sid = read_store_id(args.setup_output)
        arn = _api_key_arn(args) if args.apply or args.check else ""
        want = {"lastViewedProduct": product, "lastViewedAt": parse_viewed_at(args.viewed_at, now()),
                "viewCount": args.views}
    except ScriptError as exc:
        script_error(slog, "web_activity", exc, step="plan", decision="exit_2")
        print(f"web_activity: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(f"stage {args.stage}, store {store_id}")
    print(f"customer {customer.customer_id} ({customer.name}), profile email {customer.email}")
    print(f"group {GROUP}, traits {', '.join(TRAITS)}")
    script_step(slog, "web_activity", "plan", "ok", customer=customer.customer_id)

    if not (args.apply or args.check):
        script_step(slog, "web_activity", mode, "dry_run")
        print(f"DRY RUN: no Memory calls. Planned ({'clear' if args.clear else 'write'}):")
        for i, line in enumerate(_plan_lines(args, want, args.wait), 1):
            print(f"  {i}. {line}")
        print("re-run with --apply to make the change")
        return 0

    try:
        creds = _twilio_credentials(arn, args, sm_client)
        api_key, api_secret = creds["api_key"], creds["api_secret"]
        account_sid = account_sid or creds["account_sid"]
        writer = Writer(client_factory(api_key, api_secret, account_sid), store_id, customer,
                        sleep=sleep, clock=clock)
        if args.check:
            print(f"check with allow-list {','.join(allow)} and window {args.max_age_hours} h "
                  "(the profile traits of the recall; observations and summaries are not shown)")
            text, reason = writer.check(allow, args.max_age_hours, now())
            script_step(slog, "web_activity", "check", "ok", retrieved=text is not None)
            if text is None:
                print(f"no {GROUP} comes back: {reason}")
                return EXIT_NOT_RETRIEVED
            print(reason)
            print("recall text the agent would get:")
            print(text)
            return 0
        if args.clear:
            writer.clear(args.wait)
        else:
            writer.apply(want, args.wait)
    except ScriptError as exc:
        script_error(slog, "web_activity", exc, step=mode, decision="exit_3")
        print(f"web_activity: {exc}", file=sys.stderr)
        return EXIT_MEMORY
    except Exception as exc:  # network and SDK errors: the type only (a message could carry a URL)
        script_error(slog, "web_activity", exc, step=mode, decision="exit_1")
        print(f"web_activity: {type(exc).__name__}", file=sys.stderr)
        return EXIT_UNEXPECTED
    script_step(slog, "web_activity", mode, "changed")
    return 0


if __name__ == "__main__":
    start_logging("web_activity")
    sys.exit(main())
