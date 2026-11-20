"""Seed the demo data: fixtures, the auth policy and the prompt catalog.

Usage::

    # dry run (the default): print every item, factor hashes and real numbers masked
    uv run python scripts/seed.py --stage dev --cdk-outputs cdk-outputs.json \\
        --developer-phone +1... --test-number A=+1... --test-number B=+1...

    # write to DynamoDB
    uv run python scripts/seed.py --stage dev --cdk-outputs cdk-outputs.json --apply

    # test stack: restore every fixture and move test number A to María before a call
    uv run python scripts/seed.py --stage test --cdk-outputs cdk-outputs.json \\
        --test-number A=+1... --reseed --assign A=C-1004 --apply

Rules:

* **Fictional numbers only in data files**. Every phone-like value in
  ``fixtures.json``, ``policy.json`` and ``prompt_catalog.json`` must be in
  555-0100..0199; otherwise the script refuses before doing anything.
* **Real, owned numbers come only from arguments**: ``--developer-phone``
  and ``--test-number LABEL=+E164``, and only for the ``dev`` and ``test``
  stages. Each becomes an *ANI alias* record in ``customers``::

      {"customer_id": "ani#A#C-1001", "phone": "+1...", "alias_of": "C-1001",
       "kind": "ani_alias", "label": "A"}

  The customers table has one scalar ``phone`` per record (``phone-index``), so an
  extra ANI for a fixture is a separate record; the ANI lookup resolves
  ``alias_of`` to the fixture. Label ``D`` (and ``--assign X=none``) maps to no
  fixture: the number stays unseeded (the "no match" case).
* **Seed vs reseed**. A plain seed writes fixtures only if absent (so it
  doesn't undo a demo's balance changes) and always rewrites policy, catalog and
  aliases. ``--reseed`` restores every fixture, deletes every known alias record,
  then writes the aliases again, optionally with ``--assign LABEL=FIXTURE[+FIXTURE]``
  (test stage only). Calls on the same test number must then run one at a time.
  Tokens, call state and segments expire by TTL and aren't touched.
* **Tuned masking values survive**. Rewriting the policy keeps the
  latency-masking values already in the table: every field of the ``masking`` item named in
  :data:`owlbank.masking_config.MASKING_FIELDS` (present or absent, as stored) and each tool
  item's ``mask``. Every value that differs from ``policy.json`` is printed (field, live, seed
  file), in a dry run and an apply. ``--overwrite-masking`` writes the seed file's values
  instead. A tool or a ``masking`` item not yet in the table gets the seed file's values; a
  live value that fails validation is kept with a warning (``--overwrite-masking`` fixes it).
  An item's ``rev`` (optimistic-concurrency counter, ``scripts/masking.py``) is carried over and
  bumped, so an editor holding the old ``rev`` is refused. To see differences in a dry run,
  name the policy table (``--cdk-outputs`` or ``--table policy=NAME``): it is read, not written.
  The live values are read just before the policy is written (not atomically: don't run a seed
  and an edit at the same moment). Tune single values with ``scripts/masking.py``.
* **Tuned slot settings survive**. For each seeded
  ``slot#<name>`` item already in the table, every field the live item holds is kept exactly
  (``sensitive``, ``keypad_only``, ``barge_in``, a non-factor slot's shape and ``flow``, and any
  other field); only a field it lacks is written from ``policy.json``. A slot item not yet in the
  table gets the seed file's values. Every live value that differs from the seed file, every
  filled field and every new item is printed, in a dry run and an apply; a kept live value that
  fails validation is kept with a warning. ``rev`` is carried over and bumped (1 for a new item).
  ``meta`` (its ``slots`` list included) is always the seed file's. ``--overwrite-slots`` writes
  the seed file's slot items instead, logged at WARNING (``script_step``, ``outcome=overwrite``).
  Until a factor has its item, the policy loader treats it as sensitive.
  The ``slot_resolution`` item is merged the same way, so a phase 1
  table gains ``slot_resolution``, ``slot#card`` and the new slot fields (``mode``, ``confirm``,
  ``options``) without losing a tuned value.
* **Prompt edits survive**. Every catalog item the seed writes
  carries ``seed_hash``, a hash of the text it wrote (the admin API copies it on a console edit).
  An entry whose live text no longer matches its ``seed_hash`` (an edit in the console) and differs
  from ``prompt_catalog.json`` is not written; it is printed (key and language, never the text), in
  a dry run and an apply. An entry whose live text still matches its ``seed_hash`` gets the seed
  file's text when that changed. An entry without a ``seed_hash`` (seeded before the hash existed) whose
  text differs is kept and listed as legacy, until one ``--overwrite-prompts``. That flag writes
  the seed file's text over every differing entry and prints each one it replaced
  (``--overwrite-slots`` does the same for ``option.*`` entries only). A key or language not yet
  in the table is always written. An entry
  whose text and other fields match is not written, so its ``rev`` (the console's optimistic-
  concurrency counter) is unchanged and an open editor gets no false conflict; an entry written
  over a live one (or a language new to a live key) gets the key's view ``rev`` (the highest over
  its languages, as the console shows it) plus one. The meta item's ``keys`` index keeps every key
  added in the console that the seed file lacks: no language of it carries a ``seed_hash``
  (``--overwrite-prompts`` drops them from the index; their items stay, unserved). A key the seed
  wrote and its file then dropped leaves the index. Before anything is written the seed
  checks that every option in ``policy.json`` has its catalog entry in every catalog language and
  that no spoken name or alias belongs to two options (``alias_conflicts``); else it exits 2.
  Likewise every offer in ``policy.json``'s ``web_offers`` needs its catalog phrase ``offer.<id>``
  in every catalog language (``check_offers``); else it exits 2.
  Likewise the escalation briefing's spoken labels: every policy level (0 to ``max_level``) needs
  ``escalate.briefing.level.<n>`` and every escalation reason ``escalate.briefing.reason.<key>``
  (plus ``escalate.briefing.reason.default``) in every catalog language, with no digit in any of
  them (``check_briefing_labels``); else it exits 2.
  A ``model_failover`` section out of range is refused the same way, by
  :meth:`owlbank.policy.Policy.to_items`.
* **The stored redaction setting survives**. The ``redaction`` item is merged per field
  like a slot item, with two rules for a phase 1 item: a field the schema no longer has
  (``apply_to``, or any unknown field) is dropped, and a live empty ``entity_types`` or empty
  ``apply_to`` (phase 1's "off") becomes ``enabled: false`` with the seed file's types, so an
  operator's "off" stays off.
  Both are printed. ``--overwrite-redaction`` writes the seed file's setting, logged at WARNING.
* **Per-model prompts are validated**. Every published ``agent.system.*``
  text needs a record in ``prompt_validation.json`` (next to the catalog) with its current prompt
  hash, the current ``prompt_suite.json`` hash and full runs of every case (the entry parameters'
  hash is preflight's to compare: the seed doesn't know the deployed list); else the seed refuses (exit 2) and names each
  text. This is stricter than "a model in the deployed list" (the seed doesn't know the list, and a
  prompt for any model is only written to be used). ``--allow-unvalidated-prompt`` seeds them anyway,
  only with ``--stage dev`` (refused at any other stage); the plan then lists each unvalidated text
  and the run logs them at WARNING.
* **Agent texts keep the placeholder allow-list**. Before
  that gate, every published ``agent.*`` text is checked against
  ``owlbank.agent_prompt.placeholder_problems`` and both caller-context frames must be published in
  every catalog language; else the seed refuses (exit 2) naming each key and language. No flag
  overrides it.

Table names come from ``--table LOGICAL=NAME`` or the CDK outputs file
(``Table<Logical>``, ``<Logical>TableName``); unresolved tables fall back to the
library default ``owlbank-<logical>``, except that ``--apply`` requires the three
seeded tables (customers, policy, catalog) to be named explicitly.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a file: make `scripts` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owlbank import obs
from owlbank.agent_prompt import AGENT_PREFIX, CALLER_CONTEXT_KEYS, SYSTEM_KEY_PREFIX, placeholder_problems
from owlbank.catalog import META_KEY, META_LANG, PromptCatalog
from owlbank.config import TABLE_NAMES
from owlbank.masking_config import ARG_PREFIX, MASK_PREFIX, MASKING_FIELDS, masking_parts, tool_mask_parts
from owlbank.pii import RedactionConfig, RedactionConfigError, parse_redaction_config
from owlbank.policy import Policy, PolicyError
from owlbank.prompt_validation import Suite, ValidationFile, load_records, load_suite
from owlbank.slot_policy import (
    RESOLUTION_ITEM,
    SlotPolicyError,
    alias_conflicts,
    load_slot,
    load_slot_resolution,
    parse_slot,
)
from owlbank.store import ConditionFailed, DynamoStore, Store
from scripts._common import (
    DATA_DIR,
    E164,
    FICTIONAL_PHONE,
    PHONE_LIKE,
    ScriptError,
    add_mode_args,
    find_output,
    load_cdk_outputs,
    mask_phone,
    parse_pairs,
    resolve_region,
    script_error,
    script_start,
    script_step,
    start_logging,
    walk,
)
from scripts.hash_tool import hash_factor

log = logging.getLogger("owlbank.seed")

#: Stages where real, owned numbers may be seeded.
REAL_NUMBER_STAGES = frozenset({"dev", "test"})
#: Stages where ``--assign`` may move a test number between fixtures.
ASSIGN_STAGES = frozenset({"test"})
SEEDED_TABLES = ("customers", "policy", "catalog")
DEV_LABEL = "dev"
ALIAS_KIND = "ani_alias"
MASK = "***"


@dataclass
class SeedPlan:
    """Everything a seed run writes and deletes, by logical table."""

    customers: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[dict[str, Any]] = field(default_factory=list)
    policy: list[dict[str, Any]] = field(default_factory=list)
    catalog: list[dict[str, Any]] = field(default_factory=list)
    delete_aliases: list[str] = field(default_factory=list)
    reseed: bool = False


# --------------------------------------------------------------------------- data checks


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScriptError(f"can't read {path.name}: {type(exc).__name__}") from None


def _digits_e164(text: str) -> str:
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 10:
        digits = "1" + digits
    return "+" + digits


#: An e-mail address inside a string value.
EMAIL_LIKE = re.compile(r"[^@\s\"',;<>]+@([^@\s\"',;<>]+)")
#: Domains and top-level names RFC 2606 / RFC 6761 reserve for documentation and tests.
_RESERVED_TLDS = frozenset({"example", "invalid", "test", "localhost"})
_RESERVED_DOMAINS = frozenset({"example.com", "example.net", "example.org"})


def is_reserved_email_domain(domain: str) -> bool:
    """``True`` for a domain that can never belong to a real mailbox owner."""
    host = domain.strip().strip(".").lower()
    return (host in _RESERVED_DOMAINS or host.rsplit(".", 1)[-1] in _RESERVED_TLDS
            or any(host.endswith("." + d) for d in _RESERVED_DOMAINS))


def check_fictional(name: str, doc: Any) -> None:
    """Refuse any phone in a data file outside 555-0100..0199 (``anonymous`` is allowed) and any
    e-mail address outside a reserved domain (``.example``, ``.invalid``, ``example.com``...)."""
    bad: list[str] = []
    bad_mail: list[str] = []
    for path, key, value in walk(doc):
        if not isinstance(value, str):
            continue
        if key == "memory_email" and not EMAIL_LIKE.fullmatch(value):
            bad_mail.append(path)
            continue
        if any(not is_reserved_email_domain(m.group(1)) for m in EMAIL_LIKE.finditer(value)):
            bad_mail.append(path)
            continue
        if key in ("phone", "ani"):
            if value != "anonymous" and not FICTIONAL_PHONE.match(value):
                bad.append(path)
            continue
        for m in PHONE_LIKE.findall(value):
            if not FICTIONAL_PHONE.match(_digits_e164(m)):
                bad.append(path)
    if bad_mail:
        raise ScriptError(
            f"{name}: e-mail addresses outside a reserved domain (.example, .invalid, example.com...) "
            f"at {', '.join(sorted(set(bad_mail)))}. Fixtures hold fictional addresses only."
        )
    if bad:
        raise ScriptError(
            f"{name}: phone numbers outside the fictional 555-0100..0199 range at "
            f"{', '.join(sorted(set(bad)))}. Real numbers go in --developer-phone / --test-number only."
        )


# --------------------------------------------------------------------------- real numbers


def _alias_id(label: str, fixture_id: str) -> str:
    return f"ani#{label}#{fixture_id}"


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def resolve_real_numbers(
    fixtures: Mapping[str, Any],
    *,
    stage: str,
    developer_phone: str | None,
    test_numbers: Mapping[str, str],
    assignments: Mapping[str, str],
    reseed: bool,
) -> list[tuple[str, str, list[str]]]:
    """``[(label, phone, [fixture ids])]`` for every real number given, after the guards."""
    if (developer_phone or test_numbers) and stage not in REAL_NUMBER_STAGES:
        raise ScriptError(
            f"real phone numbers may be seeded only in {sorted(REAL_NUMBER_STAGES)} stages, "
            f"not {stage!r}"
        )
    if assignments and not reseed:
        raise ScriptError("--assign needs --reseed")
    if assignments and stage not in ASSIGN_STAGES:
        raise ScriptError(f"--assign is allowed only in the {sorted(ASSIGN_STAGES)} stage")

    placeholders = fixtures.get("realNumberPlaceholders", {})
    defaults: dict[str, list[str]] = {
        label: _as_list(spec.get("fixture"))
        for label, spec in placeholders.get("testNumbers", {}).items()
    }
    defaults[DEV_LABEL] = _as_list(placeholders.get("developerPhone", {}).get("fixture"))
    known_ids = {c["customer_id"] for c in fixtures.get("customers", [])}

    numbers: dict[str, str] = dict(test_numbers)
    if developer_phone:
        numbers[DEV_LABEL] = developer_phone
    for label, phone in numbers.items():
        if label not in defaults:
            raise ScriptError(f"unknown test-number label {label!r}; known: {sorted(defaults)}")
        if not E164.match(phone):
            raise ScriptError(f"number for {label!r} is not E.164 ({mask_phone(phone)})")
    seen: dict[str, str] = {}
    for label, phone in numbers.items():
        if phone in seen:
            raise ScriptError(f"labels {seen[phone]!r} and {label!r} share one number")
        seen[phone] = label

    mapping = {label: list(ids) for label, ids in defaults.items()}
    for label, target in assignments.items():
        if label not in numbers:
            raise ScriptError(f"--assign {label}=...: no number given for {label!r}")
        ids = [] if target.lower() == "none" else [t.strip() for t in target.split("+") if t.strip()]
        unknown = [i for i in ids if i not in known_ids]
        if unknown:
            raise ScriptError(f"--assign {label}: unknown fixture {unknown}")
        mapping[label] = ids
    return [(label, phone, mapping[label]) for label, phone in sorted(numbers.items())]


# --------------------------------------------------------------------------- plan


def build_plan(
    fixtures: Mapping[str, Any],
    policy_doc: Mapping[str, Any],
    catalog_doc: Mapping[str, Any],
    *,
    stage: str,
    developer_phone: str | None = None,
    test_numbers: Mapping[str, str] | None = None,
    assignments: Mapping[str, str] | None = None,
    reseed: bool = False,
    developer_factors: Mapping[str, str] | None = None,
) -> SeedPlan:
    """Validate the data and argument numbers, and return the items to write.

    ``developer_factors`` (dev and test only): factor values the developer wants for the fixture
    their phone is assigned to (``realNumberPlaceholders.developerPhone.fixture``), for example a
    ZIP they can remember. Like real phone numbers they come only from arguments, never from the
    repository; they are hashed here and never printed."""
    for name, doc in (("fixtures.json", fixtures), ("policy.json", policy_doc),
                      ("prompt_catalog.json", catalog_doc)):
        check_fictional(name, doc)
    real = resolve_real_numbers(
        fixtures, stage=stage, developer_phone=developer_phone,
        test_numbers=test_numbers or {}, assignments=assignments or {}, reseed=reseed,
    )
    customers = [copy.deepcopy(c) for c in fixtures.get("customers", [])]
    ids = [c["customer_id"] for c in customers]
    if len(set(ids)) != len(ids):
        raise ScriptError("fixtures.json: duplicate customer_id")

    if developer_factors:
        _apply_developer_factors(customers, fixtures, policy_doc, developer_factors, stage=stage)
    plan = SeedPlan(customers=customers, reseed=reseed)
    plan.policy = Policy.to_items(policy_doc)
    plan.catalog = PromptCatalog.to_items(catalog_doc)
    check_options(policy_doc, catalog_doc)
    check_offers(policy_doc, catalog_doc)
    check_briefing_labels(policy_doc, catalog_doc)
    for label, phone, fixture_ids in real:
        for fid in fixture_ids:
            plan.aliases.append({"customer_id": _alias_id(label, fid), "phone": phone,
                                 "alias_of": fid, "kind": ALIAS_KIND, "label": label})
    if reseed:
        labels = sorted({*fixtures.get("realNumberPlaceholders", {}).get("testNumbers", {}),
                         DEV_LABEL})
        plan.delete_aliases = [_alias_id(lb, fid) for lb in labels for fid in ids]
    return plan


def _apply_developer_factors(customers: list[dict[str, Any]], fixtures: Mapping[str, Any],
                             policy_doc: Mapping[str, Any], values: Mapping[str, str], *,
                             stage: str) -> None:
    if stage not in REAL_NUMBER_STAGES:
        raise ScriptError("--developer-factor is for dev and test stages only")
    fixture = fixtures.get("realNumberPlaceholders", {}).get("developerPhone", {}).get("fixture")
    target = next((c for c in customers if c.get("customer_id") == fixture), None)
    if target is None:
        raise ScriptError("no fixture is assigned to the developer phone (realNumberPlaceholders)")
    factors = policy_doc.get("factors", {})
    for name, value in values.items():
        spec = factors.get(name)
        if spec is None or spec.get("channel") == "verify_sms":
            raise ScriptError(f"--developer-factor: {name!r} is not a stored factor")
        length = (spec.get("constraints") or {}).get("length")
        if not value.isdigit() or (isinstance(length, int) and len(value) != length):
            raise ScriptError(f"--developer-factor: {name} must be {length} digits")
        target.setdefault("factors", {})[name] = hash_factor(value)


OPTION_PREFIX = "option."


def check_options(policy_doc: Mapping[str, Any], catalog_doc: Mapping[str, Any]) -> None:
    """The seed-time cross-check: every option of every slot in ``policy_doc``
    has its catalog entry ``option.<slot>.<value>`` in every catalog language, and no normalized
    spoken name or alias belongs to two options of one slot in one language. Raises
    :class:`ScriptError` naming each problem."""
    langs = [str(lang) for lang in (catalog_doc.get("languages") or {})]
    texts: dict[tuple[str, str], Any] = {
        (str(e.get("key")), str(e.get("lang"))): e.get("text")
        for e in catalog_doc.get("entries", []) if isinstance(e, Mapping)}
    factors = policy_doc.get("factors") or {}
    missing: list[str] = []
    conflicts: list[str] = []
    for name, raw in (policy_doc.get("slots") or {}).items():
        if not isinstance(raw, Mapping) or not raw.get("options"):
            continue
        try:
            slot = parse_slot(name, raw, factor_def=factors.get(name), langs=langs or None)
        except SlotPolicyError as exc:
            raise ScriptError(f"policy.json: {exc.field}: {exc.message}") from None
        spoken: dict[str, dict[str, str]] = {}
        for o in slot.options:
            key = f"{OPTION_PREFIX}{name}.{o.value}"
            for lang in langs:
                text = texts.get((key, lang))
                if isinstance(text, str) and text.strip():
                    spoken.setdefault(o.value, {})[lang] = text
                else:
                    missing.append(f"{key} ({lang})")
        conflicts.extend(f"slots.{name}: {c.lang} {c.phrase!r} belongs to {', '.join(c.options)}"
                         for c in alias_conflicts(slot.options, spoken, langs))
    if missing:
        raise ScriptError("prompt_catalog.json: no entry for option(s) of policy.json: "
                          + ", ".join(missing))
    if conflicts:
        raise ScriptError("policy.json: a spoken name or alias belongs to two options: "
                          + "; ".join(conflicts))


OFFER_PREFIX = "offer."


def check_offers(policy_doc: Mapping[str, Any], catalog_doc: Mapping[str, Any]) -> None:
    """The seed-time cross-check of the offer list: every offer ID in
    ``policy_doc``'s ``web_offers`` has its catalog entry ``offer.<id>`` in every catalog language.
    A catalog ``offer.*`` entry with no offer is allowed. Raises :class:`ScriptError` naming each
    missing entry. (The list itself is validated by :meth:`Policy.to_items`.)"""
    langs = [str(lang) for lang in (catalog_doc.get("languages") or {})]
    texts: dict[tuple[str, str], Any] = {
        (str(e.get("key")), str(e.get("lang"))): e.get("text")
        for e in catalog_doc.get("entries", []) if isinstance(e, Mapping)}
    section = policy_doc.get("web_offers")
    offers = section.get("offers") if isinstance(section, Mapping) else None
    missing: list[str] = []
    for offer in offers if isinstance(offers, list) else []:
        if not isinstance(offer, Mapping) or not isinstance(offer.get("id"), str):
            continue
        key = f"{OFFER_PREFIX}{offer['id']}"
        for lang in langs:
            text = texts.get((key, lang))
            if not (isinstance(text, str) and text.strip()):
                missing.append(f"{key} ({lang})")
    if missing:
        raise ScriptError("prompt_catalog.json: no entry for offer(s) of policy.json: "
                          + ", ".join(missing))


BRIEFING_PREFIX = "escalate.briefing."
BRIEFING_DEFAULT_REASON = "default"


def check_briefing_labels(policy_doc: Mapping[str, Any], catalog_doc: Mapping[str, Any]) -> None:
    """The seed-time cross-check of the escalation briefing's spoken labels: every policy level
    ``0..max_level`` has ``escalate.briefing.level.<n>`` and every escalation reason of
    ``policy_doc`` has ``escalate.briefing.reason.<key>``, plus ``escalate.briefing.reason.default``
    when there are reasons, in every catalog language; a label is read aloud, so none may hold a
    digit. A policy without levels or reasons needs nothing. Raises :class:`ScriptError` naming
    each problem."""
    langs = [str(lang) for lang in (catalog_doc.get("languages") or {})]
    texts: dict[tuple[str, str], Any] = {
        (str(e.get("key")), str(e.get("lang"))): e.get("text")
        for e in catalog_doc.get("entries", []) if isinstance(e, Mapping)}
    keys: list[str] = []
    max_level = policy_doc.get("max_level")
    if isinstance(max_level, int) and not isinstance(max_level, bool) and max_level >= 1:
        keys.extend(f"{BRIEFING_PREFIX}level.{n}" for n in range(max_level + 1))
    section = policy_doc.get("escalation")
    reasons = section.get("reasons") if isinstance(section, Mapping) else None
    if isinstance(reasons, Mapping) and reasons:
        keys.extend(f"{BRIEFING_PREFIX}reason.{r}" for r in (*reasons, BRIEFING_DEFAULT_REASON))
    missing: list[str] = []
    digits: list[str] = []
    for key in keys:
        for lang in langs:
            text = texts.get((key, lang))
            if not (isinstance(text, str) and text.strip()):
                missing.append(f"{key} ({lang})")
            elif any(c.isdigit() for c in text):
                digits.append(f"{key} ({lang})")
    if missing:
        raise ScriptError("prompt_catalog.json: no entry for the escalation briefing label(s) of policy.json: "
                          + ", ".join(missing))
    if digits:
        raise ScriptError("prompt_catalog.json: a spoken briefing label holds a digit (write numbers as words): "
                          + ", ".join(digits))


def masked(item: Mapping[str, Any]) -> dict[str, Any]:
    """A copy safe to print: factor hashes masked, real (non-fictional) numbers masked."""
    out = copy.deepcopy(dict(item))
    if isinstance(out.get("factors"), dict):
        out["factors"] = dict.fromkeys(out["factors"], MASK)
    phone = out.get("phone")
    if isinstance(phone, str) and not FICTIONAL_PHONE.match(phone):
        out["phone"] = mask_phone(phone)
    return out


def render_plan(plan: SeedPlan) -> str:
    lines = [f"mode: {'reseed (restore every fixture)' if plan.reseed else 'seed (fixtures only if absent)'}"]
    for table, items in (("customers", plan.customers), ("customers (ANI aliases)", plan.aliases),
                         ("policy", plan.policy), ("catalog", plan.catalog)):
        lines.append(f"== {table}: {len(items)} item(s)")
        lines.extend(json.dumps(masked(i), ensure_ascii=False, sort_keys=True) for i in items)
    if plan.delete_aliases:
        lines.append(f"== delete stale ANI aliases first: {len(plan.delete_aliases)} key(s)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- tuned masking values

MASKING_PK = "masking"
TOOL_PREFIX = "tool#"
REV = "rev"
#: Shown for a field the table doesn't hold (the bridge then uses the default).
NOT_SET = "(not set)"


@dataclass
class MaskingMerge:
    """The policy items to write, after keeping (or overwriting) the live masking values."""

    items: list[dict[str, Any]]
    #: ``(field, live, seed file)`` for every masking value that differs; ``NOT_SET`` = absent.
    differences: list[tuple[str, Any, Any]] = field(default_factory=list)
    #: Live values that fail strict validation (kept unless ``--overwrite-masking``).
    invalid: list[str] = field(default_factory=list)
    overwrite: bool = False


def masking_keys(policy_items: Sequence[Mapping[str, Any]]) -> list[str]:
    """The policy-table keys whose masking values a seed run reads first."""
    return [MASKING_PK, *(str(i["pk"]) for i in policy_items if str(i["pk"]).startswith(TOOL_PREFIX))]


def read_live_masking(store: Store,
                      policy_items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any] | None]:
    """The live ``masking`` item and each seeded ``tool#<name>`` item (``None`` = absent)."""
    return {pk: store.get("policy", {"pk": pk}) for pk in masking_keys(policy_items)}


def _storable(value: Any) -> Any:
    """A value read back from DynamoDB, writable again (the store takes no ``Decimal``)."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else str(value)
    if isinstance(value, list):
        return [_storable(v) for v in value]
    if isinstance(value, dict):
        return {k: _storable(v) for k, v in value.items()}
    return value


def _mask_differences(base: str, live: Any, seed_value: Any) -> list[tuple[str, Any, Any]]:
    if live == seed_value:
        return []
    if isinstance(live, dict) and isinstance(seed_value, dict):
        return [(f"{base}.{k}", live.get(k, NOT_SET), seed_value.get(k, NOT_SET))
                for k in sorted({*live, *seed_value})
                if live.get(k, NOT_SET) != seed_value.get(k, NOT_SET)]
    return [(base, NOT_SET if live is None else live, NOT_SET if seed_value is None else seed_value)]


def merge_masking(policy_items: Sequence[Mapping[str, Any]],
                  live: Mapping[str, Mapping[str, Any] | None], *, overwrite: bool) -> MaskingMerge:
    """Keep the live masking values in ``policy_items``, unless ``overwrite``.

    For an item present in ``live``: the ``masking`` item's :data:`MASKING_FIELDS` and a tool's
    ``mask`` are taken from the live item exactly (absent stays absent); every other field is
    the seed file's. An item absent from ``live`` is the seed file's. ``rev`` is carried over
    from the live item and bumped. Pure: no store access."""
    out = MaskingMerge(items=[], overwrite=overwrite)
    for seed_item in policy_items:
        pk = str(seed_item["pk"])
        item = copy.deepcopy(dict(seed_item))
        current = live.get(pk)
        if current is None or not (pk == MASKING_PK or pk.startswith(TOOL_PREFIX)):
            out.items.append(item)
            continue
        current = _storable(dict(current))
        if pk == MASKING_PK:
            for name in MASKING_FIELDS:
                lv, sv = current.get(name, NOT_SET), item.get(name, NOT_SET)
                if lv != sv:
                    out.differences.append((f"masking.{name}", lv, sv))
                if not overwrite:
                    if name in current:
                        item[name] = current[name]
                    else:
                        item.pop(name, None)
            _, problems = masking_parts({k: v for k, v in current.items() if k in MASKING_FIELDS})
        else:
            tool = pk[len(TOOL_PREFIX):]
            out.differences.extend(_mask_differences(f"tools.{tool}.mask", current.get("mask"),
                                                     item.get("mask")))
            if not overwrite:
                if "mask" in current:  # exactly as stored, an explicit null included
                    item["mask"] = current["mask"]
                else:
                    item.pop("mask", None)
            _, problems = tool_mask_parts(tool, current.get("mask"))
        out.invalid.extend(str(p) for p in problems)
        rev = current.get(REV)
        if isinstance(rev, int) and not isinstance(rev, bool):
            item[REV] = rev + 1
        out.items.append(item)
    return out


def _show(value: Any) -> str:
    return value if value == NOT_SET else json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_masking_report(merge: MaskingMerge) -> str:
    """The difference report (field, live, seed file) and a warning per invalid live value."""
    lines: list[str] = []
    if merge.differences:
        what = ("REPLACED with the seed file's values (--overwrite-masking)" if merge.overwrite
                else "KEPT (the live values); pass --overwrite-masking to write the seed file's values")
        lines.append(f"== masking values that differ from policy.json: {len(merge.differences)}, {what}")
        lines.append("   field | live | seed file")
        lines.extend(f"   {f} | {_show(lv)} | {_show(sv)}" for f, lv, sv in merge.differences)
    else:
        lines.append("== masking values: the table matches policy.json")
    for problem in merge.invalid:
        verdict = ("replaced by the seed file's value" if merge.overwrite
                   else "kept; --overwrite-masking replaces it with the seed file's value")
        lines.append(f"WARNING: live masking value invalid: {problem} ({verdict})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- tuned slot settings

SLOT_PREFIX = "slot#"


@dataclass
class SlotsMerge:
    """The policy items to write, after keeping (or overwriting) the live ``slot#<name>`` fields."""

    items: list[dict[str, Any]]
    #: ``(field, live, seed file)`` for every live value that differs from the seed file
    #: (``NOT_SET`` = the seed file has no such field).
    differences: list[tuple[str, Any, Any]] = field(default_factory=list)
    #: ``(field, seed file)`` for every field the live item lacks, filled from the seed file.
    added: list[tuple[str, Any]] = field(default_factory=list)
    #: Keys (``slot_resolution``, ``slot#<name>``) with no item in the table yet (written from the
    #: seed file).
    new: list[str] = field(default_factory=list)
    #: Kept live values that fail validation (field and reason, never a value).
    invalid: list[str] = field(default_factory=list)
    overwrite: bool = False


def slot_keys(policy_items: Sequence[Mapping[str, Any]]) -> list[str]:
    """The seeded ``slot#<name>`` keys whose live fields a seed run reads first."""
    return [str(i["pk"]) for i in policy_items if str(i["pk"]).startswith(SLOT_PREFIX)]


def slot_merge_keys(policy_items: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every key :func:`merge_slots` merges, in item order: the seeded ``slot#<name>`` items and
    ``slot_resolution``."""
    return [str(i["pk"]) for i in policy_items
            if str(i["pk"]) == RESOLUTION_ITEM or str(i["pk"]).startswith(SLOT_PREFIX)]


def read_live_slots(store: Store,
                    policy_items: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any] | None]:
    """``slot_resolution`` and each seeded ``slot#<name>`` item as the table holds it (``None`` =
    absent)."""
    return {pk: store.get("policy", {"pk": pk}) for pk in slot_merge_keys(policy_items)}


def _factor_defs(policy_items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    meta = next((i for i in policy_items if i.get("pk") == "meta"), None)
    factors = (meta or {}).get("factors")
    return factors if isinstance(factors, Mapping) else {}


def _body(item: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in ("pk", REV)}


def _next_rev(current: Mapping[str, Any] | None) -> int:
    """The live item's ``rev`` plus one, or 1 (a new item, or a live one without an integer)."""
    rev = None if current is None else current.get(REV)
    return rev + 1 if isinstance(rev, int) and not isinstance(rev, bool) else 1


def merge_slots(policy_items: Sequence[Mapping[str, Any]],
                live: Mapping[str, Mapping[str, Any] | None], *, overwrite: bool) -> SlotsMerge:
    """Keep the console-tuned slot settings in ``policy_items``, unless
    ``overwrite``.

    For the ``slot_resolution`` item and each seeded ``slot#<name>`` item present in ``live``:
    every field the live item holds is kept exactly (an invalid value too, reported in
    :attr:`SlotsMerge.invalid`); a field it lacks takes the seed file's value (so phase 2's new
    fields reach a phase 1 item). With ``overwrite`` the item is the seed file's. ``rev`` is the
    live item's plus one, or 1 for a new item (or a live one without an integer ``rev``). Every
    other item (``meta`` and its ``slots`` list included) is the seed file's. Pure: no store
    access."""
    out = SlotsMerge(items=[], overwrite=overwrite)
    factor_defs = _factor_defs(policy_items)
    for seed_item in policy_items:
        pk = str(seed_item["pk"])
        item = copy.deepcopy(dict(seed_item))
        if not (pk == RESOLUTION_ITEM or pk.startswith(SLOT_PREFIX)):
            out.items.append(item)
            continue
        where = RESOLUTION_ITEM if pk == RESOLUTION_ITEM else f"slots.{pk[len(SLOT_PREFIX):]}"
        current = live.get(pk)
        if current is None:
            out.new.append(pk)
            item[REV] = 1
            out.items.append(item)
            continue
        current = _storable(dict(current))
        seed_body, live_body = _body(item), _body(current)
        for f in sorted({*seed_body, *live_body}):
            lv, sv = live_body.get(f, NOT_SET), seed_body.get(f, NOT_SET)
            if f not in live_body:
                out.added.append((f"{where}.{f}", sv))
            elif lv != sv:
                out.differences.append((f"{where}.{f}", lv, sv))
        kept = {**seed_body, **live_body}
        if pk == RESOLUTION_ITEM:
            _, problems = load_slot_resolution(kept)
        else:
            name = pk[len(SLOT_PREFIX):]
            _, problems = load_slot(name, kept, factor_def=factor_defs.get(name))
        out.invalid.extend(f"{p.field}: {p.message}" for p in problems)
        merged = seed_body if overwrite else kept
        out.items.append({"pk": pk, **merged, REV: _next_rev(current)})
    return out


def render_slots_report(merge: SlotsMerge) -> str:
    """The difference report (field, live, seed file), the filled and new items, and a warning per
    invalid live value."""
    lines: list[str] = []
    if merge.differences:
        what = ("REPLACED with the seed file's values (--overwrite-slots)" if merge.overwrite
                else "KEPT (the live values); pass --overwrite-slots to write the seed file's values")
        lines.append(f"== slot settings that differ from policy.json: {len(merge.differences)}, {what}")
        lines.append("   field | live | seed file")
        lines.extend(f"   {f} | {_show(lv)} | {_show(sv)}" for f, lv, sv in merge.differences)
    elif not merge.added and not merge.new:
        lines.append("== slot settings: the table matches policy.json")
    else:
        lines.append("== slot settings: no live value differs from policy.json")
    if merge.added:
        lines.append(f"== slot settings missing from the table, written from policy.json: {len(merge.added)}")
        lines.extend(f"   {f} | {_show(sv)}" for f, sv in merge.added)
    if merge.new:
        lines.append("== slot items not in the table yet, written from policy.json: "
                     + ", ".join(merge.new))
    for problem in merge.invalid:
        verdict = ("replaced by the seed file's value" if merge.overwrite
                   else "kept; --overwrite-slots replaces it with the seed file's value")
        lines.append(f"WARNING: live slot setting invalid: {problem} ({verdict})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- console prompt edits

CatalogKey = tuple[str, str]
#: The fields of a catalog entry a seed compares (besides ``text``) to decide whether to write it.
_ENTRY_FIELDS = ("version", "status", "audio_url")
#: The key prefixes the console may add a new key under (the admin API's ``NEW_KEY_PREFIXES``).
CONSOLE_KEY_PREFIXES = (MASK_PREFIX, ARG_PREFIX, OPTION_PREFIX)
#: The catalog item field holding the hash of the text the seed wrote. Never served,
#: shown in the console or audited; the admin API copies it from the old item on an edit.
SEED_HASH = "seed_hash"
#: Hashers by tag: a ``seed_hash`` is ``<tag>:<hex digest>``, so the algorithm can change. A hash
#: with an unknown tag counts as none (legacy).
_HASHERS: dict[str, Callable[[str], str]] = {
    "sha256": lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()}
_HASH_TAG = "sha256"


def seed_hash(text: str) -> str:
    """The ``seed_hash`` of ``text``: ``sha256:<hex digest of its UTF-8 bytes>``."""
    return f"{_HASH_TAG}:{_HASHERS[_HASH_TAG](text)}"


def _seeded_text_unchanged(item: Mapping[str, Any]) -> bool | None:
    """Whether ``item``'s text is still the one the seed wrote: ``True`` (it hashes to its
    ``seed_hash``), ``False`` (edited in the console since), or ``None`` (no usable ``seed_hash``:
    seeded before the hash existed, so an edit can't be told from an older seed text)."""
    raw = item.get(SEED_HASH)
    if not isinstance(raw, str) or ":" not in raw:
        return None
    tag, digest = raw.split(":", 1)
    hasher = _HASHERS.get(tag)
    if hasher is None:
        return None
    text = item.get("text")
    return isinstance(text, str) and hasher(text) == digest


@dataclass
class PromptsMerge:
    """The catalog items to write, after keeping (or overwriting) the prompt texts edited live.

    Every list holds ``(key, lang)`` only: a report never shows a prompt's text."""

    items: list[dict[str, Any]]
    #: Live text edited in the console (or without a ``seed_hash``) differs from the seed file's:
    #: not written (the console edit stays).
    kept: list[CatalogKey] = field(default_factory=list)
    #: The part of ``kept`` without a ``seed_hash`` (seeded before the hash existed): an edit can't be
    #: told from an older seed text until one ``--overwrite-prompts`` writes the hashes.
    legacy: list[CatalogKey] = field(default_factory=list)
    #: Live text differed from the seed file's and was overwritten (``--overwrite-prompts``, or
    #: ``--overwrite-slots`` for an ``option.*`` entry).
    replaced: list[CatalogKey] = field(default_factory=list)
    #: Live text was the one the seed wrote (it matches ``seed_hash``) and the seed file's text
    #: changed: written, as no console edit is lost.
    updated: list[CatalogKey] = field(default_factory=list)
    #: Live text equals the seed file's but its ``seed_hash`` was missing or stale: only the hash is
    #: written, ``rev`` unchanged (no visible field changes, so an open editor gets no conflict).
    hashed: list[CatalogKey] = field(default_factory=list)
    #: Not in the table yet: written from the seed file.
    new: list[CatalogKey] = field(default_factory=list)
    #: Keys added in the console that the seed file doesn't have: kept in the index, or dropped
    #: from it with ``--overwrite-prompts`` (their items stay, unserved).
    live_only: list[str] = field(default_factory=list)
    overwrite: bool = False


def read_live_catalog(store: Store) -> dict[CatalogKey, dict[str, Any]]:
    """Every catalog table item (the meta item included), by ``(key, lang)``. One scan: the
    table holds a few hundred small items."""
    return {(str(i.get("key", "")), str(i.get("lang", ""))): i for i in store.scan("catalog")}


def _int_rev(item: Mapping[str, Any]) -> bool:
    rev = item.get(REV)
    return isinstance(rev, int) and not isinstance(rev, bool)


def _console_added(key: str, live: Mapping[CatalogKey, Mapping[str, Any]]) -> bool:
    """A key the console could have added (its prefix) with an item the console wrote (``rev`` of
    1 or more) and no language carrying a ``seed_hash``: the seed writes a hash on every item, and
    a console edit of a seeded item keeps it, so a key the seed ever wrote and the seed file then
    dropped is not one (the seed also bumps ``rev``, so ``rev`` alone can't tell)."""
    if not key.startswith(CONSOLE_KEY_PREFIXES):
        return False
    items = [item for (k, _), item in live.items() if k == key]
    return (any(_int_rev(item) and item[REV] >= 1 for item in items)
            and not any(SEED_HASH in item for item in items))


def _key_revs(live: Mapping[CatalogKey, Mapping[str, Any]]) -> dict[str, int]:
    """Per key, the highest integer ``rev`` over its live languages (the console's view ``rev``,
    as the admin API computes it; the meta item excluded)."""
    out: dict[str, int] = {}
    for (key, lang), item in live.items():
        if (key, lang) == (META_KEY, META_LANG):
            continue
        rev = item.get(REV)
        out[key] = max(out.get(key, 0), rev if isinstance(rev, int) and not isinstance(rev, bool)
                       else 0)
    return out


def merge_catalog_prompts(catalog_items: Sequence[Mapping[str, Any]],
                          live: Mapping[CatalogKey, Mapping[str, Any]], *, overwrite: bool,
                          overwrite_options: bool = False) -> PromptsMerge:
    """Keep every prompt text edited in the console, unless ``overwrite``.

    Every item written carries ``seed_hash`` (:func:`seed_hash` of its text). For each seed entry
    (``catalog_items`` from :meth:`PromptCatalog.to_items`):

    * absent from ``live``: written from the seed file (``new``), with the key's view ``rev`` + 1
      when the key already has live languages (see below), else without a ``rev``;
    * live text differs from the seed file's and still matches its ``seed_hash`` (not edited in the
      console): written with ``rev`` bumped (``updated``);
    * live text differs and doesn't match its ``seed_hash`` (a console edit), or has none (seeded
      before the hash existed; also listed in ``legacy``): not written (``kept``), or with ``overwrite``
      (``overwrite_options`` for an ``option.*`` key) written with ``rev`` bumped (``replaced``), so an editor holding the old ``rev`` gets a conflict;
    * same text, same version, status, audio URL and ``seed_hash``: not written, so its ``rev`` is
      unchanged and an editor holding it gets no false conflict;
    * same text, another field differs: written from the seed file with ``rev`` bumped;
    * same text and fields, ``seed_hash`` missing or stale: written with the live ``rev`` kept
      (``hashed``): nothing an editor sees changes.

    A bumped ``rev`` is the key's view ``rev`` (the highest over its live languages, as the admin
    API shows it) + 1, as ``put_prompt`` writes it, so the view ``rev`` rises on every seed text
    change.

    The meta item is the seed file's, but its ``keys`` index also keeps every key the live index
    lists that the seed file lacks and the console added (:func:`_console_added`; ``live_only``),
    unless ``overwrite``. So the index names only keys with stored items, and a key dropped from
    the seed file leaves it. A live meta ``rev`` is carried over and bumped.
    Pure: no store access."""
    out = PromptsMerge(items=[], overwrite=overwrite)
    meta_key = (META_KEY, META_LANG)
    key_revs = _key_revs(live)
    seed_keys: set[str] = set()
    for seed_item in catalog_items:
        item = copy.deepcopy(dict(seed_item))
        key = (str(item.get("key", "")), str(item.get("lang", "")))
        if key == meta_key:
            continue
        seed_keys.add(key[0])
        item[SEED_HASH] = seed_hash(str(item.get("text", "")))
        current = live.get(key)
        if current is None:
            out.new.append(key)
            # A language new to a key with live languages changes the console's view of that key,
            # so it gets the key's view rev + 1 (an open editor then conflicts); a new key has none.
            out.items.append({**item, REV: key_revs[key[0]] + 1} if key[0] in key_revs else item)
            continue
        over = overwrite or (overwrite_options and key[0].startswith(OPTION_PREFIX))
        if current.get("text") != item.get("text"):
            unedited = _seeded_text_unchanged(current)
            if unedited:
                out.updated.append(key)
            elif over:
                out.replaced.append(key)
            else:
                out.kept.append(key)
                if unedited is None:
                    out.legacy.append(key)
                continue
        elif all(current.get(f) == item.get(f) for f in _ENTRY_FIELDS):
            if current.get(SEED_HASH) != item[SEED_HASH]:
                out.hashed.append(key)  # the hash alone: nothing an editor sees, so rev stays
                out.items.append({**item, **({REV: current[REV]} if REV in current else {})})
            continue
        # rev = the key's view rev (max over its live languages) + 1, as put_prompt writes it:
        # bumping only this language's rev could leave the view rev unchanged, and an editor
        # holding it would overwrite this text with no conflict.
        out.items.append({**item, REV: key_revs.get(key[0], 0) + 1})
    for seed_item in catalog_items:
        if (str(seed_item.get("key", "")), str(seed_item.get("lang", ""))) != meta_key:
            continue
        meta = copy.deepcopy(dict(seed_item))
        live_meta = live.get(meta_key)
        if live_meta is not None:
            raw = live_meta.get("keys")
            for k in raw if isinstance(raw, list) else []:
                if (isinstance(k, str) and k not in seed_keys and k not in out.live_only
                        and _console_added(k, live)):
                    out.live_only.append(k)
            if not overwrite:
                meta["keys"] = [*meta.get("keys", []), *out.live_only]
            if _int_rev(live_meta):
                meta[REV] = _next_rev(live_meta)
        out.items.insert(0, meta)
    return out


def _entry(key: CatalogKey) -> str:
    return f"{key[0]} ({key[1]})"


def render_prompts_report(merge: PromptsMerge) -> str:
    """Keys and languages only, never a prompt's text."""
    lines: list[str] = []
    edited = [k for k in merge.kept if k not in merge.legacy]
    if edited:
        lines.append(f"== prompts edited in the console (text differs from prompt_catalog.json): "
                     f"{len(edited)}, KEPT (not written); pass --overwrite-prompts to write the "
                     "seed file's text")
        lines.append("   " + ", ".join(_entry(k) for k in edited))
    if merge.legacy:
        lines.append(f"== prompts whose text differs from prompt_catalog.json, legacy, no hash (seeded "
                     f"before seed hashes: a console edit can't be told from an older seed text): "
                     f"{len(merge.legacy)}, KEPT (not written); one --overwrite-prompts writes the "
                     "seed file's text and its hash")
        lines.append("   " + ", ".join(_entry(k) for k in merge.legacy))
    if merge.replaced:
        lines.append(f"== prompts edited in the catalog table: {len(merge.replaced)}, REPLACED with "
                     "the seed file's text (--overwrite-prompts; --overwrite-slots for option.*)")
        lines.append("   " + ", ".join(_entry(k) for k in merge.replaced))
    if merge.updated:
        lines.append(f"== prompts changed in prompt_catalog.json and not edited in the console: "
                     f"{len(merge.updated)}, written from prompt_catalog.json")
        lines.append("   " + ", ".join(_entry(k) for k in merge.updated))
    if not (merge.kept or merge.replaced or merge.updated):
        lines.append("== prompts: no text in the catalog table differs from prompt_catalog.json")
    if merge.hashed:
        lines.append(f"== prompts given their seed hash (text unchanged, rev kept): {len(merge.hashed)}")
    if merge.new:
        lines.append(f"== prompts not in the catalog table yet, written from prompt_catalog.json: "
                     f"{len(merge.new)}")
        lines.append("   " + ", ".join(_entry(k) for k in merge.new))
    if merge.live_only:
        what = ("DROPPED from the index (--overwrite-prompts); their items stay, unserved"
                if merge.overwrite else "kept in the index")
        lines.append(f"== prompt keys only in the catalog table (added in the console): "
                     f"{len(merge.live_only)}, {what}")
        lines.append("   " + ", ".join(merge.live_only))
    return "\n".join(lines)


# --------------------------------------------------------------------------- per-model prompts

PROMPT_VALIDATION_FILE = "prompt_validation.json"
PROMPT_SUITE_FILE = "prompt_suite.json"
#: Stages where ``--allow-unvalidated-prompt`` may seed a prompt without a current record.
UNVALIDATED_PROMPT_STAGES = frozenset({"dev"})


@dataclass
class PromptGate:
    """The ``agent.system.*`` texts (``(key, lang)``, in catalog order) without a current record,
    seeded only because ``--allow-unvalidated-prompt`` was given at stage dev."""

    unvalidated: list[CatalogKey] = field(default_factory=list)


def load_prompt_validation(data_dir: Path) -> tuple[ValidationFile, Suite | None]:
    """The records and the case file under ``data_dir`` (``None``: no case file yet, so no record
    is current). A missing records file reads as no records."""
    path = data_dir / PROMPT_VALIDATION_FILE
    try:
        records = load_records(path) if path.exists() else ValidationFile.empty()
    except (OSError, ValueError) as exc:
        raise ScriptError(f"can't read {PROMPT_VALIDATION_FILE}: {type(exc).__name__}") from None
    suite_path = data_dir / PROMPT_SUITE_FILE
    try:
        return records, load_suite(suite_path) if suite_path.exists() else None
    except (OSError, ValueError) as exc:
        raise ScriptError(f"can't read {PROMPT_SUITE_FILE}: {type(exc).__name__}") from None


def unvalidated_prompts(catalog_doc: Mapping[str, Any], records: ValidationFile,
                        suite: Suite | None) -> list[CatalogKey]:
    """Each published ``agent.system.<model key>`` text of ``catalog_doc`` without a record carrying
    its current prompt hash, ``suite``'s hash and full runs (:meth:`ValidationFile.is_validated`), as
    ``(key, lang)``."""
    catalog = PromptCatalog.from_dict(catalog_doc)
    prefix = SYSTEM_KEY_PREFIX + "."
    out: list[CatalogKey] = []
    for e in catalog_doc.get("entries", []):
        key, lang = str(e.get("key", "")), str(e.get("lang", ""))
        if (not key.startswith(prefix) or e.get("status", "published") != "published"
                or (key, lang) in out):
            continue
        if suite is None or not records.is_validated(catalog, key[len(prefix):], lang, suite):
            out.append((key, lang))
    return out


def agent_text_problems(catalog_doc: Mapping[str, Any]) -> list[str]:
    """Each published ``agent.*`` entry whose text breaks
    :func:`~owlbank.agent_prompt.placeholder_problems`, as ``"key (lang): why"`` in catalog order, then
    each caller-context frame (``agent.caller_context``, ``agent.caller_context_unverified``) with no
    published entry in a catalog language (``"key (lang): missing"``). Empty: the catalog is fine."""
    out: list[str] = []
    published: set[CatalogKey] = set()
    for e in catalog_doc.get("entries", []):
        if e.get("status", "published") != "published":
            continue
        key, lang = str(e.get("key", "")), str(e.get("lang", ""))
        published.add((key, lang))
        if key.startswith(AGENT_PREFIX):
            why = placeholder_problems(key, str(e.get("text", "")))
            if why is not None:
                out.append(f"{_entry((key, lang))}: {why}")
    for lang in catalog_doc.get("languages") or ():
        out += [f"{_entry((key, str(lang)))}: missing" for key in CALLER_CONTEXT_KEYS
                if (key, str(lang)) not in published]
    return out


def check_agent_texts(catalog_doc: Mapping[str, Any]) -> None:
    """Refuse (``ScriptError``) a catalog with any :func:`agent_text_problems`.
    It runs before the validation gate and has no override."""
    problems = agent_text_problems(catalog_doc)
    if problems:
        raise ScriptError(
            "prompt_catalog.json: agent prompt text(s) break the placeholder rule: " + "; ".join(problems)
            + ". agent.system.*, the web-activity blocks, the unverified caller context and any other agent.* "
            "text have no placeholders; agent.caller_context has exactly {firstName}, {memory}, {profile}; "
            "both caller-context frames are published in every catalog language")


def check_prompt_gate(catalog_doc: Mapping[str, Any], records: ValidationFile, suite: Suite | None, *,
                      stage: str, allow_unvalidated: bool) -> PromptGate:
    """Refuse (``ScriptError``) a seed of any unvalidated ``agent.system.*`` text, unless
    ``allow_unvalidated`` at stage dev; ``allow_unvalidated`` at any other stage is refused too."""
    if allow_unvalidated and stage not in UNVALIDATED_PROMPT_STAGES:
        raise ScriptError(f"--allow-unvalidated-prompt works only with --stage "
                          f"{'/'.join(sorted(UNVALIDATED_PROMPT_STAGES))}, not {stage!r}")
    unvalidated = unvalidated_prompts(catalog_doc, records, suite)
    if unvalidated and not allow_unvalidated:
        raise ScriptError(
            "prompt_catalog.json: agent prompt(s) without a current validation record: "
            + ", ".join(_entry(k) for k in unvalidated)
            + (f" (no {PROMPT_SUITE_FILE} yet)" if suite is None else "")
            + ". Run the prompt suite for each model (scripts/prompt_suite.py) and commit its record in "
            f"data/{PROMPT_VALIDATION_FILE}; in dev only, --allow-unvalidated-prompt seeds them anyway")
    return PromptGate(unvalidated=unvalidated)


def render_prompt_report(gate: PromptGate) -> str:
    if not gate.unvalidated:
        return "== agent prompts: every agent.system.* text has a current validation record"
    return (f"WARNING: agent prompts seeded WITHOUT a current validation record "
            f"(--allow-unvalidated-prompt, dev only): {len(gate.unvalidated)}\n   "
            + ", ".join(_entry(k) for k in gate.unvalidated))


# --------------------------------------------------------------------------- stored redaction setting

REDACTION_PK = "redaction"
#: The ``redaction`` item's fields (the schema); any other field is dropped.
REDACTION_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(RedactionConfig))


@dataclass
class RedactionMerge:
    """The policy items to write, after keeping (or overwriting) the live ``redaction`` fields."""

    items: list[dict[str, Any]]
    #: ``(field, live, seed file)`` for every kept live value that differs from the seed file.
    differences: list[tuple[str, Any, Any]] = field(default_factory=list)
    #: ``(field, written)`` for every schema field the live item lacks.
    added: list[tuple[str, Any]] = field(default_factory=list)
    #: ``(field, live)`` for every live field the schema doesn't have (``apply_to``), dropped.
    dropped: list[tuple[str, Any]] = field(default_factory=list)
    #: The live ``entity_types`` or ``apply_to`` was empty (phase 1 "off"): written
    #: ``enabled: false`` with the seed file's types.
    switched_off: bool = False
    #: Which live fields made it phase 1's "off": ``apply_to`` and/or ``entity_types``, in that order.
    switched_off_by: list[str] = field(default_factory=list)
    #: The table has no ``redaction`` item yet (written from the seed file).
    new: bool = False
    #: Kept live values that fail strict validation (field and reason).
    invalid: list[str] = field(default_factory=list)
    overwrite: bool = False


def read_live_redaction(store: Store) -> dict[str, dict[str, Any] | None]:
    """The live ``redaction`` item (``None`` = absent), keyed by its ``pk``."""
    return {REDACTION_PK: store.get("policy", {"pk": REDACTION_PK})}


def merge_redaction(policy_items: Sequence[Mapping[str, Any]],
                    live: Mapping[str, Mapping[str, Any] | None], *, overwrite: bool) -> RedactionMerge:
    """Keep the stored transcript redaction setting in ``policy_items``, unless
    ``overwrite``. Shaped like :func:`merge_slots`, with the rules for a phase 1 item:

    * a live field outside :data:`REDACTION_FIELDS` (``apply_to``) is dropped and reported, with
      or without ``overwrite``;
    * a live ``apply_to: []`` (phase 1: applied to nothing, so nothing was redacted) is
      written as ``enabled: false`` with the seed file's types, and reported, like the next rule.
      A non-empty ``apply_to`` is only dropped;
    * an empty ``entity_types`` list not paired with ``enabled: false`` (phase 1's "off": "no
      type selected means nothing is redacted") is written as ``enabled: false`` with the seed
      file's types, so an operator's "off" stays off, and reported. It is judged on the merged
      fields (live over seed);
    * every other schema field the live item holds is kept exactly (an invalid value too,
      reported); a field it lacks takes the seed file's value.

    ``rev`` is the live item's plus one, or 1. Every other item is the seed file's. Pure."""
    out = RedactionMerge(items=[], overwrite=overwrite)
    for seed_item in policy_items:
        item = copy.deepcopy(dict(seed_item))
        if item.get("pk") != REDACTION_PK:
            out.items.append(item)
            continue
        current = live.get(REDACTION_PK)
        if current is None:
            out.new = True
            out.items.append({**item, REV: 1})
            continue
        current = _storable(dict(current))
        seed_body, live_body = _body(item), _body(current)
        out.dropped = [(f"redaction.{f}", v) for f, v in sorted(live_body.items())
                       if f not in REDACTION_FIELDS]
        kept_live = {f: v for f, v in live_body.items() if f in REDACTION_FIELDS}
        merged = {**seed_body, **kept_live}
        # Judged on the merged view, so a second merge finds nothing to switch (idempotent).
        off_by = (["apply_to"] if live_body.get("apply_to") == [] else []) + (
            ["entity_types"] if merged.get("entity_types") == [] and merged.get("enabled") is not False
            else [])
        off = bool(off_by)
        if off:
            merged["enabled"] = False
            merged["entity_types"] = copy.deepcopy(seed_body.get("entity_types", []))
        for f in REDACTION_FIELDS:
            if f not in kept_live:
                out.added.append((f"redaction.{f}", (seed_body if overwrite else merged).get(f, NOT_SET)))
            elif kept_live[f] != seed_body.get(f, NOT_SET) and not (off and f in ("enabled", "entity_types")):
                out.differences.append((f"redaction.{f}", kept_live[f], seed_body.get(f, NOT_SET)))
        if overwrite:
            merged = seed_body
        else:
            out.switched_off, out.switched_off_by = off, off_by
            try:
                parse_redaction_config(merged)
            except RedactionConfigError as exc:
                out.invalid.append(str(exc))
        out.items.append({"pk": REDACTION_PK, **merged, REV: _next_rev(current)})
    return out


def render_redaction_report(merge: RedactionMerge) -> str:
    """The difference report (field, live, seed file), the dropped fields, the switch-off and
    a warning per invalid live value."""
    lines: list[str] = []
    if merge.new:
        lines.append("== redaction setting: not in the table yet, written from policy.json")
    elif merge.differences:
        what = ("REPLACED with the seed file's values (--overwrite-redaction)" if merge.overwrite
                else "KEPT (the live values); pass --overwrite-redaction to write the seed file's values")
        lines.append(f"== redaction settings that differ from policy.json: {len(merge.differences)}, {what}")
        lines.append("   field | live | seed file")
        lines.extend(f"   {f} | {_show(lv)} | {_show(sv)}" for f, lv, sv in merge.differences)
    elif not (merge.added or merge.dropped or merge.switched_off):
        lines.append("== redaction setting: the table matches policy.json")
    else:
        lines.append("== redaction setting: no live value differs from policy.json")
    if merge.added:
        lines.append(f"== redaction settings missing from the table, written: {len(merge.added)}")
        lines.extend(f"   {f} | {_show(v)}" for f, v in merge.added)
    if merge.dropped:
        lines.append(f"== redaction fields the schema no longer has, dropped: {len(merge.dropped)}")
        lines.extend(f"   {f} | {_show(v)}" for f, v in merge.dropped)
    if merge.switched_off:
        by = merge.switched_off_by or ["entity_types"]
        lines.append(f"== redaction: the live {' and '.join(by)} {'were' if len(by) > 1 else 'was'} empty "
                     "(phase 1 \"off\"): written as enabled: false with policy.json's types, so it "
                     "stays off")
    for problem in merge.invalid:
        lines.append(f"WARNING: live redaction setting invalid: {problem} "
                     "(kept; --overwrite-redaction replaces it with the seed file's value)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- apply


def apply_plan(plan: SeedPlan, store: Store) -> dict[str, int]:
    """Write the plan. Returns counts per action."""
    counts = {"customers_written": 0, "customers_kept": 0, "aliases_deleted": 0,
              "aliases_written": 0, "policy_written": 0, "catalog_written": 0}
    step = "aliases_delete"
    try:
        for key in plan.delete_aliases:
            store.delete("customers", {"customer_id": key})
            counts["aliases_deleted"] += 1
        script_step(log, "seed", step, "ok", count=counts["aliases_deleted"])
        step = "customers"
        for item in plan.customers:
            if plan.reseed:
                store.put("customers", item)
                counts["customers_written"] += 1
            else:
                try:
                    store.put("customers", item, condition="not_exists")
                    counts["customers_written"] += 1
                except ConditionFailed:  # an existing fixture is kept (edits survive a plain re-seed)
                    counts["customers_kept"] += 1
        script_step(log, "seed", step, "ok", written=counts["customers_written"],
                    kept=counts["customers_kept"])
        step = "aliases"
        for item in plan.aliases:
            store.put("customers", item)
            counts["aliases_written"] += 1
        script_step(log, "seed", step, "ok", count=counts["aliases_written"])
        step = "policy"
        for item in plan.policy:
            store.put("policy", item)
            counts["policy_written"] += 1
        script_step(log, "seed", step, "ok", count=counts["policy_written"])
        step = "catalog"
        for item in plan.catalog:
            store.put("catalog", item)
            counts["catalog_written"] += 1
        script_step(log, "seed", step, "ok", count=counts["catalog_written"])
    except Exception as exc:
        script_step(log, "seed", step, "error")
        script_error(log, "seed", exc, step=step, decision="reraise", counts=counts)
        raise
    return counts


def resolve_tables(outputs: Mapping[str, str], overrides: Mapping[str, str], *,
                   require: Sequence[str] = ()) -> dict[str, str]:
    unknown = set(overrides) - set(TABLE_NAMES)
    if unknown:
        raise ScriptError(f"--table: unknown logical table(s) {sorted(unknown)}")
    names: dict[str, str] = {}
    missing: list[str] = []
    for logical in TABLE_NAMES:
        cap = logical.capitalize()
        found = overrides.get(logical) or find_output(
            outputs, (f"Table{cap}", f"{cap}TableName", f"{cap}Table", f"{logical}_table"))
        if found is None and logical in require:
            missing.append(logical)
        names[logical] = found or f"owlbank-{logical}"
    if missing:
        raise ScriptError(f"no table name for {missing}: pass --cdk-outputs or --table LOGICAL=NAME")
    return names


def table_named(outputs: Mapping[str, str], overrides: Mapping[str, str], logical: str) -> bool:
    """Whether ``logical``'s table was named (``--table`` or the CDK outputs), not defaulted."""
    cap = logical.capitalize()
    return bool(overrides.get(logical) or find_output(
        outputs, (f"Table{cap}", f"{cap}TableName", f"{cap}Table", f"{logical}_table")))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Seed OwlBank demo data into DynamoDB (dry run by default).")
    p.add_argument("--stage", required=True, help="deployment stage, e.g. dev, test, prod")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--cdk-outputs", help="cdk deploy --outputs-file JSON")
    p.add_argument("--table", action="append", metavar="LOGICAL=NAME", help="physical table name")
    p.add_argument("--developer-phone", help="the developer's own phone (dev/test only)")
    p.add_argument("--test-number", action="append", metavar="LABEL=+E164",
                   help="an owned Twilio test number, e.g. A=+1... (dev/test only)")
    p.add_argument("--developer-factor", action="append", metavar="FACTOR=VALUE",
                   help="a factor value for the developer phone's fixture, e.g. zip=12345 (dev/test only; "
                        "hashed, never stored in the repository or printed)")
    p.add_argument("--reseed", action="store_true", help="restore every fixture and reassign aliases")
    p.add_argument("--assign", action="append", metavar="LABEL=FIXTURE[+FIXTURE]|none",
                   help="with --reseed, in test: which fixture(s) a test number belongs to")
    p.add_argument("--overwrite-masking", action="store_true",
                   help="write policy.json's masking values over the ones tuned in the table "
                        "(by default a seed keeps them and reports the differences)")
    p.add_argument("--overwrite-slots", action="store_true",
                   help="write policy.json's slot settings (slot#<name> items) over the ones tuned in "
                        "the table (by default a seed keeps every live field, fills only missing ones "
                        "and reports the differences); also writes the catalog's option.* names")
    p.add_argument("--overwrite-prompts", action="store_true",
                   help="write prompt_catalog.json's text over prompts edited in the console and "
                        "over legacy ones without a seed hash (by default a seed keeps them and "
                        "lists each key and language); drops keys added in the console from the "
                        "catalog index")
    p.add_argument("--overwrite-redaction", action="store_true",
                   help="write policy.json's transcript redaction setting over the one stored in the "
                        "table (by default a seed keeps every live field and reports the differences)")
    p.add_argument("--allow-unvalidated-prompt", action="store_true",
                   help="dev only (refused at any other stage): seed agent.system.* texts that have no "
                        "current record in prompt_validation.json; the plan lists each one")
    p.add_argument("--region", help="AWS region (default: a table ARN's, then AWS_DEFAULT_REGION or AWS_REGION, "
                                    "then us-east-1)")
    add_mode_args(p)
    return p


def _make_store(args: argparse.Namespace, tables: Mapping[str, str]) -> Store:
    region = resolve_region(args.region, arns=tables.values(), logger=log)
    return DynamoStore(tables, region_name=region)


def main(argv: list[str] | None = None, *, store: Store | None = None) -> int:
    args = build_parser().parse_args(argv)
    script_start(log, "seed", stage=args.stage, mode="apply" if args.apply else "dry_run",
                 reseed=bool(args.reseed), overwrite_masking=bool(args.overwrite_masking),
                 overwrite_slots=bool(args.overwrite_slots),
                 overwrite_redaction=bool(args.overwrite_redaction),
                 overwrite_prompts=bool(args.overwrite_prompts),
                 allow_unvalidated_prompt=bool(args.allow_unvalidated_prompt))
    try:
        d = args.data_dir
        plan = build_plan(
            load_json(d / "fixtures.json"), load_json(d / "policy.json"),
            load_json(d / "prompt_catalog.json"),
            stage=args.stage, developer_phone=args.developer_phone,
            test_numbers=parse_pairs(args.test_number, what="--test-number"),
            assignments=parse_pairs(args.assign, what="--assign"), reseed=args.reseed,
            developer_factors=parse_pairs(args.developer_factor, what="--developer-factor"),
        )
        catalog_doc = load_json(d / "prompt_catalog.json")
        check_agent_texts(catalog_doc)  # before the gate, no override
        prompt_gate = check_prompt_gate(catalog_doc, *load_prompt_validation(d),
                                        stage=args.stage, allow_unvalidated=args.allow_unvalidated_prompt)
        outputs = load_cdk_outputs(args.cdk_outputs)
        overrides = parse_pairs(args.table, what="--table")
        tables = resolve_tables(outputs, overrides, require=SEEDED_TABLES if args.apply else ())
    except (ScriptError, PolicyError) as exc:  # an invalid policy.json exits 2, like the rest
        script_error(log, "seed", exc, step="plan", decision="exit_2")
        print(f"seed: {exc}", file=sys.stderr)
        return 2
    script_step(log, "seed", "plan", "ok", customers=len(plan.customers), aliases=len(plan.aliases),
                policy_items=len(plan.policy), catalog_items=len(plan.catalog))

    # Tuned masking values, slot settings and option names and the redaction
    # setting: read the live ones first. An apply must read them (or it would overwrite
    # tuning); a dry run reads the policy ones when the policy table is named, and the option
    # names when the catalog table is named.
    slots_merge: SlotsMerge | None = None
    redaction_merge: RedactionMerge | None = None
    prompts_merge: PromptsMerge | None = None
    report_parts: list[str] = [render_prompt_report(prompt_gate)]
    if args.apply or table_named(outputs, overrides, "policy"):
        try:
            if store is None:
                store = _make_store(args, tables)
            merge = merge_masking(plan.policy, read_live_masking(store, plan.policy),
                                  overwrite=args.overwrite_masking)
            smerge = merge_slots(merge.items, read_live_slots(store, merge.items),
                                 overwrite=args.overwrite_slots)
            rmerge = merge_redaction(smerge.items, read_live_redaction(store),
                                     overwrite=args.overwrite_redaction)
        except ScriptError as exc:
            script_error(log, "seed", exc, step="masking", decision="exit_2")
            print(f"seed: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            if args.apply:
                script_step(log, "seed", "masking", "error")
                script_error(log, "seed", exc, step="masking", decision="reraise")
                raise
            script_error(log, "seed", exc, step="masking", decision="dry_run_without_live_values")
            report_parts.append(
                f"== masking values: the live table couldn't be read ({type(exc).__name__}); "
                "an apply keeps the live values unless --overwrite-masking\n"
                "== slot settings: the live table couldn't be read; an apply keeps the live "
                "values unless --overwrite-slots\n"
                "== redaction setting: the live table couldn't be read; an apply keeps the live "
                "setting unless --overwrite-redaction")
        else:
            slots_merge, redaction_merge = smerge, rmerge
            plan.policy = redaction_merge.items
            report_parts += [render_masking_report(merge), render_slots_report(slots_merge),
                             render_redaction_report(redaction_merge)]
            script_step(log, "seed", "masking", "ok", differences=len(merge.differences),
                        invalid=len(merge.invalid), overwrite=bool(args.overwrite_masking))
            script_step(log, "seed", "slots", "ok", differences=len(slots_merge.differences),
                        added=len(slots_merge.added), new=len(slots_merge.new),
                        invalid=len(slots_merge.invalid), overwrite=bool(args.overwrite_slots))
            script_step(log, "seed", "redaction", "ok", differences=len(redaction_merge.differences),
                        added=len(redaction_merge.added), dropped=len(redaction_merge.dropped),
                        switched_off=redaction_merge.switched_off, new=redaction_merge.new,
                        invalid=len(redaction_merge.invalid), overwrite=bool(args.overwrite_redaction))
    else:
        report_parts.append(
            "== masking values: the live table wasn't read (no policy table name); pass "
            "--cdk-outputs or --table policy=NAME to see differences. An apply keeps the "
            "live values unless --overwrite-masking\n"
            "== slot settings: the live table wasn't read (no policy table name). An apply "
            "keeps the live values unless --overwrite-slots\n"
            "== redaction setting: the live table wasn't read (no policy table name). An apply "
            "keeps the live setting unless --overwrite-redaction")
    if args.apply or table_named(outputs, overrides, "catalog"):
        try:
            if store is None:
                store = _make_store(args, tables)
            pmerge = merge_catalog_prompts(plan.catalog, read_live_catalog(store),
                                           overwrite=args.overwrite_prompts,
                                           overwrite_options=args.overwrite_slots)
        except Exception as exc:
            if args.apply:
                script_step(log, "seed", "catalog_prompts", "error")
                script_error(log, "seed", exc, step="catalog_prompts", decision="reraise")
                raise
            script_error(log, "seed", exc, step="catalog_prompts", decision="dry_run_without_live_values")
            report_parts.append(f"== prompts: the catalog table couldn't be read ({type(exc).__name__}); "
                                "an apply keeps the live texts unless --overwrite-prompts")
        else:
            prompts_merge = pmerge
            plan.catalog = prompts_merge.items
            report_parts.append(render_prompts_report(prompts_merge))
            script_step(log, "seed", "catalog_prompts", "ok", kept=len(prompts_merge.kept),
                        legacy=len(prompts_merge.legacy), updated=len(prompts_merge.updated),
                        hashed=len(prompts_merge.hashed), replaced=len(prompts_merge.replaced), new=len(prompts_merge.new),
                        live_only=len(prompts_merge.live_only),
                        overwrite=bool(args.overwrite_prompts))
    else:
        report_parts.append("== prompts: the catalog table wasn't read (no catalog table name). "
                            "An apply keeps the live texts unless --overwrite-prompts")
    report = "\n".join(report_parts)

    if not args.apply:
        script_step(log, "seed", "apply", "dry_run")
        print(f"DRY RUN (stage {args.stage}); nothing is written. Tables: "
              + ", ".join(f"{k}={tables[k]}" for k in SEEDED_TABLES))
        print(render_plan(plan))
        print(report)
        return 0

    if store is None:  # made above for an apply; kept for the type checker
        raise RuntimeError("no store for an apply")
    print(report)
    if args.overwrite_slots and slots_merge is not None:
        # A full overwrite of console-tuned settings is logged at WARNING.
        obs.emit(log, logging.WARNING, "script_step", script="seed", step="slots",
                 outcome="overwrite", items=len(slot_merge_keys(plan.policy)),
                 differences=len(slots_merge.differences),
                 fields=[f for f, _, _ in slots_merge.differences],
                 option_names=sum(k.startswith(OPTION_PREFIX) for k, _ in prompts_merge.replaced)
                 if prompts_merge is not None else 0)
    if prompts_merge is not None and (prompts_merge.replaced or
                                      (prompts_merge.overwrite and prompts_merge.live_only)):
        # Console prompt edits overwritten (--overwrite-prompts) are logged at WARNING: keys only.
        obs.emit(log, logging.WARNING, "script_step", script="seed", step="prompts",
                 outcome="overwrite", replaced=[_entry(k) for k in prompts_merge.replaced],
                 dropped_keys=prompts_merge.live_only if prompts_merge.overwrite else [])
    if prompt_gate.unvalidated:
        # Unvalidated agent prompts seeded in dev (--allow-unvalidated-prompt) are logged at WARNING.
        obs.emit(log, logging.WARNING, "script_step", script="seed", step="agent_prompts",
                 outcome="unvalidated", texts=[_entry(k) for k in prompt_gate.unvalidated])
    if args.overwrite_redaction and redaction_merge is not None:
        # A full overwrite of the stored redaction setting is logged at WARNING.
        obs.emit(log, logging.WARNING, "script_step", script="seed", step="redaction",
                 outcome="overwrite", differences=len(redaction_merge.differences),
                 fields=[f for f, _, _ in redaction_merge.differences])
    counts = apply_plan(plan, store)
    print(f"seeded stage {args.stage}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    start_logging("seed")
    sys.exit(main())
