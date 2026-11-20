"""Show and tune latency-masking values in the policy table.

The editor for masking values until the admin console exists. It reads and writes the live
policy table, never ``data/policy.json`` (a reseed keeps what is set here: ``scripts/seed.py``).

Usage::

    # every policy-wide value and every tool's mask (own value or default)
    uv run python scripts/masking.py show --stage dev --cdk-outputs infra/cdk-outputs.json
    uv run python scripts/masking.py show --stage dev --cdk-outputs ... --tool get_balance

    # one tool (dry run: prints the change); then the same with --apply
    uv run python scripts/masking.py set --stage dev --cdk-outputs ... --tool get_balance --after-ms 900
    uv run python scripts/masking.py set ... --tool get_balance --after-ms default   # use tool_after_ms
    uv run python scripts/masking.py set ... --tool get_balance --phrase-key mask.get_balance
    uv run python scripts/masking.py set ... --tool change_address --phrase-key off   # unmasked

    # policy-wide values (``FIELD=default`` removes a field: the bridge then uses its default)
    uv run python scripts/masking.py set ... --masking tool_after_ms=500 --masking after_ms=2500

Rules:

* **Dry run by default**: ``set`` prints each change (field: old -> new) and writes nothing
  unless ``--apply``.
* **Strict validation before any write** with :mod:`owlbank.masking_config` (the bridge's own
  rules: types, ranges 0–10,000 ms, ``end_mode``, ``silence_action``). A new ``--phrase-key`` must
  also have its whole phrase pool in the live catalog table in every catalog language, with every
  placeholder's words (:func:`~owlbank.masking_config.validate_catalog_for_tool`). An invalid value
  is refused with the field and the reason (exit 2) and nothing is written.
* **Only the named fields change.** The rest of the item is written back as read.
* **No lost updates.** The write is conditional on the item being unchanged since it was read:
  if the item has a numeric ``rev`` (the console's version), the condition is ``rev`` equal to
  the value read and the write bumps it; otherwise the condition is every attribute equal to the
  value read (an attribute added in between is not detected), and no ``rev`` is added. A
  conflict is reported (exit 3) and nothing is written: run the command again.

Exit codes: 0 done (or dry run), 2 refused (bad arguments, invalid value, unknown tool, no
table name), 3 the item changed since it was read.

Tables come from ``--table LOGICAL=NAME`` or ``--cdk-outputs`` as in ``scripts/seed.py``; the
policy table (and for ``--phrase-key``, the catalog table) must be named explicitly.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):  # run as a file: make `scripts` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owlbank.catalog import CatalogError, PromptCatalog
from owlbank.masking_config import (
    MASKING_FIELDS,
    MaskingSettings,
    masking_parts,
    parse_masking,
    parse_tool_mask,
    tool_mask_parts,
    validate_catalog_for_tool,
)
from owlbank.policy import META_PK, PolicyError
from owlbank.store import ConditionFailed, DynamoStore, Store, StoreError
from scripts._common import (
    ScriptError,
    add_mode_args,
    load_cdk_outputs,
    parse_pairs,
    resolve_region,
    script_error,
    script_start,
    script_step,
    start_logging,
)
from scripts.seed import resolve_tables

log = logging.getLogger("owlbank.masking")

SCRIPT = "masking"
MASKING_PK = "masking"
TOOL_PREFIX = "tool#"
REV = "rev"
DEFAULT = "default"
OFF = "off"
#: Item attributes that aren't masking fields.
_META_ATTRS = ("pk", REV)
_INT_FIELDS = frozenset(MASKING_FIELDS) - {"end_mode", "silence_action", "chars_per_s"}


class Conflict(Exception):
    """The item changed between the read and the write."""


# --------------------------------------------------------------------------- conditional writes


class PolicyWriter(Protocol):
    """Writes one policy item only if it still equals ``expected`` (``None`` = absent)."""

    def put_if_unchanged(self, item: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
        """Write ``item``; :class:`Conflict` if the stored item isn't ``expected`` any more."""
        ...


def _has_rev(item: Mapping[str, Any] | None) -> bool:
    rev = None if item is None else item.get(REV)
    return isinstance(rev, int) and not isinstance(rev, bool)


def with_rev(item: Mapping[str, Any], expected: Mapping[str, Any] | None) -> dict[str, Any]:
    """``item`` with ``rev`` bumped when the read item had one (else unchanged, no ``rev`` added)."""
    out = dict(item)
    if expected is not None and _has_rev(expected):
        out[REV] = int(expected[REV]) + 1
    return out


class DynamoPolicyWriter:
    """Conditional ``put_item`` on the DynamoDB policy table (atomic)."""

    def __init__(self, table: Any) -> None:
        self._table = table  # a boto3 ``Table`` resource

    @staticmethod
    def condition(expected: Mapping[str, Any] | None) -> dict[str, Any]:
        """The ``put_item`` condition arguments for ``expected``."""
        if expected is None:
            return {"ConditionExpression": "attribute_not_exists(#pk)",
                    "ExpressionAttributeNames": {"#pk": "pk"}}
        attrs = [REV] if _has_rev(expected) else sorted(expected)
        if "pk" not in attrs:
            attrs = ["pk", *attrs]
        names = {f"#a{i}": a for i, a in enumerate(attrs)}
        values = {f":v{i}": expected[a] for i, a in enumerate(attrs)}
        expr = " AND ".join(f"#a{i} = :v{i}" for i in range(len(attrs)))
        return {"ConditionExpression": expr, "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": values}

    def put_if_unchanged(self, item: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
        try:
            self._table.put_item(Item=dict(item), **self.condition(expected))
        except Exception as exc:
            code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
            if code == "ConditionalCheckFailedException":
                raise Conflict(str(item.get("pk"))) from None
            raise


class StorePolicyWriter:
    """For any :class:`Store` (tests, the in-memory store): re-read, compare, then put. The same
    rule as :class:`DynamoPolicyWriter`, but not atomic; the DynamoDB writer is used for real tables."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def put_if_unchanged(self, item: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
        now = self._store.get("policy", {"pk": item["pk"]})
        if expected is None:
            unchanged = now is None
        elif now is None:
            unchanged = False
        elif _has_rev(expected):
            unchanged = now.get(REV) == expected[REV]
        else:
            unchanged = all(k in now and now[k] == v for k, v in expected.items())
        if not unchanged:
            raise Conflict(str(item["pk"]))
        try:
            self._store.put("policy", item)
        except ConditionFailed:
            raise Conflict(str(item["pk"])) from None


# --------------------------------------------------------------------------- reading


def _fields(item: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if item is None else {k: v for k, v in item.items() if k not in _META_ATTRS}


def tool_names(store: Store) -> list[str]:
    meta = store.get("policy", {"pk": META_PK})
    if meta is None:
        raise ScriptError("the policy table has no meta item (seed it first: scripts/seed.py)")
    return [str(t) for t in meta.get("tools", [])]


def _fmt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _cps(value: float) -> str:
    return f"{value:g}"


def render_show(store: Store, tool: str | None = None) -> str:
    """Policy-wide values and every tool's mask, each marked as set in the table or default."""
    names = tool_names(store)
    if tool is not None and tool not in names:
        raise ScriptError(f"no tool {tool!r} in the policy table; known: {sorted(names)}")
    raw = _fields(store.get("policy", {"pk": MASKING_PK}))
    settings, problems = masking_parts(raw)
    bad = {p.field.split(".", 1)[1]: p.message for p in problems if "." in p.field}
    lines = ["Policy-wide (item `masking`):"]
    for name in MASKING_FIELDS:
        value = getattr(settings, name)
        shown = _cps(value) if name == "chars_per_s" else str(value)
        if name in bad and name in raw:
            note = f"default; the table's {_fmt(raw[name])} is INVALID: {bad[name]}"
        elif name in raw:
            note = "set in the table"
        else:
            note = "default; not set in the table"
        lines.append(f"  {name:<20} {shown:<10} ({note})")
    lines.extend(f"  WARNING: {p}" for p in problems if p.field.split(".", 1)[-1] not in MASKING_FIELDS)
    lines.append("Tools (item `tool#<name>`, field `mask`):")
    for name in names if tool is None else [tool]:
        lines.extend(_show_tool(store, name, settings))
    return "\n".join(lines)


def _show_tool(store: Store, name: str, settings: MaskingSettings) -> list[str]:
    item = store.get("policy", {"pk": f"{TOOL_PREFIX}{name}"})
    if item is None:
        return [f"  {name}: no item in the table"]
    raw = item.get("mask")
    mask, problems = tool_mask_parts(name, raw)
    warn = [f"    WARNING: {p} (the bridge uses the default)" for p in problems]
    if mask is None:
        return [f"  {name}: unmasked", *warn]
    after = (f"{mask.after_ms} (tool's own)" if mask.after_ms is not None
             else f"{settings.tool_after_ms} (default: masking.tool_after_ms)")
    hand = (f"{mask.handoff_wait_ms} (tool's own)" if mask.handoff_wait_ms is not None
            else f"{settings.handoff_wait_ms} (default: masking.handoff_wait_ms)")
    return [f"  {name}: phrase_key {mask.phrase_key}, after_ms {after}, handoff_wait_ms {hand}", *warn]


# --------------------------------------------------------------------------- changes


def _delay(value: str, flag: str) -> int | None:
    """``default`` -> ``None`` (remove the tool's own value); else an integer."""
    if value == DEFAULT:
        return None
    try:
        return int(value.strip())
    except ValueError:
        raise ScriptError(f"{flag}: expected an integer number of ms or 'default', got {value!r}") from None


def new_tool_item(name: str, item: Mapping[str, Any], *, after_ms: str | None,
                  handoff_wait_ms: str | None, phrase_key: str | None) -> dict[str, Any]:
    """The tool item with only ``mask`` changed, validated strictly (``ScriptError`` naming the
    field and reason). ``phrase_key`` ``off`` makes the tool unmasked (``mask`` removed)."""
    if after_ms is None and handoff_wait_ms is None and phrase_key is None:
        raise ScriptError("set --tool: give --after-ms, --handoff-wait-ms or --phrase-key")
    out = copy.deepcopy(dict(item))
    if phrase_key == OFF:
        if after_ms is not None or handoff_wait_ms is not None:
            raise ScriptError("--phrase-key off turns masking off; don't combine it with delays")
        out.pop("mask", None)
        return out
    current = item.get("mask")
    if current is None and phrase_key is None:
        raise ScriptError(f"tool {name} is unmasked; give --phrase-key mask.<...> to mask it")
    mask: dict[str, Any] = dict(current) if isinstance(current, Mapping) else {}
    if phrase_key is not None:
        mask["phrase_key"] = phrase_key
    for field_name, value, flag in (("after_ms", after_ms, "--after-ms"),
                                    ("handoff_wait_ms", handoff_wait_ms, "--handoff-wait-ms")):
        if value is None:
            continue
        parsed = _delay(value, flag)
        if parsed is None:
            mask.pop(field_name, None)
        else:
            mask[field_name] = parsed
    try:
        parse_tool_mask(name, mask, strict=True)
    except PolicyError as exc:
        raise ScriptError(f"refused: {exc}") from None
    out["mask"] = mask
    return out


def _masking_value(name: str, value: str) -> Any:
    if name in _INT_FIELDS:
        try:
            return int(value.strip())
        except ValueError:
            raise ScriptError(f"refused: masking.{name}: must be an integer, got {value!r}") from None
    if name == "chars_per_s":
        try:
            num = float(value)
        except ValueError:
            raise ScriptError(f"refused: masking.{name}: must be a number, got {value!r}") from None
        # the store holds no floats: an integral value as int, else the decimal string as typed
        return int(num) if num.is_integer() else value.strip()
    return value


def new_masking_item(item: Mapping[str, Any] | None, pairs: Mapping[str, str]) -> dict[str, Any]:
    """The ``masking`` item with only the named fields changed (``default`` removes one),
    validated strictly."""
    unknown = sorted(set(pairs) - set(MASKING_FIELDS))
    if unknown:
        raise ScriptError(f"refused: masking.{unknown[0]}: unknown field; known: {list(MASKING_FIELDS)}")
    out = copy.deepcopy(dict(item)) if item is not None else {"pk": MASKING_PK}
    for name, value in pairs.items():
        if value == DEFAULT:
            out.pop(name, None)
        else:
            out[name] = _masking_value(name, value)
    try:
        parse_masking(_fields(out), strict=True)
    except PolicyError as exc:
        raise ScriptError(f"refused: {exc}") from None
    return out


def check_phrase_key(name: str, item: Mapping[str, Any], catalog: PromptCatalog) -> None:
    """The tool's whole phrase pool must be in the catalog in every language (.20)."""
    mask = parse_tool_mask(name, item.get("mask"), strict=True)
    if mask is None:
        return
    problems = validate_catalog_for_tool(name, mask, catalog, catalog.languages)
    if problems:
        raise ScriptError("refused: " + "; ".join(str(p) for p in problems[:5])
                          + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""))


def describe_changes(base: str, old: Mapping[str, Any] | None, new: Mapping[str, Any],
                     fields: Sequence[str]) -> list[str]:
    old = old or {}
    lines: list[str] = []
    for f in fields:
        a, b = old.get(f, "(not set)"), new.get(f, "(not set)")
        if a == b:
            continue
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for k in sorted({*a, *b}):
                if a.get(k, "(not set)") != b.get(k, "(not set)"):
                    lines.append(f"  {base}.{f}.{k}: {_fmt(a.get(k, '(not set)'))} -> "
                                 f"{_fmt(b.get(k, '(not set)'))}")
        else:
            lines.append(f"  {base}.{f}: {_fmt(a)} -> {_fmt(b)}")
    return lines


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage", required=True, help="deployment stage, e.g. dev, test, prod")
    common.add_argument("--cdk-outputs", help="cdk deploy --outputs-file JSON")
    common.add_argument("--table", action="append", metavar="LOGICAL=NAME", help="physical table name")
    common.add_argument("--region", help="AWS region (default: a table ARN's, then AWS_DEFAULT_REGION "
                                         "or AWS_REGION, then us-east-1)")
    p = argparse.ArgumentParser(description="Show and tune latency-masking values in the policy table.")
    sub = p.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show", parents=[common], help="print the policy-wide values and every tool's mask")
    show.add_argument("--tool", help="only this tool")
    st = sub.add_parser("set", parents=[common], help="change one tool's or the policy-wide values "
                                                      "(dry run by default)")
    target = st.add_mutually_exclusive_group(required=True)
    target.add_argument("--tool", help="the tool whose mask to change")
    target.add_argument("--masking", action="append", metavar="FIELD=VALUE|default",
                        help=f"a policy-wide value; fields: {', '.join(MASKING_FIELDS)}")
    st.add_argument("--after-ms", metavar="N|default", help="with --tool: the phrase delay")
    st.add_argument("--handoff-wait-ms", metavar="N|default", help="with --tool: the hand-off hold")
    st.add_argument("--phrase-key", metavar="KEY|off", help="with --tool: the catalog key, or off")
    add_mode_args(st)
    return p


def _open(args: argparse.Namespace, store: Store | None, writer: PolicyWriter | None,
          require: Sequence[str]) -> tuple[Store, PolicyWriter, dict[str, str]]:
    tables = resolve_tables(load_cdk_outputs(args.cdk_outputs), parse_pairs(args.table, what="--table"),
                            require=require)
    if store is None:
        import boto3

        region = resolve_region(args.region, arns=tables.values(), logger=log)
        resource = boto3.resource("dynamodb", region_name=region)
        store = DynamoStore(tables, resource=resource, region_name=region)
        writer = writer or DynamoPolicyWriter(resource.Table(tables["policy"]))
    return store, writer or StorePolicyWriter(store), tables


def _set(args: argparse.Namespace, store: Store, writer: PolicyWriter, tables: Mapping[str, str]) -> int:
    if args.tool is not None:
        pk = f"{TOOL_PREFIX}{args.tool}"
        if args.tool not in tool_names(store):
            raise ScriptError(f"refused: no tool {args.tool!r} in the policy table")
        old = store.get("policy", {"pk": pk})
        if old is None:
            raise ScriptError(f"refused: the policy table lists {args.tool!r} but has no {pk} item")
        new = new_tool_item(args.tool, old, after_ms=args.after_ms,
                            handoff_wait_ms=args.handoff_wait_ms, phrase_key=args.phrase_key)
        if args.phrase_key not in (None, OFF):
            check_phrase_key(args.tool, new, PromptCatalog.from_store(store))
        changes = describe_changes(f"tools.{args.tool}", old, new, ["mask"])
    else:
        if args.after_ms is not None or args.handoff_wait_ms is not None or args.phrase_key is not None:
            raise ScriptError("--after-ms, --handoff-wait-ms and --phrase-key go with --tool")
        pk = MASKING_PK
        old = store.get("policy", {"pk": pk})
        new = new_masking_item(old, parse_pairs(args.masking, what="--masking"))
        changes = describe_changes("masking", old, new, MASKING_FIELDS)
    script_step(log, SCRIPT, "validate", "ok", item=pk, changes=len(changes))
    if not changes:
        print(f"no change to {pk} (stage {args.stage}, table {tables['policy']})")
        script_step(log, SCRIPT, "write", "skipped", item=pk)
        return 0
    head = f"{'DRY RUN, nothing is written' if not args.apply else 'writing'}: {pk} " \
           f"(stage {args.stage}, table {tables['policy']})"
    print("\n".join([head, *changes]))
    if not args.apply:
        script_step(log, SCRIPT, "write", "dry_run", item=pk)
        print("pass --apply to write")
        return 0
    writer.put_if_unchanged(with_rev(new, old), old)
    script_step(log, SCRIPT, "write", "changed", item=pk)
    print("written; running bridges pick it up within 60 s (policy cache)")
    return 0


def main(argv: list[str] | None = None, *, store: Store | None = None,
         writer: PolicyWriter | None = None) -> int:
    args = build_parser().parse_args(argv)
    apply = bool(getattr(args, "apply", False))
    script_start(log, SCRIPT, stage=args.stage, command=args.command,
                 mode="apply" if apply else "dry_run")
    try:
        needs_catalog = args.command == "set" and args.phrase_key not in (None, OFF)
        store, writer, tables = _open(args, store, writer,
                                      ("policy", "catalog") if needs_catalog else ("policy",))
        if args.command == "show":
            print(f"stage {args.stage}, policy table {tables['policy']}")
            print(render_show(store, args.tool))
            script_step(log, SCRIPT, "show", "ok")
            return 0
        return _set(args, store, writer, tables)
    except ScriptError as exc:
        script_error(log, SCRIPT, exc, step=args.command, decision="exit_2")
        print(f"masking: {exc}", file=sys.stderr)
        return 2
    except Conflict as exc:
        script_error(log, SCRIPT, exc, step="write", decision="exit_3")
        print(f"masking: {exc} changed since it was read; nothing was written. Run the command "
              "again.", file=sys.stderr)
        return 3
    except (PolicyError, CatalogError, StoreError) as exc:
        script_error(log, SCRIPT, exc, step=args.command, decision="exit_2")
        print(f"masking: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    start_logging(SCRIPT)
    sys.exit(main())
