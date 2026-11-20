"""Config routes: tools, masking, prompts, placeholders, audit and restore.

**Validation is the bridge's own**. A tool write is checked by
:func:`owlbank.masking_config.parse_tool_mask` (strict), :func:`owlbank.policy.parse_confirm`
and the whole policy parsed with the new row (:meth:`owlbank.policy.Policy.from_dict`, the
same rules as :meth:`~owlbank.policy.Policy.to_items`); a masked tool's phrase pool must be
valid in the catalog (:func:`~owlbank.masking_config.validate_catalog_for_tool`). A masking
write is :func:`~owlbank.masking_config.parse_masking` (strict), so the API accepts exactly
what the bridge's lenient loader uses without an ERROR. A phrase write is
:func:`~owlbank.masking_config.validate_phrase` for the tool that owns the key, in every
language; an ``arg.*`` word must be non-empty text without braces, in every language. An
``option.<slot>.<value>`` spoken name (phase 2) must be
non-empty text without braces in every language, for a choice slot ``meta.slots`` lists and a
value of the option form; it must not make a spoken name or alias belong to two of the slot's
options in one language (:func:`owlbank.slot_policy.alias_conflicts`).

**Concurrency**. Every policy and catalog item carries ``rev`` (missing =
0). A write sends the ``rev`` it read; the write is one :meth:`~owlbank.store.Store.transact_write`
of the new item (``rev + 1``, conditional on ``rev`` = the value read) and its audit entry,
so a stale ``rev`` changes nothing and every change has exactly one audit entry. A
mismatch is 409 with the current item. A prompt key's ``rev`` is the highest ``rev`` of its
language items; a write sets every language item to that plus one.

**Scope**: only the policy items ``meta`` (read),
``tool#<name>`` and ``masking``, and the catalog's entries (plus its ``_meta`` key index when a
new ``mask.*``, ``arg.*`` or ``option.*`` key is added) are read or written.

**Prompt editing by exclusion**. Every catalog key is editable except the protected
ones (:func:`protected_reason`): the deterministic read-backs and outcomes
(:func:`owlbank.policy.confirm_prompt_keys` of any tool, and every ``confirm.*`` key). The
model instructions (``agent.*``) are editable like any other key and audited the same way;
``scripts/prompt_suite.py`` is how to check an edit per model. A
protected key is refused with 403 ``protected`` before anything is written or audited. Every
edit of an existing key must be non-empty, at most :data:`MAX_PROMPT_CHARS` characters, free
of ``<`` and ``>`` (no SSML or markup on the TTS path), have no stray brace, and keep exactly
the ``{placeholder}`` set of the current text in that language (400 ``placeholders``). An
``agent.*`` text is never spoken: it may be up to :data:`MAX_AGENT_PROMPT_CHARS` characters
and contain ``<`` and ``>``; the other rules hold, except that its
placeholder set is its key's allow-list (:data:`owlbank.agent_prompt.ALLOWED_PLACEHOLDERS`),
not the current text's, for an edit, a restore or a new key alike:
so a restore of a revision from before that feature, whose ``agent.system.*`` text had
``{firstName}``, is refused. The console's
``lib/validate.js`` applies the same rules (``tests/fixtures/prompt_text_rules.json``). A
``mask.*`` phrase is exempt from that last rule only: its own check
(:func:`~owlbank.masking_config.validate_phrase`) allows any placeholder the owning tool has,
with a catalog word for every value in every language, so a phrase may gain or lose
``{account}`` but never take a placeholder the tool doesn't have. The ``mask.*``, ``arg.*``
and ``option.*`` checks stay on top, and only those prefixes may add a new key.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

from owlbank import agent_prompt, obs
from owlbank import catalog as catalog_mod
from owlbank import masking_config as mc
from owlbank import policy as policy_mod
from owlbank.catalog import CatalogError, PromptCatalog
from owlbank.masking_config import MaskingSettings
from owlbank.policy import MASK_GENERIC_KEYS, Policy, PolicyError, PolicySectionError
from owlbank.slot_policy import alias_conflicts, load_slot
from owlbank.store import Expect, Store, TransactionCancelled, TxPut

from . import audit
from .audit import AuditClock
from .config import TAKES_EFFECT_S
from .deploy_settings import DEPLOY_PK, MARKER_SK
from .http import ApiError, Caller

log = logging.getLogger(__name__)

REV = policy_mod.REV_ATTR
TOOL_FIELDS: tuple[str, ...] = ("min_level", "confirm", "mask")
OPTION_PREFIX = "option."
NEW_KEY_PREFIXES: tuple[str, ...] = (mc.MASK_PREFIX, mc.ARG_PREFIX, OPTION_PREFIX)
"""The only prefixes under which a prompt write may add a key the catalog doesn't have."""
MAX_PROMPT_CHARS = 600
"""Longest spoken prompt text the console may write (the catalog has no limit of its own)."""
AGENT_PREFIX = "agent."
"""Model instructions: system prompts the model reads, never spoken."""
MAX_AGENT_PROMPT_CHARS = 16_000
"""Longest ``agent.*`` text the console may write; the shipped ones are about 4,000-4,800."""
MARKUP_CHARS: tuple[str, ...] = ("<", ">")
CONFIRM_PREFIX = "confirm."
PROTECTED_READBACK = "protected read-back"
_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PROMPT_KEY = re.compile(r"[a-z][A-Za-z0-9_.-]{0,159}")
_PROMPT_PREFIX = re.compile(r"[a-z][a-z0-9_.-]{1,127}")
_OPTION_KEY = re.compile(r"option\.([a-z][a-z0-9_]{0,63})\.([a-z][a-z0-9_]{0,31})")
_POOL_SUFFIX = re.compile(r"[1-9][0-9]{0,5}")
_AUDIT_ITEM = re.compile(r"(policy|catalog|user|deploy)#[A-Za-z0-9_.#@+:-]{1,200}")
MAX_AUDIT_LIMIT = 200
DEFAULT_AUDIT_LIMIT = 50


def _rev(item: Mapping[str, Any] | None) -> int:
    value = (item or {}).get(REV, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _rev_of_body(body: Mapping[str, Any]) -> int:
    rev = body.get("rev")
    if not isinstance(rev, int) or isinstance(rev, bool) or rev < 0:
        raise ApiError(400, "invalid", "rev must be the item's rev as read (an integer >= 0)",
                       field="rev")
    return rev


def protected_reason(key: str, tools: Iterable[str] = ()) -> str | None:
    """Why ``key`` is read-only in the console, or ``None`` when it isn't protected.

    The read-back confirmations and outcomes a deterministic confirm speaks are protected for
    every tool that could be configured with one (``tools`` plus the default deterministic
    tools), and every ``confirm.*`` key besides. Model instructions (``agent.*``) are not
    protected: edits are audited like any other."""
    names = {*tools, *policy_mod.DEFAULT_CONFIRM}
    if key.startswith(CONFIRM_PREFIX) or any(key in policy_mod.confirm_prompt_keys(t) for t in names):
        return PROTECTED_READBACK
    return None


def _placeholder_set(text: Any) -> frozenset[str]:
    return frozenset(mc.placeholder_names(text)) if isinstance(text, str) else frozenset()


def _check_prompt_text(key: str, text: str, field: str, expected: frozenset[str] | None) -> None:
    """The generic rules for one language's text; ``expected`` is the
    current text's placeholder set, or ``None`` when no set is enforced (a key the catalog
    doesn't have yet, or a ``mask.*`` phrase, which :func:`~owlbank.masking_config.validate_phrase`
    checks instead). An ``agent.*`` text is read by the model, not spoken: it may be up to
    :data:`MAX_AGENT_PROMPT_CHARS` long and contain ``<`` and ``>``. Its
    placeholders are checked against the key's allow-list, not ``expected``, for an edit, a
    restore or a new key alike: the message is
    :func:`~owlbank.agent_prompt.placeholder_problems`', which names the allowed set."""
    agent = key.startswith(AGENT_PREFIX)
    limit = MAX_AGENT_PROMPT_CHARS if agent else MAX_PROMPT_CHARS
    if not text.strip():
        raise ApiError(400, "invalid", "must be non-empty text", field=field)
    if len(text) > limit:
        raise ApiError(400, "invalid", f"must be at most {limit} characters", field=field)
    if not agent and any(c in text for c in MARKUP_CHARS):
        raise ApiError(400, "invalid", "must not contain SSML or markup ('<' or '>')", field=field)
    if agent_prompt.allowed_placeholders(key) is not None:
        problem = agent_prompt.placeholder_problems(key, text)
        if problem is not None:
            raise ApiError(400, "placeholders", problem, field=field)
        return
    if expected is None:
        return
    if mc.has_stray_brace(text):
        raise ApiError(400, "placeholders", "has a '{' or '}' that isn't a {placeholder}", field=field)
    got = _placeholder_set(text)
    if got != expected:
        want = ", ".join(f"{{{n}}}" for n in sorted(expected)) or "none"
        raise ApiError(400, "placeholders", f"must keep exactly the placeholders of the current "
                                            f"text ({want})", field=field)


def _only_keys(body: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(str(k) for k in body if k not in allowed)
    if unknown:
        raise ApiError(400, "invalid", "unknown field", field=unknown[0])


def spoken_names(store: Store, slot: str, values: Iterable[str], cmeta: Mapping[str, Any] | None,
                 override: Mapping[tuple[str, str], str] | None = None) -> dict[str, dict[str, str]]:
    """Each option value's catalog ``option.<slot>.<value>`` text per language, as the front door's
    catalog serves it (a key in the catalog index, a published, non-empty text). ``override`` maps
    ``(key, lang)`` to a proposed text (a prompt write)."""
    keys = set((cmeta or {}).get("keys") or [])
    override = override or {}
    out: dict[str, dict[str, str]] = {}
    for value in values:
        key = f"{OPTION_PREFIX}{slot}.{value}"
        names: dict[str, str] = {}
        if key in keys:
            for it in store.query("catalog", {"key": key}):
                lang, text = it.get("lang"), it.get("text")
                if isinstance(lang, str) and isinstance(text, str) and text.strip() \
                        and it.get("status", "published") == "published":
                    names[lang] = text
        for (k, lang), text in override.items():
            if k == key:
                names[lang] = text
        out[value] = names
    return out


class ConfigRoutes:
    def __init__(self, store: Store, *, clock: AuditClock, retention_days: int) -> None:
        self.store, self.clock, self.retention_days = store, clock, retention_days

    # ------------------------------------------------------------------ loading

    def _meta(self) -> dict[str, Any]:
        meta = self.store.get("policy", {"pk": policy_mod.META_PK})
        if meta is None:
            raise ApiError(503, "not_seeded", "the policy table has no meta item")
        return meta

    @staticmethod
    def _tools(meta: Mapping[str, Any]) -> list[str]:
        tools = meta.get("tools")
        return [t for t in tools if isinstance(t, str)] if isinstance(tools, list) else []

    @staticmethod
    def _reasons(meta: Mapping[str, Any]) -> tuple[str, ...]:
        reasons = meta.get("reasons")
        return tuple(r for r in reasons if isinstance(r, str)) if isinstance(reasons, list) else ()

    def _catalog_doc(self) -> dict[str, Any]:
        try:
            return catalog_mod._doc_from_store(self.store)
        except CatalogError:
            raise ApiError(503, "not_seeded", "the catalog table has no meta item") from None

    def _catalog(self, doc: Mapping[str, Any] | None = None) -> PromptCatalog:
        try:
            return PromptCatalog.from_dict(doc if doc is not None else self._catalog_doc())
        except CatalogError:
            raise ApiError(503, "catalog_invalid", "the catalog table is invalid") from None

    def _catalog_meta(self) -> dict[str, Any]:
        meta = self.store.get("catalog", {"key": catalog_mod.META_KEY, "lang": catalog_mod.META_LANG})
        if meta is None:
            raise ApiError(503, "not_seeded", "the catalog table has no meta item")
        return meta

    def _masking_item(self) -> dict[str, Any] | None:
        return self.store.get("policy", {"pk": "masking"})

    def _masking_settings(self) -> MaskingSettings:
        item = self._masking_item()
        raw = None if item is None else {k: v for k, v in item.items() if k not in ("pk", REV)}
        return mc.masking_parts(raw)[0]

    def _tool_row(self, name: str, meta: Mapping[str, Any]) -> dict[str, Any]:
        if not _TOOL_NAME.fullmatch(name) or name not in self._tools(meta):
            raise ApiError(404, "not_found", "no such tool")
        row = self.store.get("policy", {"pk": f"tool#{name}"})
        if row is None:
            raise ApiError(404, "not_found", "no such tool")
        return row

    # ------------------------------------------------------------------ the write

    def _commit(self, put: TxPut, extra_puts: list[TxPut], *, caller: Caller, pk: str,
                action: str, kind: str, target_id: str, changes: list[dict[str, Any]],
                security_change: bool, new_rev: int,
                restored_from: Mapping[str, str] | None) -> dict[str, Any]:
        """The item, any index item, and the audit entry in one transaction."""
        entry = audit.entry(pk=pk, action=action, kind=kind, target_id=target_id, actor=caller,
                            changes=changes, security_change=security_change,
                            now=self.clock.now(), suffix=self.clock.suffix(),
                            retention_days=self.retention_days, rev=new_rev,
                            restored_from=restored_from)
        self.store.transact_write([put, *extra_puts, TxPut(audit.AUDIT_TABLE, entry, "not_exists")])
        obs.emit(log, logging.INFO, "admin_config_write", action=action, target_kind=kind,
                 target_id=target_id, actor_sub=caller.sub, rev=new_rev,
                 security_change=bool(security_change), fields=[c["field"] for c in changes])
        return {"pk": entry["pk"], "sk": entry["sk"]}

    # ------------------------------------------------------------------ tools

    def _tool_view(self, name: str, row: Mapping[str, Any], *, masking: MaskingSettings,
                   catalog: PromptCatalog, reasons: tuple[str, ...]) -> dict[str, Any]:
        kind = row.get("kind")
        mask, mask_problems = mc.tool_mask_parts(name, row.get("mask"))
        problems = [str(p) for p in mask_problems]
        try:
            confirm: str | None = policy_mod.parse_confirm(name, kind, row)
        except PolicySectionError as exc:
            confirm = None
            problems.append(str(exc))
        view: dict[str, Any] = {
            "name": name, "kind": kind, "min_level": row.get("min_level"),
            "phrase_key": row.get("phrase_key"), "confirm": confirm,
            "confirm_source": "tool" if "confirm" in row else "default",
            "rev": _rev(row), "mask": None,
        }
        if mask is not None:
            langs = catalog.languages
            view["mask"] = {
                "phrase_key": mask.phrase_key,
                "after_ms": {"value": masking.tool_after_ms if mask.after_ms is None else mask.after_ms,
                             "source": "default" if mask.after_ms is None else "tool"},
                "handoff_wait_ms": {
                    "value": masking.handoff_wait_ms if mask.handoff_wait_ms is None else mask.handoff_wait_ms,
                    "source": "default" if mask.handoff_wait_ms is None else "tool"},
                "pool": [{"key": k, "texts": {lang: catalog.template(k, lang) if catalog.has(k, lang)
                                              else None for lang in langs}}
                         for k in mc.pool_keys(mask.phrase_key, catalog.keys)],
            }
            problems += [str(p) for p in mc.validate_catalog_for_tool(name, mask, catalog, langs,
                                                                       reasons=reasons)]
        view["problems"] = problems
        return view

    def list_tools(self) -> dict[str, Any]:
        meta, catalog, masking = self._meta(), self._catalog(), self._masking_settings()
        reasons = self._reasons(meta)
        tools = []
        for name in self._tools(meta):
            row = self.store.get("policy", {"pk": f"tool#{name}"})
            if row is not None:
                tools.append(self._tool_view(name, row, masking=masking, catalog=catalog, reasons=reasons))
        return {"tools": tools, "takes_effect_s": TAKES_EFFECT_S}

    def get_tool(self, name: str) -> dict[str, Any]:
        meta = self._meta()
        row = self._tool_row(name, meta)
        catalog = self._catalog()
        view = self._tool_view(name, row, masking=self._masking_settings(), catalog=catalog,
                               reasons=self._reasons(meta))
        view["row"] = {k: v for k, v in row.items() if k not in ("pk", REV)}
        view["placeholders"] = self._placeholders(name, meta, catalog)["placeholders"]
        return {"tool": view, "takes_effect_s": TAKES_EFFECT_S}

    def put_tool(self, caller: Caller, name: str, body: Mapping[str, Any], *,
                 action: str = "update", restored_from: Mapping[str, str] | None = None
                 ) -> dict[str, Any]:
        _only_keys(body, {"rev", *TOOL_FIELDS})
        rev = _rev_of_body(body)
        updates = {k: body[k] for k in TOOL_FIELDS if k in body}
        if not updates:
            raise ApiError(400, "invalid", f"nothing to update: send any of {list(TOOL_FIELDS)}")
        meta = self._meta()
        row = self._tool_row(name, meta)
        catalog = self._catalog()
        reasons = self._reasons(meta)
        masking = self._masking_settings()
        if _rev(row) != rev:
            raise self._conflict_tool(name, row, masking, catalog, reasons)

        new_row = copy.deepcopy(row)
        for k, v in updates.items():
            if v is None and k in ("confirm", "mask"):  # absent = the default / unmasked
                new_row.pop(k, None)
            else:
                new_row[k] = v
        self._validate_tool(name, new_row, meta, catalog, reasons, mask_changed="mask" in updates)

        changes = audit.diff(row, new_row, TOOL_FIELDS)
        if not changes:
            return {"changed": False, "tool": self._tool_view(name, row, masking=masking, catalog=catalog,
                                                              reasons=reasons)}
        new_rev = rev + 1
        new_row[REV] = new_rev
        try:
            ref = self._commit(
                TxPut("policy", new_row, Expect(REV, rev, absent_as=0, must_exist=True)), [],
                caller=caller, pk=audit.tool_pk(name), action=action, kind="tool", target_id=name,
                changes=changes, security_change=any(c["field"] in audit.SECURITY_FIELDS for c in changes),
                new_rev=new_rev, restored_from=restored_from)
        except TransactionCancelled:
            current = self.store.get("policy", {"pk": f"tool#{name}"}) or row
            raise self._conflict_tool(name, current, masking, catalog, reasons) from None
        return {"changed": True, "audit": ref, "takes_effect_s": TAKES_EFFECT_S,
                "security_change": any(c["field"] in audit.SECURITY_FIELDS for c in changes),
                "tool": self._tool_view(name, new_row, masking=masking, catalog=catalog, reasons=reasons)}

    def _conflict_tool(self, name: str, row: Mapping[str, Any], masking: MaskingSettings,
                       catalog: PromptCatalog, reasons: tuple[str, ...]) -> ApiError:
        return ApiError(409, "conflict", "the tool changed since it was read",
                        extra={"current": self._tool_view(name, row, masking=masking, catalog=catalog,
                                                          reasons=reasons)})

    def _validate_tool(self, name: str, row: Mapping[str, Any], meta: Mapping[str, Any],
                       catalog: PromptCatalog, reasons: tuple[str, ...], *, mask_changed: bool) -> None:
        clean = {k: v for k, v in row.items() if k not in ("pk", REV)}
        try:
            mask = policy_mod.parse_tool_mask(name, clean)
            policy_mod.parse_confirm(name, clean.get("kind"), clean)
        except PolicySectionError as exc:
            raise ApiError(400, "invalid", str(exc), field=exc.field) from None
        ml = clean.get("min_level")
        max_level = meta.get("max_level")
        if not isinstance(ml, int) or isinstance(ml, bool) or not isinstance(max_level, int) \
                or not 0 <= ml <= max_level:
            raise ApiError(400, "invalid", f"must be an integer from 0 to {max_level}",
                           field=f"tools.{name}.min_level")
        try:  # the policy's own rules, over the whole policy with the new row
            doc = policy_mod._doc_from_store(self.store)
            doc["tools"][name] = clean
            Policy.from_dict(doc)
        except PolicyError as exc:
            raise ApiError(400, "invalid", str(exc), field=f"tools.{name}") from None
        if mask is not None and mask_changed:
            problems = mc.validate_catalog_for_tool(name, mask, catalog, catalog.languages, reasons=reasons)
            if problems:
                raise ApiError(400, "invalid", problems[0].message, field=problems[0].field)

    # ------------------------------------------------------------------ masking

    def _masking_view(self, item: Mapping[str, Any] | None, catalog: PromptCatalog) -> dict[str, Any]:
        raw = None if item is None else {k: v for k, v in item.items() if k not in ("pk", REV)}
        settings, problems = mc.masking_parts(raw)
        langs = catalog.languages
        return {
            "masking": raw or {}, "effective": settings.to_doc(),
            "defaults": mc.DEFAULT_SETTINGS.to_doc(), "problems": [str(p) for p in problems],
            "rev": _rev(item),
            "ack_pool": [{"key": k, "texts": {lang: catalog.template(k, lang) if catalog.has(k, lang)
                                              else None for lang in langs}}
                         for k in MASK_GENERIC_KEYS],
        }

    def get_masking(self) -> dict[str, Any]:
        return {**self._masking_view(self._masking_item(), self._catalog()), "takes_effect_s": TAKES_EFFECT_S}

    def put_masking(self, caller: Caller, body: Mapping[str, Any], *, action: str = "update",
                    restored_from: Mapping[str, str] | None = None) -> dict[str, Any]:
        _only_keys(body, {"rev", "masking"})
        rev = _rev_of_body(body)
        if "masking" not in body:
            raise ApiError(400, "invalid", "masking is required", field="masking")
        raw = body["masking"]
        try:
            settings = mc.parse_masking(raw, strict=True)
        except PolicySectionError as exc:
            raise ApiError(400, "invalid", str(exc), field=exc.field) from None
        item = self._masking_item()
        catalog = self._catalog()
        if _rev(item) != rev:
            raise ApiError(409, "conflict", "the masking settings changed since they were read",
                           extra={"current": self._masking_view(item, catalog)})
        normal = settings.to_doc()  # ints, and chars_per_s as the store holds it
        given = list(raw) if isinstance(raw, Mapping) else []
        new_item: dict[str, Any] = {"pk": "masking", **{k: normal[k] for k in mc.MASKING_FIELDS if k in given}}
        old = {k: v for k, v in (item or {}).items() if k not in ("pk", REV)}
        changes = audit.diff(old, new_item, mc.MASKING_FIELDS)
        if not changes:
            return {"changed": False, **self._masking_view(item, catalog)}
        new_rev = rev + 1
        new_item[REV] = new_rev
        try:
            ref = self._commit(TxPut("policy", new_item, Expect(REV, rev, absent_as=0)), [],
                               caller=caller, pk=audit.MASKING_PK, action=action, kind="masking",
                               target_id="masking", changes=changes, security_change=False,
                               new_rev=new_rev, restored_from=restored_from)
        except TransactionCancelled:
            raise ApiError(409, "conflict", "the masking settings changed since they were read",
                           extra={"current": self._masking_view(self._masking_item(), catalog)}) from None
        return {"changed": True, "audit": ref, "takes_effect_s": TAKES_EFFECT_S,
                **self._masking_view(new_item, catalog)}

    # ------------------------------------------------------------------ prompts

    def _prompt_items(self, key: str) -> dict[str, dict[str, Any]]:
        return {it["lang"]: it for it in self.store.query("catalog", {"key": key})
                if isinstance(it.get("lang"), str)}

    @staticmethod
    def _prompt_view(key: str, items: Mapping[str, Mapping[str, Any]], langs: tuple[str, ...],
                     owner: str | None) -> dict[str, Any]:
        view: dict[str, Any] = {
            "key": key, "owner_tool": owner,
            "texts": {lang: items[lang].get("text") if lang in items else None for lang in langs},
            "rev": max((_rev(it) for it in items.values()), default=0),
        }
        drafts = {lang: it.get("status") for lang, it in items.items()
                  if it.get("status", "published") != "published"}
        if drafts:
            view["status"] = drafts
        return view

    def _owner(self, key: str, meta: Mapping[str, Any]) -> tuple[bool, str | None]:
        """``(known, tool)`` for a ``mask.*`` key: the tool whose phrase pool it is in (by its
        ``mask.phrase_key``, else by the ``mask.<tool>`` convention); ``(True, None)`` for the
        acknowledgement pool; ``(False, None)`` for a key no tool owns."""
        tools = self._tools(meta)
        rows = {t: self.store.get("policy", {"pk": f"tool#{t}"}) or {} for t in tools}

        def in_pool(base: str) -> bool:
            return key == base or (key.startswith(base + ".") and bool(_POOL_SUFFIX.fullmatch(key[len(base) + 1:])))

        for t, row in rows.items():
            mask, _ = mc.tool_mask_parts(t, row.get("mask"))
            if mask is not None and in_pool(mask.phrase_key):
                return True, t
        for t in tools:
            if in_pool(f"{mc.MASK_PREFIX}{t}"):
                return True, t
        return key in MASK_GENERIC_KEYS, None

    @staticmethod
    def _word_owner(key: str, meta: Mapping[str, Any], reasons: tuple[str, ...]) -> bool:
        """Whether ``arg.<argument>.<value>`` is the word for some tool's enum value."""
        for t in ConfigRoutes._tools(meta):
            for arg, values in mc.placeholders_for(t, reasons=reasons).items():
                if any(mc.word_key(arg, v) == key for v in values):
                    return True
        return False

    def _protected(self, key: str, meta: Mapping[str, Any]) -> str | None:
        return protected_reason(key, self._tools(meta))

    def editable(self, key: str, meta: Mapping[str, Any] | None = None,
                 cmeta: Mapping[str, Any] | None = None) -> bool:
        """Whether the console may edit ``key``: a catalog key that isn't protected."""
        meta = self._meta() if meta is None else meta
        cmeta = self._catalog_meta() if cmeta is None else cmeta
        return key in (cmeta.get("keys") or []) and self._protected(key, meta) is None

    @staticmethod
    def _key_languages(key: str, cmeta: Mapping[str, Any]) -> tuple[str, ...]:
        """The languages a key is written in: one for a ``singleLanguageKeys`` key, else all."""
        langs = tuple(cmeta.get("languages") or {})
        raw_single = cmeta.get("singleLanguageKeys")
        single = raw_single.get(key) if isinstance(raw_single, Mapping) else None
        return (single,) if isinstance(single, str) and single in langs else langs

    def list_prompts(self, prefix: str | None) -> dict[str, Any]:
        if not isinstance(prefix, str) or not _PROMPT_PREFIX.fullmatch(prefix):
            raise ApiError(400, "invalid", "prefix must be 2 to 128 characters of a-z, 0-9, '_', "
                                           "'.' and '-', starting with a letter", field="prefix")
        meta = self._meta()
        cmeta = self._catalog_meta()
        langs = tuple(cmeta.get("languages") or {})
        keys = [k for k in cmeta.get("keys", []) if isinstance(k, str) and k.startswith(prefix)]
        prompts = []
        for key in keys:
            owner = self._owner(key, meta)[1] if key.startswith(mc.MASK_PREFIX) else None
            view = self._prompt_view(key, self._prompt_items(key), langs, owner)
            # The languages a save needs (one for a singleLanguageKeys key, as put_prompt checks).
            view["languages"] = list(self._key_languages(key, cmeta))
            reason = self._protected(key, meta)
            view["editable"] = reason is None
            if reason is not None:
                view["reason"] = reason
            prompts.append(view)
        return {"prefix": prefix, "languages": list(langs), "prompts": prompts,
                "takes_effect_s": TAKES_EFFECT_S}

    def put_prompt(self, caller: Caller, key: str, body: Mapping[str, Any], *, action: str = "update",
                   restored_from: Mapping[str, str] | None = None) -> dict[str, Any]:
        if not isinstance(key, str) or not _PROMPT_KEY.fullmatch(key):
            raise ApiError(400, "invalid", "not a prompt key", field="key")
        meta = self._meta()
        reason = self._protected(key, meta)
        if reason is not None:  # before anything is read from the body, written or audited
            raise ApiError(403, "protected", f"this prompt is read-only ({reason})", field="key",
                           extra={"reason": reason})
        _only_keys(body, {"rev", "texts"})
        rev = _rev_of_body(body)
        texts = body.get("texts")
        if not isinstance(texts, Mapping):
            raise ApiError(400, "invalid", "texts must map each language to its text", field="texts")
        reasons = self._reasons(meta)
        doc = self._catalog_doc()
        cmeta = self._catalog_meta()
        langs = self._key_languages(key, cmeta)
        for lang in sorted(str(k) for k in texts):
            if lang not in langs:
                raise ApiError(400, "invalid", "not a configured language for this prompt",
                               field=f"texts[{lang}]")
        for lang in langs:
            if not isinstance(texts.get(lang), str):
                raise ApiError(400, "invalid", "every language needs its text", field=f"texts[{lang}]")
        catalog = self._catalog(doc)
        items = self._prompt_items(key)
        exists = key in catalog.keys or key in (cmeta.get("keys") or [])
        if not key.startswith(NEW_KEY_PREFIXES) and not (exists and items):
            raise ApiError(400, "invalid", "not a catalog key (only mask.*, arg.* and option.* "
                                           "keys can be added)", field="key")
        same_set = not key.startswith(mc.MASK_PREFIX)  # mask phrases: validate_phrase below
        if items and same_set:  # the placeholder set of the current text in each language
            default = cmeta.get("defaultLanguage")
            fallback = items.get(default) if isinstance(default, str) else None
            fallback = fallback or next(iter(items.values()))
            for lang in langs:
                current = items.get(lang) or fallback
                _check_prompt_text(key, texts[lang], f"texts[{lang}]",
                                   _placeholder_set(current.get("text")))
        else:
            for lang in langs:
                _check_prompt_text(key, texts[lang], f"texts[{lang}]", None)

        owner: str | None = None
        if key.startswith(mc.MASK_PREFIX):
            known, owner = self._owner(key, meta)
            if not (known or exists):
                raise ApiError(400, "invalid", "not a phrase key of any tool's pool or the "
                                               "acknowledgement pool", field="key")
            for lang in langs:
                problems = mc.validate_phrase(owner or "", texts[lang], catalog, [lang],
                                              field=f"texts[{lang}]", reasons=reasons)
                if problems:
                    raise ApiError(400, "invalid", problems[0].message, field=problems[0].field)
        elif key.startswith(OPTION_PREFIX):
            self._check_option_prompt(key, meta, cmeta, {(key, lang): texts[lang] for lang in langs})
        elif key.startswith(mc.ARG_PREFIX):
            if not self._word_owner(key, meta, reasons):
                raise ApiError(400, "invalid", "not the word for any tool's enumerated argument value",
                               field="key")
            for lang in langs:
                word = texts[lang]
                if not word.strip() or "{" in word or "}" in word:
                    raise ApiError(400, "invalid", "must be non-empty text without braces",
                                   field=f"texts[{lang}]")

        current_rev = max((_rev(it) for it in items.values()), default=0)
        if current_rev != rev:
            raise ApiError(409, "conflict", "the prompt changed since it was read",
                           extra={"current": self._prompt_view(key, items, langs, owner)})
        new_rev = rev + 1
        puts: list[TxPut] = []
        changes: list[dict[str, Any]] = []
        for lang in langs:
            old = items.get(lang)
            # Built from the old item so fields the console never shows carry over: the seed's
            # seed_hash, by which a reseed tells this edit from an older seed text.
            base = dict(old) if old is not None else {"key": key, "lang": lang, "version": 0,
                                                      "status": "published", "audio_url": None}
            new = {**base, "text": texts[lang], "status": "published", REV: new_rev}
            if old is not None and old.get("text") != texts[lang] and old.get("audio_url"):
                new["audio_url"] = None  # a recording of the old text would contradict it
            for f in ("text", "status", "audio_url"):
                if (old or {}).get(f) != new.get(f) and not (old is None and f != "text"):
                    changes.append({"field": f"{f}[{lang}]", "old": (old or {}).get(f), "new": new.get(f)})
            puts.append(TxPut("catalog", new, Expect(REV, _rev(old), absent_as=0)))
        if not changes:
            return {"changed": False, "prompt": self._prompt_view(key, items, langs, owner)}
        extra: list[TxPut] = []
        if key not in (cmeta.get("keys") or []):  # a new key joins the index the bridge reads
            meta_rev = _rev(cmeta)
            extra.append(TxPut("catalog", {**cmeta, "keys": [*(cmeta.get("keys") or []), key],
                                           REV: meta_rev + 1},
                               Expect(REV, meta_rev, absent_as=0, must_exist=True)))
        first, rest = puts[0], puts[1:]
        try:
            # Model instructions steer what the agent may do, so an agent.* edit (a restore too) is
            # audited as a security change; a spoken prompt is not.
            ref = self._commit(first, [*rest, *extra], caller=caller, pk=audit.prompt_pk(key),
                               action=action, kind="prompt", target_id=key, changes=changes,
                               security_change=key.startswith(AGENT_PREFIX), new_rev=new_rev,
                               restored_from=restored_from)
        except TransactionCancelled:
            raise ApiError(409, "conflict", "the prompt changed since it was read",
                           extra={"current": self._prompt_view(key, self._prompt_items(key), langs,
                                                               owner)}) from None
        return {"changed": True, "audit": ref, "takes_effect_s": TAKES_EFFECT_S,
                "prompt": self._prompt_view(key, self._prompt_items(key), langs, owner)}

    def _check_option_prompt(self, key: str, meta: Mapping[str, Any], cmeta: Mapping[str, Any],
                             texts: Mapping[tuple[str, str], str]) -> None:
        """``option.<slot>.<value>``: a choice slot ``meta.slots`` lists, a value of the option form
        (it needn't be an option yet: the spoken name comes first), non-empty texts without braces,
        and no spoken name or alias shared by two of the slot's options."""
        m = _OPTION_KEY.fullmatch(key)
        raw_listed = meta.get("slots")
        listed: list[Any] = raw_listed if isinstance(raw_listed, list) else []
        item = self.store.get("policy", {"pk": f"slot#{m.group(1)}"}) if m and m.group(1) in listed else None
        raw = None if item is None else {k: v for k, v in item.items() if k not in ("pk", REV)}
        raw_factors = meta.get("factors")
        factors: Mapping[str, Any] = raw_factors if isinstance(raw_factors, Mapping) else {}
        settings = None
        if m is not None and raw is not None and m.group(1) not in factors:
            settings, _ = load_slot(m.group(1), raw, factor_def=None)
        if m is None or settings is None or settings.decl.slot_type != "choice":
            raise ApiError(400, "invalid", "not the spoken name of an option of a choice slot "
                                           "(option.<slot>.<value>)", field="key")
        for (_, lang), text in texts.items():
            if not text.strip() or "{" in text or "}" in text:
                raise ApiError(400, "invalid", "must be non-empty text without braces",
                               field=f"texts[{lang}]")
        names = spoken_names(self.store, settings.name, [o.value for o in settings.options], cmeta, texts)
        langs = [lang for (_, lang) in texts]
        conflicts = alias_conflicts(settings.options, names, langs)
        if conflicts:
            c = conflicts[0]
            raise ApiError(400, "invalid", "a spoken name or alias would belong to more than one "
                                           f"option ({', '.join(c.options)})", field=f"texts[{c.lang}]")

    # ------------------------------------------------------------------ placeholders

    def _placeholders(self, tool: str, meta: Mapping[str, Any], catalog: PromptCatalog) -> dict[str, Any]:
        langs = catalog.languages
        out = []
        for arg, values in mc.placeholders_for(tool, reasons=self._reasons(meta)).items():
            vals, missing = [], []
            for value in values:
                key = mc.word_key(arg, value)
                words = {lang: catalog.template(key, lang) if catalog.has(key, lang) else None
                         for lang in langs}
                missing += [f"{key}[{lang}]" for lang, w in words.items()
                            if w is None or not w.strip() or "{" in w or "}" in w]
                vals.append({"value": value, "key": key, "words": words})
            out.append({"name": arg, "placeholder": f"{{{arg}}}", "values": vals, "missing": missing})
        return {"tool": tool, "languages": list(langs), "placeholders": out}

    def get_placeholders(self, tool: str) -> dict[str, Any]:
        meta = self._meta()
        if not _TOOL_NAME.fullmatch(tool) or tool not in self._tools(meta):
            raise ApiError(404, "not_found", "no such tool")
        return self._placeholders(tool, meta, self._catalog())

    # ------------------------------------------------------------------ audit and restore

    def list_audit(self, item: str | None, limit: str | None) -> dict[str, Any]:
        if not isinstance(item, str) or not _AUDIT_ITEM.fullmatch(item):
            raise ApiError(400, "invalid", "item must be an audited item key such as "
                                           "policy#tool#get_balance", field="item")
        n = DEFAULT_AUDIT_LIMIT
        if limit is not None:
            if not limit.isdigit() or not 1 <= int(limit) <= MAX_AUDIT_LIMIT:
                raise ApiError(400, "invalid", f"limit must be 1..{MAX_AUDIT_LIMIT}", field="limit")
            n = int(limit)
        # The deploy-settings marker shares its pk with the deploy-setting entries; it isn't an entry.
        entries = [e for e in self.store.query(audit.AUDIT_TABLE, {"pk": item})
                   if not (item == DEPLOY_PK and e.get("sk") == MARKER_SK)]
        entries.sort(key=lambda e: e.get("sk", ""), reverse=True)
        return {"item": item, "entries": entries[:n]}

    def restore(self, caller: Caller, body: Mapping[str, Any]) -> dict[str, Any]:
        _only_keys(body, {"audit_pk", "audit_sk", "rev"})
        pk, sk = body.get("audit_pk"), body.get("audit_sk")
        if not isinstance(pk, str) or not _AUDIT_ITEM.fullmatch(pk):
            raise ApiError(400, "invalid", "audit_pk is required", field="audit_pk")
        if not isinstance(sk, str) or not 0 < len(sk) <= 128:
            raise ApiError(400, "invalid", "audit_sk is required", field="audit_sk")
        rev = _rev_of_body(body)
        entry = self.store.get(audit.AUDIT_TABLE, {"pk": pk, "sk": sk})
        if entry is None:
            raise ApiError(404, "not_found", "no such audit entry")
        target = entry.get("target") or {}
        kind, target_id = target.get("kind"), target.get("id")
        changes = [c for c in entry.get("changes") or [] if isinstance(c, Mapping)]
        ref = {"pk": pk, "sk": sk}
        if kind == "tool" and isinstance(target_id, str):
            updates = {c["field"]: c.get("old") for c in changes if c.get("field") in TOOL_FIELDS}
            return self.put_tool(caller, target_id, {"rev": rev, **updates}, action="restore",
                                 restored_from=ref)
        if kind == "masking":
            item = self._masking_item() or {}
            raw = {k: v for k, v in item.items() if k not in ("pk", REV)}
            for c in changes:
                f = c.get("field")
                if f in mc.MASKING_FIELDS:
                    if c.get("old") is None:
                        raw.pop(f, None)
                    else:
                        raw[f] = c["old"]
            return self.put_masking(caller, {"rev": rev, "masking": raw}, action="restore",
                                    restored_from=ref)
        if kind == "prompt" and isinstance(target_id, str):
            items = self._prompt_items(target_id)
            texts = {lang: it.get("text") for lang, it in items.items()}
            for c in changes:
                m = re.fullmatch(r"text\[(.+)\]", str(c.get("field")))
                if m:
                    if c.get("old") is None:
                        raise ApiError(409, "not_restorable", "the prompt didn't exist before this "
                                                              "change; prompts are never deleted")
                    texts[m.group(1)] = c["old"]
            return self.put_prompt(caller, target_id, {"rev": rev, "texts": texts}, action="restore",
                                   restored_from=ref)
        raise ApiError(400, "not_restorable", "only tool, masking and prompt changes can be restored")
