"""The one server-side auth policy table.

The policy says, per tool, the minimum auth level, the catalog phrase key for the
connecting phrase and the tool kind; per level, the cumulative factors; and per
escalation reason, the queue, priority and catalog phrase key.

Two sources share one validated shape:

* :meth:`Policy.from_dict` takes the seed file (``data/policy.json``) shape. It
  never refreshes.
* :meth:`Policy.from_store` reads the ``policy`` table and re-reads it after
  ``ttl_s`` seconds (default 60), so a table edit takes effect within that time.
  If a refresh fails, the last good policy stays in use and the failure is logged.

Store layout (``policy`` table, pk ``pk``), written by :meth:`Policy.to_items`:

* ``meta``: ``version``, ``max_level``, ``tools`` (names), ``reasons`` (names),
  ``default_reason``, ``queues``, ``factors`` (factor definitions), ``limits``.
  This item is the index of the other rows: a reload reads it, then the rows it names in
  one ``BatchGetItem``.
* ``tool#<name>``: the tool's row (``min_level``, ``phrase_key``, ``kind``, ...).
* ``level#<n>``: ``factors`` (the cumulative list for that level).
* ``reason#<name>``: ``queue``, ``priority``, ``phrase_key`` and, if set, ``description``.
* ``web_offers``: ``offers``, the bank's offer list (:class:`WebOffer`); no
  item (or an empty list) means no offers. :meth:`Policy.to_items` always writes it, empty when
  the section is absent, so a reseed clears a removed list.
* ``model_failover``: the failover thresholds, window, ping and recovery settings and the per-model
  overrides (:class:`ModelFailoverConfig`); always written
  (empty when the section is absent) so a reseed clears a removed override. Non-integral numbers
  are decimal strings (the store holds no floats).
* ``human_leg``: ``transcription`` (bool, default false), the human-leg conversation's transcription
  switch; always written, so a reseed resets it. A
  missing item, or anything but a boolean ``true``, reads as off. ``agent_legs`` (bool, default
  true), in the same item: transcribe each agent's call leg separately; false uses
  one both-tracks transcription. A missing or non-boolean value reads as true.
* ``memory``: ``write`` (bool, default true), the per-call recap write to Memory; always
written, so a reseed resets it. A missing item, or a non-boolean
  value, reads as on.
* ``end_call``: ``enabled`` (bool, default true), whether the model is offered the ``end_call``
  tool; always written, so a reseed resets it. A missing item, or a
  non-boolean value, reads as on (a malformed one with a warning).
* ``redaction``: the PII redaction config (:class:`owlbank.pii.RedactionConfig`;
  ``min_confidence`` as a decimal string, since the store holds no floats).
* ``ani_trust``: the caller-ID trust gate (:class:`AniTrustConfig`).
* ``masking``: the latency-masking timings (:class:`MaskingConfig`).
* ``slot#<name>``: one slot's settings (:class:`owlbank.slot_policy.SlotSettings`); ``meta.slots``
  lists the names.
* ``slot_resolution``: the mode default, the per-flow overrides, ``generative_enabled`` and
  ``resolver_budget_ms`` (:class:`owlbank.slot_policy.ResolutionSettings`).

**Slot settings** (``slots`` in the seed file): ``sensitive`` (default false),
``keypad_only`` (default false) and ``barge_in``
(default true) per slot, and for a slot that isn't a ``factors`` entry its shape and ``flow``.
The seed file's section is validated strictly (:func:`owlbank.slot_policy.parse_slot`); a
``factors`` entry with no entry in it gets ``sensitive: true``, and a ``sensitive`` key in a
``factors`` entry is refused (it moved to the slot). The table is read leniently
(:func:`owlbank.slot_policy.load_slot`): a non-boolean field loads as ``true``, a ``factors``
entry with no ``slot#`` item loads ``sensitive: true``, and each problem is logged at ERROR
naming the item and field and kept in :attr:`Policy.slot_problems` (a ``sensitive`` left in
``meta.factors`` by an older seed is one, and is not read). A non-factor slot that can't be
loaded is left out of :meth:`Policy.slots`.

**Slot modes**: each slot item may carry ``mode``, ``confirm`` and,
on a choice slot, ``options``; the ``slot_resolution`` section holds the default and the flow
overrides. The seed file is strict (:func:`owlbank.slot_policy.parse_slot_resolution`; an absent
section gives the defaults). The table is lenient
(:func:`owlbank.slot_policy.load_slot_resolution`): a missing item or an invalid mode value loads
``deterministic``, logged at ERROR and kept in :attr:`Policy.slot_resolution_problems`;
:attr:`Policy.invalid_mode_fields` names every mode field that loaded so, for
:mod:`owlbank.slots.modes`.

**Redaction config** (``redaction`` in the seed file): validated fail-closed
by :func:`owlbank.pii.load_redaction_config`. A missing or invalid section at run
time gives the built-in defaults and an ERROR naming the field; it never fails the
rest of the policy. :meth:`Policy.to_items` (seeding) refuses an invalid section.

**Caller-ID trust** (``ani_trust``): ``accepted_verstat`` (the
``StirVerstat`` values that make the ANI trusted; default ``["TN-Validation-Passed-A"]``)
and ``on_untrusted`` (``no_ani_match``, the default, or ``allow``). A missing
``StirVerstat`` is untrusted. With ``no_ani_match`` an untrusted ANI match isn't a
factor: :meth:`Policy.effective_factors` replaces it with the next level's collected
factors. The untrusted-attempt cap: ``untrusted_failure_limit`` (int
1..100, default 3) failures from an untrusted ANI per candidate customer within
``untrusted_window_s`` (int 60..86400, default 3600, from the first failure) send every
further untrusted attempt for that customer to a human (``verification_risk``). Same
fail-closed loading as ``redaction``.

**Per-tool confirmation** (``confirm`` on each tool row): ``none`` or
``deterministic`` (default: ``deterministic`` for ``make_payment`` and ``transfer_funds``,
``none`` otherwise). ``confirm`` is rejected on AgentCore Gateway tools (kind
``level0`` or :data:`owlbank.tools_contract.GATEWAY_LEVEL0_TOOLS`), and
``deterministic`` is allowed only on ``agentic_inline`` tools. An invalid value at run
time gives that tool's default and an ERROR naming ``tools.<name>.confirm``;
:meth:`Policy.to_items` refuses it.

**Latency masking**: a tool row may carry ``mask``
(``{"phrase_key": "mask.<tool>", "after_ms"?: int, "handoff_wait_ms"?: int}``; absent or
``null`` = no phrase); the ``masking`` section holds the policy-wide values (``after_ms``,
``tool_after_ms``, ``handoff_wait_ms``, ``end_mode``, ``chars_per_s``, ``silence_*``). The
fields, defaults, ranges and both validation modes are :mod:`owlbank.masking_config`; this
module delegates to it. At run time each invalid field gives that field's default and an
ERROR naming it (a bad ``phrase_key`` or a non-object ``mask``: no phrase); a missing section
gives the defaults. :meth:`Policy.to_items` refuses any invalid value.

**Web offers** (``web_offers``): ``{"offers": [{"id", "values"}]}``.
An ``id`` matches ``^[a-z0-9_]{1,32}$`` and is unique; ``values`` is a non-empty list of non-empty
strings of at most :data:`owlbank.web_activity.MAX_VALUE_CHARS` characters, unique across all offers
after normalisation (NFKC, casefold, trim, collapse whitespace); at most :data:`MAX_WEB_OFFERS`
offers. An absent section means no offers. The offer's phrase per language is the catalog entry
``offer.<id>`` (checked by ``scripts/seed.py``). At run time an invalid section gives no offers and
an ERROR naming the field (the welcome is then today's); :meth:`Policy.to_items` refuses it.

**Model failover** (``model_failover``): the global settings of
:class:`owlbank.failover.settings.Settings` (``window_calls``, ``window_age_s``, ``min_samples``,
``miss_run``, ``pings_per_cycle`` 1 to 5, ``recover_min_pings``, ``recover_period_s``, ``dwell_s``,
``idle_reset_s``, ``unavailable_recheck_s``) and ``entries``, per-model-ID overrides of
``trip_ttft_s``, ``recover_ttft_s`` (below the trip), ``deadline_s`` and ``error_rate_limit``. Every
field is optional: an absent field, entry or section means the built-in default. The
bounds are those of :class:`Settings` and :class:`EntrySettings`, which this module calls rather than
repeats. A ``deadline_s`` override on the last entry of the deployed list is ignored (it has none); an
override for a model not in the list is ignored. At run time an invalid section gives the
defaults and an ERROR naming the field; :meth:`Policy.to_items` refuses it.

**Reason descriptions** (``description`` on an escalation reason): one line of at most
:data:`MAX_REASON_DESCRIPTION_CHARS` characters with none of :data:`DESCRIPTION_MARKUP`. The
``escalate`` tool's ``reason`` description is built from them
(:func:`owlbank.tools_contract.tool_schemas`), so the model tells the reasons apart from data, not
from a prompt rule. At run time an invalid one is dropped with an ERROR; :meth:`Policy.to_items`
refuses it.

Fail closed everywhere: an unknown tool needs :attr:`Policy.max_level`; an unknown
escalation reason routes as the default reason (``unable_to_resolve``) and is logged.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any, TypeGuard

from owlbank import masking_config
from owlbank.failover.settings import Settings, SettingsError, entry_defaults
from owlbank.masking_config import DEFAULT_SETTINGS, MaskingSettings, ToolMask
from owlbank.obs import log_caught
from owlbank.pii import RedactionConfig, RedactionConfigError, load_redaction_config, parse_redaction_config
from owlbank.slot_policy import (
    RESOLUTION_ITEM,
    ResolutionSettings,
    SlotPolicyError,
    SlotSettings,
    load_slot,
    load_slot_resolution,
    parse_slot,
    parse_slot_resolution,
)
from owlbank.store import Item, Store, timed_reload
from owlbank.tools_contract import GATEWAY_LEVEL0_TOOLS
from owlbank.web_activity import MAX_VALUE_CHARS
from owlbank.welcome import OFFER_ID_RE, WebOffer, normalise

log = logging.getLogger(__name__)

#: Tool kinds (see the ``_notes`` in ``data/policy.json``).
TOOL_KINDS: frozenset[str] = frozenset({"level0", "agentic_inline", "deterministic", "handoff"})

#: Factors established by the call itself rather than collected from the caller.
ESTABLISHED_FACTORS: frozenset[str] = frozenset({"ani_match"})

#: ``confirm`` modes and the built-in defaults (every other tool: ``none``).
CONFIRM_MODES: frozenset[str] = frozenset({"none", "deterministic"})
DEFAULT_CONFIRM: Mapping[str, str] = {"make_payment": "deterministic",
                                      "transfer_funds": "deterministic"}

#: ``ani_trust``.
ON_UNTRUSTED_MODES: frozenset[str] = frozenset({"no_ani_match", "allow"})
DEFAULT_ACCEPTED_VERSTAT: tuple[str, ...] = ("TN-Validation-Passed-A",)
MAX_VERSTAT_LEN = 64
#: Untrusted-attempt cap: default and allowed range of each field.
DEFAULT_UNTRUSTED_FAILURE_LIMIT = 3
DEFAULT_UNTRUSTED_WINDOW_S = 3600
UNTRUSTED_FAILURE_LIMIT_RANGE = (1, 100)
UNTRUSTED_WINDOW_S_RANGE = (60, 86400)
DEFAULT_ASK_PHONE_ON_NO_MATCH = True

#: ``masking``: defaults, ranges and parsing live in :mod:`owlbank.masking_config`.
#: The timed acknowledgements, rotated within a call. Neutral: they never name an action.
MASK_GENERIC_KEYS: tuple[str, ...] = tuple(f"mask.generic.{n}" for n in range(1, 6))
MASK_GENERIC_KEY = MASK_GENERIC_KEYS[0]  # kept for callers that need one key

#: ``web_offers``.
WEB_OFFERS_ITEM = "web_offers"
#: ``model_failover``.
MODEL_FAILOVER_ITEM = "model_failover"
#: ``human_leg``: ``{"transcription": false}``.
HUMAN_LEG_ITEM = "human_leg"
#: ``memory``: ``{"write": true}``.
MEMORY_ITEM = "memory"
#: ``end_call``: ``{"enabled": true}``.
END_CALL_ITEM = "end_call"
MAX_WEB_OFFERS = 50

#: An escalation reason's ``description``: one line, this long at most, none of these.
MAX_REASON_DESCRIPTION_CHARS = 160
DESCRIPTION_MARKUP: frozenset[str] = frozenset("{}<>[]*#`|~^_")

META_PK = "meta"
DEFAULT_TTL_S = 60
#: The admin console's optimistic-concurrency counter on each item (missing = 0).
REV_ATTR = "rev"


class PolicyError(ValueError):
    """The policy data is invalid."""


class PolicySectionError(PolicyError):
    """One field of a fail-closed section (``ani_trust``, a tool's ``confirm``) is invalid."""

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(f"{field_name}: {message}")
        self.field = field_name


@dataclass(frozen=True, slots=True)
class AniTrustConfig:
    """The caller-ID trust gate."""

    accepted_verstat: tuple[str, ...] = DEFAULT_ACCEPTED_VERSTAT
    on_untrusted: str = "no_ani_match"
    untrusted_failure_limit: int = DEFAULT_UNTRUSTED_FAILURE_LIMIT
    untrusted_window_s: int = DEFAULT_UNTRUSTED_WINDOW_S
    #: Ask for the account phone number when the ANI has no match or is withheld.
    #: Not fail-closed: asking more of the caller only adds a step, so a bad value means on.
    ask_phone_on_no_match: bool = DEFAULT_ASK_PHONE_ON_NO_MATCH

    def trusted(self, verstat: Any) -> bool:
        """``True`` only for a ``StirVerstat`` in ``accepted_verstat``. Missing is untrusted."""
        return isinstance(verstat, str) and verstat in self.accepted_verstat

    def ani_match_counts(self, verstat: Any) -> bool:
        """Whether an ANI match counts as a factor for a call with this ``StirVerstat``."""
        return self.trusted(verstat) or self.on_untrusted == "allow"

    def to_doc(self) -> dict[str, Any]:
        return {"accepted_verstat": list(self.accepted_verstat), "on_untrusted": self.on_untrusted,
                "untrusted_failure_limit": self.untrusted_failure_limit,
                "untrusted_window_s": self.untrusted_window_s,
                "ask_phone_on_no_match": self.ask_phone_on_no_match}


DEFAULT_ANI_TRUST = AniTrustConfig()


def parse_ani_trust(raw: Any) -> AniTrustConfig:
    """Strict: :class:`PolicySectionError` naming the bad field."""
    if not isinstance(raw, Mapping):
        raise PolicySectionError("ani_trust", "must be an object")
    unknown = set(raw) - {"accepted_verstat", "on_untrusted", "untrusted_failure_limit",
                          "untrusted_window_s", "ask_phone_on_no_match"}
    if unknown:
        raise PolicySectionError(f"ani_trust.{sorted(unknown)[0]}", "unknown field")
    accepted = raw.get("accepted_verstat", list(DEFAULT_ACCEPTED_VERSTAT))
    if not isinstance(accepted, list) or not all(
            isinstance(v, str) and 0 < len(v) <= MAX_VERSTAT_LEN for v in accepted):
        raise PolicySectionError("ani_trust.accepted_verstat",
                                 f"must be a list of non-empty strings (max {MAX_VERSTAT_LEN} chars)")
    if len(set(accepted)) != len(accepted):
        raise PolicySectionError("ani_trust.accepted_verstat", "lists a value twice")
    mode = raw.get("on_untrusted", "no_ani_match")
    if mode not in ON_UNTRUSTED_MODES:
        raise PolicySectionError("ani_trust.on_untrusted",
                                 f"must be one of {sorted(ON_UNTRUSTED_MODES)}")
    limit = _bounded_int(raw, "untrusted_failure_limit", DEFAULT_UNTRUSTED_FAILURE_LIMIT,
                         UNTRUSTED_FAILURE_LIMIT_RANGE)
    window = _bounded_int(raw, "untrusted_window_s", DEFAULT_UNTRUSTED_WINDOW_S,
                          UNTRUSTED_WINDOW_S_RANGE)
    ask_phone = raw.get("ask_phone_on_no_match", DEFAULT_ASK_PHONE_ON_NO_MATCH)
    if not isinstance(ask_phone, bool):
        log.warning("policy field ani_trust.ask_phone_on_no_match must be true or false; using %s",
                    DEFAULT_ASK_PHONE_ON_NO_MATCH, extra={"decision": "fallback_defaults"})
        ask_phone = DEFAULT_ASK_PHONE_ON_NO_MATCH
    return AniTrustConfig(tuple(accepted), mode, limit, window, ask_phone)


def _bounded_int(raw: Mapping[str, Any], name: str, default: int, bounds: tuple[int, int]) -> int:
    value = raw.get(name, default)
    lo, hi = bounds
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise PolicySectionError(f"ani_trust.{name}", f"must be an integer from {lo} to {hi}")
    return value


def load_ani_trust(raw: Any) -> AniTrustConfig:
    """:func:`parse_ani_trust` that never raises: missing or invalid gives the defaults
    (untrusted unless ``TN-Validation-Passed-A``; no ANI match when untrusted) and an ERROR."""
    if raw is None:
        log.error("policy ani_trust section is missing; using the built-in defaults",
                  extra={"decision": "fallback_defaults"})
        return DEFAULT_ANI_TRUST
    try:
        return parse_ani_trust(raw)
    except PolicySectionError as exc:
        log.error("policy field %s is invalid; using the built-in defaults", exc.field,
                  extra={"decision": "fallback_defaults"})
        return DEFAULT_ANI_TRUST


def _is_gateway_tool(name: str, kind: Any) -> bool:
    return kind == "level0" or name in GATEWAY_LEVEL0_TOOLS


def default_confirm(name: str) -> str:
    return DEFAULT_CONFIRM.get(name, "none")


def parse_confirm(name: str, kind: Any, row: Mapping[str, Any]) -> str:
    """A tool's ``confirm`` mode (strict). Absent means the built-in default."""
    field_name = f"tools.{name}.confirm"
    if _is_gateway_tool(name, kind):
        if "confirm" in row:
            raise PolicySectionError(
                field_name, "not allowed on an AgentCore Gateway tool: the bridge doesn't intercept "
                "tool calls that run through the Gateway, so it can't hold one for the read-back. Move the tool to an "
                "inline tool (kind agentic_inline) to confirm it")
        return "none"
    value = row.get("confirm", default_confirm(name))
    if value not in CONFIRM_MODES:
        raise PolicySectionError(field_name, f"must be one of {sorted(CONFIRM_MODES)}")
    if value == "deterministic" and kind != "agentic_inline":
        raise PolicySectionError(field_name, "deterministic is allowed only on agentic_inline tools")
    return value


def load_confirm(name: str, kind: Any, row: Mapping[str, Any]) -> str:
    """:func:`parse_confirm` that never raises: invalid gives the tool's default and an ERROR."""
    try:
        return parse_confirm(name, kind, row)
    except PolicySectionError as exc:
        fallback = "none" if _is_gateway_tool(name, kind) else (
            default_confirm(name) if kind == "agentic_inline" else "none")
        log.error("policy field %s is invalid (%s); using %r", exc.field, exc, fallback,
                  extra={"decision": "fallback_default_confirm"})
        return fallback


#: The policy-wide masking values. The old name of
#: :class:`owlbank.masking_config.MaskingSettings`.
MaskingConfig = MaskingSettings
DEFAULT_MASKING = DEFAULT_SETTINGS


def _masking_parts(raw: Any) -> tuple[MaskingConfig, list[PolicySectionError]]:
    """Each valid field, the default for each invalid one, and the errors in field order."""
    cfg, problems = masking_config.parse_masking(raw, strict=False)
    return cfg, [PolicySectionError(p.field, p.message) for p in problems]


def parse_masking(raw: Any) -> MaskingConfig:
    """Strict: :class:`PolicySectionError` naming the first bad field. Omitted fields take
    the defaults. Delegates to :func:`owlbank.masking_config.parse_masking`."""
    return masking_config.parse_masking(raw, strict=True)


def load_masking(raw: Any) -> MaskingConfig:
    """Never raises: a missing section gives the defaults; each invalid field gives that
    field's default and an ERROR naming it."""
    cfg, problems = masking_config.parse_masking(raw, strict=False)
    for p in problems:
        log.error("policy field %s is invalid; using the built-in default", p.field,
                  extra={"decision": "fallback_default"})
    return cfg


def parse_tool_mask(name: str, row: Mapping[str, Any]) -> ToolMask | None:
    """A tool's full masking settings (strict). Absent or ``null`` means unmasked."""
    return masking_config.parse_tool_mask(name, row.get("mask"), strict=True)


def load_tool_mask(name: str, row: Mapping[str, Any]) -> ToolMask | None:
    """:func:`parse_tool_mask` that never raises: each bad field gives its default (a bad
    ``phrase_key`` or a non-object ``mask``: unmasked) and an ERROR naming it."""
    mask, problems = masking_config.parse_tool_mask(name, row.get("mask"), strict=False)
    for p in problems:
        log.error("policy field %s is invalid (%s); %s", p.field, p.message,
                  "the tool gets no masking phrase" if mask is None else "using the default",
                  extra={"decision": "fallback_no_mask" if mask is None else "fallback_default"})
    return mask


def parse_mask(name: str, row: Mapping[str, Any]) -> str | None:
    """A tool's masking phrase key (strict). Absent or ``null`` means no phrase."""
    mask = parse_tool_mask(name, row)
    return None if mask is None else mask.phrase_key


def load_mask(name: str, row: Mapping[str, Any]) -> str | None:
    """:func:`parse_mask` that never raises: invalid gives no phrase and an ERROR."""
    mask = load_tool_mask(name, row)
    return None if mask is None else mask.phrase_key


def parse_web_offers(raw: Any) -> tuple[WebOffer, ...]:
    """The ``web_offers`` section (strict): :class:`PolicySectionError` naming the bad field.
    ``None`` (absent) means no offers."""
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise PolicySectionError("web_offers", "must be an object")
    unknown = set(raw) - {"offers"}
    if unknown:
        raise PolicySectionError(f"web_offers.{sorted(map(str, unknown))[0]}", "unknown field")
    offers = raw.get("offers", [])
    if not isinstance(offers, (list, tuple)):
        raise PolicySectionError("web_offers.offers", "must be a list")
    if len(offers) > MAX_WEB_OFFERS:
        raise PolicySectionError("web_offers.offers", f"at most {MAX_WEB_OFFERS} offers")
    out: list[WebOffer] = []
    ids: set[str] = set()
    seen: dict[str, str] = {}  # normalised value -> offer ID
    for i, entry in enumerate(offers):
        where = f"web_offers.offers[{i}]"
        if not isinstance(entry, Mapping):
            raise PolicySectionError(where, "must be an object")
        extra = set(entry) - {"id", "values"}
        if extra:
            raise PolicySectionError(f"{where}.{sorted(map(str, extra))[0]}", "unknown field")
        oid = entry.get("id")
        if not isinstance(oid, str) or not OFFER_ID_RE.fullmatch(oid):
            raise PolicySectionError(f"{where}.id", "must match ^[a-z0-9_]{1,32}$")
        if oid in ids:
            raise PolicySectionError(f"{where}.id", f"offer {oid!r} is listed twice")
        ids.add(oid)
        values = entry.get("values")
        if not isinstance(values, (list, tuple)) or not values:
            raise PolicySectionError(f"{where}.values", "must be a non-empty list")
        for j, value in enumerate(values):
            vwhere = f"{where}.values[{j}]"
            if not isinstance(value, str) or not value.strip():
                raise PolicySectionError(vwhere, "must be a non-empty string")
            if len(value) > MAX_VALUE_CHARS:
                raise PolicySectionError(vwhere, f"at most {MAX_VALUE_CHARS} characters")
            norm = normalise(value)  # the matcher's own rule, so uniqueness agrees with matching
            if not norm:
                raise PolicySectionError(vwhere, "must be a non-empty string")
            if norm in seen:
                raise PolicySectionError(vwhere, f"the same value (after normalisation) is already "
                                                 f"listed for offer {seen[norm]!r}")
            seen[norm] = oid
        out.append(WebOffer(oid, tuple(values)))
    return tuple(out)


def load_web_offers(raw: Any) -> tuple[WebOffer, ...]:
    """:func:`parse_web_offers` that never raises: invalid gives no offers and an ERROR naming
    the field (never a value)."""
    try:
        return parse_web_offers(raw)
    except PolicySectionError as exc:
        log.error("policy field %s is invalid; no web offers", exc.field,
                  extra={"decision": "fallback_no_offers"})
        return ()


# --------------------------------------------------------------------------- model_failover

_FAILOVER_COUNTS: tuple[str, ...] = ("window_calls", "min_samples", "miss_run", "pings_per_cycle",
                                     "recover_min_pings")
_FAILOVER_SECONDS: tuple[str, ...] = ("window_age_s", "recover_period_s", "dwell_s", "idle_reset_s",
                                      "unavailable_recheck_s")
_FAILOVER_ENTRY_FIELDS: tuple[str, ...] = ("trip_ttft_s", "recover_ttft_s", "deadline_s", "error_rate_limit")
_PROBE_ID = "probe"  # a stand-in entry, to validate the global fields through Settings


def load_human_leg_transcription(raw: Any) -> bool:
    """The ``human_leg.transcription`` switch; a missing or malformed section reads as off."""
    return isinstance(raw, Mapping) and raw.get("transcription") is True


def load_human_leg_agent_legs(raw: Any) -> bool:
    """The ``human_leg.agent_legs`` switch; a missing or malformed value reads as on."""
    return not (isinstance(raw, Mapping) and raw.get("agent_legs") is False)


def load_memory_write(raw: Any) -> bool:
    """The ``memory.write`` switch; a missing or malformed value reads as on."""
    return not (isinstance(raw, Mapping) and raw.get("write") is False)


def load_end_call(raw: Any) -> bool:
    """The ``end_call.enabled`` switch; a missing or malformed value reads as
    on. A present but malformed section logs a warning (an absent one is the pre-switch seed)."""
    enabled = raw.get("enabled") if isinstance(raw, Mapping) else None
    if isinstance(enabled, bool):
        return enabled
    if raw is not None:
        log.warning("policy field end_call.enabled must be true or false; using true",
                    extra={"decision": "fallback_defaults"})
    return True


def _failover_number(where: str, value: Any, *, count: bool) -> int | float:
    """One section value as a number: an ``int`` for a count; an int, float or decimal string (the
    store form) for seconds and rates. Anything else (``bool``, ``None``, text) is refused."""
    if isinstance(value, bool):
        raise PolicySectionError(where, "must be a number")
    if count:
        if not _is_int(value):
            raise PolicySectionError(where, "must be an integer")
        return value
    if isinstance(value, str):
        try:
            return float(Decimal(value))
        except (InvalidOperation, ValueError):
            raise PolicySectionError(where, "must be a number") from None
    if not isinstance(value, int | float | Decimal):
        raise PolicySectionError(where, "must be a number")
    return float(value)


def _failover_doc_number(value: float) -> int | str:
    """The store form of a float: an ``int`` when integral, else its shortest decimal string."""
    return int(value) if value == int(value) else repr(value)


@dataclass(frozen=True, slots=True)
class ModelFailoverConfig:
    """The ``model_failover`` section: the overrides of the built-in defaults, validated.

    ``settings`` holds the global fields set by the section; ``entries`` the per-model-ID overrides
    (model ID -> field -> value). The list of models is a deploy setting, so :meth:`settings_for`
    builds the :class:`Settings` for it. An empty config gives the defaults.
    """

    settings: Mapping[str, int | float]
    entries: Mapping[str, Mapping[str, float]]

    def settings_for(self, model_ids: tuple[str, ...] | list[str]) -> Settings:
        """The :class:`Settings` for an ordered list of model IDs: the defaults with this section's
        values applied. Overrides for models not in the list are ignored, and so is ``deadline_s``
        for the last entry (it has none)."""
        base = Settings.for_models(model_ids, **self.settings)
        last = len(base.entries) - 1
        out = []
        for i, entry in enumerate(base.entries):
            ov: dict[str, Any] = dict(self.entries.get(entry.model_id, {}))
            if i == last:
                ov.pop("deadline_s", None)
            out.append(replace(entry, **ov))
        return replace(base, entries=tuple(out))

    def to_doc(self) -> dict[str, Any]:
        """The seed-file / policy-item shape (no floats: see :func:`_failover_doc_number`)."""
        doc: dict[str, Any] = {k: v if k in _FAILOVER_COUNTS else _failover_doc_number(float(v))
                               for k, v in self.settings.items()}
        doc["entries"] = {m: {k: _failover_doc_number(v) for k, v in ov.items()}
                          for m, ov in self.entries.items()}
        return doc


def parse_model_failover(raw: Any) -> ModelFailoverConfig:
    """The ``model_failover`` section (strict): :class:`PolicySectionError` naming the bad field.
    ``None`` (absent) means the defaults. The bounds come from :class:`Settings`."""
    if raw is None:
        return ModelFailoverConfig({}, {})
    if not isinstance(raw, Mapping):
        raise PolicySectionError(MODEL_FAILOVER_ITEM, "must be an object")
    known = {*_FAILOVER_COUNTS, *_FAILOVER_SECONDS, "entries"}
    unknown = set(raw) - known
    if unknown:
        raise PolicySectionError(f"{MODEL_FAILOVER_ITEM}.{sorted(map(str, unknown))[0]}", "unknown field")
    values: dict[str, int | float] = {
        k: _failover_number(f"{MODEL_FAILOVER_ITEM}.{k}", v, count=k in _FAILOVER_COUNTS)
        for k, v in raw.items() if k != "entries"}
    try:
        Settings.for_models([_PROBE_ID], **values)
    except SettingsError as exc:
        raise PolicySectionError(f"{MODEL_FAILOVER_ITEM}.{exc.field}", "out of range") from None
    entries_raw = raw.get("entries", {})
    where = f"{MODEL_FAILOVER_ITEM}.entries"
    if not isinstance(entries_raw, Mapping):
        raise PolicySectionError(where, "must be an object keyed by model ID")
    entries: dict[str, dict[str, float]] = {}
    for model_id, ov_raw in entries_raw.items():
        if not isinstance(model_id, str) or not model_id:
            raise PolicySectionError(where, "keys must be model IDs")
        ewhere = f"{where}.{model_id}"
        if not isinstance(ov_raw, Mapping):
            raise PolicySectionError(ewhere, "must be an object")
        extra = set(ov_raw) - set(_FAILOVER_ENTRY_FIELDS)
        if extra:
            raise PolicySectionError(f"{ewhere}.{sorted(map(str, extra))[0]}", "unknown field")
        ov: dict[str, Any] = {k: float(_failover_number(f"{ewhere}.{k}", v, count=False))
                              for k, v in ov_raw.items()}
        try:  # against the model's defaults, as a non-last entry (so a deadline is allowed)
            replace(entry_defaults(model_id, last=False), **ov)
        except SettingsError as exc:
            raise PolicySectionError(f"{ewhere}.{exc.field}", "out of range") from None
        entries[model_id] = ov
    return ModelFailoverConfig(values, entries)


def load_model_failover(raw: Any) -> ModelFailoverConfig:
    """:func:`parse_model_failover` that never raises: invalid gives the defaults and an ERROR
    naming the field (never a value)."""
    try:
        return parse_model_failover(raw)
    except PolicySectionError as exc:
        log.error("policy field %s is invalid; using the default failover settings", exc.field,
                  extra={"decision": "fallback_default", "field": exc.field})
        return ModelFailoverConfig({}, {})


def parse_reason_description(name: str, value: Any) -> str:
    """An escalation reason's ``description`` (strict): one line of at most
    :data:`MAX_REASON_DESCRIPTION_CHARS` characters, no markup characters."""
    field_name = f"escalation.reasons.{name}.description"
    if not isinstance(value, str) or not value.strip():
        raise PolicySectionError(field_name, "must be a non-empty string")
    if value != value.strip() or len(value.splitlines()) != 1 or any(
            unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") for c in value):
        raise PolicySectionError(field_name, "must be one line with no surrounding space or control characters")
    if len(value) > MAX_REASON_DESCRIPTION_CHARS:
        raise PolicySectionError(field_name, f"at most {MAX_REASON_DESCRIPTION_CHARS} characters")
    bad = sorted(set(value) & DESCRIPTION_MARKUP)
    if bad:
        raise PolicySectionError(field_name, f"no markup characters ({''.join(bad)})")
    return value


def _reason_descriptions(raw_reasons: Mapping[str, Any], *, strict: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, row in raw_reasons.items():
        if not isinstance(row, Mapping) or row.get("description") is None:
            continue
        try:
            out[name] = parse_reason_description(name, row["description"])
        except PolicySectionError as exc:
            if strict:
                raise
            log.error("policy field %s is invalid; the reason has no description", exc.field,
                      extra={"decision": "fallback_no_description"})
    return out


@dataclass(frozen=True, slots=True)
class EscalationRoute:
    """Where an escalation goes. ``reason`` is the effective (possibly defaulted) reason."""

    queue_key: str
    priority: int
    phrase_key: str
    reason: str


@dataclass(frozen=True, slots=True)
class _ToolRow:
    min_level: int
    phrase_key: str | None
    kind: str
    extra: Mapping[str, Any]
    confirm: str = "none"
    mask: ToolMask | None = None

    @property
    def mask_key(self) -> str | None:
        return None if self.mask is None else self.mask.phrase_key


@dataclass(frozen=True, slots=True)
class _PolicyData:
    version: int
    max_level: int
    levels: Mapping[int, tuple[str, ...]]
    factor_defs: Mapping[str, Mapping[str, Any]]
    tools: Mapping[str, _ToolRow]
    default_reason: str
    reasons: Mapping[str, EscalationRoute]
    reason_descriptions: Mapping[str, str]
    web_offers: tuple[WebOffer, ...]
    model_failover: ModelFailoverConfig
    human_leg_transcription: bool
    human_leg_agent_legs: bool
    memory_write: bool
    end_call_enabled: bool
    queues: Mapping[str, Mapping[str, str]]
    limits: Mapping[str, int]
    redaction: RedactionConfig
    ani_trust: AniTrustConfig
    masking: MaskingConfig
    slots: Mapping[str, SlotSettings]
    slot_problems: tuple[SlotPolicyError, ...]
    slot_resolution: ResolutionSettings
    slot_resolution_problems: tuple[SlotPolicyError, ...]
    raw: Mapping[str, Any]


# --------------------------------------------------------------------------- parsing


def _is_int(v: object) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool)


def _parse(doc: Mapping[str, Any]) -> _PolicyData:
    """Validate a seed-file-shaped dict and return immutable policy data."""
    if not isinstance(doc, Mapping):
        raise PolicyError("policy must be a mapping")
    try:
        max_level = doc["max_level"]
        raw_levels = doc["levels"]
        raw_tools = doc["tools"]
        esc = doc["escalation"]
    except KeyError as exc:
        raise PolicyError(f"policy missing {exc.args[0]!r}") from None
    if not _is_int(max_level) or max_level < 1:
        raise PolicyError("max_level must be a positive int")

    # Levels: contiguous from 1 to max_level, factors cumulative.
    try:
        levels = {int(k): v for k, v in raw_levels.items()}
    except (TypeError, ValueError, AttributeError):
        raise PolicyError("level keys must be integers") from None
    if sorted(levels) != list(range(1, max_level + 1)):
        raise PolicyError(f"levels must be contiguous 1..{max_level}, got {sorted(levels)}")
    factor_defs = doc.get("factors", {}) or {}
    if not isinstance(factor_defs, Mapping):
        raise PolicyError("factors must be a mapping")
    parsed_levels: dict[int, tuple[str, ...]] = {}
    prev: tuple[str, ...] = ()
    for n in range(1, max_level + 1):
        fs = levels[n]
        if not isinstance(fs, list) or not all(isinstance(f, str) and f for f in fs):
            raise PolicyError(f"level {n} factors must be a list of names")
        if len(set(fs)) != len(fs):
            raise PolicyError(f"level {n} lists a factor twice")
        if not set(prev) <= set(fs):
            raise PolicyError(f"level {n} factors are not cumulative over level {n - 1}")
        for f in fs:
            if f not in factor_defs and f not in ESTABLISHED_FACTORS:
                raise PolicyError(f"level {n} uses undefined factor {f!r}")
        parsed_levels[n] = tuple(fs)
        prev = tuple(fs)

    # Tools.
    if not isinstance(raw_tools, Mapping):
        raise PolicyError("tools must be a mapping")
    tools: dict[str, _ToolRow] = {}
    for name, row in raw_tools.items():
        if not isinstance(name, str) or not name or not isinstance(row, Mapping):
            raise PolicyError(f"bad tool row {name!r}")
        if "phrase_key" not in row:
            raise PolicyError(f"tool {name!r} has no phrase_key field")
        ml, pk, kind = row.get("min_level"), row["phrase_key"], row.get("kind")
        if not _is_int(ml) or not 0 <= ml <= max_level:
            raise PolicyError(f"tool {name!r} min_level must be an int in 0..{max_level}")
        if kind not in TOOL_KINDS:
            raise PolicyError(f"tool {name!r} has unknown kind {kind!r}")
        if pk is not None and (not isinstance(pk, str) or not pk):
            raise PolicyError(f"tool {name!r} phrase_key must be a non-empty string or null")
        if pk is None and ml > 0 and kind in ("agentic_inline", "deterministic"):
            raise PolicyError(f"tool {name!r} needs a phrase_key (min_level {ml})")
        extra = {k: v for k, v in row.items()
                 if k not in ("min_level", "phrase_key", "kind", "confirm", "mask")}
        tools[name] = _ToolRow(ml, pk, kind, copy.deepcopy(extra), load_confirm(name, kind, row),
                               load_tool_mask(name, row))

    # Escalation.
    if not isinstance(esc, Mapping):
        raise PolicyError("escalation must be a mapping")
    queues = esc.get("queues", {})
    raw_reasons = esc.get("reasons", {})
    default_reason = esc.get("default_reason")
    if not isinstance(queues, Mapping) or not isinstance(raw_reasons, Mapping):
        raise PolicyError("escalation queues and reasons must be mappings")
    reasons: dict[str, EscalationRoute] = {}
    for name, row in raw_reasons.items():
        if not isinstance(row, Mapping):
            raise PolicyError(f"bad reason row {name!r}")
        q, pri, ph = row.get("queue"), row.get("priority"), row.get("phrase_key")
        if not isinstance(q, str) or q not in queues:
            raise PolicyError(f"reason {name!r} routes to unknown queue {q!r}")
        if not _is_int(pri):
            raise PolicyError(f"reason {name!r} priority must be an int")
        if not isinstance(ph, str) or not ph:
            raise PolicyError(f"reason {name!r} needs a phrase_key")
        reasons[name] = EscalationRoute(q, pri, ph, name)
    if not isinstance(default_reason, str) or default_reason not in reasons:
        raise PolicyError(f"default_reason {default_reason!r} is not a defined reason")

    limits = doc.get("limits", {}) or {}
    if not isinstance(limits, Mapping) or not all(_is_int(v) for v in limits.values()):
        raise PolicyError("limits must map names to ints")
    version = doc.get("version", 0)
    if not _is_int(version):
        raise PolicyError("version must be an int")
    slots = _parse_slots(doc.get("slots"), factor_defs, from_store=_SLOT_PROBLEMS_KEY in doc)
    problems = _internal_problems(doc, _SLOT_PROBLEMS_KEY)
    slot_resolution = _parse_slot_resolution(doc.get(RESOLUTION_ITEM))
    resolution_problems = _internal_problems(doc, _RESOLUTION_PROBLEMS_KEY)

    return _PolicyData(
        version=version,
        max_level=max_level,
        levels=parsed_levels,
        factor_defs=copy.deepcopy(dict(factor_defs)),
        tools=tools,
        default_reason=default_reason,
        reasons=reasons,
        reason_descriptions=_reason_descriptions(raw_reasons, strict=False),
        web_offers=load_web_offers(doc.get(WEB_OFFERS_ITEM)),
        model_failover=load_model_failover(doc.get(MODEL_FAILOVER_ITEM)),
        human_leg_transcription=load_human_leg_transcription(doc.get(HUMAN_LEG_ITEM)),
        human_leg_agent_legs=load_human_leg_agent_legs(doc.get(HUMAN_LEG_ITEM)),
        memory_write=load_memory_write(doc.get(MEMORY_ITEM)),
        end_call_enabled=load_end_call(doc.get(END_CALL_ITEM)),
        queues=copy.deepcopy(dict(queues)),
        limits=dict(limits),
        redaction=load_redaction_config(doc.get("redaction")),
        ani_trust=load_ani_trust(doc.get("ani_trust")),
        masking=load_masking(doc.get("masking")),
        slots=slots,
        slot_problems=tuple(problems),
        slot_resolution=slot_resolution,
        slot_resolution_problems=tuple(resolution_problems),
        raw=copy.deepcopy({k: v for k, v in doc.items()
                           if k not in ("_notes", _SLOT_PROBLEMS_KEY, _RESOLUTION_PROBLEMS_KEY)}),
    )


#: Where :func:`_doc_from_store` hands the lenient loaders' problems to :func:`_parse`.
_SLOT_PROBLEMS_KEY = "_slot_problems"
_RESOLUTION_PROBLEMS_KEY = "_slot_resolution_problems"


def _internal_problems(doc: Mapping[str, Any], key: str) -> list[SlotPolicyError]:
    problems = doc.get(key, ())
    if not isinstance(problems, (list, tuple)) or not all(isinstance(e, SlotPolicyError) for e in problems):
        raise PolicyError(f"{key} is internal to the store loader")
    return list(problems)


def _parse_slot_resolution(raw: Any) -> ResolutionSettings:
    """The ``slot_resolution`` section, strict; absent gives the defaults (the store loader always
    hands one over, lenient-loaded)."""
    if raw is None:
        return ResolutionSettings()
    try:
        return parse_slot_resolution(raw)
    except SlotPolicyError as exc:
        raise PolicySectionError(exc.field, exc.message) from None


def _parse_slots(raw: Any, factor_defs: Mapping[str, Any], *, from_store: bool) -> dict[str, SlotSettings]:
    """The ``slots`` section, strict. Every ``factors`` entry is a slot; one with no entry in the
    section gets ``sensitive: true``. Order: the factors, then the other slots.

    ``from_store``: the section was rebuilt by :func:`_slots_from_store`, which already gave every
    loadable slot an entry; a factor without one couldn't be loaded and stays out."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise PolicyError("slots must be a mapping")
    if not from_store:  # the seed file: ``sensitive`` moved to the slot's settings
        for name, fd in factor_defs.items():
            if isinstance(fd, Mapping) and "sensitive" in fd:
                raise PolicySectionError(f"factors.{name}.sensitive",
                                         f"moved to slots.{name}.sensitive")
    names = [*(n for n in factor_defs if not from_store or n in raw),
             *(n for n in raw if n not in factor_defs)]
    out: dict[str, SlotSettings] = {}
    for name in names:
        entry = raw.get(name, {"sensitive": True})
        try:
            out[name] = parse_slot(name, entry, factor_def=factor_defs.get(name))
        except SlotPolicyError as exc:
            raise PolicySectionError(exc.field, exc.message) from None
    return out


def _doc_from_store(store: Store) -> dict[str, Any]:
    """Rebuild the seed-file-shaped dict from the ``policy`` table.

    Two round trips: the ``meta`` item, then one ``BatchGetItem`` of every
    item it names (levels, tools, reasons, ``slot#*``, ``slot_resolution`` and the sections). The
    gathered items are then read in the same order, with the same checks and errors, as one
    ``GetItem`` each would be. Levels past the first :data:`_LEVELS_IN_FIRST_BATCH` are
    fetched a chunk at a time, only while every earlier level exists."""
    with timed_reload("policy") as call:
        reader = _BatchReader(store)
        doc = _doc_from_items(reader, store)
        call.items = reader.count
    return doc


#: Levels fetched with the first batch; a larger ``max_level`` reads the rest in chunks as needed.
_LEVELS_IN_FIRST_BATCH = 100


class _BatchReader:
    """``read(pk)``: the ``policy`` item, from what :meth:`fetch` gathered (``None`` if absent)."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._items: dict[str, Item] = {}
        self._fetched: set[str] = set()
        self.count = 0

    def fetch(self, pks: list[str]) -> None:
        want = [pk for pk in dict.fromkeys(pks) if pk not in self._fetched]
        if not want:
            return
        got = self._store.batch_get("policy", [{"pk": pk} for pk in want])
        self._fetched.update(want)
        self.count += len(got)
        for item in got:
            self._items[item["pk"]] = item

    def read(self, pk: str) -> Item | None:
        if pk not in self._fetched:
            self.fetch([pk])
        item = self._items.get(pk)
        return None if item is None else copy.deepcopy(item)


def _names(meta: Mapping[str, Any], field: str) -> list[Any] | TypeError:
    """``list(meta[field])`` (default empty), or the error iterating it raises, for later."""
    try:
        return list(meta.get(field, []))
    except TypeError as exc:
        return exc


def _listed(names: list[Any] | TypeError) -> list[Any]:
    if isinstance(names, TypeError):
        raise names
    return names


def _doc_from_items(reader: _BatchReader, store: Store) -> dict[str, Any]:
    meta = store.get("policy", {"pk": META_PK})
    if meta is None:
        raise PolicyError("policy table has no meta item")
    reader.count += 1
    max_level = meta.get("max_level")
    if not _is_int(max_level) or max_level < 1:
        raise PolicyError("policy meta max_level must be a positive int")
    tool_names, reason_names = _names(meta, "tools"), _names(meta, "reasons")
    listed_slots = meta.get("slots")
    first_levels = min(max_level, _LEVELS_IN_FIRST_BATCH)
    reader.fetch([
        *(f"level#{n}" for n in range(1, first_levels + 1)),
        *(f"tool#{name}" for name in ([] if isinstance(tool_names, TypeError) else tool_names)),
        *(f"reason#{name}" for name in ([] if isinstance(reason_names, TypeError) else reason_names)),
        *(f"slot#{name}" for name in (listed_slots if isinstance(listed_slots, list) else [])
          if isinstance(name, str) and name),
        RESOLUTION_ITEM, "redaction", "ani_trust", "masking", WEB_OFFERS_ITEM,
        MODEL_FAILOVER_ITEM, HUMAN_LEG_ITEM, MEMORY_ITEM, END_CALL_ITEM,
    ])
    levels: dict[str, Any] = {}
    for n in range(1, max_level + 1):
        if n > first_levels and (n - 1) % _LEVELS_IN_FIRST_BATCH == 0:
            reader.fetch([f"level#{m}" for m in range(n, min(max_level, n + _LEVELS_IN_FIRST_BATCH - 1) + 1)])
        item = reader.read(f"level#{n}")
        if item is None:
            raise PolicyError(f"policy table has no level#{n}")
        levels[str(n)] = item.get("factors")
    tools: dict[str, Any] = {}
    for name in _listed(tool_names):
        item = reader.read(f"tool#{name}")
        if item is None:
            raise PolicyError(f"policy meta lists tool {name!r} but has no row")
        tools[name] = {k: v for k, v in item.items() if k not in _STORE_ONLY_ATTRS}
    reasons: dict[str, Any] = {}
    for name in _listed(reason_names):
        item = reader.read(f"reason#{name}")
        if item is None:
            raise PolicyError(f"policy meta lists reason {name!r} but has no row")
        reasons[name] = {k: v for k, v in item.items() if k not in _STORE_ONLY_ATTRS}
    factors = meta.get("factors", {})
    slots, problems = _slots_from_store(reader.read, listed_slots,
                                        factors if isinstance(factors, Mapping) else {})
    resolution, resolution_problems = load_slot_resolution(_section_from_store(reader.read, RESOLUTION_ITEM))
    for p in resolution_problems:
        log.error("policy slot_resolution field %s is invalid (%s)", p.field, p.message,
                  extra={"decision": "slot_resolution_fail_closed", "field": p.field})
    return {
        "version": meta.get("version", 0),
        "max_level": max_level,
        "levels": levels,
        "factors": factors,
        "slots": slots,
        _SLOT_PROBLEMS_KEY: problems,
        RESOLUTION_ITEM: resolution.to_doc(),
        _RESOLUTION_PROBLEMS_KEY: resolution_problems,
        "tools": tools,
        "escalation": {
            "default_reason": meta.get("default_reason"),
            "queues": meta.get("queues", {}),
            "reasons": reasons,
        },
        "limits": meta.get("limits", {}),
        "redaction": _section_from_store(reader.read, "redaction"),
        "ani_trust": _section_from_store(reader.read, "ani_trust"),
        "masking": _section_from_store(reader.read, "masking"),
        WEB_OFFERS_ITEM: _section_from_store(reader.read, WEB_OFFERS_ITEM),
        MODEL_FAILOVER_ITEM: _section_from_store(reader.read, MODEL_FAILOVER_ITEM),
        HUMAN_LEG_ITEM: _section_from_store(reader.read, HUMAN_LEG_ITEM),
        MEMORY_ITEM: _section_from_store(reader.read, MEMORY_ITEM),
        END_CALL_ITEM: _section_from_store(reader.read, END_CALL_ITEM),
    }


#: Item attributes that aren't policy data: the key and the admin console's ``rev``,
#: which the bridge ignores (a section validator would otherwise see ``rev`` as unknown).
_STORE_ONLY_ATTRS: frozenset[str] = frozenset({"pk", REV_ATTR})


def _slots_from_store(read: Callable[[str], Item | None], listed: Any, factor_defs: Mapping[str, Any]
                      ) -> tuple[dict[str, Any], list[SlotPolicyError]]:
    """The ``slots`` section rebuilt from ``meta.slots`` and the ``slot#<name>`` items, lenient
    Each slot's loaded settings as a valid entry, and the problems, each logged at
    ERROR naming the item and field (never a value). A ``factors`` entry not listed or with no
    item loads ``sensitive: true``; a non-factor slot that can't be loaded is left out."""
    problems: list[SlotPolicyError] = []
    if listed is None:
        listed = []
    if not isinstance(listed, list) or not all(isinstance(n, str) and n for n in listed):
        problems.append(SlotPolicyError("meta.slots", "must be a list of names; ignored"))
        listed = []
    # A table seeded before ``sensitive`` moved still has it in ``meta.factors``: it is not read
    # (the slot item decides); the next seed rewrites ``meta`` without it.
    problems.extend(SlotPolicyError(f"factors.{name}.sensitive", f"moved to slot#{name}; ignored")
                    for name, fd in factor_defs.items() if isinstance(fd, Mapping) and "sensitive" in fd)
    names = [*factor_defs, *(n for n in dict.fromkeys(listed) if n not in factor_defs)]
    out: dict[str, Any] = {}
    for name in names:
        item = _section_from_store(read, f"slot#{name}") if name in listed else None
        settings, errors = load_slot(name, item, factor_def=factor_defs.get(name))
        problems.extend(errors)
        if settings is not None:
            out[name] = settings.to_doc()
    for p in problems:
        log.error("policy slot field %s is invalid (%s)", p.field, p.message,
                  extra={"decision": "slot_fail_closed", "field": p.field})
    return out, problems


def _section_from_store(read: Callable[[str], Item | None], pk: str) -> Any:
    item = read(pk)
    return None if item is None else {k: v for k, v in item.items() if k not in _STORE_ONLY_ATTRS}


def confirm_prompt_keys(tool: str) -> tuple[str, ...]:
    """Catalog keys a ``deterministic`` confirmation of ``tool`` speaks."""
    return (f"confirm.{tool}", f"confirm.{tool}.done", "confirm.cancelled", "confirm.failed",
            "confirm.insufficient_funds", "clarify.yes_no")


# --------------------------------------------------------------------------- Policy


class Policy:
    """Read-only view of the auth and escalation policy, with an optional refresh cache."""

    def __init__(
        self,
        loader: Callable[[], Mapping[str, Any]],
        *,
        ttl_s: float | None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loader = loader
        self._ttl = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._data: _PolicyData = _parse(loader())  # first load must succeed
        self._loaded_at = clock()

    # -- constructors

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> Policy:
        """A static policy from the seed-file shape (validated now; never refreshed)."""
        frozen = copy.deepcopy(dict(doc))
        return cls(lambda: frozen, ttl_s=None)

    @classmethod
    def from_store(
        cls, store: Store, *, ttl_s: float = DEFAULT_TTL_S, clock: Callable[[], float] = time.monotonic
    ) -> Policy:
        """A policy read from the ``policy`` table, re-read after ``ttl_s`` seconds."""
        return cls(lambda: _doc_from_store(store), ttl_s=ttl_s, clock=clock)

    @staticmethod
    def to_items(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Store items for the ``policy`` table from a seed-file-shaped dict (validated)."""
        data = _parse(doc)
        redaction: RedactionConfig | None = None
        if doc.get("redaction") is not None:
            try:
                redaction = parse_redaction_config(doc["redaction"])
            except RedactionConfigError as exc:
                raise PolicyError(str(exc)) from None
        ani_trust: AniTrustConfig | None = None
        if doc.get("ani_trust") is not None:
            ani_trust = parse_ani_trust(doc["ani_trust"])
        masking: MaskingConfig | None = None
        if doc.get("masking") is not None:
            masking = parse_masking(doc["masking"])
        for name, row in doc["tools"].items():  # strict: seeding refuses a bad confirm or mask
            parse_confirm(name, row.get("kind"), row)
            parse_tool_mask(name, row)
        descriptions = _reason_descriptions(doc["escalation"].get("reasons", {}), strict=True)
        offers = parse_web_offers(doc.get(WEB_OFFERS_ITEM))
        failover = parse_model_failover(doc.get(MODEL_FAILOVER_ITEM))
        items: list[dict[str, Any]] = [
            {
                "pk": META_PK,
                "version": data.version,
                "max_level": data.max_level,
                "tools": list(data.tools),
                "reasons": list(data.reasons),
                "default_reason": data.default_reason,
                "queues": copy.deepcopy(dict(data.queues)),
                "factors": copy.deepcopy(dict(data.factor_defs)),
                "limits": dict(data.limits),
                "slots": list(data.slots),
            }
        ]
        for n, fs in data.levels.items():
            items.append({"pk": f"level#{n}", "factors": list(fs)})
        for name, row in data.tools.items():
            items.append(
                {"pk": f"tool#{name}", "min_level": row.min_level, "phrase_key": row.phrase_key,
                 "kind": row.kind, **copy.deepcopy(dict(row.extra)),
                 **({} if _is_gateway_tool(name, row.kind) else {"confirm": row.confirm}),
                 **({} if row.mask is None else {"mask": row.mask.to_doc()})}
            )
        for name, r in data.reasons.items():
            items.append({"pk": f"reason#{name}", "queue": r.queue_key, "priority": r.priority,
                          "phrase_key": r.phrase_key,
                          **({"description": descriptions[name]} if name in descriptions else {})})
        # Always written (empty when the section is absent): the seed only puts items, so a
        # reseed without the section must overwrite a live list rather than leave it in place.
        items.append({"pk": WEB_OFFERS_ITEM, "offers": [o.to_doc() for o in offers]})
        items.append({"pk": MODEL_FAILOVER_ITEM, **failover.to_doc()})  # likewise
        items.append({"pk": HUMAN_LEG_ITEM,  # likewise: a reseed resets the switch to the file's value
                      "transcription": load_human_leg_transcription(doc.get(HUMAN_LEG_ITEM)),
                      "agent_legs": load_human_leg_agent_legs(doc.get(HUMAN_LEG_ITEM))})
        items.append({"pk": MEMORY_ITEM, "write": load_memory_write(doc.get(MEMORY_ITEM))})  # likewise
        items.append({"pk": END_CALL_ITEM, "enabled": load_end_call(doc.get(END_CALL_ITEM))})  # likewise
        if redaction is not None:
            items.append({"pk": "redaction", **redaction.to_doc()})
        if ani_trust is not None:
            items.append({"pk": "ani_trust", **ani_trust.to_doc()})
        if masking is not None:
            items.append({"pk": "masking", **masking.to_doc()})
        for name, s in data.slots.items():
            items.append({"pk": f"slot#{name}", **s.to_doc()})
        items.append({"pk": RESOLUTION_ITEM, **data.slot_resolution.to_doc()})
        return items

    # -- cache

    def _current(self) -> _PolicyData:
        if self._ttl is None or self._clock() - self._loaded_at < self._ttl:
            return self._data
        with self._lock:
            if self._clock() - self._loaded_at >= self._ttl:
                try:
                    self._data = _parse(self._loader())
                except Exception as exc:
                    log_caught(log, "policy refresh failed; keeping the last good policy", exc,
                               level=logging.ERROR, operation="refresh_policy", decision="keep_last_good")
                self._loaded_at = self._clock()
        return self._data

    def refresh(self) -> None:
        """Force a reload now (raises if the source is invalid; the old policy stays)."""
        data = _parse(self._loader())
        with self._lock:
            self._data, self._loaded_at = data, self._clock()

    # -- reads

    @property
    def max_level(self) -> int:
        return self._current().max_level

    @property
    def version(self) -> int:
        return self._current().version

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(self._current().tools)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(self._current().reasons)

    @property
    def web_offers(self) -> tuple[WebOffer, ...]:
        """The bank's offer list, in policy order; empty without a (valid) section."""
        return self._current().web_offers

    @property
    def model_failover(self) -> ModelFailoverConfig:
        """The failover overrides; the defaults without a (valid) section. Build the
        settings for the deployed list with :meth:`ModelFailoverConfig.settings_for`."""
        return self._current().model_failover

    @property
    def human_leg_transcription(self) -> bool:
        """The human-leg transcription switch; off without a (valid) ``human_leg`` item."""
        return self._current().human_leg_transcription

    @property
    def human_leg_agent_legs(self) -> bool:
        """Per-agent-leg transcription; on without a (valid) ``human_leg`` item."""
        return self._current().human_leg_agent_legs

    @property
    def memory_write(self) -> bool:
        """The per-call Memory recap switch; on without a (valid) ``memory`` item."""
        return self._current().memory_write

    def end_call_enabled(self) -> bool:
        """The ``end_call`` tool switch; on without a (valid) ``end_call`` item.
        A method, not a property: the bridge reads it per session and per call."""
        return self._current().end_call_enabled

    @property
    def reason_descriptions(self) -> dict[str, str]:
        """``{reason: description}`` for the reasons that have one, in policy order."""
        return dict(self._current().reason_descriptions)

    def has_tool(self, tool: str) -> bool:
        return tool in self._current().tools

    def min_level(self, tool: str) -> int:
        """Minimum level for ``tool``. A tool with no row needs the max level (fail closed)."""
        data = self._current()
        row = data.tools.get(tool)
        if row is None:
            log.warning("policy: no row for tool %r; failing closed at level %d",
                        str(tool)[:64], data.max_level)
            return data.max_level
        return row.min_level

    def phrase_key(self, tool: str) -> str | None:
        """Catalog key for the tool's connecting phrase, or ``None`` (use a generic phrase)."""
        row = self._current().tools.get(tool)
        return row.phrase_key if row else None

    def kind(self, tool: str) -> str | None:
        """The tool's kind, or ``None`` for an unknown tool."""
        row = self._current().tools.get(tool)
        return row.kind if row else None

    def tool_row(self, tool: str) -> dict[str, Any] | None:
        """A copy of the tool's full row (including e.g. ``template_key``), or ``None``."""
        row = self._current().tools.get(tool)
        if row is None:
            return None
        return {"min_level": row.min_level, "phrase_key": row.phrase_key, "kind": row.kind,
                "confirm": row.confirm, **copy.deepcopy(dict(row.extra)),
                **({} if row.mask is None else {"mask": row.mask.to_doc()})}

    def tool_mask(self, tool: str) -> ToolMask | None:
        """The tool's masking settings, or ``None`` (unmasked, also for an
        unknown tool or an invalid ``mask``)."""
        row = self._current().tools.get(tool)
        return row.mask if row else None

    def mask_phrase_key(self, tool: str) -> str | None:
        """Catalog key of the tool's latency-masking phrase, or ``None`` (no phrase,
        also for an unknown tool or an invalid ``mask``)."""
        row = self._current().tools.get(tool)
        return row.mask_key if row else None

    def confirm(self, tool: str) -> str:
        """The tool's ``confirm`` mode. An unknown tool gets the built-in default."""
        row = self._current().tools.get(tool)
        return row.confirm if row else default_confirm(str(tool))

    def factors(self, level: int) -> list[str]:
        """Cumulative factors for ``level`` in policy order. Level 0 needs none."""
        data = self._current()
        if not _is_int(level) or not 0 <= level <= data.max_level:
            raise PolicyError(f"no such level {level!r} (0..{data.max_level})")
        return [] if level == 0 else list(data.levels[level])

    def missing_factors(self, have: set[str] | frozenset[str], target_level: int) -> list[str]:
        """Factors of ``target_level`` not in ``have``, in policy order."""
        return [f for f in self.factors(target_level) if f not in have]

    def effective_factors(self, level: int, *, ani_counts: bool) -> list[str]:
        """What ``level`` actually needs on a call.

        If the ANI match counts, the level's factors. If it doesn't (untrusted caller ID
        with ``on_untrusted: no_ani_match``) and the level uses ``ani_match``, the
        *collected* factors of the next level instead (of the level itself at the
        maximum level), so the caller is asked for the next level's factors."""
        fs = self.factors(level)
        if ani_counts or "ani_match" not in fs:
            return fs
        nxt = self.factors(min(level + 1, self._current().max_level))
        return [f for f in nxt if f not in ESTABLISHED_FACTORS]

    def slots(self) -> tuple[str, ...]:
        """Every slot's name: the ``factors`` entries in policy order, then the other slots."""
        return tuple(self._current().slots)

    def slot(self, name: str) -> SlotSettings:
        """A slot's settings. ``KeyError`` for a name that isn't a slot."""
        return self._current().slots[name]

    @property
    def slot_problems(self) -> tuple[SlotPolicyError, ...]:
        """What the lenient table loader found wrong with the slot items (empty for a seed file)."""
        return self._current().slot_problems

    @property
    def slot_resolution(self) -> ResolutionSettings:
        """The ``slot_resolution`` item: ``default_mode``, flow overrides, ``generative_enabled``,
        ``resolver_budget_ms``."""
        return self._current().slot_resolution

    @property
    def slot_resolution_problems(self) -> tuple[SlotPolicyError, ...]:
        """What the lenient table loader found wrong with ``slot_resolution`` (empty for a seed file)."""
        return self._current().slot_resolution_problems

    @property
    def invalid_mode_fields(self) -> frozenset[str]:
        """Every mode field (``slot#zip.mode``, ``slot_resolution.default_mode``,
        ``slot_resolution.flows.login.mode``) that was missing where required, unknown or of the
        wrong type and loaded as ``deterministic``."""
        data = self._current()
        return frozenset(e.field for e in (*data.slot_problems, *data.slot_resolution_problems)
                         if e.mode_invalid)

    def factor(self, name: str) -> dict[str, Any] | None:
        """A copy of a collected factor's definition (slot type, prompt key...), or ``None``."""
        d = self._current().factor_defs.get(name)
        return copy.deepcopy(dict(d)) if d is not None else None

    def escalation_route(self, reason: str) -> EscalationRoute:
        """Route for an escalation reason; an unknown reason routes as the default reason."""
        data = self._current()
        route = data.reasons.get(reason) if isinstance(reason, str) else None
        if route is None:
            log.warning("policy: unknown escalation reason %r; using %r",
                        str(reason)[:64], data.default_reason)
            return data.reasons[data.default_reason]
        return route

    def queue_name(self, queue_key: str, lang: str, *, default_lang: str = "en-US") -> str:
        """TaskRouter queue name for a queue key and language (falls back to ``default_lang``)."""
        names = self._current().queues[queue_key]
        if lang in names:
            return names[lang]
        log.warning("policy: queue %r has no %r entry; using %r", queue_key, lang, default_lang)
        return names[default_lang]

    @property
    def redaction(self) -> RedactionConfig:
        """The PII redaction config (defaults if the section is missing or invalid)."""
        return self._current().redaction

    @property
    def ani_trust(self) -> AniTrustConfig:
        """The caller-ID trust gate (defaults if the section is missing or invalid)."""
        return self._current().ani_trust

    @property
    def masking(self) -> MaskingConfig:
        """The latency-masking timings (defaults if missing, per-field defaults if invalid)."""
        return self._current().masking

    def ani_trusted(self, verstat: Any) -> bool:
        return self._current().ani_trust.trusted(verstat)

    def ani_match_counts(self, verstat: Any) -> bool:
        return self._current().ani_trust.ani_match_counts(verstat)

    def limit(self, name: str) -> int:
        """A numeric limit (e.g. ``stepup_attempts_per_call``); ``KeyError`` if absent."""
        return self._current().limits[name]

    def referenced_prompt_keys(self) -> set[str]:
        """Every catalog key the policy points to (for cross-validation with the catalog)."""
        data = self._current()
        keys: set[str] = set()
        for row in data.tools.values():
            if row.phrase_key:
                keys.add(row.phrase_key)
            if row.mask_key:
                keys.add(row.mask_key)
            tk = row.extra.get("template_key")
            if isinstance(tk, str):
                keys.add(tk)
        for name, row in data.tools.items():
            if row.confirm == "deterministic":
                keys.update(confirm_prompt_keys(name))
        keys.update(r.phrase_key for r in data.reasons.values())
        if data.masking.after_ms > 0:
            keys.update(MASK_GENERIC_KEYS)
        for d in data.factor_defs.values():
            for f in ("promptKey", "labelKey"):
                if isinstance(d.get(f), str):
                    keys.add(d[f])
        return keys
