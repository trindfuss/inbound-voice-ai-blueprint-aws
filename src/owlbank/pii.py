"""PII detection with Amazon Comprehend, masking in our own code.

The factor guard in :mod:`owlbank.segments` only knows the values the front door
collected, and only in that process. Free text the caller says to the AI (card
numbers, SSNs, account numbers...) can reach the journal (``ai_turn``), the
escalation summary, task attributes and Flex. This module finds and masks it:

* :class:`ComprehendPiiRedactor` calls ``DetectPiiEntities`` and masks the spans
  itself. It never uses Comprehend's own redaction output.
* :class:`FakePiiRedactor` (tests) finds a few shapes with regexes and shares the
  masking code.
* :class:`PiiGuard` applies the deploying bank's :class:`RedactionConfig` (from
  ``data/policy.json`` ``redaction``, through :class:`owlbank.policy.Policy`) to
  a surface's text: whether transcript redaction is on (``enabled``; it covers every
  surface), which types, the threshold, the mask, and what
  happens when the check is unavailable (``on_failure``). The surface is a label for
  the logs only.

**Off** (``enabled: false``): no Comprehend call is made and the text passes
through; each component logs one ``transcript_redaction_off`` WARNING per process.
With ``enabled: true`` the type list must not be empty.

**Fail closed.** Any Comprehend error or timeout raises :class:`PiiCheckUnavailable`.
With ``on_failure: withhold`` (the default) :meth:`PiiGuard.text` returns the
caller's ``withheld`` text instead; ``allow`` returns the text unredacted and logs
a WARNING each time. Text is never logged.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from owlbank import obs
from owlbank.obs import log_caught, timed_call

log = logging.getLogger(__name__)

#: ``DetectPiiEntities`` entity types (AWS Comprehend docs, "Detecting PII entities";
#: matches botocore's ``PiiEntityType`` enum without ``ALL``, which isn't a detected type).
COMPREHEND_PII_ENTITY_TYPES: frozenset[str] = frozenset({
    "ADDRESS", "AGE", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "BANK_ACCOUNT_NUMBER", "BANK_ROUTING",
    "CA_HEALTH_NUMBER", "CA_SOCIAL_INSURANCE_NUMBER", "CREDIT_DEBIT_CVV", "CREDIT_DEBIT_EXPIRY",
    "CREDIT_DEBIT_NUMBER", "DATE_TIME", "DRIVER_ID", "EMAIL", "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
    "IN_AADHAAR", "IN_NREGA", "IN_PERMANENT_ACCOUNT_NUMBER", "IN_VOTER_NUMBER", "IP_ADDRESS",
    "LICENSE_PLATE", "MAC_ADDRESS", "NAME", "PASSPORT_NUMBER", "PASSWORD", "PHONE", "PIN", "SSN",
    "SWIFT_CODE", "UK_NATIONAL_HEALTH_SERVICE_NUMBER", "UK_NATIONAL_INSURANCE_NUMBER",
    "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER", "URL", "USERNAME",
    "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "VEHICLE_IDENTIFICATION_NUMBER",
})

#: Surface labels (for the logs; the setting covers the whole transcript).
SURFACE_ESCALATION_SUMMARY = "escalation_summary"
SURFACE_AI_TURN = "ai_turn"
#: The ``speech`` field of the front door's ``factor_answer`` log line.
SURFACE_SPEECH_LOG = "speech_log"

ON_FAILURE_WITHHOLD = "withhold"
ON_FAILURE_ALLOW = "allow"
ON_FAILURE: frozenset[str] = frozenset({ON_FAILURE_WITHHOLD, ON_FAILURE_ALLOW})

#: Built-in safe defaults (also the ``redaction`` section shipped in ``data/policy.json``).
DEFAULT_ENTITY_TYPES: tuple[str, ...] = (
    "SSN", "CREDIT_DEBIT_NUMBER", "CREDIT_DEBIT_CVV", "CREDIT_DEBIT_EXPIRY",
    "BANK_ACCOUNT_NUMBER", "BANK_ROUTING", "PIN", "PASSWORD",
)
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MASK = "[{type}]"

#: The journal text for an ``ai_turn`` whose PII check was unavailable.
WITHHELD_TURN = "[WITHHELD]"

#: Comprehend's input limit is 100 KB of UTF-8; stay under it.
COMPREHEND_MAX_BYTES = 99_000
#: Short client timeouts: the check sits on the caller's path.
COMPREHEND_CONNECT_TIMEOUT_S = 1.0
COMPREHEND_READ_TIMEOUT_S = 2.0

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
#: An unknown field is named in the ERROR only when it looks like a field name (no free text).
#: Used with ``fullmatch``: ``$`` also matches before a trailing newline.
_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,39}")
_FIELDS = ("enabled", "entity_types", "min_confidence", "mask", "on_failure")


class PiiCheckUnavailable(RuntimeError):
    """The PII check could not run (service error, timeout, bad response). Fail closed."""


class RedactionConfigError(ValueError):
    """A ``redaction`` config field is invalid. ``field`` names it (never the text)."""

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(f"redaction.{field_name}: {message}")
        self.field = field_name


# --------------------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class RedactionConfig:
    """Whether to redact the transcript, and what to detect and mask (``data/policy.json``
    ``redaction``). :func:`parse_redaction_config` refuses ``enabled`` with no types."""

    enabled: bool = True
    entity_types: tuple[str, ...] = DEFAULT_ENTITY_TYPES
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    mask: str = DEFAULT_MASK
    on_failure: str = ON_FAILURE_WITHHOLD

    def render_mask(self, entity_type: str) -> str:
        return self.mask.replace("{type}", entity_type)

    def to_doc(self) -> dict[str, Any]:
        """The seed-file shape. ``min_confidence`` as a decimal string (the store has no floats)."""
        return {"enabled": self.enabled, "entity_types": list(self.entity_types),
                "min_confidence": repr(self.min_confidence), "mask": self.mask,
                "on_failure": self.on_failure}


DEFAULT_CONFIG = RedactionConfig()


def _str_list(raw: Any, name: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not all(isinstance(x, str) for x in raw):
        raise RedactionConfigError(name, "must be a list of strings")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise RedactionConfigError(name, f"unknown values {unknown[:5]}")
    return tuple(dict.fromkeys(raw))


def parse_redaction_config(raw: Any) -> RedactionConfig:
    """Validate a ``redaction`` section; :class:`RedactionConfigError` on the first bad field.

    Missing fields take the defaults. ``min_confidence`` may be a number or a decimal string.
    ``enabled`` must be a boolean; with ``enabled: true``, ``entity_types`` must not be empty. With ``enabled:
    false`` the other fields are still validated (they are kept for
    when it is switched back on).
    """
    if not isinstance(raw, Mapping):
        raise RedactionConfigError("(section)", "must be a mapping")
    extra = sorted(str(k) for k in raw if k not in _FIELDS)
    if extra:
        named = [k for k in extra if _FIELD_NAME.fullmatch(k)]
        raise RedactionConfigError(named[0] if named else "(unknown field)", "unknown field")
    d = DEFAULT_CONFIG
    enabled = raw.get("enabled", d.enabled)
    if not isinstance(enabled, bool):
        raise RedactionConfigError("enabled", "must be true or false")
    types = _str_list(raw.get("entity_types", list(d.entity_types)), "entity_types",
                      COMPREHEND_PII_ENTITY_TYPES)
    if enabled and not types:
        raise RedactionConfigError("entity_types", "must not be empty while enabled")
    conf_raw = raw.get("min_confidence", d.min_confidence)
    if isinstance(conf_raw, bool):
        raise RedactionConfigError("min_confidence", "must be a number")
    try:
        conf = float(conf_raw)
    except (TypeError, ValueError):
        raise RedactionConfigError("min_confidence", "must be a number") from None
    if not 0.0 <= conf <= 1.0:  # also rejects NaN
        raise RedactionConfigError("min_confidence", "must be between 0 and 1")
    mask = raw.get("mask", d.mask)
    if not isinstance(mask, str) or not mask:
        raise RedactionConfigError("mask", "must be a non-empty string")
    if any(m != "type" for m in _PLACEHOLDER.findall(mask)) or \
            "{" in _PLACEHOLDER.sub("", mask) or "}" in _PLACEHOLDER.sub("", mask):
        raise RedactionConfigError("mask", "may contain only the {type} placeholder")
    on_failure = raw.get("on_failure", d.on_failure)
    if not isinstance(on_failure, str) or on_failure not in ON_FAILURE:
        raise RedactionConfigError("on_failure", f"must be one of {sorted(ON_FAILURE)}")
    return RedactionConfig(enabled=enabled, entity_types=types, min_confidence=conf, mask=mask,
                           on_failure=on_failure)


def load_redaction_config(raw: Any) -> RedactionConfig:
    """:func:`parse_redaction_config`, but never raises: a missing or invalid section gives
    :data:`DEFAULT_CONFIG` (enabled, the default types) and one ERROR naming the field, never
    a value."""
    if raw is None:
        log.error("policy redaction section is missing; using the built-in defaults",
                  extra={"field": "(section)", "decision": "fallback"})
        return DEFAULT_CONFIG
    try:
        return parse_redaction_config(raw)
    except RedactionConfigError as exc:
        log.error("policy redaction field %s is invalid; using the built-in defaults", exc.field,
                  extra={"field": exc.field, "decision": "fallback"})
        return DEFAULT_CONFIG


# --------------------------------------------------------------------------- results and masking


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    entity_types: tuple[str, ...] = ()  # types masked, sorted, no repeats


@dataclass(frozen=True, slots=True)
class Entity:
    """A detected span: ``text[begin:end]`` (character offsets), Comprehend's shape."""

    begin: int
    end: int
    type: str
    score: float = 1.0


_GAP = re.compile(r"[\s\-]*")


def mask_entities(text: str, entities: Iterable[Entity], config: RedactionConfig) -> RedactionResult:
    """Mask every entity of a configured type with ``score >= min_confidence``.

    Overlapping spans, and spans separated only by spaces or hyphens, are merged
    into one mask labelled with the first span's type.
    """
    wanted = set(config.entity_types)
    spans = sorted(((max(0, e.begin), min(len(text), e.end), e.type) for e in entities
                    if e.type in wanted and e.score >= config.min_confidence and e.end > e.begin),
                   key=lambda s: (s[0], -s[1]))
    merged: list[list[Any]] = []
    for b, e, t in spans:
        if b >= e:
            continue
        if merged and (b <= merged[-1][1] or _GAP.fullmatch(text, merged[-1][1], b)):
            last = merged[-1]
            last[1] = max(last[1], e)
            last[3].add(t)
        else:
            merged.append([b, e, t, {t}])
    if not merged:
        return RedactionResult(text, ())
    out, pos, types = [], 0, set()
    for b, e, t, ts in merged:
        out.append(text[pos:b])
        out.append(config.render_mask(t))
        types |= ts
        pos = e
    out.append(text[pos:])
    return RedactionResult("".join(out), tuple(sorted(types)))


def chunks(text: str, max_bytes: int = COMPREHEND_MAX_BYTES) -> list[tuple[int, str]]:
    """``(offset, chunk)`` pieces of ``text``, each at most ``max_bytes`` of UTF-8, cut at the
    last whitespace when there is one (a hard cut at a character boundary otherwise)."""
    if max_bytes < 4:
        raise ValueError("max_bytes must be at least 4")
    out: list[tuple[int, str]] = []
    start, n = 0, len(text)
    while start < n:
        size, i = 0, start
        while i < n:
            b = len(text[i].encode("utf-8", "surrogatepass"))
            if size + b > max_bytes:
                break
            size += b
            i += 1
        end = i
        if end < n:
            cut = max((m.end() for m in re.finditer(r"\s+", text[start:end])), default=0)
            if cut > 0:
                end = start + cut
        out.append((start, text[start:end]))
        start = end
    return out


def comprehend_language(language: str | None) -> str:
    """``en-US`` -> ``en``, ``es-US`` -> ``es``; anything else -> ``en``."""
    return "es" if isinstance(language, str) and language.lower().startswith("es") else "en"


# --------------------------------------------------------------------------- redactors


class PiiRedactor(Protocol):
    def redact(self, text: str, *, language: str,
               config: RedactionConfig | None = None) -> RedactionResult: ...


class ComprehendPiiRedactor:
    """``DetectPiiEntities`` per chunk; masking with :func:`mask_entities`."""

    def __init__(self, client: Any, *, max_bytes: int = COMPREHEND_MAX_BYTES) -> None:
        self._client = client
        self._max_bytes = max_bytes

    def redact(self, text: str, *, language: str,
               config: RedactionConfig | None = None) -> RedactionResult:
        config = config or DEFAULT_CONFIG
        if not isinstance(text, str) or not text.strip() or not config.entity_types:
            return RedactionResult(text if isinstance(text, str) else "", ())
        lang = comprehend_language(language)
        entities: list[Entity] = []
        for offset, chunk in chunks(text, self._max_bytes):
            if not chunk.strip():
                continue
            try:
                with timed_call("comprehend", "detect_pii_entities"):
                    resp = self._client.detect_pii_entities(Text=chunk, LanguageCode=lang)
                for e in resp.get("Entities") or []:
                    entities.append(Entity(offset + int(e["BeginOffset"]), offset + int(e["EndOffset"]),
                                           str(e["Type"]), float(e.get("Score", 0.0))))
            except Exception as exc:  # ClientError, timeouts, endpoint errors, bad shapes
                raise PiiCheckUnavailable(f"PII check unavailable ({type(exc).__name__})") from None
        return mask_entities(text, entities, config)


def make_comprehend_client(region_name: str | None = None) -> Any:  # pragma: no cover - needs boto3
    """A Comprehend client with short timeouts and no retries (the caller is waiting)."""
    import boto3  # noqa: PLC0415
    from botocore.config import Config  # noqa: PLC0415

    return boto3.client("comprehend", region_name=region_name, config=Config(
        connect_timeout=COMPREHEND_CONNECT_TIMEOUT_S, read_timeout=COMPREHEND_READ_TIMEOUT_S,
        retries={"max_attempts": 1}))


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


_DIGIT_RUN = re.compile(r"(?<!\d)\d(?:[ \-]?\d)*(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}[ \-]?\d{2}[ \-]?\d{4}(?!\d)")
_LABELLED = (
    ("BANK_ROUTING", re.compile(r"routing(?:\s+number)?(?:\s+is)?[\s:]*(\d(?:[ \-]?\d){3,}(?![\d]))", re.I)),
    ("BANK_ACCOUNT_NUMBER", re.compile(r"account\s+number(?:\s+is)?[\s:]*(\d(?:[ \-]?\d){3,}(?![\d]))",
                                       re.I)),
    ("PIN", re.compile(r"\bpin(?:\s+is)?[\s:]*(\d{4,6})(?!\d)", re.I)),
    ("ADDRESS", re.compile(r"\b(\d{1,6}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
                           r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd))\b")),
)


@dataclass
class FakePiiRedactor:
    """Offline stand-in for Comprehend (tests only). Detects Luhn-valid card numbers (13-19
    digits, any spaces or hyphens), 9-digit SSN shapes, and digits after "routing",
    "account number" or "PIN", plus simple street addresses. ``fail = True`` makes every
    call raise :class:`PiiCheckUnavailable`. ``calls`` counts detection calls."""

    fail: bool = False
    calls: int = 0
    languages: list[str] = field(default_factory=list)

    def detect(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        for m in _DIGIT_RUN.finditer(text):
            found.extend(_cards_in(text, m.start(), m.end()))
        found.extend(Entity(m.start(), m.end(), "SSN") for m in _SSN.finditer(text))
        for etype, rx in _LABELLED:
            found.extend(Entity(m.start(1), m.end(1), etype) for m in rx.finditer(text))
        return found

    def redact(self, text: str, *, language: str,
               config: RedactionConfig | None = None) -> RedactionResult:
        config = config or DEFAULT_CONFIG
        if not isinstance(text, str) or not text.strip() or not config.entity_types:
            return RedactionResult(text if isinstance(text, str) else "", ())
        self.calls += 1
        self.languages.append(comprehend_language(language))
        if self.fail:
            raise PiiCheckUnavailable("PII check unavailable (fake)")
        return mask_entities(text, self.detect(text), config)


def _cards_in(text: str, start: int, end: int) -> list[Entity]:
    """The whole digit run, if any 13-19 digit window of it is Luhn-valid. Masking the
    whole run (not just the window) keeps masking idempotent: no digit tail is left to
    look like a new number on a second pass."""
    digits = "".join(ch for ch in text[start:end] if ch.isdigit())
    for size in range(min(19, len(digits)), 12, -1):
        if any(luhn_ok(digits[i:i + size]) for i in range(len(digits) - size + 1)):
            return [Entity(start, end, "CREDIT_DEBIT_NUMBER")]
    return []


# --------------------------------------------------------------------------- the guard

#: Components that have logged ``transcript_redaction_off`` in this process.
_warned_off: set[str] = set()


def _warn_off_once() -> None:
    # The component the log record will carry (``configure_logging`` or a bound context).
    component = obs._default_component()
    if component in _warned_off:
        return
    _warned_off.add(component)
    log.warning("transcript_redaction_off", extra={"component": component, "decision": "pass_through"})


class PiiGuard:
    """Applies the current :class:`RedactionConfig` to a surface's text.

    ``config`` is a callable (normally ``lambda: policy.redaction``) so a reseeded
    policy takes effect within the policy cache TTL.
    """

    def __init__(self, redactor: PiiRedactor,
                 config: Callable[[], RedactionConfig] | RedactionConfig = DEFAULT_CONFIG) -> None:
        self._redactor = redactor
        self._config = config if callable(config) else (lambda: config)

    @classmethod
    def for_policy(cls, redactor: PiiRedactor, policy: Any) -> PiiGuard:
        return cls(redactor, lambda: policy.redaction)

    def config(self) -> RedactionConfig:
        try:
            cfg = self._config()
        except Exception as exc:
            log_caught(log, "redaction config unavailable; using the built-in defaults", exc,
                       level=logging.ERROR, decision="fallback")
            return DEFAULT_CONFIG
        return cfg if isinstance(cfg, RedactionConfig) else DEFAULT_CONFIG

    def text(self, surface: str, text: str, *, language: str, withheld: str,
             on_failure: str | None = None) -> str:
        """``text`` with PII masked, or ``withheld`` if the check is unavailable
        (``on_failure: withhold``). With redaction off, ``text`` unchanged and no call.
        ``surface`` is a label for the logs. ``on_failure`` overrides the config's for this call
        (the speech log always withholds)."""
        if not isinstance(text, str):
            return ""
        cfg = self.config()
        failure = on_failure if on_failure in ON_FAILURE else cfg.on_failure
        if not cfg.enabled:
            _warn_off_once()
            return text
        if not cfg.entity_types or not text.strip():
            return text
        try:
            return self._redactor.redact(text, language=language, config=cfg).text
        except PiiCheckUnavailable:
            pass
        except Exception as exc:  # a redactor bug is treated like an outage
            log_caught(log, "PII redactor failed", exc, level=logging.ERROR, operation="redact",
                       decision="treated_as_outage")
        if failure == ON_FAILURE_ALLOW:
            log.warning("PII check unavailable; text sent unredacted (on_failure=allow)",
                        extra={"surface": surface, "decision": "allow_unredacted"})
            return text
        log.warning("PII check unavailable; text withheld",
                    extra={"surface": surface, "decision": "withhold"})
        return withheld


__all__: Sequence[str] = (
    "COMPREHEND_PII_ENTITY_TYPES", "ComprehendPiiRedactor", "DEFAULT_CONFIG", "Entity",
    "FakePiiRedactor", "PiiCheckUnavailable", "PiiGuard", "PiiRedactor", "RedactionConfig",
    "RedactionConfigError", "RedactionResult", "SURFACE_AI_TURN", "SURFACE_ESCALATION_SUMMARY",
    "SURFACE_SPEECH_LOG",
    "WITHHELD_TURN", "chunks",
    "comprehend_language", "load_redaction_config", "luhn_ok", "make_comprehend_client",
    "mask_entities", "parse_redaction_config",
)
