"""Slot declarations and resolution results.

A deterministic flow step only *declares* a slot; :func:`owlbank.slots.engine.resolve`
does all recognition. These types are plain frozen dataclasses so they can be
logged, compared and built from JSON flow definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

SlotType = Literal["digits", "yes_no", "choice", "amount", "date"]
SLOT_TYPES: tuple[str, ...] = ("digits", "yes_no", "choice", "amount", "date")

#: Where a resolution came from: the keypad, the deterministic speech
#: parse (a spoken option name is ``speech``, an alias ``alias``, an ordinal reference to the
#: options just listed ``ordinal``), a universal command, the resolver, a digit answer cut
#: short, or nothing.
Source = Literal["dtmf", "speech", "alias", "ordinal", "command", "model", "short_digits", "none"]
SOURCES: tuple[str, ...] = ("dtmf", "speech", "alias", "ordinal", "command", "model",
                            "short_digits", "none")

# ------------------------------------------------------------- slot modes

Match = Literal["exact", "inferred", "ambiguous", "none", "correction", "intent_switch",
                "repeat", "escalate", "cancel", "decline"]
MATCHES: tuple[str, ...] = ("exact", "inferred", "ambiguous", "none", "correction",
                            "intent_switch", "repeat", "escalate", "cancel", "decline")

ModelOutcome = Literal["not_called", "ok", "timeout", "error", "refusal", "malformed", "ungrounded"]
MODEL_OUTCOMES: tuple[str, ...] = ("not_called", "ok", "timeout", "error", "refusal", "malformed",
                                   "ungrounded")

PendingKind = Literal["confirm_digits", "confirm_option", "disambiguate", "not_available_one",
                      "not_available_many", "correction_reask"]
PENDING_KINDS: tuple[str, ...] = ("confirm_digits", "confirm_option", "disambiguate",
                                  "not_available_one", "not_available_many", "correction_reask")

Scope = Literal["current", "earlier"]
SCOPES: tuple[str, ...] = ("current", "earlier")


@dataclass(frozen=True, slots=True)
class SlotDecl:
    """What a prompt step asks for.

    ``constraints`` by slot type:

    * ``digits``: ``length`` or ``minLength``/``maxLength``.
    * ``yes_no``: none.
    * ``choice``: ``options``: list of ``{"value", "labelKey"}`` (``label_key`` also
      accepted), optional ``"dtmf"`` key per option (default: 1-based position).
    * ``amount``: ``min``, ``max`` (numbers or decimal strings), ``currency`` (``USD``).
    * ``date``: ``min``, ``max`` (ISO dates).
    """

    slot_type: SlotType
    constraints: Mapping[str, Any] = field(default_factory=dict)
    label_key: str | None = None
    prompt_key: str | None = None
    max_retries: int = 1

    def __post_init__(self) -> None:
        if self.slot_type not in SLOT_TYPES:
            raise ValueError(f"unknown slot type {self.slot_type!r}")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SlotDecl:
        """Build from the handover's camelCase declaration. A ``sensitive`` key is not read:
        whether a slot is sensitive is a slot setting (:class:`owlbank.slot_policy.SlotSettings`,
        model-assisted slots)."""
        return cls(
            slot_type=data["slotType"],
            constraints=dict(data.get("constraints") or {}),
            label_key=data.get("labelKey"),
            prompt_key=data.get("promptKey"),
            max_retries=int(data.get("maxRetries", 1)),
        )

    def options(self) -> list[dict[str, str]]:
        """``choice`` options normalized to ``{"value", "label_key", "dtmf"}``."""
        out: list[dict[str, str]] = []
        for i, opt in enumerate(self.constraints.get("options") or [], start=1):
            out.append(
                {
                    "value": str(opt["value"]),
                    "label_key": str(opt.get("label_key") or opt.get("labelKey") or ""),
                    "dtmf": str(opt.get("dtmf") or i),
                }
            )
        return out


@dataclass(frozen=True, slots=True)
class SpeechParse:
    """A deterministic speech parse and where it read the value.

    ``value`` is the candidate (validated by the caller). ``value_runs`` are indices into
    ``number_runs(NFC(speech), ("en", "es"))`` of the runs that supplied it, and ``span`` is the
    ``(start, end)`` of the matched phrase in the NFC speech (``yes_no``, ``choice``). Both are
    ``None`` when the parse can't say (no value, an ``amount`` or ``date`` slot, or the words
    can't be located): :func:`owlbank.sensitive.placeholder_text` then falls back."""

    value: str | None
    value_runs: tuple[int, ...] | None = None
    span: tuple[int, int] | None = None


# --------------------------------------------------------------------------- resolution


@dataclass(frozen=True, slots=True)
class OptionView:
    """One known choice option as the call sees it: its value, its spoken name and its aliases,
    all in the call's language."""

    value: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectedSlot:
    """A slot already collected in this flow (``current``) or an earlier one (``earlier``).
    Never its value.

    A choice slot also carries its options (value, spoken name, aliases in the call's language)
    and the caller's options among them (the card in the confirm flow: the cards on file), so
    a correction of it can name a valid option. Which option was chosen is never here."""

    name: str
    slot_type: SlotType
    label: str                       # catalog text in the call's language
    scope: Scope
    verified: bool                   # sent to the auth service
    options: tuple[OptionView, ...] = ()     # choice only: every known option
    caller_options: tuple[str, ...] = ()     # choice only: the caller's, a subset of options

    def __post_init__(self) -> None:
        if self.slot_type not in SLOT_TYPES:
            raise ValueError(f"unknown slot type {self.slot_type!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r}")
        if not isinstance(self.options, tuple) or not isinstance(self.caller_options, tuple):
            raise ValueError("options and caller_options must be tuples")
        if self.slot_type != "choice" and (self.options or self.caller_options):
            raise ValueError("only a choice slot carries options")
        if any(not isinstance(o, OptionView) for o in self.options):
            raise ValueError("options must be OptionView")
        values = {o.value for o in self.options}
        if len(values) != len(self.options):
            raise ValueError("option values must be distinct")
        if any(v not in values for v in self.caller_options):
            raise ValueError("caller_options must be option values")
        if len(set(self.caller_options)) != len(self.caller_options):
            raise ValueError("caller_options must be distinct")


@dataclass(frozen=True, slots=True)
class Pending:
    """A follow-up question the caller is answering."""

    kind: PendingKind
    slot: str
    option: str | None = None                 # the option the question names
    candidates: tuple[str, ...] = ()
    sealed_value: str | None = None           # confirm_digits: the grounded value, sealed
    #: The option values in the order the question just read them (for ordinals); empty
    #: when the question listed nothing.
    listed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in PENDING_KINDS:
            raise ValueError(f"unknown pending kind {self.kind!r}")
        if not isinstance(self.listed, tuple):
            raise ValueError("listed must be a tuple")
        if any(not isinstance(v, str) or not v for v in self.listed):
            raise ValueError("listed values must be non-empty strings")
        if len(set(self.listed)) != len(self.listed):
            raise ValueError("listed values must be distinct")


@dataclass(frozen=True, slots=True)
class ResolveContext:
    """What :func:`resolve` knows beyond the slot and the input."""

    question_text: str                        # as the caller heard it
    labels: Mapping[str, str]
    commands: Mapping[str, Sequence[str]]
    known_options: tuple[OptionView, ...]     # value, spoken name, aliases in lang
    caller_options: tuple[str, ...]           # subset (cards on file); = known when no subset
    collected: tuple[CollectedSlot, ...]
    pending: Pending | None
    budget_ms: int


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of resolving one answer. Grounding and validation have already run:
    ``value`` is canonical, and ``candidates`` are caller options."""

    match: Match
    source: Source
    value: str | None = None                  # exact / inferred / correction (canonical, validated)
    candidates: tuple[str, ...] = ()          # ambiguous: >= 2 caller options
    target: str | None = None                 # correction: collected slot name
    reply: Literal["yes", "no"] | None = None # answer to a yes/no question that names an option
    reply_deterministic: bool = False         # keypad or lists
    model: ModelOutcome = "not_called"
    say: str | None = None                    # generative only, unchecked here
    utterance_for_ai: str | None = None       # sensitive slot: [<slot name>] placeholder
    redacted_spans: tuple[tuple[int, int], ...] = ()   # character spans of the speech replaced

    def __post_init__(self) -> None:
        if self.match not in MATCHES:
            raise ValueError(f"unknown match {self.match!r}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        if self.model not in MODEL_OUTCOMES:
            raise ValueError(f"unknown model outcome {self.model!r}")
        if self.reply is not None and self.reply not in ("yes", "no"):
            raise ValueError(f"unknown reply {self.reply!r}")
