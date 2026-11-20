"""Slot resolution: the engine, the resolver, grounding, the dialog policy, modes and the generative
check."""

import importlib
from typing import Any

from . import dialog, grounding, resolver
from .commands import commands_from_catalog, match_command
from .dialog import Action, Turn, decide
from .engine import (
    YES_NO_PENDING,
    hints,
    make_context,
    option_labels,
    resolve,
    slot_options,
    utterance_for_ai,
)
from .normalize import (
    LocatedToken,
    NumberRun,
    normalize,
    normalize_located,
    number_runs,
    spoken_digits,
)
from .resolver import (
    DEFAULT_MODEL_ID,
    BedrockResolver,
    FakeResolver,
    Resolver,
    ResolverOutput,
    ResolverRequest,
    client_for_budget,
)
from .types import (
    CollectedSlot,
    OptionView,
    Pending,
    Resolution,
    ResolveContext,
    SlotDecl,
    SpeechParse,
)
from .validators import (
    EARLY_OUTCOMES,
    early_digits,
    early_phone,
    parse_dtmf,
    parse_speech,
    parse_speech_located,
    validate,
)

#: Imported on first use: ``modes`` needs ``owlbank.slot_policy`` at import time, and
#: ``slot_policy`` imports this package, so they can't load with it (PEP 562).
_LAZY = ("generative", "modes")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_MODEL_ID",
    "EARLY_OUTCOMES",
    "YES_NO_PENDING",
    "Action",
    "BedrockResolver",
    "CollectedSlot",
    "FakeResolver",
    "LocatedToken",
    "NumberRun",
    "OptionView",
    "Pending",
    "Resolution",
    "ResolveContext",
    "Resolver",
    "ResolverOutput",
    "ResolverRequest",
    "SlotDecl",
    "SpeechParse",
    "Turn",
    "client_for_budget",
    "commands_from_catalog",
    "decide",
    "dialog",
    "early_digits",
    "early_phone",
    "generative",
    "grounding",
    "hints",
    "make_context",
    "match_command",
    "modes",
    "normalize",
    "normalize_located",
    "number_runs",
    "option_labels",
    "parse_dtmf",
    "parse_speech",
    "parse_speech_located",
    "resolve",
    "resolver",
    "slot_options",
    "spoken_digits",
    "utterance_for_ai",
    "validate",
]
