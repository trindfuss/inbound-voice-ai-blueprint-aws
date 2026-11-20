"""Universal commands (repeat, agent, cancel): whole-utterance matches only.

Phrases come from the prompt catalog's ``command.*`` entries, where ``text`` is a
``|``-separated list of phrases per language. A command matches only when the
*whole* normalized utterance equals one normalized phrase, so "my agent at the
branch told me to call" is not an escalation request (it goes to the resolver).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from .normalize import normalize

#: ``decline`` is never a catalog phrase: only the resolver or ``#`` alone
#: on a declinable slot gives it.
CommandKind = Literal["repeat", "escalate", "cancel", "decline"]

#: Catalog key -> SlotResult kind.
COMMAND_KEYS: dict[str, CommandKind] = {
    "command.repeat": "repeat",
    "command.agent": "escalate",
    "command.cancel": "cancel",
}


def _split(phrases: str | Sequence[str]) -> list[str]:
    if isinstance(phrases, str):
        return [p for p in phrases.split("|") if p.strip()]
    return [p for p in phrases if p and p.strip()]


def match_command(
    utterance: str | None, lang: str, catalog_commands: Mapping[str, str | Sequence[str]]
) -> CommandKind | None:
    """Return the command kind if the whole utterance is a command phrase.

    ``catalog_commands`` maps ``command.repeat`` / ``command.agent`` /
    ``command.cancel`` (or the short names ``repeat`` / ``agent`` / ``cancel``) to
    the catalog text for the call's language.
    """
    norm = normalize(utterance, lang)
    if not norm:
        return None
    for key, phrases in catalog_commands.items():
        full = key if key.startswith("command.") else f"command.{key}"
        kind = COMMAND_KEYS.get(full)
        if kind is None:
            continue
        if any(normalize(p, lang) == norm for p in _split(phrases)):
            return kind
    return None


def commands_from_catalog(entries: Iterable[Mapping[str, object]], lang: str) -> dict[str, str]:
    """Pick the ``command.*`` texts for ``lang`` out of catalog entries
    (``data/prompt_catalog.json`` ``entries`` shape: ``{key, lang, text, ...}``)."""
    return {
        str(e["key"]): str(e["text"])
        for e in entries
        if str(e.get("key", "")) in COMMAND_KEYS and e.get("lang") == lang
    }
