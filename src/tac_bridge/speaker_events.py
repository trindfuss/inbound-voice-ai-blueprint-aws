"""ConversationRelay speaker events: recognising the frames, nothing else.

The front door subscribes each session with ``events="speaker-events"``. Twilio's
ConversationRelay noun reference says this "will notify your application about ``agentSpeaking``
and ``clientSpeaking`` events" but doesn't show the message. The shape assumed here, to be
confirmed on a real call, is::

    {"type": "info", "name": "clientSpeaking" | "agentSpeaking", "value": "on" | "off"}

The parser is tolerant, and this module is the one place to change if the real shape differs:

* :func:`is_speaker_frame`: a frame this module owns: ``type`` ``"info"``, or ``type`` being
  the event name itself (``{"type": "clientSpeaking", "value": "on"}``). The app swallows these
  (TAC gets ``{"type": "ignored"}``), so none can reach TAC as caller input.
* :func:`parse_speaker_event`: the event, from ``name`` (or the ``type`` itself) and ``value``
  (``"on"``/``"off"`` in any case, or a JSON boolean). Anything else is ``None``: the app logs it
  as unrecognised (``type`` and key names only) and the silence watch keeps estimate timing.

No value is ever logged from an unrecognised frame, since it could hold text.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tac_bridge.silence import AGENT, CALLER

#: ConversationRelay event name -> who it is about.
SPEAKER_NAMES: Mapping[str, str] = {"clientSpeaking": CALLER, "agentSpeaking": AGENT}
INFO_TYPE = "info"
_ON = frozenset({"on"})
_OFF = frozenset({"off"})
#: Bounds on what an unrecognised frame's log line can carry (key names only).
MAX_KEYS = 12
MAX_KEY_CHARS = 32


@dataclass(frozen=True, slots=True)
class SpeakerEvent:
    """A recognised speaker event: the ConversationRelay ``name``, ``who`` (``caller`` or
    ``agent``) and whether they are now speaking."""

    name: str
    who: str
    speaking: bool

    @property
    def value(self) -> str:
        return "on" if self.speaking else "off"


def is_speaker_frame(msg: Mapping[str, Any]) -> bool:
    """Whether the frame belongs to this module (see the module docstring)."""
    kind = msg.get("type")
    return kind == INFO_TYPE or (isinstance(kind, str) and kind in SPEAKER_NAMES)


def parse_speaker_event(msg: Mapping[str, Any]) -> SpeakerEvent | None:
    """The speaker event in ``msg``, or ``None`` for any shape not recognised."""
    kind = msg.get("type")
    name = msg.get("name") if kind == INFO_TYPE else kind
    if not isinstance(name, str) or name not in SPEAKER_NAMES:
        return None
    value = msg.get("value")
    if isinstance(value, bool):
        speaking = value
    elif isinstance(value, str) and value.strip().lower() in _ON:
        speaking = True
    elif isinstance(value, str) and value.strip().lower() in _OFF:
        speaking = False
    else:
        return None
    return SpeakerEvent(name=name, who=SPEAKER_NAMES[name], speaking=speaking)


def shape_of(msg: Mapping[str, Any]) -> dict[str, Any]:
    """Log fields for an unrecognised frame: its ``type`` (when it is one this module owns, so a
    known word) and its key names, bounded. Never a value."""
    kind = msg.get("type")
    keys = sorted(str(k)[:MAX_KEY_CHARS] for k in list(msg.keys())[:MAX_KEYS])
    return {"type": kind if is_speaker_frame(msg) else "other", "keys": keys}


__all__ = ["INFO_TYPE", "SPEAKER_NAMES", "SpeakerEvent", "is_speaker_frame", "parse_speaker_event",
           "shape_of"]
