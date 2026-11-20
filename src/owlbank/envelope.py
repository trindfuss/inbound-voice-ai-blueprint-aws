"""The result envelope every deterministic tool returns to the agent.

``spoken`` is exactly what the caller heard, in order; ``data`` is the facts behind
it; ``next`` tells the agent what to do now.

Validation (on construction and on :meth:`ResultEnvelope.from_json`):

* ``status`` and ``next`` are from :data:`STATUSES` and :data:`NEXT_ACTIONS`.
* ``next == "await_caller"`` needs a non-empty ``spoken`` whose last item's template
  is a follow-up prompt (:data:`FOLLOW_UP_TEMPLATES`): the flow already asked it.
* ``next == "handle_utterance"`` needs a non-empty ``callerUtterance``.
* ``data`` must be JSON-serializable.

JSON uses the handover's camelCase field names. Optional fields that are ``None``
(``forTool``, ``callerUtterance``, ``completed``) are left out.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

#: ``CANCELLED_BY_CALLER``: a ``deterministic``-confirm tool the caller didn't
#: confirm (no, no answer, cancel); the tool did not run.
CANCELLED_BY_CALLER = "CANCELLED_BY_CALLER"
STATUSES: frozenset[str] = frozenset(
    {"completed", "not_found", "failed", "declined", "interrupted", "verified", "not_verified",
     CANCELLED_BY_CALLER}
)
NEXT_ACTIONS: frozenset[str] = frozenset(
    {"await_caller", "retry_original_tool", "offer_alternative", "handle_utterance"}
)
#: Catalog keys that ask the caller a follow-up question, so the agent can stay silent.
FOLLOW_UP_TEMPLATES: frozenset[str] = frozenset(
    {"common.anything_else", "escalate.unavailable", "escalate.after_hours"}
)

_FIELDS = ("status", "tool", "spoken", "data", "next", "forTool", "callerUtterance", "completed")


class EnvelopeError(ValueError):
    """The envelope breaks a rule."""


@dataclass(frozen=True, slots=True)
class Spoken:
    """One prompt the caller heard: its catalog key and the rendered text."""

    template: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.template, str) or not self.template:
            raise EnvelopeError("spoken.template must be a non-empty string")
        if not isinstance(self.text, str):
            raise EnvelopeError("spoken.text must be a string")


def _check_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EnvelopeError(f"{path}: non-finite number")
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            _check_json_value(v, f"{path}[{i}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise EnvelopeError(f"{path}: keys must be strings")
            _check_json_value(v, f"{path}.{k}")
        return
    raise EnvelopeError(f"{path}: {type(value).__name__} is not JSON")


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    status: str
    tool: str
    spoken: list[Spoken] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    next: str = "await_caller"
    forTool: str | None = None
    callerUtterance: str | None = None
    completed: bool | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise EnvelopeError(f"unknown status {self.status!r}")
        if self.next not in NEXT_ACTIONS:
            raise EnvelopeError(f"unknown next {self.next!r}")
        if not isinstance(self.tool, str) or not self.tool:
            raise EnvelopeError("tool must be a non-empty string")
        if not isinstance(self.spoken, list) or not all(isinstance(s, Spoken) for s in self.spoken):
            raise EnvelopeError("spoken must be a list of Spoken")
        if not isinstance(self.data, dict):
            raise EnvelopeError("data must be an object")
        _check_json_value(self.data, "data")
        if self.forTool is not None and (not isinstance(self.forTool, str) or not self.forTool):
            raise EnvelopeError("forTool must be a non-empty string")
        if self.callerUtterance is not None and not isinstance(self.callerUtterance, str):
            raise EnvelopeError("callerUtterance must be a string")
        if self.completed is not None and not isinstance(self.completed, bool):
            raise EnvelopeError("completed must be a bool")
        if self.next == "await_caller":
            if not self.spoken:
                raise EnvelopeError("await_caller needs a non-empty spoken list")
            if self.spoken[-1].template not in FOLLOW_UP_TEMPLATES:
                raise EnvelopeError(
                    f"await_caller must end with a follow-up prompt, not {self.spoken[-1].template!r}"
                )
        if self.next == "handle_utterance" and not self.callerUtterance:
            raise EnvelopeError("handle_utterance needs callerUtterance")

    # -- serialization

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "tool": self.tool,
            "spoken": [{"template": s.template, "text": s.text} for s in self.spoken],
            "data": json.loads(json.dumps(self.data)),
            "next": self.next,
        }
        for name in ("forTool", "callerUtterance", "completed"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, obj: Any) -> ResultEnvelope:
        if not isinstance(obj, dict):
            raise EnvelopeError("envelope must be an object")
        unknown = set(obj) - set(_FIELDS)
        if unknown:
            raise EnvelopeError(f"unknown envelope fields {sorted(unknown)}")
        for required in ("status", "tool", "next"):
            if required not in obj:
                raise EnvelopeError(f"envelope missing {required!r}")
        raw_spoken = obj.get("spoken", [])
        if not isinstance(raw_spoken, list):
            raise EnvelopeError("spoken must be a list")
        spoken = []
        for s in raw_spoken:
            if not isinstance(s, dict) or set(s) != {"template", "text"}:
                raise EnvelopeError("each spoken item needs exactly template and text")
            spoken.append(Spoken(s["template"], s["text"]))
        return cls(
            status=obj["status"],
            tool=obj["tool"],
            spoken=spoken,
            data=obj.get("data", {}) if obj.get("data") is not None else {},
            next=obj["next"],
            forTool=obj.get("forTool"),
            callerUtterance=obj.get("callerUtterance"),
            completed=obj.get("completed"),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> ResultEnvelope:
        try:
            obj = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise EnvelopeError(f"envelope is not JSON: {exc}") from None
        return cls.from_dict(obj)
