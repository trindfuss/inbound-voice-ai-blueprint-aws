"""A permissive TaskRouter reader for tests of the context API that are not about authorisation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

TASK_SID = "WT" + "4" * 32


class PermissiveTaskRouter:
    """The task carries this call's SID and journal ref; callers also send ``roles=supervisor``."""

    def __init__(self, call_sid: str, journal_ref: str) -> None:
        self.call_sid, self.journal_ref = call_sid, journal_ref

    def task_attributes(self, task_sid: str, *, timeout_s=None) -> Mapping[str, str]:
        return {"call_sid": self.call_sid, "journal_ref": self.journal_ref}

    def reservations(self, task_sid: str, *, timeout_s=None) -> Sequence[Mapping[str, str]]:
        return []
