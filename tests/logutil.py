"""Read structured log records the way the JSON formatter shows them.

The audit and security helpers log through :mod:`owlbank.obs`, so a record's fields are
attributes (``record.audit``, ``record.event``, ``record.reason``...), not a JSON message.
"""

from __future__ import annotations

import logging
from typing import Any

from owlbank.obs import _STD_ATTRS

_SKIP = _STD_ATTRS | {"_owlbank_scrubbed"}


def fields(record: logging.LogRecord) -> dict[str, Any]:
    """The extra fields of a record (everything that is not a standard LogRecord attribute)."""
    return {k: v for k, v in record.__dict__.items() if k not in _SKIP and not k.startswith("_")}


def lines(records: list[logging.LogRecord], logger: str) -> list[dict[str, Any]]:
    """Fields of every record from ``logger``, with ``event`` always present."""
    out = []
    for r in records:
        if r.name == logger:
            f = fields(r)
            f.setdefault("event", r.getMessage())
            out.append(f)
    return out


def text_of(records: list[logging.LogRecord]) -> str:
    """Message plus every field of every record, for "this value is nowhere in the logs" checks."""
    return "\n".join(f"{r.getMessage()} {fields(r)}" for r in records)


def with_context(caplog: Any) -> None:
    """Make ``caplog`` records carry ``call_sid``/``component`` like the JSON handler's filter does.

    Call it inside the test body (the capture handler of the "call" phase is only the current one there)."""
    from owlbank.obs import ContextFilter

    caplog.handler.addFilter(ContextFilter())
