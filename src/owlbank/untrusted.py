"""Untrusted stored content in model inputs.

Memory recall, account nicknames and transaction descriptions are text that someone
other than the bank wrote (a caller on another channel, a payee's memo). They reach
the model only inside a delimited block::

    <untrusted_data source="memory">...</untrusted_data>

and the catalog's ``agent.system.<model>`` tells the model that such a block is data, never
instructions.

The content is escaped so it can't close the block or open a new one: ``&``, ``<`` and
``>`` become ``&amp;``, ``&lt;`` and ``&gt;``. After that no ``<untrusted_data`` or
``</untrusted_data`` tag (in any case or spacing) can appear inside the content, so the
only delimiter tags in the output are the two this module writes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

TAG = "untrusted_data"
OPEN_RE = re.compile(r"<\s*untrusted_data\b", re.IGNORECASE)
CLOSE_RE = re.compile(r"<\s*/\s*untrusted_data\s*>", re.IGNORECASE)
_SOURCE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

#: Bank result fields that carry third-party text, and the source each is labelled with.
RESULT_FIELDS: Mapping[str, str] = {
    "description": "transaction_description",
    "memo": "transaction_description",
    "nickname": "account_nickname",
}


def neutralize(text: str) -> str:
    """Escape ``text`` so it can't contain a delimiter tag (or any tag)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(source: str, text: Any) -> str:
    """``text`` (as a string) inside one ``untrusted_data`` block labelled ``source``."""
    if not _SOURCE.match(source):
        raise ValueError("source must be a short lowercase identifier")
    return f'<{TAG} source="{source}">{neutralize(str(text))}</{TAG}>'


def wrap_result(value: Any, fields: Mapping[str, str] = RESULT_FIELDS) -> Any:
    """A copy of a tool result with every string in a third-party text field wrapped."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k in fields and isinstance(v, str):
                out[k] = wrap(fields[k], v)
            else:
                out[k] = wrap_result(v, fields)
        return out
    if isinstance(value, list):
        return [wrap_result(v, fields) for v in value]
    return value


__all__ = ["CLOSE_RE", "OPEN_RE", "RESULT_FIELDS", "TAG", "neutralize", "wrap", "wrap_result"]
