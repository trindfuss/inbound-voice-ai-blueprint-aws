"""Policy documents for tests that pin latency masking on or off explicitly.

``data/policy.json`` masks ``list_recent_transactions``, ``change_address`` and ``get_balance``
and arms the timed acknowledgement at 1500 ms. Tests written before latency masking that assert the exact
words the caller hears, or the timing of a hand-off, use :func:`unmasked_policy_doc` so the
policy they run with is stated rather than implied.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[2] / "data"


def _policy_json() -> dict[str, Any]:
    return json.loads((DATA / "policy.json").read_text())


def unmasked_policy_doc(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    """``doc`` (default ``data/policy.json``) with no masking phrase and the timer off."""
    out = copy.deepcopy(doc) if doc is not None else _policy_json()
    for row in out["tools"].values():
        if isinstance(row, dict):
            row.pop("mask", None)
    out["masking"] = {**out.get("masking", {}), "after_ms": 0}
    return out


def masked_policy_doc(doc: dict[str, Any] | None = None, *, masks: dict[str, str | None] | None = None,
                      **masking: Any) -> dict[str, Any]:
    """``doc`` (default ``data/policy.json``) with ``masking`` fields overridden and, per tool in
    ``masks``, the phrase key set (``None`` removes the tool's phrase)."""
    out = copy.deepcopy(doc) if doc is not None else _policy_json()
    for tool, key in (masks or {}).items():
        if key is None:
            out["tools"][tool].pop("mask", None)
        else:
            out["tools"][tool]["mask"] = {"phrase_key": key}
    out["masking"] = {**out.get("masking", {}), **masking}
    return out
