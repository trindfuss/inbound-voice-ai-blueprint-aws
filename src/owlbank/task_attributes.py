"""Flex/TaskRouter task attributes for an escalation.

Identity, level, language and customer fields come only from the call state and
the store; the journal summary supplies what tools ran. The model contributes
only ``reason`` (mapped to a known value, else ``unable_to_resolve``) and
``summary`` (control characters stripped, registered factor values redacted,
capped at 1,000 characters, and labelled ``ai_summary``). With a ``pii`` guard
 the summary is also PII-masked; if the check is unavailable it becomes
``withheld_summary`` (``escalation.summary_withheld``) and the escalation goes ahead. ``test`` comes from an
explicit server parameter, never from the model. The result passes the
redaction guard, so it never contains factor values or tokens.
"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from typing import Any

from owlbank.callstate import CallState
from owlbank.obs import log_caught
from owlbank.pii import SURFACE_ESCALATION_SUMMARY, WITHHELD_TURN, PiiGuard
from owlbank.segments import assert_clean, redact_text
from owlbank.store import Store
from owlbank.tools_contract import DEFAULT_ESCALATION_REASONS, FALLBACK_REASON, SUMMARY_MAX_LEN

log = logging.getLogger(__name__)

_TO_SPACE = {"\t", "\n", "\r", "\v", "\f", " ", " ", "\x85"}


def normalize_reason(reason: Any, reasons: Collection[str] = DEFAULT_ESCALATION_REASONS) -> tuple[str, bool]:
    """``(reason, known)``; an unknown or non-string reason becomes ``unable_to_resolve``."""
    if isinstance(reason, str) and reason in reasons:
        return reason, True
    return FALLBACK_REASON, False


def clean_summary(call_sid: str, summary: Any, *, max_len: int = SUMMARY_MAX_LEN) -> str:
    """Strip control/format characters, redact factor values, cap at ``max_len``."""
    text = summary if isinstance(summary, str) else ""
    out = []
    for ch in text:
        if ch in _TO_SPACE or ch.isspace():  # includes \x1c-\x1f separators, so words stay apart
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn"):
            continue
        else:
            out.append(ch)
    text = " ".join("".join(out).split())
    text = redact_text(call_sid, text)
    return text[:max_len].rstrip()


def _summary(call_state: CallState, summary: Any, pii: PiiGuard | None, withheld: str) -> str:
    return _prepared(call_state.call_sid, summary, pii, call_state.language, withheld)


def prepare_summary(call_sid: str, summary: Any, pii: PiiGuard | None, language: str,
                    withheld: str, *, on_failure: str | None = None) -> str | None:
    """The escalation summary as the task's ``ai_summary`` carries it, or ``None`` when there is
    none to use: empty after cleaning, or the withheld text (the PII check was unavailable). One
    preparation for both the Flex task and the briefing."""
    text = _prepared(call_sid, summary, pii, language, withheld, on_failure=on_failure)
    return text if text and text != withheld else None


def _prepared(call_sid: str, summary: Any, pii: PiiGuard | None, language: str, withheld: str,
              *, on_failure: str | None = None) -> str:
    text = clean_summary(call_sid, summary)
    if pii is None:
        return text
    try:
        text = pii.text(SURFACE_ESCALATION_SUMMARY, text, language=language, withheld=withheld,
                        on_failure=on_failure)
    except Exception as exc:  # PiiGuard doesn't raise; belt and braces
        log_caught(log, "summary_pii_check_failed", exc, operation="escalation_summary",
                   decision="fallback_withheld_text")
        text = withheld
    return text[:SUMMARY_MAX_LEN].rstrip()


def _iso(now: float) -> str:
    return datetime.fromtimestamp(int(now), tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_task_attributes(
    call_state: CallState,
    segment_summary: Mapping[str, Any],
    reason: Any,
    summary: Any,
    *,
    now: float,
    reasons: Collection[str] = DEFAULT_ESCALATION_REASONS,
    conversation_id: str | None = None,
    profile_id: str | None = None,
    journal_ref: str | None = None,
    test: bool = False,
    store: Store | None = None,
    pii: PiiGuard | None = None,
    withheld_summary: str = WITHHELD_TURN,
) -> dict[str, Any]:
    """Task attributes for ``<Enqueue><Task>``.

    ``segment_summary`` is :meth:`owlbank.segments.SegmentLog.summary_for_resume`
    output. ``store`` (optional) supplies the verified customer's ``name``.
    """
    if not isinstance(test, bool):
        raise TypeError("test must be a bool")
    sid = call_state.call_sid
    escalation_reason, _ = normalize_reason(reason, reasons)

    customer_id = call_state.verified_customer_id
    name = None
    if store is not None and customer_id:
        customer = store.get("customers", {"customer_id": customer_id})
        if customer is not None and isinstance(customer.get("name"), str):
            name = customer["name"]
        elif customer is not None:  # fixtures carry first_name/last_name (handover example: full name)
            parts = [customer.get(k) for k in ("first_name", "last_name")]
            name = " ".join(p for p in parts if isinstance(p, str) and p) or None

    tools = [t for t in segment_summary.get("toolsCalled", []) if isinstance(t, str)]
    last_tool = call_state.step_up_target or (
        call_state.pending_tool.name if call_state.pending_tool else None
    ) or (tools[-1] if tools else None)

    attrs: dict[str, Any] = {
        "type": "inbound",
        "name": name,
        "from": call_state.ani,
        "call_sid": sid,
        "language": call_state.language,
        "escalation_reason": escalation_reason,
        "ai_summary": _summary(call_state, summary, pii, withheld_summary),
        "auth_level": call_state.level,
        # Keyed-in phone login: how identity was claimed, never the keyed number.
        "identity_source": call_state.identity_source,
        "customer_id": customer_id,
        "factors_verified": list(call_state.factors_verified),
        "last_tool_attempted": last_tool,
        "tools_run": tools,
        "conversation_id": conversation_id,
        "profile_id": profile_id,
        "journal_ref": journal_ref,
        "escalated_at": _iso(now),
        "test": test,
    }
    assert_clean(sid, attrs)
    return attrs
