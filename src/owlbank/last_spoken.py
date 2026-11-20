"""The last message spoken to the caller at a session's start, sealed for the bridge.

After caller silence the bridge repeats the last message it knows the caller heard. For
a turn that's the text the bridge sent itself; for a ConversationRelay session's welcome greeting
it's text only the front door knows, so the front door stores it in call state
(:data:`owlbank.callstate.F_LAST_SPOKEN`) when it connects. The greeting can carry a balance or a
confirmation number, so it is sealed like the call token (the same KMS key, the ``CallSid``
as encryption context): a record copied to another call doesn't unseal. It expires after
:data:`TTL_S` and is used for repeats only.

Stored shape: ``{"sealed": <Sealer.seal output>, "expires_at": <epoch seconds>}``.

The text is never logged, here or by callers: failures log the reason only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from owlbank.obs import emit, log_caught
from owlbank.token_seal import Sealer, UnsealError

__all__ = ["TTL_S", "seal", "unseal"]

log = logging.getLogger(__name__)

#: How long a sealed greeting may be repeated (10 minutes).
TTL_S = 600


def seal(sealer: Sealer, text: str, *, call_sid: str, now: float) -> dict[str, Any] | None:
    """The call-state record for ``text``, or ``None`` when there is nothing to repeat (empty
    text) or sealing failed (logged without the text; the caller then gets no repeat, and the
    silence action still applies,– error handling)."""
    if not text or not text.strip():
        return None
    try:
        sealed = sealer.seal(text, call_sid=call_sid)
    except Exception as exc:
        log_caught(log, "last_spoken could not be sealed", exc, level=logging.ERROR,
                   operation="seal_last_spoken", decision="store_none_no_repeat")
        return None
    return {"sealed": sealed, "expires_at": int(now) + TTL_S}


def unseal(sealer: Sealer, record: Any, *, call_sid: str, now: float) -> str | None:
    """The text of a :func:`seal` record for this call, or ``None`` if the record is missing,
    malformed, expired, or doesn't unseal for ``call_sid``. Never raises."""
    if record is None:
        return None
    if not isinstance(record, Mapping):
        emit(log, logging.WARNING, "last_spoken_unusable", reason="bad_shape", decision="no_repeat")
        return None
    sealed, expires_at = record.get("sealed"), record.get("expires_at")
    if (not isinstance(sealed, str) or not sealed or isinstance(expires_at, bool)
            or not isinstance(expires_at, (int, float)) or not math.isfinite(expires_at)):
        emit(log, logging.WARNING, "last_spoken_unusable", reason="bad_shape", decision="no_repeat")
        return None
    if now >= expires_at:
        emit(log, logging.INFO, "last_spoken_unusable", reason="expired", decision="no_repeat")
        return None
    try:
        return sealer.unseal(sealed, call_sid=call_sid)
    except UnsealError:
        emit(log, logging.WARNING, "last_spoken_unusable", reason="unseal_failed", decision="no_repeat")
        return None
    except Exception as exc:
        log_caught(log, "last_spoken could not be unsealed", exc, operation="unseal_last_spoken",
                   decision="no_repeat")
        return None
