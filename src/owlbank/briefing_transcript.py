"""The scrubbed tail of the call's conversation, for the escalation briefing.

:func:`fetch` reads the Orchestrator conversation's communications and participants, takes the last
turns, scrubs them as the recap writer scrubs its transcript (:func:`owlbank.recap.scrub`: factor
values removed, digit runs masked, the PII guard, fail closed), masks every number that scrub keeps
(small numbers, amounts, dates, times: the model gets no digit) and returns the end of the result.
It never raises and returns ``None`` for anything short of a scrubbed transcript: no conversation,
a failed or late read, no text, a scrub that fails closed. It never logs the text.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Protocol

from owlbank.obs import log_caught
from owlbank.pii import PiiGuard
from owlbank.recap import scrub, transcript_lines

log = logging.getLogger(__name__)

MAX_TURNS = 12
#: Whole lines are dropped from the start until the text to scrub is at most this long.
MAX_SCRUB_CHARS = 3000
MAX_CHARS = 1500
DIGITS_MASK = "[number]"
#: Any number still in the scrubbed text (``recap.scrub`` keeps small numbers, money amounts, ISO dates
#: and clock times): a digit run with the separators numbers carry. The model gets no digit at all.
_NUMBER = re.compile(r"\d(?:[\d.,:/\-]*\d)?")

#: Workers for the bounded input reads. A call past its deadline keeps its worker until the client's
#: own timeouts end it (the PII client's are 1 s connect and 2 s read).
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="briefing-input")


def gather(tasks: Mapping[str, Callable[[], Any]], seconds: float) -> dict[str, Any]:
    """Run ``tasks`` at once and wait at most ``seconds`` for all of them: the results of those that
    finished (``None`` for one that raised or was late; the late result is discarded). Never raises.
    Each task runs in a copy of the caller's context (the call's telemetry)."""
    futures = {name: _POOL.submit(contextvars.copy_context().run, fn) for name, fn in tasks.items()}
    done, _ = concurrent.futures.wait(futures.values(), timeout=max(0.0, seconds))
    out: dict[str, Any] = {}
    for name, fut in futures.items():
        out[name] = None
        if fut in done:
            try:
                out[name] = fut.result()
            except Exception:  # noqa: BLE001 - the task logs its own failure; no message here
                out[name] = None
        else:
            fut.cancel()
    return out


class ConversationReader(Protocol):
    def communications(self, conversation_id: str, *,
                       more: Callable[[], bool] | None = None) -> Sequence[Mapping[str, Any]] | None: ...

    def participants(self, conversation_id: str, *,
                     more: Callable[[], bool] | None = None) -> Sequence[Mapping[str, Any]] | None: ...


def _tail(lines: Sequence[str], limit: int) -> list[str]:
    """The last whole lines whose joined length is at most ``limit`` (at least none)."""
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        extra = len(line) + (1 if kept else 0)
        if total + extra > limit:
            break
        kept.append(line)
        total += extra
    kept.reverse()
    return kept


def fetch(reader: ConversationReader | None, conversation_id: str | None, *, call_sid: str,
          factor_values: Collection[str], pii: PiiGuard, language: str, deadline: float,
          now: Callable[[], float] = time.monotonic) -> str | None:
    """The scrubbed tail of the conversation, read and scrubbed before ``deadline`` (on the ``now``
    clock), else ``None``. The reads stop before a page that would start after the deadline; the
    caller bounds each request with :func:`owlbank.humanleg_client.budget`."""
    if reader is None or not conversation_id:
        return None
    try:
        def more() -> bool:
            return now() < deadline

        communications = reader.communications(conversation_id, more=more)
        if communications is None:
            return None
        participants = reader.participants(conversation_id, more=more)
        if participants is None:
            return None
        lines = transcript_lines(communications, participants)[-MAX_TURNS:]
        if not lines or now() >= deadline:
            return None
        kept = _tail(lines, MAX_SCRUB_CHARS)
        if not kept:
            return None
        scrubbed = scrub("\n".join(kept), factor_values=factor_values, pii=pii, language=language)
        if scrubbed is None or now() >= deadline:
            return None
        scrubbed = _NUMBER.sub(DIGITS_MASK, scrubbed)
        # Cut after the scrub, on a line boundary: a cut only ever removes scrubbed text.
        out = "\n".join(_tail([ln for ln in scrubbed.split("\n") if ln.strip()], MAX_CHARS))
        return out or None
    except Exception as exc:  # noqa: BLE001 - the briefing never fails the call
        log_caught(log, "briefing_transcript_failed", exc, call_sid=call_sid,
                   operation="transcript", decision="without_transcript")
        return None
