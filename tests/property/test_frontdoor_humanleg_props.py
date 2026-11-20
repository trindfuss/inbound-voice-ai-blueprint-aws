"""Human-leg properties over generated escalations and queue waits.

- Under any combination of escalation dependency failures (the
  conversation lookup answers, finds none, or raises; the ``handoff`` journal write fails or not),
  the ``<Enqueue>``, its ``<Task>`` and the task attributes equal the healthy case, apart from
  ``conversation_id``, which is the looked-up id when one was found and ``null`` otherwise; with
  the id known from the hand-off, nothing differs at all.
- Under any escalation setup, the ``<Enqueue>`` carries an ``action``
  and no ``waitUrl`` (Twilio plays its default hold music), and its ``<Task>`` carries no
  timeout; the ``action`` answer depends on ``QueueResult`` only, never on the wait, and has a ``<Hangup>`` only when the
  caller left the queue or the enqueue failed.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.app import (
    QUEUE_FAILED_RESULTS,
    QUEUE_LEFT_RESULTS,
)
from owlbank import callstate
from owlbank.catalog import PromptCatalog
from tests.component.test_frontdoor_humanleg import CONV, FakeHumanLeg, escalate
from tests.component.test_frontdoor_support import (
    DATA, ENQUEUE_ACTION_URL, MARIA, THOMAS, build, sid,
)

PROPS = settings(max_examples=60, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
CATALOG_DOC = json.loads((DATA / "prompt_catalog.json").read_text())
CATALOG = PromptCatalog.from_dict(CATALOG_DOC)
LANGS = tuple(sorted(CATALOG_DOC["languages"]))
KNOWN_RESULTS = ("leave", "bridged", "hangup", "redirected", "bridging-in-process",
                 "redirected-from-bridged", "queue-full", "error", "system-error")


def _enqueue(root: ET.Element) -> tuple[dict, dict, dict]:
    enq = root.find("Enqueue")
    return dict(enq.attrib), dict(enq.find("Task").attrib), json.loads(enq.find("Task").text)


def _escalated(lookup: str, journal_fails: bool, known: bool) -> ET.Element:
    h = build(human_leg=FakeHumanLeg(None) if lookup == "none" else FakeHumanLeg(fail=lookup == "raises"))
    if journal_fails:
        append = h.app.journal.append

        def failing(call_sid, event):
            if event.get("type") == "handoff" and event.get("to") == "human":
                raise RuntimeError("journal down")
            return append(call_sid, event)

        h.app.journal.append = failing
    return escalate(h, sid(1), **({"conversationId": CONV} if known else {}))


HEALTHY = {known: _enqueue(_escalated("found", False, known)) for known in (False, True)}


@PROPS
@given(lookup=st.sampled_from(["found", "none", "raises"]), journal_fails=st.booleans(),
       known=st.booleans())
def test_failures_never_change_the_enqueue(lookup, journal_fails, known):
    enq, task, attrs = _enqueue(_escalated(lookup, journal_fails, known))
    h_enq, h_task, h_attrs = HEALTHY[known]
    assert enq == h_enq and task == h_task
    assert "timeout" not in task  # No <Task> timeout
    assert "waitUrl" not in enq and enq["action"] == ENQUEUE_ACTION_URL
    expected = CONV if known or lookup == "found" else None
    assert attrs["conversation_id"] == expected
    assert {k: v for k, v in attrs.items() if k != "conversation_id"} == \
        {k: v for k, v in h_attrs.items() if k != "conversation_id"}


# --------------------------------------------------------------------------- action


def _queued(lang: str):
    """A harness with one call in the queue, in ``lang``."""
    h = build(human_leg=FakeHumanLeg())
    h.start(sid(3), THOMAS if lang == "en-US" else MARIA, lang_digit="" if lang == "en-US" else "2")
    callstate.mutate(h.store, sid(3), lambda s: {"escalation": {"reason": "unable_to_resolve"}})
    return h


QUEUED = {lang: _queued(lang) for lang in LANGS}


@PROPS
@given(lookup=st.sampled_from(["found", "none", "raises"]), journal_fails=st.booleans(),
       known=st.booleans())
def test_escalation_has_an_action_and_no_wait_url_or_task_timeout(lookup, journal_fails, known):
    root = _escalated(lookup, journal_fails, known)
    enq, task, _ = _enqueue(root)
    assert enq["action"] == ENQUEUE_ACTION_URL and "waitUrl" not in enq
    assert "timeout" not in task and "timeout" not in enq
    assert root.find(".//Enqueue/Task") is not None and root.find(".//Hangup") is None


@PROPS
@given(result=st.one_of(st.sampled_from(KNOWN_RESULTS), st.text(max_size=20), st.none()),
       wait_a=st.integers(0, 10 ** 6), wait_b=st.integers(0, 10 ** 6))
def test_action_depends_on_the_result_only(result, wait_a, wait_b):
    h = QUEUED["en-US"]

    def answer(waited: int) -> bytes:
        params = {"QueueTime": str(waited), **({"QueueResult": result} if result is not None else {})}
        return ET.tostring(h.xml("enqueue-action", sid(3), follow=False, **params))

    a = answer(wait_a)
    assert a == answer(wait_b)
    hangs_up = ET.fromstring(a).find("Hangup") is not None
    norm = (result or "").strip().lower()
    assert hangs_up == (norm in QUEUE_LEFT_RESULTS or norm in QUEUE_FAILED_RESULTS)
