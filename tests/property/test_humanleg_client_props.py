"""Human-leg properties of the Orchestrator client against a fake Orchestrator.

* Over any sequence of add, duplicate add, close, hand-off metadata and
  transcription-stop calls, with or without injected failures, the conversation holds at most one
  participant per worker and at most one close request reaches it (a close refused by an injected
  failure may be retried; one that went through is never repeated).
* No operation, whatever the sequence, sends a request that could create a conversation (a POST
  to the conversations collection); the only writes are POST participants, POST transcription stop, PUT
  close and PUT hand-off on the one existing conversation.
* Once the close PUT has succeeded, no participant add or metadata PUT is sent, and
  every metadata PUT that is sent is within Orchestrator's limits and status ACTIVE.
"""

from __future__ import annotations

import urllib.error
from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.humanleg_client import OrchestratorHumanLeg, TwilioCalls, metadata_problem
from tests.unit.test_humanleg_client import CALL, CONV, CREDS, FakeOrchestrator, active, customer

WORKERS = ["WK" + c * 32 for c in "abc"]

metadata = st.dictionaries(
    st.text(min_size=0, max_size=130), st.text(max_size=515), max_size=10,
) | st.just({"handoff_at": "1762340000", "handoff_to": "human"})

operation = st.one_of(
    st.tuples(st.just("add"), st.sampled_from(WORKERS), st.sampled_from(["x", "y"])),
    st.tuples(st.just("close"), st.none(), st.none()),
    st.tuples(st.just("resolve"), st.none(), st.none()),
    st.tuples(st.just("handoff"), metadata, st.none()),
    st.tuples(st.just("stop"), st.sampled_from(["humanleg", "bad name"]), st.none()),
)
# A failure injected for one step: (method, kind) the next request of that kind fails with.
failure = st.one_of(
    st.none(),
    st.tuples(st.sampled_from([("GET", "participants"), ("POST", "participants"), ("GET", "conversation"),
                               ("GET", "conversations"), ("PUT", "conversation"), ("POST", "transcription")]),
              st.sampled_from([500, 503, 409, TimeoutError("t"), urllib.error.URLError("down")])),
)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(operation, failure), max_size=25))
def test_one_participant_per_worker_at_most_one_close_and_never_a_create(steps):
    fake = FakeOrchestrator(conversations=[active()])
    fake.participants[CONV] = [customer()]
    client = OrchestratorHumanLeg(lambda: CREDS, opener=fake)
    calls = TwilioCalls(lambda: CREDS, opener=fake)
    for (kind, arg, identity), injected in steps:
        if injected is not None:
            fake.fail[injected[0]] = injected[1]
        if kind == "add":
            assert isinstance(client.add_human_participant(CONV, arg, identity), bool)
        elif kind == "close":
            assert isinstance(client.close(CONV), bool)
        elif kind == "handoff":
            assert isinstance(client.mark_handoff(CONV, arg), bool)
        elif kind == "stop":
            assert isinstance(calls.stop_transcription(CALL, arg), bool)
        else:
            client.resolve_conversation(CALL)
        fake.fail.clear()

    workers = [p for p in fake.participants.get(CONV, []) if p.get("type") == "HUMAN_AGENT"]
    stored = Counter(a["channelId"] for p in workers for a in p["addresses"])
    assert all(n == 1 for n in stored.values())  # One participant per worker
    assert all(p.get("type") for p in fake.participants.get(CONV, []))  # explicit type on every body
    closes = [i for i, r in enumerate(fake.requests) if r[0] == "PUT" and (r[2] or {}).get("status") == "CLOSED"
              and i not in fake.refused]
    assert len(closes) <= 1  # At most one close request reaches the conversation (a refused one may be retried)
    for i, (method, path, body) in enumerate(fake.requests):
        assert not (method == "POST" and path.rstrip("/") == "/v2/Conversations")
        assert method in ("GET", "PUT", "POST")
        assert method != "POST" or path.endswith(("/Participants", "/Transcriptions/humanleg.json"))
        assert method != "PUT" or path == f"/v2/Conversations/{CONV}"
        if method == "PUT" and (body or {}).get("status") != "CLOSED":
            assert body is not None and body["status"] == "ACTIVE"
            assert metadata_problem(body["metadata"]) is None  # limits enforced before sending
    if closes and fake.conversations[CONV]["status"] == "CLOSED":  # The close took effect
        after = fake.requests[closes[0] + 1:]
        assert not any(m == "PUT" for m, _p, _b in after)
        assert not any(m == "POST" and p.endswith("/Participants") for m, p, _b in after)
