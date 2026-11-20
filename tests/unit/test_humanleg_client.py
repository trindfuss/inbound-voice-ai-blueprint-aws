"""Unit tests for owlbank.humanleg_client (human-leg: resolve, add participant, close: hand-off
metadata, list active, stop transcription, task reservations)."""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pytest

from owlbank import humanleg
from owlbank.humanleg_client import (
    OrchestratorHumanLeg,
    Reservation,
    TaskReservations,
    TwilioCalls,
    metadata_problem,
    taskrouter_reservations,
)

CALL = "CA" + "1" * 32
CONV = "comms_conversation_01abc"
WORKER = "WK" + "2" * 32
KEY, SECRET = "SK" + "3" * 32, "s3cr3t-value-xyz"
ACCOUNT = "AC" + "4" * 32
CREDS = (ACCOUNT, KEY, SECRET)
CONFIG = "conv_configuration_01xyz"
TASK = "WT" + "6" * 32
WORKSPACE = "WS" + "7" * 32


class _Resp:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._raw = b"" if body is None else json.dumps(body).encode()

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


class FakeOrchestrator:
    """An in-memory Conversations v2 (plus the Calls API's transcription start and stop) behind the client's
    injectable opener. Records every request as ``(method, path, body)``; a form body is a dict too."""

    def __init__(self, *, conversations: list[dict[str, Any]] | None = None) -> None:
        self.conversations: dict[str, dict[str, Any]] = {c["id"]: dict(c) for c in conversations or []}
        self.participants: dict[str, list[dict[str, Any]]] = {}
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []
        self.queries: list[dict[str, list[str]]] = []
        self.fail: dict[tuple[str, str], Any] = {}  # (method, kind) -> status int or Exception
        self.stopped: list[tuple[str, str]] = []  # (call_sid, name)
        self.started: list[tuple[str, dict[str, Any]]] = []  # (call_sid, form params)
        self.refused: set[int] = set()  # indexes into ``requests`` that an injected failure refused

    def __call__(self, req: urllib.request.Request, timeout: float) -> _Resp:
        parts = urllib.parse.urlsplit(req.full_url)
        method = req.get_method()
        body: dict[str, Any] | None = None
        if req.data:
            if req.get_header("Content-type") == "application/x-www-form-urlencoded":
                body = {k: v[0] for k, v in urllib.parse.parse_qs(req.data.decode()).items()}  # type: ignore[union-attr]
            else:
                body = json.loads(req.data)  # type: ignore[arg-type]
        self.requests.append((method, parts.path, body))
        q = urllib.parse.parse_qs(parts.query)
        self.queries.append(q)
        segs = [s for s in parts.path.split("/") if s]
        if segs[0] == "2010-04-01":  # 2010-04-01, Accounts, AC, Calls, CA, Transcriptions[.json | , name.json]
            kind = "transcription"
        else:  # v2, Conversations[, id[, Participants]]
            kind = {2: "conversations", 3: "conversation", 4: "participants"}[len(segs)]
        injected = self.fail.get((method, kind))
        if injected is not None:
            self.refused.add(len(self.requests) - 1)
        if isinstance(injected, Exception):
            raise injected
        if isinstance(injected, int):
            raise urllib.error.HTTPError(req.full_url, injected, "err", {}, io.BytesIO(b'{"secret": "body"}'))  # type: ignore[arg-type]
        if kind == "transcription" and method == "POST" and len(segs) == 6:
            self.started.append((segs[4], dict(body or {})))
            return _Resp(201, {"sid": "GT" + "0" * 32, "status": "in-progress"})
        if kind == "transcription" and method == "POST":
            self.stopped.append((segs[4], segs[6].removesuffix(".json")))
            return _Resp(200, {"sid": "GT" + "0" * 32, "status": "stopped"})
        if kind == "conversations" and method == "GET":
            wanted = [c for c in self.conversations.values()
                      if ("status" not in q or c["status"] == q["status"][0])
                      and ("channelId" not in q or c.get("channelId") == q["channelId"][0])
                      and ("configurationId" not in q or c.get("configurationId") == q["configurationId"][0])]
            size = int(q.get("pageSize", ["50"])[0])
            start = int(q.get("pageToken", ["0"])[0])
            page = wanted[start:start + size]
            nxt = str(start + size) if start + size < len(wanted) else None
            return _Resp(200, {"conversations": page, "meta": {"nextToken": nxt}})
        cid = segs[2]
        if kind == "conversation" and method == "GET":
            return _Resp(200, self.conversations[cid]) if cid in self.conversations else _Resp404(req)
        if kind == "conversation" and method == "PUT":
            conv = self.conversations[cid]
            if conv["status"] == "CLOSED" and (body or {}).get("status") != "CLOSED":
                return _Resp400(req)  # CLOSED is final
            conv["status"] = (body or {})["status"]
            if "metadata" in (body or {}):
                conv["metadata"] = dict((body or {})["metadata"])
            return _Resp(200, conv)
        if kind == "participants" and method == "GET":
            return _Resp(200, {"participants": self.participants.get(cid, [])})
        if kind == "participants" and method == "POST":
            self.participants.setdefault(cid, []).append(dict(body or {}))
            return _Resp(201, body)
        raise AssertionError(f"unexpected request {method} {parts.path}")

    def methods(self, method: str) -> list[tuple[str, str, dict[str, Any] | None]]:
        return [r for r in self.requests if r[0] == method]


def _Resp400(req: urllib.request.Request) -> Any:  # noqa: N802
    raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]


def _Resp404(req: urllib.request.Request) -> Any:  # noqa: N802
    raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]


def active(cid: str = CONV, call: str = CALL, config: str = CONFIG) -> dict[str, Any]:
    return {"id": cid, "status": "ACTIVE", "channelId": call, "configurationId": config,
            "createdAt": "2026-11-05T10:15:30Z"}


def customer(call: str = CALL) -> dict[str, Any]:
    return {"type": "CUSTOMER", "name": "caller",
            "addresses": [{"channel": "VOICE", "address": "+15550100", "channelId": call}]}


def make(fake: FakeOrchestrator, **kw: Any) -> OrchestratorHumanLeg:
    return OrchestratorHumanLeg(lambda: CREDS, opener=fake, **kw)


# --------------------------------------------------------------------------- resolve


def test_resolve_returns_the_one_active_conversation():
    fake = FakeOrchestrator(conversations=[active()])
    assert make(fake).resolve_conversation(CALL) == CONV
    method, path, _ = fake.requests[0]
    assert (method, path) == ("GET", "/v2/Conversations")


@pytest.mark.parametrize("convs", [[], [active("comms_conversation_a"), active("comms_conversation_b")]])
def test_resolve_needs_exactly_one(convs):
    assert make(FakeOrchestrator(conversations=convs)).resolve_conversation(CALL) is None


def test_resolve_ignores_other_calls_and_closed():
    fake = FakeOrchestrator(conversations=[{**active(), "status": "CLOSED"}, active("comms_conversation_o", "CA" + "9" * 32)])
    assert make(fake).resolve_conversation(CALL) is None


def test_resolve_rejects_a_malformed_call_sid_without_a_request():
    fake = FakeOrchestrator()
    assert make(fake).resolve_conversation("not-a-sid") is None
    assert fake.requests == []


def test_resolve_sends_basic_auth_and_only_https():
    seen: list[urllib.request.Request] = []

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        seen.append(req)
        return _Resp(200, {"conversations": []})

    OrchestratorHumanLeg(lambda: CREDS, opener=opener, timeout_s=1.5).resolve_conversation(CALL)
    assert seen[0].full_url.startswith("https://conversations.twilio.com/v2/Conversations?")
    assert seen[0].get_header("Authorization", "").startswith("Basic ")
    with pytest.raises(ValueError):
        OrchestratorHumanLeg(lambda: CREDS, base_url="http://conversations.twilio.com")


@pytest.mark.parametrize("failure", [500, 401, 404, TimeoutError("t"), urllib.error.URLError("down"), OSError("x")])
def test_resolve_never_raises(failure):
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[("GET", "conversations")] = failure
    assert make(fake).resolve_conversation(CALL) is None


def test_resolve_survives_a_credentials_failure_and_a_bad_body():
    def boom() -> tuple[str, str, str]:
        raise KeyError("api_key")

    assert OrchestratorHumanLeg(boom, opener=FakeOrchestrator()).resolve_conversation(CALL) is None

    def junk(req: urllib.request.Request, timeout: float) -> Any:
        class R(_Resp):
            def read(self) -> bytes:
                return b"<html>"
        return R(200, None)

    assert OrchestratorHumanLeg(lambda: CREDS, opener=junk).resolve_conversation(CALL) is None


# --------------------------------------------------------------------------- latest


def _conv(cid: str, status: str, created: str | None, call: str = CALL) -> dict[str, Any]:
    c: dict[str, Any] = {"id": cid, "status": status, "channelId": call, "configurationId": CONFIG}
    if created is not None:
        c["createdAt"] = created
    return c


def test_latest_lists_by_call_sid_without_a_status_filter():
    fake = FakeOrchestrator(conversations=[_conv(CONV, "CLOSED", "2026-11-05T10:15:30Z")])
    assert make(fake).latest_conversation(CALL) == CONV
    method, path, _ = fake.requests[0]
    assert (method, path) == ("GET", "/v2/Conversations")
    assert "status" not in fake.queries[0] and fake.queries[0]["channelId"] == [CALL]


def test_latest_picks_the_newest_whatever_its_status():
    fake = FakeOrchestrator(conversations=[
        _conv("comms_conversation_old", "ACTIVE", "2026-11-05T10:00:00Z"),
        _conv("comms_conversation_new", "INACTIVE", "2026-11-05T10:05:00+00:00"),
        _conv("comms_conversation_mid", "CLOSED", "2026-11-05T10:02:00Z"),
        _conv("comms_conversation_oth", "ACTIVE", "2026-11-05T11:00:00Z", call="CA" + "9" * 32)])
    assert make(fake).latest_conversation(CALL) == "comms_conversation_new"


@pytest.mark.parametrize("convs", [
    [],
    [_conv("comms_conversation_a", "CLOSED", None), _conv("comms_conversation_b", "CLOSED", "junk")],
    [_conv("comms_conversation_a", "CLOSED", "2026-11-05T10:00:00Z"),
     _conv("comms_conversation_b", "ACTIVE", "2026-11-05T10:00:00Z")],
])
def test_latest_is_none_with_nothing_or_no_clear_newest(convs):
    assert make(FakeOrchestrator(conversations=convs)).latest_conversation(CALL) is None


def test_latest_takes_one_undated_conversation_alone():
    fake = FakeOrchestrator(conversations=[_conv(CONV, "INACTIVE", None)])
    assert make(fake).latest_conversation(CALL) == CONV


def test_latest_rejects_a_malformed_call_sid_without_a_request():
    fake = FakeOrchestrator()
    assert make(fake).latest_conversation("not-a-sid") is None
    assert fake.requests == []


@pytest.mark.parametrize("failure", [500, 401, 404, TimeoutError("t"), urllib.error.URLError("down"), OSError("x")])
def test_latest_never_raises(failure):
    fake = FakeOrchestrator(conversations=[_conv(CONV, "CLOSED", "2026-11-05T10:15:30Z")])
    fake.fail[("GET", "conversations")] = failure
    assert make(fake).latest_conversation(CALL) is None


def test_latest_survives_a_credentials_failure_and_a_bad_body():
    def boom() -> tuple[str, str, str]:
        raise KeyError("api_key")

    assert OrchestratorHumanLeg(boom, opener=FakeOrchestrator()).latest_conversation(CALL) is None

    def odd(req: urllib.request.Request, timeout: float) -> _Resp:
        return _Resp(200, {"conversations": "nope"})

    assert OrchestratorHumanLeg(lambda: CREDS, opener=odd).latest_conversation(CALL) is None


def test_resolve_still_asks_for_active_only():
    fake = FakeOrchestrator(conversations=[active()])
    make(fake).resolve_conversation(CALL)
    assert fake.queries[0]["status"] == ["ACTIVE"]


# --------------------------------------------------------------------------- add participant


def test_add_posts_a_human_agent_once_and_second_add_is_a_no_op():
    fake = FakeOrchestrator(conversations=[active()])
    client = make(fake)
    assert client.add_human_participant(CONV, WORKER, "agent@example.com") is True
    assert client.add_human_participant(CONV, WORKER, "agent@example.com") is False
    posts = fake.methods("POST")
    assert len(posts) == 1
    assert posts[0][1] == f"/v2/Conversations/{CONV}/Participants"
    assert posts[0][2]["type"] == "HUMAN_AGENT"  # type: ignore[index]


def test_add_two_different_workers_adds_both():
    fake = FakeOrchestrator(conversations=[active()])
    other = "WK" + "5" * 32
    assert make(fake).add_human_participant(CONV, WORKER, "a") is True
    assert make(fake).add_human_participant(CONV, other, "b") is True
    assert len(fake.participants[CONV]) == 2


def test_add_treats_409_as_already_present():
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[("POST", "participants")] = 409
    assert make(fake).add_human_participant(CONV, WORKER, "a") is False


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("failure", [400, 403, 404, 500, 503, TimeoutError("t"), urllib.error.URLError("down")])
def test_add_never_raises(method, failure):
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[(method, "participants")] = failure
    assert make(fake).add_human_participant(CONV, WORKER, "a") is False


def test_add_rejects_bad_ids_without_a_request():
    fake = FakeOrchestrator()
    client = make(fake)
    assert client.add_human_participant("../x", WORKER, "a") is False
    assert client.add_human_participant(CONV, "not-a-worker", "a") is False
    assert fake.requests == []


# --------------------------------------------------------------------------- close


def test_close_puts_closed_and_a_double_close_sends_one_put():
    fake = FakeOrchestrator(conversations=[active()])
    client = make(fake)
    assert client.close(CONV) is True
    assert client.close(CONV) is True
    puts = fake.methods("PUT")
    assert len(puts) == 1
    assert puts[0][1:] == (f"/v2/Conversations/{CONV}", {"status": "CLOSED"})
    assert fake.conversations[CONV]["status"] == "CLOSED"


def test_close_of_an_already_closed_conversation_is_success_with_no_put():
    fake = FakeOrchestrator(conversations=[{**active(), "status": "CLOSED"}])
    assert make(fake).close(CONV) is True
    assert fake.methods("PUT") == []


def test_close_refused_put_but_closed_meanwhile_is_success():
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[("PUT", "conversation")] = 409
    gets = {"n": 0}
    real = fake.__call__

    def opener(req: urllib.request.Request, timeout: float) -> Any:
        if req.get_method() == "GET":
            gets["n"] += 1
            if gets["n"] == 2:  # the re-read after the refused PUT
                fake.conversations[CONV]["status"] = "CLOSED"
        return real(req, timeout)

    assert OrchestratorHumanLeg(lambda: CREDS, opener=opener).close(CONV) is True


@pytest.mark.parametrize("failure", [400, 404, 500, 503, TimeoutError("t"), urllib.error.URLError("down")])
def test_close_failure_returns_false_and_never_raises(failure):
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[("PUT", "conversation")] = failure
    assert make(fake).close(CONV) is False


def test_close_still_puts_when_the_read_fails():
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[("GET", "conversation")] = 500
    assert make(fake).close(CONV) is True
    assert len(fake.methods("PUT")) == 1


def test_close_rejects_a_bad_id():
    fake = FakeOrchestrator()
    assert make(fake).close("a/b") is False
    assert fake.requests == []


# --------------------------------------------------------------------------- logs


def test_failures_are_logged_with_ids_and_never_with_credentials(caplog):
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[("GET", "conversations")] = 500
    fake.fail[("PUT", "conversation")] = TimeoutError(f"secret {SECRET} {KEY}")
    fake.fail[("POST", "participants")] = 500
    client = make(fake)
    with caplog.at_level(logging.DEBUG):
        client.resolve_conversation(CALL)
        client.add_human_participant(CONV, WORKER, "agent")
        client.close(CONV)
    text = "\n".join(
        r.getMessage() + " " + json.dumps(r.__dict__, default=str) for r in caplog.records)
    basic = base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()
    for leaked in (SECRET, KEY, basic):
        assert leaked not in text
    assert CALL in text and CONV in text


# --------------------------------------------------------------------------- participant type


def test_every_participant_body_carries_an_explicit_type():
    fake = FakeOrchestrator(conversations=[active()])
    make(fake).add_human_participant(CONV, WORKER, "")
    (post,) = fake.methods("POST")
    assert post[2]["type"] == "HUMAN_AGENT"  # type: ignore[index]
    assert post[2]["name"] == WORKER  # type: ignore[index]


def test_the_worker_is_a_voice_client_address():
    # channel "API" got HTTP 400 (v2 enum: VOICE, SMS, RCS, WHATSAPP, CHAT, VIDEO).
    fake = FakeOrchestrator(conversations=[active()])
    make(fake).add_human_participant(CONV, WORKER, "ana")
    (post,) = fake.methods("POST")
    assert post[2]["addresses"] == [  # type: ignore[index]
        {"channel": "VOICE", "address": "client:ana", "channelId": WORKER}]


@pytest.mark.parametrize("listed", [
    {"type": "HUMAN_AGENT", "name": "ana", "addresses": [{"channel": "VOICE", "address": "client:ana"}]},
    {"type": "HUMAN_AGENT", "name": "x", "addresses": [{"channel": "VOICE", "address": "client:x",
                                                        "channelId": WORKER}]},
])
def test_a_listed_worker_is_not_added_again(listed):
    # Orchestrator may not echo channelId: the client address alone also counts.
    fake = FakeOrchestrator(conversations=[active()])
    fake.participants[CONV] = [customer(), listed]
    assert make(fake).add_human_participant(CONV, WORKER, "ana") is False
    assert fake.methods("POST") == []


def test_the_ai_side_voice_address_is_not_the_worker():
    fake = FakeOrchestrator(conversations=[active()])
    fake.participants[CONV] = [customer(), {"type": "AGENT", "name": "+15550199",
                                            "addresses": [{"channel": "VOICE", "address": "client:ana"}]}]
    assert make(fake).add_human_participant(CONV, WORKER, "ana") is True


def test_add_after_close_sends_nothing():
    fake = FakeOrchestrator(conversations=[active()])
    client = make(fake)
    assert client.close(CONV) is True
    before = len(fake.requests)
    assert client.add_human_participant(CONV, WORKER, "a") is False
    assert len(fake.requests) == before


# --------------------------------------------------------------------------- mark_handoff

HANDOFF = {"handoff_at": "1762340000", "handoff_to": "human"}


def test_mark_handoff_puts_active_with_the_full_merged_metadata():
    fake = FakeOrchestrator(conversations=[{**active(), "metadata": {"bank_key": "kept"}}])
    assert make(fake).mark_handoff(CONV, HANDOFF) is True
    (put,) = fake.methods("PUT")
    assert put[1] == f"/v2/Conversations/{CONV}"
    assert put[2] == {"status": "ACTIVE", "metadata": {"bank_key": "kept", **HANDOFF}}
    assert fake.conversations[CONV]["metadata"] == {"bank_key": "kept", **HANDOFF}


def test_mark_handoff_twice_sends_one_put():
    fake = FakeOrchestrator(conversations=[active()])
    client = make(fake)
    assert client.mark_handoff(CONV, HANDOFF) is True
    assert client.mark_handoff(CONV, HANDOFF) is True
    assert len(fake.methods("PUT")) == 1


def test_mark_handoff_on_a_closed_conversation_sends_no_put():
    fake = FakeOrchestrator(conversations=[{**active(), "status": "CLOSED"}])
    client = make(fake)
    assert client.mark_handoff(CONV, HANDOFF) is False
    assert fake.methods("PUT") == []
    # and remembered: a second try sends nothing at all
    before = len(fake.requests)
    assert client.mark_handoff(CONV, HANDOFF) is False
    assert len(fake.requests) == before


def test_mark_handoff_after_own_close_sends_nothing():
    fake = FakeOrchestrator(conversations=[active()])
    client = make(fake)
    client.close(CONV)
    before = len(fake.requests)
    assert client.mark_handoff(CONV, HANDOFF) is False
    assert len(fake.requests) == before


@pytest.mark.parametrize("bad", [
    {f"k{i}": "v" for i in range(9)},
    {"k" * 129: "v"},
    {"": "v"},
    {"k": "v" * 513},
    {"k": 1},
    {1: "v"},
    "not-a-mapping",
])
def test_mark_handoff_limits_are_enforced_before_any_request(bad):
    fake = FakeOrchestrator(conversations=[active()])
    assert make(fake).mark_handoff(CONV, bad) is False  # type: ignore[arg-type]
    assert fake.requests == []


def test_metadata_limits_at_the_edges_are_accepted():
    edge = {f"k{i}" + "x" * (128 - len(f"k{i}")): "v" * 512 for i in range(8)}
    assert metadata_problem(edge) is None
    fake = FakeOrchestrator(conversations=[active()])
    assert make(fake).mark_handoff(CONV, edge) is True


def test_mark_handoff_refuses_when_the_merge_would_exceed_eight_keys():
    existing = {f"bank{i}": "v" for i in range(7)}
    fake = FakeOrchestrator(conversations=[{**active(), "metadata": existing}])
    assert make(fake).mark_handoff(CONV, HANDOFF) is False
    assert fake.methods("PUT") == []


@pytest.mark.parametrize("odd", ["a string", ["list"], {"k": 5}])
def test_mark_handoff_does_not_overwrite_metadata_of_an_unknown_shape(odd):
    fake = FakeOrchestrator(conversations=[{**active(), "metadata": odd}])
    assert make(fake).mark_handoff(CONV, HANDOFF) is False
    assert fake.methods("PUT") == []


@pytest.mark.parametrize("method", ["GET", "PUT"])
@pytest.mark.parametrize("failure", [400, 404, 500, TimeoutError("t"), urllib.error.URLError("down")])
def test_mark_handoff_never_raises(method, failure):
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[(method, "conversation")] = failure
    assert make(fake).mark_handoff(CONV, HANDOFF) is False


def test_mark_handoff_rejects_a_bad_id():
    fake = FakeOrchestrator()
    assert make(fake).mark_handoff("../x", HANDOFF) is False
    assert fake.requests == []


# --------------------------------------------------------------------------- start transcription

START = humanleg.transcription_start("conv_1", enabled=True, language="en-US",
                                     status_url="https://h/voice/transcription", engine="deepgram",
                                     speech_model="nova-3-general")


def test_start_transcription_posts_the_params_to_the_calls_transcriptions_resource():
    seen: list[urllib.request.Request] = []

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        seen.append(req)
        return _Resp(201, {"status": "in-progress"})

    assert START is not None
    assert TwilioCalls(lambda: CREDS, opener=opener).start_transcription(CALL, START) is True
    (req,) = seen
    assert req.get_method() == "POST"
    assert req.full_url == f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT}/Calls/{CALL}/Transcriptions.json"
    assert dict(urllib.parse.parse_qsl(req.data.decode())) == dict(START)  # type: ignore[union-attr]
    assert req.get_header("Content-type") == "application/x-www-form-urlencoded"
    assert req.get_header("Authorization") == "Basic " + base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()


def test_start_and_stop_share_the_transcription_name():
    fake = FakeOrchestrator()
    calls = TwilioCalls(lambda: CREDS, opener=fake)
    assert START is not None
    assert calls.start_transcription(CALL, START) is True
    assert calls.stop_transcription(CALL, humanleg.TRANSCRIPTION_NAME) is True
    ((started_call, params),) = fake.started
    assert (started_call, params["Name"]) == fake.stopped[0] == (CALL, humanleg.TRANSCRIPTION_NAME)


@pytest.mark.parametrize("failure", [400, 404, 409, 500, TimeoutError("t"), urllib.error.URLError("down")])
def test_start_transcription_failure_is_false_and_never_raises(failure, caplog):
    fake = FakeOrchestrator()
    fake.fail[("POST", "transcription")] = failure
    assert START is not None
    with caplog.at_level(logging.DEBUG):
        assert TwilioCalls(lambda: CREDS, opener=fake).start_transcription(CALL, START) is False
    text = "\n".join(r.getMessage() + " " + json.dumps(r.__dict__, default=str) for r in caplog.records)
    assert "human_leg_start_failed" in text and CALL in text
    basic = base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()
    for leaked in (SECRET, KEY, basic):
        assert leaked not in text


@pytest.mark.parametrize(("call", "params"), [
    ("bad", {"Name": "humanleg"}), (CALL, {}), (CALL, {"Name": "../x"}), (CALL, {"Name": "a b"}),
    (CALL, {"Name": "humanleg", "Track": None}), (CALL, None)])
def test_start_transcription_rejects_bad_input_without_a_request(call, params):
    fake = FakeOrchestrator()
    assert TwilioCalls(lambda: CREDS, opener=fake).start_transcription(call, params) is False
    assert fake.requests == []


def test_start_transcription_bad_account_or_credentials_failure():
    fake = FakeOrchestrator()
    assert START is not None
    assert TwilioCalls(lambda: ("nope", KEY, SECRET), opener=fake).start_transcription(CALL, START) is False

    def boom() -> tuple[str, str, str]:
        raise KeyError("api_key")

    assert TwilioCalls(boom, opener=fake).start_transcription(CALL, START) is False
    assert fake.requests == []


# --------------------------------------------------------------------------- stop transcription


def test_stop_transcription_posts_status_stopped_to_the_named_transcription():
    seen: list[urllib.request.Request] = []

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        seen.append(req)
        return _Resp(200, {"status": "stopped"})

    assert TwilioCalls(lambda: CREDS, opener=opener).stop_transcription(CALL, "humanleg") is True
    (req,) = seen
    assert req.get_method() == "POST"
    assert req.full_url == (f"https://api.twilio.com/2010-04-01/Accounts/{ACCOUNT}/Calls/{CALL}"
                            "/Transcriptions/humanleg.json")
    assert req.data == b"Status=stopped"
    assert req.get_header("Content-type") == "application/x-www-form-urlencoded"
    assert req.get_header("Authorization") == "Basic " + base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()


@pytest.mark.parametrize("failure", [400, 404, 409, 500, TimeoutError("t"), urllib.error.URLError("down")])
def test_stop_transcription_failure_is_false_and_never_raises(failure):
    fake = FakeOrchestrator()
    fake.fail[("POST", "transcription")] = failure
    assert TwilioCalls(lambda: CREDS, opener=fake).stop_transcription(CALL, "humanleg") is False


@pytest.mark.parametrize(("call", "name"), [("bad", "humanleg"), (CALL, "../x"), (CALL, ""), (CALL, "a b")])
def test_stop_transcription_rejects_bad_input_without_a_request(call, name):
    fake = FakeOrchestrator()
    assert TwilioCalls(lambda: CREDS, opener=fake).stop_transcription(call, name) is False
    assert fake.requests == []


def test_stop_transcription_bad_account_or_credentials_failure_and_https_only():
    fake = FakeOrchestrator()
    assert TwilioCalls(lambda: ("nope", KEY, SECRET), opener=fake).stop_transcription(CALL, "humanleg") is False

    def boom() -> tuple[str, str, str]:
        raise KeyError("api_key")

    assert TwilioCalls(boom, opener=fake).stop_transcription(CALL, "humanleg") is False
    assert fake.requests == []
    with pytest.raises(ValueError):
        TwilioCalls(lambda: CREDS, base_url="http://api.twilio.com")


# --------------------------------------------------------------------------- reservations


class _Source:
    def __init__(self, rows: Any = None, exc: Exception | None = None) -> None:
        self.rows, self.exc = rows or [], exc

    def reservations(self, task_sid: str) -> Any:
        if self.exc:
            raise self.exc
        return self.rows


def test_task_reservations_keeps_well_formed_rows():
    rows = [{"worker_sid": WORKER, "status": "accepted"}, {"worker_sid": "nope", "status": "accepted"},
            {"worker_sid": WORKER, "status": None}, "junk"]
    assert TaskReservations(_Source(rows)).list(TASK) == [Reservation(WORKER, "accepted")]


def test_task_reservations_failure_is_none():
    from owlbank.taskrouter import TaskRouterUnavailable

    assert TaskReservations(_Source(exc=TaskRouterUnavailable("x"))).list(TASK) is None
    assert TaskReservations(_Source(exc=RuntimeError("x"))).list(TASK) is None


def test_taskrouter_reservations_reuses_the_context_api_reader():
    seen: list[urllib.request.Request] = []

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        seen.append(req)
        return _Resp(200, {"reservations": [{"worker_sid": WORKER, "reservation_status": "accepted"},
                                            {"worker_sid": "WK" + "9" * 32, "reservation_status": "rejected"}],
                           "meta": {}})

    got = taskrouter_reservations(WORKSPACE, lambda: CREDS, opener=opener).list(TASK)
    assert got == [Reservation(WORKER, "accepted"), Reservation("WK" + "9" * 32, "rejected")]
    assert seen[0].full_url.startswith(
        f"https://taskrouter.twilio.com/v1/Workspaces/{WORKSPACE}/Tasks/{TASK}/Reservations")
    assert seen[0].get_header("Authorization") == "Basic " + base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()


def test_taskrouter_reservations_incomplete_page_or_no_workspace_is_none():
    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        return _Resp(200, {"reservations": [], "meta": {"next_page_url": "https://x"}})

    assert taskrouter_reservations(WORKSPACE, lambda: CREDS, opener=opener).list(TASK) is None
    assert taskrouter_reservations(None, lambda: CREDS, opener=opener).list(TASK) is None


# --------------------------------------------------------------------------- logs


def test_new_operations_never_log_credentials(caplog):
    fake = FakeOrchestrator(conversations=[active()])
    fake.fail[("GET", "conversation")] = TimeoutError(f"secret {SECRET} {KEY}")
    fake.fail[("GET", "conversations")] = 500
    fake.fail[("POST", "transcription")] = TimeoutError(f"secret {SECRET}")
    with caplog.at_level(logging.DEBUG):
        make(fake).mark_handoff(CONV, HANDOFF)
        make(fake).resolve_conversation(CALL)
        TwilioCalls(lambda: CREDS, opener=fake).stop_transcription(CALL, "humanleg")
    text = "\n".join(r.getMessage() + " " + json.dumps(r.__dict__, default=str) for r in caplog.records)
    basic = base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()
    for leaked in (SECRET, KEY, basic):
        assert leaked not in text
    assert CALL in text and CONV in text


# --------------------------------------------------------------------------- budget


def test_budget_cuts_each_timeout_to_what_is_left_and_sends_nothing_once_spent(monkeypatch):
    from owlbank import humanleg_client

    now = [100.0]
    monkeypatch.setattr(humanleg_client.time, "monotonic", lambda: now[0])
    timeouts: list[float] = []

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        timeouts.append(timeout)
        now[0] += 0.8  # each request takes 0.8 s
        return _Resp(200, {"conversations": []})

    client = OrchestratorHumanLeg(lambda: CREDS, opener=opener, timeout_s=2.0)
    with humanleg_client.budget(2.0):
        for _ in range(4):
            client.resolve_conversation(CALL)
    assert timeouts == pytest.approx([2.0, 1.2, 0.4])  # the fourth is never sent
    client.resolve_conversation(CALL)  # outside the budget: the client's own timeout again
    assert timeouts[-1] == 2.0


def test_a_nested_budget_keeps_the_earlier_deadline(monkeypatch):
    from owlbank import humanleg_client

    monkeypatch.setattr(humanleg_client.time, "monotonic", lambda: 0.0)
    seen: list[float] = []

    def opener(req: urllib.request.Request, timeout: float) -> _Resp:
        seen.append(timeout)
        return _Resp(200, {"conversations": []})

    calls = OrchestratorHumanLeg(lambda: CREDS, opener=opener, timeout_s=3.0)
    with humanleg_client.budget(1.0), humanleg_client.budget(5.0):
        calls.resolve_conversation(CALL)
    assert seen == [1.0]


def test_a_spent_budget_is_a_plain_failure_for_every_operation(monkeypatch):
    from owlbank import humanleg_client

    fake = FakeOrchestrator(conversations=[active()])
    client, calls = make(fake), TwilioCalls(lambda: CREDS, opener=fake)
    with humanleg_client.budget(0.0):
        assert client.add_human_participant(CONV, WORKER, "Alice") is False
        assert client.mark_handoff(CONV, {"handoff_to": "human"}) is False
        assert calls.start_transcription(CALL, {"Name": "humanleg"}) is False
    assert fake.requests == []


# --------------------------------------------------------------------------- communications / participants


class _Paged:
    """Serves ``pages`` for a list GET; page i is returned for token ``str(i)`` (none for the first)."""

    def __init__(self, key: str, pages: list[Any], *, endless: bool = False) -> None:
        self.key, self.pages, self.endless = key, pages, endless
        self.seen: list[tuple[str, dict[str, list[str]]]] = []

    def __call__(self, req: urllib.request.Request, timeout: float) -> _Resp:
        parts = urllib.parse.urlsplit(req.full_url)
        q = urllib.parse.parse_qs(parts.query)
        self.seen.append((parts.path, q))
        i = int(q.get("pageToken", ["0"])[0])
        page = self.pages[min(i, len(self.pages) - 1)] if self.endless else self.pages[i]
        if isinstance(page, int):
            raise urllib.error.HTTPError(req.full_url, page, "err", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]
        more = self.endless or i + 1 < len(self.pages)
        return _Resp(200, {self.key: page, "meta": {"nextToken": str(i + 1) if more else None}})


@pytest.mark.parametrize(("method", "segment", "key"), [("communications", "Communications", "communications"),
                                                         ("participants", "Participants", "participants")])
def test_list_reads_follow_the_next_token_over_three_pages(method, segment, key):
    pages = [[{"id": "a"}, {"id": "b"}], [{"id": "c"}], [{"id": "d"}]]
    fake = _Paged(key, pages)
    got = getattr(OrchestratorHumanLeg(lambda: CREDS, opener=fake), method)(CONV)
    assert [i["id"] for i in got] == ["a", "b", "c", "d"]
    assert [p for p, _ in fake.seen] == [f"/v2/Conversations/{CONV}/{segment}"] * 3
    assert [q.get("pageToken") for _, q in fake.seen] == [None, ["1"], ["2"]]
    assert all(q["pageSize"] == ["100"] for _, q in fake.seen)


@pytest.mark.parametrize("method", ["communications", "participants"])
def test_list_reads_stop_at_the_page_cap_and_return_none(method):
    key = method
    fake = _Paged(key, [[{"id": "x"}]], endless=True)
    assert getattr(OrchestratorHumanLeg(lambda: CREDS, opener=fake), method)(CONV) is None
    assert len(fake.seen) == 20


@pytest.mark.parametrize("method", ["communications", "participants"])
def test_list_reads_ask_more_before_each_page_and_stop_when_told(method):
    fake = _Paged(method, [[{"id": "a"}], [{"id": "b"}], [{"id": "c"}]])
    answers = iter([True, True, False])
    asked: list[int] = []

    def more() -> bool:
        asked.append(len(fake.seen))
        return next(answers)

    assert getattr(OrchestratorHumanLeg(lambda: CREDS, opener=fake), method)(CONV, more=more) is None
    assert asked == [0, 1, 2] and len(fake.seen) == 2  # never a part: None, the third page not sent
    always = _Paged(method, [[{"id": "a"}], [{"id": "b"}]])
    got = getattr(OrchestratorHumanLeg(lambda: CREDS, opener=always), method)(CONV, more=lambda: True)
    assert [i["id"] for i in got] == ["a", "b"]


@pytest.mark.parametrize("method", ["communications", "participants"])
@pytest.mark.parametrize("failure", [500, 404, OSError("boom"), TimeoutError()])
def test_list_reads_return_none_on_any_failure(method, failure):
    fake = _Paged(method, [[{"id": "a"}], failure])
    client = OrchestratorHumanLeg(lambda: CREDS, opener=fake)
    assert getattr(client, method)(CONV) is None  # first page already failing or mid-paging


@pytest.mark.parametrize("method", ["communications", "participants"])
def test_list_reads_return_none_on_bad_body_or_id_and_credentials_failure(method):
    def bad(req: Any, timeout: float) -> _Resp:
        return _Resp(200, {"unrelated": 1})

    assert getattr(OrchestratorHumanLeg(lambda: CREDS, opener=bad), method)(CONV) is None
    fake = _Paged(method, [[]])
    assert getattr(OrchestratorHumanLeg(lambda: CREDS, opener=fake), method)("../x") is None
    assert fake.seen == []

    def no_creds() -> Any:
        raise RuntimeError("nope")

    assert getattr(OrchestratorHumanLeg(no_creds, opener=fake), method)(CONV) is None
    assert getattr(OrchestratorHumanLeg(lambda: CREDS, opener=_Paged(method, [[]])), method)(CONV) == []
