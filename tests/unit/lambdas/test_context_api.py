"""Unit tests for lambdas.context_api."""

from __future__ import annotations

import json
import logging
import urllib.error

import pytest

from lambdas.context_api import handler as ctx
from lambdas.context_api.handler import (
    ALLOWED_ORIGIN,
    authorize_task,
    build_timeline,
    get_context,
    parse_roles,
)
from owlbank import callstate, taskrouter
from owlbank.callstate import CallState
from owlbank.pii import FakePiiRedactor
from owlbank.segments import SegmentLog
from owlbank.store import MemoryStore
from owlbank.taskrouter import TaskRouterUnavailable, TwilioTaskRouterReader

SID = "CA" + "e" * 32
OTHER_SID = "CA" + "f" * 32
REF = "jr_7f3c2a"
NOW = 1_760_108_645
WORKER = "WK" + "1" * 32
ZIP = "94107"
SSN4 = "4821"


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore()
    callstate.create(s, CallState(
        call_sid=SID, expires_at=NOW + 3600, language="es-US", ani="+12025550101",
        verified_customer_id="C-1042", level=2, factors_verified=("zip", "ssn4"),
        escalation={"reason": "fraud_suspected", "journal_ref": REF}))
    log = SegmentLog(s, now=lambda: NOW)
    log.append(SID, {"type": "call_start", "ani": "+12025550101"})
    log.append(SID, {"type": "prompt", "prompt_key": "verify.zip", "rendered_text": "Please enter your ZIP code."})
    log.append(SID, {"type": "factor_asked", "factor": "zip"})
    log.append(SID, {"type": "verification", "outcome": "verified", "level": 2})
    log.append(SID, {"type": "ai_turn", "role": "caller", "text": "my card was stolen"})
    log.append(SID, {"type": "tool_call", "tool": "list_recent_transactions", "args": {"account": "checking"}})
    log.append(SID, {"type": "tool_result", "tool": "list_recent_transactions", "outcome": "completed",
                     "data": {"items": [{"amount": "12.00"}]}})
    log.append(SID, {"type": "escalation", "reason": "fraud_suspected", "queue": "Fraud",
                     "summary": "AI text"})
    # Written behind the guard's back (a buggy writer): must never be served.
    s.put("segments", {"call_sid": SID, "seq": 50, "ts": NOW, "type": "prompt", "prompt_key": "x",
                       "rendered_text": "leak", "digits": ZIP})
    s.put("segments", {"call_sid": SID, "seq": 51, "ts": NOW, "type": "prompt", "prompt_key": "verify.echo",
                       "rendered_text": f"You entered {SSN4}, and 9 4 1 0 7."})
    return s


def request(ref: str = REF, call_sid: str | None = SID, origin: str | None = ALLOWED_ORIGIN,
            method: str = "GET", worker: str | None = WORKER) -> dict:
    headers = {"X-Flex-Token": "tok"}
    if origin is not None:
        headers["Origin"] = origin
    return {
        "httpMethod": method, "resource": "/context/{journal_ref}", "headers": headers,
        "pathParameters": {"journal_ref": ref},
        "queryStringParameters": {"call_sid": call_sid} if call_sid else None,
        "requestContext": {"authorizer": {"worker_sid": worker} if worker else {}},
    }


def call(store, **kw) -> tuple[int, dict, dict]:
    req = request(**kw)
    if req["queryStringParameters"]:
        req["queryStringParameters"]["task_sid"] = TASK
    req["requestContext"]["authorizer"]["roles"] = "supervisor"
    attrs = {"call_sid": kw.get("call_sid", SID), "journal_ref": kw.get("ref", REF)}
    resp = get_context(req, store, now=lambda: NOW, taskrouter=FakeTaskRouter(tasks={TASK: attrs}))
    return resp["statusCode"], (json.loads(resp["body"]) if resp["body"] else {}), resp["headers"]


def test_returns_redacted_timeline(store) -> None:
    status, body, headers = call(store)
    assert status == 200
    assert body["language"] == "es-US"
    assert body["verified_level"] == 2
    assert body["escalation_reason"] == "fraud_suspected"
    assert body["journal_ref"] == REF
    types = [e["type"] for e in body["timeline"]]
    assert "ai_turn" not in types
    assert types[:4] == ["call_start", "prompt", "factor_asked", "verification"]
    prompt = body["timeline"][1]
    assert prompt == {"seq": 2, "ts": NOW, "type": "prompt", "prompt_key": "verify.zip",
                      "rendered_text": "Please enter your ZIP code."}
    tool_call = next(e for e in body["timeline"] if e["type"] == "tool_call")
    assert tool_call == {"seq": 6, "ts": NOW, "type": "tool_call", "tool": "list_recent_transactions"}
    esc = next(e for e in body["timeline"] if e["type"] == "escalation")
    assert esc["queue"] == "Fraud" and "summary" not in esc
    assert headers["Cache-Control"] == "no-store"


def test_never_returns_factor_values_or_tokens(store) -> None:
    _, body, _ = call(store)
    text = json.dumps(body)
    for secret in (ZIP, SSN4, "9 4 1 0 7", "leak", "tok", "args", "data", "ani", "+1202"):
        assert secret not in text
    assert all(e["seq"] != 50 for e in body["timeline"])
    echo = next(e for e in body["timeline"] if e["seq"] == 51)
    assert echo["rendered_text"] == "You entered [redacted], and [redacted]."


def test_build_timeline_drops_nested_denied_keys() -> None:
    events = [{"seq": 1, "type": "tool_result", "outcome": "ok", "meta": {"token_hash": "abc"}},
              {"seq": 2, "type": "handoff", "outcome": "step_up", "tool": "make_payment"}]
    assert build_timeline(events) == [{"seq": 2, "type": "handoff", "outcome": "step_up",
                                       "tool": "make_payment"}]


def test_unknown_call_is_404(store) -> None:
    status, body, _ = call(store, call_sid=OTHER_SID)
    assert status == 404 and body == {"error": "not_found"}


def test_unknown_ref_is_404(store) -> None:
    status, body, _ = call(store, ref="jr_other")
    assert status == 404 and body == {"error": "not_found"}


def test_ref_of_another_call_is_404(store) -> None:
    callstate.create(store, CallState(call_sid=OTHER_SID, expires_at=NOW + 3600,
                                      escalation={"reason": "complaint", "journal_ref": "jr_b"}))
    assert call(store, ref="jr_b")[0] == 404          # ref B with call A
    assert call(store, ref=REF, call_sid=OTHER_SID)[0] == 404  # ref A with call B
    assert call(store, ref="jr_b", call_sid=OTHER_SID)[0] == 200


def test_call_without_escalation_is_404(store) -> None:
    callstate.create(store, CallState(call_sid=OTHER_SID, expires_at=NOW + 3600))
    assert call(store, call_sid=OTHER_SID)[0] == 404


def test_expired_call_is_404(store) -> None:
    resp = get_context(request(), store, now=lambda: NOW + 7200, taskrouter=FakeTaskRouter())
    assert resp["statusCode"] == 404


@pytest.mark.parametrize("kw,status", [
    ({"call_sid": None}, 400), ({"call_sid": "CA123"}, 400), ({"ref": "bad ref/../x"}, 400),
    ({"worker": None}, 403), ({"method": "POST"}, 405),
])
def test_bad_requests(store, kw, status) -> None:
    assert call(store, **kw)[0] == status


def test_cors_allows_only_flex_origin(store) -> None:
    _, _, headers = call(store)
    assert headers["Access-Control-Allow-Origin"] == "https://flex.twilio.com"
    assert headers["Vary"] == "Origin"
    for origin in ("https://evil.example", "https://flex.twilio.com.evil.example", "null", None):
        _, _, h = call(store, origin=origin)
        assert "Access-Control-Allow-Origin" not in h


def test_cors_preflight(store) -> None:
    ok = get_context(request(method="OPTIONS"), store)
    assert ok["statusCode"] == 204
    assert ok["headers"]["Access-Control-Allow-Origin"] == ALLOWED_ORIGIN
    assert "X-Flex-Token" in ok["headers"]["Access-Control-Allow-Headers"]
    bad = get_context(request(method="OPTIONS", origin="https://evil.example"), store)
    assert bad["statusCode"] == 403
    assert "Access-Control-Allow-Origin" not in bad["headers"]


def test_no_reader_configured_is_503_and_reads_no_timeline(store, monkeypatch) -> None:
    real_query = store.query

    def guarded_query(table, *a, **kw):
        assert table != "segments", "segments read without a TaskRouter reader"
        return real_query(table, *a, **kw)

    monkeypatch.setattr(store, "query", guarded_query)
    req = request()
    req["queryStringParameters"]["task_sid"] = TASK
    req["requestContext"]["authorizer"]["roles"] = "supervisor"
    resp = get_context(req, store, now=lambda: NOW)
    assert resp["statusCode"] == 503 and json.loads(resp["body"]) == {"error": "unavailable"}


def test_store_failure_is_500_without_details(store) -> None:
    class Broken(MemoryStore):
        def get(self, table, key):
            raise RuntimeError("secret internals")

    resp = get_context(request(), Broken(), now=lambda: NOW)
    assert resp["statusCode"] == 500
    assert "internals" not in resp["body"]


# ------------------------------------------------------------------ per-task check

TASK = "WT" + "7" * 32
OTHER_TASK = "WT" + "8" * 32
OTHER_WORKER = "WK" + "9" * 32
KEY, SECRET = "SK" + "k" * 32, "s3cr3t-api-secret"
TOKEN = "flex-token-value"


class FakeTaskRouter:
    """Tasks by SID (attributes) and reservations by task SID; ``fail`` makes every read raise."""

    def __init__(self, tasks=None, reservations=None, fail: Exception | None = None) -> None:
        self.tasks = tasks if tasks is not None else {TASK: {"call_sid": SID, "journal_ref": REF}}
        self.res = reservations if reservations is not None else {
            TASK: [{"worker_sid": WORKER, "status": "accepted"}]}
        self.fail = fail
        self.reads: list[str] = []

    def task_attributes(self, task_sid, *, timeout_s=None):
        self.reads.append("task")
        if self.fail:
            raise self.fail
        return self.tasks.get(task_sid)

    def reservations(self, task_sid, *, timeout_s=None):
        self.reads.append("reservations")
        if self.fail:
            raise self.fail
        return self.res.get(task_sid, [])


def guarded(store, tr, *, task_sid: str | None = TASK, roles: str | None = "agent", **kw):
    req = request(**kw)
    req["queryStringParameters"] = {"call_sid": SID, **({"task_sid": task_sid} if task_sid else {})}
    req["requestContext"]["authorizer"]["roles"] = roles
    resp = get_context(req, store, now=lambda: NOW, taskrouter=tr)
    return resp["statusCode"], (json.loads(resp["body"]) if resp["body"] else {}), resp


def test_worker_with_a_reservation_sees_the_timeline(store) -> None:
    status, body, _ = guarded(store, FakeTaskRouter())
    assert status == 200 and body["timeline"]


@pytest.mark.parametrize("res_status", ["pending", "accepted", "wrapping"])
def test_held_reservation_statuses_allow(store, res_status) -> None:
    tr = FakeTaskRouter(reservations={TASK: [{"worker_sid": WORKER, "status": res_status}]})
    assert guarded(store, tr)[0] == 200


@pytest.mark.parametrize("res_status", ["rejected", "timeout", "canceled", "rescinded", "completed", None])
def test_other_reservation_statuses_deny(store, res_status) -> None:
    tr = FakeTaskRouter(reservations={TASK: [{"worker_sid": WORKER, "status": res_status}]})
    status, body, _ = guarded(store, tr)
    assert status == 404 and body == {"error": "not_found"}


def test_worker_without_a_reservation_gets_the_same_404(store) -> None:
    tr = FakeTaskRouter(reservations={TASK: [{"worker_sid": OTHER_WORKER, "status": "accepted"}]})
    denied = guarded(store, tr)
    unknown = call(store, call_sid=OTHER_SID)
    assert denied[0] == 404 and denied[1] == unknown[1] == {"error": "not_found"}
    assert denied[2]["body"] == json.dumps({"error": "not_found"})


@pytest.mark.parametrize("roles", ["supervisor", "admin", "agent,supervisor", " admin "])
def test_supervisor_and_admin_roles_allow_without_a_reservation(store, roles) -> None:
    tr = FakeTaskRouter(reservations={})
    assert guarded(store, tr, roles=roles)[0] == 200


@pytest.mark.parametrize("roles", [None, "", "agent", "Supervisor", "supervisors", "super visor"])
def test_other_roles_do_not_allow_without_a_reservation(store, roles) -> None:
    assert guarded(store, FakeTaskRouter(reservations={}), roles=roles)[0] == 404


def test_privileged_role_still_needs_matching_task_attributes(store) -> None:
    for attrs in ({"call_sid": OTHER_SID, "journal_ref": REF}, {"call_sid": SID, "journal_ref": "jr_x"},
                  {"call_sid": SID}, {"journal_ref": REF}, {}):
        tr = FakeTaskRouter(tasks={TASK: attrs})
        assert guarded(store, tr, roles="admin")[0] == 404


def test_task_attributes_must_be_strings_that_match(store) -> None:
    for attrs in ({"call_sid": [SID], "journal_ref": REF}, {"call_sid": SID, "journal_ref": 5}):
        assert guarded(store, FakeTaskRouter(tasks={TASK: attrs}))[0] == 404


def test_unknown_task_is_404(store) -> None:
    assert guarded(store, FakeTaskRouter(), task_sid=OTHER_TASK)[0] == 404


@pytest.mark.parametrize("task_sid", [None, "", "WT123", "CA" + "7" * 32, TASK + "/../x", TASK.lower()])
def test_missing_or_malformed_task_sid_is_404_without_a_lookup(store, task_sid) -> None:
    tr = FakeTaskRouter()
    status, body, _ = guarded(store, tr, task_sid=task_sid)
    assert status == 404 and body == {"error": "not_found"}
    assert tr.reads == []


@pytest.mark.parametrize("exc", [TaskRouterUnavailable("down"), RuntimeError("boom"), TimeoutError()])
def test_taskrouter_failure_is_503_with_no_data(store, exc) -> None:
    status, body, resp = guarded(store, FakeTaskRouter(fail=exc))
    assert status == 503 and body == {"error": "unavailable"}
    for leaked in ("timeline", SID, REF, "boom"):
        assert leaked not in resp["body"]


def test_ref_mismatch_never_reaches_taskrouter(store) -> None:
    tr = FakeTaskRouter()
    assert guarded(store, tr, ref="jr_other")[0] == 404
    assert tr.reads == []


def test_timeline_allow_list_is_unchanged() -> None:
    """Pinned; adds only the ``conversation_closed`` type, shown with its ``closed_by``."""
    assert ctx.TIMELINE_FIELDS == ("seq", "ts", "type", "prompt_key", "outcome", "tool", "factor",
                                   "level", "reason", "queue", "answer")
    assert {"status", "transcription", "sweep"} == ctx.CLOSED_BY_VALUES
    assert frozenset({
        "call_start", "lookup", "prompt", "factor_asked", "verification", "tool_call", "tool_result",
        "handoff", "escalation", "call_end", "confirm_asked", "confirm_answer",
        "conversation_closed"}) == ctx.TIMELINE_TYPES


def test_parse_roles() -> None:
    assert parse_roles("agent, supervisor,,") == ("agent", "supervisor")
    assert parse_roles(None) == () and parse_roles(5) == () and parse_roles("") == ()


def test_authorize_task_direct() -> None:
    tr = FakeTaskRouter()
    assert authorize_task(WORKER, (), TASK, SID, REF, taskrouter=tr) is True
    assert authorize_task(OTHER_WORKER, (), TASK, SID, REF, taskrouter=tr) is False
    assert authorize_task(OTHER_WORKER, ("admin",), TASK, SID, REF, taskrouter=tr) is True
    with pytest.raises(TaskRouterUnavailable):
        authorize_task(WORKER, (), TASK, SID, REF, taskrouter=FakeTaskRouter(fail=OSError("x")))


class SlowTaskRouter(FakeTaskRouter):
    """Each read takes ``costs[i]`` seconds of a fake clock (capped at the timeout it was given)."""

    def __init__(self, clock: list[float], costs: list[float]) -> None:
        super().__init__()
        self.clock, self.costs, self.timeouts = clock, list(costs), []

    def _spend(self, timeout_s):
        self.timeouts.append(timeout_s)
        self.clock[0] += min(self.costs.pop(0), timeout_s)

    def task_attributes(self, task_sid, *, timeout_s=None):
        self._spend(timeout_s)
        return super().task_attributes(task_sid)

    def reservations(self, task_sid, *, timeout_s=None):
        self._spend(timeout_s)
        return super().reservations(task_sid)


@pytest.mark.parametrize("first", [0.0, 0.4, 1.0, 2.5, 2.99])
def test_both_reads_share_one_deadline(first) -> None:
    clock = [100.0]
    tr = SlowTaskRouter(clock, [first, 10.0])  # the second read would hang: it gets the rest only
    assert authorize_task(WORKER, (), TASK, SID, REF, taskrouter=tr, clock=lambda: clock[0]) is True
    t_task, t_res = tr.timeouts
    assert t_task == ctx.TASKROUTER_TIMEOUT_S == 3.0
    assert t_res == pytest.approx(3.0 - first)
    assert clock[0] - 100.0 == pytest.approx(ctx.TASKROUTER_TIMEOUT_S)  # never more in total


@pytest.mark.parametrize("first", [3.0, 5.0])
def test_no_time_left_for_the_second_read_is_unavailable(first) -> None:
    clock = [100.0]
    tr = SlowTaskRouter(clock, [first])
    with pytest.raises(TaskRouterUnavailable):
        authorize_task(WORKER, (), TASK, SID, REF, taskrouter=tr, clock=lambda: clock[0])
    assert tr.reads == ["task"]  # the reservations read is never started
    assert clock[0] - 100.0 <= ctx.TASKROUTER_TIMEOUT_S


def test_privileged_role_needs_no_time_for_the_second_read() -> None:
    clock = [100.0]
    tr = SlowTaskRouter(clock, [3.0])
    assert authorize_task(WORKER, ("supervisor",), TASK, SID, REF, taskrouter=tr,
                          clock=lambda: clock[0]) is True


def test_reader_refuses_a_spent_timeout_without_a_request() -> None:
    def opener(req, timeout):
        raise AssertionError("no request expected")

    for spent in (0.0, -1.0, float("nan")):
        with pytest.raises(TaskRouterUnavailable):
            reader(opener).reservations(TASK, timeout_s=spent)


def test_reader_passes_the_given_timeout_to_the_opener() -> None:
    seen = []

    def opener(req, timeout):
        seen.append(timeout)
        return Resp({"reservations": [], "meta": {}})

    reader(opener).reservations(TASK, timeout_s=0.7)
    reader(opener).reservations(TASK)
    assert seen == [0.7, taskrouter.DEFAULT_TIMEOUT_S]


# ------------------------------------------------------------------ access log


def access_records(caplog):
    return [r for r in caplog.records if r.getMessage() == "context api: access"]


@pytest.mark.parametrize("kw,decision", [
    ({}, "served"),
    ({"roles": "agent", "task_sid": OTHER_TASK}, "return_404_task_denied"),
    ({"ref": "jr_other"}, "return_404_no_match"),
])
def test_one_access_line_per_read(store, caplog, kw, decision) -> None:
    caplog.set_level(logging.INFO)
    guarded(store, FakeTaskRouter(), **kw)
    (rec,) = access_records(caplog)
    assert rec.decision == decision and rec.worker_sid == WORKER and rec.call_sid == SID
    assert rec.journal_ref == kw.get("ref", REF)


def test_access_line_on_503_and_no_content_or_secrets_in_logs(store, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    guarded(store, FakeTaskRouter(fail=TaskRouterUnavailable("down")))
    (rec,) = access_records(caplog)
    assert rec.decision == "return_503_taskrouter_unavailable"
    guarded(store, FakeTaskRouter())
    text = "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    for forbidden in ("Please enter your ZIP", ZIP, SSN4, TOKEN, SECRET, "my card was stolen"):
        assert forbidden not in text


# ------------------------------------------------------------------ the real reader (fake opener)


class Resp:
    def __init__(self, body, status: int = 200) -> None:
        self.status, self._body = status, json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._body


WS = "WS" + "5" * 32


def reader(opener, ws=WS, creds=lambda: (KEY, SECRET)):
    return TwilioTaskRouterReader(ws, creds, opener=opener)


def test_reader_requests_are_https_basic_and_short(monkeypatch) -> None:
    seen = []

    def opener(req, timeout):
        seen.append((req.full_url, req.get_header("Authorization"), timeout, req.get_method()))
        if req.full_url.endswith(TASK):
            return Resp({"attributes": json.dumps({"call_sid": SID, "journal_ref": REF})})
        return Resp({"reservations": [{"worker_sid": WORKER, "reservation_status": "accepted"}],
                     "meta": {"next_page_url": None}})

    r = reader(opener)
    assert r.task_attributes(TASK) == {"call_sid": SID, "journal_ref": REF}
    assert r.reservations(TASK) == [{"worker_sid": WORKER, "status": "accepted"}]
    assert all(u.startswith("https://taskrouter.twilio.com/v1/Workspaces/" + WS + "/Tasks/" + TASK)
               and a.startswith("Basic ") and t <= 5 and m == "GET" for u, a, t, m in seen)


def test_reader_404_task_is_none() -> None:
    def opener(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)

    assert reader(opener).task_attributes(TASK) is None


@pytest.mark.parametrize("make", [
    lambda req, t: (_ for _ in ()).throw(urllib.error.HTTPError(req.full_url, 500, "x", {}, None)),
    lambda req, t: (_ for _ in ()).throw(urllib.error.HTTPError(req.full_url, 401, "x", {}, None)),
    lambda req, t: (_ for _ in ()).throw(TimeoutError()),
    lambda req, t: Resp({"attributes": "not json"}),
    lambda req, t: Resp([1]),
    lambda req, t: Resp({"attributes": "[1]"}),
])
def test_reader_failures_are_unavailable(make) -> None:
    with pytest.raises(TaskRouterUnavailable) as err:
        reader(make).task_attributes(TASK)
    assert KEY not in str(err.value) and SECRET not in str(err.value)


def test_reader_incomplete_reservation_list_is_unavailable() -> None:
    page = Resp({"reservations": [], "meta": {"next_page_url": "https://x/next"}})
    with pytest.raises(TaskRouterUnavailable):
        reader(lambda req, t: page).reservations(TASK)


@pytest.mark.parametrize("ws", [None, "", "WS123", "AC" + "5" * 32])
def test_reader_without_a_workspace_is_unavailable(ws) -> None:
    with pytest.raises(TaskRouterUnavailable):
        reader(lambda req, t: Resp({}), ws=ws).task_attributes(TASK)


def test_reader_bad_task_sid_makes_no_request() -> None:
    def opener(req, timeout):
        raise AssertionError("no request expected")

    with pytest.raises(TaskRouterUnavailable):
        reader(opener).task_attributes("WT/../../x")


def test_reader_credential_failure_is_unavailable_without_the_message() -> None:
    def creds():
        raise RuntimeError("arn:aws:secret-details")

    with pytest.raises(TaskRouterUnavailable) as err:
        reader(lambda req, t: Resp({}), creds=creds).task_attributes(TASK)
    assert "arn:aws" not in str(err.value)


def test_non_https_url_is_refused() -> None:
    with pytest.raises(TaskRouterUnavailable):
        taskrouter._require_https("http://taskrouter.twilio.com/v1")
    with pytest.raises(TaskRouterUnavailable):
        taskrouter._require_https("file:///etc/passwd")


def test_handler_without_configuration_is_503_not_data(store, monkeypatch) -> None:
    monkeypatch.delenv("OWLBANK_TASKROUTER_WORKSPACE_SID", raising=False)
    monkeypatch.setattr(ctx, "_store", store)
    monkeypatch.setattr(ctx, "_pii", None)
    monkeypatch.setattr(ctx, "_redactor", FakePiiRedactor)
    callstate.create(store, CallState(call_sid=OTHER_SID, expires_at=4_000_000_000,  # real clock
                                      escalation={"reason": "complaint", "journal_ref": REF}))
    req = request()
    req["queryStringParameters"] = {"call_sid": OTHER_SID, "task_sid": TASK}
    req["requestContext"]["authorizer"]["roles"] = "admin"
    resp = ctx.handler(req, None)
    assert resp["statusCode"] == 503
    assert "timeline" not in resp["body"]
