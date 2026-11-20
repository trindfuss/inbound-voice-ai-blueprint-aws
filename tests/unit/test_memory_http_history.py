"""owlbank.memory_http observation and summary calls against a stub HTTP client: paths, bodies,
pagination, ids quoted, and a 404 raising. No network. No PBT (I/O against a stub)."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from owlbank import memory_http as mh

STORE = "mem_store_" + "2" * 26
PID = "mem_profile_" + "a" * 26
BASE = f"{mh.MEMORY_URL}/v1/Stores/{STORE}/Profiles/{PID}"


class Stub:
    def __init__(self, *docs: Any, status: int = 200) -> None:
        self.docs, self.status = list(docs), status
        self.log: list[dict[str, Any]] = []
        self.username, self.password = "SKkey", "secret-9"
        self.http_client = SimpleNamespace(request=self.request)

    def request(self, method: str, url: str, **kw: Any) -> Any:
        self.log.append({"method": method, "url": url, **kw})
        doc = self.docs.pop(0) if len(self.docs) > 1 else (self.docs[0] if self.docs else None)
        return SimpleNamespace(status_code=self.status, text=json.dumps(doc))


def test_create_observations() -> None:
    stub = Stub({"id": "x"}, status=202)
    obs = [{"content": "c", "source": "owlbank-demo", "occurredAt": "2026-10-28T00:00:00Z",
            "conversationIds": ["conv_conversation_a"]}]
    mh.create_observations(stub, STORE, PID, obs, timeout_s=3)
    (call,) = stub.log
    assert call["method"] == "POST" and call["url"] == f"{BASE}/Observations"
    assert call["data"] == {"observations": obs} and call["timeout"] == 3
    assert call["auth"] == ("SKkey", "secret-9")


def test_create_summaries() -> None:
    stub = Stub({"message": "accepted"}, status=202)
    summ = [{"content": "c", "conversationId": "conv_conversation_a", "occurredAt": "t", "source": "s"}]
    mh.create_summaries(stub, STORE, PID, summ)
    (call,) = stub.log
    assert call["method"] == "POST" and call["url"] == f"{BASE}/ConversationSummaries"
    assert call["data"] == {"summaries": summ}


@pytest.mark.parametrize(("fn", "path", "key"), [
    (mh.list_observations, "Observations", "observations"),
    (mh.list_summaries, "ConversationSummaries", "summaries"),
])
def test_list_follows_tokens(fn: Callable[..., Any], path: str, key: str) -> None:
    stub = Stub({key: [{"id": "1"}], "meta": {"nextToken": "tok2"}},
                {key: [{"id": "2"}, "junk"], "meta": {"nextToken": None}})
    assert fn(stub, STORE, PID) == [{"id": "1"}, {"id": "2"}]
    first, second = stub.log
    assert first["method"] == "GET" and first["url"] == f"{BASE}/{path}"
    assert first["params"] == {"pageSize": str(mh.LIST_PAGE_SIZE)}
    assert second["params"] == {"pageSize": str(mh.LIST_PAGE_SIZE), "pageToken": "tok2"}


def test_list_empty_or_odd_bodies() -> None:
    assert mh.list_summaries(Stub({}), STORE, PID) == []
    assert mh.list_observations(Stub(None), STORE, PID) == []


def test_deletes() -> None:
    stub = Stub(None, status=204)
    mh.delete_observation(stub, STORE, PID, "mem_observation_1")
    mh.delete_summary(stub, STORE, PID, "mem_summary/2")
    a, b = stub.log
    assert (a["method"], a["url"]) == ("DELETE", f"{BASE}/Observations/mem_observation_1")
    assert (b["method"], b["url"]) == ("DELETE", f"{BASE}/ConversationSummaries/mem_summary%2F2")


@pytest.mark.parametrize("call", [
    lambda c: mh.create_observations(c, STORE, PID, []),
    lambda c: mh.create_summaries(c, STORE, PID, []),
    lambda c: mh.list_observations(c, STORE, PID),
    lambda c: mh.list_summaries(c, STORE, PID),
    lambda c: mh.delete_observation(c, STORE, PID, "x"),
    lambda c: mh.delete_summary(c, STORE, PID, "x"),
])
def test_404_raises_without_body_or_credential(call: Callable[[Any], Any]) -> None:
    with pytest.raises(mh.MemoryApiError) as exc:
        call(Stub({"code": 20404, "message": "secret-9 leaked?"}, status=404))
    assert exc.value.status == 404 and exc.value.code == 20404
    assert "secret-9" not in str(exc.value) and "leaked" not in str(exc.value)
