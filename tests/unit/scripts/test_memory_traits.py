"""scripts/_memory_traits.py against a Memory stub: group present, created with a poll, poll timeout,
empty trait list on a present group, lookup, PATCH with null. No network. No PBT (I/O against a stub)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from owlbank.web_activity import GROUP, TRAITS
from scripts import _memory_traits as mt
from scripts._common import ScriptError

STORE = "mem_store_" + "1" * 26
SECRET = "memory-stub-SECRET-0123"
OP_PATH = "/v1/ControlPlane/Operations/op_1"


class MemoryStub:
    """The ``http_client`` of a Twilio client, answering the Memory calls the helpers make.

    ``group``: None (absent) or the group's trait map as reads show it. ``ops``: the ``status`` values
    the operation reads return in turn (the last repeats). ``visible_after``: reads of the group 404 until
    that many operation reads were made (0: readable at once after the POST)."""

    def __init__(self, *, group: dict[str, Any] | None = None, ops: tuple[str, ...] = ("RUNNING", "COMPLETED"),
                 visible_after: int = 0, status_url: str | None = OP_PATH, post_status: int = 202,
                 profiles: list[str] | None = None, patch_status: int = 202) -> None:
        self.group, self.ops, self.visible_after = group, list(ops), visible_after
        self.status_url, self.post_status, self.profiles = status_url, post_status, profiles or []
        self.patch_status = patch_status
        self.pending = False
        self.op_reads = 0
        self.log: list[tuple[str, str, Any]] = []
        self.auths: list[Any] = []
        self.username, self.password = "SKkey", SECRET
        self.http_client = SimpleNamespace(request=self.request)

    def calls(self, method: str) -> list[tuple[str, str, Any]]:
        return [c for c in self.log if c[0] == method]

    def request(self, method: str, url: str, params: Any = None, data: Any = None, headers: Any = None,
                auth: Any = None, timeout: Any = None, allow_redirects: bool = False) -> Any:
        self.log.append((method, url, data))
        self.auths.append(auth)
        assert headers["Content-Type"] == "application/json" and timeout and not allow_redirects
        if url.endswith(f"/TraitGroups/{GROUP}") and method == "GET":
            if self.group is None or (self.pending and self.op_reads < self.visible_after):
                return _resp(404, {"code": 520047, "message": "no such group"})
            return _resp(200, {"displayName": GROUP, "traits": self.group})
        if method == "POST" and url.endswith("/TraitGroups"):
            self.group, self.pending = {}, True  # created, traits not listed yet
            return _resp(self.post_status, {"statusUrl": self.status_url, "message": "accepted"})
        if method == "GET" and url.endswith(OP_PATH):
            status = self.ops.pop(0) if len(self.ops) > 1 else self.ops[0]
            self.op_reads += 1
            return _resp(200, {"operationId": "op_1", "status": status})
        if method == "POST" and url.endswith("/Profiles/Lookup"):
            return _resp(200, {"normalizedValue": data["value"], "profiles": self.profiles})
        if method == "PATCH":
            return _resp(self.patch_status, {"message": "accepted for processing"})
        raise AssertionError(f"unexpected {method} {url}")


def _resp(status: int, doc: Any) -> Any:
    return SimpleNamespace(status_code=status, text=json.dumps(doc))


class Clock:
    def __init__(self) -> None:
        self.t, self.sleeps = 0.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def ensure(stub: MemoryStub, clock: Clock | None = None, **kw: Any) -> tuple[bool, list[str]]:
    clock = clock or Clock()
    lines: list[str] = []
    made = mt.ensure_trait_group(stub, STORE, log=lines.append, sleep=clock.sleep, clock=clock.now, **kw)
    return made, lines


def test_group_present_is_read_and_left_alone() -> None:
    stub = MemoryStub(group={n: {"dataType": t} for n, t in TRAITS.items()})
    made, lines = ensure(stub)
    assert made is False and lines == [f"ok      trait group {GROUP}"]
    assert [c[0] for c in stub.log] == ["GET"]


def test_empty_trait_list_on_a_present_group_is_not_missing() -> None:
    stub = MemoryStub(group={})
    made, _ = ensure(stub)
    assert made is False and stub.calls("POST") == []


def test_absent_group_is_created_with_exactly_the_traits_then_polled() -> None:
    stub = MemoryStub(visible_after=2)
    clock = Clock()
    made, lines = ensure(stub, clock)
    assert made is True and lines == [f"create  trait group {GROUP}"]
    (_, url, body), = stub.calls("POST")
    assert url == f"{mt.MEMORY_URL}/v1/ControlPlane/Stores/{STORE}/TraitGroups"
    assert body["displayName"] == GROUP
    assert body["traits"] == {n: {"dataType": t} for n, t in TRAITS.items()}
    assert stub.op_reads == 2 and clock.sleeps == [mt.GROUP_POLL_S]


def test_create_polls_the_operation_status_url_with_the_api_key_auth() -> None:
    stub = MemoryStub(status_url=mt.MEMORY_URL + OP_PATH)
    ensure(stub)
    assert any(c[1] == mt.MEMORY_URL + OP_PATH for c in stub.calls("GET"))
    assert set(stub.auths) == {("SKkey", SECRET)}


def test_group_readable_settles_it_even_if_the_operation_never_says_done() -> None:
    stub = MemoryStub(ops=("RUNNING",))
    made, _ = ensure(stub)
    assert made is True and stub.op_reads == 1


def test_poll_timeout_is_bounded_and_says_how_to_recover() -> None:
    stub = MemoryStub(ops=("RUNNING",), visible_after=10**6)
    clock = Clock()
    with pytest.raises(ScriptError, match="not readable after 10 s"):
        ensure(stub, clock, timeout_s=10, poll_s=2)
    assert clock.t >= 10 and len(clock.sleeps) <= 6


@pytest.mark.parametrize("final", ["FAILED", "CANCELLED"])
def test_failed_operation_is_an_error(final: str) -> None:
    with pytest.raises(ScriptError, match="could not create"):
        ensure(MemoryStub(ops=("PENDING", final), visible_after=10**6))


def test_operation_url_outside_memory_is_not_followed() -> None:
    stub = MemoryStub(status_url="https://evil.example/v1/ControlPlane/Operations/op_1")
    with pytest.raises(ScriptError, match="outside memory.twilio.com"):
        ensure(stub)
    assert all("evil" not in c[1] for c in stub.log)


def test_http_error_carries_status_and_code_but_no_body_or_secret() -> None:
    stub = MemoryStub()
    real = stub.request

    def boom(method: str, url: str, **kw: Any) -> Any:
        if method == "POST":
            return _resp(403, {"code": 20003, "message": f"leak {SECRET}"})
        return real(method, url, **kw)

    stub.http_client = SimpleNamespace(request=boom)
    with pytest.raises(mt.MemoryApiError) as err:
        ensure(stub)
    assert err.value.status == 403 and err.value.code == 20003
    assert SECRET not in str(err.value) and "leak" not in str(err.value)


def test_lookup_profile_by_email() -> None:
    stub = MemoryStub(profiles=["mem_profile_a"])
    assert mt.lookup_profile(stub, STORE, "a@example.com") == ["mem_profile_a"]
    (_, url, body), = stub.calls("POST")
    assert url.endswith(f"/v1/Stores/{STORE}/Profiles/Lookup")
    assert body == {"idType": "email", "value": "a@example.com"}
    assert mt.lookup_profile(MemoryStub(), STORE, "none@example.com") == []


def test_patch_sends_null_to_remove_a_trait() -> None:
    stub = MemoryStub()
    mt.patch_traits(stub, STORE, "mem_profile_a", {GROUP: {"viewCount": None, "lastViewedProduct": "HELOC"}})
    (method, url, body), = stub.calls("PATCH")
    assert url.endswith(f"/v1/Stores/{STORE}/Profiles/mem_profile_a")
    assert body == {"traits": {GROUP: {"viewCount": None, "lastViewedProduct": "HELOC"}}}


def test_patch_308_is_raised_with_its_status_for_the_caller_to_relookup() -> None:
    with pytest.raises(mt.MemoryApiError) as err:
        mt.patch_traits(MemoryStub(patch_status=308), STORE, "mem_profile_a", {GROUP: {"viewCount": 1}})
    assert err.value.status == 308


def test_patch_on_a_missing_profile_is_an_error_not_a_silent_success() -> None:
    with pytest.raises(mt.MemoryApiError) as err:
        mt.patch_traits(MemoryStub(patch_status=404), STORE, "mem_profile_gone", {GROUP: {"viewCount": 1}})
    assert err.value.status == 404


def test_create_answering_404_fails_at_once_without_polling() -> None:
    stub, clock = MemoryStub(post_status=404), Clock()
    with pytest.raises(mt.MemoryApiError) as err:
        ensure(stub, clock)
    assert err.value.status == 404 and clock.sleeps == []
