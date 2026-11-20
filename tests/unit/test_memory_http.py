"""owlbank.memory_http against a stub HTTP client: the lookup, the profile read with and without
``groups``, a 404 and a 308 raising, and the scripts' re-export raising their ScriptError subclass.
No network. No PBT (I/O against a stub)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from owlbank import memory_http as mh

STORE = "mem_store_" + "2" * 26
SECRET = "memory-http-SECRET-42"


class Stub:
    def __init__(self, status: int = 200, doc: Any = None) -> None:
        self.status, self.doc = status, doc
        self.log: list[dict[str, Any]] = []
        self.username, self.password = "SKkey", SECRET
        self.http_client = SimpleNamespace(request=self.request)

    def request(self, method: str, url: str, **kw: Any) -> Any:
        self.log.append({"method": method, "url": url, **kw})
        return SimpleNamespace(status_code=self.status, text=json.dumps(self.doc))


def test_lookup() -> None:
    stub = Stub(doc={"profiles": ["mem_profile_a", 7]})
    assert mh.lookup_profile(stub, STORE, "a@example.com", timeout_s=2) == ["mem_profile_a"]
    (call,) = stub.log
    assert call["method"] == "POST" and call["url"] == f"{mh.MEMORY_URL}/v1/Stores/{STORE}/Profiles/Lookup"
    assert call["data"] == {"idType": "email", "value": "a@example.com"} and call["timeout"] == 2
    assert call["auth"] == ("SKkey", SECRET) and call["allow_redirects"] is False


def test_lookup_by_sends_the_id_type_and_lookup_is_the_email_wrapper() -> None:
    stub = Stub(doc={"profiles": ["mem_profile_a"]})
    assert mh.lookup_profile_by(stub, STORE, "phone", "+15550100") == ["mem_profile_a"]
    (call,) = stub.log
    assert call["method"] == "POST" and call["url"] == f"{mh.MEMORY_URL}/v1/Stores/{STORE}/Profiles/Lookup"
    assert call["data"] == {"idType": "phone", "value": "+15550100"}
    assert mh.lookup_profile_by(Stub(doc={}), STORE, "phone", "+1") == []
    with pytest.raises(mh.MemoryApiError):
        mh.lookup_profile_by(Stub(status=500, doc={"code": 1}), STORE, "phone", "+1")


def test_read_every_group_by_default() -> None:
    stub = Stub(doc={"traits": {"Contact": {"email": "a@example.com"}}})
    assert mh.get_profile_traits(stub, STORE, "mem_profile_a") == {"Contact": {"email": "a@example.com"}}
    (call,) = stub.log
    assert call["method"] == "GET" and call["url"].endswith("/Profiles/mem_profile_a") and call["params"] is None
    assert call["timeout"] == mh.REQUEST_TIMEOUT_S


def test_read_named_groups() -> None:
    stub = Stub(doc={"traits": {"WebActivity": {"viewCount": 1}}})
    mh.get_profile_traits(stub, STORE, "mem_profile_a", ("WebActivity", "Contact"))
    assert stub.log[0]["params"] == {"traitGroups": "WebActivity,Contact"}


@pytest.mark.parametrize("groups", ["WebActivity", ()])
def test_read_refuses_a_str_or_no_groups(groups: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        mh.get_profile_traits(Stub(), STORE, "mem_profile_a", groups)


@pytest.mark.parametrize("status", [404, 308, 403, 500])
def test_errors_raise_with_status_and_code_only(status: int) -> None:
    stub = Stub(status=status, doc={"code": 20404, "message": f"leak {SECRET}"})
    with pytest.raises(mh.MemoryApiError) as err:
        mh.get_profile_traits(stub, STORE, "mem_profile_gone")
    assert err.value.status == status and err.value.code == 20404
    assert SECRET not in str(err.value) and "leak" not in str(err.value)


def test_scripts_reexport_raises_their_script_error() -> None:
    from scripts import _memory_traits as mt
    from scripts._common import ScriptError

    with pytest.raises(mt.MemoryApiError) as err:
        mt.lookup_profile(Stub(status=403, doc={"code": 20003}), STORE, "a@example.com")
    assert isinstance(err.value, ScriptError) and isinstance(err.value, mh.MemoryApiError)
    assert err.value.status == 403 and err.value.code == 20003
