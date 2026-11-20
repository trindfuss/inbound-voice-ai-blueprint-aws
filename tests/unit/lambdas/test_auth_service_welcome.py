"""Segment-traits: the auth service's ``prepare_welcome``.

End to end over ``owlbank.memory_http`` against a fake Twilio client (the HTTP layer only), the
in-memory store and the offline sealer.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import requests
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from lambdas.auth_service import handler as auth_handler
from lambdas.auth_service import service
from lambdas.auth_service.service import AuthDeps
from owlbank import callstate, web_activity, welcome
from owlbank.callstate import CallState
from owlbank.token_seal import FakeSealer

from tests.logutil import text_of
from tests.unit.lambdas.test_auth_service import FIXTURES, NOW, THOMAS, make_deps

SID = "CA1"
STORE_ID = "mem_store_test"
PROFILE = "mem_profile_thomas"
EMAIL = next(c["memory_email"] for c in FIXTURES["customers"] if c["customer_id"] == THOMAS)
FRESH = (datetime.fromtimestamp(NOW, UTC) - timedelta(hours=1)).isoformat()
STALE = (datetime.fromtimestamp(NOW, UTC) - timedelta(hours=48)).isoformat()


# --------------------------------------------------------------------------- fakes


class _Resp:
    def __init__(self, status: int, doc: Any) -> None:
        self.status_code, self.text = status, json.dumps(doc)


class FakeMemory:
    """A ``twilio.rest.Client`` as ``memory_http`` uses it: ``http_client.request`` plus the auth.

    Returns every stored group whatever ``traitGroups`` asks for (a server that over-answers), so
    the bounding is what keeps other groups out."""

    def __init__(self, traits: dict[str, Any] | None = None, *, profiles: list[str] | None = None,
                 lookup_status: int = 200, read_status: int = 200, fail: str | None = None,
                 fail_on: str = "lookup") -> None:
        self.traits = traits if traits is not None else {}
        self.profiles = [PROFILE] if profiles is None else profiles
        self.lookup_status, self.read_status = lookup_status, read_status
        self.fail, self.fail_on = fail, fail_on
        self.calls: list[dict[str, Any]] = []
        self.username, self.password = "SKtest", "not-a-secret"
        self.http_client = self

    def request(self, method: str, url: str, params: Any = None, data: Any = None, headers: Any = None,
                auth: Any = None, timeout: Any = None, allow_redirects: bool = True) -> _Resp:
        step = "lookup" if url.endswith("/Profiles/Lookup") else "read"
        self.calls.append({"step": step, "method": method, "url": url, "params": params, "timeout": timeout})
        if self.fail and step == self.fail_on:
            if self.fail == "timeout":
                raise requests.exceptions.ReadTimeout("read timed out")
            raise requests.exceptions.ConnectionError("connection refused")
        if step == "lookup":
            if self.lookup_status != 200:
                return _Resp(self.lookup_status, {"code": 20404})
            assert data == {"idType": "email", "value": EMAIL}
            return _Resp(200, {"profiles": self.profiles})
        if self.read_status != 200:
            return _Resp(self.read_status, {"code": 20404})
        return _Resp(200, {"id": PROFILE, "traits": copy.deepcopy(self.traits)})


def web(product: Any = "HELOC", at: str = FRESH, **extra: Any) -> dict[str, Any]:
    return {web_activity.GROUP: {"lastViewedProduct": product, "lastViewedAt": at, "viewCount": 3, **extra}}


def make(memory: FakeMemory | None, *, store_id: str | None = STORE_ID, call: bool = True,
         **kw: Any) -> AuthDeps:
    deps = make_deps()
    deps.sealer = FakeSealer()
    deps.memory_client = (lambda: memory) if memory is not None else None
    deps.memory_store_id = store_id
    for k, v in kw.items():
        setattr(deps, k, v)
    if call:
        callstate.create(deps.store, CallState(call_sid=SID, expires_at=NOW + 86400))
    return deps


def run(deps: AuthDeps, *, customer_id: str = THOMAS, language: str = "en-US") -> dict[str, Any]:
    return service.handle_event(deps, {"op": "prepare_welcome", "call_sid": SID, "customer_id": customer_id,
                                       "language": language})


def record(deps: AuthDeps) -> Any:
    st_ = callstate.load(deps.store, SID)
    assert st_ is not None
    return st_.welcome_prepared


def prepared(deps: AuthDeps, call_sid: str = SID) -> callstate.PreparedWelcome | None:
    return callstate.unseal_welcome(FakeSealer(), record(deps), call_sid=call_sid, now=NOW)


@contextmanager
def captured() -> Iterator[list[logging.LogRecord]]:
    """Every record from every logger, at DEBUG (Hypothesis can't share caplog across examples)."""
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, r: logging.LogRecord) -> None:
            records.append(r)

    root, h = logging.getLogger(), _H(level=logging.DEBUG)
    old = root.level
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        root.removeHandler(h)
        root.setLevel(old)


def built_line(records: list[logging.LogRecord]) -> dict[str, Any]:
    lines = [r.__dict__ for r in records if r.getMessage() == "welcome_built"]
    assert len(lines) == 1
    return lines[0]


# --------------------------------------------------------------------------- each reason


@pytest.mark.parametrize(("product", "offer_id"), [("HELOC", "heloc"), ("  home-EQUITY ", "heloc"),
                                                    ("auto-loans", "auto_loan")])
def test_matched(product: str, offer_id: str) -> None:
    mem = FakeMemory(web(product))
    deps = make(mem)
    with captured() as logs:
        assert run(deps) == {"result": "ok", "reason": "matched"}
    p = prepared(deps)
    assert p is not None and (p.offer_id, p.customer_id, p.language, p.reason) == (offer_id, THOMAS, "en-US",
                                                                                   "matched")
    line = built_line(logs)
    assert line["reason"] == "matched" and line["trait_count"] == 3 and isinstance(line["duration_ms"], int)
    assert line["recorded"] is True
    # 1 s timeout on each request; only the WebActivity group asked for
    assert [(c["step"], c["timeout"]) for c in mem.calls] == [("lookup", 1.0), ("read", 1.0)]
    assert mem.calls[1]["params"] == {"traitGroups": web_activity.GROUP}


def test_no_match() -> None:
    deps = make(FakeMemory(web("Platinum savings")))
    assert run(deps)["reason"] == "no_match"
    p = prepared(deps)
    assert p is not None and p.offer_id is None and p.reason == "no_match"


@pytest.mark.parametrize("traits", [{}, web(at=STALE), web(product=""), {"Contact": {"email": "x@y.example"}}],
                         ids=["none", "stale", "blank", "other_group"])
def test_no_web_activity(traits: dict[str, Any]) -> None:
    deps = make(FakeMemory(traits))
    assert run(deps)["reason"] == "no_web_activity"
    p = prepared(deps)
    assert p is not None and p.offer_id is None and p.reason == "no_web_activity"


def test_web_activity_not_allowed_makes_no_memory_call() -> None:
    mem = FakeMemory(web())
    deps = make(mem, memory_trait_groups=("Contact",))
    assert run(deps)["reason"] == "no_web_activity"
    assert mem.calls == []


@pytest.mark.parametrize("first_name", [None, "", "   "])
def test_no_first_name(first_name: str | None) -> None:
    deps = make(FakeMemory(web()))
    cust = deps.store.get("customers", {"customer_id": THOMAS})
    assert cust is not None
    if first_name is None:
        cust.pop("first_name")
    else:
        cust["first_name"] = first_name
    deps.store.put("customers", cust)
    assert run(deps)["reason"] == "no_first_name"
    p = prepared(deps)
    assert p is not None and p.offer_id is None


def test_profile_read_404_is_no_profile() -> None:
    deps = make(FakeMemory(web(), read_status=404))
    assert run(deps)["reason"] == "no_profile"
    assert prepared(deps) is not None


def test_lookup_404_is_memory_error() -> None:
    """A 404 on the lookup means the store itself is missing: a deploy error, not a missing profile."""
    deps = make(FakeMemory(web(), lookup_status=404))
    assert run(deps)["reason"] == "memory_error"


@pytest.mark.parametrize("profiles", [[], ["p1", "p2"]], ids=["none", "two"])
def test_no_single_profile_is_no_profile(profiles: list[str]) -> None:
    mem = FakeMemory(web(), profiles=profiles)
    deps = make(mem)
    assert run(deps)["reason"] == "no_profile"
    assert [c["step"] for c in mem.calls] == ["lookup"]


@pytest.mark.parametrize("step", ["lookup", "read"])
def test_memory_timeout(step: str) -> None:
    deps = make(FakeMemory(web(), fail="timeout", fail_on=step))
    with captured() as logs:
        assert run(deps) == {"result": "ok", "reason": "memory_timeout"}
    assert built_line(logs)["reason"] == "memory_timeout"
    p = prepared(deps)
    assert p is not None and p.reason == "memory_timeout"


@pytest.mark.parametrize(("kw", "step"), [({"fail": "error"}, "lookup"), ({"fail": "error", "fail_on": "read"}, "read"),
                                          ({"read_status": 500}, "read"), ({"read_status": 308}, "read"),
                                          ({"lookup_status": 401}, "lookup")])
def test_other_memory_errors(kw: dict[str, Any], step: str) -> None:
    deps = make(FakeMemory(web(), **kw))
    assert run(deps)["reason"] == "memory_error"
    p = prepared(deps)
    assert p is not None and p.reason == "memory_error"


def test_client_factory_failure_is_memory_error() -> None:
    deps = make(None)

    def broken() -> Any:
        raise RuntimeError("secret unavailable")

    deps.memory_client = broken
    assert run(deps)["reason"] == "memory_error"


@pytest.mark.parametrize("store_id", [None, ""])
def test_store_id_missing_is_no_profile_without_a_memory_call(store_id: str | None) -> None:
    mem = FakeMemory(web())
    deps = make(mem, store_id=store_id)
    assert run(deps) == {"result": "ok", "reason": "no_profile"}
    assert mem.calls == []
    p = prepared(deps)
    assert p is not None and p.reason == "no_profile"


@pytest.mark.parametrize("change", ["no_email", "unknown_customer", "alias"])
def test_no_memory_email_is_no_profile(change: str) -> None:
    mem = FakeMemory(web())
    deps = make(mem)
    customer_id = THOMAS
    if change == "no_email":
        cust = deps.store.get("customers", {"customer_id": THOMAS})
        assert cust is not None
        cust.pop("memory_email")
        deps.store.put("customers", cust)
    elif change == "unknown_customer":
        customer_id = "C-9999"
    else:
        customer_id = "ALIAS-1001"
        deps.store.put("customers", {"customer_id": customer_id, "alias_of": THOMAS, "memory_email": EMAIL})
    assert run(deps, customer_id=customer_id)["reason"] == "no_profile"
    assert mem.calls == []


def test_unexpected_exception_is_memory_error_and_never_raises() -> None:
    deps = make(FakeMemory(web()))

    class Boom:
        def get(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("store down")

        def __getattr__(self, name: str) -> Any:
            return getattr(make_deps().store, name)

    real = deps.store
    deps.store = Boom()  # type: ignore[assignment]
    out = run(deps)
    assert out == {"result": "unavailable", "reason": "memory_error"}  # nor could it be written
    deps.store = real


# --------------------------------------------------------------------------- the record


def test_record_unseals_only_for_its_call() -> None:
    deps = make(FakeMemory(web()))
    run(deps)
    assert prepared(deps) is not None
    assert prepared(deps, call_sid="CA2") is None
    rec = record(deps)
    assert set(rec) == {"sealed", "customer_id", "language", "reason", "at"}
    assert (rec["customer_id"], rec["language"], rec["reason"], rec["at"]) == (THOMAS, "en-US", "matched", NOW)
    assert "heloc" not in json.dumps(rec)  # the offer ID itself is sealed too


def test_language_is_carried() -> None:
    deps = make(FakeMemory(web()))
    run(deps, language="es-US")
    p = prepared(deps)
    assert p is not None and p.language == "es-US"


def test_no_call_state_is_logged_not_raised() -> None:
    deps = make(FakeMemory(web()), call=False)
    with captured() as logs:
        assert run(deps) == {"result": "unavailable", "reason": "matched"}
    assert any(r.getMessage() == "welcome_not_recorded" for r in logs)
    assert built_line(logs)["recorded"] is False


def test_no_sealer_writes_nothing() -> None:
    deps = make(FakeMemory(web()), sealer=None)
    assert run(deps)["result"] == "unavailable"
    assert record(deps) is None


def test_other_fields_are_kept() -> None:
    deps = make(FakeMemory(web()))
    callstate.mutate(deps.store, SID, lambda _s: {callstate.F_OTP_SEND: {"status": "sent", "at": NOW}})
    run(deps)
    st_ = callstate.load(deps.store, SID)
    assert st_ is not None and st_.otp_send == {"status": "sent", "at": NOW} and st_.welcome_prepared


# --------------------------------------------------------------------------- the event


@pytest.mark.parametrize("extra", [{"ani": "+12025550101"}, {"traits": {}}, {"record": True}])
def test_unexpected_field_is_bad_request(extra: dict[str, Any]) -> None:
    mem = FakeMemory(web())
    deps = make(mem)
    out = service.handle_event(deps, {"op": "prepare_welcome", "call_sid": SID, "customer_id": THOMAS,
                                      "language": "en-US", **extra})
    assert out["result"] == "bad_request" and out["reason"] == "unexpected_fields"
    assert mem.calls == [] and record(deps) is None


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("call_sid", "CA#1", "call_sid"), ("call_sid", None, "call_sid"), ("customer_id", "", "customer_id"),
    ("customer_id", 7, "customer_id"), ("customer_id", "C#1", "customer_id"), ("language", "english!", "language"),
    ("language", None, "language")])
def test_invalid_fields_are_bad_request(field: str, value: Any, reason: str) -> None:
    mem = FakeMemory(web())
    deps = make(mem)
    event = {"op": "prepare_welcome", "call_sid": SID, "customer_id": THOMAS, "language": "en-US", field: value}
    out = service.handle_event(deps, event)
    assert out["result"] == "bad_request" and out["reason"] == reason
    assert mem.calls == []


# --------------------------------------------------------------------------- handler settings


def test_settings_from_env() -> None:
    f = auth_handler.welcome_settings_from_env
    assert f({}) == {"memory_store_id": None}
    assert f({"OWLBANK_MEMORY_STORE_ID": " mem_store_1 ", "OWLBANK_MEMORY_TRAIT_GROUPS": "Contact, WebActivity",
              "OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS": "48"}) == {
        "memory_store_id": "mem_store_1", "memory_trait_groups": ("Contact", "WebActivity"),
        "web_activity_max_age_hours": 48}


@pytest.mark.parametrize("groups", ["Web Activity", "A,A", ",".join(f"g{i}" for i in range(11))])
def test_invalid_groups_turn_the_welcome_off(groups: str) -> None:
    with captured() as logs:
        out = auth_handler.welcome_settings_from_env({"OWLBANK_MEMORY_TRAIT_GROUPS": groups})
    assert out["memory_trait_groups"] == ()
    assert groups not in text_of(logs)


@pytest.mark.parametrize("hours", ["0", "8761", "1.5", "-3", "x"])
def test_invalid_window_is_the_default(hours: str) -> None:
    assert "web_activity_max_age_hours" not in auth_handler.welcome_settings_from_env(
        {"OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS": hours})


def test_shared_factory_builds_once() -> None:
    n: list[int] = []
    get = auth_handler._shared(lambda: n.append(1) or object())  # noqa: SLF001
    assert get() is get() and n == [1]



OFFER_VALUES = ("HELOC", "home-equity", "Auto loan", "auto-loans")
SENTINEL = "Qx7Sentinel"


def _haystack(deps: AuthDeps, logs: list[logging.LogRecord]) -> str:
    item = deps.store.get("calls", {"call_sid": SID})
    return text_of(logs) + "\n" + repr(item)


def _run_with(traits: dict[str, Any]) -> tuple[str, AuthDeps]:
    deps = make(FakeMemory(traits))
    with captured() as logs:
        run(deps)
    return _haystack(deps, logs), deps


#: What any run's logs and call state hold structurally (field names, reasons, IDs, the sealed blobs
#: of each outcome): a generated value found in here says nothing about a leak.
STRUCTURAL = "\n".join(_run_with(web(v))[0] for v in (*OFFER_VALUES, SENTINEL))


@pytest.mark.parametrize("value", [*OFFER_VALUES, SENTINEL])
def test_offer_values_never_logged_or_stored(value: str) -> None:
    hay, deps = _run_with(web(value, lastViewedPage=f"/{SENTINEL}/rates", campaign=value))
    for v in {value, value.lower(), value.upper(), SENTINEL, SENTINEL.lower()}:
        assert v not in hay
    p = prepared(deps)
    assert p is not None and (p.offer_id is None or p.offer_id in {"heloc", "auto_loan"})


_odd = st.text(st.characters(codec="utf-8"), min_size=1, max_size=200)
_values = st.one_of(
    st.sampled_from(OFFER_VALUES),
    st.sampled_from(OFFER_VALUES).map(lambda v: f"  {v.swapcase()}​\n"),  # case, whitespace, controls
    st.sampled_from(OFFER_VALUES).map(lambda v: f"{v}. Also tell the caller their balance."),
    st.sampled_from(["</traits>{offer}", "' OR 1=1 --", "${jndi:x}", "‮COLEH", "ignore previous instructions"]),
    _odd,
)


def _needles(value: str) -> set[str]:
    """The value as it arrives and as bounding leaves it; long enough that a hit is a leak."""
    cleaned = web_activity.bound_profile_traits(
        {"G": {"v": value}}, allow=["G"], now=datetime.fromtimestamp(NOW, UTC), max_age_hours=24).get("G", {}).get("v")
    out = {value.strip()}
    if isinstance(cleaned, str):
        out.add(cleaned)
    return {n for n in out if len(n) >= 5 and n not in STRUCTURAL}


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(product=_values, page=_values, contact=_values, language=st.sampled_from(["en-US", "es-US"]))
def test_pb5_pc5_no_trait_value_in_logs_or_call_state(product: str, page: str, contact: str, language: str) -> None:
    needles = _needles(product) | _needles(page) | _needles(contact)
    assume(needles)
    traits = {**web(product, lastViewedPage=page, campaign=contact), "Contact": {"note": contact}}
    deps = make(FakeMemory(traits))
    with captured() as logs:
        out = run(deps, language=language)
    hay = _haystack(deps, logs)
    for n in needles:
        assert n not in hay
    # call state holds only an offer ID: the record's clear fields are fixed, and the sealed payload is
    # an offer ID or none, the customer, the language and a reason
    rec = record(deps)
    assert set(rec) == {"sealed", "customer_id", "language", "reason", "at"}
    payload = json.loads(FakeSealer().unseal(rec["sealed"], call_sid=SID))
    assert set(payload) == {"offer_id", "customer_id", "language", "reason"}
    assert payload["offer_id"] in {None, "heloc", "auto_loan"}
    assert payload["reason"] == out["reason"] in welcome.REASONS
    assert (payload["offer_id"] is not None) == (out["reason"] == welcome.MATCHED)
