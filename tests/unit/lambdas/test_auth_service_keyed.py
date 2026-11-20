"""Auth service: the keyed-in phone ``identity_source``.

A ``keyed_phone`` request is untrusted and never counts ``ani_match``, whatever the verstat and
``ani_trust.on_untrusted`` say (also under ``allow``). ``ani`` (or absent) is today's behaviour;
any other value is ``bad_request``."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest

from lambdas.auth_service import service
from lambdas.auth_service.service import AuthDeps
from owlbank import callstate
from owlbank.callstate import CallState
from owlbank.policy import Policy
from owlbank.store import MemoryStore
from tests.logutil import lines as lines_of
from tests.unit.lambdas.test_auth_service import (
    ANI, FACTORS, POLICY_DOC, THOMAS, TRUSTED, Clock, FakeVerify, seed,
)

KEYED = "keyed_phone"
Z, S = FACTORS[THOMAS]["zip"], FACTORS[THOMAS]["ssn4"]
CODE = "123456"


def make(mode: str = "allow") -> AuthDeps:
    doc = copy.deepcopy(POLICY_DOC)
    doc["ani_trust"]["on_untrusted"] = mode
    store = MemoryStore()
    seed(store)
    return AuthDeps(store=store, policy=Policy.from_dict(doc), verify_client=FakeVerify(CODE),
                    clock=Clock(), token_max_ttl_s=1800, record_ttl_s=86400)


def kverify(d: AuthDeps, level: int, *, verstat: str | None = None, source: Any = KEYED,
            **factors: str) -> dict[str, Any]:
    return service.verify(d, call_sid="CA1", ani=ANI[THOMAS], factors=factors,
                          target_level=level, stir_verstat=verstat, identity_source=source)


def audits(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return lines_of(caplog.records, "owlbank.audit")


# --------------------------------------------------------------------------- under allow


@pytest.mark.parametrize("verstat", [None, TRUSTED])
def test_keyed_zip_alone_is_not_level_1_under_allow(verstat: str | None) -> None:
    d = make("allow")
    out = kverify(d, 1, verstat=verstat, zip=Z)
    assert out == {"result": "bad_request", "level": 0, "reason": "missing_factors",
                   "missing": ["ssn4"]}
    # The same request as an ANI call under ``allow`` and no verstat: the trusted rule, ZIP is enough.
    ani = kverify(make("allow"), 1, verstat=None, source="ani", zip=Z)
    assert ani["result"] == "verified" and "ani_match" in ani["factors_verified"]


@pytest.mark.parametrize("verstat", [None, TRUSTED])
def test_keyed_zip_and_ssn4_reach_level_1_without_ani_match(
        verstat: str | None, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    d = make("allow")
    out = kverify(d, 1, verstat=verstat, zip=Z, ssn4=S)
    assert out["result"] == "verified" and out["level"] == 1 and out["customer_id"] == THOMAS
    assert out["factors_verified"] == ["zip", "ssn4"]
    (issued,) = [a for a in audits(caplog) if a["event"] == "token_issued"]
    assert issued["ani_trusted"] is False and issued["ani_match_counted"] is False


def test_keyed_level2_step_up_under_allow_succeeds_end_to_end(
        caplog: pytest.LogCaptureFixture) -> None:
    """The gap that was fixed: verify asked for ``ssn4`` again (``bad_request``)."""
    caplog.set_level(logging.DEBUG)
    d = make("allow")
    assert service.precheck(d, call_sid="CA1", ani=ANI[THOMAS], target_level=1,
                            identity_source=KEYED) == {"result": "ok"}
    assert kverify(d, 1, zip=Z, ssn4=S)["level"] == 1
    assert service.precheck(d, call_sid="CA1", ani=ANI[THOMAS], target_level=2,
                            identity_source=KEYED) == {"result": "ok"}
    sent = service.send_otp(d, call_sid="CA1", ani=ANI[THOMAS], identity_source=KEYED)
    assert sent == {"result": "sent"}
    assert d.verify_client.sent == [(ANI[THOMAS], "sms")]  # type: ignore[union-attr]
    out = kverify(d, 2, otp=CODE)
    assert out["result"] == "verified" and out["step_up"] is True
    # The untrusted level 2 holds level 3's collected factors, as for any untrusted ANI.
    assert out["level"] == 2 and out["factors_verified"] == ["zip", "ssn4", "otp"]
    rows = audits(caplog)
    assert [a["ani_trusted"] for a in rows if "ani_trusted" in a] and \
        all(a["ani_trusted"] is False for a in rows if "ani_trusted" in a)
    (otp_sent,) = [a for a in rows if a["event"] == "otp_sent"]
    assert otp_sent["ani_trusted"] is False
    # a keyed send is counted (otp_sent_keyed_phone), never the spoofed-ANI alarm.
    security = [r.getMessage() for r in caplog.records if r.name == "owlbank.security"]
    assert security == ["otp_sent_keyed_phone"], security
    assert not any("otp_sent_untrusted_ani" in m for m in security)


def test_keyed_trusted_verstat_is_still_untrusted_at_precheck() -> None:
    """A trusted verstat never short-cuts the gate on a keyed call: no code before login."""
    d = make("allow")
    out = service.precheck(d, call_sid="CA1", ani=ANI[THOMAS], target_level=2,
                           stir_verstat=TRUSTED, identity_source=KEYED)
    assert out == {"result": "escalate", "level": 0, "reason": "verification_risk"}
    out = service.precheck(d, call_sid="CA1", ani=ANI[THOMAS], target_level=2,
                           stir_verstat=TRUSTED, identity_source="ani")
    assert out == {"result": "ok"}


def test_keyed_failure_counts_untrusted_never_the_customer_lockout() -> None:
    d = make("allow")
    out = kverify(d, 1, verstat=TRUSTED, zip="00000", ssn4="0000")
    assert out["result"] == "not_verified"
    keys = [str(k[0]) for k in d.store._tables["calls"]]  # noqa: SLF001
    assert not [k for k in keys if k.startswith("ratelimit#customer#")]
    assert list(d.store._tables["risk"])  # noqa: SLF001  (the untrusted-attempt cap)


# --------------------------------------------------------------------------- the field


BAD_SOURCES = ["", "ANI", "keyed", "Keyed_Phone", "keyed_phone ", None, 0, 1, True, ["ani"],
               {"v": "ani"}]


@pytest.mark.parametrize("source", BAD_SOURCES)
def test_unknown_identity_source_is_bad_request_everywhere(source: Any) -> None:
    d = make("allow")
    bad = {"result": "bad_request", "level": 0, "reason": "identity_source"}
    assert kverify(d, 1, source=source, zip=Z, ssn4=S) == bad
    assert service.precheck(d, call_sid="CA1", ani=ANI[THOMAS], target_level=1,
                            identity_source=source) == bad
    assert kverify(d, 1, source="ani", verstat=TRUSTED, zip=Z)["result"] == "verified"
    assert service.send_otp(d, call_sid="CA1", ani=ANI[THOMAS], identity_source=source) == bad
    assert d.verify_client.sent == []  # type: ignore[union-attr]


def test_unknown_identity_source_on_the_async_send_is_recorded_as_error() -> None:
    d = make("allow")
    callstate.create(d.store, CallState(call_sid="CA1", expires_at=d.clock() + 86400))
    out = service.handle_event(d, {"op": "send_otp", "call_sid": "CA1", "ani": ANI[THOMAS],
                                   "identity_source": "bogus", "record": True})
    assert out["result"] == "bad_request"
    st = callstate.load(d.store, "CA1")
    assert st is not None and st.otp_send is not None and st.otp_send["status"] == "error"


@pytest.mark.parametrize("op, extra", [
    ("verify", {"factors": {"zip": Z, "ssn4": S}, "target_level": 1}),
    ("precheck", {"target_level": 1}),
    ("send_otp", {"language": "en-US"}),
])
def test_handle_event_accepts_identity_source_and_rejects_junk(op: str, extra: dict) -> None:
    base = {"op": op, "call_sid": "CA1", "ani": ANI[THOMAS], "stir_verstat": None, **extra}
    for source in ("ani", KEYED):
        out = service.handle_event(make("allow"), {**base, "identity_source": source})
        assert out["result"] != "bad_request" or out.get("reason") != "identity_source"
        assert out.get("reason") != "unexpected_fields"
    for junk in ("x", None, 3):
        out = service.handle_event(make("allow"), {**base, "identity_source": junk})
        assert out == {"result": "bad_request", "level": 0, "reason": "identity_source"}


def test_handle_event_keyed_verify_under_allow_needs_ssn4() -> None:
    event = {"op": "verify", "call_sid": "CA1", "ani": ANI[THOMAS], "factors": {"zip": Z},
             "target_level": 1, "stir_verstat": None}
    assert service.handle_event(make("allow"), event)["result"] == "verified"  # absent: today
    out = service.handle_event(make("allow"), {**event, "identity_source": KEYED})
    assert out["reason"] == "missing_factors" and out["missing"] == ["ssn4"]


# --------------------------------------------------------------------------- absent == "ani"


def _strip(out: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in out.items() if k != "token"}


@pytest.mark.parametrize("mode", ["allow", "no_ani_match"])
@pytest.mark.parametrize("verstat", [None, TRUSTED, "TN-Validation-Failed-A"])
@pytest.mark.parametrize("factors", [{"zip": Z}, {"zip": Z, "ssn4": S}, {"zip": "00000"}])
def test_absent_and_ani_are_identical(mode: str, verstat: str | None,
                                      factors: dict[str, str]) -> None:
    runs = []
    for kw in ({}, {"identity_source": "ani"}):
        d = make(mode)
        level = 1
        res = [_strip(service.precheck(d, call_sid="CA1", ani=ANI[THOMAS], target_level=level,
                                       stir_verstat=verstat, **kw)),
               _strip(service.verify(d, call_sid="CA1", ani=ANI[THOMAS], factors=factors,
                                     target_level=level, stir_verstat=verstat, **kw)),
               service.send_otp(d, call_sid="CA1", ani=ANI[THOMAS], stir_verstat=verstat, **kw)]
        runs.append((res, sorted(map(str, d.store._tables["calls"])),  # noqa: SLF001
                     sorted(map(str, d.store._tables["risk"])),  # noqa: SLF001
                     d.verify_client.sent))  # type: ignore[union-attr]
    assert runs[0] == runs[1]


def test_handler_payload_round_trip_keeps_the_field() -> None:
    """The front door's wire payload (``LambdaBackendClient``) reaches ``handle_event`` intact."""
    payload = json.loads(json.dumps({"op": "precheck", "call_sid": "CA1", "ani": ANI[THOMAS],
                                     "target_level": 2, "stir_verstat": TRUSTED,
                                     "identity_source": KEYED}))
    assert service.handle_event(make("allow"), payload)["result"] == "escalate"
