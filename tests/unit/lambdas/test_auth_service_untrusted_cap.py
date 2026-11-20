"""Unit tests: the untrusted-attempt cap and no one-time code before login.

The PBT for the counter is ``tests/property/test_untrusted_cap_props.py``.
"""

from __future__ import annotations

from tests.logutil import lines as lines_of

import copy
import json
import logging
from typing import Any

import pytest

from lambdas.auth_service import service
from lambdas.auth_service.service import AuthDeps, UntrustedCounter
from owlbank.policy import Policy
from owlbank.store import MemoryStore

from tests.unit.lambdas.test_auth_service import (
    ANI, FACTORS, NOW, POLICY_DOC, THOMAS, Clock, FakeVerify, make_deps,
)

TRUSTED = "TN-Validation-Passed-A"
WRONG = {"zip": "00000", "ssn4": "0000"}


def right(cid: str = THOMAS) -> dict[str, str]:
    return {"zip": FACTORS[cid]["zip"], "ssn4": FACTORS[cid]["ssn4"]}


def risk_items(d: AuthDeps) -> dict[str, dict[str, Any]]:
    return {k[0]: v for k, v in d.store._tables["risk"].items()}  # noqa: SLF001


def untrusted_verify(d: AuthDeps, n: int, factors: dict[str, str], *, level: int = 1,
                     verstat: str | None = None) -> dict[str, Any]:
    return service.verify(d, call_sid=f"CAr4{n:04d}", ani=ANI[THOMAS], factors=factors,
                          target_level=level, stir_verstat=verstat)


# --------------------------------------------------------------------------- the cap


def test_limit_failures_then_escalate_without_checking_factors(
        caplog: pytest.LogCaptureFixture) -> None:
    d = make_deps()
    for n in range(3):
        assert untrusted_verify(d, n, WRONG)["result"] == "not_verified"
    with caplog.at_level(logging.INFO, logger="owlbank.audit"):
        out = untrusted_verify(d, 3, right())  # the right answers no longer help
    assert out == {"result": "escalate", "reason": "verification_risk", "level": 0}
    lines = lines_of(caplog.records, "owlbank.audit")
    assert [ln["audit"] for ln in lines] == ["verification_risk"]
    assert lines[0]["cause"] == "untrusted_failure_limit" and lines[0]["customer_ids"] == [THOMAS]
    blob = caplog.text
    assert ANI[THOMAS] not in blob and FACTORS[THOMAS]["zip"] not in blob
    # No call slot was taken and no token issued for the escalated attempt.
    assert not [k for k in d.store._tables["calls"] if str(k[0]).startswith("ratelimit#call#CAr40003")]  # noqa: SLF001


def test_risk_items_hold_no_pii_and_share_the_window_expiry() -> None:
    clock = Clock()
    d = make_deps(clock=clock)
    for n in range(3):
        clock.t = NOW + 10 * n
        untrusted_verify(d, n, WRONG)
    items = risk_items(d)
    assert set(items) == {f"untrusted#{THOMAS}#{n}" for n in (1, 2, 3)}
    for item in items.values():
        assert set(item) == {"risk_key", "kind", "customer_id", "claimed_at", "expires_at"}
        assert item["customer_id"] == THOMAS
        assert item["expires_at"] == NOW + 3600  # from the first failure
    text = json.dumps(items)
    assert ANI[THOMAS] not in text  # the field set above already excludes factor values


def test_window_restarts_after_it_lapses() -> None:
    clock = Clock()
    d = make_deps(clock=clock)
    for n in range(3):
        untrusted_verify(d, n, WRONG)
    clock.t = NOW + 3599
    assert untrusted_verify(d, 10, right())["result"] == "escalate"
    clock.t = NOW + 3600  # the window ends 3600 s after the first failure
    out = untrusted_verify(d, 11, right())
    assert out["result"] == "verified"


def test_trusted_caller_never_reads_or_writes_the_risk_table() -> None:
    d = make_deps()
    for n in range(3):
        untrusted_verify(d, n, WRONG)
    before = copy.deepcopy(risk_items(d))
    seen: list[str] = []
    real_get, real_put = d.store.get, d.store.put
    d.store.get = lambda t, k: (seen.append(t), real_get(t, k))[1]  # type: ignore[method-assign]
    d.store.put = lambda t, i, **kw: (seen.append(t), real_put(t, i, **kw))[1]  # type: ignore[method-assign]
    assert untrusted_verify(d, 20, {"zip": "00000"}, verstat=TRUSTED)["result"] == "not_verified"
    assert service.precheck(d, call_sid="CAr40021", ani=ANI[THOMAS], target_level=1,
                            stir_verstat=TRUSTED) == {"result": "ok"}
    out = untrusted_verify(d, 22, {"zip": FACTORS[THOMAS]["zip"]}, verstat=TRUSTED)
    assert out["result"] == "verified"
    assert "risk" not in seen
    assert risk_items(d) == before


def test_shared_phone_counts_every_candidate_and_any_at_limit_escalates() -> None:
    d = make_deps()
    sam_ani = "+12025550102"  # Sam and Jordan share it
    for n in range(3):
        service.verify(d, call_sid=f"CAsh{n}", ani=sam_ani, factors=WRONG, target_level=1)
    ids = {item["customer_id"] for item in risk_items(d).values()}
    assert ids == {"C-1002", "C-1003"}
    assert service.precheck(d, call_sid="CAsh9", ani=sam_ani, target_level=1)["result"] == "escalate"


def test_counter_never_exceeds_the_limit() -> None:
    store = MemoryStore()
    c = UntrustedCounter(store, limit=3, window_s=60)
    assert [c.record("C-1", NOW) for _ in range(5)] == [True, True, True, False, False]
    assert c.count("C-1", NOW) == 3 and c.at_limit("C-1", NOW)
    assert c.count("C-1", NOW + 60) == 0
    assert c.record("C-1", NOW + 60)
    assert store.get("risk", {"risk_key": "untrusted#C-1#1"})["expires_at"] == NOW + 120


def test_precheck_escalates_only_at_the_limit() -> None:
    d = make_deps()
    kw = {"ani": ANI[THOMAS], "target_level": 1}
    for n in range(3):
        assert service.precheck(d, call_sid=f"CAp{n}", **kw) == {"result": "ok"}
        untrusted_verify(d, n, WRONG)
    out = service.precheck(d, call_sid="CAp9", **kw)
    assert out == {"result": "escalate", "reason": "verification_risk", "level": 0}


def test_precheck_rejects_bad_input() -> None:
    d = make_deps()
    assert service.precheck(d, call_sid="", ani=None, target_level=1)["result"] == "bad_request"
    assert service.precheck(d, call_sid="CA1", ani=None, target_level=9)["result"] == "bad_request"
    ev = {"op": "precheck", "call_sid": "CA1", "ani": ANI[THOMAS], "target_level": 1, "x": 1}
    assert service.handle_event(d, ev)["reason"] == "unexpected_fields"
    ev.pop("x")
    assert service.handle_event(d, ev) == {"result": "ok"}


def test_policy_limit_and_window_are_used() -> None:
    doc = copy.deepcopy(POLICY_DOC)
    doc["ani_trust"].update(untrusted_failure_limit=1, untrusted_window_s=60)
    d = make_deps()
    d.policy = Policy.from_dict(doc)
    assert untrusted_verify(d, 0, WRONG)["result"] == "not_verified"
    assert untrusted_verify(d, 1, right())["result"] == "escalate"
    assert risk_items(d)[f"untrusted#{THOMAS}#1"]["expires_at"] == NOW + 60


# --------------------------------------------------------------------------- OTP before login


def test_untrusted_verify_that_needs_an_otp_before_a_token_escalates() -> None:
    verify = FakeVerify()
    d = make_deps(verify=verify)
    # Untrusted, level 2: the next level's factors (zip, ssn4, otp), and no token yet.
    out = untrusted_verify(d, 0, {**right(), "otp": "123456"}, level=2)
    assert out == {"result": "escalate", "reason": "verification_risk", "level": 0}
    assert verify.sent == [] and verify.checked == []
    assert risk_items(d) == {}  # not a failure: nothing was checked
    pre = service.precheck(d, call_sid="CAr40001", ani=ANI[THOMAS], target_level=3)
    assert pre["result"] == "escalate"


def test_trusted_verify_needing_an_otp_is_unaffected() -> None:
    d = make_deps()
    out = untrusted_verify(d, 0, {"zip": FACTORS[THOMAS]["zip"]}, level=1, verstat=TRUSTED)
    assert out["result"] == "verified"
    assert service.precheck(d, call_sid="CAr49999", ani=ANI[THOMAS], target_level=3,
                            stir_verstat=TRUSTED) == {"result": "ok"}


def test_untrusted_step_up_to_otp_with_a_token_is_allowed() -> None:
    verify = FakeVerify()
    d = make_deps(verify=verify)
    assert untrusted_verify(d, 0, right())["result"] == "verified"  # level 1, token issued
    assert service.precheck(d, call_sid="CAr40000", ani=ANI[THOMAS],
                            target_level=2) == {"result": "ok"}
    assert service.send_otp(d, call_sid="CAr40000", ani=ANI[THOMAS]) == {"result": "sent"}
    out = untrusted_verify(d, 0, {"otp": "123456"}, level=2)
    assert out["result"] == "verified" and out["level"] == 2


def test_send_otp_still_refuses_without_a_token() -> None:
    verify = FakeVerify()
    d = make_deps(verify=verify)
    assert service.send_otp(d, call_sid="CAnotoken", ani=ANI[THOMAS]) == {"result": "not_verified"}
    assert verify.sent == []
