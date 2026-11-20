"""With ``on_untrusted: no_ani_match``, for any ``StirVerstat`` outside
``accepted_verstat`` (including missing), the auth service never counts ``ani_match`` and
never increments a customer's lockout counter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.auth_service import service
from owlbank.policy import Policy
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
FIXTURES = json.loads((DATA / "fixtures.json").read_text())["customers"]
FACTORS = json.loads((DATA / "fixtures.test-factors.json").read_text())["customers"]
POLICY = Policy.from_dict(json.loads((DATA / "policy.json").read_text()))
ACCEPTED = set(POLICY.ani_trust.accepted_verstat)
THOMAS, ANI = "C-1001", "+12025550101"

UNTRUSTED = st.one_of(
    st.none(),
    st.sampled_from(["", "TN-Validation-Passed-B", "TN-Validation-Passed-C", "No-TN-Validation",
                     "TN-Validation-Failed-A", "tn-validation-passed-a", "TN-Validation-Passed-A ",
                     "TN-Validation-Passed-A\n"]),
    st.text(max_size=80),
    st.integers(), st.booleans(),
).filter(lambda v: v not in ACCEPTED)

ATTEMPT = st.sampled_from(["right_zip", "wrong_zip", "right_both", "wrong_both", "wrong_ssn"])


def deps() -> service.AuthDeps:
    store = MemoryStore()
    for c in FIXTURES:
        store.put("customers", copy.deepcopy(c))
    return service.AuthDeps(store=store, policy=POLICY, clock=lambda: 1_760_000_000)


def factors_for(kind: str) -> dict[str, str]:
    z, s = FACTORS[THOMAS]["zip"], FACTORS[THOMAS]["ssn4"]
    return {"right_zip": {"zip": z}, "wrong_zip": {"zip": "00000"},
            "right_both": {"zip": z, "ssn4": s}, "wrong_both": {"zip": "00000", "ssn4": "0000"},
            "wrong_ssn": {"zip": z, "ssn4": "0000"}}[kind]


def customer_slots(d: service.AuthDeps) -> list[str]:
    return [str(k[0]) for k in d.store._tables["calls"]  # noqa: SLF001
            if str(k[0]).startswith("ratelimit#customer#")]


@settings(max_examples=150, deadline=None)
@given(UNTRUSTED, st.lists(ATTEMPT, min_size=1, max_size=8), st.integers(min_value=1, max_value=3))
def test_cp_r3b_untrusted_ani_never_counts_and_never_locks(verstat: Any, attempts: list[str],
                                                           calls: int) -> None:
    d = deps()
    for c in range(calls):
        for kind in attempts:
            out = service.verify(d, call_sid=f"CA{c}", ani=ANI, factors=factors_for(kind),
                                 target_level=1, stir_verstat=verstat)
            if kind == "right_zip":
                # The ANI match isn't a factor, so the ZIP alone never verifies level 1.
                # (Once the untrusted-attempt cap is reached, the gate escalates first.)
                assert out["result"] in ("bad_request", "rate_limited", "escalate")
                if out["result"] == "bad_request":
                    assert out["missing"] == ["ssn4"]
            if out["result"] == "verified":
                assert kind == "right_both"
                assert "ani_match" not in out["factors_verified"]
                break  # this call is verified; a further level-1 verify is invalid_level
    assert customer_slots(d) == []  # the real customer was never put toward a lockout


@settings(max_examples=50, deadline=None)
@given(st.lists(st.sampled_from(["wrong_zip"]), min_size=1, max_size=4))
def test_trusted_failures_still_count_toward_the_lockout(attempts: list[str]) -> None:
    """The control: the same failures from a trusted ANI do count."""
    d = deps()
    for kind in attempts:
        service.verify(d, call_sid="CA1", ani=ANI, factors=factors_for(kind), target_level=1,
                       stir_verstat="TN-Validation-Passed-A")
    assert customer_slots(d)
