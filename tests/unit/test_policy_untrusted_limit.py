"""Policy fields: ``ani_trust.untrusted_failure_limit`` and ``untrusted_window_s``.
Same fail-closed pattern as the rest of ``ani_trust``."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from owlbank.policy import (
    DEFAULT_ANI_TRUST,
    Policy,
    PolicyError,
    PolicySectionError,
    parse_ani_trust,
)
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
DOC: dict[str, Any] = json.loads((DATA / "policy.json").read_text())


def doc(ani_trust: Any) -> dict[str, Any]:
    d = copy.deepcopy(DOC)
    d["ani_trust"] = ani_trust
    return d


def test_seed_file_and_defaults_have_the_documented_cap() -> None:
    p = Policy.from_dict(DOC)
    assert p.ani_trust.untrusted_failure_limit == 3 and p.ani_trust.untrusted_window_s == 3600
    assert DEFAULT_ANI_TRUST.untrusted_failure_limit == 3
    assert DEFAULT_ANI_TRUST.untrusted_window_s == 3600
    # Omitted fields take the defaults.
    assert parse_ani_trust({}).untrusted_failure_limit == 3


def test_verification_risk_routes_to_the_fraud_queue() -> None:
    route = Policy.from_dict(DOC).escalation_route("verification_risk")
    assert (route.reason, route.queue_key, route.phrase_key) == (
        "verification_risk", "fraud", "escalation.verification_risk")


@pytest.mark.parametrize("field,value", [
    ("untrusted_failure_limit", 1), ("untrusted_failure_limit", 100),
    ("untrusted_window_s", 60), ("untrusted_window_s", 86400),
])
def test_bounds_are_inclusive(field: str, value: int) -> None:
    assert getattr(parse_ani_trust({field: value}), field) == value


@pytest.mark.parametrize("field,value", [
    ("untrusted_failure_limit", 0), ("untrusted_failure_limit", 101),
    ("untrusted_failure_limit", "3"), ("untrusted_failure_limit", 3.0),
    ("untrusted_failure_limit", True), ("untrusted_failure_limit", None),
    ("untrusted_window_s", 59), ("untrusted_window_s", 86401), ("untrusted_window_s", "3600"),
    ("untrusted_window_s", False), ("untrusted_window_s", -1),
])
def test_invalid_cap_gives_defaults_and_an_error_naming_the_field(
        field: str, value: Any, caplog: pytest.LogCaptureFixture) -> None:
    raw = {field: value}
    with pytest.raises(PolicySectionError) as exc:
        parse_ani_trust(raw)
    assert exc.value.field == f"ani_trust.{field}"
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_dict(doc(raw))
    assert p.ani_trust == DEFAULT_ANI_TRUST
    assert any(f"ani_trust.{field}" in r.getMessage() for r in caplog.records)
    with pytest.raises(PolicyError):
        Policy.to_items(doc(raw))


def test_cap_round_trips_through_the_store() -> None:
    store = MemoryStore()
    cfg = {"accepted_verstat": ["TN-Validation-Passed-A"], "on_untrusted": "no_ani_match",
           "untrusted_failure_limit": 5, "untrusted_window_s": 600}
    for item in Policy.to_items(doc(cfg)):
        store.put("policy", item)
    # The stored item also carries the keyed-phone switch (default on).
    assert store.get("policy", {"pk": "ani_trust"}) == {"pk": "ani_trust", **cfg,
                                                        "ask_phone_on_no_match": True}
    got = Policy.from_store(store).ani_trust
    assert (got.untrusted_failure_limit, got.untrusted_window_s) == (5, 600)
