"""Seed option --developer-factor: a developer's own factor value, hashed at seed time, dev/test only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._common import ScriptError
from scripts.hash_tool import verify
from scripts.seed import build_plan, render_plan

DATA = Path(__file__).resolve().parents[2] / "data"
FIX, POL, CAT = (json.loads((DATA / n).read_text()) for n in ("fixtures.json", "policy.json", "prompt_catalog.json"))
DEV_FIXTURE = FIX["realNumberPlaceholders"]["developerPhone"]["fixture"]
VALUE = "55555"  # a test value; the real one is passed only on the command line


def _plan(stage: str = "dev", **factors: str):
    return build_plan(FIX, POL, CAT, stage=stage, developer_factors=factors)


def test_the_value_is_hashed_on_the_developer_fixture_only() -> None:
    plan = _plan(zip=VALUE)
    by_id = {c["customer_id"]: c for c in plan.customers}
    assert verify(VALUE, by_id[DEV_FIXTURE]["factors"]["zip"])
    others = [c for cid, c in by_id.items() if cid != DEV_FIXTURE and "zip" in c.get("factors", {})]
    assert others and not any(verify(VALUE, c["factors"]["zip"]) for c in others)


def test_the_value_is_never_printed() -> None:
    assert VALUE not in render_plan(_plan(zip=VALUE))


def test_the_repository_fixtures_are_untouched() -> None:
    before = json.dumps(FIX, sort_keys=True)
    _plan(zip=VALUE)
    assert json.dumps(FIX, sort_keys=True) == before


@pytest.mark.parametrize("stage,factors", [("prod", {"zip": VALUE}), ("dev", {"zip": "1234"}),
                                           ("dev", {"zip": "abcde"}), ("dev", {"otp": "123456"}),
                                           ("dev", {"pin": "1234"})])
def test_refused(stage, factors) -> None:
    with pytest.raises(ScriptError):
        _plan(stage, **factors)
