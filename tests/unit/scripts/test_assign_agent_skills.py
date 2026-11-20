"""scripts/assign_agent_skills.py and the pure merge in scripts/_agent_skills.py. No network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts import assign_agent_skills as aas
from scripts._agent_skills import AGENT_SKILLS, merge_worker_skills

from .test_twilio_setup import (
    API_KEY_ARN,
    SECRET_BODY,
    SUB,
    FakeTwilio,
    StubSM,
)

ARGS = ["--secret", f"twilio_api_key={API_KEY_ARN}"]


def _worker(n: int, attributes: Any) -> SimpleNamespace:
    return SimpleNamespace(sid=f"WK{n:032d}", friendly_name=f"agent{n}",
                           attributes=attributes if isinstance(attributes, str) else json.dumps(attributes))


def run(fake: FakeTwilio, *args: str) -> int:
    return aas.main([*ARGS, *args], sm_client=StubSM(), client_factory=lambda *a: fake)


def attrs(w: Any) -> dict[str, Any]:
    return dict(json.loads(w.attributes))


def updates(fake: FakeTwilio) -> list[Any]:
    return [c for c in fake.calls if c[0] == "workers.update"]


def test_email_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        aas.main(ARGS, sm_client=StubSM())


def test_dry_run_changes_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    fake.workers = [_worker(1, {"email": "A@example.com"})]
    assert run(fake, "--email", "a@example.com") == 0
    assert updates(fake) == []
    assert "would" in capsys.readouterr().out


def test_apply_merges_skills_for_the_matching_worker_only() -> None:
    fake = FakeTwilio()
    fake.workers = [
        _worker(1, {"email": "Agent@example.com", "routing": {"skills": ["billing"], "levels": {"billing": 2}},
                    "contact_uri": "client:a"}),
        _worker(2, {"email": "other@example.com"}),
    ]
    assert run(fake, "--email", "agent@example.com", "--apply") == 0
    a = attrs(fake.workers[0])
    assert a["routing"] == {"skills": ["billing", *AGENT_SKILLS], "levels": {"billing": 2}}
    assert a["email"] == "Agent@example.com" and a["contact_uri"] == "client:a"
    assert attrs(fake.workers[1]) == {"email": "other@example.com"}
    assert len(updates(fake)) == 1


def test_already_has_them_is_not_updated(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    fake.workers = [_worker(1, {"email": "a@example.com", "routing": {"skills": list(AGENT_SKILLS)}})]
    assert run(fake, "--email", "a@example.com", "--apply") == 0
    assert updates(fake) == []
    assert "already has" in capsys.readouterr().out


def test_second_apply_is_a_no_op() -> None:
    fake = FakeTwilio()
    fake.workers = [_worker(1, {"email": "a@example.com"})]
    assert run(fake, "--email", "a@example.com", "--apply") == 0
    fake.calls.clear()
    assert run(fake, "--email", "a@example.com", "--apply") == 0
    assert updates(fake) == []


def test_no_worker_found_is_reported_and_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    fake.workers = [_worker(1, {"email": "a@example.com"})]
    assert run(fake, "--email", "a@example.com", "--email", "ghost@example.com", "--apply") == 1
    out = capsys.readouterr().out
    assert "ghost@example.com: no worker found" in out and "signed in" in out
    assert len(updates(fake)) == 1  # the other email was still done


def test_two_workers_with_one_email_warn_and_both_update(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    fake.workers = [_worker(1, {"email": "a@example.com"}), _worker(2, {"email": "a@example.com"})]
    assert run(fake, "--email", "a@example.com", "--apply") == 0
    out = capsys.readouterr().out
    assert "matches 2 workers (agent1, agent2)" in out
    assert len(updates(fake)) == 2


def test_non_json_attributes_are_never_overwritten() -> None:
    fake = FakeTwilio()
    fake.workers = [_worker(1, "not json")]
    assert run(fake, "--email", "a@example.com", "--apply") == 1
    assert updates(fake) == [] and fake.workers[0].attributes == "not json"


def test_missing_flex_workspace_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio(workspace=False)
    assert run(fake, "--email", "a@example.com", "--apply") == 3
    assert "Flex" in capsys.readouterr().err


def test_no_secret_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    fake.workers = [_worker(1, {"email": "a@example.com"})]
    run(fake, "--email", "a@example.com", "--apply")
    cap = capsys.readouterr()
    assert SECRET_BODY not in cap.out + cap.err
    assert "SK" + "b2" * 16 not in cap.out + cap.err and SUB not in cap.err


def test_merge_worker_skills_basics() -> None:
    assert merge_worker_skills("[]", ["en"]) is None
    assert merge_worker_skills("", ["en"]) == json.dumps({"routing": {"skills": ["en"]}})
    assert merge_worker_skills(json.dumps({"routing": {"skills": ["en"]}}), ["en"]) is None
    assert merge_worker_skills('{"routing": "x"}', ["en"]) is None


_names = st.text(alphabet=st.characters(whitelist_categories=("Ll", "Nd")), min_size=1, max_size=6)
_json_leaf = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=8))


@settings(max_examples=100, deadline=None)
@given(skills=st.lists(_names, max_size=5, unique=True), wanted=st.lists(_names, min_size=1, max_size=5),
       levels=st.dictionaries(_names, st.integers(0, 9), max_size=3),
       other=st.dictionaries(st.text(min_size=1, max_size=5).filter(lambda k: k != "routing"),
                             _json_leaf, max_size=4),
       with_routing=st.booleans())
def test_property_merge_worker_skills(skills: list[str], wanted: list[str], levels: dict[str, int],
                                      other: dict[str, Any], with_routing: bool) -> None:
    routing: dict[str, Any] = {"skills": skills, "levels": levels} if with_routing else {}
    old = {**other, "routing": routing} if with_routing else dict(other)
    merged = merge_worker_skills(json.dumps(old), wanted)
    if merged is None:
        assert with_routing and set(wanted) <= set(skills)
        return
    new = json.loads(merged)
    old_skills = skills if with_routing else []
    assert new["routing"]["skills"][:len(old_skills)] == old_skills  # old ones kept, in order
    assert set(wanted) <= set(new["routing"]["skills"])
    assert {k: v for k, v in new.items() if k != "routing"} == other
    if with_routing:
        assert new["routing"]["levels"] == levels
    assert merge_worker_skills(merged, wanted) is None  # a second merge has nothing to do
