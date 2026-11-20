"""Web-activity traits: ``memoryTraitGroups`` and ``webActivityMaxAgeHours``."""

from __future__ import annotations

from pathlib import Path

import pytest

from owlbank_infra.config import ConfigError, from_context

BASE = Path(__file__).resolve().parents[1]


def test_defaults() -> None:
    cfg = from_context({}, base_dir=BASE)
    assert cfg.memory_trait_groups == ("Contact", "WebActivity")
    assert cfg.web_activity_max_age_hours == 24


def test_overrides() -> None:
    cfg = from_context({"memoryTraitGroups": "Contact, Other.Group-1_x", "webActivityMaxAgeHours": "8760"},
                       base_dir=BASE)
    assert cfg.memory_trait_groups == ("Contact", "Other.Group-1_x")
    assert cfg.web_activity_max_age_hours == 8760
    assert from_context({"webActivityMaxAgeHours": 1}, base_dir=BASE).web_activity_max_age_hours == 1


@pytest.mark.parametrize("bad", ["Contact,Contact", "Bad Name", "a/b", "A" * 65, "Contact,,Web", ",",
                                 ",".join(f"G{i}" for i in range(11)), 5])
def test_bad_trait_groups_are_refused_naming_the_key(bad) -> None:
    with pytest.raises(ConfigError, match="memoryTraitGroups"):
        from_context({"memoryTraitGroups": bad}, base_dir=BASE)


def test_ten_groups_are_allowed() -> None:
    names = ",".join(f"G{i}" for i in range(10))
    assert len(from_context({"memoryTraitGroups": names}, base_dir=BASE).memory_trait_groups) == 10


@pytest.mark.parametrize("bad", ["0", "8761", "-1", "day", "2.5", True])
def test_bad_max_age_is_refused_naming_the_key(bad) -> None:
    with pytest.raises(ConfigError, match="webActivityMaxAgeHours"):
        from_context({"webActivityMaxAgeHours": bad}, base_dir=BASE)


def test_bridge_task_gets_both_settings(synth_default) -> None:
    tpl = synth_default.templates["OwlBankBridge"]
    env = {}
    for r in tpl["Resources"].values():
        if r["Type"] == "AWS::ECS::TaskDefinition":
            env = {e["Name"]: e.get("Value") for e in r["Properties"]["ContainerDefinitions"][0]["Environment"]}
    assert env["OWLBANK_MEMORY_TRAIT_GROUPS"] == "Contact,WebActivity"
    assert env["OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS"] == "24"


@pytest.mark.parametrize("synth_name", ["synth_agentcore_vpc_dev", "synth_agentcore_public_dev",
                                        "synth_agentcore_vpc_prod", "synth_agentcore_public_prod"])
def test_agentcore_runtime_gets_both_settings(synth_name, request) -> None:
    tpl = request.getfixturevalue(synth_name).templates["OwlBankBridge"]
    runtimes = [r for r in tpl["Resources"].values() if r["Type"] == "AWS::BedrockAgentCore::Runtime"]
    assert len(runtimes) == 1
    env = runtimes[0]["Properties"]["EnvironmentVariables"]
    assert env["OWLBANK_MEMORY_TRAIT_GROUPS"] == "Contact,WebActivity"
    assert env["OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS"] == "24"
