"""``timezone`` reaches the bridge; an invalid zone is refused at synth."""

from __future__ import annotations

from pathlib import Path

import pytest

from owlbank_infra.config import ConfigError, from_context

BASE = Path(__file__).resolve().parents[1]


def test_default_and_override() -> None:
    assert from_context({}, base_dir=BASE).timezone == "America/New_York"
    assert from_context({"timezone": "America/Chicago"}, base_dir=BASE).timezone == "America/Chicago"


@pytest.mark.parametrize("bad", ["Eastern", "America/Nowhere", "../etc"])
def test_invalid_zone_is_refused(bad) -> None:
    with pytest.raises(ConfigError, match="timezone"):
        from_context({"timezone": bad}, base_dir=BASE)


def test_bridge_gets_the_zone(synth_default) -> None:
    tpl = synth_default.templates["OwlBankBridge"]
    env = {}
    for r in tpl["Resources"].values():
        if r["Type"] == "AWS::ECS::TaskDefinition":
            env = {e["Name"]: e.get("Value") for e in r["Properties"]["ContainerDefinitions"][0]["Environment"]}
    assert env.get("OWLBANK_TIMEZONE") == "America/New_York"
