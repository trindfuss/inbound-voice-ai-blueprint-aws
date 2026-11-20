"""AI keypad shortcuts: ``aiKeypadShortcuts`` and ``OWLBANK_AI_KEYPAD``.

``on`` (default) or ``off`` on the front door and on the bridge, in both hostings; any other value
fails synth naming the key; the README settings row exists once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import INFRA, _synth
from test_bridge_agentcore import AGENTCORE

from owlbank_infra.app_builder import build_app
from owlbank_infra.config import ConfigError, from_context

README = Path(INFRA) / "README.md"
VAR = "OWLBANK_AI_KEYPAD"


def _fd_env(synth) -> dict:
    tpl = synth.templates["OwlBankCompute"]
    fn = next(r for lid, r in tpl["Resources"].items()
              if r["Type"] == "AWS::Lambda::Function" and lid.startswith("FrontdoorFn"))
    return fn["Properties"]["Environment"]["Variables"]


def _bridge_env(synth, hosting: str) -> dict:
    tpl = synth.templates["OwlBankBridge"]
    if hosting == "agentcore":
        rt = next(r for r in tpl["Resources"].values() if r["Type"] == "AWS::BedrockAgentCore::Runtime")
        return rt["Properties"]["EnvironmentVariables"]
    td = next(r for r in tpl["Resources"].values() if r["Type"] == "AWS::ECS::TaskDefinition")
    env = td["Properties"]["ContainerDefinitions"][0]["Environment"]
    return {e["Name"]: e["Value"] for e in env}


@pytest.mark.parametrize("hosting", ["fargate", "agentcore"])
@pytest.mark.parametrize(("extra", "want"), [({}, "on"), ({"aiKeypadShortcuts": "on"}, "on"),
                                             ({"aiKeypadShortcuts": "off"}, "off")])
def test_env_on_both_stacks_and_hostings(tmp_path_factory, hosting, extra, want):
    ctx = {**(AGENTCORE if hosting == "agentcore" else {}), **extra}
    synth = _synth(tmp_path_factory, f"aikp-{hosting}-{want}", ctx)
    assert _fd_env(synth)[VAR] == want
    assert _bridge_env(synth, hosting)[VAR] == want


def test_default_is_on():
    assert from_context({}, base_dir=INFRA).ai_keypad_shortcuts == "on"


@pytest.mark.parametrize("bad", ["maybe", "true", "1"])
def test_bad_value_is_refused_naming_the_key(tmp_path, bad):
    with pytest.raises(ConfigError, match="aiKeypadShortcuts"):
        build_app({"aiKeypadShortcuts": bad}, outdir=str(tmp_path))


def test_readme_has_one_settings_row():
    rows = [ln for ln in README.read_text().splitlines() if ln.startswith("| `aiKeypadShortcuts` |")]
    assert len(rows) == 1
    default, meaning = [c.strip() for c in rows[0].strip("|").split("|")][1:3]
    assert default == "`on`"
    for needle in ("OWLBANK_AI_KEYPAD", "OwlBankCompute", "OwlBankBridge", "never sends them to the model"):
        assert needle in meaning
