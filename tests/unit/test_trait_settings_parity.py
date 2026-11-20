"""infra's copy of the Memory trait settings equals ``owlbank.web_activity``.

infra/ is its own uv project and can't import src/, so ``infra/owlbank_infra/config.py`` keeps a
copy. That module imports only the standard library, so it is loaded here by file path (not as the
``owlbank_infra`` package, whose other modules need CDK).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from owlbank.web_activity import (
    DEFAULT_MAX_AGE_HOURS,
    DEFAULT_TRAIT_GROUPS,
    MAX_AGE_HOURS_RANGE,
    MAX_TRAIT_GROUPS,
    TRAIT_GROUP_NAME_RE,
)
from tac_bridge.bridge import BridgeConfig

INFRA_CONFIG = Path(__file__).resolve().parents[2] / "infra" / "owlbank_infra" / "config.py"


@pytest.fixture(scope="module")
def infra_config() -> ModuleType:
    name = "_infra_config_for_parity"
    spec = importlib.util.spec_from_file_location(name, INFRA_CONFIG)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while building the class
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_infra_trait_settings_equal_web_activity(infra_config: ModuleType) -> None:
    assert infra_config.DEFAULT_MEMORY_TRAIT_GROUPS == DEFAULT_TRAIT_GROUPS
    assert infra_config.TRAIT_GROUP_NAME_RE.pattern == TRAIT_GROUP_NAME_RE.pattern
    assert infra_config.TRAIT_GROUP_NAME_RE.flags == TRAIT_GROUP_NAME_RE.flags
    assert infra_config.MAX_TRAIT_GROUPS == MAX_TRAIT_GROUPS
    assert infra_config.WEB_ACTIVITY_MAX_AGE_HOURS_RANGE == MAX_AGE_HOURS_RANGE
    assert infra_config.DEFAULT_WEB_ACTIVITY_MAX_AGE_HOURS == DEFAULT_MAX_AGE_HOURS


def _defaults(cls: type) -> dict[str, object]:
    return {f.name: f.default for f in dataclasses.fields(cls)}


def test_infra_and_bridge_dataclass_defaults_agree(infra_config: ModuleType) -> None:
    infra, bridge = _defaults(infra_config.InfraConfig), _defaults(BridgeConfig)
    assert infra["memory_trait_groups"] == bridge["memory_trait_groups"] == DEFAULT_TRAIT_GROUPS
    assert infra["web_activity_max_age_hours"] == bridge["web_activity_max_age_hours"] == DEFAULT_MAX_AGE_HOURS
