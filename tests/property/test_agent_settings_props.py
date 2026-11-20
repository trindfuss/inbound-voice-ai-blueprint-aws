"""The ``promptCaching`` parser accepts
exactly the allowed values, over arbitrary strings.

The parser lives in the CDK app (``infra/owlbank_infra/config.py``), a separate uv project without
Hypothesis. That module is stdlib-only, so it is loaded here by path; the synth-level checks
(grants, env vars, cdk-nag) are in ``infra/tests/test_agent_loop.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

INFRA = Path(__file__).resolve().parents[2] / "infra"
_PKG = "_owlbank_infra_under_test"


def _load_config():
    """``owlbank_infra.config`` under a private package name (it has a lazy relative import of
    ``contract``, so it must load as a submodule), without adding ``infra/`` to ``sys.path``."""
    if _PKG not in sys.modules:
        pkg_dir = INFRA / "owlbank_infra"
        spec = importlib.util.spec_from_file_location(
            _PKG, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
        assert spec and spec.loader
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[_PKG] = pkg
        spec.loader.exec_module(pkg)
    return importlib.import_module(f"{_PKG}.config")


config = _load_config()
SETTINGS = (
    ("promptCaching", config.PROMPT_CACHING_MODES, "on", "prompt_caching"),
    ("promptCacheTtl", config.PROMPT_CACHE_TTLS, "5m", "prompt_cache_ttl"),
)
#: Per-entry ``modelParams`` fields have their own tests in ``infra/tests/test_agent_loop.py`` and
#: ``test_model_list.py``.


def _ctx(key: str, value: object) -> dict[str, object]:
    return {key: value}


def _variants(allowed: tuple[str, ...]):
    """Allowed values in random case with surrounding whitespace, plus arbitrary text."""
    cased = st.sampled_from(allowed).flatmap(
        lambda v: st.lists(st.booleans(), min_size=len(v), max_size=len(v)).map(
            lambda ups: "".join(c.upper() if u else c for c, u in zip(v, ups, strict=True))))
    ws = st.sampled_from(["", " ", "\t", "\n "])
    return st.one_of(st.tuples(ws, cased, ws).map("".join), st.text(max_size=20))


def test_allowed_values_and_defaults():
    assert config.PROMPT_CACHING_MODES == ("on", "off")
    cfg = config.from_context({}, base_dir=INFRA)
    assert cfg.prompt_caching == "on"
    assert cfg.prompt_cache_ttl == "5m"
    assert not hasattr(config, "AGENT_BACKENDS")  # No backend choice


@pytest.mark.parametrize(("key", "allowed", "default", "field"), SETTINGS)
@settings(max_examples=300, deadline=None)
@given(data=st.data())
def test_parser_accepts_exactly_the_allowed_values(key, allowed, default, field, data):
    """A value is accepted iff, trimmed and lower-cased, it is an allowed value (empty means the
    default), and then the setting is that canonical value; anything else fails with an error
    naming the setting."""
    value = data.draw(_variants(allowed))
    norm = value.strip().lower() or default
    if norm in allowed:
        assert getattr(config.from_context(_ctx(key, value), base_dir=INFRA), field) == norm
    else:
        with pytest.raises(config.ConfigError, match=key):
            config.from_context(_ctx(key, value), base_dir=INFRA)


@pytest.mark.parametrize(("key", "allowed", "default", "field"), SETTINGS)
@given(value=st.one_of(st.integers(), st.booleans(), st.lists(st.text(max_size=3), min_size=1)))
def test_non_strings_are_rejected(key, allowed, default, field, value):
    with pytest.raises(config.ConfigError, match=key):
        config.from_context(_ctx(key, value), base_dir=INFRA)
