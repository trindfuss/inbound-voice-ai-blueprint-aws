"""The Lambda layers install only hashed, pinned requirements exported from ``uv.lock``."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_lambda_deps.py"
spec = importlib.util.spec_from_file_location("build_lambda_deps", SCRIPT)
assert spec and spec.loader
bld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bld)


@pytest.fixture(scope="module")
def exported() -> str:
    return bld.export_all()


def test_install_command_requires_hashes_and_skips_resolution(tmp_path):
    cmd = bld.install_command(tmp_path / "python", tmp_path / "requirements.txt")
    assert "--require-hashes" in cmd
    assert "--no-deps" in cmd
    assert cmd[cmd.index("-r") + 1] == str(tmp_path / "requirements.txt")


@pytest.mark.parametrize("layer", list(bld.LAYERS))
def test_every_transitive_dependency_is_pinned_with_hashes(layer, exported):
    graph = bld.lock_graph()
    names = bld.closure(bld.LAYERS[layer], graph)
    assert len(names) > len(bld.LAYERS[layer])  # transitive deps are included
    text = bld.layer_requirements(exported, names)
    entries = re.split(r"\n(?=\S)", text.strip())
    assert {bld._entry_name(e) for e in entries} == names
    for e in entries:
        assert re.match(r"[A-Za-z0-9_.-]+==\S+", e), e
        assert "--hash=sha256:" in e, e
