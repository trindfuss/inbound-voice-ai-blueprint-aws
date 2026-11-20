"""node_bootstrap: jsii runs on the venv's pinned Node, not whatever ``node`` is on PATH."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import nodejs_wheel

from owlbank_infra.node_bootstrap import ensure_node

BUNDLED = os.path.join(os.path.dirname(nodejs_wheel.__file__), "bin")


def test_the_bundled_node_comes_first_even_with_another_node_on_path(monkeypatch, tmp_path: Path) -> None:
    other = tmp_path / "node"
    other.write_text("#!/bin/sh\necho v99.0.0\n")
    other.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}/usr/bin")
    ensure_node()
    assert os.environ["PATH"].split(os.pathsep)[0] == BUNDLED
    assert shutil.which("node") == os.path.join(BUNDLED, "node")


def test_ensure_node_is_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    ensure_node()
    ensure_node()
    assert os.environ["PATH"] == f"{BUNDLED}{os.pathsep}/usr/bin"


def test_the_bundled_node_is_a_release_jsii_supports() -> None:
    # jsii's supported majors (its untested-version warning): 20, 22, 24. pyproject pins <25.
    out = subprocess.run([os.path.join(BUNDLED, "node"), "--version"],  # noqa: S603 - our node
                         capture_output=True, text=True, check=True).stdout
    assert int(out.strip().lstrip("v").split(".")[0]) in (22, 24)
