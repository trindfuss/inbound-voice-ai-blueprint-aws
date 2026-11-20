"""The Fargate bridge image's asset hash covers only what ``src/tac_bridge/Dockerfile`` COPYs.

``bridge_stack.IMAGE_EXCLUDE`` narrows the build context (the repo root). A change to a file the
image never contains (the admin console, the Lambdas, scripts, data, the uv project files) must not
change the asset hash, or every such change rebuilds and redeploys the bridge; a change to a file
the Dockerfile COPYs must change it.

The hash is computed the way the stack computes it (a ``DockerImageAsset`` with the same
Dockerfile, ``exclude`` and platform) over a scratch copy of the relevant tree, so the real
checkout is never written to. No PBT: a fixed list of paths, checked by assertions.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
from aws_cdk import App, Stack
from aws_cdk import aws_ecr_assets as ecr_assets
from conftest import REPO

from owlbank_infra.bridge_stack import DOCKERFILE, IMAGE_EXCLUDE

DOCKERFILE_REL = DOCKERFILE.relative_to(REPO).as_posix()
#: Repo paths copied into the scratch tree: everything the old, wider context included and more.
MIRRORED = ["pyproject.toml", "uv.lock", "src", "data", "scripts"]
#: Files the image does not use; touching one must leave the hash unchanged.
UNUSED = [
    "src/admin_console/app.js",
    "src/admin_console/new_file.css",
    "src/lambdas/new_module.py",
    "scripts/new_script.py",
    "data/new_fixture.json",
    "pyproject.toml",
    "uv.lock",
]


def _copy_sources() -> list[str]:
    """The source paths of the Dockerfile's COPY lines (flags and the destination dropped)."""
    sources = []
    for line in DOCKERFILE.read_text().splitlines():
        m = re.match(r"\s*COPY\s+(.*)", line)
        if m:
            args = [a for a in m.group(1).split() if not a.startswith("--")]
            sources += args[:-1]
    return sources


COPY_SOURCES = _copy_sources()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in MIRRORED:
        src = REPO / rel
        if src.is_dir():
            shutil.copytree(src, root / rel, ignore=ignore)
        elif src.is_file():
            shutil.copy2(src, root / rel)
    return root


def _asset_hash(root: Path) -> str:
    stack = Stack(App(), "S")
    return ecr_assets.DockerImageAsset(
        stack, "BridgeImage", directory=str(root), file=DOCKERFILE_REL, exclude=IMAGE_EXCLUDE,
        platform=ecr_assets.Platform.LINUX_ARM64).asset_hash


def _touch(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write("\n# changed by test_bridge_image_context\n")


def test_dockerfile_copies_only_from_src() -> None:
    """The COPY sources this test relies on were parsed (guards against a silent empty list)."""
    assert COPY_SOURCES, "no COPY lines found in the bridge Dockerfile"
    assert all(s.startswith("src/") for s in COPY_SOURCES), COPY_SOURCES


@pytest.mark.parametrize("rel", UNUSED)
def test_unused_file_change_keeps_the_hash(tree: Path, rel: str) -> None:
    before = _asset_hash(tree)
    _touch(tree, rel)
    assert _asset_hash(tree) == before, f"{rel} is not in the image but changes its asset hash"


@pytest.mark.parametrize("source", COPY_SOURCES)
def test_copied_file_change_changes_the_hash(tree: Path, source: str) -> None:
    """Every COPY source is in the context: a change under it changes the hash."""
    rel = source if (REPO / source).is_file() else f"{source.rstrip('/')}/new_module.py"
    before = _asset_hash(tree)
    _touch(tree, rel)
    assert _asset_hash(tree) != before, f"{rel} is copied into the image but is excluded from the hash"


def test_dockerfile_change_changes_the_hash(tree: Path) -> None:
    before = _asset_hash(tree)
    _touch(tree, DOCKERFILE_REL)
    assert _asset_hash(tree) != before
