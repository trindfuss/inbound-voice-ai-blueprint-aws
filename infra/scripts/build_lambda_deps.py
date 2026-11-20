#!/usr/bin/env python3
"""Build the Lambda dependency layers into ``infra/build``. Deploy-time only.

Synth never runs this (it ships an empty layer if a directory is empty).

* ``lambda-deps/python``: ``twilio``, the one package the Lambdas need beyond the runtime's boto3.
* ``lambda-otel/python``: OpenTelemetry for the front door (call observability), attached
  only with ``callObservability=on``. Pinned from the root ``pyproject.toml``'s ``lambda-otel`` group.

Every package, transitive ones included, is pinned with hashes from the repo's ``uv.lock``
(``uv export``, filtered to each layer's dependency closure) and installed with ``--require-hashes``. Wheels are for Linux arm64 / CPython 3.12
(the functions' architecture and runtime), binary only.

    cd infra && uv run python scripts/build_lambda_deps.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

INFRA = Path(__file__).resolve().parents[1]
LOCK = INFRA.parent / "uv.lock"
#: Layer directory (under ``infra/build``) -> packages pinned from ``uv.lock``.
LAYERS: dict[str, tuple[str, ...]] = {
    "lambda-deps": ("twilio",),
    "lambda-otel": ("opentelemetry-sdk", "opentelemetry-exporter-otlp-proto-common",
                    "opentelemetry-sdk-extension-aws", "opentelemetry-propagator-aws-xray"),
}


def lock_graph() -> dict[str, list[str]]:
    """Package name -> names of its dependencies, from ``uv.lock``."""
    pkgs = tomllib.loads(LOCK.read_text())["package"]
    return {p["name"]: [d["name"] for d in p.get("dependencies", [])] for p in pkgs}


def closure(roots: tuple[str, ...], graph: dict[str, list[str]]) -> set[str]:
    """``roots`` plus every transitive dependency."""
    seen: set[str] = set()
    todo = list(roots)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        if name not in graph:
            sys.exit(f"{name} not found in {LOCK}")
        seen.add(name)
        todo.extend(graph[name])
    return seen


def export_all() -> str:
    """Every locked package with hashes, as a requirements file (project itself excluded)."""
    cmd = ["uv", "export", "--frozen", "--all-groups", "--no-emit-project", "--no-header", "--no-annotate"]
    return subprocess.run(cmd, check=True, cwd=LOCK.parent, capture_output=True, text=True).stdout  # noqa: S603


def _entry_name(entry: str) -> str:
    m = re.match(r"[A-Za-z0-9_.-]+", entry)
    return m.group(0).lower().replace("_", "-") if m else ""


def layer_requirements(exported: str, names: set[str]) -> str:
    """The hashed requirement entries of ``exported`` whose package is in ``names``."""
    entries = re.split(r"\n(?=\S)", exported.strip())
    kept = [e for e in entries if _entry_name(e) in names]
    missing = names - {_entry_name(e) for e in kept}
    if missing:
        sys.exit(f"not exported with hashes: {sorted(missing)}")
    return "\n".join(kept) + "\n"


def install_command(target: Path, requirements: Path) -> list[str]:
    return ["uv", "pip", "install", "--target", str(target), "--python-platform", "aarch64-manylinux2014",
            "--python-version", "3.12", "--only-binary", ":all:", "--no-deps", "--require-hashes",
            "-r", str(requirements)]


def build(layer: str, packages: tuple[str, ...], exported: str, graph: dict[str, list[str]]) -> None:
    root = INFRA / "build" / layer
    target = root / "python"
    if root.exists():
        shutil.rmtree(root)
    target.mkdir(parents=True)
    requirements = root / "requirements.txt"
    requirements.write_text(layer_requirements(exported, closure(packages, graph)))
    cmd = install_command(target, requirements)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603
    requirements.unlink()


def main() -> None:
    exported, graph = export_all(), lock_graph()
    for layer, packages in LAYERS.items():
        build(layer, packages, exported, graph)


if __name__ == "__main__":
    main()
