"""Run the admin console's own tests (``src/admin_console/tests``, Node's built-in test
runner) from the main suite: PKCE, preview rendering and placeholder insertion, form checks, API
error mapping, and the site's CSP rules. Skipped when no Node binary is available: a ``node`` on
``PATH``, else the one the infra venv ships (``nodejs-wheel-binaries``, used by CDK synth)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SITE_TESTS = REPO / "src" / "admin_console" / "tests"
INFRA_NODE = REPO / "infra" / ".venv" / "lib" / "python3.12" / "site-packages" / "nodejs_wheel" / "bin" / "node"


def _node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    return str(INFRA_NODE) if INFRA_NODE.exists() else None


def test_site_tests_exist():
    # The original suites stay; the console restyle adds more.
    assert {"api.test.js", "pkce.test.js", "preview.test.js", "site.test.js",
            "validate.test.js"} <= {p.name for p in SITE_TESTS.glob("*.test.js")}


def test_admin_console_node_tests_pass():
    node = _node()
    if node is None:
        pytest.skip("no Node binary (install one, or run `uv sync` in infra/ for its bundled node)")
    version = subprocess.run([node, "--version"], capture_output=True, text=True,  # noqa: S603 - our node
                             check=True).stdout.strip()
    if int(version.lstrip("v").split(".")[0]) < 22:
        pytest.skip(f"node {version}: the console's tests need Node 22+ (test-runner globs, WebCrypto)")
    # Node expands the glob itself (no shell).
    proc = subprocess.run([node, "--test", "src/admin_console/tests/*.test.js"], cwd=REPO,  # noqa: S603
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    assert "# fail 0" in proc.stdout or "ℹ fail 0" in proc.stdout, proc.stdout[-2000:]
