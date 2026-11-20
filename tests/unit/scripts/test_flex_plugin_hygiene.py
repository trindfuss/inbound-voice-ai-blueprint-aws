"""flex-plugin hygiene: runs the plugin's own check and pins the manifest."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from scripts._common import REPO_ROOT

PLUGIN = REPO_ROOT / "flex-plugin"
CHECK = PLUGIN / "scripts" / "check-hygiene.mjs"


def test_manifest_pins_flex_ui_and_node() -> None:
    pkg = json.loads((PLUGIN / "package.json").read_text())
    m = re.fullmatch(r"[\^~]?2\.(\d+)\.(\d+)", pkg["devDependencies"]["@twilio/flex-ui"])
    assert m and (int(m[1]), int(m[2])) >= (17, 1)
    # Flex UI 2.17.1 needs Node >= 20; flex-plugin-scripts 7.0.x allows up to 20.
    assert pkg["engines"]["node"] == "^20 || >=22.18"


def test_context_request_sends_both_sids() -> None:
    src = (PLUGIN / "src" / "contextApi.ts").read_text()
    assert "call_sid=" in src and "task_sid=" in src
    panel = (PLUGIN / "src" / "components" / "OwlBankContextPanel.tsx").read_text()
    assert "task?.taskSid" in panel and "dangerouslySetInnerHTML" not in panel


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_hygiene_check_passes_on_the_plugin() -> None:
    proc = subprocess.run(["node", str(CHECK)], capture_output=True, text=True, check=False)  # noqa: S603, S607 (fixed arguments)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_hygiene_check_flags_a_token(tmp_path) -> None:
    root = tmp_path / "p"
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "scripts" / "check-hygiene.mjs").write_text(CHECK.read_text())
    (root / "src" / "a.ts").write_text("const t = 'SK" + "0123456789abcdef" * 2 + "'; process.env.OTHER;\n")
    proc = subprocess.run(["node", str(root / "scripts" / "check-hygiene.mjs")], capture_output=True,  # noqa: S603, S607 (fixed arguments)
                          text=True, check=False)
    assert proc.returncode == 1
    assert "SID-like" in proc.stderr and "process.env.OTHER" in proc.stderr
