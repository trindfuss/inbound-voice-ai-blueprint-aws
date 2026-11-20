"""Model failover 9.3, 9.7: the per-model prompt suite on Bedrock, for one model (on demand; ``-m prompt_suite``).

Skipped by the default ``pytest`` run (``addopts`` in ``pyproject.toml``). It runs
``scripts/prompt_suite.py`` for one model of the default list (``PROMPT_SUITE_MODEL``, default
``haiku-5-5``) in every language, with the case file's run count, and fails unless every run of
every case passes. Records go to a copy in ``tmp_path``: validating a prompt for the repository is
the script's job (its records are reviewed in a PR with the prompt). Fixture data only::

    AWS_PROFILE=<dev profile> uv run pytest -m prompt_suite -rA tests/live/test_prompt_suite.py

No property-based test (a live service).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from scripts import prompt_suite

pytestmark = pytest.mark.prompt_suite

MODEL = os.environ.get("PROMPT_SUITE_MODEL", "haiku-5-5")


def test_suite_passes_for_one_model(tmp_path: Path) -> None:
    records = tmp_path / "prompt_validation.json"
    shutil.copy(prompt_suite.RECORDS_PATH, records)
    rc = prompt_suite.main(["--model", MODEL, "--records", str(records)])
    assert rc != 2, "the suite stopped on an error (see stderr); no record written"
    assert rc == 0, f"{MODEL}: some run of some case failed (see the table and FAIL lines)"
