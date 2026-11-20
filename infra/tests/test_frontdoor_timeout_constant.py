"""The front door's request-timeout constant (briefing budget cap) equals its Lambda timeout.

No PBT: one equality between a source constant and an infra table.
"""

from __future__ import annotations

import re
from pathlib import Path

from owlbank_infra.compute_stack import TIMEOUTS

APP = Path(__file__).resolve().parents[2] / "src" / "lambdas" / "frontdoor" / "app.py"


def test_request_timeout_ms_matches_the_frontdoor_lambda_timeout() -> None:
    found = re.search(r"^REQUEST_TIMEOUT_MS = ([\d_]+)$", APP.read_text(encoding="utf-8"), re.M)
    assert found is not None
    assert int(found.group(1).replace("_", "")) == TIMEOUTS["frontdoor"] * 1000
