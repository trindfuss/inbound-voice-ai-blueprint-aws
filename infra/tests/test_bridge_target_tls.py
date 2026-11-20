"""The ALB reaches the bridge task over HTTPS in every exposure mode and stage.

The container serves TLS on its port with a self-signed certificate (``src/tac_bridge/serve.py``),
so the target group's protocol and its health check are both HTTPS. Also: the bridge image's base
is pinned by digest, not only by tag.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _target_groups(synth):
    return [r["Properties"] for s, _, r in synth.resources("AWS::ElasticLoadBalancingV2::TargetGroup")
            if s == "OwlBankBridge"]


def test_target_group_and_health_check_are_https(synth_any):
    tgs = _target_groups(synth_any)
    assert len(tgs) == 1
    tg = tgs[0]
    assert tg["Protocol"] == "HTTPS"
    assert tg["Port"] == 8080
    assert tg["HealthCheckProtocol"] == "HTTPS"
    assert tg["HealthCheckPath"] == "/health"


def test_dockerfile_base_image_is_pinned_by_digest():
    text = (REPO / "src" / "tac_bridge" / "Dockerfile").read_text()
    froms = [line for line in text.splitlines() if line.startswith("FROM ")]
    assert froms, "no FROM line"
    for line in froms:
        assert re.fullmatch(r"FROM python:3\.12-slim@sha256:[0-9a-f]{64}", line.strip()), line
