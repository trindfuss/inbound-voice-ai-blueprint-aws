#!/usr/bin/env python3
"""CDK entry point: ``uv run python app.py`` synthesizes to ``cdk.out``. Never deploys."""

import owlbank_infra.node_bootstrap  # noqa: F401  (puts the bundled node on PATH before aws_cdk loads)
from owlbank_infra.app_builder import build_app  # noqa: E402

app, _ = build_app(outdir="cdk.out")
app.synth()
