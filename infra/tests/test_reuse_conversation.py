"""``reuseConversation`` reaches both the front door and the bridge with the same value; on by default."""

from __future__ import annotations

import json

import pytest

from owlbank_infra.app_builder import build_app


def _envs(tmp_path, ctx):
    app, stacks = build_app(ctx, outdir=str(tmp_path))
    asm = app.synth()
    fd_env = br_env = None
    for st in ("OwlBankCompute", "OwlBankBridge"):
        tpl = asm.get_stack_by_name(st).template
        for lid, r in tpl["Resources"].items():
            if r["Type"] == "AWS::Lambda::Function" and lid.startswith("FrontdoorFn"):
                fd_env = r["Properties"]["Environment"]["Variables"]
            if r["Type"] == "AWS::ECS::TaskDefinition":
                br_env = {e["Name"]: e.get("Value") for e in r["Properties"]["ContainerDefinitions"][0]["Environment"]}
    return fd_env, br_env


@pytest.mark.parametrize("ctx,want", [({}, "true"), ({"reuseConversation": "true"}, "true"),
                                      ({"reuseConversation": "false"}, "false")])
def test_both_sides_get_the_same_value(tmp_path, ctx, want):
    fd_env, br_env = _envs(tmp_path, ctx)
    assert fd_env["OWLBANK_REUSE_CONVERSATION"] == want
    assert br_env["OWLBANK_REUSE_CONVERSATION"] == want, json.dumps(br_env)[:300]
