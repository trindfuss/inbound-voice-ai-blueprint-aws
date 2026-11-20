"""AgentCore hosting: the security matrix over the final templates.

The matrix is conftest's ``synth_matrix``: ``fargate`` in the four context sets and ``agentcore``
x {``vpc``, ``public``} x {dev (CloudFront mode), prod with a domain and, for ``public``,
``agentcoreNetworkModeAck=public-at-prod``}. Every assertion reads all templates of the app at once,
so a grant or rule that moves to another stack is still seen.

* No ``*`` principal outside a Deny; no ``AWS::Lambda::Url`` and no ``AuthType`` (so no
  ``AuthType NONE``); no public bucket (Block Public Access on, ``BucketOwnerEnforced``, no ACL); the
  Runtime has no ``AuthorizerConfiguration`` and no ``RequestHeaderConfiguration`` (IAM SigV4 only); in ``vpc`` its security group has egress on 443 only and no ingress. The only way into a
  Lambda from outside is the one API Gateway stage, behind its WAF.
* Across every template, the Runtime actions are held by exactly three roles: the presign role
  (``InvokeAgentRuntimeWithWebSocketStream``, its one statement, trust only the front-door role), the
  warmer (``InvokeAgentRuntime``) and the front door (``StopRuntimeSession``), each on exactly {R, E};
  only the front door may assume the presign role; the execution role is the Fargate task role plus
  the Runtime's additions. On ``fargate`` none of this exists.
* Every acknowledgement ``agentcore`` adds to the Fargate app of the same stage has a written
  reason, and no template carries a cdk-nag metadata suppression. (cdk-nag clean in every context:
  ``test_synth_and_nag.py``; the AWS defaults in both hostings: ``test_aws_defaults.py``.)

No PBT: the matrix is exhaustive over the supported contexts; the property tests of the hosting config and the template layout are
in ``test_stage2_config_props.py`` and ``test_bridge_agentcore.py``.
"""

from __future__ import annotations

import json

import pytest
import test_aws_defaults as cp11
from conftest import MATRIX
from test_bridge_agentcore import ADDED_SIDS, LOG_SIDS

ACCOUNT, REGION = "111111111111", "us-east-1"
BRIDGE, COMPUTE = "OwlBankBridge", "OwlBankCompute"
RUNTIME = "AWS::BedrockAgentCore::Runtime"
WS, INVOKE, STOP = (f"bedrock-agentcore:{a}" for a in
                    ("InvokeAgentRuntimeWithWebSocketStream", "InvokeAgentRuntime", "StopRuntimeSession"))


@pytest.fixture(scope="module", params=sorted(k for k in MATRIX if k.startswith("agentcore-")))
def ac(request):
    return request.getfixturevalue(MATRIX[request.param])


@pytest.fixture
def fargate(ac, request):
    """The Fargate synth with the same stage and domain mode as ``ac``."""
    return request.getfixturevalue("synth_prod" if ac.cfg.is_prod else "synth_default")


# --------------------------------------------------------------------------- helpers

def _actions(st) -> list[str]:
    a = st.get("Action", [])
    return a if isinstance(a, list) else [a]


def _resources(st) -> list:
    r = st.get("Resource", [])
    return r if isinstance(r, list) else [r]


def _role_names(synth) -> dict[tuple[str, str], str]:
    """(stack, logical id) -> a readable name for every role: its RoleName, or the logical ID."""
    return {(s, lid): r["Properties"].get("RoleName", lid) for s, lid, r in synth.resources("AWS::IAM::Role")}


def _attached(synth):
    """(role name, statement) for every IAM::Policy statement, by the roles it is attached to."""
    names = _role_names(synth)
    for stack, _, res in synth.resources("AWS::IAM::Policy"):
        for ref in res["Properties"].get("Roles", []):
            # A local Ref, or a role name string when the role lives in another stack.
            role = names[(stack, ref["Ref"])] if isinstance(ref, dict) else ref
            for st in res["Properties"]["PolicyDocument"]["Statement"]:
                yield role, st


def _role_props(synth, name: str) -> dict:
    (props,) = [synth.templates[s]["Resources"][lid]["Properties"]
                for (s, lid), n in _role_names(synth).items() if n == name]
    return props


def _warmer_role(synth) -> str:
    (fn,) = [r for s, lid, r in synth.resources("AWS::Lambda::Function")
             if s == COMPUTE and lid.startswith("RuntimeWarmerFn")]
    return _role_names(synth)[(COMPUTE, fn["Properties"]["Role"]["Fn::GetAtt"][0])]


def _arn(tail: str) -> dict:
    return {"Fn::Join": ["", ["arn:", {"Ref": "AWS::Partition"}, tail]]}


def _runtime(synth) -> tuple[str, dict]:
    found = [(lid, r) for s, lid, r in synth.resources(RUNTIME)]
    assert len(found) == 1 and synth.templates[BRIDGE]["Resources"].get(found[0][0]) is not None, found
    return found[0]


def _r_and_e(synth, stack: str) -> list:
    """{R, E} as each stack spells them: the Runtime's ARN in Bridge; its fixed name prefix in
    Compute, which can't reference Bridge."""
    if stack == BRIDGE:
        arn = {"Fn::GetAtt": [_runtime(synth)[0], "AgentRuntimeArn"]}
        return [arn, {"Fn::Join": ["", [arn, "/runtime-endpoint/DEFAULT"]]}]
    r = f":bedrock-agentcore:{REGION}:{ACCOUNT}:runtime/{synth.cfg.runtime_name}-*"
    return [_arn(r), _arn(r + "/runtime-endpoint/DEFAULT")]



def test_no_star_principal_outside_a_deny(synth_matrix):
    for stack, lid, _, st in cp11._statements_typed(synth_matrix):
        assert "NotPrincipal" not in st, (stack, lid, st)
        if cp11._has_star(st.get("Principal")):
            assert st["Effect"] == "Deny", (stack, lid, st)


def test_no_function_url_and_no_auth_type(synth_matrix):
    """7.9: no ``AWS::Lambda::Url``, and no ``AuthType``/``FunctionUrlAuthType`` key anywhere, so no
    ``AuthType NONE`` (API Gateway's ``AuthorizationType`` is a different property)."""
    assert not list(synth_matrix.resources("AWS::Lambda::Url"))
    for stack, lid, res in synth_matrix.resources():
        for path, node in cp11._walk(res.get("Properties", {})):
            if isinstance(node, dict):
                assert not {"AuthType", "FunctionUrlAuthType"} & set(node), (stack, lid, path)


def test_no_public_bucket(synth_matrix):
    for stack, lid, res in synth_matrix.resources("AWS::S3::Bucket"):
        p = res["Properties"]
        assert "AccessControl" not in p and "WebsiteConfiguration" not in p, (stack, lid)
        assert p["OwnershipControls"]["Rules"] == [{"ObjectOwnership": "BucketOwnerEnforced"}], (stack, lid)
        if synth_matrix.bpa_mode == "explicit":
            assert p["PublicAccessBlockConfiguration"] == cp11.BPA_ALL_ON, (stack, lid)
    for stack, lid, rtype, st in cp11._statements_typed(synth_matrix):
        if rtype == "AWS::S3::BucketPolicy" and st["Effect"] == "Allow":
            assert not cp11._has_star(st.get("Principal")), (stack, lid, st)
    assert not list(synth_matrix.resources("AWS::S3::AccessPoint"))


def test_runtime_has_no_authorizer_and_no_header_allow_list(synth_matrix):
    """4.5: IAM (SigV4) only, at any depth of the Runtime's properties; no extra endpoint."""
    if not synth_matrix.cfg.is_agentcore:
        assert not [t for _, _, r in synth_matrix.resources() if (t := r["Type"]).startswith(RUNTIME)]
        return
    _, rt = _runtime(synth_matrix)
    for path, node in cp11._walk(rt["Properties"]):
        if isinstance(node, dict):
            assert not {"AuthorizerConfiguration", "RequestHeaderConfiguration"} & set(node), path
    assert not list(synth_matrix.resources("AWS::BedrockAgentCore::RuntimeEndpoint"))


def test_runtime_security_group_egress_443_only_and_no_ingress(ac):
    net = _runtime(ac)[1]["Properties"]["NetworkConfiguration"]
    if ac.cfg.agentcore_network_mode == "public":
        assert net == {"NetworkMode": "PUBLIC"}
        assert not list(ac.resources("AWS::EC2::SecurityGroup"))
        return
    assert net["NetworkMode"] == "VPC"
    (ref,) = net["NetworkModeConfig"]["SecurityGroups"]
    sg_lid = ref["Fn::GetAtt"][0]
    sg = ac.templates[BRIDGE]["Resources"][sg_lid]["Properties"]
    assert sg["SecurityGroupEgress"] == [{
        "CidrIp": "0.0.0.0/0", "Description": "HTTPS to Twilio and AWS APIs",
        "FromPort": 443, "IpProtocol": "tcp", "ToPort": 443}]
    assert "SecurityGroupIngress" not in sg
    # No separate rule resource adds to it, in any stack.
    for rtype in ("AWS::EC2::SecurityGroupIngress", "AWS::EC2::SecurityGroupEgress"):
        for stack, lid, r in ac.resources(rtype):
            assert sg_lid not in json.dumps(r["Properties"].get("GroupId")), (stack, lid)


def test_lambdas_reachable_only_through_the_api_stage_behind_its_waf(synth_matrix):
    """7.9: one REST API; its stage has the WAF association; every Lambda permission is for
    API Gateway on that API (no other public entry to a function, in either hosting)."""
    (api_stack, api_lid, _), = synth_matrix.resources("AWS::ApiGateway::RestApi")
    stage_lid = cp11._logical_id(synth_matrix, api_stack, "ApiDeploymentStage", "AWS::ApiGateway::Stage")
    arns = [json.dumps(r["Properties"]["ResourceArn"]) for s, _, r in
            synth_matrix.resources("AWS::WAFv2::WebACLAssociation") if s == api_stack]
    assert any(f'"Ref": "{api_lid}"' in a and f'"Ref": "{stage_lid}"' in a for a in arns), arns
    for stack, lid, r in synth_matrix.resources("AWS::Lambda::Permission"):
        p = r["Properties"]
        if p["Principal"] == "events.amazonaws.com":  # the model-failover poller's schedule only
            assert synth_matrix.cfg.failover_on and lid.startswith("ModelPollerSchedule"), (stack, lid)
            assert p["SourceArn"]["Fn::GetAtt"][0].startswith("ModelPollerSchedule"), (stack, lid)
            continue
        assert p["Principal"] == "apigateway.amazonaws.com", (stack, lid)
        assert stack == api_stack and f'"Ref": "{api_lid}"' in json.dumps(p["SourceArn"]), (stack, lid)
    if synth_matrix.cfg.is_agentcore:
        paths = {r["Properties"]["PathPart"] for _, _, r in synth_matrix.resources("AWS::ApiGateway::Resource")}
        assert "orchestrator" in paths  # the orchestrator sink is on this API, not a Function URL



def test_runtime_actions_held_only_by_their_three_roles(synth_matrix):
    """3.3, 7.1: who may call which Runtime API, over every template. Any ``bedrock-agentcore``
    action on a Runtime, a wildcard included, counts."""
    held: dict[str, set[str]] = {}
    for role, st in _attached(synth_matrix):
        runtime_actions = {a for a in _actions(st) if a.startswith("bedrock-agentcore:")
                           and ("Runtime" in a or "*" in a)}
        if runtime_actions:
            assert st["Effect"] == "Allow" and "Condition" not in st, (role, st)
            held.setdefault(role, set()).update(runtime_actions)
    if not synth_matrix.cfg.is_agentcore:
        assert held == {}
        return
    cfg = synth_matrix.cfg
    assert held == {cfg.presign_role_name: {WS}, _warmer_role(synth_matrix): {INVOKE},
                    cfg.frontdoor_role_name: {STOP}}, held


def test_runtime_grants_name_exactly_r_and_e(ac):
    stacks = {ac.cfg.presign_role_name: BRIDGE, _warmer_role(ac): COMPUTE, ac.cfg.frontdoor_role_name: COMPUTE}
    seen = set()
    for role, st in _attached(ac):
        if {WS, INVOKE, STOP} & set(_actions(st)):
            assert len(_actions(st)) == 1, (role, st)
            assert _resources(st) == _r_and_e(ac, stacks[role]), (role, st)
            seen.add(role)
    assert seen == set(stacks)


def test_presign_role_one_statement_and_trusts_only_the_front_door(ac):
    cfg = ac.cfg
    mine = [st for role, st in _attached(ac) if role == cfg.presign_role_name]
    assert [(st["Effect"], _actions(st)) for st in mine] == [("Allow", [WS])]
    role = _role_props(ac, cfg.presign_role_name)
    assert "ManagedPolicyArns" not in role and "Policies" not in role
    (trust,) = role["AssumeRolePolicyDocument"]["Statement"]
    assert trust["Effect"] == "Allow" and trust["Action"] == "sts:AssumeRole"
    assert trust["Principal"] == {"AWS": _arn(f":iam::{ACCOUNT}:root")}
    assert trust["Condition"] == {"ArnEquals": {"aws:PrincipalArn": _arn(f":iam::{ACCOUNT}:role/{cfg.frontdoor_role_name}")}}


def test_only_the_front_door_may_assume_the_presign_role(synth_matrix):
    """Any identity policy granting ``sts:AssumeRole`` (fargate: none at all)."""
    holders = {(role, json.dumps(_resources(st))) for role, st in _attached(synth_matrix)
               if any(a in ("sts:AssumeRole", "sts:*", "*") for a in _actions(st))}
    if not synth_matrix.cfg.is_agentcore:
        assert holders == set()
        return
    cfg = synth_matrix.cfg
    assert holders == {(cfg.frontdoor_role_name,
                        json.dumps([_arn(f":iam::{ACCOUNT}:role/{cfg.presign_role_name}")]))}


def test_warmer_holds_only_invoke_among_agentcore_actions(ac):
    """The warmer: only ``InvokeAgentRuntime`` among ``bedrock-agentcore:*``, and no managed or
    inline policy that could add one."""
    warmer = _warmer_role(ac)
    agentcore = {a for role, st in _attached(ac) if role == warmer for a in _actions(st)
                 if a.startswith("bedrock-agentcore:")}
    assert agentcore == {INVOKE}
    props = _role_props(ac, warmer)
    assert "ManagedPolicyArns" not in props and "Policies" not in props


def test_execution_role_is_the_fargate_task_role_plus_the_additions(ac, fargate):
    """From every template, the execution role's statements are the Fargate task role's
    (identical) plus the Runtime's added Sids; no managed or inline policy; only AgentCore may assume it,
    and besides it only the Agent stack's Gateway role trusts AgentCore."""
    name = ac.cfg.bridge_task_role_name
    mine = {st["Sid"]: st for role, st in _attached(ac) if role == name}
    theirs = {st["Sid"]: st for role, st in _attached(fargate) if role == name}
    assert {s: mine.get(s) for s in theirs} == theirs
    assert set(mine) - set(theirs) == ADDED_SIDS | LOG_SIDS
    props = _role_props(ac, name)
    assert "ManagedPolicyArns" not in props and "Policies" not in props
    names = _role_names(ac)
    (gateway,) = [n for (s, lid), n in names.items() if s == "OwlBankAgent" and lid.startswith("GatewayRole")]
    trusted = {}
    for (stack, lid), role_name in names.items():
        trust = ac.templates[stack]["Resources"][lid]["Properties"]["AssumeRolePolicyDocument"]["Statement"]
        services = {st["Principal"].get("Service") for st in trust}
        if "bedrock-agentcore.amazonaws.com" in services:
            assert services == {"bedrock-agentcore.amazonaws.com"}, role_name
            trusted[role_name] = stack
    assert trusted == {name: BRIDGE, gateway: "OwlBankAgent"}


# --------------------------------------------------------------------------- 7.8 acknowledgements

def test_new_acknowledgements_have_written_reasons(ac, fargate):
    """What ``agentcore`` acknowledges beyond the Fargate app of the same stage (the Runtime log
    streams, the Runtime name prefix, the warmer's role...): each with a written reason."""
    theirs = {(path, rule) for path, rule, _ in fargate.acks}
    new = [(path, rule, why) for path, rule, why in ac.acks if (path, rule) not in theirs]
    assert new
    for path, rule, why in new:
        assert rule.startswith("AwsSolutions-") and len(why.strip()) >= 40, (path, rule)


def test_no_metadata_suppression_in_any_template(synth_matrix):
    """Every acknowledgement goes through ``nag.ack`` (which needs a reason); none hides in a
    template's ``cdk_nag`` metadata."""
    for stack, lid, res in synth_matrix.resources():
        assert "cdk_nag" not in json.dumps(res.get("Metadata", {})), (stack, lid)
