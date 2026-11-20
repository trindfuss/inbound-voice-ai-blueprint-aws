"""Human-leg conversation infrastructure.

Template assertions over offline synths:

* ``humanLegConversation`` (default ``on``): ``POST /voice/taskrouter`` and ``POST /voice/transcription``
  on the front door, Twilio webhooks like ``/voice/status`` (no API Gateway authorizer; the handler
  checks the signature and Basic credentials), in the WAF allow-list. ``off`` removes both, except
  that ``callObservability=on`` keeps ``taskrouter`` for its own telemetry.
* The Context API reads exactly one secret, by ARN: its own ``context_api_key`` secret with
  ``contextApiRestrictedKey=on`` (the default; created with a generated placeholder, the value
  written by scripts/twilio_setup.py, never by the template), or ``twilio_api_key`` with it ``off``.
* Env: ``OWLBANK_TASKROUTER_WORKSPACE_SID`` (from ``twilioOutputs.workspaceSid``) and
  ``OWLBANK_SECRET_TWILIO_API_KEY`` on the Context API; ``OWLBANK_HUMAN_LEG_CONVERSATION`` and, when
  on, the workspace SID on the front door.
* WAF: the per-IP rate rule has no scope-down and the ACL is associated with the stage, so
  it covers every path, ``/context/*`` included.
* Stale-conversation sweep: with the flag on, one ``conversation_sweep``
  Lambda (no URL, no API, no trigger but its schedule), an EventBridge Scheduler schedule every 10
  minutes whose role may invoke that one function ARN, the ``twilio_api_key`` secret by its one ARN
  and the CMK's decrypt through Secrets Manager, the ``calls`` table, and an Errors alarm; no wildcard
  action or resource in the new grants; ``off`` removes all of it.

PBT: skipped (config and IAM glue; fixed policy assertions over a small context matrix).
"""

from __future__ import annotations

import json

import pytest
from conftest import _synth
from test_synth_and_nag import _nag_report

from owlbank_infra.api_stack import TASKROUTER_ROUTE, TRANSCRIPTION_ROUTE, VOICE_ROUTES, voice_routes
from owlbank_infra.compute_stack import (
    CONVERSATION_SWEEP_RATE,
    HUMAN_LEG_ENV,
    SWEEP_INDEX_ACTIONS,
    SWEEP_INDEX_SID,
    SWEEP_TABLE_ACTIONS,
    TASKROUTER_WORKSPACE_ENV,
)
from owlbank_infra.config import ConfigError, from_context
from owlbank_infra.data_stack import CONTEXT_API_KEY_SECRET, OPTIONAL_SECRETS

API, COMPUTE, DATA = "OwlBankApi", "OwlBankCompute", "OwlBankData"
WORKSPACE = "WS" + "4" * 32
CONFIGURATION = "conv_configuration_01xyz"
SECRET_ACTIONS = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]


def _outputs(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("hl-tw") / "twilio.context.json"
    path.write_text(json.dumps({"accountSid": "AC" + "1" * 32, "workflowSid": "WW" + "2" * 32,
                                "workspaceSid": WORKSPACE, "conversationConfigurationId": CONFIGURATION}))
    return str(path)


@pytest.fixture(scope="module")
def hl(tmp_path_factory) -> dict:
    """One synth per context, built once for the module (``default`` is the shared fixture's)."""
    tw = _outputs(tmp_path_factory)
    contexts = {
        "off": {"twilioOutputs": tw, "humanLegConversation": "off"},
        "off-obs": {"twilioOutputs": tw, "humanLegConversation": "off", "callObservability": "on"},
        "noflex-obs": {"callObservability": "on"},
        "ws": {"twilioOutputs": tw},
        "ws-off": {"twilioOutputs": tw, "humanLegConversation": "off"},
        "restricted": {"twilioOutputs": tw, "contextApiRestrictedKey": "on"},
        "shared": {"twilioOutputs": tw, "contextApiRestrictedKey": "off"},
    }
    return {name: _synth(tmp_path_factory, f"hl-{name}", ctx) for name, ctx in contexts.items()}


# --------------------------------------------------------------------------- helpers

def _res(synth, stack: str) -> dict:
    return synth.templates[stack]["Resources"]


def _one(synth, stack: str, prefix: str, rtype: str) -> tuple[str, dict]:
    found = [(lid, r) for lid, r in _res(synth, stack).items()
             if r["Type"] == rtype and lid.startswith(prefix)]
    assert len(found) == 1, (prefix, [lid for lid, _ in found])
    return found[0]


def _voice_methods(synth) -> dict[str, dict]:
    """``/voice/<route>`` -> its method's properties."""
    res = _res(synth, API)
    voice = next(lid for lid, r in res.items() if r["Type"] == "AWS::ApiGateway::Resource"
                 and r["Properties"]["PathPart"] == "voice")
    routes = {lid: r["Properties"]["PathPart"] for lid, r in res.items()
              if r["Type"] == "AWS::ApiGateway::Resource"
              and r["Properties"].get("ParentId") == {"Ref": voice}}
    out: dict[str, dict] = {}
    for r in res.values():
        if r["Type"] == "AWS::ApiGateway::Method":
            ref = r["Properties"].get("ResourceId", {}).get("Ref")
            if ref in routes:
                assert routes[ref] not in out, routes[ref]
                out[routes[ref]] = r["Properties"]
    return out


def _acl(synth) -> dict:
    return _one(synth, API, "WebAcl", "AWS::WAFv2::WebACL")[1]["Properties"]


def _allow_list(synth) -> str:
    rule = next(r for r in _acl(synth)["Rules"] if r["Name"] == "AllowedPathsAndMethodsOnly")
    return json.dumps(rule)


def _env(synth, fn_prefix: str) -> dict:
    return _one(synth, COMPUTE, fn_prefix, "AWS::Lambda::Function")[1]["Properties"]["Environment"]["Variables"]


def _role_statements(synth, role_prefix: str) -> list[dict]:
    role, _ = _one(synth, COMPUTE, role_prefix, "AWS::IAM::Role")
    out: list[dict] = []
    for r in _res(synth, COMPUTE).values():
        if r["Type"] == "AWS::IAM::Policy" and {"Ref": role} in r["Properties"].get("Roles", []):
            out += r["Properties"]["PolicyDocument"]["Statement"]
    return out


def _actions(st: dict) -> list[str]:
    return st["Action"] if isinstance(st["Action"], list) else [st["Action"]]


def _secret_statements(synth, role_prefix: str) -> list[dict]:
    return [st for st in _role_statements(synth, role_prefix)
            if any(a.startswith("secretsmanager:") for a in _actions(st))]


def _export_of(synth, secret_logical: str) -> str:
    """The Data stack's export name for ``secret_logical``'s ARN (``Ref``)."""
    lid = next(lid for lid, r in _res(synth, DATA).items() if r["Type"] == "AWS::SecretsManager::Secret"
               and lid.startswith("".join(p.capitalize() for p in secret_logical.split("_")) + "Secret"))
    names = [o["Export"]["Name"] for o in synth.templates[DATA].get("Outputs", {}).values()
             if o.get("Value") == {"Ref": lid} and "Export" in o]
    assert len(names) == 1, (secret_logical, names)
    return names[0]


# --------------------------------------------------------------------------- config

def _flex_ctx(tmp_path, **ctx) -> dict:
    path = tmp_path / "flex.json"
    path.write_text(json.dumps({"workflowSid": "WW" + "2" * 32, "workspaceSid": WORKSPACE}))
    return {"twilioOutputs": str(path), **ctx}


def test_flag_defaults_and_values(tmp_path) -> None:
    c = from_context(_flex_ctx(tmp_path), base_dir=tmp_path)
    assert c.flex_mode and c.human_leg_conversation_on and c.context_api_restricted_key_on
    assert c.taskrouter_route_on
    off = from_context(_flex_ctx(tmp_path, humanLegConversation="off"), base_dir=tmp_path)
    assert not off.human_leg_conversation_on and not off.taskrouter_route_on
    assert from_context(_flex_ctx(tmp_path, humanLegConversation="off", callObservability="on"),
                        base_dir=tmp_path).taskrouter_route_on
    # Without Flex nothing human-leg is on, whatever the flags say.
    bare = from_context({"callObservability": "on"}, base_dir=tmp_path)
    assert not bare.flex_mode and not bare.human_leg_conversation_on and not bare.taskrouter_route_on
    assert from_context({"contextApiRestrictedKey": "ON"}, base_dir=tmp_path).context_api_restricted_key_on
    assert not from_context({"contextApiRestrictedKey": "off"}, base_dir=tmp_path).context_api_restricted_key_on


@pytest.mark.parametrize("key", ["humanLegConversation", "contextApiRestrictedKey"])
@pytest.mark.parametrize("raw", ["yes", "true", "1", "enabled"])
def test_bad_values_are_refused_naming_the_key(tmp_path, key, raw) -> None:
    with pytest.raises(ConfigError, match=key):
        from_context({key: raw}, base_dir=tmp_path)


def test_voice_routes_by_flag(tmp_path) -> None:
    def routes(**ctx):
        return voice_routes(from_context(_flex_ctx(tmp_path, **ctx), base_dir=tmp_path))
    assert routes() == (*VOICE_ROUTES, TASKROUTER_ROUTE, TRANSCRIPTION_ROUTE)
    assert routes(humanLegConversation="off") == VOICE_ROUTES
    assert routes(humanLegConversation="off", callObservability="on") == (*VOICE_ROUTES, TASKROUTER_ROUTE)
    assert routes(callObservability="on") == (*VOICE_ROUTES, TASKROUTER_ROUTE, TRANSCRIPTION_ROUTE)
    # No Flex: neither route, whatever the flags say (R 1.3).
    assert voice_routes(from_context({"callObservability": "on"}, base_dir=tmp_path)) == VOICE_ROUTES


# --------------------------------------------------------------------------- routes

def test_both_routes_present_by_default(synth_flex_any) -> None:
    methods = _voice_methods(synth_flex_any)
    assert {TASKROUTER_ROUTE, TRANSCRIPTION_ROUTE} <= set(methods)
    assert TASKROUTER_ROUTE in _allow_list(synth_flex_any) and TRANSCRIPTION_ROUTE in _allow_list(synth_flex_any)


def test_transcription_route_is_a_twilio_webhook_like_status(synth_flex_any) -> None:
    """Same front door integration, POST only, no API Gateway authorizer (the handler validates
    ``X-Twilio-Signature`` and Basic first, as for ``/voice/status``), the request validator on."""
    methods = _voice_methods(synth_flex_any)
    status, tx, tr = methods["status"], methods[TRANSCRIPTION_ROUTE], methods[TASKROUTER_ROUTE]
    for m in (tx, tr):
        assert m["HttpMethod"] == "POST"
        assert m["AuthorizationType"] == "NONE" and "AuthorizerId" not in m
        assert m["Integration"] == status["Integration"]
        assert m["RequestValidatorId"] == status["RequestValidatorId"]
    # The WAF's Basic challenge covers every /voice/ path, so these are challenged at the edge too.
    assert "/voice/" in json.dumps(_acl(synth_flex_any))


def test_queue_routes_are_voice_webhooks_on_or_off(synth_flex_any, hl) -> None:
    """The ``<Enqueue>`` ``action`` route exists with the flag
    on or off (every escalation uses it; no ``waitUrl`` route), wired like ``/voice/status`` and in the WAF allow-list."""
    for synth in (synth_flex_any, hl["off"]):
        methods = _voice_methods(synth)
        for route in ("enqueue-action",):
            assert route in VOICE_ROUTES
            m = methods[route]
            assert m["HttpMethod"] == "POST"
            assert m["AuthorizationType"] == "NONE" and "AuthorizerId" not in m
            assert m["Integration"] == methods["status"]["Integration"]
            assert m["RequestValidatorId"] == methods["status"]["RequestValidatorId"]
            assert route in _allow_list(synth)


def test_off_removes_both_routes(hl) -> None:
    methods = _voice_methods(hl["off"])
    assert set(methods) == set(VOICE_ROUTES)
    assert TASKROUTER_ROUTE not in _allow_list(hl["off"])
    assert TRANSCRIPTION_ROUTE not in _allow_list(hl["off"])


def test_off_keeps_taskrouter_for_call_observability(hl) -> None:
    methods = _voice_methods(hl["off-obs"])
    assert set(methods) == {*VOICE_ROUTES, TASKROUTER_ROUTE}
    assert TASKROUTER_ROUTE in _allow_list(hl["off-obs"])
    assert TRANSCRIPTION_ROUTE not in _allow_list(hl["off-obs"])


def test_no_new_unauthenticated_route(synth_matrix) -> None:
    """10.5: every method without an API Gateway authorizer is a Twilio webhook under /voice/ (signed,
    checked in the handler) or the Context API's CORS preflight (no data)."""
    res = _res(synth_matrix, API)
    voice = set(_voice_methods(synth_matrix))
    expected = set(voice_routes(synth_matrix.cfg)) | ({"orchestrator"} if synth_matrix.cfg.is_agentcore else set())
    assert voice == expected
    open_methods = [r["Properties"] for r in res.values() if r["Type"] == "AWS::ApiGateway::Method"
                    and r["Properties"]["AuthorizationType"] == "NONE"]
    assert sum(m["HttpMethod"] == "POST" for m in open_methods) == len(expected)
    assert sum(m["HttpMethod"] == "OPTIONS" for m in open_methods) == 1
    assert {m["HttpMethod"] for m in open_methods} == {"POST", "OPTIONS"}


# --------------------------------------------------------------------------- Context API

def test_context_api_reads_its_own_secret_by_default(synth_flex_any) -> None:
    sts = _secret_statements(synth_flex_any, "ContextApiRole")
    assert len(sts) == 1
    assert _actions(sts[0]) == SECRET_ACTIONS
    assert sts[0]["Resource"] == {"Fn::ImportValue": _export_of(synth_flex_any, CONTEXT_API_KEY_SECRET)}
    assert _env(synth_flex_any, "ContextApiFn")["OWLBANK_SECRET_TWILIO_API_KEY"] == sts[0]["Resource"]
    assert _export_of(synth_flex_any, "twilio_api_key") not in json.dumps(_role_statements(synth_flex_any, "ContextApiRole"))


def test_flag_off_reads_only_the_shared_api_key_secret_by_arn(hl) -> None:
    synth = hl["shared"]
    sts = _secret_statements(synth, "ContextApiRole")
    assert len(sts) == 1
    assert _actions(sts[0]) == SECRET_ACTIONS
    assert sts[0]["Resource"] == {"Fn::ImportValue": _export_of(synth, "twilio_api_key")}
    assert _env(synth, "ContextApiFn")["OWLBANK_SECRET_TWILIO_API_KEY"] == sts[0]["Resource"]
    assert not [r for r in _res(synth, DATA).values() if r["Type"] == "AWS::SecretsManager::Secret"
                and r["Properties"]["Name"].endswith(OPTIONAL_SECRETS[CONTEXT_API_KEY_SECRET][0])]


@pytest.mark.parametrize("name", ["default", "restricted"])
def test_restricted_key_secret_replaces_the_shared_one(hl, synth_flex, name) -> None:
    synth = synth_flex if name == "default" else hl[name]
    sts = _secret_statements(synth, "ContextApiRole")
    assert len(sts) == 1 and _actions(sts[0]) == SECRET_ACTIONS
    own = {"Fn::ImportValue": _export_of(synth, CONTEXT_API_KEY_SECRET)}
    assert sts[0]["Resource"] == own
    env = _env(synth, "ContextApiFn")
    assert env["OWLBANK_SECRET_TWILIO_API_KEY"] == own
    # The shared key is not reachable from the Context API's role any more.
    assert _export_of(synth, "twilio_api_key") not in json.dumps(_role_statements(synth, "ContextApiRole"))
    # Other functions keep the shared secret.
    fd = _secret_statements(synth, "FrontdoorRole")
    assert own not in [st["Resource"] for st in fd]


def test_restricted_key_secret_is_a_placeholder_with_no_value(hl) -> None:
    """scripts/twilio_setup.py writes the value; the template holds only a generated placeholder
    (no SecretString), with the data CMK, retained, and an ARN output."""
    synth = hl["restricted"]
    _, secret = _one(synth, DATA, "ContextApiKeySecret", "AWS::SecretsManager::Secret")
    props = secret["Properties"]
    assert props["Name"] == "owlbank/twilio/context-api-key"
    assert "SecretString" not in props
    assert props["GenerateSecretString"]["SecretStringTemplate"] == '{"placeholder": true}'
    assert "KmsKeyId" in props
    assert secret["DeletionPolicy"] == "Retain"
    assert "SecretContextApiKey" in synth.templates[DATA]["Outputs"]


def test_secret_decrypt_only_through_secrets_manager(hl, synth_default) -> None:
    for synth in (synth_default, hl["restricted"], hl["shared"]):
        kms = [st for st in _role_statements(synth, "ContextApiRole")
               if st.get("Sid") == "DecryptSecretsViaSecretsManager"]
        assert len(kms) == 1 and _actions(kms[0]) == ["kms:Decrypt"]
        assert kms[0]["Condition"] == {"StringEquals": {"kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"}}
        assert "*" not in json.dumps(kms[0]["Resource"])


@pytest.mark.parametrize("name", ["default", "ws", "restricted", "shared"])
def test_no_wildcard_in_the_new_grants(hl, synth_flex, name) -> None:
    """The statements this task adds (the secret read and its decrypt) name one ARN each
    and list their actions; the role's only ``*`` resources stay X-Ray and Comprehend."""
    synth = synth_flex if name == "default" else hl[name]
    for st in _role_statements(synth, "ContextApiRole"):
        acts = _actions(st)
        assert not [a for a in acts if "*" in a], st
        if st["Resource"] == "*":
            assert all(a.startswith(("xray:", "comprehend:")) for a in acts), st
        if any(a.startswith(("secretsmanager:", "kms:")) for a in acts):
            assert isinstance(st["Resource"], dict) and "*" not in json.dumps(st["Resource"]), st


def test_cdk_nag_clean_with_the_restricted_key(hl) -> None:
    report = _nag_report(hl["restricted"])
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]


# --------------------------------------------------------------------------- env

def test_workspace_sid_reaches_the_context_api_and_front_door(hl) -> None:
    assert _env(hl["ws"], "ContextApiFn")[TASKROUTER_WORKSPACE_ENV] == WORKSPACE
    assert _env(hl["ws"], "FrontdoorFn")[TASKROUTER_WORKSPACE_ENV] == WORKSPACE
    assert _env(hl["restricted"], "ContextApiFn")[TASKROUTER_WORKSPACE_ENV] == WORKSPACE


def test_workspace_sid_absent_without_twilio_outputs(synth_default) -> None:
    """No ``workspaceSid``: the variable is left out and the Context API fails closed (503)."""
    assert TASKROUTER_WORKSPACE_ENV not in _env(synth_default, "ContextApiFn")
    assert TASKROUTER_WORKSPACE_ENV not in _env(synth_default, "FrontdoorFn")


def test_front_door_human_leg_switch(hl, synth_flex, synth_default) -> None:
    assert _env(synth_flex, "FrontdoorFn")[HUMAN_LEG_ENV] == "on"
    assert _env(synth_default, "FrontdoorFn")[HUMAN_LEG_ENV] == "off"  # no Flex: derived, not the raw flag
    assert _env(hl["off"], "FrontdoorFn")[HUMAN_LEG_ENV] == "off"
    # Off: the Context API keeps its per-task check (workspace), the front door gets no workspace.
    assert _env(hl["ws-off"], "ContextApiFn")[TASKROUTER_WORKSPACE_ENV] == WORKSPACE
    assert TASKROUTER_WORKSPACE_ENV not in _env(hl["ws-off"], "FrontdoorFn")
    # The flag adds no front-door IAM: its statements are the same on and off.
    assert _role_statements(hl["off"], "FrontdoorRole") == _role_statements(synth_flex, "FrontdoorRole")


# --------------------------------------------------------------------------- WAF

def _transcription_regex(stmt: dict) -> str:
    return stmt["RegexMatchStatement"]["RegexString"]


def test_waf_rate_rule_covers_the_context_api(synth_matrix) -> None:
    """The per-IP rate rule counts every path but ``/voice/transcription`` (that one has its
    own count, below), and the ACL is attached to the API's stage, so ``/context/*`` (its 404s and
    authorizer Denies) is counted. The allow-list lets ``/context/<ref>`` through to it."""
    acl = _acl(synth_matrix)
    rate = next(r for r in acl["Rules"] if r["Name"] == "RateLimitPerIp")
    stmt = rate["Statement"]["RateBasedStatement"]
    assert stmt["AggregateKeyType"] == "IP"
    if synth_matrix.cfg.human_leg_conversation_on:
        excluded = stmt["ScopeDownStatement"]["NotStatement"]["Statement"]
        assert _transcription_regex(excluded) == "^(/v1)?/voice/transcription$"
    else:
        assert "ScopeDownStatement" not in stmt
    assert rate["Action"] == {"Block": {}}
    assert stmt["Limit"] == synth_matrix.cfg.waf_api_rate_limit
    _, assoc = _one(synth_matrix, API, "WebAclAssociation", "AWS::WAFv2::WebACLAssociation")
    arn = json.dumps(assoc["Properties"]["ResourceArn"])
    assert "/restapis/" in arn and "/stages/" in arn
    assert assoc["Properties"]["WebACLArn"] == {"Fn::GetAtt": ["WebAcl", "Arn"]}
    assert "/context/" in _allow_list(synth_matrix)


def test_transcript_callbacks_have_their_own_rate_count(synth_matrix) -> None:
    """Transcript callbacks (every transcript event, both tracks) are counted per IP by their own
    rule, with the same limit, so they cannot use up the voice webhooks' count from the same Twilio
    IP. Without the human leg there is no such route and no such rule."""
    rules = {r["Name"]: r for r in _acl(synth_matrix)["Rules"]}
    if not synth_matrix.cfg.human_leg_conversation_on:
        assert "RateLimitPerIpOwnPaths" not in rules
        return
    own = rules["RateLimitPerIpOwnPaths"]
    stmt = own["Statement"]["RateBasedStatement"]
    assert stmt["AggregateKeyType"] == "IP" and stmt["Limit"] == synth_matrix.cfg.waf_api_rate_limit
    assert _transcription_regex(stmt["ScopeDownStatement"]) == "^(/v1)?/voice/transcription$"
    assert own["Action"] == {"Block": {}}
    assert len({r["Priority"] for r in rules.values()}) == len(rules)
    # Right after the per-IP count, before the body, Basic-auth and other custom rules.
    assert own["Priority"] == rules["RateLimitPerIp"]["Priority"] + 1
    after = [r for r in rules.values() if r["Priority"] < 10 and r["Name"] not in
             ("AllowedPathsAndMethodsOnly", "RateLimitPerIp", "RateLimitPerIpOwnPaths")]
    assert after and all(r["Priority"] > own["Priority"] for r in after)
    assert rules["AuthorizationHeaderRequired"]["Priority"] > own["Priority"]


# --------------------------------------------------------------------------- stale-conversation sweep

SWEEP_FN, SWEEP_ROLE, SCHED_ROLE = "ConversationSweepFn", "ConversationSweepRole", "ConversationSweepSchedulerRole"


def _of_type(synth, rtype: str, prefix: str = "") -> list[tuple[str, dict]]:
    return [(lid, r) for lid, r in _res(synth, COMPUTE).items() if r["Type"] == rtype and lid.startswith(prefix)]


def _sweep_role_statements(synth) -> list[dict]:
    """The sweep function's own role (not the scheduler's, which shares the prefix)."""
    role = next(lid for lid, r in _of_type(synth, "AWS::IAM::Role", SWEEP_ROLE)
                if not lid.startswith(SCHED_ROLE))
    return [st for r in _res(synth, COMPUTE).values() if r["Type"] == "AWS::IAM::Policy"
            and {"Ref": role} in r["Properties"].get("Roles", [])
            for st in r["Properties"]["PolicyDocument"]["Statement"]]


def test_sweep_lambda_and_schedule_exist_when_on(synth_flex_any) -> None:
    fn_lid, fn = _one(synth_flex_any, COMPUTE, SWEEP_FN, "AWS::Lambda::Function")
    assert fn["Properties"]["Handler"] == "lambdas.conversation_sweep.handler.handler"
    (_, sched), = _of_type(synth_flex_any, "AWS::Scheduler::Schedule")
    props = sched["Properties"]
    assert props["ScheduleExpression"] == "rate(10 minutes)" == f"rate({int(CONVERSATION_SWEEP_RATE.to_minutes())} minutes)"
    assert props["State"] == "ENABLED" and props["FlexibleTimeWindow"] == {"Mode": "OFF"}
    assert props["Target"]["Arn"] == {"Fn::GetAtt": [fn_lid, "Arn"]}
    assert props["Target"]["RetryPolicy"]["MaximumRetryAttempts"] == 0
    # Never retried by Lambda either; no public endpoint or other trigger.
    (_, invoke_cfg), = [(lid, r) for lid, r in _of_type(synth_flex_any, "AWS::Lambda::EventInvokeConfig")
                        if r["Properties"]["FunctionName"] == {"Ref": fn_lid}]
    assert invoke_cfg["Properties"]["MaximumRetryAttempts"] == 0
    assert not [r for r in _res(synth_flex_any, COMPUTE).values() if r["Type"] in (
        "AWS::Lambda::Url", "AWS::Lambda::EventSourceMapping", "AWS::Events::Rule")
        and fn_lid in json.dumps(r)]
    assert not [r for _, r in _of_type(synth_flex_any, "AWS::Lambda::Permission") if fn_lid in json.dumps(r)]
    assert fn_lid not in json.dumps(_res(synth_flex_any, API)) and "conversation_sweep" not in json.dumps(_res(synth_flex_any, API))


def test_scheduler_role_invokes_only_the_sweep(synth_flex_any) -> None:
    fn_lid, _ = _one(synth_flex_any, COMPUTE, SWEEP_FN, "AWS::Lambda::Function")
    role_lid, role = _one(synth_flex_any, COMPUTE, SCHED_ROLE, "AWS::IAM::Role")
    trust = role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert [t["Principal"] for t in trust] == [{"Service": "scheduler.amazonaws.com"}]
    cond = trust[0]["Condition"]["StringEquals"]
    assert "aws:SourceAccount" in cond
    # And the sweep's own schedule group (Scheduler's aws:SourceArn is the group's ARN).
    group_lid, group = _one(synth_flex_any, COMPUTE, "ConversationSweepScheduleGroup", "AWS::Scheduler::ScheduleGroup")
    assert cond["aws:SourceArn"] == {"Fn::GetAtt": [group_lid, "Arn"]}
    (_, sched), = _of_type(synth_flex_any, "AWS::Scheduler::Schedule")
    assert sched["Properties"]["GroupName"] == {"Ref": group_lid}
    assert group["Properties"]["Name"].endswith("-conversation-sweep")
    sts = [st for r in _res(synth_flex_any, COMPUTE).values() if r["Type"] == "AWS::IAM::Policy"
           and {"Ref": role_lid} in r["Properties"]["Roles"] for st in r["Properties"]["PolicyDocument"]["Statement"]]
    assert sts == [{"Action": "lambda:InvokeFunction", "Effect": "Allow", "Sid": "InvokeConversationSweep",
                    "Resource": {"Fn::GetAtt": [fn_lid, "Arn"]}}]


def test_sweep_reads_only_the_api_key_secret_by_arn(synth_flex_any) -> None:
    sts = [st for st in _sweep_role_statements(synth_flex_any) if any(a.startswith("secretsmanager:") for a in _actions(st))]
    assert len(sts) == 1 and _actions(sts[0]) == SECRET_ACTIONS
    assert sts[0]["Resource"] == {"Fn::ImportValue": _export_of(synth_flex_any, "twilio_api_key")}
    env = _env(synth_flex_any, SWEEP_FN)
    assert env["OWLBANK_SECRET_TWILIO_API_KEY"] == sts[0]["Resource"]
    kms = [st for st in _sweep_role_statements(synth_flex_any) if st.get("Sid") == "DecryptSecretsViaSecretsManager"]
    assert len(kms) == 1 and _actions(kms[0]) == ["kms:Decrypt"]
    assert kms[0]["Condition"] == {"StringEquals": {"kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"}}


@pytest.mark.parametrize("name", ["default", "ws"])
def test_no_wildcard_in_the_sweep_grants(hl, synth_flex, name) -> None:
    """Every sweep statement lists its actions and names ARNs; the only ``*`` resource is the
    factory's X-Ray statement. The tables it reaches: ``calls`` and ``segments`` (the close's
    ``conversation_closed`` journal event), by ARN, and no other; the one index it reaches is the
    ``calls`` table's ``sweep-due``."""
    synth = synth_flex if name == "default" else hl[name]
    sts = _sweep_role_statements(synth)
    assert sts
    for st in sts:
        acts = _actions(st)
        assert not [a for a in acts if "*" in a], st
        if st["Resource"] == "*":
            assert all(a.startswith("xray:") for a in acts), st
        else:
            assert "*" not in json.dumps(st["Resource"]), st
    tables = [st for st in sts if any(a.startswith("dynamodb:") for a in _actions(st))
              and st.get("Sid") != SWEEP_INDEX_SID]
    assert len(tables) == 2
    reached = json.dumps([t["Resource"] for t in tables])
    assert "CallsTable" in reached and "SegmentsTable" in reached
    assert "/index/" not in reached
    assert not [t for t in ("Customers", "Policy", "Catalog", "Tokens", "Risk") if f"{t}Table" in reached]


@pytest.mark.parametrize("name", ["default", "ws"])
def test_sweep_table_actions_are_exactly_those_it_uses(hl, synth_flex, name) -> None:
    """Per table, exactly the DynamoDB actions the sweep and its close path call (no
    ``BatchWriteItem``, ``UpdateItem`` or ``DescribeTable``; ``DeleteItem`` on ``calls`` only, for the
    version-claim takeover in ``callstate.update``)."""
    synth = synth_flex if name == "default" else hl[name]
    tables = {st["Sid"]: st for st in _sweep_role_statements(synth)
              if any(a.startswith("dynamodb:") for a in _actions(st))}
    assert set(tables) == {"TableCalls", "TableSegments", SWEEP_INDEX_SID}
    assert sorted(_actions(tables["TableCalls"])) == sorted(SWEEP_TABLE_ACTIONS["calls"]) == sorted(
        ["dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"])
    assert sorted(_actions(tables["TableSegments"])) == sorted(SWEEP_TABLE_ACTIONS["segments"]) == sorted(
        ["dynamodb:Query", "dynamodb:PutItem"])
    assert "CallsTable" in json.dumps(tables["TableCalls"]["Resource"])
    assert "SegmentsTable" in json.dumps(tables["TableSegments"]["Resource"])


@pytest.mark.parametrize("name", ["default", "ws"])
def test_sweep_queries_only_the_sweep_due_index(hl, synth_flex, name) -> None:
    """``dynamodb:Query`` on the ``calls`` table's ``sweep-due`` index ARN, its own
    statement, and nothing else: no other index, no table ARN, no other action."""
    synth = synth_flex if name == "default" else hl[name]
    (st,) = [st for st in _sweep_role_statements(synth) if st.get("Sid") == SWEEP_INDEX_SID]
    assert _actions(st) == list(SWEEP_INDEX_ACTIONS) == ["dynamodb:Query"]
    resource = st["Resource"]
    assert not isinstance(resource, list)
    text = json.dumps(resource)
    assert "CallsTable" in text and text.count("/index/") == 1 and text.endswith('/index/sweep-due"]]}')
    assert not [s for s in _sweep_role_statements(synth) if s.get("Sid") != SWEEP_INDEX_SID
                and "/index/" in json.dumps(s.get("Resource"))]


def test_no_other_role_can_query_the_sweep_index(synth_flex_any) -> None:
    """``grant_tables`` leaves the ``sweep-due`` index out (``TableDef.own_grant``), so the front
    door, bridge and other roles that read ``calls`` get no access to it."""
    seen = 0
    for stack, template in synth_flex_any.templates.items():
        for lid, r in template.get("Resources", {}).items():
            if r["Type"] != "AWS::IAM::Policy":
                continue
            for st in r["Properties"]["PolicyDocument"]["Statement"]:
                if "/index/sweep-due" in json.dumps(st.get("Resource")):
                    seen += 1
                    assert st.get("Sid") == SWEEP_INDEX_SID, (stack, lid, st.get("Sid"))
    assert seen == 1


def test_sweep_env_from_twilio_outputs(hl, synth_default) -> None:
    env = _env(hl["ws"], SWEEP_FN)
    assert "OWLBANK_CONVERSATION_CONFIGURATION_ID" not in env  # Nothing is listed from Twilio
    assert env[TASKROUTER_WORKSPACE_ENV] == WORKSPACE
    assert sorted(k for k in env if k.startswith("OWLBANK_TABLE_")) == [  # + the journal
        "OWLBANK_TABLE_CALLS", "OWLBANK_TABLE_SEGMENTS"]
    assert "ConversationSweep" not in json.dumps(_res(synth_default, COMPUTE))  # no Flex: no sweep at all


def test_sweep_errors_alarm(synth_flex_any) -> None:
    fn_lid, _ = _one(synth_flex_any, COMPUTE, SWEEP_FN, "AWS::Lambda::Function")
    alarms = {r["Properties"]["MetricName"]: r["Properties"] for _, r in _of_type(synth_flex_any, "AWS::CloudWatch::Alarm")
              if r["Properties"].get("Dimensions") == [{"Name": "FunctionName", "Value": {"Ref": fn_lid}}]}
    assert set(alarms) == {"Errors", "Throttles"}
    errors = alarms["Errors"]
    assert errors["AlarmActions"] and errors["Period"] == 1800 and errors["Threshold"] == 2


@pytest.mark.parametrize("name", ["off", "off-obs", "ws-off"])
def test_flag_off_removes_the_sweep(hl, name) -> None:
    text = json.dumps(_res(hl[name], COMPUTE))
    assert "ConversationSweep" not in text and "conversation_sweep" not in text
    assert not _of_type(hl[name], "AWS::Scheduler::Schedule")
