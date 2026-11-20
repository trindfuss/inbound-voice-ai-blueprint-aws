"""Compute: the six Lambda functions (seven with ``bridgeHosting=agentcore``: ``runtime_warmer``;
one more with model failover on: ``model_poller``; one more with ``humanLegConversation=on``:
``conversation_sweep``; one more with ``memoryStoreId`` in ``twilioOutputs``: ``recap_writer``),
each with its own least-privilege role.

Code is one asset built from ``../src`` (the ``owlbank`` library plus
``lambdas/<name>/handler.py``), so every function ships the shared library.
Third-party packages (today only twilio; see ``build_lambda_deps.py``) go in a layer built
from ``infra/build/lambda-deps`` by ``infra/scripts/build_lambda_deps.py`` before a
deploy; synth never runs pip or Docker.

Nothing here has a Function URL. ``auth_service``, ``bank_tools`` and
``level0_tools`` have no API at all: only the front door, the bridge task role
and the Gateway role may invoke them.

Latency: ``frontdoor``, ``auth_service`` and ``bank_tools`` get ``lambdaMemoryMb`` and,
unless ``lambdaProvisionedConcurrency`` is 0, a ``live`` alias with provisioned concurrency.
Everything that invokes them (API Gateway, the front door, the bridge) uses the alias ARN, and
each grant names exactly that ARN.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from aws_cdk import Duration, Stack
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_scheduler as scheduler
from aws_cdk import aws_secretsmanager as sm
from aws_cdk import aws_sns as sns
from constructs import Construct

from . import call_observability, model_failover
from .config import DEFAULT_BRIEFING_BUDGET_MS, LIVE_ALIAS, PROVISIONED_FUNCTIONS, InfraConfig
from .contract import DATA_DIR, REPO_ROOT, SRC_DIR, SWEEP_DUE_INDEX, TABLES
from .data_stack import CONTEXT_API_KEY_SECRET, role_arn
from .gateway_target import level0_target
from .grants import (
    PII_DETECT_REASON,
    TOKEN_KEY_ENV,
    ack_runtime_prefix,
    catalog_scan_statement,
    grant_invoke,
    grant_pii_detect,
    grant_relay_parameter_read,
    grant_table_actions,
    grant_tables,
    grant_token_key,
    runtime_invoke_resources,
)
from .nag import ack
from .observability import (
    FRONTDOOR_REJECT_MESSAGE,
    OTP_KEYED_PHONE_EVENT,
    OTP_KEYED_PHONE_METRIC,
    OTP_UNTRUSTED_ANI_EVENT,
    Alarms,
)
from .observability import log_group as log_group_fn

INFRA_DIR = Path(__file__).resolve().parents[1]
LAMBDA_DEPS_DIR = INFRA_DIR / "build" / "lambda-deps"
LAMBDA_SRC_DIR = INFRA_DIR / "build" / "lambda-src"
#: Data files the Lambdas read from the bundle (level0_tools: data/faq.json).
BUNDLED_DATA = ("faq.json",)
BUNDLED_SCRIPTS = ("__init__.py", "hash_tool.py")
#: Functions other stacks name by ARN (the Agent's Gateway role, the bridge's task role), so
#: they get deterministic names (``InfraConfig.function_name``) and need no stack reference.
NAMED_FUNCTIONS = ("bank_tools", "level0_tools")


def stage_lambda_source() -> Path:
    """Copy ``src`` (minus the bridge) plus the bundled data files into build/lambda-src.

    Layout at the bundle root (``/var/task``): ``owlbank/``, ``lambdas/``, ``scripts/``
    (``hash_tool`` only), ``data/``.
    ``level0_tools`` gets ``OWLBANK_FAQ_PATH=/var/task/data/faq.json``. Plain file
    copies: no pip, no Docker.
    """
    if LAMBDA_SRC_DIR.exists():
        shutil.rmtree(LAMBDA_SRC_DIR)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "tac_bridge")
    for pkg in ("owlbank", "lambdas"):
        shutil.copytree(SRC_DIR / pkg, LAMBDA_SRC_DIR / pkg, ignore=ignore)
    # auth_service imports scripts.hash_tool (the factor-hash scheme shared with seeding).
    (LAMBDA_SRC_DIR / "scripts").mkdir()
    for name in BUNDLED_SCRIPTS:
        shutil.copy2(REPO_ROOT / "scripts" / name, LAMBDA_SRC_DIR / "scripts" / name)
    (LAMBDA_SRC_DIR / "data").mkdir(parents=True)
    for name in BUNDLED_DATA:
        shutil.copy2(DATA_DIR / name, LAMBDA_SRC_DIR / "data" / name)
    return LAMBDA_SRC_DIR

#: Function -> (tables read-only, tables read-write, secrets). Taken from what each
#: handler in src/lambdas reads and writes.
ACCESS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "frontdoor": (("customers", "policy", "catalog"), ("calls", "segments", "tokens"),
                  ("twilio_auth_token", "basic_auth", "twilio_api_key")),
    "auth_service": (("customers", "policy"), ("tokens", "calls", "risk"), ("twilio_api_key",)),
    "bank_tools": (("tokens", "policy"), ("customers", "calls"), ()),
    "level0_tools": ((), (), ()),
    # ``policy`` for the ``redaction`` item (timeline redaction):
    # ``catalog`` for ``escalation.summary_withheld``. Same data CMK as ``calls``/``segments``, so
    # no new KMS statement; never the token key.
    # Human leg: one Twilio key secret each. ``twilio_api_key`` here is the slot
    # :func:`secrets_of` fills: the Context API's own Restricted key (``context_api_key``) with
    # ``contextApiRestrictedKey=on`` (the default); the shared Standard key with the flag off. The Flex
    # authorizer always reads the shared key: Twilio has no Restricted permission for ``Tokens/validate``.
    "context_api": (("calls", "segments", "policy", "catalog"), (), ("twilio_api_key",)),
    "flex_authorizer": ((), (), ("twilio_api_key",)),
    # AgentCore hosting only: starts the call's Runtime session; no tables or secrets.
    "runtime_warmer": ((), (), ()),
}
#: Functions that exist only with ``bridgeHosting=agentcore``.
AGENTCORE_ONLY = ("runtime_warmer",)
SECRET_ENV = {
    "twilio_api_key": "OWLBANK_SECRET_TWILIO_API_KEY",
    "twilio_auth_token": "OWLBANK_SECRET_TWILIO_AUTH_TOKEN",
    "basic_auth": "OWLBANK_SECRET_BASIC_AUTH",
    # Same variable as ``twilio_api_key`` and the same JSON shape: each handler reads
    # ``OWLBANK_SECRET_TWILIO_API_KEY`` whichever secret it is given (only its function gets it).
    CONTEXT_API_KEY_SECRET: "OWLBANK_SECRET_TWILIO_API_KEY",
}
#: Function -> (its own Restricted-key secret, whether this configuration gives it that secret
#: instead of the shared Standard ``twilio_api_key``).
RESTRICTED_KEY_OF: dict[str, tuple[str, Callable[[InfraConfig], bool]]] = {
    "context_api": (CONTEXT_API_KEY_SECRET, lambda cfg: cfg.context_api_restricted_key_on),
}
#: Human-leg: the Flex TaskRouter workspace (``workspaceSid`` in ``twilioOutputs``, written by
#: scripts/twilio_setup.py) for the Context API's per-task check and the front door's reservation reads.
TASKROUTER_WORKSPACE_ENV = "OWLBANK_TASKROUTER_WORKSPACE_SID"
#: Human-leg: ``humanLegConversation`` on the front door (``on`` | ``off``): the closer and the
#: participant adds run only when ``on`` (the routes exist only then). Not the transcription switch,
#: which is the policy item ``human_leg``.
HUMAN_LEG_ENV = "OWLBANK_HUMAN_LEG_CONVERSATION"
#: The escalation briefing's overall deadline in ms (default :data:`DEFAULT_BRIEFING_BUDGET_MS`).
BRIEFING_BUDGET_ENV = "OWLBANK_BRIEFING_BUDGET_MS"


def secrets_of(name: str, cfg: InfraConfig) -> tuple[str, ...]:
    """The secrets function ``name`` reads: :data:`ACCESS`, except that a function in
    :data:`RESTRICTED_KEY_OF` reads its own Restricted-key secret instead of ``twilio_api_key`` when its
    flag is on (``contextApiRestrictedKey``, on by default)."""
    secret_names = ACCESS[name][2]
    own = RESTRICTED_KEY_OF.get(name)
    if own is not None and own[1](cfg):
        return tuple(own[0] if s == "twilio_api_key" else s for s in secret_names)
    return secret_names
#: Human-leg: the stale-conversation sweep, only with ``humanLegConversation=on``.
CONVERSATION_SWEEP = "conversation_sweep"
#: Every 10 minutes; a call is closed once it has been over for more than 5 (in the handler).
#: A run reads only the escalated calls whose conversation is not closed, from the ``calls``
#: table's ``sweep-due`` index, and the Calls API only for those with no recorded end.
CONVERSATION_SWEEP_RATE = Duration.minutes(10)
#: Up to a few hundred Twilio requests of at most 3 s each (500 live calls in queue are 250 to 500
#: Calls API reads per run, about 40 to 75 s); the handler stops starting work 20 s before.
CONVERSATION_SWEEP_TIMEOUT_S = 300
#: Exactly the DynamoDB actions the sweep uses. ``calls``: the run's call-state read
#: (``BatchGetItem``) and ``callstate.mutate`` for the reschedule and the close path (``GetItem``,
#: conditional ``PutItem`` of the version claim and the state, and ``DeleteItem`` when it takes over a
#: stale version claim, ``callstate._take_over_stale_claim``). ``segments``: the ``conversation_closed``
#: journal event (``SegmentLog.append``: ``Query`` for the last sequence number, conditional
#: ``PutItem``). No ``BatchWriteItem``, no ``UpdateItem``, no ``DescribeTable``.
SWEEP_TABLE_ACTIONS = {
    "calls": ("dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"),
    "segments": ("dynamodb:Query", "dynamodb:PutItem"),
}
#: The one index action, on the ``calls`` table's ``sweep-due`` index ARN only (its own
#: statement, ``SweepDueIndexQuery``); the table ARN itself gets no ``Query``.
SWEEP_INDEX_ACTIONS = ("dynamodb:Query",)
SWEEP_INDEX_SID = "SweepDueIndexQuery"
#: The sweep runs 3 times in 30 minutes: 2 failed runs in that window alarm (one is a Twilio blip).
CONVERSATION_SWEEP_ALARM_PERIOD = Duration.minutes(30)
#: The post-call recap writer, only with ``memoryStoreId`` in
#: ``twilioOutputs`` (:attr:`InfraConfig.recap_writer_on`). Invoked asynchronously by the front door
#: and the conversation sweep (``OWLBANK_RECAP_FUNCTION``); one Lambda async retry.
RECAP_WRITER = "recap_writer"
RECAP_FUNCTION_ENV = "OWLBANK_RECAP_FUNCTION"
RECAP_MODEL_ENV = "OWLBANK_RECAP_MODEL_ID"
RECAP_RETRY_ATTEMPTS = 1
#: Exactly the DynamoDB actions the writer uses. ``calls``: ``callstate.mutate`` for the claim, the
#: ``writing_at`` mark and the outcome (``GetItem``, conditional ``PutItem``, and ``DeleteItem`` when
#: it takes over a stale version claim); ``customers``: the verified customer's ``GetItem``;
#: ``policy``: ``Policy.from_store`` (``GetItem`` of the meta item, ``BatchGetItem``). ``catalog``
#: (``PromptCatalog.from_store``) gets only its ``Scan`` statement (:func:`catalog_scan_statement`).
#: No ``UpdateItem``, ``Query``, ``BatchWriteItem`` or ``DescribeTable``; no index.
RECAP_TABLE_ACTIONS = {
    "calls": ("dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"),
    "customers": ("dynamodb:GetItem",),
    "policy": ("dynamodb:GetItem", "dynamodb:BatchGetItem"),
}
RECAP_TABLES = (*RECAP_TABLE_ACTIONS, "catalog")
TIMEOUTS = {"frontdoor": 10, "auth_service": 5, "bank_tools": 8, "level0_tools": 5,
            "context_api": 5, "flex_authorizer": 5, "runtime_warmer": 30,
            model_failover.POLLER: model_failover.POLLER_TIMEOUT_S,
            CONVERSATION_SWEEP: CONVERSATION_SWEEP_TIMEOUT_S, RECAP_WRITER: 60}
#: Memory overrides (MB); the rest follow ``InfraConfig.memory_mb``.
MEMORY_MB = {"runtime_warmer": 256, model_failover.POLLER: model_failover.POLLER_MEMORY_MB,
             RECAP_WRITER: 256}
#: Names the relay parameter for the front door and the warmer (``owlbank.relay_param``).
RELAY_PARAMETER_ENV = "OWLBANK_RELAY_PARAMETER"
#: The deploy-time slot-mode ceiling (``slotModeCeiling``), on the front door and ``admin_api``.
SLOT_MODE_CEILING_ENV = "OWLBANK_SLOT_MODE_CEILING"
#: Segment traits: the trait groups (front door and auth service) and the Twilio Memory store
#: (auth service).
MEMORY_TRAIT_GROUPS_ENV = "OWLBANK_MEMORY_TRAIT_GROUPS"
MEMORY_STORE_ENV = "OWLBANK_MEMORY_STORE_ID"
#: Validity of a pre-signed Runtime URL (the AgentCore maximum).
URL_TTL_S = 300
#: The warmer's throttling or quota failures: any one in 5 minutes alarms.
PREWARM_THROTTLE_CODES = ("ThrottlingException", "ServiceQuotaExceededException")


#: Geographic cross-Region inference-profile prefixes (system-defined profiles).
GEO_PROFILE_PREFIXES = ("us", "eu", "apac", "au", "jp", "us-gov")
GLOBAL_PROFILE_PREFIX = "global"


def bedrock_invoke_statements(stack: Stack, model_id: str, *, sid: str,
                              actions: list[str]) -> list[iam.PolicyStatement]:
    """Least-privilege statements to invoke ``model_id``.

    For a geo cross-Region inference profile (``us.``/``eu.``/...), the AWS-documented
    pattern: one statement for the profile in this Region, and one for the one
    foundation model in any Region, usable only through that profile.

    Why the Region wildcard on the foundation model is safe: the
    ``bedrock:InferenceProfileArn`` condition means the model can be invoked only through
    our geo profile (a US profile routes only to US Regions, by definition), and the
    profile decides the destination Region. Direct invocation of the model, in any Region,
    stays denied. Listing Regions instead breaks when the profile adds one (us-west-1).
    ``scripts/preflight.py`` prints the profile's destination Regions.

    For a ``global.`` profile, AWS's documented shape:
    the profile in this Region, and the one foundation model as the empty-Region ARN
    (``arn:<part>:bedrock:::foundation-model/<base>``, how Bedrock authorizes a global route) and
    the source-Region ARN, both only through that profile. No Region wildcard, so no IAM5 ack.
    """
    region, account, part = stack.region, stack.account, stack.partition
    prefix = model_id.split(".", 1)[0]
    if prefix not in (*GEO_PROFILE_PREFIXES, GLOBAL_PROFILE_PREFIX):
        return [iam.PolicyStatement(sid=sid, actions=actions,
                                    resources=[f"arn:{part}:bedrock:{region}::foundation-model/{model_id}"])]
    base = model_id.split(".", 1)[1]
    profile = f"arn:{part}:bedrock:{region}:{account}:inference-profile/{model_id}"
    models = ([f"arn:{part}:bedrock:::foundation-model/{base}",
               f"arn:{part}:bedrock:{region}::foundation-model/{base}"]
              if prefix == GLOBAL_PROFILE_PREFIX else [f"arn:{part}:bedrock:*::foundation-model/{base}"])
    return [
        iam.PolicyStatement(sid=f"{sid}Profile", actions=actions, resources=[profile]),
        iam.PolicyStatement(
            sid=f"{sid}ViaProfile", actions=actions, resources=models,
            conditions={"StringEquals": {"bedrock:InferenceProfileArn": profile}}),
    ]


BEDROCK_REGION_WILDCARD_REASON = (
    "The foundation-model resource has a Region wildcard but one specific model, and a "
    "bedrock:InferenceProfileArn condition: the model is invocable only through our geo "
    "inference profile, which routes only to its geography's Regions. Direct invocation stays "
    "denied in every Region (AWS-documented cross-Region inference policy).")


def bedrock_nag_ack(role: iam.IRole, model_id: str) -> None:
    """IAM5 acknowledgement for the Region wildcard (geo profiles only; ``global.`` has none)."""
    if "." not in model_id or model_id.split(".", 1)[0] not in GEO_PROFILE_PREFIXES:
        return
    base = model_id.split(".", 1)[1]
    ack(role, f"AwsSolutions-IAM5[Resource::arn:<AWS::Partition>:bedrock:*::foundation-model/{base}]",
        BEDROCK_REGION_WILDCARD_REASON)


def import_secret(scope: Construct, cid: str, secret: sm.ISecret) -> sm.ISecret:
    """The secret, imported by ARN without its key, so grants stay on the principal side.

    Granting on the Data stack's own Secret object would add the consumer's role to the
    CMK's key policy, which makes the Data stack depend on its consumers (a cycle).
    """
    return sm.Secret.from_secret_attributes(scope, cid, secret_complete_arn=secret.secret_arn)


def grant_secret_decrypt(scope: Stack, principal: iam.IGrantable, key: kms.IKey) -> None:
    """kms:Decrypt on the CMK, only when called through Secrets Manager."""
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid="DecryptSecretsViaSecretsManager",
        actions=["kms:Decrypt"],
        resources=[key.key_arn],
        conditions={"StringEquals": {"kms:ViaService": f"secretsmanager.{scope.region}.amazonaws.com"}},
    ))


def table_env(tables: dict[str, ddb.ITable], names: tuple[str, ...]) -> dict[str, str]:
    return {f"OWLBANK_TABLE_{n.upper()}": tables[n].table_name for n in names}


class ComputeStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, key: kms.IKey,
                 token_key: kms.IKey, tables: dict[str, ddb.ITable], secrets: dict[str, sm.ISecret],
                 alarm_topic: sns.ITopic, gateway_identifier: str, **kw) -> None:
        super().__init__(scope, cid, **kw)
        self.cfg = cfg
        self.log_groups: dict[str, logs.LogGroup] = {}

        code = lambda_.Code.from_asset(str(stage_lambda_source()))
        LAMBDA_DEPS_DIR.joinpath("python").mkdir(parents=True, exist_ok=True)
        deps = lambda_.LayerVersion(
            self, "DepsLayer",
            code=lambda_.Code.from_asset(str(LAMBDA_DEPS_DIR)),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
            compatible_architectures=[lambda_.Architecture.ARM_64],
            description="Third-party packages for the OwlBank Lambdas (built by infra/scripts/build_lambda_deps.py)",
        )

        self.functions: dict[str, lambda_.Function] = {}
        for name, (ro, rw, _) in ACCESS.items():
            if name in AGENTCORE_ONLY and not cfg.is_agentcore:
                continue
            secret_names = secrets_of(name, cfg)
            env = {
                "OWLBANK_DEFAULT_LANGUAGE": "en-US",
                "OWLBANK_SUPPORTED_LANGUAGES": "en-US,es-US",
                "OWLBANK_LOG_LEVEL": cfg.python_log_level,  # The app's own log level
                # Every table name (DynamoStore needs them all); IAM limits what is reachable.
                **table_env(tables, tuple(TABLES)),
                **{SECRET_ENV[s]: secrets[s].secret_arn for s in secret_names},
            }
            self.functions[name] = self._function(name, code, [deps], key, env)
            grant_tables(self, self.functions[name], key, tables, read_only=ro, read_write=rw)
            for s in secret_names:
                import_secret(self, f"{name}-{s}", secrets[s]).grant_read(self.functions[name])
            if secret_names:
                grant_secret_decrypt(self, self.functions[name], key)

        if cfg.failover_on:
            self._model_poller(cfg, code, deps, key, tables["policy"])

        # The ``live`` alias of the three critical-path functions (provisioned concurrency).
        # ``invoke_targets`` is what callers invoke: the alias for those three, the function itself
        # for the rest (and for all of them with lambdaProvisionedConcurrency=0).
        self.aliases: dict[str, lambda_.Alias] = {}
        self.invoke_targets: dict[str, lambda_.IFunction] = dict(self.functions)
        if cfg.provisioned_concurrency_on:
            for name in PROVISIONED_FUNCTIONS:
                self.aliases[name] = self.functions[name].add_alias(
                    LIVE_ALIAS, provisioned_concurrent_executions=cfg.lambda_provisioned_concurrency)
                self.invoke_targets[name] = self.aliases[name]
            # The front door's background code send goes to the alias; never retry it (a retry
            # would text a second code after the front door has given up).
            self.aliases["auth_service"].configure_async_invoke(retry_attempts=0)

        self.functions["level0_tools"].add_environment("OWLBANK_FAQ_PATH", "/var/task/data/faq.json")
        # Legacy export, kept for ONE deploy. Api used to import the front door's plain function ARN; it now
        # imports the `live` alias. CloudFormation refuses to delete an export another stack still imports
        # ("Cannot delete export ... as it is in use by OwlBankApi"), so the old export stays until Api has
        # moved to the alias. After that deploy, delete this block and deploy again (infra README hazards).
        self.export_value(self.functions["frontdoor"].function_arn)
        fd = self.functions["frontdoor"]
        for target in ("auth_service", "bank_tools"):
            grant_invoke(fd, self.invoke_targets[target], sid=f"Invoke{target.title().replace('_', '')}")
            fd.add_environment(f"OWLBANK_{target.upper()}_FUNCTION", self.invoke_targets[target].function_arn)
        fd.add_environment("OWLBANK_RESOLVER_MODEL_ID", cfg.resolver_model_id)
        # Spoken verification factors (on in dev/test, off in prod unless overridden).
        fd.add_environment("OWLBANK_ACCEPT_SPOKEN_FACTORS", "true" if cfg.accept_spoken_factors else "false")
        fd.add_environment("OWLBANK_SCRIPTED_BARGE_IN", cfg.scripted_barge_in)
        fd.add_environment("OWLBANK_SPEECH_TIMEOUT", cfg.speech_timeout)
        fd.add_environment("OWLBANK_CONFIRM_SPEECH_TIMEOUT", cfg.confirm_speech_timeout)
        fd.add_environment("OWLBANK_KEYPAD_TIMEOUT", cfg.keypad_timeout)
        fd.add_environment("OWLBANK_REUSE_CONVERSATION", "true" if cfg.reuse_conversation else "false")
        # Early finish for spoken digits. The Twilio REST call update uses the
        # ``twilio_api_key`` secret granted above (that one secret ARN only, ACCESS["frontdoor"]).
        fd.add_environment("OWLBANK_EARLY_DIGITS", "true" if cfg.early_digits else "false")
        fd.add_environment("OWLBANK_AI_KEYPAD", cfg.ai_keypad_shortcuts)  # keypad shortcuts
        # The slot-mode ceiling, and the resolver's Bedrock statement only when a
        # model may be involved. With ``deterministic`` the role cannot call Bedrock at all.
        fd.add_environment(SLOT_MODE_CEILING_ENV, cfg.slot_mode_ceiling)
        # The escalation briefing (no Flex) reuses the resolver model and grant below; no new IAM.
        fd.add_environment(BRIEFING_BUDGET_ENV, str(DEFAULT_BRIEFING_BUDGET_MS))
        if cfg.slot_resolver_allowed:
            for st in bedrock_invoke_statements(self, cfg.resolver_model_id, sid="SlotResolver",
                                                actions=["bedrock:InvokeModel"]):
                fd.add_to_role_policy(st)
            bedrock_nag_ack(fd.role, cfg.resolver_model_id)
        # The speech log of Gather turns, and its optional redaction (the Comprehend call
        # is the PiiRedactor's, granted just below; no new permission).
        fd.add_environment("OWLBANK_SPEECH_LOG", cfg.speech_log)
        fd.add_environment("OWLBANK_SPEECH_LOG_REDACTION", cfg.speech_log_redaction)
        grant_pii_detect(fd)  # PiiRedactor (Comprehend)
        # The screen-pop timeline's prompts are redacted at read time.
        grant_pii_detect(self.functions["context_api"])
        # The relay comes from the Bridge stack's fixed-name SSM parameter, read once at INIT
        # (fargate: the relay host, from which the front door builds wss://<host>/ws; agentcore:
        # the Runtime ARN). No CloudFormation export between Bridge and Compute in either hosting.
        fd.add_environment(RELAY_PARAMETER_ENV, cfg.relay_parameter_name)
        grant_relay_parameter_read(self, fd, cfg.relay_parameter_name)
        if cfg.is_agentcore:
            self._agentcore_front_door(cfg, fd)
        # Call observability: OpenTelemetry in its own layer on the front door only, and
        # its settings. Spans go to the Lambda X-Ray daemon over loopback UDP, which active
        # tracing's xray:PutTraceSegments/PutTelemetryRecords grant already covers: no new IAM.
        if cfg.call_observability_on:
            fd.add_layers(call_observability.observability_layer(self))
            for env_name, value in call_observability.frontdoor_env(cfg).items():
                fd.add_environment(env_name, value)
        # The Gateway's only target. Here, not in the Agent stack: it needs this function to
        # exist, and Agent is deployed before Compute.
        self.level0_target = level0_target(self, "Level0Target", gateway_identifier=gateway_identifier,
                                           lambda_arn=self.functions["level0_tools"].function_arn)
        fd.add_environment(TOKEN_KEY_ENV, token_key.key_arn)
        grant_token_key(fd, token_key, ["kms:Encrypt", "kms:Decrypt"], sid="TokenSealing")
        tw = cfg.twilio_outputs
        for out_key, env_name in (("conversationConfigurationId", "OWLBANK_CONVERSATION_CONFIGURATION_ID"),
                                  ("workflowSid", "OWLBANK_WORKFLOW_SID")):
            if out_key in tw:
                fd.add_environment(env_name, tw[out_key])
        self._human_leg_env(cfg, fd)
        if cfg.verify_service_sid:
            self.functions["auth_service"].add_environment("OWLBANK_VERIFY_SERVICE_SID",
                                                           cfg.verify_service_sid)
        # Segment traits: the front door triggers the welcome only when WebActivity is listed.
        # Nothing new in IAM: it already invokes the auth service.
        fd.add_environment(MEMORY_TRAIT_GROUPS_ENV, ",".join(cfg.memory_trait_groups))
        self._auth_service_welcome(cfg, token_key)
        if cfg.recap_writer_on:
            self._recap_writer(cfg, code, deps, key, token_key, tables, secrets["twilio_api_key"])

        # ---- Errors and throttles per function, and front-door auth failures.
        alarms = Alarms(self, alarm_topic, cfg)
        for name, f in self.functions.items():
            cid = "".join(p.capitalize() for p in name.split("_"))
            alarms.alarm(f"{cid}ErrorsAlarm", f.metric_errors(period=Duration.minutes(5), statistic="Sum"),
                         threshold=3, description=f"{name} Lambda errors")
            alarms.alarm(f"{cid}ThrottlesAlarm", f.metric_throttles(period=Duration.minutes(5), statistic="Sum"),
                         threshold=1, description=f"{name} Lambda throttles")
        alarms.reject_filter("FrontdoorAuthRejected", self.log_groups["frontdoor"], (FRONTDOOR_REJECT_MESSAGE,),
                             "FrontdoorRejected",
                             "Front door rejected requests (signature, Basic auth, route or unverifiable)")
        # A one-time code was sent on a call with an untrusted caller ID.
        alarms.slot_mode_invalid(self.log_groups["frontdoor"])
        alarms.reject_filter("OtpUntrustedAni", self.log_groups["auth_service"],
                             (OTP_UNTRUSTED_ANI_EVENT,), "OtpSentUntrustedAni",
                             "A one-time code was sent on a call with an untrusted caller ID "
                             "(possible spoofed caller ID aimed at a real customer)", threshold=1)
        # Keyed-phone: a code sent on a keyed-in number's call, counted only (never the alarm above).
        alarms.count_filter("OtpKeyedPhone", self.log_groups["auth_service"], OTP_KEYED_PHONE_EVENT,
                            OTP_KEYED_PHONE_METRIC)
        if cfg.call_observability_on:  # The factor_answer step timings
            call_observability.frontdoor_filters(self, cfg, self.log_groups["frontdoor"])
        if cfg.is_agentcore:
            self._prewarm_alarm(alarms)
        if cfg.failover_on:
            model_failover.poller_alarms(cfg, alarms)  # Stale state, every entry down
        if cfg.human_leg_conversation_on:
            self._conversation_sweep(cfg, code, deps, key, {n: tables[n] for n in ("calls", "segments")},
                                     secrets["twilio_api_key"], alarms)
            if RECAP_WRITER in self.functions:  # The sweep's close triggers the recap too
                self._grant_recap_trigger(self.functions[CONVERSATION_SWEEP], "InvokeRecapWriter")

    def _model_poller(self, cfg: InfraConfig, code: lambda_.Code, deps: lambda_.ILayerVersion,
                      key: kms.IKey, policy: ddb.ITable) -> None:
        """Model failover's poller, only with ``modelFailover=on`` and two or
        more entries: arm64, 128 MB, every minute. No secret; its env names ``modelstats``, the
        ``policy`` table (read only), the stage and the model list (the bridge's ``OWLBANK_MODELS``)."""
        from .bridge_stack import FAILOVER_TEST_ENV, MODELS_ENV, models_env_value  # bridge_stack imports this

        models = {MODELS_ENV: models_env_value(cfg),
                  **({FAILOVER_TEST_ENV: f"{cfg.failover_test:g}"} if cfg.failover_test is not None else {})}
        env = {"OWLBANK_LOG_LEVEL": cfg.python_log_level,
               **model_failover.poller_env(cfg, models, policy.table_name)}
        # Never retried (an async retry would replay stale cycles; the next run is a minute away).
        fn = self._function(model_failover.POLLER, code, [deps], key, env, retry_attempts=0)
        self.functions[model_failover.POLLER] = fn
        model_failover.grant_poller(self, fn, cfg, key, policy.table_arn)
        model_failover.schedule(self, fn)

    def _human_leg_env(self, cfg: InfraConfig, fd: lambda_.Function) -> None:
        """Human-leg. No new IAM here: the Context API's one key secret is
        granted in the loop above (:func:`secrets_of`, ARN-scoped) and the front door already reads
        ``twilio_api_key``.

        * Context API: the workspace SID for the per-task check. Without ``workspaceSid`` in
          ``twilioOutputs`` the variable is left out and the API answers 503 (fails closed).
        * Front door: :data:`HUMAN_LEG_ENV` always, and the workspace SID (reservation reads for the
          close repair) only with ``humanLegConversation=on``."""
        workspace = cfg.twilio_outputs.get("workspaceSid")
        if workspace:
            self.functions["context_api"].add_environment(TASKROUTER_WORKSPACE_ENV, workspace)
        fd.add_environment(HUMAN_LEG_ENV, "on" if cfg.human_leg_conversation_on else "off")
        if workspace and cfg.human_leg_conversation_on:
            fd.add_environment(TASKROUTER_WORKSPACE_ENV, workspace)

    def _conversation_sweep(self, cfg: InfraConfig, code: lambda_.Code, deps: lambda_.ILayerVersion,
                            key: kms.IKey, tables: dict[str, ddb.ITable], api_key: sm.ISecret,
                            alarms: Alarms) -> None:
        """The human leg's stale-conversation sweep, only with
        ``humanLegConversation=on`` (``off`` removes it, with its schedule and alarms).

        * Lambda ``conversation_sweep`` from the shared factory (own role and log group), no trigger
          but its schedule, never retried (the next run is 10 minutes away).
        * EventBridge Scheduler every 10 minutes; its role may invoke this one function only.
        * IAM: on the ``calls`` and ``segments`` tables exactly :data:`SWEEP_TABLE_ACTIONS` (the
          run's call-state read, the reschedule's and the close path's call state, the close's
          ``conversation_closed`` journal event), :data:`SWEEP_INDEX_ACTIONS` on the
          ``calls`` table's ``sweep-due`` index ARN only, the ``twilio_api_key`` secret by its one ARN, and the data CMK's
          ``kms:Decrypt`` through Secrets Manager only. No other secret, no wildcard action or resource
          beyond the factory's X-Ray statement. The scheduler role's trust names this account and the
          sweep's own schedule group (``aws:SourceAccount``, ``aws:SourceArn``).
        * Env: the TaskRouter workspace (the close's repair). No Orchestrator configuration: the sweep
          lists nothing from Twilio; the close looks a conversation up by its CallSid.
        * Alarms: Errors (a run whose index query failed raises) and Throttles, over 30 minutes,
          sized for a 10-minute cadence (the per-function 5-minute alarms above never see 3 runs)."""
        env = {"OWLBANK_LOG_LEVEL": cfg.python_log_level,
               **table_env(tables, ("calls", "segments")),
               SECRET_ENV["twilio_api_key"]: api_key.secret_arn}
        tw = cfg.twilio_outputs
        if tw.get("workspaceSid"):
            env[TASKROUTER_WORKSPACE_ENV] = tw["workspaceSid"]
        fn = self._function(CONVERSATION_SWEEP, code, [deps], key, env, retry_attempts=0)
        self.functions[CONVERSATION_SWEEP] = fn
        grant_table_actions(self, fn, key, tables, SWEEP_TABLE_ACTIONS)
        fn.add_to_role_policy(iam.PolicyStatement(
            sid=SWEEP_INDEX_SID, actions=list(SWEEP_INDEX_ACTIONS),
            resources=[f"{tables['calls'].table_arn}/index/{SWEEP_DUE_INDEX}"]))
        import_secret(self, f"{CONVERSATION_SWEEP}-twilio_api_key", api_key).grant_read(fn)
        grant_secret_decrypt(self, fn, key)
        # The schedule in its own group, so the scheduler role's trust can name it
        # (aws:SourceArn must be a schedule group ARN, never a schedule's; Scheduler user guide,
        # "Confused deputy prevention").
        group = scheduler.CfnScheduleGroup(self, "ConversationSweepScheduleGroup",
                                           name=f"{cfg.name_prefix}-conversation-sweep")
        # L1 with our own role: the L2 target's grantInvoke adds ``<function ARN>:*`` (every version
        # and alias); this role may invoke the one unqualified function ARN, for this account and
        # this schedule group only.
        role = iam.Role(self, "ConversationSweepSchedulerRole",
                        assumed_by=iam.ServicePrincipal(
                            "scheduler.amazonaws.com",
                            conditions={"StringEquals": {"aws:SourceAccount": self.account,
                                                         "aws:SourceArn": group.attr_arn}}),
                        description="EventBridge Scheduler: invoke the conversation sweep only")
        grant_invoke(role, fn, sid="InvokeConversationSweep")
        scheduler.CfnSchedule(
            self, "ConversationSweepSchedule",
            group_name=group.ref,
            schedule_expression=f"rate({int(CONVERSATION_SWEEP_RATE.to_minutes())} minutes)",
            flexible_time_window=scheduler.CfnSchedule.FlexibleTimeWindowProperty(mode="OFF"),
            state="ENABLED",
            target=scheduler.CfnSchedule.TargetProperty(
                arn=fn.function_arn, role_arn=role.role_arn,
                retry_policy=scheduler.CfnSchedule.RetryPolicyProperty(
                    maximum_retry_attempts=0,
                    maximum_event_age_in_seconds=int(CONVERSATION_SWEEP_RATE.to_seconds()))),
            description="Human-leg sweep: close conversations whose call ended over 5 minutes ago")
        alarms.alarm("ConversationSweepErrorsAlarm",
                     fn.metric_errors(period=CONVERSATION_SWEEP_ALARM_PERIOD, statistic="Sum"),
                     threshold=2, description="conversation_sweep Lambda errors (2 failed runs in 30 minutes)")
        alarms.alarm("ConversationSweepThrottlesAlarm",
                     fn.metric_throttles(period=CONVERSATION_SWEEP_ALARM_PERIOD, statistic="Sum"),
                     threshold=1, description="conversation_sweep Lambda throttles")

    def _recap_writer(self, cfg: InfraConfig, code: lambda_.Code, deps: lambda_.ILayerVersion,
                      key: kms.IKey, token_key: kms.IKey, tables: dict[str, ddb.ITable],
                      api_key: sm.ISecret) -> None:
        """The post-call recap writer.

        * Lambda ``recap_writer`` from the shared factory (own role with a fixed name, own log group),
          256 MB, 60 s, one async retry. No URL, no API, no trigger: the front door (and the
          sweep, :meth:`_conversation_sweep`) invoke it by its unqualified ARN only.
        * IAM, nothing else: :data:`RECAP_TABLE_ACTIONS` plus the catalog ``Scan`` (data CMK through
          DynamoDB only); ``bedrock:InvokeModel`` (the Converse API's action) on the recap model's
          inference profile and the foundation model only through it
          (:func:`bedrock_invoke_statements`); ``comprehend:DetectPiiEntities`` on ``*``
          (:data:`PII_DETECT_REASON`: no resource-level permissions exist for it); ``kms:Decrypt``
          on the token key with a ``call_sid`` context (unsealing ``sealed_sensitive``; the key
          policy names this role, data_stack.py); the ``twilio_api_key`` secret by its one ARN and
          the data CMK's ``kms:Decrypt`` through Secrets Manager only.
        * Env: the Memory store, the model, the token key, the secret and the four table names."""
        env = {"OWLBANK_LOG_LEVEL": cfg.python_log_level,
               **table_env(tables, RECAP_TABLES),
               SECRET_ENV["twilio_api_key"]: api_key.secret_arn,
               MEMORY_STORE_ENV: cfg.twilio_outputs["memoryStoreId"],
               RECAP_MODEL_ENV: cfg.recap_model_id,
               TOKEN_KEY_ENV: token_key.key_arn}
        fn = self._function(RECAP_WRITER, code, [deps], key, env, retry_attempts=RECAP_RETRY_ATTEMPTS)
        self.functions[RECAP_WRITER] = fn
        self.invoke_targets[RECAP_WRITER] = fn
        grant_table_actions(self, fn, key, {n: tables[n] for n in RECAP_TABLE_ACTIONS}, RECAP_TABLE_ACTIONS)
        fn.add_to_role_policy(catalog_scan_statement(tables["catalog"].table_arn))
        for st in bedrock_invoke_statements(self, cfg.recap_model_id, sid="RecapModel",
                                            actions=["bedrock:InvokeModel"]):
            fn.add_to_role_policy(st)
        bedrock_nag_ack(fn.role, cfg.recap_model_id)
        grant_pii_detect(fn)  # PiiGuard over the transcript and the model's output
        grant_token_key(fn, token_key, ["kms:Decrypt"], sid="SensitiveUnseal")
        import_secret(self, f"{RECAP_WRITER}-twilio_api_key", api_key).grant_read(fn)
        grant_secret_decrypt(self, fn, key)
        self._grant_recap_trigger(self.functions["frontdoor"], "InvokeRecapWriter")

    def _grant_recap_trigger(self, caller: lambda_.Function, sid: str) -> None:
        """``lambda:InvokeFunction`` on the writer's one unqualified ARN, and ``OWLBANK_RECAP_FUNCTION``."""
        writer = self.functions[RECAP_WRITER]
        grant_invoke(caller, writer, sid=sid)
        caller.add_environment(RECAP_FUNCTION_ENV, writer.function_arn)

    def _auth_service_welcome(self, cfg: InfraConfig, token_key: kms.IKey) -> None:
        """Segment traits: the auth service's web-activity welcome.

        It reads the traits to choose the welcome offer, with no model call. It already reads
        ``customers`` (``first_name``, ``memory_email``) and the ``twilio_api_key`` secret (the
        Memory reads' credentials; ACCESS above). It seals the prepared welcome to the call
         with the token key, so it gets ``kms:Encrypt`` only (the front door unseals), with
        the same ``call_sid`` encryption-context condition. The Memory
        store ID is ``memoryStoreId`` in ``twilioOutputs``."""
        auth = self.functions["auth_service"]
        auth.add_environment(TOKEN_KEY_ENV, token_key.key_arn)
        grant_token_key(auth, token_key, ["kms:Encrypt"], sid="WelcomeSealing")
        auth.add_environment(MEMORY_TRAIT_GROUPS_ENV, ",".join(cfg.memory_trait_groups))
        auth.add_environment("OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS", str(cfg.web_activity_max_age_hours))
        if "memoryStoreId" in cfg.twilio_outputs:
            auth.add_environment(MEMORY_STORE_ENV, cfg.twilio_outputs["memoryStoreId"])

    def _agentcore_front_door(self, cfg: InfraConfig, fd: lambda_.Function) -> None:
        """``agentcore``: the front door's Runtime settings and
        grants, and the warmer's. The Runtime is named by its fixed name prefix and DEFAULT endpoint,
        the presign role by its fixed name: Compute references nothing in the Bridge stack.
        No ``OWLBANK_RELAY_WS_URL`` and no nonce-carrier setting (the nonce is always
        ``<Parameter name="n">`` here)."""
        warmer = self.functions["runtime_warmer"]
        runtime = runtime_invoke_resources(self, cfg.runtime_name)
        presign_arn = role_arn(self, cfg.presign_role_name)
        for name, value in (("OWLBANK_HOSTING", "agentcore"),
                            ("OWLBANK_PRESIGN_ROLE_ARN", presign_arn),
                            ("OWLBANK_WARMER_FUNCTION", warmer.function_arn),
                            ("OWLBANK_URL_TTL_S", str(URL_TTL_S))):
            fd.add_environment(name, value)
        fd.add_to_role_policy(iam.PolicyStatement(
            sid="AssumePresignRole", actions=["sts:AssumeRole"], resources=[presign_arn]))
        grant_invoke(fd, warmer, sid="InvokeRuntimeWarmer")
        fd.add_to_role_policy(iam.PolicyStatement(
            sid="StopRuntimeSession", actions=["bedrock-agentcore:StopRuntimeSession"],
            resources=runtime))
        ack_runtime_prefix(self, fd.role, cfg.runtime_name)
        # The warmer: InvokeAgentRuntime on {R, E} only, and the relay parameter.
        warmer.add_environment(RELAY_PARAMETER_ENV, cfg.relay_parameter_name)
        grant_relay_parameter_read(self, warmer, cfg.relay_parameter_name)
        warmer.add_to_role_policy(iam.PolicyStatement(
            sid="InvokeRuntime", actions=["bedrock-agentcore:InvokeAgentRuntime"], resources=runtime))
        ack_runtime_prefix(self, warmer.role, cfg.runtime_name)

    def _prewarm_alarm(self, alarms: Alarms) -> None:
        """A ``runtime_prewarm_failed`` line with a throttling or quota code, 1 in 5 minutes."""
        pattern = logs.FilterPattern.all(
            logs.FilterPattern.string_value("$.event", "=", "runtime_prewarm_failed"),
            logs.FilterPattern.any(*(logs.FilterPattern.string_value("$.code", "=", c)
                                     for c in PREWARM_THROTTLE_CODES)))
        mf = logs.MetricFilter(self, "RuntimePrewarmThrottledFilter",
                               log_group=self.log_groups["runtime_warmer"], filter_pattern=pattern,
                               metric_namespace=f"OwlBank/{self.cfg.stage}/Runtime",
                               metric_name="RuntimePrewarmThrottled", metric_value="1", default_value=0)
        alarms.alarm("RuntimePrewarmThrottledAlarm",
                     mf.metric(statistic="Sum", period=Duration.minutes(5)), threshold=1,
                     description="Runtime pre-warm throttled or over quota (calls start cold, 5.3)")

    def _function(self, name: str, code: lambda_.Code, layers: list[lambda_.ILayerVersion],
                  key: kms.IKey, env: dict[str, str], *,
                  retry_attempts: int | None = None) -> lambda_.Function:
        cid = "".join(p.capitalize() for p in name.split("_"))
        fn, log_group = lambda_function(
            self, cid, cfg=self.cfg, handler=f"lambdas.{name}.handler.handler", code=code,
            layers=layers, key=key, env=env, log_group_name=f"/owlbank/lambda/{name}",
            description=f"OwlBank {name} Lambda (least privilege)",
            # Fixed names: the token key policy names these roles (the token key's policy).
            role_name={"frontdoor": self.cfg.frontdoor_role_name,
                       "auth_service": self.cfg.auth_service_role_name,
                       RECAP_WRITER: self.cfg.recap_writer_role_name}.get(name),
            function_name=self.cfg.function_name(name) if name in NAMED_FUNCTIONS else None,
            timeout_s=TIMEOUTS[name], memory_mb=MEMORY_MB.get(name) or self.cfg.memory_mb(name),
            # Async calls only: a retry would send a second one-time code after the front door has
            # given up (auth_service's background send; with an alias, the alias carries the
            # setting, see above), or pre-warm a session the call has already reached.
            retry_attempts=0 if name == "runtime_warmer" or (
                name == "auth_service" and not self.cfg.provisioned_concurrency_on) else retry_attempts,
            pii_detect=name in ("frontdoor", "context_api", RECAP_WRITER))
        self.log_groups[name] = log_group
        return fn


X_RAY_STAR_REASON = ("X-Ray tracing: xray:PutTraceSegments and xray:PutTelemetryRecords don't support "
                     "resource-level permissions, so their resource must be '*'.")


def lambda_function(scope: Stack, cid: str, *, cfg: InfraConfig, handler: str, code: lambda_.Code,
                    layers: list[lambda_.ILayerVersion], key: kms.IKey, env: dict[str, str],
                    log_group_name: str, description: str, timeout_s: int, memory_mb: int,
                    role_name: str | None = None, function_name: str | None = None,
                    retry_attempts: int | None = None,
                    pii_detect: bool = False) -> tuple[lambda_.Function, logs.LogGroup]:
    """One OwlBank Lambda: its CMK-encrypted log group ``<cid>Logs``, its own role ``<cid>Role``
    (writes to that log group only) and the function ``<cid>Fn`` (Python 3.12 on arm64, X-Ray,
    JSON logs). ``pii_detect``: the role's ``Resource::*`` reason also names Comprehend's PII
    detection (granted by the caller)."""
    log_group = log_group_fn(scope, f"{cid}Logs", cfg=cfg, key=key, name=log_group_name)
    role = iam.Role(scope, f"{cid}Role",
                    role_name=role_name,
                    assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
                    description=description)
    role.add_to_policy(iam.PolicyStatement(
        sid="WriteOwnLogs",
        actions=["logs:CreateLogStream", "logs:PutLogEvents"],
        resources=[log_group.log_group_arn],
    ))
    fn = lambda_.Function(
        scope, f"{cid}Fn",
        runtime=lambda_.Runtime.PYTHON_3_12,  # Python 3.12 everywhere
        architecture=lambda_.Architecture.ARM_64,
        handler=handler,
        code=code,
        layers=layers,
        role=role,
        function_name=function_name,
        log_group=log_group,
        timeout=Duration.seconds(timeout_s),
        memory_size=memory_mb,
        retry_attempts=retry_attempts,
        environment=env,
        tracing=lambda_.Tracing.ACTIVE,
        # Structured JSON logs. The app's own records show up as top-level keys next
        # to the runtime's timestamp/level/message/requestId, so Logs Insights can filter
        # on call_sid, event, component and so on.
        logging_format=lambda_.LoggingFormat.JSON,
        application_log_level_v2=getattr(lambda_.ApplicationLogLevel, cfg.lambda_application_log_level),
        # Lambda system logs have no ERROR level; config maps ERROR to WARN.
        system_log_level_v2=getattr(lambda_.SystemLogLevel, cfg.lambda_system_log_level),
    )
    reason = X_RAY_STAR_REASON
    if pii_detect:
        reason += " Also " + PII_DETECT_REASON
    ack(role, "AwsSolutions-IAM5[Resource::*]", reason + " Nothing else uses '*'.")
    ack(fn, "AwsSolutions-L1",
        "Python 3.12 is pinned on purpose (one Python version for the Lambdas, "
        "the bridge and the tests). Move all components together when upgrading.")
    ack(role, f"AwsSolutions-IAM5[Resource::<{scope.get_logical_id(log_group.node.default_child)}.Arn>:*]",
        "Log stream names are created at runtime, so the log-stream ARN under this "
        "function's own log group needs a trailing wildcard.")
    return fn, log_group
