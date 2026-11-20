"""CDK context parameters parsed into one typed object.

Every value comes from CDK context (``cdk.json``, ``-c key=value`` or the
``context`` dict passed to :func:`owlbank_infra.app_builder.build_app`). None of
them is a secret: secret values never pass through CDK. Synth needs no AWS
credentials: the account is a fake default unless ``accountId`` is given, and
nothing is looked up.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAKE_ACCOUNT = "111111111111"
PLACEHOLDER_CERT_SUFFIX = "certificate/00000000-0000-0000-0000-000000000000"
_E164 = re.compile(r"^\+[1-9][0-9]{6,14}$")
_ACCOUNT = re.compile(r"^[0-9]{12}$")
_REGION = re.compile(r"^[a-z]{2}(-[a-z]+)+-[0-9]$")
_DOMAIN = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_SECRETISH = re.compile(r"(token|secret|password|passwd|auth|credential|key)", re.I)
#: Keys from the twilio_setup output that are identifiers, even though they match _SECRETISH.
_SECRETISH_ALLOW = re.compile(r"(sid|_id|id)$", re.I)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,63}$", re.I)
#: Characters AWS allows in tag values.
_TAG_VALUE = re.compile(r"^[\w\s+=.:/@-]{1,256}$")


#: ``s3PublicAccessBlockMode`` values. ``explicit`` sets Block Public Access on each bucket;
#: ``account`` leaves it to S3's default for new buckets plus the account-level setting, for an
#: account whose organization SCP denies ``s3:PutBucketPublicAccessBlock``.
S3_PUBLIC_ACCESS_BLOCK_MODES = ("explicit", "account")
DEFAULT_S3_PUBLIC_ACCESS_BLOCK_MODE = "explicit"


#: ``promptCaching`` values.
PROMPT_CACHING_MODES = ("on", "off")
DEFAULT_PROMPT_CACHING = "on"
#: ``promptCacheTtl`` values: how long Bedrock keeps the cached prefix.
PROMPT_CACHE_TTLS = ("5m", "1h")
DEFAULT_PROMPT_CACHE_TTL = "5m"
#: Model-assisted slots: ``slotModeCeiling``, the most permissive slot mode
#: this deployment allows, least to most permissive. The front door and the admin API get it as
#: ``OWLBANK_SLOT_MODE_CEILING``; ``deterministic`` also drops the front door's Bedrock statement.
SLOT_MODES = ("deterministic", "assisted", "generative")
DEFAULT_SLOT_MODE_CEILING = "assisted"
#: Model-assisted slots: ``speechLog`` (each Gather turn's transcript on ``factor_answer``,
#: a sensitive slot's value as its placeholder) and ``speechLogRedaction`` (that text also through
#: the transcript redaction, a Comprehend call per answer). The front door gets them as
#: ``OWLBANK_SPEECH_LOG`` and ``OWLBANK_SPEECH_LOG_REDACTION``.
SPEECH_LOG_MODES = ("on", "off")
DEFAULT_SPEECH_LOG = "on"
DEFAULT_SPEECH_LOG_REDACTION = "on"

#: Human-leg conversation.
#: ``humanLegConversation``: the ``/voice/taskrouter`` and ``/voice/transcription`` routes (and the
#: front door's closer); both routes also require Flex mode (``workspaceSid`` and ``workflowSid``
#: in ``twilioOutputs``). Transcription itself is the policy item ``human_leg``, not a deploy flag.
HUMAN_LEG_MODES = ("off", "on")
DEFAULT_HUMAN_LEG_CONVERSATION = "on"
#: The ``twilioOutputs`` keys that together mean Flex mode.
FLEX_OUTPUT_KEYS = ("workspaceSid", "workflowSid")
#: Overall deadline of the escalation briefing (ms). Must stay <= 5000 (the front door Lambda
#: timeout is 10 s and Twilio waits 15 s).
DEFAULT_BRIEFING_BUDGET_MS = 4000
#: ``contextApiRestrictedKey``: the Context API reads its own secret (a Twilio Restricted key
#: with only the two TaskRouter reads it makes, written by ``scripts/twilio_setup.py``) instead of the
#: shared Standard ``twilio_api_key`` secret. On by default: the Context API never holds the Standard key.
DEFAULT_CONTEXT_API_RESTRICTED_KEY = "on"

#: Call observability. ``callObservability`` turns
#: tracing and the call metrics on; every other key below is read only when it is ``on``.
CALL_OBSERVABILITY_MODES = ("off", "on")
DEFAULT_CALL_OBSERVABILITY = "off"
#: ``traceContent``: what the bridge's spans carry of the agent's text. ``off`` in every
#: stage by default; ``full`` is refused at ``stage=prod``.
TRACE_CONTENT_MODES = ("off", "redacted", "full")
DEFAULT_TRACE_CONTENT = "off"
DEFAULT_TRACE_SAMPLE_RATE = 1.0
#: ``traceContentRetentionDays``: the traces log group's retention while text is exported.
TRACE_CONTENT_RETENTION_MIN_DAYS = 1
TRACE_CONTENT_RETENTION_MAX_DAYS = 30
DEFAULT_TRACE_CONTENT_RETENTION_DAYS = 7
#: The CloudWatch Logs retention values up to 30 days.
TRACE_CONTENT_RETENTION_VALUES = (1, 3, 5, 7, 14, 30)
#: Fixed-ceiling backstop alarms: seconds for the latency ceilings, tokens for the last.
CEILING_SECONDS_MIN = 0.1
CEILING_SECONDS_MAX = 60.0
DEFAULT_TTFT_CEILING_S = 3.0
DEFAULT_TURN_WAIT_CEILING_S = 5.0
DEFAULT_SILENCE_CEILING_S = 4.0
TOKENS_PER_CALL_CEILING_MAX = 10_000_000
DEFAULT_TOKENS_PER_CALL_CEILING = 60000
#: The context keys that only matter with ``callObservability=on`` (the "no effect" warning).
CALL_OBSERVABILITY_KEYS = ("traceSampleRate", "traceContent", "traceContentRetentionDays",
                           "tracesEnableTransactionSearch", "ttftCeilingSeconds",
                           "turnWaitCeilingSeconds", "silenceCeilingSeconds", "tokensPerCallCeiling")


#: Memory trait settings. A copy of ``owlbank.web_activity``'s
#: ``DEFAULT_TRAIT_GROUPS``, ``TRAIT_GROUP_NAME_RE``, ``MAX_TRAIT_GROUPS``, ``MAX_AGE_HOURS_RANGE`` and
#: ``DEFAULT_MAX_AGE_HOURS`` (infra can't import src/); tests/unit/test_trait_settings_parity.py
#: keeps them equal.
DEFAULT_MEMORY_TRAIT_GROUPS: tuple[str, ...] = ("Contact", "WebActivity")
TRAIT_GROUP_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAX_TRAIT_GROUPS = 10
WEB_ACTIVITY_MAX_AGE_HOURS_RANGE: tuple[int, int] = (1, 8760)
DEFAULT_WEB_ACTIVITY_MAX_AGE_HOURS = 24
#: The trait group that turns on the web-activity offer and the auth service's welcome model.
WEB_ACTIVITY_GROUP = "WebActivity"


#: Accepted ``logLevel`` values (canonical spelling). ``WARNING`` is accepted as ``WARN``.
LOG_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")
DEFAULT_LOG_LEVEL = "INFO"


def normalize_log_level(value: object) -> str:
    """Canonical log level (``DEBUG|INFO|WARN|ERROR``) for a ``logLevel`` context value.

    Case-insensitive; ``WARNING`` is an alias of ``WARN``. ``None`` or an empty string is the
    default (``INFO``). Anything else (an unknown word, a non-string) **falls back to INFO**
    with a ``UserWarning`` instead of failing synth: a typo in a logging knob must not block
    a deploy, and INFO is the safe, diagnosable default.
    """
    if value is None or value == "":
        return DEFAULT_LOG_LEVEL
    if isinstance(value, str):
        level = value.strip().upper()
        if level == "WARNING":
            level = "WARN"
        if level in LOG_LEVELS:
            return level
    warnings.warn(f"context 'logLevel' {value!r} is not one of {'|'.join(LOG_LEVELS)}; using "
                  f"{DEFAULT_LOG_LEVEL}", UserWarning, stacklevel=2)
    return DEFAULT_LOG_LEVEL


#: Lambda memory limits (MB): the service accepts 128 to 10240.
LAMBDA_MEMORY_MIN_MB, LAMBDA_MEMORY_MAX_MB = 128, 10240
#: Default memory of the latency-critical functions (``MEMORY_SIZED_FUNCTIONS``); CPU scales with it.
#: The others stay at 512 MB.
DEFAULT_LAMBDA_MEMORY_MB = 1024
#: The functions on the call's critical path: they get ``lambdaMemoryMb`` and, with provisioned
#: concurrency, the ``live`` alias. Everything that invokes them uses the alias ARN.
PROVISIONED_FUNCTIONS = ("frontdoor", "auth_service", "bank_tools")
MEMORY_SIZED_FUNCTIONS = PROVISIONED_FUNCTIONS
LIVE_ALIAS = "live"
#: ``lambdaProvisionedConcurrency`` bounds. The real ceiling is the account's unreserved
#: concurrency (100 must stay free), so a low default fits new accounts; see the production guide.
LAMBDA_PROVISIONED_CONCURRENCY_MAX = 100
DEFAULT_LAMBDA_PROVISIONED_CONCURRENCY = 1


#: AgentCore hosting: a container image digest, ``bridgeImageDigest``.
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: ``AWS::BedrockAgentCore::Runtime`` ``AgentRuntimeName`` pattern (CFN): letters, digits and
#: underscores, starting with a letter, at most 48 characters.
RUNTIME_NAME_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{0,47}")
#: AZ IDs where AgentCore supports VPC connectivity, from the AgentCore docs page *Configure Amazon
#: Bedrock AgentCore Runtime and tools for VPC* ("Supported Availability Zones").
#: Only the Region this blueprint deploys to is listed; add a Region from that table when needed.
#: The Runtime gets only the private subnets whose AZ ID is in this list.
AGENTCORE_SUPPORTED_AZ_IDS: Mapping[str, tuple[str, ...]] = {
    "us-east-1": ("use1-az1", "use1-az2", "use1-az4"),
}
#: The VPC spans exactly two AZs (two public, two private subnets); ``availabilityZones`` keeps that.
VPC_AZ_COUNT = 2

#: AgentCore hosting: ``bridgeHosting``. ``fargate`` (default) is today's ECS bridge.
BRIDGE_HOSTINGS = ("fargate", "agentcore")
DEFAULT_BRIDGE_HOSTING = "fargate"
#: ``agentcoreNetworkMode``: ``vpc`` (default, the private subnets) or ``public``.
AGENTCORE_NETWORK_MODES = ("vpc", "public")
DEFAULT_AGENTCORE_NETWORK_MODE = "vpc"
#: ``agentcoreNetworkModeAck``: its only accepted value, required for ``stage=prod`` + ``public``.
AGENTCORE_PUBLIC_AT_PROD_ACK = "public-at-prod"
#: The Runtime's ``LifecycleConfiguration.MaxLifetime`` (fixed, seconds); the idle timeout must be
#: below it. ``agentcoreIdleTimeoutSeconds`` lower bound and default.
AGENTCORE_MAX_LIFETIME_S = 28800
AGENTCORE_IDLE_TIMEOUT_MIN_S = 60
DEFAULT_AGENTCORE_IDLE_TIMEOUT_S = 900
#: ``agentcoreSessionQuota``: the account's active-session quota, the basis of the 80 %
#: alarm. Defaults from the AgentCore quotas page (5,000 confirmed in the sandbox).
AGENTCORE_SESSION_QUOTA_MAX = 1_000_000
AGENTCORE_SESSION_QUOTA_DEFAULTS: Mapping[str, int] = {"us-east-1": 5000, "us-west-2": 5000}
AGENTCORE_SESSION_QUOTA_OTHER_REGIONS = 2500
#: What ``agentcoreNetworkMode=public`` gives up, named in the synth warning.
AGENTCORE_PUBLIC_TRADE_OFFS = "egress control, VPC flow logs and the DynamoDB gateway endpoint"


class ConfigError(ValueError):
    """A context value is missing or malformed."""


#: ``basicAuthUser`` (``user`` of the secret): the user name in the generated
#: ``owlbank/twilio/basic-auth`` secret. Not a secret; the password is generated by Secrets Manager.
DEFAULT_BASIC_AUTH_USER = "owlbank"
BASIC_AUTH_USER_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
#: Context keys that belong to CDK itself, skipped by :func:`check_context_keys`: the
#: ``aws:*`` and ``@aws-cdk/*`` namespaces, and cached lookups, whose keys all contain ``:``.
CDK_CONTEXT_PREFIXES = ("aws:", "@aws-cdk/")
#: Plain-named keys the CDK CLI itself writes to ``cdk.context.json`` (``cdk acknowledge <id>``
#: stores ``acknowledged-issue-numbers``; ``cdk cli-telemetry --disable`` stores ``cli-telemetry``),
#: also skipped.
CDK_CLI_CONTEXT_KEYS = frozenset({"acknowledged-issue-numbers", "cli-telemetry"})


def check_context_keys(keys: Iterable[str], known: Iterable[str]) -> None:
    """Refuse any context key that isn't a known, reviewed setting.

    Secret values never travel as context, so a key nobody listed (``twilioAuthToken``, or a
    typo of a real key) fails synth instead of being silently ignored. Only the key is named,
    never its value."""
    allowed = set(known) | CDK_CLI_CONTEXT_KEYS
    unknown = sorted(k for k in keys
                     if k not in allowed and not k.startswith(CDK_CONTEXT_PREFIXES) and ":" not in k)
    if unknown:
        raise ConfigError(
            f"unknown context key(s): {', '.join(map(repr, unknown))}. Every setting is in the "
            "known-keys list in infra/owlbank_infra/app_builder.py; check the spelling. Secret "
            "values are never context: set them in Secrets Manager after deploying OwlBankData "
            "(see infra/README.md, \"First deploy\")")


#: The agent's model list:
#: ``modelIds`` (one to three inference-profile IDs in preference order), ``modelParams`` (per-entry
#: overrides keyed by model ID) and ``failoverTest`` (the test-only latency injection, never at prod).
MODEL_LIST_MIN, MODEL_LIST_MAX = 1, 3
#: Geography prefix of a deployment Region (the system-defined inference-profile prefix), the same
#: table as ``infra/scripts/resolve_models.py``. A Region not listed has no geographic default.
REGION_GEOS: tuple[tuple[str, str], ...] = (("us-gov-", "us-gov"), ("us-", "us"), ("eu-", "eu"),
                                            ("ap-", "apac"))
GLOBAL_PROFILE_PREFIX = "global"
#: The default list: Haiku 5.5, Haiku 4.5, Sonnet 5.5, each through the Region's geography.
DEFAULT_MODEL_BASES: tuple[str, ...] = ("anthropic.claude-haiku-5-5",
                                        "anthropic.claude-haiku-4-5-20251001-v1:0",
                                        "anthropic.claude-sonnet-5-5")
#: An inference-profile ID: ``<prefix>.<provider>.<model>``.
_PROFILE_ID = re.compile(r"(?P<prefix>[a-z]+(?:-[a-z]+)*)\.(?P<base>[a-z0-9-]+\.[A-Za-z0-9._:-]{1,120})")
#: Per-entry thinking settings: ``adaptive``, ``off``, ``default``, ``between_tools`` and
#: ``enabled:<budget>`` (a budget of at least :data:`MIN_THINKING_BUDGET`, below ``max_tokens``).
ENTRY_THINKING_MODES = ("adaptive", "off", "default", "between_tools")
ENABLED_THINKING_PREFIX = "enabled:"
MIN_THINKING_BUDGET = 1024
#: Output tokens per model call, per entry.
MAX_TOKENS_RANGE: tuple[int, int] = (64, 8192)
#: The keys a ``modelParams`` entry may set (the ``OWLBANK_MODELS`` entry fields but ``id``).
MODEL_PARAM_KEYS = ("thinking", "max_tokens", "text_release", "prompt_caching")
#: Known per-model defaults (live): ``(thinking, text_release)`` by the model's base ID (a
#: base matches its name or ``<name>-<anything>``). Any other model gets reasoning off with whole
#: messages, which every model accepts (Haiku 4.5's default).
KNOWN_MODEL_DEFAULTS: Mapping[str, tuple[str, str]] = {
    "anthropic.claude-haiku-5-5": ("adaptive", "stream"),
    "anthropic.claude-sonnet-5-5": ("adaptive", "stream"),
}
OTHER_MODEL_DEFAULT: tuple[str, str] = ("off", "message")
#: Thinking settings Bedrock refuses for a known model (live): ``enabled`` stands for any
#: ``enabled:<budget>``. Refused at synth naming the entry.
KNOWN_THINKING_REFUSALS: Mapping[str, tuple[str, ...]] = {
    "anthropic.claude-haiku-5-5": ("enabled",),
    "anthropic.claude-haiku-4-5": ("adaptive",),
    "anthropic.claude-sonnet-5-5": ("off", "enabled"),
}
#: ``failoverTest``: ``off`` or the seconds the preferred entry's first token is delayed (the
#: a test-only latency injection), set as ``OWLBANK_FAILOVER_TEST=<seconds>`` on the
#: bridge; above 0, at most this; refused at ``stage=prod``.
FAILOVER_TEST_MAX_S = 30.0
#: ``modelFailover``: ``on`` (default) builds the fleet picture (the ``modelstats``
#: table, the poller, its alarms and the table env) when ``modelIds`` lists two or more entries;
#: ``off`` keeps only the first entry, so the stacks are today's single-model shape.
MODEL_FAILOVER_MODES = ("on", "off")
DEFAULT_MODEL_FAILOVER = "on"


@dataclass(frozen=True)
class ModelEntry:
    """One entry of the agent's model list, as the bridge reads it from ``OWLBANK_MODELS``."""

    id: str
    thinking: str
    max_tokens: int
    text_release: str
    prompt_caching: bool

    def as_env(self) -> dict[str, Any]:
        return {"id": self.id, "thinking": self.thinking, "max_tokens": self.max_tokens,
                "text_release": self.text_release, "prompt_caching": self.prompt_caching}

    @property
    def profile_prefix(self) -> str:
        return self.id.split(".", 1)[0]

    @property
    def base_model(self) -> str:
        return self.id.split(".", 1)[1]

    @property
    def is_global(self) -> bool:
        return self.profile_prefix == GLOBAL_PROFILE_PREFIX


def region_geo(region: str) -> str | None:
    """The geography prefix of ``region`` (``us-east-1`` -> ``us``), or ``None`` if not known."""
    for start, geo in REGION_GEOS:
        if region.startswith(start):
            return geo
    return None


def _known(table: Mapping[str, Any], base: str) -> Any:
    for name, value in table.items():
        if base == name or base.startswith(f"{name}-"):
            return value
    return None


def default_model_ids(region: str) -> tuple[str, ...]:
    """The default list for ``region``: the three default models through its geography's profiles."""
    geo = region_geo(region)
    if geo is None:
        raise ConfigError(f"modelIds: no geography is known for region {region!r}, so there is no "
                          "default model list; pass -c modelIds='[\"global.<model>\", ...]' (global. "
                          "profiles may be processed in any commercial Region)")
    return tuple(f"{geo}.{base}" for base in DEFAULT_MODEL_BASES)


def _default_models() -> tuple[ModelEntry, ...]:
    return tuple(_model_entry(mid, {}, DEFAULT_PROMPT_CACHING == "on") for mid in default_model_ids("us-east-1"))


@dataclass(frozen=True)
class InfraConfig:
    account: str = FAKE_ACCOUNT
    region: str = "us-east-1"
    stage: str = "dev"
    domain_name: str | None = None
    certificate_arn: str | None = None
    developer_phone: str | None = None
    test_numbers: tuple[str, ...] = ()
    bridge_desired_count: int = 1
    #: The agent's ordered model list (``modelIds`` + ``modelParams``), the
    #: bridge's ``OWLBANK_MODELS``. Every entry is the Region's geography profile or a ``global.``
    #: one, and the bridge role may invoke exactly these.
    models: tuple[ModelEntry, ...] = field(default_factory=_default_models)
    #: ``failoverTest``: the test-only first-token delay of the first entry, in seconds
    #: (``None`` = off), never at ``stage=prod``.
    failover_test: float | None = None
    #: ``modelFailover``: ``on`` or ``off``. Read :attr:`failover_on`.
    model_failover: str = DEFAULT_MODEL_FAILOVER
    resolver_model_id: str = "us.anthropic.claude-haiku-5-5"
    alb_idle_timeout_s: int = 900
    deregistration_delay_s: int = 300
    nat_gateways: int = 1
    flex_origin: str = "https://flex.twilio.com"
    #: The Twilio number the bridge's TAC config needs (TWILIO_PHONE_NUMBER); not a secret.
    twilio_phone_number: str | None = None
    #: Twilio Verify service SID for auth_service (context, or the twilio_setup output).
    verify_service_sid: str | None = None
    twilio_outputs: Mapping[str, str] = field(default_factory=dict)
    #: Optional email subscribed to the alarm topic. Not a secret.
    alarm_email: str | None = None
    #: WAF rate-based rule limits, requests per 5 minutes per source IP.
    #: The API's default is 2000: in the sandbox Twilio calls the webhooks from a handful of
    #: egress IPs at low volume (a call is a few dozen requests), so 2000 per 5 minutes per IP
    #: leaves wide headroom for tests and still cuts off a flood from one source. It is a
    #: per-IP backstop, not the volume control (that is the stage throttling, 50 rps). Every
    #: Twilio call shares those IPs, so production must raise it after a load test.
    waf_api_rate_limit: int = 2000
    waf_alb_rate_limit: int = 1000
    #: Rate rule of the CloudFront web ACL (CloudFront mode). High on purpose, like the API's:
    #: Twilio's egress IPs are shared by all calls, so a per-IP rule sees all traffic at once.
    waf_cloudfront_rate_limit: int = 10000
    #: CloudFront mode: origin response (read) and keep-alive timeouts of the VPC origin, in
    #: seconds. The quotas are 1-120 and 1-300 (both raisable).
    cloudfront_origin_read_timeout_s: int = 60
    cloudfront_origin_keepalive_timeout_s: int = 60
    #: CloudFront mode: ID of the managed prefix list com.amazonaws.global.cloudfront.origin-facing
    #: in this Region (IDs differ per Region; CloudFront mode runs in us-east-1 only). The
    #: internal ALB's security group accepts port 80 from it. Check it with
    #: ``aws ec2 describe-managed-prefix-lists --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing``.
    cloudfront_prefix_list_id: str = "pl-3b927c52"
    #: Action of the AWS managed rule groups in every web ACL: ``count`` observes (the
    #: groups' override action is Count, so nothing they match is blocked and the WAF logs show
    #: what would have been); ``block`` enforces. ``None`` in the context means the stage's
    #: default, see :attr:`waf_managed_mode`.
    waf_managed_rule_mode: str | None = None
    #: CloudFront mode: version ID of the origin-verify secret to pin the dynamic references to.
    #: Unset: the references follow the secret's current version, but CloudFormation only
    #: re-reads it when the referencing resource changes. Set it after rotating the secret.
    origin_verify_secret_version: str | None = None
    tag_owner: str = "unassigned"
    tag_cost_center: str = "unassigned"
    tag_data_classification: str = "confidential"
    #: Canonical log level (``DEBUG|INFO|WARN|ERROR``) for the Lambdas and the bridge.
    log_level: str = DEFAULT_LOG_LEVEL
    #: ``explicit`` (default): every bucket sets all four Block Public Access flags itself.
    #: ``account``: the buckets omit ``PublicAccessBlockConfiguration`` because an organization SCP
    #: denies ``s3:PutBucketPublicAccessBlock``; protection comes from S3's default for
    #: new buckets (all four on) plus the account-level setting, both checked by preflight.
    s3_public_access_block_mode: str = DEFAULT_S3_PUBLIC_ACCESS_BLOCK_MODE
    #: Memory (MB) of ``frontdoor``, ``auth_service`` and ``bank_tools``. Lambda gives CPU in
    #: proportion to memory, so this is also the cold-start and per-call speed knob.
    lambda_memory_mb: int = DEFAULT_LAMBDA_MEMORY_MB
    #: Provisioned concurrency on the ``live`` alias of ``frontdoor``, ``auth_service`` and
    #: ``bank_tools``. 0 turns it off: no alias, and every invoker uses the unqualified ARN.
    lambda_provisioned_concurrency: int = DEFAULT_LAMBDA_PROVISIONED_CONCURRENCY
    #: Memory trait groups whose traits may reach the model (``memoryTraitGroups``, the bridge's
    #: ``OWLBANK_MEMORY_TRAIT_GROUPS``). Names match
    #: ``^[A-Za-z0-9_.-]{1,64}$``, 1 to 10, no duplicates.
    memory_trait_groups: tuple[str, ...] = DEFAULT_MEMORY_TRAIT_GROUPS
    #: Oldest web activity the agent is offered, in hours (``webActivityMaxAgeHours``, the bridge's
    #: ``OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS``): 1 to 8760.
    web_activity_max_age_hours: int = DEFAULT_WEB_ACTIVITY_MAX_AGE_HOURS
    #: Accept spoken factor digits as well as keypad. ``None`` = the default: on in
    #: every stage, prod included. ``false`` is a ceiling: every verification factor is
    #: keypad-only (spoken values pass through speech recognition and can be overheard).
    accept_spoken_factors_override: bool | None = None
    #: How the caller may interrupt scripted read-outs: the front door's
    #: ``welcomeGreetingInterruptible`` (``any``, ``speech``, ``dtmf``, ``none``).
    scripted_barge_in: str = "any"
    #: The front door's Gather ``speechTimeout`` for spoken answers,
    #: "auto" or whole seconds 1-10. With "auto" the Gather drops ``speechModel`` (Twilio requires an
    #: integer with a model); with seconds it keeps Deepgram.
    speech_timeout: str = "3"
    #: The Gather ``speechTimeout`` for spoken yes/no confirmations, which callers
    #: answer in phrases ("Yeah, that's correct"); same values and "auto" rule as ``speech_timeout``.
    confirm_speech_timeout: str = "2"
    #: Keyed-phone login: whole seconds (1-10) a keypad Gather waits for the next key (front door
    #: ``OWLBANK_KEYPAD_TIMEOUT``).
    keypad_timeout: str = "3"
    #: One Orchestrator conversation per call (front door ``conversationId`` on
    #: reconnects, bridge stops closing it per segment). Default on: a measurement call showed
    #: one conversation reconnects after step-up in 2.9 s instead of about 8 s. ``false``
    #: restores a new conversation per segment, each closed when its segment ends.
    reuse_conversation: bool = True
    #: Spoken digit factors end as soon as a partial result holds a complete
    #: answer (``partialResultCallback`` + a REST call update). ``false`` gives today's Gathers.
    early_digits: bool = True
    #: AI keypad shortcuts: in the AI session the keypad is read by the bridge, never
    #: sent to the model (1 yes, 2 no, 0 agent, 9 repeat). ``off`` restores today's TwiML.
    ai_keypad_shortcuts: str = "on"
    #: The bank's IANA time zone, for the date and time in the agent's system prompt.
    timezone: str = "America/New_York"
    #: Bedrock prompt caching in the agent loop (``on`` | ``off``, default on).
    prompt_caching: str = DEFAULT_PROMPT_CACHING
    #: How long Bedrock keeps the cached prefix (``5m`` | ``1h``), one value for every entry.
    prompt_cache_ttl: str = DEFAULT_PROMPT_CACHE_TTL
    #: Deploy the ``OwlBankAdmin`` stack (the admin console). Default on for dev
    #: and test in us-east-1, off for prod (until reviewed) and in other Regions (the console's
    #: CLOUDFRONT-scope web ACL must live in us-east-1, see :func:`_admin_console`).
    admin_console: bool = True
    #: Email of the first console user (an ``admin``), created at deploy. Optional,
    #: command line only (never stored in the repository), never logged. Not a secret.
    admin_email: str | None = field(default=None, repr=False)
    #: Version ID of the admin console's ``X-Origin-Verify`` secret to pin its dynamic
    #: references to (the distribution's ``/api/*`` origin header and the API's WAF rule), as
    #: ``originVerifySecretVersion`` does for the bridge. Set it after rotating that secret.
    admin_origin_verify_secret_version: str | None = None
    #: ``bridgeImageDigest``: ``sha256:<64 hex>`` of the bridge image in our ECR repository.
    #: Optional here; the stacks that need it call :meth:`require_bridge_image_digest`.
    bridge_image_digest: str | None = None
    #: Accept a Runtime in a single supported AZ at ``stage=prod``.
    agentcore_single_az_ack: bool = False
    #: Optional explicit AZ names of the VPC (exactly two, in ``region``). ``None`` keeps
    #: :attr:`azs` as before (``<region>a``, ``<region>b``). The only setting that changes the VPC.
    availability_zones: tuple[str, ...] | None = None
    #: AgentCore hosting: ``bridgeHosting`` (``fargate`` | ``agentcore``).
    bridge_hosting: str = DEFAULT_BRIDGE_HOSTING
    #: ``agentcoreNetworkMode`` (``vpc`` | ``public``). Meaningful only in ``agentcore``: read
    #: :attr:`agentcore_in_vpc`, which is False on ``fargate`` whatever this says.
    agentcore_network_mode: str = DEFAULT_AGENTCORE_NETWORK_MODE
    #: ``agentcoreIdleTimeoutSeconds``: the Runtime's idle session timeout, below
    #: :data:`AGENTCORE_MAX_LIFETIME_S`.
    agentcore_idle_timeout_s: int = DEFAULT_AGENTCORE_IDLE_TIMEOUT_S
    #: ``agentcoreSessionQuota``: active Runtime sessions the account may hold.
    agentcore_session_quota: int = 5000
    #: ``slotModeCeiling``: one of :data:`SLOT_MODES`. Code review only: not
    #: settable from the console or the policy table.
    slot_mode_ceiling: str = DEFAULT_SLOT_MODE_CEILING
    #: ``speechLog``: ``on`` (default, every stage) or ``off``.
    speech_log: str = DEFAULT_SPEECH_LOG
    #: ``speechLogRedaction``: ``on`` (default, every stage) or ``off``.
    speech_log_redaction: str = DEFAULT_SPEECH_LOG_REDACTION
    #: ``basicAuthUser``: the fixed user in the generated Basic auth secret.
    basic_auth_user: str = DEFAULT_BASIC_AUTH_USER
    #: ``callObservability``: ``off`` (default) or ``on``. Read :attr:`call_observability_on`.
    call_observability: str = DEFAULT_CALL_OBSERVABILITY
    #: ``traceSampleRate``: the share of calls traced, 0.0 to 1.0.
    trace_sample_rate: float = DEFAULT_TRACE_SAMPLE_RATE
    #: ``traceContent``: one of :data:`TRACE_CONTENT_MODES`.
    trace_content: str = DEFAULT_TRACE_CONTENT
    #: ``traceContentRetentionDays``: traces log group retention while ``traceContent`` != off.
    trace_content_retention_days: int = DEFAULT_TRACE_CONTENT_RETENTION_DAYS
    #: ``tracesEnableTransactionSearch``: the stack enables X-Ray Transaction Search itself.
    traces_enable_transaction_search: bool = False
    #: Fixed-ceiling alarms: ``ttftCeilingSeconds``, ``turnWaitCeilingSeconds``,
    #: ``silenceCeilingSeconds`` (seconds) and ``tokensPerCallCeiling`` (tokens).
    ttft_ceiling_seconds: float = DEFAULT_TTFT_CEILING_S
    turn_wait_ceiling_seconds: float = DEFAULT_TURN_WAIT_CEILING_S
    silence_ceiling_seconds: float = DEFAULT_SILENCE_CEILING_S
    tokens_per_call_ceiling: int = DEFAULT_TOKENS_PER_CALL_CEILING
    #: ``humanLegConversation``: ``on`` (default) or ``off``. Read
    #: :attr:`human_leg_conversation_on`.
    human_leg_conversation: str = DEFAULT_HUMAN_LEG_CONVERSATION
    #: ``contextApiRestrictedKey``: ``on`` (default) or ``off``. Read
    #: :attr:`context_api_restricted_key_on`.
    context_api_restricted_key: str = DEFAULT_CONTEXT_API_RESTRICTED_KEY

    @property
    def flex_mode(self) -> bool:
        """True iff the outputs file holds both ``workspaceSid`` and ``workflowSid`` (written by
        ``twilio_setup --with-flex``). Exactly one of them is a synth error: the setup was partial."""
        has = [bool(self.twilio_outputs.get(k)) for k in FLEX_OUTPUT_KEYS]
        if any(has) and not all(has):
            missing = FLEX_OUTPUT_KEYS[has.index(False)]
            raise ConfigError(f"twilioOutputs has one of {' and '.join(FLEX_OUTPUT_KEYS)} but not {missing}: "
                              "re-run twilio_setup with --with-flex")
        return all(has)

    @property
    def human_leg_conversation_on(self) -> bool:
        return self.human_leg_conversation == "on" and self.flex_mode

    @property
    def context_api_restricted_key_on(self) -> bool:
        return self.context_api_restricted_key == "on"

    @property
    def recap_writer_on(self) -> bool:
        """The recap writer exists only when ``twilioOutputs`` names the Memory
        store (``memoryStoreId``), which its handler requires. Without it there is no function, no
        token-key statement, and no ``OWLBANK_RECAP_FUNCTION`` (the front door uses ``NoopRecap``)."""
        return bool(self.twilio_outputs.get("memoryStoreId"))

    @property
    def recap_model_id(self) -> str:
        """The recap writer's model: the slot resolver's (the handler's default is the resolver's
        Haiku), passed explicitly so the IAM statement and ``OWLBANK_RECAP_MODEL_ID`` always agree."""
        return self.resolver_model_id

    @property
    def taskrouter_route_on(self) -> bool:
        """``/voice/taskrouter`` exists when either user needs it: call observability's queue-wait
        span or the human leg's participants. It is the TaskRouter event callback, so Flex only."""
        return (self.call_observability_on or self.human_leg_conversation_on) and self.flex_mode

    @property
    def briefing_on(self) -> bool:
        """The escalation briefing: on without Flex, and only when a model may be involved (it reuses
        the slot resolver's model and Bedrock grant)."""
        return not self.flex_mode and self.slot_resolver_allowed

    @property
    def slot_resolver_allowed(self) -> bool:
        """False with ``slotModeCeiling=deterministic``: no slot is resolved by a model, so the
        front-door role gets no ``bedrock:InvokeModel`` statement."""
        return self.slot_mode_ceiling != "deterministic"

    @property
    def is_agentcore(self) -> bool:
        """True with ``bridgeHosting=agentcore``: the bridge runs as an AgentCore Runtime."""
        return self.bridge_hosting == "agentcore"

    @property
    def agentcore_in_vpc(self) -> bool:
        """True when the Runtime runs in the private subnets (``agentcore`` + ``vpc``)."""
        return self.is_agentcore and self.agentcore_network_mode == "vpc"

    def require_twilio_phone_number(self, needed_by: str) -> str:
        """``twilioPhoneNumber`` (TAC's ``TWILIO_PHONE_NUMBER``; the bridge exits without it), or a
        :class:`ConfigError` naming the key and ``needed_by``. Passed on the command line only."""
        if not self.twilio_phone_number:
            raise ConfigError(f"twilioPhoneNumber is required by {needed_by} (the bridge's TAC "
                              "configuration needs it): pass -c twilioPhoneNumber=<E.164>")
        return self.twilio_phone_number

    def require_bridge_image_digest(self, needed_by: str) -> str:
        """The image digest, or a :class:`ConfigError` naming ``bridgeImageDigest`` and ``needed_by``.
        :func:`from_context` doesn't require it, so the image repository can be deployed before the
        first image exists (push-then-deploy)."""
        if self.bridge_image_digest is None:
            raise ConfigError(f"bridgeImageDigest is required by {needed_by}: push the image with "
                              "infra/scripts/push_bridge_image.py and pass -c bridgeImageDigest=sha256:...")
        return self.bridge_image_digest

    @property
    def runtime_name(self) -> str:
        """``AgentRuntimeName`` of the bridge Runtime (``agentcore``): ``owlbank_<stage>_bridge``."""
        return runtime_name_for(self.stage, "bridge")

    @property
    def presign_role_name(self) -> str:
        """Fixed name of the role the front door assumes to pre-sign Runtime URLs, so its
        trust and the front door's grant can name it without a reference."""
        return f"{self.name_prefix}-relay-presign"

    @property
    def relay_parameter_name(self) -> str:
        """The SSM String parameter the Bridge stack writes and Compute's Lambdas read at start-up
        on both hostings: the relay host (``fargate``) or the Runtime ARN
        (``agentcore``). A fixed name instead of a CloudFormation export, so a hosting switch never
        has to remove an export another stack still imports."""
        return f"/owlbank/{self.stage}/bridge/relay"

    @property
    def image_repo_name(self) -> str:
        """ECR repository of the bridge image (``OwlBankBridgeImage``), on both hostings."""
        return f"{self.name_prefix}-bridge"

    @property
    def supported_az_ids(self) -> tuple[str, ...]:
        """AZ IDs where AgentCore VPC mode is supported in :attr:`region` (empty if not listed)."""
        return AGENTCORE_SUPPORTED_AZ_IDS.get(self.region, ())

    @property
    def prompt_caching_on(self) -> bool:
        return self.prompt_caching == "on"

    @property
    def call_observability_on(self) -> bool:
        return self.call_observability == "on"

    @property
    def failover_on(self) -> bool:
        """True when the fleet picture is built: ``modelFailover=on`` and two or more entries.
        Otherwise the stacks are today's single-model shape (no table, poller, alarms or table env)."""
        return self.model_failover == "on" and len(self.models) > 1

    @property
    def failover_table_name(self) -> str:
        """Fixed name of the ``modelstats`` table, so the Bridge and Compute stacks name
        it by ARN without a CloudFormation export, and ``modelFailover`` can be turned off without
        removing an export another stack still imports."""
        return f"{self.name_prefix}-modelstats"

    @property
    def provisioned_concurrency_on(self) -> bool:
        return self.lambda_provisioned_concurrency > 0

    def memory_mb(self, logical: str) -> int:
        """Memory of a function: ``lambdaMemoryMb`` for the critical-path three, 512 for the rest."""
        if logical in MEMORY_SIZED_FUNCTIONS:
            return self.lambda_memory_mb
        return 512

    def invoke_arn(self, logical: str, function_arn: str) -> str:
        """The ARN callers invoke (and IAM grants name): ``<function_arn>:live`` for the three
        provisioned functions while provisioned concurrency is on, else ``function_arn`` itself.
        Never a ``:*`` pattern and never an unqualified fallback next to the alias."""
        if self.provisioned_concurrency_on and logical in PROVISIONED_FUNCTIONS:
            return f"{function_arn}:{LIVE_ALIAS}"
        return function_arn

    @property
    def explicit_public_access_block(self) -> bool:
        """True when buckets set Block Public Access themselves (``s3PublicAccessBlockMode=explicit``)."""
        return self.s3_public_access_block_mode == "explicit"

    @property
    def python_log_level(self) -> str:
        """Level name for ``OWLBANK_LOG_LEVEL`` (Python's names: ``WARN`` becomes ``WARNING``)."""
        return "WARNING" if self.log_level == "WARN" else self.log_level

    @property
    def lambda_application_log_level(self) -> str:
        """Name of ``lambda.ApplicationLogLevel`` (DEBUG, INFO, WARN or ERROR)."""
        return self.log_level

    @property
    def lambda_system_log_level(self) -> str:
        """Name of ``lambda.SystemLogLevel``. Lambda's system logs only have DEBUG, INFO and
        WARN, so ``ERROR`` maps to ``WARN`` (the closest, quietest level)."""
        return "WARN" if self.log_level == "ERROR" else self.log_level

    @property
    def custom_domain_mode(self) -> bool:
        """True with a ``domainName`` (public ALB, ACM certificate, ``execute-api`` off).

        False is the CloudFront mode (no domain): internal ALB behind a CloudFront VPC origin.
        """
        return self.domain_name is not None

    @property
    def waf_managed_mode(self) -> str:
        """``count`` or ``block``: the context value, else ``count`` for dev and test, ``block`` for prod."""
        return self.waf_managed_rule_mode or ("block" if self.is_prod else "count")

    @property
    def is_dev_or_test(self) -> bool:
        return self.stage in ("dev", "test")

    @property
    def is_prod(self) -> bool:
        return self.stage == "prod"

    @property
    def name_prefix(self) -> str:
        """Prefix for named resources (roles, web ACLs, log groups, topics)."""
        return f"owlbank-{self.stage}"

    @property
    def frontdoor_role_name(self) -> str:
        """Fixed name, so the Data stack's token-key policy can name it without a cycle."""
        return f"{self.name_prefix}-frontdoor"

    @property
    def auth_service_role_name(self) -> str:
        """Fixed name, so the token-key policy can name it (it seals the prepared welcome)."""
        return f"{self.name_prefix}-auth-service"

    @property
    def bridge_task_role_name(self) -> str:
        """Fixed name, so the Data stack's token-key policy can name it without a cycle."""
        return f"{self.name_prefix}-bridge-task"

    @property
    def recap_writer_role_name(self) -> str:
        """Fixed name, so the token-key policy can name it (it unseals the call's factor values to
        scrub them from the recap)."""
        return f"{self.name_prefix}-recap-writer"

    def function_name(self, logical: str) -> str:
        """Deterministic Lambda name, so other stacks can build its ARN without a reference.
        Only the functions other stacks need are given this name."""
        return f"{self.name_prefix}-{logical.replace('_', '-')}"

    @property
    def azs(self) -> list[str]:
        # Explicit AZs: no availability-zone lookup at synth.
        if self.availability_zones is not None:
            return list(self.availability_zones)
        return [f"{self.region}a", f"{self.region}b"]

    @property
    def certificate_is_placeholder(self) -> bool:
        return self.certificate_arn is None

    def effective_certificate_arn(self) -> str:
        if self.certificate_arn:
            return self.certificate_arn
        return f"arn:aws:acm:{self.region}:{self.account}:{PLACEHOLDER_CERT_SUFFIX}"

    @property
    def bridge_host(self) -> str | None:
        return f"relay.{self.domain_name}" if self.domain_name else None

    @property
    def accept_spoken_factors(self) -> bool:
        if self.accept_spoken_factors_override is not None:
            return self.accept_spoken_factors_override
        return True

    @property
    def api_host(self) -> str | None:
        return f"voice.{self.domain_name}" if self.domain_name else None


def runtime_name_for(stage: str, kind: str) -> str:
    """``owlbank_<stage>_<kind>``, checked against the CFN ``AgentRuntimeName`` pattern."""
    name = f"owlbank_{stage}_{kind}"
    if not RUNTIME_NAME_PATTERN.fullmatch(name):
        raise ConfigError(f"runtime name {name!r} does not match the AgentRuntimeName pattern "
                          "(a letter, then letters, digits or underscores, at most 48)")
    return name


def _image_digest(ctx: Mapping[str, Any]) -> str | None:
    """``bridgeImageDigest``: unset, or exactly ``sha256:`` and 64 lowercase hex digits."""
    raw = ctx.get("bridgeImageDigest")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not _IMAGE_DIGEST.fullmatch(raw.strip()):
        raise ConfigError(f"bridgeImageDigest must be sha256:<64 lowercase hex>, got {raw!r}")
    return raw.strip()


def _bridge_hosting(ctx: Mapping[str, Any], *, stage: str, region: str,
                    reuse_conversation: bool, image_digest: str | None) -> dict[str, Any]:
    """The AgentCore hosting keys, as ``InfraConfig`` keyword arguments.

    Every key is checked against its domain in both hostings, so a typo fails synth either way.
    The combination rules apply to ``agentcore`` only: ``reuseConversation=false`` is refused, and ``public`` warns about what it gives up
    and needs ``agentcoreNetworkModeAck=public-at-prod`` at ``stage=prod``. On ``fargate`` an
    explicit ``agentcoreNetworkMode`` is ignored with a warning. ``bridgeImageDigest`` is not required
    here: the image repository deploys before any image exists (push-then-deploy), so the stack
    that builds the Runtime calls :meth:`InfraConfig.require_bridge_image_digest` itself.
    """
    hosting = _choice(ctx, "bridgeHosting", BRIDGE_HOSTINGS, DEFAULT_BRIDGE_HOSTING)
    mode = _choice(ctx, "agentcoreNetworkMode", AGENTCORE_NETWORK_MODES, DEFAULT_AGENTCORE_NETWORK_MODE)
    # Unset (or blank) means no acknowledgement; any value but the exact word fails.
    acked = bool(_str(ctx, "agentcoreNetworkModeAck"))
    if acked:
        _choice(ctx, "agentcoreNetworkModeAck", (AGENTCORE_PUBLIC_AT_PROD_ACK,), AGENTCORE_PUBLIC_AT_PROD_ACK)
    idle = _int(ctx, "agentcoreIdleTimeoutSeconds", DEFAULT_AGENTCORE_IDLE_TIMEOUT_S,
                AGENTCORE_IDLE_TIMEOUT_MIN_S, AGENTCORE_MAX_LIFETIME_S)
    if idle >= AGENTCORE_MAX_LIFETIME_S:
        raise ConfigError(f"context 'agentcoreIdleTimeoutSeconds' must be below the Runtime's maximum "
                          f"lifetime ({AGENTCORE_MAX_LIFETIME_S} s), got {idle}")
    quota = _int(ctx, "agentcoreSessionQuota",
                 AGENTCORE_SESSION_QUOTA_DEFAULTS.get(region, AGENTCORE_SESSION_QUOTA_OTHER_REGIONS),
                 1, AGENTCORE_SESSION_QUOTA_MAX)

    if hosting == "fargate":
        if _str(ctx, "agentcoreNetworkMode"):
            warnings.warn("agentcoreNetworkMode is ignored with bridgeHosting=fargate (it applies to "
                          "bridgeHosting=agentcore only)", UserWarning, stacklevel=3)
    else:
        if not reuse_conversation:
            raise ConfigError("bridgeHosting=agentcore needs reuseConversation=true (one conversation "
                              "per call): drop reuseConversation=false or use "
                              "bridgeHosting=fargate")
        if mode == "public":
            if stage == "prod" and not acked:
                raise ConfigError("agentcoreNetworkMode=public at stage=prod needs "
                                  f"agentcoreNetworkModeAck={AGENTCORE_PUBLIC_AT_PROD_ACK} (the mode gives "
                                  f"up {AGENTCORE_PUBLIC_TRADE_OFFS}; see the production guide, "
                                  "\"Opting in to public networking for the Runtime\")")
            warnings.warn(f"agentcoreNetworkMode=public gives up {AGENTCORE_PUBLIC_TRADE_OFFS}: the "
                          "Runtime runs outside the VPC and OwlBankNetwork is not created. See the "
                          "production guide, \"Opting in to public networking for the Runtime\"",
                          UserWarning, stacklevel=3)
    return {"bridge_hosting": hosting, "agentcore_network_mode": mode,
            "agentcore_idle_timeout_s": idle, "agentcore_session_quota": quota}


def _call_observability(ctx: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    """The call-observability keys, as ``InfraConfig``
    keyword arguments.

    Every key is validated whether ``callObservability`` is ``on`` or ``off``: a typo fails synth
    now rather than on the day observability is turned on, and the ``off`` warning below can then
    name only well-formed keys. For the same reason ``traceContent=full`` is refused at
    ``stage=prod`` in both modes. With ``off``, any of the other keys given (non-blank) gives
    one warning naming them; with ``on``, ``traceContent`` other than ``off`` warns that the agent's
    text is exported.
    """
    mode = _choice(ctx, "callObservability", CALL_OBSERVABILITY_MODES, DEFAULT_CALL_OBSERVABILITY)
    content = _choice(ctx, "traceContent", TRACE_CONTENT_MODES, DEFAULT_TRACE_CONTENT)
    if content == "full" and stage == "prod":
        raise ConfigError("traceContent=full is not allowed at stage=prod (it exports the agent's "
                          "unredacted text); use traceContent=redacted or off")
    values = {
        "call_observability": mode,
        "trace_sample_rate": _float(ctx, "traceSampleRate", DEFAULT_TRACE_SAMPLE_RATE, 0.0, 1.0),
        "trace_content": content,
        "trace_content_retention_days": _retention_days(ctx),
        "traces_enable_transaction_search": _bool(ctx, "tracesEnableTransactionSearch", False),
        "ttft_ceiling_seconds": _float(ctx, "ttftCeilingSeconds", DEFAULT_TTFT_CEILING_S,
                                       CEILING_SECONDS_MIN, CEILING_SECONDS_MAX),
        "turn_wait_ceiling_seconds": _float(ctx, "turnWaitCeilingSeconds", DEFAULT_TURN_WAIT_CEILING_S,
                                            CEILING_SECONDS_MIN, CEILING_SECONDS_MAX),
        "silence_ceiling_seconds": _float(ctx, "silenceCeilingSeconds", DEFAULT_SILENCE_CEILING_S,
                                          CEILING_SECONDS_MIN, CEILING_SECONDS_MAX),
        "tokens_per_call_ceiling": _int(ctx, "tokensPerCallCeiling", DEFAULT_TOKENS_PER_CALL_CEILING,
                                        1, TOKENS_PER_CALL_CEILING_MAX),
    }
    if mode == "off":
        given = [k for k in CALL_OBSERVABILITY_KEYS if ctx.get(k) not in (None, "")]
        if given:
            warnings.warn(f"{', '.join(given)} {'has' if len(given) == 1 else 'have'} no effect with "
                          "callObservability=off (set callObservability=on to use them)",
                          UserWarning, stacklevel=3)
    elif content != "off":
        how = ("after redaction" if content == "redacted" else "unredacted")
        warnings.warn(f"traceContent={content}: the agent's text (system prompt, messages, tool "
                      f"inputs and outputs) is exported {how} to the traces log group",
                      UserWarning, stacklevel=3)
    return values


def _availability_zones(ctx: Mapping[str, Any], region: str) -> tuple[str, ...] | None:
    """``availabilityZones``: unset, or exactly two distinct AZ names of ``region`` (``<region><letter>``),
    as a list, a JSON list string or a comma-separated string."""
    raw = ctx.get("availabilityZones")
    if raw is None or raw == "":
        return None
    error = ConfigError(f"availabilityZones must be {VPC_AZ_COUNT} distinct AZ names in {region} "
                        f"(such as {region}a,{region}b), got {raw!r}")
    if isinstance(raw, str):
        if raw.strip().startswith("["):
            try:
                raw = json.loads(raw)
            except ValueError:
                raise error from None
        else:
            raw = raw.split(",")
    if not isinstance(raw, (list, tuple)) or not all(isinstance(v, str) for v in raw):
        raise error
    names = tuple(v.strip() for v in raw)
    pattern = re.compile(re.escape(region) + r"[a-z]")
    if (len(names) != VPC_AZ_COUNT or len(set(names)) != VPC_AZ_COUNT
            or not all(pattern.fullmatch(n) for n in names)):
        raise error
    return names


def _str(ctx: Mapping[str, Any], key: str) -> str | None:
    value = ctx.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ConfigError(f"context {key!r} must be a string")
    return value.strip()


def _speech_timeout(ctx: Mapping[str, Any], key: str = "speechTimeout", default: int = 3) -> str:
    """Context ``speechTimeout`` or ``confirmSpeechTimeout``: "auto" or an
    integer 1-10 (seconds); unset means ``default``."""
    raw = ctx.get(key)
    if raw is None or raw == "":
        return str(default)
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        return "auto"
    return str(_int(ctx, key, default, 1, 10))


def _int(ctx: Mapping[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    raw = ctx.get(key)
    if raw is None or raw == "":
        return default
    try:
        if isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer()):
            raise ValueError(raw)  # int() would silently truncate 2.5 and accept True
        value = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(f"context {key!r} must be an integer") from None
    if not lo <= value <= hi:
        raise ConfigError(f"context {key!r} must be between {lo} and {hi}, got {value}")
    return value


def _retention_days(ctx: Mapping[str, Any]) -> int:
    """``traceContentRetentionDays``: an integer CloudWatch Logs accepts (1, 3, 5, 7, 14 or 30)."""
    days = _int(ctx, "traceContentRetentionDays", DEFAULT_TRACE_CONTENT_RETENTION_DAYS,
                TRACE_CONTENT_RETENTION_MIN_DAYS, TRACE_CONTENT_RETENTION_MAX_DAYS)
    if days not in TRACE_CONTENT_RETENTION_VALUES:
        raise ConfigError("traceContentRetentionDays must be one of "
                          f"{', '.join(map(str, TRACE_CONTENT_RETENTION_VALUES))}, got {days}")
    return days


def _float(ctx: Mapping[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    """A finite number from ``lo`` to ``hi`` (inclusive); unset or blank means ``default``. A number
    or a numeric string (``-c key=0.5`` gives a string); booleans, NaN and infinities are refused."""
    raw = ctx.get(key)
    if raw is None or raw == "":
        return default
    try:
        if isinstance(raw, bool):
            raise ValueError(raw)  # float(True) would be 1.0
        value = float(raw.strip() if isinstance(raw, str) else raw)
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(f"context {key!r} must be a number") from None
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ConfigError(f"context {key!r} must be between {lo} and {hi}, got {raw!r}")
    return value


def _choice(ctx: Mapping[str, Any], key: str, allowed: tuple[str, ...], default: str) -> str:
    value = (_str(ctx, key) or default).lower()
    if value not in allowed:
        raise ConfigError(f"{key} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _timezone(ctx: Mapping[str, Any]) -> str:
    """``timezone``: an IANA zone name (validated with zoneinfo at synth)."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    name = _str(ctx, "timezone") or "America/New_York"
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigError(f"timezone must be an IANA zone name such as America/New_York, got {name!r}") from None
    return name


#: ``text_release`` values.
AGENT_TEXT_RELEASE_MODES = ("stream", "message")
#: Agent-wide context keys removed in favour of ``modelParams``; passing
#: one fails synth.
REMOVED_AGENT_KEYS = ("agentThinking", "agentTextRelease", "agentMaxTokens")


def _thinking_error(thinking: str) -> str | None:
    """Why ``thinking`` is not an entry thinking setting, or ``None``."""
    if thinking.startswith(ENABLED_THINKING_PREFIX):
        budget = thinking[len(ENABLED_THINKING_PREFIX):]
        if not re.fullmatch(r"[0-9]{1,6}", budget) or int(budget) < MIN_THINKING_BUDGET:
            return (f"thinking {thinking!r}: enabled:<budget> needs a whole-number budget of at least "
                    f"{MIN_THINKING_BUDGET}")
        return None
    if thinking not in ENTRY_THINKING_MODES:
        return (f"thinking {thinking!r} must be one of {', '.join(ENTRY_THINKING_MODES)} or "
                f"{ENABLED_THINKING_PREFIX}<budget>")
    return None


def _model_entry(model_id: str, params: Mapping[str, Any], prompt_caching: bool) -> ModelEntry:
    """One entry: ``modelParams`` over ``promptCaching`` and the model's known defaults.
    Raises :class:`ConfigError` naming the entry when the parameter set is invalid for it."""
    base = model_id.split(".", 1)[1]
    known_thinking, known_release = _known(KNOWN_MODEL_DEFAULTS, base) or OTHER_MODEL_DEFAULT
    thinking = params.get("thinking", known_thinking)
    text_release = params.get("text_release", known_release)
    max_tokens = params.get("max_tokens", 1000 if thinking == "off" else 2000)
    prompt_caching = params.get("prompt_caching", prompt_caching)
    where = f"modelIds entry {model_id!r}"
    problem = _thinking_error(thinking)
    if problem is None and thinking.split(":", 1)[0] in (_known(KNOWN_THINKING_REFUSALS, base) or ()):
        problem = (f"thinking {thinking!r} is refused by this model (Bedrock rejects it); set another "
                   "thinking for it in modelParams")
    if problem is None and text_release not in AGENT_TEXT_RELEASE_MODES:
        problem = f"text_release {text_release!r} must be one of {', '.join(AGENT_TEXT_RELEASE_MODES)}"
    if problem is None and text_release == "stream" and thinking == "off":
        problem = ("text_release=stream needs reasoning on: with reasoning off the model narrates "
                   "before tool calls; set text_release=message in modelParams for it")
    if problem is None and thinking.startswith(ENABLED_THINKING_PREFIX) \
            and int(thinking[len(ENABLED_THINKING_PREFIX):]) >= max_tokens:
        problem = f"the thinking budget of {thinking!r} must be below max_tokens ({max_tokens})"
    if problem is not None:
        raise ConfigError(f"{where}: {problem} (set it per entry in "
                          "modelParams)")
    return ModelEntry(id=model_id, thinking=thinking, max_tokens=max_tokens,
                      text_release=text_release, prompt_caching=prompt_caching)


def _json_value(ctx: Mapping[str, Any], key: str) -> Any:
    """``ctx[key]`` with a JSON string decoded (``-c key='[...]'`` gives a string)."""
    raw = ctx.get(key)
    if isinstance(raw, str) and raw.strip()[:1] in ("[", "{"):
        try:
            return json.loads(raw)
        except ValueError:
            raise ConfigError(f"context {key!r} is not valid JSON") from None
    return raw


def _model_ids(ctx: Mapping[str, Any], region: str) -> tuple[str, ...]:
    """``modelIds``: one to three distinct inference-profile IDs, each of the Region's geography or
    ``global.``; unset means :func:`default_model_ids`. A JSON list, a list, or a
    comma-separated string."""
    raw = _json_value(ctx, "modelIds")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default_model_ids(region)
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, (list, tuple)) or not all(isinstance(v, str) for v in raw):
        raise ConfigError("context 'modelIds' must be a list of model IDs (a JSON list or a "
                          "comma-separated string)")
    ids = tuple(v.strip() for v in raw)
    if not MODEL_LIST_MIN <= len(ids) <= MODEL_LIST_MAX:
        raise ConfigError(f"context 'modelIds' must list {MODEL_LIST_MIN} to {MODEL_LIST_MAX} models, "
                          f"got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ConfigError("context 'modelIds' must not repeat a model")
    geo = region_geo(region)
    allowed = tuple(p for p in (geo, GLOBAL_PROFILE_PREFIX) if p)
    for mid in ids:
        m = _PROFILE_ID.fullmatch(mid)
        if m is None:
            raise ConfigError(f"modelIds entry {mid!r} is not an inference-profile ID "
                              f"(<{'|'.join(allowed)}>.<provider>.<model>): every entry is called "
                              "through an inference profile")
        if m["prefix"] not in allowed:
            raise ConfigError(f"modelIds entry {mid!r} is a {m['prefix']}. profile, another geography's: "
                              f"from {region} use {' or '.join(p + '.' for p in allowed)}")
    return ids


def _model_params(ctx: Mapping[str, Any], ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """``modelParams``: ``{"<model id>": {"thinking": ..., "max_tokens": ..., "text_release": ...,
    "prompt_caching": ...}}``, each key one of ``modelIds`` and each field optional."""
    raw = _json_value(ctx, "modelParams")
    if raw is None or raw == "":
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError("context 'modelParams' must be a JSON object keyed by model ID")
    out: dict[str, dict[str, Any]] = {}
    for mid, fields in raw.items():
        if mid not in ids:
            raise ConfigError(f"modelParams key {mid!r} is not in modelIds ({', '.join(ids)})")
        if not isinstance(fields, Mapping):
            raise ConfigError(f"modelParams[{mid!r}] must be an object of {', '.join(MODEL_PARAM_KEYS)}")
        unknown = sorted(set(fields) - set(MODEL_PARAM_KEYS))
        if unknown:
            raise ConfigError(f"modelParams[{mid!r}]: unknown field(s) {', '.join(map(repr, unknown))}; "
                              f"allowed: {', '.join(MODEL_PARAM_KEYS)}")
        params: dict[str, Any] = {}
        for name in ("thinking", "text_release"):
            if name in fields:
                if not isinstance(fields[name], str):
                    raise ConfigError(f"modelParams[{mid!r}].{name} must be a string")
                params[name] = fields[name].strip().lower()
        if "max_tokens" in fields:
            value, (lo, hi) = fields["max_tokens"], MAX_TOKENS_RANGE
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ConfigError(f"modelParams[{mid!r}].max_tokens must be an integer from {lo} to {hi}")
            params["max_tokens"] = value
        if "prompt_caching" in fields:
            value = fields["prompt_caching"]
            if isinstance(value, str) and value.strip().lower() in PROMPT_CACHING_MODES:
                value = value.strip().lower() == "on"
            if not isinstance(value, bool):
                raise ConfigError(f"modelParams[{mid!r}].prompt_caching must be on, off, true or false")
            params["prompt_caching"] = value
        out[mid] = params
    return out


def _models(ctx: Mapping[str, Any], *, region: str, prompt_caching: bool) -> tuple[ModelEntry, ...]:
    """The agent's model list. ``modelId`` (one model) was replaced by ``modelIds``
    and is refused rather than ignored. A ``global.`` entry warns: it may be processed in any
    commercial Region."""
    if ctx.get("modelId") not in (None, ""):
        raise ConfigError("modelId was replaced by modelIds (a list of one to three models): pass "
                          "-c modelIds='[\"<model id>\"]' for a single model")
    ids = _model_ids(ctx, region)
    params = _model_params(ctx, ids)
    entries = tuple(_model_entry(mid, params.get(mid, {}), prompt_caching) for mid in ids)
    for entry in entries:
        if entry.is_global:
            warnings.warn(f"modelIds entry {entry.id!r} is a global. profile: its requests may be "
                          "processed in any commercial Region, not only this geography",
                          UserWarning, stacklevel=3)
    return entries


def _failover_test(ctx: Mapping[str, Any], *, stage: str) -> float | None:
    """``failoverTest``: the test-only delay, in seconds, of the preferred (first)
    entry's first token; above 0 and at most :data:`FAILOVER_TEST_MAX_S`. Unset, blank or ``off``
    is off (``None``). Refused at ``stage=prod``."""
    raw = ctx.get("failoverTest")
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "off")):
        return None
    try:
        if isinstance(raw, bool):
            raise ValueError(raw)
        value = float(raw.strip() if isinstance(raw, str) else raw)
    except (TypeError, ValueError, OverflowError):
        raise ConfigError(f"failoverTest must be off or a delay in seconds (above 0, at most "
                          f"{FAILOVER_TEST_MAX_S:g}), got {raw!r}") from None
    if not math.isfinite(value) or not 0 < value <= FAILOVER_TEST_MAX_S:
        raise ConfigError(f"failoverTest must be off or a delay in seconds (above 0, at most "
                          f"{FAILOVER_TEST_MAX_S:g}), got {raw!r}")
    if stage == "prod":
        raise ConfigError("failoverTest is not allowed at stage=prod (it delays the agent's model "
                          "calls on purpose); use it in dev or test only")
    return value


def _memory_trait_groups(ctx: Mapping[str, Any]) -> tuple[str, ...]:
    """``memoryTraitGroups``: comma-separated Memory trait group names (1 to 10, no duplicates)."""
    raw = ctx.get("memoryTraitGroups")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return DEFAULT_MEMORY_TRAIT_GROUPS
    if not isinstance(raw, str):
        raise ConfigError("context 'memoryTraitGroups' must be a comma-separated list of group names")
    names = tuple(part.strip() for part in raw.split(","))
    if not 1 <= len(names) <= MAX_TRAIT_GROUPS:
        raise ConfigError(f"context 'memoryTraitGroups' must list 1 to {MAX_TRAIT_GROUPS} group names, "
                          f"got {len(names)}")
    for name in names:
        if not TRAIT_GROUP_NAME_RE.fullmatch(name):
            raise ConfigError(f"context 'memoryTraitGroups': each name must match {TRAIT_GROUP_NAME_RE.pattern}")
    if len(set(names)) != len(names):
        raise ConfigError("context 'memoryTraitGroups' must not repeat a group name")
    return names


def _bool(ctx: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = ctx.get(key)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
        return raw.strip().lower() == "true"
    raise ConfigError(f"context {key!r} must be true or false, got {raw!r}")


def _admin_console(ctx: Mapping[str, Any], *, stage: str, region: str) -> bool:
    """``adminConsole``: default on for ``dev`` and ``test``, off for ``prod``.

    The console's CloudFront distribution needs a CLOUDFRONT-scope web ACL, which only exists in
    us-east-1, and the stack also holds the ACL. So outside us-east-1 the default is off, and an
    explicit ``adminConsole=true`` there is an error rather than a stack that can't deploy.
    """
    explicit = ctx.get("adminConsole") not in (None, "")
    enabled = _bool(ctx, "adminConsole", stage in ("dev", "test") and region == "us-east-1")
    if enabled and explicit and region != "us-east-1":
        raise ConfigError(f"adminConsole needs region us-east-1 (its CloudFront web ACL lives there); "
                          f"got {region!r}. Set adminConsole=false for this Region")
    return enabled


def _admin_email(ctx: Mapping[str, Any]) -> str | None:
    """``adminEmail``: optional; must look like one email address. The value is never put
    in an error message, so a typo doesn't land in a CI log."""
    raw = ctx.get("adminEmail")
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str) or not _EMAIL.match(raw.strip()) or len(raw.strip()) > 254:
        raise ConfigError("adminEmail must be a single email address (value not shown)")
    return raw.strip()


def _list(ctx: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = ctx.get(key)
    if raw is None or raw == "":
        return ()
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip().startswith("[") else raw.split(",")
    if not isinstance(raw, (list, tuple)):
        raise ConfigError(f"context {key!r} must be a list or comma-separated string")
    return tuple(str(v).strip() for v in raw if str(v).strip())


def load_twilio_outputs(path: str | None, base_dir: Path) -> dict[str, str]:
    """Read the twilio_setup output JSON: identifiers only (SIDs, URLs), never secrets."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    if not p.is_file():
        raise ConfigError(f"twilioOutputs file not found: {p}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ConfigError("twilioOutputs must be a JSON object")
    out: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            continue  # nested plan details are for humans, not the stacks
        if _SECRETISH.search(key) and not _SECRETISH_ALLOW.search(key):
            raise ConfigError(
                f"twilioOutputs key {key!r} looks like a secret; secrets go in Secrets Manager only")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", key):
            raise ConfigError(f"twilioOutputs key {key!r} is not a plain identifier")
        out[key] = str(value)
    return out


def _basic_auth_user(ctx: Mapping[str, Any]) -> str:
    user = _str(ctx, "basicAuthUser") or DEFAULT_BASIC_AUTH_USER
    if not BASIC_AUTH_USER_RE.fullmatch(user):
        raise ConfigError(f"basicAuthUser must match {BASIC_AUTH_USER_RE.pattern} "
                          "(1 to 64 letters, digits, '.', '_' or '-')")
    return user


def from_context(ctx: Mapping[str, Any], *, base_dir: Path) -> InfraConfig:
    removed = [k for k in REMOVED_AGENT_KEYS if ctx.get(k) not in (None, "")]
    if removed:
        raise ConfigError(f"{', '.join(removed)} {'was' if len(removed) == 1 else 'were'} removed: set thinking, "
                          "max_tokens and text_release per model in modelParams, for example "
                          "-c 'modelParams={\"<model id>\":{\"thinking\":\"off\",\"text_release\":\"message\"}}'")
    account = _str(ctx, "accountId") or FAKE_ACCOUNT
    if not _ACCOUNT.match(account):
        raise ConfigError("accountId must be 12 digits")
    region = _str(ctx, "region") or "us-east-1"
    if not _REGION.match(region):
        raise ConfigError(f"bad region {region!r}")
    stage = _str(ctx, "stage") or "dev"
    if stage not in ("dev", "test", "prod"):
        raise ConfigError("stage must be dev, test or prod")

    domain = _str(ctx, "domainName")
    if domain is not None:
        domain = domain.lower().rstrip(".")
        if not _DOMAIN.match(domain):
            raise ConfigError(f"bad domainName {domain!r}")
    cert = _str(ctx, "certificateArn")
    if cert is not None and not re.match(r"^arn:aws[a-z-]*:acm:[a-z0-9-]+:[0-9]{12}:certificate/.+$", cert):
        raise ConfigError("certificateArn must be an ACM certificate ARN")
    if domain and not cert:
        raise ConfigError("domainName needs certificateArn (HTTPS only)")
    if not domain and region != "us-east-1":
        # CloudFront mode: the CLOUDFRONT-scope web ACL (and its log group and metrics) must be
        # in us-east-1, and the Bridge stack holds it. Set domainName for another Region.
        raise ConfigError(f"no domainName selects the CloudFront mode, which needs region us-east-1 "
                          f"(CloudFront WAF lives there); got {region!r}. Use us-east-1, or set "
                          "domainName and certificateArn for the custom-domain mode")

    dev_phone = _str(ctx, "developerPhone")
    tests = _list(ctx, "testNumbers")
    for num in ((dev_phone,) if dev_phone else ()) + tests:
        if not _E164.match(num):
            raise ConfigError(f"phone numbers must be E.164, got {num!r}")
    if stage == "prod" and (dev_phone or tests):
        # Owned real numbers are seeded only in dev and test stacks.
        raise ConfigError("developerPhone/testNumbers are for dev and test stages only")

    if stage == "prod" and not domain:
        # Waiver: the unencrypted CloudFront-to-ALB and ALB-to-task hops are accepted for the
        # sandbox only. Anything at stage prod needs the custom-domain mode.
        raise ConfigError(
            "stage=prod needs the custom-domain mode: set domainName and certificateArn. The "
            "no-domain CloudFront mode sends traffic over plain HTTP from CloudFront to the ALB "
            "(port 80) and from the ALB to the task (port 8080); that is a sandbox-only waiver "
            "See docs/production-guide.md")

    twilio_outputs = load_twilio_outputs(_str(ctx, "twilioOutputs"), base_dir)
    verify_sid = _str(ctx, "verifyServiceSid") or twilio_outputs.get("verifyServiceSid")
    if verify_sid is not None and not re.fullmatch(r"VA[0-9a-f]{32}", verify_sid):
        raise ConfigError("verifyServiceSid must be a Verify service SID (VA + 32 hex)")
    prefix_list = _str(ctx, "cloudFrontOriginPrefixListId") or InfraConfig.cloudfront_prefix_list_id
    if not re.fullmatch(r"pl-[0-9a-f]{8,17}", prefix_list):
        raise ConfigError("cloudFrontOriginPrefixListId must be a managed prefix list ID (pl-...)")
    twilio_number = _str(ctx, "twilioPhoneNumber")
    if twilio_number is not None and not _E164.match(twilio_number):
        raise ConfigError("twilioPhoneNumber must be E.164")

    waf_mode = _str(ctx, "wafManagedRuleMode")
    if waf_mode is not None:
        waf_mode = waf_mode.lower()
        if waf_mode not in ("count", "block"):
            raise ConfigError("wafManagedRuleMode must be 'count' or 'block'")
    secret_version = _str(ctx, "originVerifySecretVersion")
    if secret_version is not None and not re.fullmatch(r"[A-Za-z0-9-]{32,64}", secret_version):
        raise ConfigError("originVerifySecretVersion must be a Secrets Manager version ID (32 to 64 characters)")
    admin_secret_version = _str(ctx, "adminOriginVerifySecretVersion")
    if admin_secret_version is not None and not re.fullmatch(r"[A-Za-z0-9-]{32,64}", admin_secret_version):
        raise ConfigError("adminOriginVerifySecretVersion must be a Secrets Manager version ID "
                          "(32 to 64 characters)")

    bpa_mode = ctx.get("s3PublicAccessBlockMode")
    if bpa_mode is None or bpa_mode == "":
        bpa_mode = DEFAULT_S3_PUBLIC_ACCESS_BLOCK_MODE
    if not isinstance(bpa_mode, str) or bpa_mode not in S3_PUBLIC_ACCESS_BLOCK_MODES:
        # Exact, case-sensitive match, and no silent fallback: a typo here would otherwise decide
        # whether a bucket is protected.
        raise ConfigError(f"s3PublicAccessBlockMode must be one of {'|'.join(S3_PUBLIC_ACCESS_BLOCK_MODES)}, "
                          f"got {bpa_mode!r}")

    alarm_email = _str(ctx, "alarmEmail")
    if alarm_email is not None and not _EMAIL.match(alarm_email):
        raise ConfigError("alarmEmail must be an email address")
    tags = {}
    for key, default in (("owner", "unassigned"), ("costCenter", "unassigned"),
                         ("dataClassification", "confidential")):
        value = _str(ctx, key) or default
        if not _TAG_VALUE.match(value):
            raise ConfigError(f"context {key!r} is not a valid tag value")
        tags[key] = value

    reuse_conversation = _bool(ctx, "reuseConversation", True)
    image_digest = _image_digest(ctx)
    hosting = _bridge_hosting(ctx, stage=stage, region=region, reuse_conversation=reuse_conversation,
                              image_digest=image_digest)
    observability = _call_observability(ctx, stage=stage)
    prompt_caching = _choice(ctx, "promptCaching", PROMPT_CACHING_MODES, DEFAULT_PROMPT_CACHING)
    prompt_cache_ttl = _choice(ctx, "promptCacheTtl", PROMPT_CACHE_TTLS, DEFAULT_PROMPT_CACHE_TTL)
    models = _models(ctx, region=region, prompt_caching=prompt_caching == "on")
    model_failover = _choice(ctx, "modelFailover", MODEL_FAILOVER_MODES, DEFAULT_MODEL_FAILOVER)
    if model_failover == "off":
        models = models[:1]  # Only the first entry reaches IAM and OWLBANK_MODELS
    if prompt_cache_ttl == "1h" and not any(m.prompt_caching for m in models):
        warnings.warn("promptCacheTtl=1h has no effect: prompt caching is off on every modelIds entry",
                      UserWarning, stacklevel=2)
    failover_test = _failover_test(ctx, stage=stage)
    if failover_test is not None and (model_failover == "off" or len(models) < 2):
        raise ConfigError("failoverTest needs modelFailover=on and at least two modelIds entries "
                          "(it delays the preferred entry to test a switch, and there is nothing to "
                          "switch to)")
    cfg = InfraConfig(
        account=account,
        region=region,
        stage=stage,
        domain_name=domain,
        certificate_arn=cert,
        developer_phone=dev_phone,
        test_numbers=tests,
        bridge_desired_count=_int(ctx, "bridgeDesiredCount", 1, 0, 20),
        models=models,
        failover_test=failover_test,
        model_failover=model_failover,
        resolver_model_id=_str(ctx, "resolverModelId") or InfraConfig.resolver_model_id,
        alb_idle_timeout_s=_int(ctx, "albIdleTimeoutSeconds", 900, 1, 4000),
        deregistration_delay_s=_int(ctx, "deregistrationDelaySeconds", 300, 0, 3600),
        nat_gateways=_int(ctx, "natGateways", 1, 1, 2),
        flex_origin=_str(ctx, "flexOrigin") or InfraConfig.flex_origin,
        twilio_phone_number=twilio_number,
        verify_service_sid=verify_sid,
        twilio_outputs=twilio_outputs,
        alarm_email=alarm_email,
        waf_api_rate_limit=_int(ctx, "wafApiRateLimit", InfraConfig.waf_api_rate_limit, 100, 2_000_000_000),
        waf_alb_rate_limit=_int(ctx, "wafAlbRateLimit", 1000, 100, 2_000_000_000),
        waf_cloudfront_rate_limit=_int(ctx, "wafCloudFrontRateLimit", 10000, 100, 2_000_000_000),
        cloudfront_origin_read_timeout_s=_int(ctx, "cloudFrontOriginReadTimeoutSeconds", 60, 1, 120),
        cloudfront_origin_keepalive_timeout_s=_int(ctx, "cloudFrontOriginKeepaliveTimeoutSeconds", 60, 1, 300),
        cloudfront_prefix_list_id=prefix_list,
        waf_managed_rule_mode=waf_mode,
        origin_verify_secret_version=secret_version,
        tag_owner=tags["owner"],
        tag_cost_center=tags["costCenter"],
        tag_data_classification=tags["dataClassification"],
        log_level=normalize_log_level(ctx.get("logLevel")),
        s3_public_access_block_mode=bpa_mode,
        lambda_memory_mb=_int(ctx, "lambdaMemoryMb", DEFAULT_LAMBDA_MEMORY_MB,
                              LAMBDA_MEMORY_MIN_MB, LAMBDA_MEMORY_MAX_MB),
        lambda_provisioned_concurrency=_int(ctx, "lambdaProvisionedConcurrency",
                                            DEFAULT_LAMBDA_PROVISIONED_CONCURRENCY, 0,
                                            LAMBDA_PROVISIONED_CONCURRENCY_MAX),
        memory_trait_groups=_memory_trait_groups(ctx),
        web_activity_max_age_hours=_int(ctx, "webActivityMaxAgeHours", DEFAULT_WEB_ACTIVITY_MAX_AGE_HOURS,
                                         *WEB_ACTIVITY_MAX_AGE_HOURS_RANGE),
        reuse_conversation=reuse_conversation,
        early_digits=_bool(ctx, "earlyDigits", True),
        ai_keypad_shortcuts=_choice(ctx, "aiKeypadShortcuts", ("on", "off"), "on"),
        timezone=_timezone(ctx),
        scripted_barge_in=_choice(ctx, "scriptedBargeIn", ("any", "speech", "dtmf", "none"), "any"),
        speech_timeout=_speech_timeout(ctx),
        confirm_speech_timeout=_speech_timeout(ctx, "confirmSpeechTimeout", 2),
        keypad_timeout=str(_int(ctx, "keypadTimeout", 3, 1, 10)),
        prompt_caching=prompt_caching,
        prompt_cache_ttl=prompt_cache_ttl,
        accept_spoken_factors_override=(_bool(ctx, "acceptSpokenFactors", False)
                                        if ctx.get("acceptSpokenFactors") not in (None, "") else None),
        admin_console=_admin_console(ctx, stage=stage, region=region),
        admin_email=_admin_email(ctx),
        admin_origin_verify_secret_version=admin_secret_version,
        bridge_image_digest=image_digest,
        agentcore_single_az_ack=_bool(ctx, "agentcoreSingleAzAck", False),
        availability_zones=_availability_zones(ctx, region),
        slot_mode_ceiling=_choice(ctx, "slotModeCeiling", SLOT_MODES, DEFAULT_SLOT_MODE_CEILING),
        speech_log=_choice(ctx, "speechLog", SPEECH_LOG_MODES, DEFAULT_SPEECH_LOG),
        speech_log_redaction=_choice(ctx, "speechLogRedaction", SPEECH_LOG_MODES,
                                     DEFAULT_SPEECH_LOG_REDACTION),
        basic_auth_user=_basic_auth_user(ctx),
        human_leg_conversation=_choice(ctx, "humanLegConversation", HUMAN_LEG_MODES,
                                       DEFAULT_HUMAN_LEG_CONVERSATION),
        context_api_restricted_key=_choice(ctx, "contextApiRestrictedKey", HUMAN_LEG_MODES,
                                           DEFAULT_CONTEXT_API_RESTRICTED_KEY),
        **hosting,
        **observability,
    )
    _ = cfg.flex_mode  # a partial Flex setup fails at synth, not at first use
    return cfg
