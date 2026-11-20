"""Twilio-side setup for the blueprint. Not run in v1.

Usage::

    # dry run (the default): print the plan; no Twilio call, no secret read
    uv run python scripts/twilio_setup.py --phone-number +1... \\
        --api-domain api.owlbank.example --relay-domain relay.owlbank.example

    # CloudFront mode (no domain): hosts and stage come from the CDK outputs
    # (ApiDomain = <id>.execute-api.<region>.amazonaws.com, ApiStage = v1, RelayDomain = d....cloudfront.net)
    uv run python scripts/twilio_setup.py --phone-number +1... --cdk-outputs cdk-outputs.json

    # apply: API key and Basic credentials from Secrets Manager (ARNs from the CDK outputs or
    # --secret NAME=ARN), Account SID as an argument
    uv run python scripts/twilio_setup.py --apply --account-sid AC... --phone-number +1... \\
        --cdk-outputs cdk-outputs.json --output infra/twilio.context.json

**Secrets.** ``--apply`` reads the deployed secrets with ``get_secret_value``:
``twilio_api_key`` (every phase) and ``basic_auth`` (phases ``post`` and ``all``; ``pre`` puts no
webhook URL anywhere). Each ARN comes from ``--secret NAME=ARN`` or the Data stack's ``Secret*``
output in ``--cdk-outputs``; ``--region`` defaults to the ARN's Region. A secret that is still the
placeholder or malformed stops the run naming the secret and the field, never the value. A dry run
reads no secret.

What the plan does, in order (every step finds before it creates, so a re-run is a no-op):

1. **Account check:** the account must be active and dedicated to the blueprint:
   a subaccount, or a main account only with ``--dedicated-main-account`` (e.g. a sandbox
   account used only for OwlBank).
2. **Number:** voice URL ``https://<basic creds>@<api-domain>[/<stage>]/voice/incoming`` and
   status callback ``.../voice/status``, both POST. These keep their credentials: Twilio sends
   them only after the API's ``401`` Basic challenge (the front door and its WAF issue it). With API Gateway's default endpoint (the
   CloudFront mode: ``<id>.execute-api.<region>.amazonaws.com``) the stage is part of the path
   (``/v1``, from ``--api-stage`` or the CDK output ``ApiStage``), because the front door
   rebuilds the signed URL from the request path *with* the stage. On a custom domain there is
   no stage in the path.
3. **TaskRouter, only with ``--with-flex``** (in the Flex TaskRouter workspace, which Flex creates; see
   "Which workspace" below). Without ``--with-flex`` the script makes no Flex or TaskRouter call and writes
   no ``workspaceSid``, ``workflowSid`` or ``queueSids``:
   the queues OwlBank Support EN / Support ES / Fraud / Disputes, and the workflow
   "OwlBank AI Escalation" from the handover (queue names from ``data/policy.json``).
4. **Memory Store** ``owlbank-inbound-voice``, and in it the **trait group** ``WebActivity``
   (``owlbank.web_activity.TRAITS``): Memory refuses traits for a group that doesn't exist. The
   group is made last and **fails open**: if it can't be made, setup warns and finishes (the
   web-activity offer stays off until a re-run makes it); nothing a call needs waits on it.
5. **Conversation Configuration** ``owlbank-inbound-voice``: VOICE channel with *no*
   capture rules (ConversationRelay is active ingestion, TAC channel), status timeouts
   (inactive above the longest deterministic segment, and below "closed": Twilio
   rejects the reverse; defaults 15 and 30 minutes), and ``statusCallbacks`` to the
   bridge's ``/webhook`` **without credentials**: that route is
   signature-only and never challenges for Basic, so Twilio would never send them. Twilio measures both from the last
   transcription event, so "closed" is a callback window; ``statusTimeouts: null`` (close when the
   transcription stream ends, "on hangup") is the production aim once re-entry across deterministic
   segments is verified. The front door's ``ConversationCloser`` is a no-op until then.

6. **Verify Service**: found by friendly name (``--verify-friendly-name``,
   default "OwlBank"), else created with a 6-digit code and Verify's default 10-minute
   validity, and ``do_not_share_warning_enabled`` on (Verify's own "don't share this code"
   warning in the text; it works without a custom template). Its SID is written as
   ``verifyServiceSid``, the key ``infra/owlbank_infra/config.py`` reads from the
   ``twilioOutputs`` file. An existing service whose code length isn't 6 or whose warning
   flag is off is updated. ``--verify-template-sid HJ...`` also sets ``DefaultTemplateSid``
   (a Twilio-approved custom template; see ``docs/verify-sms-template.md``); without it the
   service keeps its current template (Verify's default for a new one).

The resulting SIDs are written to ``--output`` (JSON) for CDK context.

**Phases (first-deploy order).** The bridge container calls Twilio at startup and needs real
secrets and the Conversation Configuration ID, but the webhooks need the API and relay domains,
which exist only after the stacks are deployed. So the plan splits (``--phase``, default ``all``):

* ``pre`` (no domain, no phone number, no Basic credentials needed; runs after ``OwlBankData`` is
  deployed and the Twilio secrets are written, since it reads the API key): steps 1, 3, 4, 5 and 6
  above. The Conversation Configuration is created **without** ``statusCallbacks`` (the API makes
  them optional, and they can be set later on an active configuration, taking effect at once).
  Writes ``--output`` with ``verifyServiceSid`` and ``conversationConfigurationId``, which the
  stacks read as ``twilioOutputs``. Re-running ``pre`` after ``post`` leaves the status callbacks as they are.
* ``post`` (needs the phone number, both domains and the Basic credentials): the account check,
  step 2 (the number's webhooks), the Conversation Configuration's ``statusCallbacks``
  (``https://<relay>/webhook``, without credentials), then a read-back that checks both and that the configuration is the
  one the bridge was deployed with (the ID in ``--output``). Merges into ``--output``.
* ``all``: every step in one run, when the domains are already known (the original behaviour).

Every phase finds before it creates, so each is idempotent; the default is a dry run with no network.

**Flex (opt-in ``--with-flex``; ``--create-flex`` and ``--taskrouter-events`` need it).** Step 3 needs the Flex TaskRouter workspace, which exists only once
Flex is activated on the account. Either activate Flex in the Console, or pass ``--create-flex``
(phases ``pre`` and ``all``): a step ``ensure_flex`` right after the account check does nothing when the
Flex TaskRouter workspace already exists ("Flex already provisioned"); otherwise it creates a Flex
instance through the API (``POST https://flex-api.twilio.com/v2/Instances``, body
``{"conversations": {"default": false}}`` so the account's default conversation service is untouched;
**billable**), polls ``GET .../v2/Instances/<GO...>`` until ``status`` is ``ready`` (every 10 seconds, up to
``--flex-timeout-min`` minutes, default 15), then waits briefly for the workspace to appear. The installed
``twilio`` package has no v2 ``Instances`` resource, so the calls go through ``client.request``
(the client's own HTTP layer and Basic auth with the API key); no new dependency. The instance SID is
written as ``flexInstanceSid`` and, if a run stops early, kept in ``--output`` so the re-run polls that
instance instead of creating another. Errors name the HTTP status and Twilio error code only, never the body.

**Which workspace.** A Flex made in the Console names its TaskRouter workspace exactly "Flex Task Assignment";
one made through the Instances API names it "Flex Task Assignment for GO<instance SID>". One resolver
(``Applier._find_workspace``) serves every step: (a) when the instance SID is known (the output file, or
just created), the workspace named exactly "Flex Task Assignment for <GO...>"; else (b) the one named exactly
"Flex Task Assignment"; else (c) the single workspace whose name starts with "Flex Task Assignment". Several
matches in (c) are an error listing the names and SIDs: it never guesses. The resolved SID is written as
``workspaceSid``.

**Status callback by hosting.** Phases ``post`` and ``all`` read the
``BridgeHosting`` CDK output (missing = ``fargate``). ``fargate``: today's ``https://<relay>/webhook``,
without credentials. ``agentcore``: the front door's Orchestrator sink,
``https://<user>:<password>@<api-domain>[/<stage>]/voice/orchestrator``, with the Basic credentials
(Orchestrator answers the API WAF's Basic challenge); no relay domain is needed (the
``agentcore`` bridge has none). Switching hosting and re-running ``--phase post`` moves the callback.

**TaskRouter events (opt-in ``--taskrouter-events``, phases ``post`` and ``all``; call observability).** Sets the Flex workspace's ``EventCallbackUrl`` to the front door's
``https://<user>:<password>@<api-domain>[/<stage>]/voice/taskrouter`` (the same Basic credential style as
the number's voice URL) and ``EventsFilter`` to ``reservation.accepted,task.canceled,workflow.timeout,task.updated``, so the
front door can record each hand-off's wait in the queue. The callback is **workspace-wide** and Flex or
another integration may already use it, so the step sets it only when it is empty or already points at
this front door (same host and path; the credentials and the filter are brought up to date). Any other
value is reported (host and path only, never credentials) and the run stops without changing it. The
number's status callback (``/voice/status``, step 2) is what produces the ``call.end`` span.

**Restricted API keys (phases ``pre`` and ``all``).** The Context API reads its own Twilio key
instead of the account's Standard key (CDK flag ``contextApiRestrictedKey``, on by default). The Flex
authorizer keeps the Standard key: Twilio has no Restricted permission for ``Tokens/validate``. The step
``ensure_restricted_key`` creates the Restricted key (``POST https://iam.twilio.com/v1/Keys`` with ``KeyType=restricted``
and ``Policy={"allow": [...]}``) and writes ``{"account_sid", "api_key", "api_secret"}`` straight to its
secret (``owlbank/twilio/context-api-key``) after the same
customer-managed-KMS check as ``populate_secrets.py``. Twilio returns the key's secret only in that
answer, so it is never printed, logged or kept. The key may only read a task
(``/twilio/taskrouter/workspaces.tasks/read``) and list its reservations
(``/twilio/taskrouter/workspaces.tasks.reservations/list``).
The Key resource and the Accounts resource refuse Standard API keys, so the key request and the account
check use the account's Auth Token, read from the ``twilio_auth_token`` secret once, before any change
(a missing or placeholder secret stops the run first). A secret that already holds a key for
the account is left as it is; ``--rotate-restricted-keys`` makes a new key and prints the old key's SID
for you to delete once the functions use the new one. ``--skip-context-api-key`` leaves the Context
API's out (needed with ``contextApiRestrictedKey=off``, where its secret doesn't exist). The ARN comes
from the Data stack output ``SecretContextApiKey`` or ``--restricted-key-secret NAME=ARN``.

**The Basic credentials are never printed:** the plan holds the number's URLs (and, in ``agentcore``,
the sink URL) without them, shown as ``https://***:***@host/path``, and they are inserted only when a
request is made. The Fargate bridge status callback has none at all (``https://<relay>/webhook``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote, urlsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from owlbank.obs import scrub, timed_call
from owlbank.secret_schema import validate
from owlbank.web_activity import GROUP as WEB_ACTIVITY_GROUP
from scripts._agent_skills import AGENT_SKILLS
from scripts._common import (
    DATA_DIR,
    E164,
    ScriptError,
    add_mode_args,
    add_secret_args,
    find_output,
    load_cdk_outputs,
    mask_phone,
    parse_pairs,
    parse_secret_overrides,
    read_secret,
    resolve_region,
    script_error,
    script_start,
    script_step,
    secret_arn,
    start_logging,
)
from scripts._memory_traits import ensure_trait_group as ensure_memory_trait_group
from scripts.populate_secrets import check_kms

slog = logging.getLogger("owlbank.twilio_setup")

FLEX_WORKSPACE = "Flex Task Assignment"
#: What the account check accepts: shown in the dry-run plan.
ACCOUNT_REQUIRE = "active account (subaccount, or main account with --dedicated-main-account)"
#: How plan steps and the dry run name the workspace (its real name differs, see the resolver).
FLEX_WORKSPACE_LABEL = "the Flex TaskRouter workspace"
#: Flex Instances API (no resource for it in the installed twilio SDK, so plain requests).
FLEX_INSTANCES_URL = "https://flex-api.twilio.com/v2/Instances"
FLEX_INSTANCE_BODY: dict[str, Any] = {"conversations": {"default": False}}
FLEX_INSTANCE_SID = re.compile(r"GO[0-9a-fA-F]{32}")
DEFAULT_FLEX_TIMEOUT_MIN = 15
MAX_FLEX_TIMEOUT_MIN = 120
FLEX_POLL_S = 10
#: After ``ready`` the TaskRouter workspace can lag: look this many times, FLEX_POLL_S apart.
FLEX_WORKSPACE_TRIES = 6
WORKFLOW_NAME = "OwlBank AI Escalation"
MEMORY_STORE_NAME = "owlbank-inbound-voice"
CONVERSATION_CONFIG_NAME = "owlbank-inbound-voice"   # displayName: max 32 characters
#: Separate conversations per channel; we bind Memory to the *verified* profile ourselves.
GROUPING_TYPE = "GROUP_BY_PARTICIPANT_ADDRESSES_AND_CHANNEL_TYPE"
#: Longest deterministic segment, minutes: an SMS passcode plus retries.
LONGEST_DETERMINISTIC_SEGMENT_MIN = 10
DEFAULT_INACTIVE_MIN = 15
#: Twilio: the voice "inactive" timeout must be less than "closed" (a 400 otherwise). Minutes since the last
#: transcription event. The sandbox keeps a callback window; production aims at ``null`` = on hangup.
DEFAULT_CLOSED_MIN = 30
TWILIO_MESSAGE_MAX = 300

DEFAULT_VERIFY_FRIENDLY_NAME = "OwlBank"
VERIFY_CODE_LENGTH = 6   # the front door's prompts say "six-digit code"
VERIFY_NAME_MAX = 30     # Verify Service friendly name limit
VERIFY_TEMPLATE_SID = re.compile(r"HJ[0-9a-fA-F]{32}")

#: The API Gateway default endpoint's host (no custom domain). Its URLs carry the stage.
EXECUTE_API_HOST = re.compile(r"^[a-z0-9]+\.execute-api\.[a-z0-9-]+\.amazonaws\.com(\.cn)?$")
#: The stage infra/owlbank_infra/api_stack.py deploys (STAGE); overridden by the ApiStage output.
DEFAULT_API_STAGE = "v1"
_STAGE = re.compile(r"[A-Za-z0-9_-]{1,128}")

VOICE_INCOMING_PATH = "/voice/incoming"
VOICE_STATUS_PATH = "/voice/status"
BRIDGE_WEBHOOK_PATH = "/webhook"
#: The front door's Orchestrator sink (``agentcore``; infra ``bridge_stack.ORCHESTRATOR_CALLBACK_PATH``).
ORCHESTRATOR_CALLBACK_PATH = "/voice/orchestrator"
#: The front door's TaskRouter event route.
TASKROUTER_EVENTS_PATH = "/voice/taskrouter"
#: The events the front door turns into ``handoff.queue`` spans, plus ``task.updated``.
TASKROUTER_EVENTS_FILTER = "reservation.accepted,task.canceled,workflow.timeout,task.updated"
HOSTINGS = ("fargate", "agentcore")

#: Twilio's Key resource (IAM v1): ``POST`` creates a key; the ``secret`` is in that answer only.
IAM_KEYS_URL = "https://iam.twilio.com/v1/Keys"
#: The Context API's two TaskRouter reads (``owlbank.taskrouter``): ``GET .../Tasks/<WT>`` and
#: ``GET .../Tasks/<WT>/Reservations``. Names from Twilio's Restricted API key permission list
#: for TaskRouter; the ``Policy`` shape (``{"allow": [...]}``) from the Key resource v1 docs.
CONTEXT_API_PERMISSIONS: tuple[str, ...] = (
    "/twilio/taskrouter/workspaces.tasks/read",
    "/twilio/taskrouter/workspaces.tasks.reservations/list",
)


@dataclass(frozen=True)
class RestrictedKey:
    """A Twilio Restricted API key this script creates and writes straight to its own secret."""

    #: The logical secret (``infra/owlbank_infra/data_stack.OPTIONAL_SECRETS``).
    secret: str
    #: Its Secrets Manager name and the Data stack outputs that hold its ARN.
    secret_name: str
    outputs: tuple[str, ...]
    friendly_name: str
    permissions: tuple[str, ...]
    #: The CDK flag that makes the stacks create the secret and give it to the function.
    infra_flag: str


RESTRICTED_KEYS: dict[str, RestrictedKey] = {
    "context_api_key": RestrictedKey(
        "context_api_key", "owlbank/twilio/context-api-key", ("SecretContextApiKey",),
        "OwlBank Context API (TaskRouter read)", CONTEXT_API_PERMISSIONS, "contextApiRestrictedKey"),
}

#: Queue key -> (policy.json queue key, language, target workers expression). Handover table.
QUEUES: tuple[tuple[str, str, str, str], ...] = (
    ("support-en", "support", "en-US", 'routing.skills HAS "en"'),
    ("support-es", "support", "es-US", 'routing.skills HAS "es"'),
    ("fraud", "fraud", "en-US", 'routing.skills HAS "fraud"'),
    ("disputes", "disputes", "en-US", 'routing.skills HAS "disputes"'),
)


#: Flex Configuration API: ``taskrouter_skills`` is the Add-skill dropdown (the names come from
#: ``scripts/_agent_skills.py``; a test checks them against the queue targets).
FLEX_CONFIGURATION_URL = "https://flex-api.twilio.com/v1/Configuration"
FLEX_CREATE_HINT = (" If Flex can't be created through the API for this account, activate Flex in the "
                    "Console (Flex > Overview) and re-run without --create-flex")

#: Escalation reasons whose policy queue is ``fraud`` (``data/policy.json``); a unit test keeps them in step.
FRAUD_REASONS: tuple[str, ...] = ("fraud_suspected", "verification_risk")
#: Escalation reasons whose policy queue is ``disputes``; kept in step with ``data/policy.json`` by a unit test.
DISPUTES_REASONS: tuple[str, ...] = ("card_dispute",)


# --------------------------------------------------------------------------- credentials


@dataclass(frozen=True)
class BasicCreds:
    user: str = field(repr=False)
    password: str = field(repr=False)

    def __repr__(self) -> str:
        return "BasicCreds(***)"

    __str__ = __repr__


@dataclass(frozen=True)
class CredUrl:
    """An https URL that gets Basic credentials only when rendered for a request."""

    host: str
    path: str

    def __str__(self) -> str:
        return f"https://***:***@{self.host}{self.path}"

    __repr__ = __str__

    def render(self, creds: BasicCreds | None) -> str:
        if creds is None:
            return f"https://{self.host}{self.path}"
        return f"https://{quote(creds.user, safe='')}:{quote(creds.password, safe='')}@{self.host}{self.path}"


# --------------------------------------------------------------------------- plan


PHASES = ("pre", "post", "all")
#: A status-callback URL: plain (the Fargate bridge's ``/webhook``) or with Basic credentials (the sink).
Webhook = str | CredUrl


@dataclass
class SetupConfig:
    #: Needed for phases ``post`` and ``all`` only.
    phone_number: str = ""
    api_domain: str = ""
    relay_domain: str = ""
    account_sid: str | None = None
    queue_names: Mapping[str, str] = field(default_factory=dict)
    inactive_timeout_min: int = DEFAULT_INACTIVE_MIN
    closed_timeout_min: int = DEFAULT_CLOSED_MIN
    dedicated_main_account: bool = False
    verify_friendly_name: str = DEFAULT_VERIFY_FRIENDLY_NAME
    verify_template_sid: str | None = None
    #: Path prefix of the front-door URLs: ``/<stage>`` on the execute-api default endpoint, else "".
    api_base_path: str = ""
    phase: str = "all"
    #: ``post``: the Conversation Configuration ID the earlier ``pre`` run wrote (the one the bridge
    #: was deployed with); a different ID found by name is an error.
    expected_conversation_configuration_id: str | None = None
    #: Opt-in (``--with-flex``): the Flex TaskRouter steps (workspace, queues, workflow, skills, events).
    #: Off by default: no Flex API call and no Flex key in the outputs file.
    with_flex: bool = False
    #: Opt-in (needs ``with_flex``): create the Flex instance through the API when the workspace is missing.
    create_flex: bool = False
    flex_timeout_min: int = DEFAULT_FLEX_TIMEOUT_MIN
    #: The instance SID an earlier run kept in ``--output`` (resume instead of a second POST).
    known_flex_instance_sid: str | None = None
    #: ``BridgeHosting``: where the Conversation Configuration's status callback points.
    bridge_hosting: str = "fargate"
    #: Opt-in: point the Flex workspace's event callback at the front door, if it is free.
    taskrouter_events: bool = False
    #: The Restricted keys to ensure (:data:`RESTRICTED_KEYS`; phases ``pre`` and ``all``).
    restricted_keys: tuple[str, ...] = ()
    #: Create a new key even when the secret already holds one (the old key is left for the operator).
    rotate_restricted_keys: bool = False

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ScriptError(f"--phase must be one of {', '.join(PHASES)}")
        if self.create_flex and not self.with_flex:
            raise ScriptError("--create-flex needs --with-flex (Flex is opt-in; add --with-flex)")
        if self.taskrouter_events and not self.with_flex:
            raise ScriptError("--taskrouter-events needs --with-flex (the events belong to the Flex "
                              "workspace; add --with-flex)")
        if self.create_flex and self.phase == "post":
            raise ScriptError("--create-flex applies to --phase pre or all (Flex must exist before the "
                              "TaskRouter queues)")
        if self.taskrouter_events and self.phase == "pre":
            raise ScriptError("--taskrouter-events applies to --phase post or all (it needs the API domain "
                              "and the Basic credentials)")
        if self.restricted_keys and self.phase == "post":
            raise ScriptError("Restricted keys are made in --phase pre or all (before the stacks that "
                              "read them); pass --skip-context-api-key with --phase post")
        unknown = [k for k in self.restricted_keys if k not in RESTRICTED_KEYS]
        if unknown:
            raise ScriptError(f"unknown Restricted key(s) {unknown}")
        if self.rotate_restricted_keys and not self.restricted_keys:
            raise ScriptError("--rotate-restricted-keys needs a Restricted key to make "
                              "(not --phase post, not with every key skipped)")
        if self.bridge_hosting not in HOSTINGS:
            raise ScriptError(f"bridge hosting must be one of {', '.join(HOSTINGS)}")
        if not 1 <= self.flex_timeout_min <= MAX_FLEX_TIMEOUT_MIN:
            raise ScriptError(f"--flex-timeout-min must be 1 to {MAX_FLEX_TIMEOUT_MIN}")
        if self.phase != "pre":
            if not E164.match(self.phone_number):
                raise ScriptError(f"--phone-number is not E.164 ({mask_phone(self.phone_number)})")
            hosts = ("api_domain",) if self.bridge_hosting == "agentcore" else ("api_domain", "relay_domain")
            for name in hosts:
                v = getattr(self, name)
                if not v or "/" in v or "@" in v or ":" in v:
                    raise ScriptError(f"--{name.replace('_', '-')} must be a bare host name")
        if self.api_base_path and not re.fullmatch(r"/[A-Za-z0-9_-]{1,128}", self.api_base_path):
            raise ScriptError("--api-stage must be a stage name (letters, digits, - and _)")
        if self.account_sid is not None and not (self.account_sid.startswith("AC")
                                                 and len(self.account_sid) == 34):
            raise ScriptError("--account-sid must be an Account SID (AC..., 34 characters)")
        if self.inactive_timeout_min <= LONGEST_DETERMINISTIC_SEGMENT_MIN:
            raise ScriptError(f"inactive timeout must exceed the longest deterministic segment "
                              f"({LONGEST_DETERMINISTIC_SEGMENT_MIN} min)")
        if self.closed_timeout_min < 1:
            raise ScriptError("closed timeout must be at least 1 minute")
        if self.closed_timeout_min <= self.inactive_timeout_min:
            raise ScriptError(f"closed timeout ({self.closed_timeout_min} min) must be greater than the "
                              f"inactive timeout ({self.inactive_timeout_min} min); Twilio rejects it with "
                              "\"inactive timeout must be less than closed timeout\"")
        if not self.verify_friendly_name.strip() or len(self.verify_friendly_name) > VERIFY_NAME_MAX:
            raise ScriptError(f"--verify-friendly-name must be 1 to {VERIFY_NAME_MAX} characters")
        if self.verify_template_sid is not None and not VERIFY_TEMPLATE_SID.fullmatch(
                self.verify_template_sid):
            raise ScriptError("--verify-template-sid must be a Verify template SID (HJ + 32 hex)")


#: A short plain-words label per step kind, printed in the dry-run plan so steps can be told apart.
STEP_LABELS = {
    "check_account": "account check",
    "ensure_flex": "Flex instance",
    "ensure_queue": "TaskRouter queue",
    "ensure_workflow": "TaskRouter workflow",
    "ensure_flex_skills": "Flex agent skills",
    "ensure_memory_store": "Memory store",
    "ensure_conversation_configuration": "Conversation configuration",
    "ensure_verify_service": "Verify service",
    "ensure_restricted_key": "Restricted API key",
    "ensure_trait_group": "web-activity trait group",
    "ensure_taskrouter_events": "TaskRouter event callback",
    "configure_number": "number webhooks",
    "ensure_status_callbacks": "status callbacks",
    "verify_setup": "read-back",
}


@dataclass
class Step:
    kind: str
    name: str
    params: dict[str, Any]

    @property
    def label(self) -> str:
        return STEP_LABELS.get(self.kind, self.kind)

    def describe(self) -> str:
        shown = json.dumps(self.params, default=str, ensure_ascii=False, sort_keys=True)
        return f"{self.kind} [{self.label}]: {self.name} {shown}"


def queue_names_from_policy(policy_doc: Mapping[str, Any]) -> dict[str, str]:
    queues = policy_doc["escalation"]["queues"]
    return {key: queues[pkey][lang] for key, pkey, lang, _ in QUEUES}


def workflow_configuration(queue_sids: Mapping[str, str]) -> dict[str, Any]:
    """The handover's "OwlBank AI Escalation" workflow with real queue SIDs.

    A task matching no filter goes to the English queue (the catch-all ``default_filter``). No target and
    not the catch-all carries a ``timeout``: callers may wait an hour or two, and a timeout on
    a filter's last target cancels the task (it does not fall through to the catch-all)."""
    en, es, fraud = queue_sids["support-en"], queue_sids["support-es"], queue_sids["fraud"]
    disputes = queue_sids["disputes"]
    return {
        "task_routing": {
            "filters": [
                {"filter_friendly_name": "Fraud",
                 "expression": "escalation_reason IN [" + ", ".join(f"'{r}'" for r in FRAUD_REASONS) + "]",
                 "targets": [{"queue": fraud, "priority": 100}]},
                {"filter_friendly_name": "Disputes",
                 "expression": "escalation_reason IN [" + ", ".join(f"'{r}'" for r in DISPUTES_REASONS) + "]",
                 "targets": [{"queue": disputes, "priority": 50}]},
                {"filter_friendly_name": "Vulnerable customer",
                 "expression": "escalation_reason == 'vulnerable_customer' AND language == 'es-US'",
                 "targets": [{"queue": es, "priority": 50}]},
                {"filter_friendly_name": "Vulnerable customer EN",
                 "expression": "escalation_reason == 'vulnerable_customer'",
                 "targets": [{"queue": en, "priority": 50}]},
                {"filter_friendly_name": "Spanish",
                 "expression": "language == 'es-US'",
                 "targets": [{"queue": es}]},
            ],
            "default_filter": {"queue": en},
        }
    }


PLACEHOLDER_QUEUE_SIDS = {key: f"WQ<{key}>" for key, *_ in QUEUES}


def bridge_webhook_url(cfg: SetupConfig) -> str:
    """The bridge's status-callback URL: ``https://<relay>/webhook``, **no credentials**.

    The bridge's ``/webhook`` is signature-only (TAC's check): it never answers with a Basic
    challenge, so Twilio would never send credentials to it, and a URL carrying them might be
    signed with them, which the bridge's check would then fail. Only the API routes (the number's
    voice and status URLs) keep ``user:pass@``; those challenge correctly."""
    return f"https://{cfg.relay_domain}{BRIDGE_WEBHOOK_PATH}"


def status_callback_url(cfg: SetupConfig) -> Webhook:
    """The Conversation Configuration's status callback for ``cfg.bridge_hosting``.

    ``fargate``: :func:`bridge_webhook_url`. ``agentcore``: the front door's sink on the API, with the
    Basic credentials inserted only when rendered for a request (:class:`CredUrl`): the API WAF
    challenges for Basic and Orchestrator answers the challenge."""
    if cfg.bridge_hosting == "agentcore":
        return CredUrl(cfg.api_domain, cfg.api_base_path + ORCHESTRATOR_CALLBACK_PATH)
    return bridge_webhook_url(cfg)


def render_webhook(webhook: Webhook, creds: BasicCreds | None) -> str:
    """The URL to send to Twilio: a :class:`CredUrl` with ``creds``, a plain URL as it is."""
    return webhook.render(creds) if isinstance(webhook, CredUrl) else webhook


#: Read-backs of the Conversation Configuration after an update (it can lag briefly), and the wait between them.
VERIFY_READS = 4
VERIFY_READ_WAIT_S = 2.0


def same_url(a: str | None, b: str | None) -> bool:
    """Whether two URLs are the same once percent-encoding is undone: Twilio can return the Basic
    credentials in a status-callback URL encoded differently from how they were sent."""
    if a is None or b is None:
        return a == b
    x, y = urlsplit(a), urlsplit(b)
    return ((x.scheme, (x.hostname or "").lower(), x.port, unquote(x.path), x.query, x.fragment)
            == (y.scheme, (y.hostname or "").lower(), y.port, unquote(y.path), y.query, y.fragment)
            and unquote(x.username or "") == unquote(y.username or "")
            and unquote(x.password or "") == unquote(y.password or "")
            and (x.username is None) == (y.username is None))


def same_callbacks(got: Any, want: list[dict[str, str]]) -> bool:
    """``statusCallbacks`` read back equals ``want``: same length and order, same method, and
    :func:`same_url` for each URL."""
    if not isinstance(got, list) or len(got) != len(want):
        return False
    return all(isinstance(g, Mapping) and set(g) == set(w) and g.get("method") == w["method"]
               and same_url(g.get("url"), w["url"]) for g, w in zip(got, want, strict=True))


def conversation_configuration(cfg: SetupConfig, memory_store_id: str,
                               webhook: Webhook | None,
                               intelligence_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Request body for the Conversation Configuration (``webhook``: the status-callback URL).

    ``intelligence_ids``: the ``intelligenceConfigurationIds`` already on a live configuration, re-sent
    because a PUT replaces the whole object. Never added by this script; left out when empty.

    ``webhook=None`` (phase ``pre``) leaves ``statusCallbacks`` out: the API makes it optional."""
    body: dict[str, Any] = {
        "displayName": CONVERSATION_CONFIG_NAME,
        "description": "OwlBank inbound voice blueprint (ConversationRelay via TAC)",
        "conversationGroupingType": GROUPING_TYPE,
        "memoryStoreId": memory_store_id,
        "memoryExtractionEnabled": False,  # the solution writes its own recap; extraction would write caller-ID/Anonymous profiles
        # VOICE without captureRules: ConversationRelay passes conversationConfiguration
        # (active ingestion); capture rules would double the STT.
        "channelSettings": {"VOICE": {"statusTimeouts": {
            "inactive": cfg.inactive_timeout_min, "closed": cfg.closed_timeout_min}}},
    }
    if intelligence_ids:
        body["intelligenceConfigurationIds"] = list(intelligence_ids)
    if webhook is not None:
        body["statusCallbacks"] = [{"url": webhook, "method": "POST"}]
    return body


def _flex_step(cfg: SetupConfig) -> Step:
    return Step("ensure_flex", "Flex instance", {
        "first": f"find {FLEX_WORKSPACE_LABEL}: if it exists, do nothing (Flex already provisioned)",
        "create": f"POST {FLEX_INSTANCES_URL} {json.dumps(FLEX_INSTANCE_BODY)} (billable)",
        "poll": f"GET {FLEX_INSTANCES_URL}/<flex_instance_sid> every {FLEX_POLL_S} s until status is "
                f"'ready', up to {cfg.flex_timeout_min} min",
        "then": f"wait for {FLEX_WORKSPACE_LABEL} to appear"})


def _with_flex(cfg: SetupConfig, plan: list[Step]) -> list[Step]:
    """``--create-flex`` (under ``--with-flex``): ``ensure_flex`` goes right after ``check_account``."""
    if cfg.with_flex and cfg.create_flex:
        plan.insert(1, _flex_step(cfg))
    return plan


def _skills_step() -> Step:
    """``ensure_flex_skills``: defines the skill names in Flex; assigns them to no one."""
    return Step("ensure_flex_skills", ", ".join(AGENT_SKILLS), {
        "define": f"add any missing skill to taskrouter_skills in {FLEX_CONFIGURATION_URL} (existing ones kept)",
        "assign": "nothing: no worker is touched (use Flex admin or scripts/assign_agent_skills.py)"})


def _flex_resource_steps(cfg: SetupConfig) -> list[Step]:
    """Queues, workflow and skills: only under ``--with-flex`` (they need the Flex workspace)."""
    if not cfg.with_flex:
        return []
    steps = [Step("ensure_queue", cfg.queue_names[key],
                  {"key": key, "workspace": FLEX_WORKSPACE_LABEL, "target_workers": target})
             for key, _, _, target in QUEUES]
    steps.append(Step("ensure_workflow", WORKFLOW_NAME,
                      {"workspace": FLEX_WORKSPACE_LABEL,
                       "configuration": workflow_configuration(PLACEHOLDER_QUEUE_SIDS)}))
    steps.append(_skills_step())
    return steps


def _pre_plan(cfg: SetupConfig) -> list[Step]:
    """Phase ``pre``: everything that needs no domain."""
    plan = [Step("check_account", cfg.account_sid or "<--account-sid>", {"require": ACCOUNT_REQUIRE})]
    plan.extend(_flex_resource_steps(cfg))
    plan.append(Step("ensure_memory_store", MEMORY_STORE_NAME, {"description": "OwlBank blueprint profiles"}))
    plan.append(Step("ensure_conversation_configuration", CONVERSATION_CONFIG_NAME,
                     {"body": conversation_configuration(cfg, "<memory store id>", None),
                      "webhook": None,
                      "status_callbacks": "omitted until --phase post (kept as they are if already set)"}))
    plan.append(_verify_step(cfg))
    return _with_trait_group(_with_restricted_keys(cfg, _with_flex(cfg, plan)))


def _with_trait_group(plan: list[Step]) -> list[Step]:
    """The ``WebActivity`` trait group, last: it fails open, so nothing a call needs waits on it."""
    plan.append(Step("ensure_trait_group", WEB_ACTIVITY_GROUP,
                     {"store": MEMORY_STORE_NAME, "action": f"create trait group {WEB_ACTIVITY_GROUP}",
                      "only_if": "missing", "on_failure": "warn and finish (web-activity offer off)"}))
    return plan


def _verify_step(cfg: SetupConfig) -> Step:
    return Step("ensure_verify_service", cfg.verify_friendly_name,
                {"code_length": VERIFY_CODE_LENGTH, "validity": "default (10 minutes)",
                 "do_not_share_warning_enabled": True,
                 "default_template_sid": cfg.verify_template_sid or "(Verify default)"})


def _restricted_key_step(cfg: SetupConfig, name: str) -> Step:
    key = RESTRICTED_KEYS[name]
    return Step("ensure_restricted_key", name, {
        "secret": key.secret_name,
        "create": f"POST {IAM_KEYS_URL} KeyType=restricted",
        "friendly_name": key.friendly_name,
        "permissions": list(key.permissions) or "none",
        "policy": json.dumps({"allow": list(key.permissions)}),
        "credentials": "the account's Auth Token (twilio_auth_token secret): a Standard key can't make keys",
        "only_if": ("always (--rotate-restricted-keys; the old key stays until you delete it)"
                    if cfg.rotate_restricted_keys else "the secret holds no key for this account"),
        "value": "written straight to the secret, never shown",
        "used_with": f"{key.infra_flag}=on"})


def _with_restricted_keys(cfg: SetupConfig, plan: list[Step]) -> list[Step]:
    """The Restricted keys, before the trait group (which fails open and stays last)."""
    plan.extend(_restricted_key_step(cfg, name) for name in cfg.restricted_keys)
    return plan


def _taskrouter_events_step(cfg: SetupConfig) -> Step:
    return Step("ensure_taskrouter_events", FLEX_WORKSPACE_LABEL, {
        "event_callback_url": CredUrl(cfg.api_domain, cfg.api_base_path + TASKROUTER_EVENTS_PATH),
        "events_filter": TASKROUTER_EVENTS_FILTER,
        "only_if": "EventCallbackUrl is empty or already this front door's; otherwise report it and stop"})


def _with_taskrouter_events(cfg: SetupConfig, plan: list[Step]) -> list[Step]:
    """``--taskrouter-events``: the workspace event callback, last (nothing else depends on it)."""
    if cfg.with_flex and cfg.taskrouter_events:
        plan.append(_taskrouter_events_step(cfg))
    return plan


def _post_plan(cfg: SetupConfig) -> list[Step]:
    """Phase ``post``: the webhooks, which need the deployed domains."""
    voice_url = CredUrl(cfg.api_domain, cfg.api_base_path + VOICE_INCOMING_PATH)
    status_url = CredUrl(cfg.api_domain, cfg.api_base_path + VOICE_STATUS_PATH)
    webhook = status_callback_url(cfg)
    number = {"phone_number": cfg.phone_number, "voice_url": voice_url, "voice_method": "POST",
              "status_callback": status_url, "status_callback_method": "POST"}
    return _with_taskrouter_events(cfg, [
        Step("check_account", cfg.account_sid or "<--account-sid>", {"require": ACCOUNT_REQUIRE}),
        Step("configure_number", cfg.phone_number, number),
        Step("ensure_status_callbacks", CONVERSATION_CONFIG_NAME,
             {"webhook": webhook, "requires": "the configuration made by --phase pre"}),
        Step("verify_setup", "read back", {
            "checks": ["number voice and status URLs", "status callbacks",
                       "configuration ID matches the output file"],
            "number": number, "webhook": webhook}),
    ])


def build_plan(cfg: SetupConfig) -> list[Step]:
    if cfg.phase == "pre":
        return _pre_plan(cfg)
    if cfg.phase == "post":
        return _post_plan(cfg)
    voice_url = CredUrl(cfg.api_domain, cfg.api_base_path + VOICE_INCOMING_PATH)
    status_url = CredUrl(cfg.api_domain, cfg.api_base_path + VOICE_STATUS_PATH)
    webhook = status_callback_url(cfg)
    plan = [
        Step("check_account", cfg.account_sid or "<--account-sid>",
             {"require": ACCOUNT_REQUIRE}),
        Step("configure_number", cfg.phone_number,
             {"phone_number": cfg.phone_number, "voice_url": voice_url, "voice_method": "POST",
              "status_callback": status_url, "status_callback_method": "POST"}),
    ]
    plan.extend(_flex_resource_steps(cfg))
    plan.append(Step("ensure_memory_store", MEMORY_STORE_NAME,
                     {"description": "OwlBank blueprint profiles"}))
    plan.append(Step("ensure_conversation_configuration", CONVERSATION_CONFIG_NAME,
                     {"body": conversation_configuration(cfg, "<memory store id>", webhook),
                      "webhook": webhook}))
    plan.append(_verify_step(cfg))
    return _with_taskrouter_events(cfg, _with_trait_group(_with_restricted_keys(cfg, _with_flex(cfg, plan))))


# --------------------------------------------------------------------------- apply


def _same_target(url: str, ours: CredUrl) -> bool:
    """Whether ``url`` is this front door's route (https, same host and path), whatever its credentials."""
    parts = urlsplit(url)
    return (parts.scheme == "https" and (parts.hostname or "").lower() == ours.host.lower()
            and parts.port in (None, 443) and unquote(parts.path) == ours.path)


def _shown(url: str) -> str:
    """A URL for a message: scheme, host and path only (no credentials, query or fragment)."""
    parts = urlsplit(url)
    host = parts.hostname or "?"
    return f"{parts.scheme or '?'}://{'***:***@' if parts.username else ''}{host}{parts.path}"


def _masked(url: Any) -> str:
    """A webhook URL for output: scheme, host and path only (never credentials, query or fragment)."""
    if not url:
        return "(none)"
    parts = urlsplit(str(url))
    return f"{parts.scheme or '?'}://{parts.hostname or '?'}{parts.path}"


def describe_number_hooks(num: Any) -> str:
    """A number's voice and status webhooks and its voice application, masked."""
    app = getattr(num, "voice_application_sid", None)
    return (f"voice {getattr(num, 'voice_method', None)} {_masked(getattr(num, 'voice_url', None))}; "
            f"status {getattr(num, 'status_callback_method', None)} "
            f"{_masked(getattr(num, 'status_callback', None))}; voice_application_sid {app or '(none)'}"
            + (" (an application on the number overrides its voice URL)" if app else ""))


def _norm_channels(settings: Any) -> dict[str, Any]:
    """Channel settings in a comparable shape (missing capture rules == none)."""
    out: dict[str, Any] = {}
    for channel, value in (settings or {}).items():
        value = value or {}
        out[channel] = {"statusTimeouts": value.get("statusTimeouts"),
                        "captureRules": value.get("captureRules") or []}
    return out


class FlexApiError(ScriptError):
    """A Flex API call failed. ``status`` (HTTP) and ``code`` (Twilio error number) are what
    ``timed_call`` and ``script_error`` log; the response body is never kept."""

    def __init__(self, message: str, *, status: int, code: int | None = None) -> None:
        super().__init__(message)
        self.status, self.code = status, code


class TwilioRejected(ScriptError):
    """Twilio answered a request with a 4xx: carries the API's own ``message`` (scrubbed, at most
    :data:`TWILIO_MESSAGE_MAX` characters), which describes the validation failure and holds no secrets."""

    def __init__(self, message: str, *, status: int, code: int | None) -> None:
        super().__init__(message)
        self.status, self.code = status, code


def _rejected(step: Step, exc: Exception) -> TwilioRejected | None:
    """``exc`` as a :class:`TwilioRejected` if it is a Twilio 4xx that has a message, else None."""
    status, msg = getattr(exc, "status", None), getattr(exc, "msg", None)
    if (isinstance(exc, ScriptError) or isinstance(status, bool) or not isinstance(status, int)
            or not 400 <= status < 500 or not isinstance(msg, str) or not msg.strip()):
        return None
    code = getattr(exc, "code", None)
    code = code if isinstance(code, int) and not isinstance(code, bool) else None
    detail = " ".join(scrub(msg).split())
    if len(detail) > TWILIO_MESSAGE_MAX:
        detail = detail[:TWILIO_MESSAGE_MAX] + "..."
    return TwilioRejected(f"Twilio rejected {step.kind} (HTTP {status}, code {code}): {detail}",
                          status=status, code=code)


def _twilio_error_code(text: str) -> int | None:
    """The numeric ``code`` of a Twilio JSON error body, else None (nothing else is read from it)."""
    try:
        doc = json.loads(text)
    except ValueError:
        return None
    code = doc.get("code") if isinstance(doc, dict) else None
    return code if isinstance(code, int) and not isinstance(code, bool) else None


_KEY_SID = re.compile(r"SK[0-9a-f]{32}")


def _aws_code(exc: Exception) -> str:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        return str(resp.get("Error", {}).get("Code", type(exc).__name__))
    return type(exc).__name__


class KeyStore:
    """Creates Twilio Restricted keys and writes each straight to its Secrets Manager secret.

    The key's ``secret`` exists only between the Twilio answer and ``put_secret_value``: it is never
    printed, logged, returned or put in an error message. Twilio's Key resource can't be called with a
    Standard API key, so the request uses ``auth_client``, the client built from the account's Auth Token
    (the ``twilio_auth_token`` secret, read once before any change; ``twilio.rest.Client``-shaped: its
    ``request`` method is used)."""

    def __init__(self, sm: Any, kms: Any | None, arns: Mapping[str, str], account_sid: str,
                 auth_client: Any) -> None:
        self.sm, self.kms, self.arns, self.account_sid = sm, kms, dict(arns), account_sid
        self._iam = auth_client

    def current(self, name: str) -> dict[str, str] | None:
        """The key the secret holds (account SID and key SID only), or None if it is the placeholder or
        not a well-formed key. A failed read stops the run (the code only)."""
        arn = self.arns[name]
        try:
            resp = self.sm.get_secret_value(SecretId=arn)
        except Exception as exc:
            raise ScriptError(f"secret {name} ({arn}): get_secret_value failed: {_aws_code(exc)}") from None
        raw = resp.get("SecretString") if isinstance(resp, Mapping) else None
        try:
            value = json.loads(raw) if isinstance(raw, str) else None
        except ValueError:
            value = None
        if validate("twilio_api_key", value):
            return None
        held = cast(Mapping[str, str], value)
        return {"account_sid": held["account_sid"], "api_key": held["api_key"]}

    def check_kms(self, name: str) -> None:
        """populate_secrets' rule: write only to a secret under a customer-managed KMS key."""
        check_kms(self.arns[name], self.sm, self.kms)

    def _iam_request(self) -> Any:
        return self._iam

    def create_and_store(self, key: RestrictedKey) -> str:
        """Create ``key`` at Twilio and write it to its secret; return the new key SID."""
        body = {"AccountSid": self.account_sid, "FriendlyName": key.friendly_name,
                "KeyType": "restricted", "Policy": json.dumps({"allow": list(key.permissions)})}
        with timed_call("twilio", "restricted_key_create"):
            resp = self._iam_request().request("POST", IAM_KEYS_URL, data=body)
        status = int(resp.status_code)
        if status >= 400:
            code = _twilio_error_code(resp.text)
            raise ScriptError(f"Twilio refused to create the Restricted key for {key.secret} "
                              f"(HTTP {status}, Twilio error {code})")
        try:
            doc = json.loads(resp.text)
        except ValueError:
            doc = None
        sid = doc.get("sid") if isinstance(doc, dict) else None
        secret = doc.get("secret") if isinstance(doc, dict) else None
        if not isinstance(sid, str) or not _KEY_SID.fullmatch(sid):
            raise ScriptError(f"Twilio's answer for the {key.secret} key has no valid key SID; check "
                              "API keys in the Console and delete any key named "
                              f"{key.friendly_name!r} it shows")
        value = {"account_sid": self.account_sid, "api_key": sid, "api_secret": secret}
        if validate("twilio_api_key", value):
            raise ScriptError(f"Twilio's answer for key {sid} ({key.secret}) has no usable secret; "
                              f"delete key {sid} in the Console and re-run")
        try:
            self.sm.put_secret_value(SecretId=self.arns[key.secret], SecretString=json.dumps(value))
        except Exception as exc:
            raise ScriptError(f"created Restricted key {sid} but could not write secret {key.secret} "
                              f"({_aws_code(exc)}); its secret is lost: delete key {sid} in the Console "
                              "and re-run") from None
        return sid


class Applier:
    """Executes a plan against a ``twilio.rest.Client``-shaped object."""

    def __init__(self, client: Any, cfg: SetupConfig, creds: BasicCreds | None,
                 log: Callable[[str], None] = print, *, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic, keys: KeyStore | None = None,
                 account_client: Any | None = None) -> None:
        self.client, self.cfg, self.creds, self.log = client, cfg, creds, log
        #: The client built from the Auth Token: a Standard API key can't read the Accounts resource,
        #: so there is no fallback to ``client``: ``check_account`` refuses to run without it.
        self.account_client = account_client
        self.keys = keys
        self.sleep, self.clock = sleep, clock
        self.out: dict[str, Any] = {"accountSid": cfg.account_sid}
        if cfg.with_flex:
            self.out["queueSids"] = {}
        self._workspace: str | None = None

    def run(self, plan: list[Step]) -> dict[str, Any]:
        for step in plan:
            # configure_number is named by the phone number: leave names off that line.
            name = None if step.kind == "configure_number" else step.name
            try:
                with timed_call("twilio", step.kind):
                    getattr(self, step.kind)(step)
            except Exception as exc:
                rejected = _rejected(step, exc)
                script_step(slog, "twilio_setup", step.kind, "error", name=name)
                script_error(slog, "twilio_setup", rejected or exc, step=step.kind, name=name,
                             decision="abort")
                if rejected is not None:
                    raise rejected from None
                raise
            script_step(slog, "twilio_setup", step.kind, "ok", name=name)
        return self.out

    # -- steps

    def check_account(self, step: Step) -> None:
        if self.account_client is None:
            raise ScriptError("check_account needs the Auth Token client (account_client); the Standard "
                              "API key client can't read the Accounts resource")
        acct = self.account_client.api.v2010.accounts(self.cfg.account_sid).fetch()
        if acct.status != "active":
            raise ScriptError(f"account {acct.sid} is {acct.status!r}, not active")
        if not acct.owner_account_sid or acct.owner_account_sid == acct.sid:
            if not self.cfg.dedicated_main_account:
                raise ScriptError(f"account {acct.sid} is a main account; the blueprint runs in a "
                                  "dedicated subaccount. If this main account is used only "
                                  "by the blueprint, pass --dedicated-main-account")
            self.log(f"warn    account {acct.sid} is a main account, accepted as dedicated "
                     "(--dedicated-main-account)")
            return
        self.log(f"ok      account {acct.sid} is an active subaccount")

    def configure_number(self, step: Step) -> None:
        p = step.params
        found = self.client.incoming_phone_numbers.list(phone_number=p["phone_number"])
        if len(found) != 1:
            raise ScriptError(f"number {step.name}: expected 1 owned number, found {len(found)}")
        num = found[0]
        want = {"voice_url": p["voice_url"].render(self.creds), "voice_method": p["voice_method"],
                "status_callback": p["status_callback"].render(self.creds),
                "status_callback_method": p["status_callback_method"]}
        diff = [k for k, v in want.items() if getattr(num, k, None) != v]
        if diff:
            self.log(f"previous number {num.sid}: {describe_number_hooks(num)}")
            self.client.incoming_phone_numbers(num.sid).update(**want)
            self.log(f"update  number {num.sid}: {', '.join(diff)}")
        else:
            self.log(f"ok      number {num.sid}")
        self.out["phoneNumberSid"] = num.sid

    def _find_workspace(self) -> str | None:
        """The Flex TaskRouter workspace's SID, or None. The one resolver for every step.

        Order: (a) with the instance SID known, the name ``Flex Task Assignment for <GO...>`` (what the
        Instances API makes); (b) the exact name ``Flex Task Assignment`` (the Console's); (c) the single
        workspace whose name starts with ``Flex Task Assignment``. Several in (c) is an error."""
        if self._workspace is not None:
            return self._workspace
        every = list(self.client.taskrouter.v1.workspaces.list())
        known = self.out.get("flexInstanceSid") or self.cfg.known_flex_instance_sid
        wanted = [f"{FLEX_WORKSPACE} for {known}"] if known else []
        wanted.append(FLEX_WORKSPACE)
        for name in wanted:
            hit = [w for w in every if w.friendly_name == name]
            if hit:
                self._workspace = hit[0].sid
                break
        else:
            prefixed = [w for w in every if str(w.friendly_name).startswith(FLEX_WORKSPACE)]
            if len(prefixed) > 1:
                listed = ", ".join(f"{w.friendly_name!r} ({w.sid})" for w in prefixed)
                raise ScriptError(f"more than one TaskRouter workspace name starts with "
                                  f"{FLEX_WORKSPACE!r}: {listed}; can't tell which is Flex's; "
                                  "remove or rename the extra one")
            if not prefixed:
                return None
            self._workspace = prefixed[0].sid
        self.out["workspaceSid"] = self._workspace
        return self._workspace

    def _workspace_sid(self) -> str:
        sid = self._find_workspace()
        if sid is None:
            raise ScriptError("the Flex TaskRouter workspace (named 'Flex Task Assignment...') not found; "
                              "set up Flex first, then re-run setup with --with-flex")
        return sid

    # -- Flex (--create-flex)

    def _flex_call(self, method: str, operation: str, url: str,
                   body: dict[str, Any] | None = None, *, api: str = "Flex Instances API",
                   hint: str = FLEX_CREATE_HINT) -> dict[str, Any]:
        """One Flex request through the client's HTTP layer (Basic auth with the API key).

        Raises :class:`FlexApiError` on an HTTP error or an unreadable answer; the message and the
        log carry the HTTP status and Twilio error code only, never the response body."""
        with timed_call("twilio", operation):
            resp = self.client.request(method, url, data=body,
                                       headers={"Content-Type": "application/json"} if body else None)
            status = int(resp.status_code)
            if status >= 400:
                code = _twilio_error_code(resp.text)
                raise FlexApiError(
                    f"{api} {method} failed (HTTP {status}, Twilio error {code}).{hint}",
                    status=status, code=code)
            try:
                doc = json.loads(resp.text)
            except ValueError:
                doc = None
            if not isinstance(doc, dict):
                raise FlexApiError(f"{api} {method} returned an unreadable answer "
                                   f"(HTTP {status})", status=status)
            return doc

    @staticmethod
    def _flex_sid(doc: dict[str, Any]) -> str:
        sid = doc.get("flex_instance_sid")
        if not isinstance(sid, str) or not FLEX_INSTANCE_SID.fullmatch(sid):
            raise ScriptError("Flex Instances API answered without a valid flex_instance_sid; "
                              "check Flex in the Console")
        return sid

    def ensure_flex(self, step: Step) -> None:
        if self._find_workspace() is not None:
            self.log(f"ok      Flex already provisioned (workspace {self._workspace})")
            if self.cfg.known_flex_instance_sid:
                self.out["flexInstanceSid"] = self.cfg.known_flex_instance_sid
            return
        sid = self.cfg.known_flex_instance_sid
        status: Any = None
        if sid:  # an earlier run created it and stopped before the workspace appeared
            try:
                status = self._flex_call("GET", "flex_instance_fetch", f"{FLEX_INSTANCES_URL}/{sid}"
                                         ).get("status")
                self.log(f"ok      Flex instance {sid} found from the output file (status {status!r})")
            except FlexApiError as exc:
                if exc.status != 404:
                    raise
                sid = None
        if not sid:
            created = self._flex_call("POST", "flex_instance_create", FLEX_INSTANCES_URL,
                                      FLEX_INSTANCE_BODY)
            sid, status = self._flex_sid(created), created.get("status")
            self.log(f"create  Flex instance {sid} (status {status!r}; billable)")
        self.out["flexInstanceSid"] = sid
        self._await_flex_ready(sid, status)
        self._await_flex_workspace(sid)

    def _await_flex_ready(self, sid: str, status: Any) -> None:
        deadline = self.clock() + self.cfg.flex_timeout_min * 60
        while True:
            if status == "ready":
                self.log(f"ok      Flex instance {sid} is ready")
                return
            if status != "creation-in-progress":
                raise ScriptError(f"Flex instance {sid} has status {str(status)[:40]!r}, expected "
                                  "'creation-in-progress' or 'ready'; check Flex in the Console")
            if self._find_workspace() is not None:  # visible already: nothing left to wait for
                self.log(f"ok      Flex workspace {self._workspace} is visible; not waiting longer")
                return
            if self.clock() >= deadline:
                raise ScriptError(
                    f"Flex instance {sid} is not ready after {self.cfg.flex_timeout_min} minutes. It is "
                    "still being created: wait, then re-run the same command (safe: the existing "
                    "instance is detected, nothing is created twice; the SID is kept in --output). "
                    "Raise the wait with --flex-timeout-min")
            self.sleep(FLEX_POLL_S)
            status = self._flex_call("GET", "flex_instance_fetch", f"{FLEX_INSTANCES_URL}/{sid}"
                                     ).get("status")

    def _await_flex_workspace(self, sid: str) -> None:
        """The workspace can lag behind ``ready``; ensure_queue needs it."""
        for attempt in range(FLEX_WORKSPACE_TRIES):
            if attempt:
                self.sleep(FLEX_POLL_S)
            if self._find_workspace() is not None:
                self.log(f"ok      Flex workspace {self._workspace} is available")
                return
        raise ScriptError(f"Flex instance {sid} is ready but the TaskRouter workspace "
                          f"({FLEX_WORKSPACE!r}...) isn't visible yet; re-run the same command in a few "
                          "minutes (safe)")

    def ensure_queue(self, step: Step) -> None:
        ws = self.client.taskrouter.v1.workspaces(self._workspace_sid())
        target = step.params["target_workers"]
        found = ws.task_queues.list(friendly_name=step.name)
        if found:
            q = found[0]
            if q.target_workers != target:
                ws.task_queues(q.sid).update(target_workers=target)
                self.log(f"update  queue {step.name} ({q.sid}): target_workers")
            else:
                self.log(f"ok      queue {step.name} ({q.sid})")
        else:
            q = ws.task_queues.create(friendly_name=step.name, target_workers=target)
            self.log(f"create  queue {step.name} ({q.sid})")
        self.out["queueSids"][step.params["key"]] = q.sid

    def ensure_workflow(self, step: Step) -> None:
        ws = self.client.taskrouter.v1.workspaces(self._workspace_sid())
        config = workflow_configuration(self.out["queueSids"])
        found = ws.workflows.list(friendly_name=step.name)
        if found:
            wf = found[0]
            try:
                current = json.loads(wf.configuration or "null")
            except ValueError:
                slog.debug("workflow configuration is not JSON", extra={
                    "decision": "treated_as_changed_and_updated"})
                current = None
            if current != config:
                ws.workflows(wf.sid).update(configuration=json.dumps(config))
                self.log(f"update  workflow {step.name} ({wf.sid}): configuration")
            else:
                self.log(f"ok      workflow {step.name} ({wf.sid})")
        else:
            wf = ws.workflows.create(friendly_name=step.name, configuration=json.dumps(config))
            self.log(f"create  workflow {step.name} ({wf.sid})")
        self.out["workflowSid"] = wf.sid

    def ensure_flex_skills(self, step: Step) -> None:
        """Define the agent skills in Flex (add the missing names, keep the existing entries). No worker is
        touched: ``scripts/assign_agent_skills.py`` or Flex admin gives skills to agents.

        Not fail-open: without the skills no escalation reaches an agent."""
        api = "Flex Configuration API"
        hint = " Check the API key's access to Flex, then re-run (safe)"
        conf = self._flex_call("GET", "flex_configuration_fetch", FLEX_CONFIGURATION_URL, api=api, hint=hint)
        listed = conf.get("taskrouter_skills")
        entries = [e for e in listed if isinstance(e, dict)] if isinstance(listed, list) else []
        have = {e.get("name") for e in entries}
        missing = [n for n in AGENT_SKILLS if n not in have]
        if missing:
            account = conf.get("account_sid")
            if not isinstance(account, str) or not account:
                raise ScriptError("Flex Configuration API answered without an account_sid; check Flex in the Console")
            added = [{"name": n, "multivalue": False, "minimum": None, "maximum": None} for n in missing]
            self._flex_call("POST", "flex_configuration_update", FLEX_CONFIGURATION_URL,
                            {"account_sid": account, "taskrouter_skills": [*entries, *added]},
                            api=api, hint=hint)
            self.log(f"create  Flex skills {', '.join(missing)}")
        else:
            self.log(f"ok      Flex skills {', '.join(AGENT_SKILLS)}")

    def _find_memory_store(self, name: str) -> str | None:
        for store_id in self.client.memory.v1.stores.list():
            if self.client.memory.v1.stores(store_id).fetch().display_name == name:
                return store_id
        return None

    def ensure_memory_store(self, step: Step) -> None:
        store_id = self._find_memory_store(step.name)
        if store_id:
            self.log(f"ok      memory store {step.name} ({store_id})")
        else:
            from twilio.rest.memory.v1.store import StoreList

            created = self.client.memory.v1.stores.create(StoreList.ServiceRequest(
                {"displayName": step.name, "description": step.params["description"]}))
            store_id = created.id or self._find_memory_store(step.name)
            if not store_id:
                raise ScriptError(f"memory store {step.name} is still provisioning; re-run later")
            self.log(f"create  memory store {step.name} ({store_id})")
        self.out["memoryStoreId"] = store_id

    def ensure_trait_group(self, step: Step) -> None:
        """Fails open: a failure warns and setup finishes; only the web-activity
        offer waits for a re-run. A :class:`ScriptError` message is secret-free by contract; any other
        error is named by its type only."""
        try:
            ensure_memory_trait_group(self.client, self.out["memoryStoreId"], log=self.log,
                                      sleep=self.sleep, clock=self.clock)
        except Exception as exc:
            script_error(slog, "twilio_setup", exc, step=step.kind, name=step.name, decision="continue")
            why = str(exc) if isinstance(exc, ScriptError) else type(exc).__name__
            self.log(f"warn    trait group {step.name} not made ({why}); setup continues, the web-activity "
                     "offer stays off until a re-run makes it")

    def _sync_configuration(self, name: str, memory_store_id: str | None, webhook: Webhook | None, *,
                            create: bool) -> str:
        """Find the configuration by name and bring it to the wanted body; return its ID.

        ``webhook=None`` (phase ``pre``) never touches ``statusCallbacks``: an update (a PUT replaces the
        whole configuration) re-sends whatever is set. ``create=False`` (phase ``post``) refuses to
        create one: the bridge needs the ID from the earlier run.
        """
        from twilio.rest.conversations.v2.configuration import ConfigurationList

        url = None if webhook is None else render_webhook(webhook, self.creds)
        configs = self.client.conversations.v2.configurations
        found = [c for c in configs.list() if c.display_name == name]
        if not found and not create:
            raise ScriptError(f"conversation configuration {name!r} not found; run --phase pre first")
        if found:
            cur = found[0]
            body = conversation_configuration(self.cfg, memory_store_id or cur.memory_store_id, url,
                                              getattr(cur, "intelligence_configuration_ids", None) or ())
            current = {"conversationGroupingType": cur.conversation_grouping_type,
                       "memoryStoreId": cur.memory_store_id,
                       "channelSettings": _norm_channels(cur.channel_settings),
                       "statusCallbacks": cur.status_callbacks,
                       "memoryExtractionEnabled": bool(cur.memory_extraction_enabled)}
            if webhook is None:
                current.pop("statusCallbacks")
                if cur.status_callbacks:
                    body["statusCallbacks"] = cur.status_callbacks
            want = {**body, "channelSettings": _norm_channels(body["channelSettings"])}
            diff = [k for k, v in current.items() if v != want.get(k)]
            if "conversationGroupingType" in diff:
                raise ScriptError(f"configuration {name} ({cur.id}) has grouping "
                                  f"{cur.conversation_grouping_type!r}, which can't change; delete it first")
            if diff:
                configs(cur.id).update(update_configuration_request=
                                       ConfigurationList.UpdateConfigurationRequest(body))
                self.log(f"update  conversation configuration {name} ({cur.id}): {', '.join(diff)}")
            else:
                self.log(f"ok      conversation configuration {name} ({cur.id})")
            self.out["memoryStoreId"] = self.out.get("memoryStoreId") or cur.memory_store_id
            return str(cur.id)
        body = conversation_configuration(self.cfg, memory_store_id or "", url)
        created = configs.create(create_configuration_request=
                                 ConfigurationList.CreateConfigurationRequest(body))
        config_id = created.id or next((c.id for c in configs.list() if c.display_name == name), None)
        if not config_id:
            raise ScriptError(f"configuration {name} is still being created; re-run later")
        self.log(f"create  conversation configuration {name} ({config_id})")
        return str(config_id)

    def ensure_conversation_configuration(self, step: Step) -> None:
        self.out["conversationConfigurationId"] = self._sync_configuration(
            step.name, self.out["memoryStoreId"], step.params["webhook"], create=True)

    def ensure_status_callbacks(self, step: Step) -> None:
        self.out["conversationConfigurationId"] = self._sync_configuration(
            step.name, None, step.params["webhook"], create=False)

    def verify_setup(self, step: Step) -> None:
        """Read the number and the configuration back and compare; never prints a URL."""
        p = step.params["number"]
        (num,) = self.client.incoming_phone_numbers.list(phone_number=p["phone_number"])
        for attr, key in (("voice_url", "voice_url"), ("status_callback", "status_callback")):
            if getattr(num, attr, None) != p[key].render(self.creds):
                raise ScriptError(f"number {num.sid}: {attr} is not what was set")
        # Fargate: no credentials; agentcore: the sink URL with them
        want = [{"url": render_webhook(step.params["webhook"], self.creds), "method": "POST"}]
        for attempt in range(VERIFY_READS):
            cur = next((c for c in self.client.conversations.v2.configurations.list()
                        if c.display_name == CONVERSATION_CONFIG_NAME), None)
            if cur is None:
                raise ScriptError("conversation configuration vanished after the update")
            if same_callbacks(cur.status_callbacks, want):
                break
            if attempt + 1 < VERIFY_READS:
                # A read right after the update can still return the previous value.
                self.sleep(VERIFY_READ_WAIT_S)
        else:
            raise ScriptError(f"configuration {cur.id}: statusCallbacks are not what was set")
        expected = self.cfg.expected_conversation_configuration_id
        if expected and expected != cur.id:
            raise ScriptError(
                f"configuration {cur.id} differs from {expected} in the output file, which the bridge "
                "was deployed with; update twilioOutputs and redeploy the Bridge stack, or fix the file")
        self.log(f"ok      verified number {num.sid} and configuration {cur.id}")

    def ensure_taskrouter_events(self, step: Step) -> None:
        """Point the Flex workspace's event callback at the front door, only if it is empty or already
        ours (same host and path); any other value is reported (host and path) and the run stops."""
        sid = self._workspace_sid()
        ws = self.client.taskrouter.v1.workspaces(sid).fetch()
        ours: CredUrl = step.params["event_callback_url"]
        want_url = ours.render(self.creds)
        want_filter = step.params["events_filter"]
        current = (getattr(ws, "event_callback_url", None) or "").strip()
        if current and not _same_target(current, ours):
            raise ScriptError(
                f"workspace {sid} already sends its TaskRouter events to {_shown(current)}; it is "
                "workspace-wide and may be used by Flex or another integration, so it was left as it is. "
                "Re-run without --taskrouter-events, or clear it in the Console if it is no longer used")
        if same_url(current or None, want_url) and (getattr(ws, "events_filter", None) or "") == want_filter:
            self.log(f"ok      workspace {sid} TaskRouter events")
            return
        self.client.taskrouter.v1.workspaces(sid).update(event_callback_url=want_url,
                                                         events_filter=want_filter)
        self.log(f"{'update' if current else 'set   '}  workspace {sid} TaskRouter events: "
                 f"{ours} ({want_filter})")

    def ensure_restricted_key(self, step: Step) -> None:
        """Create the Restricted key and write it to its secret, unless the secret already holds a key
        for this account (``--rotate-restricted-keys`` creates a new one anyway). Prints key SIDs only."""
        if self.keys is None:
            raise ScriptError("no secret store for the Restricted keys")
        key = RESTRICTED_KEYS[step.name]
        held = self.keys.current(key.secret)
        if held is not None and not self.cfg.rotate_restricted_keys:
            if held["account_sid"] != self.cfg.account_sid:
                raise ScriptError(f"secret {key.secret_name} holds a key for another account; pass "
                                  "--rotate-restricted-keys to replace it")
            self.log(f"ok      restricted key {key.secret} ({held['api_key']}) already in {key.secret_name}")
            return
        self.keys.check_kms(key.secret)
        sid = self.keys.create_and_store(key)
        perms = ", ".join(key.permissions) or "no permissions"
        self.log(f"create  restricted key {key.secret} ({sid}; {perms}) -> {key.secret_name}")
        if held is not None:
            self.log(f"next    once the functions read the new key (their secret cache is 300 s), delete "
                     f"the old key {held['api_key']} in the Console")

    def ensure_verify_service(self, step: Step) -> None:
        services = self.client.verify.v2.services
        want: dict[str, Any] = {"code_length": step.params["code_length"],
                                "do_not_share_warning_enabled": True}
        if self.cfg.verify_template_sid:
            want["default_template_sid"] = self.cfg.verify_template_sid
        found = [s for s in services.list() if s.friendly_name == step.name]
        if found:
            svc = found[0]
            diff = {k: v for k, v in want.items() if getattr(svc, k, None) != v}
            if diff:
                services(svc.sid).update(**diff)
                self.log(f"update  verify service {step.name} ({svc.sid}): {', '.join(diff)}")
            else:
                self.log(f"ok      verify service {step.name} ({svc.sid})")
        else:  # no lifetime argument: Verify's default validity (10 minutes)
            svc = services.create(friendly_name=step.name, **want)
            self.log(f"create  verify service {step.name} ({svc.sid})")
        self.out["verifyServiceSid"] = svc.sid


# --------------------------------------------------------------------------- --check (read only)


def _failure(exc: Exception) -> str:
    """An error for output: its type, HTTP status and Twilio code only (no URL, no body)."""
    return f"{type(exc).__name__} (status {getattr(exc, 'status', None)}, code {getattr(exc, 'code', None)})"


class Checker:
    """``--check``: reads what Twilio holds and prints it; makes no write call and prints no secret.

    Every call is a GET or a list. ``problems`` counts what makes ``--check`` exit non-zero."""

    def __init__(self, client: Any, account_client: Any, cfg: SetupConfig,
                 log: Callable[[str], None] = print, keys: KeyStore | None = None) -> None:
        self.client, self.account_client, self.cfg, self.log = client, account_client, cfg, log
        self.keys = keys
        self.problems = 0

    def _say(self, tag: str, text: str) -> None:
        self.log(f"{tag:<8}{text}")

    def _problem(self, text: str) -> None:
        self.problems += 1
        self._say("PROBLEM", text)

    def run(self) -> int:
        self.log(f"CHECK (read only): account {self.cfg.account_sid}")
        self._account()
        self._same_account()
        self._keys()
        self._pre_resources()
        if self.cfg.phone_number and self.cfg.phase != "pre":
            self._number()
        if self.cfg.with_flex:
            self._flex()
        self.log(f"check done: {self.problems} problem(s)")
        return 0 if self.problems == 0 else 1

    def _account(self) -> None:
        sid = self.cfg.account_sid
        try:
            acct = self.account_client.api.v2010.accounts(sid).fetch()
        except Exception as exc:
            self._problem(f"account {sid}: the Auth Token can't read it: {_failure(exc)}")
            return
        main = not acct.owner_account_sid or acct.owner_account_sid == acct.sid
        kind = "main account" if main else "subaccount"
        if acct.status != "active":
            self._problem(f"account {acct.sid} is a {kind} with status {acct.status!r}, not active")
        elif main and not self.cfg.dedicated_main_account:
            self._problem(f"account {acct.sid} is a main account; apply refuses it without "
                          "--dedicated-main-account")
        else:
            self._say("ok", f"account {acct.sid}: active {kind}")

    def _same_account(self) -> None:
        """A Standard key lists Calls only under the account it belongs to."""
        sid = self.cfg.account_sid
        try:
            list(self.client.api.v2010.accounts(sid).calls.list(limit=1))
        except Exception as exc:
            self._problem(f"the Standard API key can't list Calls under {sid}: {_failure(exc)}; the key, "
                          "the Auth Token and the account SID may belong to different accounts")
            return
        self._say("ok", f"the Standard API key lists Calls under {sid}: it belongs to this account")

    def _keys(self) -> None:
        """Report each Restricted key the way apply decides: from the key the secret holds."""
        for name in self.cfg.restricted_keys:
            key = RESTRICTED_KEYS[name]
            if self.keys is None:
                self._problem(f"Restricted key {key.friendly_name}: no secret store to read {key.secret_name}")
                continue
            try:
                held = self.keys.current(key.secret)
            except ScriptError as exc:
                self._problem(f"Restricted key {key.friendly_name}: {exc}")
                continue
            if held is None:
                self._say("missing", f"Restricted key {key.friendly_name}: {key.secret_name} holds no key; "
                                     "apply will create one")
            elif self.cfg.rotate_restricted_keys:
                self._say("todo", f"Restricted key {key.friendly_name}: {key.secret_name} holds "
                                  f"{held['api_key']}; apply will create a new key and replace it "
                                  "(--rotate-restricted-keys)")
            elif held["account_sid"] != self.cfg.account_sid:
                self._problem(f"Restricted key {key.friendly_name}: {key.secret_name} holds a key for "
                              f"another account ({held['account_sid']}); apply will stop unless "
                              "--rotate-restricted-keys is passed")
            else:
                self._say("ok", f"Restricted key {key.friendly_name}: {key.secret_name} holds "
                                f"{held['api_key']}; apply keeps it")

    def _pre_resources(self) -> None:
        cfg = self.cfg
        try:
            verify = [s for s in self.client.verify.v2.services.list()
                      if s.friendly_name == cfg.verify_friendly_name]
            self._say("ok" if verify else "missing",
                      f"Verify service {cfg.verify_friendly_name}" + (f" ({verify[0].sid})" if verify else ""))
            store = Applier(self.client, cfg, None)._find_memory_store(MEMORY_STORE_NAME)
            self._say("ok" if store else "missing",
                      f"Memory store {MEMORY_STORE_NAME}" + (f" ({store})" if store else ""))
            conf = next((c for c in self.client.conversations.v2.configurations.list()
                         if c.display_name == CONVERSATION_CONFIG_NAME), None)
            if conf is None:
                self._say("missing", f"Conversation configuration {CONVERSATION_CONFIG_NAME}")
            else:
                hooks = ", ".join(_masked(c.get("url")) for c in (conf.status_callbacks or [])) or "(none)"
                self._say("ok", f"Conversation configuration {CONVERSATION_CONFIG_NAME} ({conf.id}); "
                                f"status callbacks: {hooks}")
        except Exception as exc:
            self._problem(f"resources not readable: {_failure(exc)}")

    def _number(self) -> None:
        cfg = self.cfg
        try:
            found = list(self.client.incoming_phone_numbers.list(phone_number=cfg.phone_number))
        except Exception as exc:
            self._problem(f"number {cfg.phone_number}: not readable: {_failure(exc)}")
            return
        if len(found) != 1:
            self._problem(f"number {cfg.phone_number}: expected 1 owned number, found {len(found)}")
            return
        num = found[0]
        owner = getattr(num, "account_sid", None)
        if owner and owner != cfg.account_sid:
            self._problem(f"number {num.sid} belongs to account {owner}, not {cfg.account_sid}")
        self._say("ok", f"number {num.sid}: {describe_number_hooks(num)}")
        if cfg.api_domain:
            ours = CredUrl(cfg.api_domain, cfg.api_base_path + VOICE_INCOMING_PATH)
            current = getattr(num, "voice_url", None)
            same = bool(current) and _same_target(str(current), ours)
            self._say("ok" if same else "todo",
                      "the number's voice URL " + ("already targets this front door"
                                                    if same else "is not this front door's (post sets it)"))
        app_sid = getattr(num, "voice_application_sid", None)
        if app_sid:
            self._application(app_sid)

    def _application(self, app_sid: str) -> None:
        try:
            app = self.client.applications(app_sid).fetch()
        except Exception as exc:
            self._say("warn", f"voice application {app_sid}: not readable: {_failure(exc)}")
            return
        self._say("ok", f"voice application {app_sid}: voice {_masked(getattr(app, 'voice_url', None))}")
        owner = getattr(app, "account_sid", None)
        if owner and owner != self.cfg.account_sid:
            self._problem(f"voice application {app_sid} belongs to account {owner}, "
                          f"not {self.cfg.account_sid}")

    def _flex(self) -> None:
        cfg = self.cfg
        try:
            applier = Applier(self.client, cfg, None, log=self.log)
            ws_sid = applier._find_workspace()
            if ws_sid is None:
                self._say("missing", f"{FLEX_WORKSPACE_LABEL} (Flex isn't set up in this account)")
                return
            self._say("ok", f"{FLEX_WORKSPACE_LABEL} ({ws_sid})")
            ws = self.client.taskrouter.v1.workspaces(ws_sid)
            for key, *_ in QUEUES:
                name = cfg.queue_names[key]
                hit = ws.task_queues.list(friendly_name=name)
                self._say("ok" if hit else "missing", f"queue {name}" + (f" ({hit[0].sid})" if hit else ""))
            wf = ws.workflows.list(friendly_name=WORKFLOW_NAME)
            self._say("ok" if wf else "missing",
                      f"workflow {WORKFLOW_NAME}" + (f" ({wf[0].sid})" if wf else ""))
        except ScriptError as exc:
            self._problem(f"Flex resources: {exc}")
        except Exception as exc:
            self._problem(f"Flex resources not readable: {_failure(exc)}")


# --------------------------------------------------------------------------- CLI


def default_client_factory(api_key: str, api_secret: str, account_sid: str) -> Any:
    from twilio.rest import Client

    return Client(api_key, api_secret, account_sid)


def default_sm_client(region: str) -> Any:
    import boto3

    return boto3.client("secretsmanager", region_name=region)


def default_kms_client(region: str) -> Any:
    import boto3

    return boto3.client("kms", region_name=region)


def secrets_for(phase: str) -> tuple[str, ...]:
    """The secrets ``--apply`` reads: the API key and the Auth Token (the account check and the Restricted
    keys use it) always, Basic auth only where webhook URLs are set."""
    base = ("twilio_api_key", "twilio_auth_token")
    return base if phase == "pre" else (*base, "basic_auth")


def restricted_keys_for(phase: str, *, skip_context_api_key: bool) -> tuple[str, ...]:
    """The Restricted keys a run makes: none in ``post``; else the Context API's unless skipped."""
    if phase == "post" or skip_context_api_key:
        return ()
    return ("context_api_key",)


def restricted_key_arns(names: Sequence[str], outputs: Mapping[str, str],
                        explicit: Mapping[str, str]) -> dict[str, str]:
    """Each named Restricted key's secret ARN (``--restricted-key-secret`` or the Data stack output).
    No AWS call."""
    unknown = sorted(set(explicit) - set(RESTRICTED_KEYS))
    if unknown:
        raise ScriptError(f"--restricted-key-secret: unknown name(s) {unknown}; known: {sorted(RESTRICTED_KEYS)}")
    arns: dict[str, str] = {}
    for name in names:
        key = RESTRICTED_KEYS[name]
        arn = explicit.get(name) or find_output(outputs, key.outputs)
        if not arn:
            raise ScriptError(f"no ARN for secret {name}: CDK output {', '.join(key.outputs)} not found. Deploy "
                              f"OwlBankData with {key.infra_flag}=on and pass --cdk-outputs, or pass "
                              f"--restricted-key-secret {name}=ARN"
                              + (", or --skip-context-api-key" if name == "context_api_key" else ""))
        arns[name] = arn
    return arns


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Configure the Twilio side of the OwlBank blueprint "
                                            "(dry run by default).")
    p.add_argument("--account-sid", help="the blueprint's account SID (AC...); required for --apply")
    p.add_argument("--dedicated-main-account", action="store_true",
                   help="accept a main (non-sub) account that is used only by the blueprint")
    p.add_argument("--phase", choices=PHASES, default="all",
                   help="pre: everything that needs no domain (run after OwlBankData and the Twilio secrets, "
                        "before the other stacks); post: the webhooks (after them); all: both in one run "
                        "(default, when the domains are known)")
    p.add_argument("--phone-number", help="the owned Twilio number to configure (E.164); not needed for --phase pre")
    p.add_argument("--api-domain", help="front-door API host: the custom domain, or API Gateway's "
                                        "<id>.execute-api.<region>.amazonaws.com (or CDK output ApiDomain)")
    p.add_argument("--api-stage", help="API Gateway stage for the webhook URL path (default: CDK output "
                                       f"ApiStage, else {DEFAULT_API_STAGE!r}, only for an execute-api host; "
                                       "never added on a custom domain unless you pass it)")
    p.add_argument("--relay-domain", help="bridge host: the custom domain, or the CloudFront domain "
                                          "d....cloudfront.net (or CDK output RelayDomain)")
    p.add_argument("--verify-friendly-name", default=DEFAULT_VERIFY_FRIENDLY_NAME,
                   help="friendly name of the Verify Service to find or create "
                        f"(default {DEFAULT_VERIFY_FRIENDLY_NAME!r}; shown in code messages)")
    p.add_argument("--verify-template-sid", metavar="HJ...",
                   help="a Twilio-approved Verify message template to make the service default "
                        "(docs/verify-sms-template.md); without it the default template is used")
    p.add_argument("--with-flex", action="store_true",
                   help="opt in: also set up Flex (find the Flex TaskRouter workspace, the queues, the "
                        "workflow, the agent skills). Without it no Flex API call is made and the outputs "
                        "file has no workspace, workflow or queue SIDs. Required by --create-flex and "
                        "--taskrouter-events")
    p.add_argument("--create-flex", action="store_true",
                   help="opt in (needs --with-flex): if the Flex workspace is missing, create a Flex instance through the API "
                        "(BILLABLE; conversations.default false) and wait for it; phases pre and all. "
                        "Without it Flex must already be activated (Console)")
    p.add_argument("--flex-timeout-min", type=int, default=DEFAULT_FLEX_TIMEOUT_MIN,
                   help="--create-flex: how long to wait for the instance to be ready "
                        f"(default {DEFAULT_FLEX_TIMEOUT_MIN}, polled every {FLEX_POLL_S} s)")
    p.add_argument("--taskrouter-events", action="store_true",
                   help="opt in (phases post and all): send the Flex workspace's TaskRouter events "
                        f"({TASKROUTER_EVENTS_FILTER}) to the front door's {TASKROUTER_EVENTS_PATH}, for the "
                        "queue wait after a hand-off. Workspace-wide: set only when the callback is empty "
                        "or already this front door's; any other value is reported and left as it is")
    p.add_argument("--skip-context-api-key", action="store_true",
                   help="don't make the Context API's Restricted key (phases pre and all make it by default, "
                        "for contextApiRestrictedKey=on, the default); needed with contextApiRestrictedKey=off, "
                        "where its secret doesn't exist")
    p.add_argument("--rotate-restricted-keys", action="store_true",
                   help="make new Restricted keys even when their secrets already hold one; the old keys "
                        "are left for you to delete in the Console")
    p.add_argument("--restricted-key-secret", action="append", metavar="NAME=ARN",
                   help="ARN of a Restricted key's secret, overriding the outputs file; NAME in "
                        f"{sorted(RESTRICTED_KEYS)}")
    p.add_argument("--policy", type=Path, default=DATA_DIR / "policy.json")
    p.add_argument("--inactive-timeout-min", type=int, default=DEFAULT_INACTIVE_MIN)
    p.add_argument("--closed-timeout-min", type=int, default=DEFAULT_CLOSED_MIN)
    p.add_argument("--output", type=Path, default=Path("infra/twilio.context.json"),
                   help="where --apply writes the resulting SIDs (merged into the file if it exists)")
    p.add_argument("--check", action="store_true",
                   help="read-only: verify the account and print what Twilio holds today (resources, the "
                        "number's webhooks, its voice application, the Flex workspace under --with-flex); "
                        "changes nothing and prints no secret; can't be combined with --apply")
    add_secret_args(p)  # --cdk-outputs (domains, stage, hosting and the Secret* ARNs), --secret, --region
    add_mode_args(p)
    return p


def api_base_path_for(api_domain: str, stage_arg: str | None, stage_output: str | None) -> str:
    """``/<stage>`` for the webhook URL path, or ``""``.

    An explicit ``--api-stage`` always applies. For API Gateway's default endpoint the stage is
    the CDK output ``ApiStage``, else :data:`DEFAULT_API_STAGE`: that endpoint serves
    ``/<stage>/voice/...`` and the front door's signature check uses the path with the stage.
    A custom domain's base path mapping has no stage, so nothing is added.
    """
    if stage_arg:
        stage: str | None = stage_arg
    elif EXECUTE_API_HOST.match(api_domain):
        stage = stage_output or DEFAULT_API_STAGE
    else:
        stage = None
    if stage is None:
        return ""
    if not _STAGE.fullmatch(stage):
        raise ScriptError("--api-stage must be a stage name (letters, digits, - and _)")
    return f"/{stage}"


def bridge_hosting_from(outputs: Mapping[str, str]) -> str:
    """The ``BridgeHosting`` CDK output (only the ``agentcore`` bridge stack has it): missing is
    ``fargate``; an unknown value is an error (it would point the callback at the wrong place)."""
    value = find_output(outputs, ("BridgeHosting",))
    if not value:
        return "fargate"
    if value not in HOSTINGS:
        raise ScriptError(f"CDK output BridgeHosting must be one of {', '.join(HOSTINGS)}")
    return value


def _read_output_file(path: Path) -> dict[str, Any]:
    """The existing ``--output`` file as a dict, or ``{}`` (missing or not a JSON object)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


#: Edited by hand to the README's first-deploy order (a test checks the order and that each script exists).
NEXT_STEPS = {
    "pre": "next: scripts/check_secrets.py --strict, then infra/scripts/preflight.py --before-bridge, "
           "then scripts/seed.py --apply, then deploy the remaining stacks (README, first deploy)",
    "post": "next: scripts/assign_agent_skills.py --apply, then scripts/deploy_flex_plugin.py, "
            "then the test calls (README, first deploy)",
    "post_no_flex": "next: the test calls (README, first deploy)",
}


def next_steps(phase: str, with_flex: bool) -> str | None:
    """The hint for ``phase``: post's Flex steps only when this run set Flex up (``--with-flex``)."""
    if phase == "post" and not with_flex:
        return NEXT_STEPS["post_no_flex"]
    return NEXT_STEPS.get(phase)


def _keep_flex_sid(path: Path, previous: dict[str, Any], applier: Applier | None) -> None:
    """A run that stops after creating the Flex instance keeps its SID in ``--output``, so the re-run
    resumes that instance (nothing else is written on failure)."""
    sid = applier.out.get("flexInstanceSid") if applier else None
    if not sid or previous.get("flexInstanceSid") == sid:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**previous, "flexInstanceSid": sid}, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    except OSError as exc:
        script_error(slog, "twilio_setup", exc, step="keep_flex_sid", decision="ignored")


def _run_check(args: argparse.Namespace, cfg: SetupConfig, secret_arns: Mapping[str, str],
               sm_client: Any, client_factory: Callable[[str, str, str], Any],
               key_arns: Mapping[str, str] | None = None) -> int:
    """``--check``: read the two Twilio secrets, then only read from Twilio."""
    script_step(slog, "twilio_setup", "check", "start")
    try:
        sm = sm_client
        if sm is None:
            sm = default_sm_client(resolve_region(
                args.region, arns=[*secret_arns.values(), *(key_arns or {}).values()], logger=slog))
        key = read_secret("twilio_api_key", secret_arns["twilio_api_key"], sm)
        token = read_secret("twilio_auth_token", secret_arns["twilio_auth_token"], sm)
        client = client_factory(key["api_key"], key["api_secret"], args.account_sid)
        auth_client = client_factory(args.account_sid, token["auth_token"], args.account_sid)
        keys = KeyStore(sm, None, key_arns, args.account_sid, auth_client) if key_arns else None
        problems = Checker(client, auth_client, cfg, keys=keys).run()
    except ScriptError as exc:
        script_error(slog, "twilio_setup", exc, step="check", decision="exit_3")
        print(f"twilio_setup: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        script_error(slog, "twilio_setup", exc, step="check", decision="exit_1")
        print(f"twilio_setup: {_failure(exc)}", file=sys.stderr)
        return 1
    script_step(slog, "twilio_setup", "check", "ok" if problems == 0 else "problems")
    return 3 if problems else 0


def main(argv: list[str] | None = None, *, sm_client: Any = None, kms_client: Any = None,
         client_factory: Callable[[str, str, str], Any] = default_client_factory,
         sleep: Callable[[float], None] = time.sleep,
         clock: Callable[[], float] = time.monotonic) -> int:
    args = build_parser().parse_args(argv)
    phase = args.phase
    script_start(slog, "twilio_setup", mode="apply" if args.apply else "check" if args.check else "dry_run", phase=phase,
                 dedicated_main_account=bool(args.dedicated_main_account),
                 with_flex=bool(args.with_flex), create_flex=bool(args.create_flex),
                 taskrouter_events=bool(args.taskrouter_events),
                 skip_context_api_key=bool(args.skip_context_api_key),
                 rotate_restricted_keys=bool(args.rotate_restricted_keys))
    try:
        api_domain = relay_domain = ""
        api_base_path = ""
        bridge_hosting = "fargate"
        outputs = load_cdk_outputs(args.cdk_outputs)
        explicit_secrets = parse_secret_overrides(args.secret)
        if phase != "pre":
            if not args.phone_number:
                raise ScriptError(f"--phone-number is required for --phase {phase}")
            api_domain = args.api_domain or find_output(outputs, ("ApiDomain", "ApiDomainName", "FrontDoorDomain")) or ""
            relay_domain = (args.relay_domain
                            or find_output(outputs, ("RelayDomain", "RelayDomainName", "BridgeDomain")) or "")
            bridge_hosting = bridge_hosting_from(outputs)
            if not api_domain or (bridge_hosting == "fargate" and not relay_domain):
                raise ScriptError("need --api-domain and --relay-domain (or CDK outputs ApiDomain, RelayDomain)")
            api_base_path = api_base_path_for(api_domain, args.api_stage,
                                              find_output(outputs, ("ApiStage",)))
        if args.check and args.apply:
            raise ScriptError("--check is read-only and can't be combined with --apply")
        if (args.apply or args.check) and not args.account_sid:
            raise ScriptError(f"{'--apply' if args.apply else '--check'} needs --account-sid")
        # ARNs only (no AWS call); a dry run needs none. --check reads the API key and the Auth Token only.
        secret_arns = ({name: secret_arn(name, outputs, explicit_secrets)
                        for name in secrets_for(phase) if args.apply or name != "basic_auth"}
                       if args.apply or args.check else {})
        restricted = restricted_keys_for(phase, skip_context_api_key=args.skip_context_api_key)
        key_overrides = parse_pairs(args.restricted_key_secret, what="--restricted-key-secret")
        key_arns: dict[str, str] = {}
        if (args.apply or args.check) and restricted:
            key_arns = restricted_key_arns(restricted, outputs, key_overrides)
        try:
            policy_doc = json.loads(args.policy.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ScriptError(f"can't read {args.policy}: {type(exc).__name__}") from None
        previous = _read_output_file(args.output)
        expected_id = previous.get("conversationConfigurationId")
        known_flex = previous.get("flexInstanceSid")
        cfg = SetupConfig(phone_number=args.phone_number or "", api_domain=api_domain,
                          relay_domain=relay_domain, account_sid=args.account_sid,
                          queue_names=queue_names_from_policy(policy_doc),
                          inactive_timeout_min=args.inactive_timeout_min,
                          closed_timeout_min=args.closed_timeout_min,
                          dedicated_main_account=args.dedicated_main_account,
                          verify_friendly_name=args.verify_friendly_name,
                          verify_template_sid=args.verify_template_sid,
                          api_base_path=api_base_path, phase=phase, bridge_hosting=bridge_hosting,
                          with_flex=args.with_flex, create_flex=args.create_flex,
                          flex_timeout_min=args.flex_timeout_min,
                          taskrouter_events=args.taskrouter_events,
                          restricted_keys=restricted,
                          rotate_restricted_keys=args.rotate_restricted_keys,
                          known_flex_instance_sid=(
                              known_flex if isinstance(known_flex, str)
                              and FLEX_INSTANCE_SID.fullmatch(known_flex) else None),
                          expected_conversation_configuration_id=(
                              expected_id if phase == "post" and isinstance(expected_id, str) else None))
        plan = build_plan(cfg)
    except ScriptError as exc:
        script_error(slog, "twilio_setup", exc, step="plan", decision="exit_2")
        print(f"twilio_setup: {exc}", file=sys.stderr)
        return 2
    script_step(slog, "twilio_setup", "plan", "ok", steps=len(plan), phase=phase)

    if args.check:
        return _run_check(args, cfg, secret_arns, sm_client, client_factory, key_arns)

    if not args.apply:
        script_step(slog, "twilio_setup", "apply", "dry_run")
        print("DRY RUN: no Twilio calls. Planned changes (find before create)"
              + (f", phase {phase}:" if phase != "all" else ":"))
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step.describe()}")
        return 0

    applier: Applier | None = None
    try:
        region = ""
        if sm_client is None or (key_arns and kms_client is None):
            region = resolve_region(args.region, arns=[*secret_arns.values(), *key_arns.values()],
                                    logger=slog)
        sm = sm_client if sm_client is not None else default_sm_client(region)
        key = read_secret("twilio_api_key", secret_arns["twilio_api_key"], sm)
        api_key, api_secret = key["api_key"], key["api_secret"]
        creds = None
        if "basic_auth" in secret_arns:  # the Basic credentials only go into webhook URLs
            basic = read_secret("basic_auth", secret_arns["basic_auth"], sm)
            creds = BasicCreds(basic["user"], basic["password"])
        token = read_secret("twilio_auth_token", secret_arns["twilio_auth_token"], sm)
        client = client_factory(api_key, api_secret, args.account_sid)
        auth_client = client_factory(args.account_sid, token["auth_token"], args.account_sid)
        keys = None
        if key_arns:
            kms = kms_client if kms_client is not None else default_kms_client(region)
            keys = KeyStore(sm, kms, key_arns, args.account_sid, auth_client)
        applier = Applier(client, cfg, creds, sleep=sleep, clock=clock, keys=keys,
                          account_client=auth_client)
        out = applier.run(plan)
    except ScriptError as exc:
        _keep_flex_sid(args.output, previous, applier)
        script_error(slog, "twilio_setup", exc, step="apply", decision="exit_3")
        print(f"twilio_setup: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # Twilio errors: print the type and status only (no request URL)
        _keep_flex_sid(args.output, previous, applier)
        script_error(slog, "twilio_setup", exc, step="apply", decision="exit_1")
        status = getattr(exc, "status", None)
        code = getattr(exc, "code", None)
        print(f"twilio_setup: {type(exc).__name__} (status {status}, code {code})", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A later phase adds to what an earlier one wrote (an empty value, like post's no queues, never erases).
    merged = {**previous, **{k: v for k, v in out.items() if v not in ({}, None)}}
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    script_step(slog, "twilio_setup", "write_output", "ok")
    print(f"wrote {args.output}")
    hint = next_steps(phase, bool(args.with_flex))
    if hint:
        print(hint)
    return 0


if __name__ == "__main__":
    start_logging("twilio_setup")
    sys.exit(main())
