"""Data: the customer-managed KMS key, the token-sealing key, DynamoDB tables, the
secrets, the alarm topic, the Bedrock invocation-logging rule and the account's IAM Access
Analyzer."""

from __future__ import annotations

import json

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_accessanalyzer as aa
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_events as events
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_secretsmanager as sm
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from constructs import Construct

from . import call_observability, model_failover
from .config import InfraConfig
from .contract import NUMERIC_KEYS, TABLES, TTL_ATTRIBUTE
from .nag import ack
from .observability import LOG_GROUP_PATTERNS, Alarms

#: Where AgentCore writes a Runtime's logs: ``<prefix><AgentRuntimeId>-DEFAULT``.
RUNTIME_LOG_GROUP_PREFIX = "/aws/bedrock-agentcore/runtimes/"

#: Crypto actions on the token key. The key-policy Deny names the wildcard forms on
#: purpose: a Deny should be broad, so it also covers every ReEncrypt/GenerateDataKey variant.
TOKEN_KEY_DENIED_ACTIONS = ["kms:Decrypt", "kms:Encrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*"]
#: Key administration for the account root (delegated to IAM). No crypto actions and no
#: CreateGrant, so the account can manage the key but only the three sealing roles can use it.
KEY_ADMIN_ACTIONS = [
    "kms:CancelKeyDeletion", "kms:CreateAlias", "kms:DeleteAlias", "kms:DescribeKey",
    "kms:DisableKey", "kms:DisableKeyRotation", "kms:EnableKey", "kms:EnableKeyRotation",
    "kms:GetKeyPolicy", "kms:GetKeyRotationStatus", "kms:ListAliases", "kms:ListGrants",
    "kms:ListKeyPolicies", "kms:ListKeyRotations", "kms:ListResourceTags", "kms:PutKeyPolicy",
    "kms:RetireGrant", "kms:RevokeGrant", "kms:RotateKeyOnDemand", "kms:ScheduleKeyDeletion",
    "kms:TagResource", "kms:UntagResource", "kms:UpdateAlias", "kms:UpdateKeyDescription",
]


#: CloudTrail's record of a change to Bedrock model-invocation logging.
BEDROCK_LOGGING_EVENT_PATTERN = {
    "source": ["aws.bedrock"],
    "detail-type": ["AWS API Call via CloudTrail"],
    "detail": {"eventSource": ["bedrock.amazonaws.com"],
               "eventName": ["PutModelInvocationLoggingConfiguration"]},
}


def role_arn(stack: Stack, name: str) -> str:
    return f"arn:{stack.partition}:iam::{stack.account}:role/{name}"

#: Secret logical name -> (Secrets Manager name suffix, description).
SECRETS: dict[str, tuple[str, str]] = {
    "twilio_api_key": ("twilio/api-key",
                       "Twilio API key for REST and Memory calls: JSON {account_sid, api_key, api_secret}"),
    "twilio_auth_token": ("twilio/auth-token",
                          "Twilio Auth Token, used only to validate X-Twilio-Signature: JSON {auth_token}"),
    "basic_auth": ("twilio/basic-auth",
                   "HTTP Basic credentials in the Twilio webhook URLs: JSON {user, password}"),
}
#: Only with ``contextApiRestrictedKey=on`` (the default): the Context API's own Twilio
#: Restricted key (TaskRouter task read and reservation list). Same JSON shape as ``twilio_api_key``,
#: so the handler is unchanged. ``scripts/twilio_setup.py`` creates the key and writes the value.
CONTEXT_API_KEY_SECRET = "context_api_key"  # noqa: S105 (a logical name, not a value)
OPTIONAL_SECRETS: dict[str, tuple[str, str]] = {
    CONTEXT_API_KEY_SECRET: ("twilio/context-api-key",
                             "Twilio Restricted API key for the Context API's TaskRouter reads: "
                             "JSON {account_sid, api_key, api_secret}"),
}


def secrets_for(cfg: InfraConfig) -> dict[str, tuple[str, str]]:
    """The secrets this configuration creates: :data:`SECRETS`, plus the Context API's own key with
    ``contextApiRestrictedKey=on``."""
    wanted = ((CONTEXT_API_KEY_SECRET, cfg.context_api_restricted_key_on),)
    return {**SECRETS, **{name: OPTIONAL_SECRETS[name] for name, on in wanted if on}}


#: The Basic auth password's character rules. With punctuation, upper case and g-z excluded,
#: the generator draws only from 0-9 and a-f, so the password is lower-case hex.
BASIC_AUTH_PASSWORD_LENGTH = 48
BASIC_AUTH_EXCLUDE_CHARACTERS = "ghijklmnopqrstuvwxyz"


def basic_auth_generator(user: str) -> sm.SecretStringGenerator:
    """The ``basic_auth`` secret's generator: ``{"user": <user>, "password": <48 hex>}``."""
    return sm.SecretStringGenerator(
        secret_string_template=json.dumps({"user": user}),
        generate_string_key="password",
        password_length=BASIC_AUTH_PASSWORD_LENGTH,
        exclude_punctuation=True, exclude_uppercase=True, include_space=False,
        exclude_characters=BASIC_AUTH_EXCLUDE_CHARACTERS)


def _attr(name: str) -> ddb.Attribute:
    return ddb.Attribute(name=name,
                         type=ddb.AttributeType.NUMBER if name in NUMERIC_KEYS else ddb.AttributeType.STRING)


class DataStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, **kw) -> None:
        super().__init__(scope, cid, **kw)

        self.key = kms.Key(
            self, "DataKey",
            alias="alias/owlbank-data",
            description="OwlBank blueprint CMK: DynamoDB, Secrets Manager, CloudWatch Logs, AgentCore",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        # CloudWatch Logs encrypts every OwlBank log group with this key (AWS defaults).
        self.key.add_to_resource_policy(iam.PolicyStatement(
            sid="AllowCloudWatchLogs",
            principals=[iam.ServicePrincipal(f"logs.{self.region}.amazonaws.com")],
            actions=["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:Describe*"],
            resources=["*"],  # in a key policy, "*" means this key
            # Only OwlBank's own log groups: /owlbank/* (Lambdas, bridge, API access, flow
            # logs) and the WAF log groups, whose names must start with aws-waf-logs-.
            conditions={"ArnLike": {"kms:EncryptionContext:aws:logs:arn": [
                f"arn:{self.partition}:logs:{self.region}:{self.account}:log-group:{pat}"
                for pat in LOG_GROUP_PATTERNS]}},
        ))
        # CloudWatch alarms publish to the CMK-encrypted alarm topic; SNS calls KMS for them.
        # [unverified at deploy] the kms:ViaService value for alarm -> SNS deliveries.
        self.key.add_to_resource_policy(iam.PolicyStatement(
            sid="AllowCloudWatchAlarmsToEncryptedTopic",
            principals=[iam.ServicePrincipal("cloudwatch.amazonaws.com")],
            actions=["kms:Decrypt", "kms:GenerateDataKey"],
            resources=["*"],  # in a key policy, "*" means this key
            conditions={"StringEquals": {"kms:ViaService": f"sns.{self.region}.amazonaws.com",
                                         "aws:SourceAccount": self.account}},
        ))
        # AgentCore hosting: a Runtime's pre-created log group,
        # /aws/bedrock-agentcore/runtimes/<AgentRuntimeId>-DEFAULT, is CMK-encrypted too. Its
        # ID isn't known before the Runtime exists, but it starts with the fixed
        # AgentRuntimeName, so the condition names that prefix only, with bridgeHosting=agentcore.
        # Fargate adds nothing, so its template is unchanged.
        if cfg.is_agentcore:
            runtime_arn = (f"arn:{self.partition}:logs:{self.region}:{self.account}:log-group:"
                           f"{RUNTIME_LOG_GROUP_PREFIX}{cfg.runtime_name}-*")
            self.key.add_to_resource_policy(iam.PolicyStatement(
                sid="RuntimeLogsCmk",
                principals=[iam.ServicePrincipal(f"logs.{self.region}.amazonaws.com")],
                actions=["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*",
                         "kms:Describe*"],
                resources=["*"],  # in a key policy, "*" means this key
                conditions={"ArnLike": {"kms:EncryptionContext:aws:logs:arn": runtime_arn}},
            ))

        # A dedicated key for sealing call tokens. Its policy, not just IAM,
        # limits who can use it: only the front door Lambda role (encrypt + decrypt), the
        # auth service role (encrypt: it seals the prepared welcome) and
        # the bridge task role (decrypt), each only with a `call_sid` encryption context.
        # Any other principal, even one with kms:* in IAM, is denied. The roles have fixed
        # names (cfg.*_role_name), so this stack names their ARNs without a cycle.
        # With the recap writer on (``memoryStoreId`` in twilioOutputs),
        # its role may also decrypt (it unseals the call's ``sealed_sensitive`` factor values to
        # scrub them from the recap), with the same ``call_sid`` context condition.
        fd_arn = role_arn(self, cfg.frontdoor_role_name)
        auth_arn = role_arn(self, cfg.auth_service_role_name)
        bridge_arn = role_arn(self, cfg.bridge_task_role_name)
        recap_arn = role_arn(self, cfg.recap_writer_role_name) if cfg.recap_writer_on else None
        root = iam.AccountRootPrincipal()
        call_sid = {"ForAnyValue:StringEquals": {"kms:EncryptionContextKeys": ["call_sid"]}}
        recap_unseal = [iam.PolicyStatement(
            sid="RecapWriterUnseal", principals=[root],
            actions=["kms:Decrypt"], resources=["*"],
            conditions={"ArnEquals": {"aws:PrincipalArn": recap_arn}, **call_sid})] if recap_arn else []
        token_policy = iam.PolicyDocument(statements=[
            iam.PolicyStatement(
                sid="KeyAdministration", principals=[root], actions=KEY_ADMIN_ACTIONS,
                resources=["*"]),  # in a key policy, "*" means this key
            iam.PolicyStatement(
                sid="FrontdoorSealUnseal", principals=[root],
                actions=["kms:Encrypt", "kms:Decrypt"], resources=["*"],
                conditions={"ArnEquals": {"aws:PrincipalArn": fd_arn}, **call_sid}),
            iam.PolicyStatement(
                sid="AuthServiceSeal", principals=[root],
                actions=["kms:Encrypt"], resources=["*"],
                conditions={"ArnEquals": {"aws:PrincipalArn": auth_arn}, **call_sid}),
            iam.PolicyStatement(
                sid="BridgeUnseal", principals=[root],
                actions=["kms:Decrypt"], resources=["*"],
                conditions={"ArnEquals": {"aws:PrincipalArn": bridge_arn}, **call_sid}),
            *recap_unseal,
            # A Deny guard, not a grant: "*" is the only way to say "everyone else". The
            # A test allows exactly this statement shape on this key.
            iam.PolicyStatement(
                sid="DenyCryptoExceptSealingRoles", effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()], actions=TOKEN_KEY_DENIED_ACTIONS,
                resources=["*"],
                conditions={"ArnNotEquals": {"aws:PrincipalArn": [
                    fd_arn, auth_arn, bridge_arn, *([recap_arn] if recap_arn else [])]}}),
        ])
        self.token_key = kms.Key(
            self, "TokenKey",
            alias="alias/owlbank-token-sealing",
            description="OwlBank call-token sealing; encryption context must include call_sid",
            enable_key_rotation=True,
            policy=token_policy,  # replaces CDK's default root kms:* statement
            removal_policy=RemovalPolicy.RETAIN,
        )

        self.tables: dict[str, ddb.Table] = {}
        for logical, td in TABLES.items():
            table = ddb.Table(
                self, f"{logical.capitalize()}Table",
                partition_key=_attr(td.pk),
                sort_key=_attr(td.sk) if td.sk else None,
                billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
                encryption=ddb.TableEncryption.CUSTOMER_MANAGED,
                encryption_key=self.key,
                point_in_time_recovery_specification=ddb.PointInTimeRecoverySpecification(
                    point_in_time_recovery_enabled=True),
                time_to_live_attribute=TTL_ATTRIBUTE if td.ttl else None,
                deletion_protection=cfg.is_prod,
                removal_policy=RemovalPolicy.RETAIN,
            )
            for index_name, ipk, isk in td.indexes:
                table.add_global_secondary_index(
                    index_name=index_name, partition_key=_attr(ipk),
                    sort_key=_attr(isk) if isk else None,
                    projection_type=ddb.ProjectionType(td.projection(index_name)))
            self.tables[logical] = table

        # CDK creates every secret and its access policy; no value is ever in the template.
        # Basic auth is generated by Secrets Manager: the fixed
        # user from basicAuthUser and a 48-character lower-case hex password, safe inside the
        # webhook URLs without escaping. The two Twilio secrets start as a generated placeholder
        # (CloudFormation needs an initial value) until the operator sets their values after
        # the deploy, in the console, the CLI or scripts/populate_secrets.py. The Restricted-key
        # secrets (OPTIONAL_SECRETS) start the same way; scripts/twilio_setup.py writes them.
        self.secrets: dict[str, sm.Secret] = {}
        for logical, (suffix, desc) in secrets_for(cfg).items():
            if logical == "basic_auth":
                generator = basic_auth_generator(cfg.basic_auth_user)
                desc += ". Generated by Secrets Manager at deploy; no one sets it"
            else:
                generator = sm.SecretStringGenerator(
                    secret_string_template='{"placeholder": true}',  # noqa: S106
                    generate_string_key="unset", exclude_punctuation=True)
                desc += (". Created in Twilio and written by scripts/twilio_setup.py --phase pre; "
                         "placeholder until then" if logical in OPTIONAL_SECRETS else
                         ". Set after deploy by the operator (console, CLI or "
                         "scripts/populate_secrets.py); placeholder until then")
            secret = sm.Secret(
                self, "".join(p.capitalize() for p in logical.split("_")) + "Secret",
                secret_name=f"owlbank/{suffix}",
                description=desc + ".",
                encryption_key=self.key,
                generate_secret_string=generator,
                removal_policy=RemovalPolicy.RETAIN,
            )
            ack(secret, "AwsSolutions-SMG4",
                "Twilio credentials can't be rotated by a Secrets Manager rotation Lambda: they "
                "are rotated in Twilio (secondary Auth Token flow, new API key), then written by "
                "the operator (console, CLI or scripts/populate_secrets.py). The generated Basic "
                "auth password is rotated by writing a new generated value, then re-running "
                "twilio_setup.py --phase post.")
            self.secrets[logical] = secret

        # Every alarm notifies this topic, encrypted with the CMK. No subscription
        # unless alarmEmail is set (the owner confirms it from the email).
        self.alarm_topic = sns.Topic(self, "AlarmTopic", topic_name=f"{cfg.name_prefix}-alarms",
                                     master_key=self.key, enforce_ssl=True)
        if cfg.alarm_email:
            self.alarm_topic.add_subscription(subs.EmailSubscription(cfg.alarm_email))
        self._bedrock_logging_rule(cfg)
        # Model failover: the fleet table, always; its write-budget alarm only
        # when on. Not in self.tables: only the bridge and the poller name it, by its fixed name.
        self.failover_table: ddb.Table = model_failover.build_table(
            self, cfg, self.key, Alarms(self, self.alarm_topic, cfg))

        # Call observability: optionally Transaction Search, only when on. The traces log
        # group's key (TracesKey) lives in OwlBankBridge with the group, so nothing is exported.
        if cfg.call_observability_on and cfg.traces_enable_transaction_search:
            call_observability.transaction_search(self, cfg)

        # Any resource shared outside the account becomes a finding after deploy
        # (checked by scripts/preflight.py --post-deploy).
        aa.CfnAnalyzer(self, "AccessAnalyzer", type="ACCOUNT",
                       analyzer_name=f"{cfg.name_prefix}-account")

        # Owned real numbers are seeded only in dev/test stacks, by scripts/seed.py.
        if cfg.is_dev_or_test and (cfg.developer_phone or cfg.test_numbers):
            CfnOutput(self, "SeedOwnedNumbers",
                      description="Owned numbers for scripts/seed.py (dev/test only)",
                      value=",".join(([cfg.developer_phone] if cfg.developer_phone else [])
                                     + list(cfg.test_numbers)))
        # Output names are the contract with scripts/seed.py and scripts/populate_secrets.py.
        for logical, table in self.tables.items():
            CfnOutput(self, f"Table{logical.capitalize()}", value=table.table_name)
        for logical, secret in self.secrets.items():
            CfnOutput(self, "Secret" + "".join(p.capitalize() for p in logical.split("_")),
                      value=secret.secret_arn)

    def _bedrock_logging_rule(self, cfg: InfraConfig) -> None:
        """Alarm when anyone turns Bedrock model-invocation logging on (or changes it),
        since the resolver's prompts would then be logged with callers' digits.

        A rule on the default bus matches CloudTrail's PutModelInvocationLoggingConfiguration
        (it needs a trail logging management write events; preflight checks for one) and
        publishes the event to the alarm topic. Not ``targets.SnsTopic``: it grants EventBridge
        on the topic and the key with no condition. Here both grants name this rule's ARN.

        Deployment Region only: calls routed through
        the US inference profile are logged by the source Region's configuration, so this Region's
        is the one to watch.
        """
        name = f"{cfg.name_prefix}-bedrock-logging-change"
        # Built from the fixed name, not the rule's GetAtt: key -> rule -> topic -> key would cycle.
        rule_arn = f"arn:{self.partition}:events:{self.region}:{self.account}:rule/{name}"
        only_this_rule = {"ArnEquals": {"aws:SourceArn": rule_arn}}
        events.CfnRule(
            self, "BedrockLoggingChangeRule",
            name=name,
            description="Bedrock model-invocation logging configuration changed: "
                        "the resolver's prompts may now be logged",
            event_bus_name="default",
            event_pattern=BEDROCK_LOGGING_EVENT_PATTERN,
            state="ENABLED",
            targets=[events.CfnRule.TargetProperty(id="AlarmTopic", arn=self.alarm_topic.topic_arn)],
        )
        self.alarm_topic.add_to_resource_policy(iam.PolicyStatement(
            sid="AllowBedrockLoggingRulePublish",
            principals=[iam.ServicePrincipal("events.amazonaws.com")],
            actions=["sns:Publish"],
            resources=[self.alarm_topic.topic_arn],
            conditions=only_this_rule,
        ))
        # The topic is CMK-encrypted, so EventBridge also needs the key for its deliveries.
        # [unverified at deploy] that KMS sees aws:SourceArn = the rule ARN on these calls.
        self.key.add_to_resource_policy(iam.PolicyStatement(
            sid="AllowBedrockLoggingRuleToEncryptedTopic",
            principals=[iam.ServicePrincipal("events.amazonaws.com")],
            actions=["kms:GenerateDataKey*", "kms:Decrypt"],
            resources=["*"],  # in a key policy, "*" means this key
            conditions=only_this_rule,
        ))
