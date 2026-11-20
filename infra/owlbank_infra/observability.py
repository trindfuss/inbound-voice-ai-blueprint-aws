"""Log groups, WAF logging, metric filters and alarms.

* Every OwlBank log group is named under ``/owlbank/`` (or ``aws-waf-logs-owlbank-``,
  the prefix WAF requires), so the CMK's CloudWatch Logs statement can be scoped to
  :data:`LOG_GROUP_PATTERNS`. The one exception is an AgentCore Runtime's group, whose name
  AgentCore fixes (``/aws/bedrock-agentcore/runtimes/<RuntimeId>-DEFAULT``); the
  Data stack's key policy covers it with its own statement, per Runtime name.
* ``stage=prod``: log groups are retained, with 1-year retention. Other stages keep
  the demo defaults (destroyed with the stack, 1 month).
* Every alarm notifies the Data stack's CMK-encrypted SNS topic.
* ``bridgeHosting=agentcore``: :meth:`Alarms.runtime_alarms` puts the bridge's reject and
  error filters on the Runtime's log group and alarms on the ``AWS/Bedrock-AgentCore`` metrics;
  the saved queries use :func:`agentcore_log_group_names`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_wafv2 as waf
from constructs import Construct

from .config import InfraConfig

#: Log-group name patterns the CMK may encrypt (the key policy's ArnLike values).
LOG_GROUP_PATTERNS = ("/owlbank/*", "aws-waf-logs-owlbank-*")
#: Prefix of the log group AgentCore writes a Runtime's logs to. Not in
#: :data:`LOG_GROUP_PATTERNS`: the key policy names each Runtime's prefix separately.
RUNTIME_LOG_GROUP_PREFIX = "/aws/bedrock-agentcore/runtimes/"
#: Every prefix :func:`log_group` accepts.
LOG_GROUP_PREFIXES = ("/owlbank/", "aws-waf-logs-owlbank-", RUNTIME_LOG_GROUP_PREFIX)
#: Both components log a stable JSON line on logger ``owlbank.security`` for every
#: rejected Twilio request: {"event":"twilio_auth_failed","component":"frontdoor|bridge",
#: "reason":"signature|basic|nonce|route|unverifiable"} (src/owlbank). The filters
#: match the event name only, so the alarm doesn't depend on log wording or TAC's text.
AUTH_FAILED_EVENT = "twilio_auth_failed"
FRONTDOOR_REJECT_MESSAGE = AUTH_FAILED_EVENT
BRIDGE_REJECT_MESSAGES = (AUTH_FAILED_EVENT,)
#: The auth service logs {"event":"otp_sent_untrusted_ani","component":"auth_service"} on
#: ``owlbank.security`` when it sends a one-time code on a call with an untrusted caller ID
#: (src/owlbank/security_log.py): a possible spoofed caller ID aimed at a real
#: customer. The filter matches the event name; any single occurrence alarms.
OTP_UNTRUSTED_ANI_EVENT = "otp_sent_untrusted_ani"
#: A code sent on a keyed-in number's call logs {"event":"otp_sent_keyed_phone",...} instead
#: The expected path for those calls, so counted, never alarmed.
OTP_KEYED_PHONE_EVENT = "otp_sent_keyed_phone"
OTP_KEYED_PHONE_METRIC = "OtpSentKeyedPhone"
#: Model-assisted slots: the front door logs {"event":"slot_mode_invalid","level":"ERROR",
#: "item":...,"field":...} once per process per (item, field, policy version) when a slot mode value
#: is missing, unknown or of the wrong type, and uses ``deterministic`` for that slot. The filter
#: matches the event name as a literal term, like the others (no JSON filters on the call
#: path); any single occurrence in 5 minutes alarms.
SLOT_MODE_INVALID_EVENT = "slot_mode_invalid"
SLOT_MODE_INVALID_METRIC = "SlotModeInvalid"
SLOT_MODE_INVALID_THRESHOLD = 1
#: Request headers every WAF log redacts (lowercase, as WAF wants them).
REDACTED_HEADERS = ("authorization", "cookie", "x-twilio-signature", "x-origin-verify")
WAF_BLOCKED_THRESHOLD = 100  # per 5 minutes: a burst, not the steady scanner background
#: The bridge's reject filter (same construct ID and metric on both hostings).
BRIDGE_AUTH_REJECTED_ID = "BridgeAuthRejected"
BRIDGE_REJECT_METRIC = "BridgeWsRejected"
#: Log levels the error filter counts, and its threshold per 5 minutes (the Lambdas' Errors one).
ERROR_LEVELS = ("ERROR", "CRITICAL")
APP_ERROR_THRESHOLD = 3

# ---- AgentCore Runtime metrics (AgentCore devguide "AgentCore generated runtime
# observability data"). Per-Runtime metrics carry the dimensions the CDK's ``Runtime.metric*``
# helpers use: Operation=InvokeAgentRuntime, Name=<runtime name>::<endpoint>, Resource=<Runtime ARN>.
# ``ActiveSessionCount`` is an account-wide gauge with the single dimension Service=AgentCore.Runtime.
AGENTCORE_NAMESPACE = "AWS/Bedrock-AgentCore"
RUNTIME_OPERATION = "InvokeAgentRuntime"
RUNTIME_ENDPOINT_NAME = "DEFAULT"
RUNTIME_THROTTLES_THRESHOLD = 1  # per 5 minutes (any throttle is visible)
RUNTIME_SYSTEM_ERRORS_THRESHOLD = 5  # per 5 minutes
#: Alarm at 80 % of ``agentcoreSessionQuota``.
ACTIVE_SESSION_ALARM_FRACTION = 0.8


def active_session_threshold(quota: int) -> int:
    """80 % of the active-session quota, rounded up, and at least 1."""
    return max(1, math.ceil(quota * ACTIVE_SESSION_ALARM_FRACTION))


def runtime_metric(metric_name: str, runtime_name: str, runtime_arn: str, period: Duration) -> cw.Metric:
    """A per-Runtime ``AWS/Bedrock-AgentCore`` metric on the DEFAULT endpoint, summed per period."""
    return cw.Metric(namespace=AGENTCORE_NAMESPACE, metric_name=metric_name, statistic="Sum", period=period,
                     dimensions_map={"Operation": RUNTIME_OPERATION,
                                     "Name": f"{runtime_name}::{RUNTIME_ENDPOINT_NAME}",
                                     "Resource": runtime_arn})


def log_settings(cfg: InfraConfig) -> tuple[logs.RetentionDays, RemovalPolicy]:
    if cfg.is_prod:
        return logs.RetentionDays.ONE_YEAR, RemovalPolicy.RETAIN
    return logs.RetentionDays.ONE_MONTH, RemovalPolicy.DESTROY


def log_group(scope: Construct, cid: str, *, cfg: InfraConfig, key: kms.IKey, name: str) -> logs.LogGroup:
    if not name.startswith(LOG_GROUP_PREFIXES):
        raise ValueError(f"log group {name!r} is outside the CMK's log-group scope")
    retention, removal = log_settings(cfg)
    return logs.LogGroup(scope, cid, log_group_name=name, encryption_key=key,
                         retention=retention, removal_policy=removal)


def waf_logging(stack: Stack, cid: str, *, cfg: InfraConfig, key: kms.IKey, acl: waf.CfnWebACL,
                which: str, redact_query_string: bool) -> logs.LogGroup:
    """WAF logs to ``aws-waf-logs-owlbank-<stage>-<which>``, with credentials redacted.

    ``authorization`` (HTTP Basic, or the admin console's Cognito token), ``cookie``,
    ``x-twilio-signature`` (a valid signature for a captured URL and body is replayable) and
    ``x-origin-verify`` (the admin API's CloudFront-only header) are always redacted. The bridge's query string
    carries the WebSocket nonce, so it is redacted there too.
    """
    name = f"aws-waf-logs-{cfg.name_prefix}-{which}"
    group = log_group(stack, f"{cid}Logs", cfg=cfg, key=key, name=name)
    F = waf.CfnLoggingConfiguration.FieldToMatchProperty
    redacted = [F(single_header={"Name": header}) for header in REDACTED_HEADERS]
    if redact_query_string:
        redacted.append(F(query_string={}))
    # WAF wants the log-group ARN without the trailing ":*" that LogGroup.log_group_arn has.
    arn = f"arn:{stack.partition}:logs:{stack.region}:{stack.account}:log-group:{name}"
    conf = waf.CfnLoggingConfiguration(stack, cid, resource_arn=acl.attr_arn,
                                       log_destination_configs=[arn], redacted_fields=redacted)
    conf.node.add_dependency(group)
    return group


class Alarms:
    """Creates alarms in one stack, all notifying the alarm topic."""

    def __init__(self, scope: Construct, topic: sns.ITopic, cfg: InfraConfig) -> None:
        self.scope, self.cfg = scope, cfg
        self.action = cw_actions.SnsAction(topic)
        self.namespace = f"OwlBank/{cfg.stage}/Security"

    def alarm(self, cid: str, metric: cw.IMetric, *, threshold: float, description: str,
              comparison: cw.ComparisonOperator = cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
              evaluation_periods: int = 1,
              missing: cw.TreatMissingData = cw.TreatMissingData.NOT_BREACHING) -> cw.Alarm:
        a = cw.Alarm(self.scope, cid, metric=metric, threshold=threshold,
                     evaluation_periods=evaluation_periods, comparison_operator=comparison,
                     treat_missing_data=missing, alarm_description=description)
        a.add_alarm_action(self.action)
        return a

    def reject_filter(self, cid: str, group: logs.ILogGroup, terms: tuple[str, ...], metric_name: str,
                      description: str, threshold: float = 10) -> cw.Alarm:
        """Metric filter on auth-failure log lines, and an alarm on its rate."""
        pattern = (logs.FilterPattern.literal(f'"{terms[0]}"') if len(terms) == 1
                   else logs.FilterPattern.any_term(*terms))
        mf = logs.MetricFilter(self.scope, f"{cid}Filter", log_group=group, filter_pattern=pattern,
                               metric_namespace=self.namespace, metric_name=metric_name,
                               metric_value="1", default_value=0)
        return self.alarm(f"{cid}Alarm", mf.metric(statistic="Sum", period=Duration.minutes(5)),
                          threshold=threshold, description=description)

    def count_filter(self, cid: str, group: logs.ILogGroup, term: str, metric_name: str) -> logs.MetricFilter:
        """A metric filter on a literal event name, with no alarm (a count to watch, not a page)."""
        return logs.MetricFilter(self.scope, f"{cid}Filter", log_group=group,
                                 filter_pattern=logs.FilterPattern.literal(f'"{term}"'),
                                 metric_namespace=self.namespace, metric_name=metric_name,
                                 metric_value="1", default_value=0)

    def error_filter(self, cid: str, group: logs.ILogGroup, metric_name: str, description: str,
                     threshold: float = APP_ERROR_THRESHOLD) -> cw.Alarm:
        """Metric filter on JSON log lines at ERROR or CRITICAL (``src/owlbank/obs.py`` writes the
        level as ``$.level``), and an alarm on their count per 5 minutes."""
        pattern = logs.FilterPattern.any(*(logs.FilterPattern.string_value("$.level", "=", lvl)
                                           for lvl in ERROR_LEVELS))
        mf = logs.MetricFilter(self.scope, f"{cid}Filter", log_group=group, filter_pattern=pattern,
                               metric_namespace=self.namespace, metric_name=metric_name,
                               metric_value="1", default_value=0)
        return self.alarm(f"{cid}Alarm", mf.metric(statistic="Sum", period=Duration.minutes(5)),
                          threshold=threshold, description=description)

    def slot_mode_invalid(self, group: logs.ILogGroup) -> cw.Alarm:
        """An invalid slot mode value fell back to ``deterministic`` (front door)."""
        return self.reject_filter(
            "SlotModeInvalid", group, (SLOT_MODE_INVALID_EVENT,), SLOT_MODE_INVALID_METRIC,
            "A slot mode value in the policy table is missing, unknown or of the wrong type; the "
            "front door uses deterministic for that slot (log event slot_mode_invalid names the item "
            "and field)", threshold=SLOT_MODE_INVALID_THRESHOLD)

    def runtime_alarms(self, *, runtime_name: str, runtime_arn: str, group: logs.ILogGroup,
                       session_quota: int) -> dict[str, cw.Alarm]:
        """``bridgeHosting=agentcore``: the bridge's
        log alarms on the Runtime's log group, and the ``AWS/Bedrock-AgentCore`` metric alarms
        that replace the ECS, ALB, CloudFront and WAF ones."""
        five = Duration.minutes(5)
        return {
            "auth": self.reject_filter(
                BRIDGE_AUTH_REJECTED_ID, group, BRIDGE_REJECT_MESSAGES, BRIDGE_REJECT_METRIC,
                "Bridge WebSocket auth failures on the AgentCore Runtime (signature, nonce, binding)"),
            "errors": self.error_filter(
                "RuntimeAppErrors", group, "RuntimeAppErrors",
                "Bridge ERROR or CRITICAL log lines on the AgentCore Runtime"),
            "throttles": self.alarm(
                "RuntimeThrottlesAlarm", runtime_metric("Throttles", runtime_name, runtime_arn, five),
                threshold=RUNTIME_THROTTLES_THRESHOLD,
                description="AgentCore Runtime requests throttled (TPS or quota, HTTP 429)"),
            "system_errors": self.alarm(
                "RuntimeSystemErrorsAlarm", runtime_metric("SystemErrors", runtime_name, runtime_arn, five),
                threshold=RUNTIME_SYSTEM_ERRORS_THRESHOLD,
                description="AgentCore Runtime server-side errors"),
            "sessions": self.alarm(
                "RuntimeActiveSessionsAlarm",
                cw.Metric(namespace=AGENTCORE_NAMESPACE, metric_name="ActiveSessionCount",
                          dimensions_map={"Service": "AgentCore.Runtime"}, statistic="Maximum", period=five),
                threshold=active_session_threshold(session_quota),
                description=(f"Active AgentCore Runtime sessions in the account at "
                             f"{int(ACTIVE_SESSION_ALARM_FRACTION * 100)} % of the quota ({session_quota})")),
        }

    def waf_blocked(self, cid: str, acl_name: str, region: str) -> cw.Alarm:
        """``region`` is the Region of a REGIONAL ACL, or ``Global`` for a CLOUDFRONT one."""
        metric = cw.Metric(namespace="AWS/WAFV2", metric_name="BlockedRequests",
                           dimensions_map={"WebACL": acl_name, "Region": region, "Rule": "ALL"},
                           statistic="Sum", period=Duration.minutes(5))
        return self.alarm(cid, metric, threshold=WAF_BLOCKED_THRESHOLD,
                          description=f"Spike in requests blocked by WAF web ACL {acl_name}")


# ---- Saved CloudWatch Logs Insights queries ------------------------------------------
#: Log groups that carry application (JSON) logs: the six Lambdas and the bridge. The names
#: are deterministic strings (see ``compute_stack`` and ``bridge_stack``), so the queries
#: need no cross-stack references. The API Gateway access-log group is left out on purpose:
#: its lines are access records, not application events, and have none of these fields.
LAMBDA_LOG_GROUP_NAMES = tuple(f"/owlbank/lambda/{n}" for n in (
    "frontdoor", "auth_service", "bank_tools", "level0_tools", "context_api", "flex_authorizer"))
BRIDGE_LOG_GROUP_NAME = "/owlbank/ecs/tac-bridge"
APP_LOG_GROUP_NAMES = LAMBDA_LOG_GROUP_NAMES + (BRIDGE_LOG_GROUP_NAME,)
#: ``bridgeHosting=agentcore`` only: the warmer Lambda (``compute_stack``'s seventh function).
RUNTIME_WARMER_LOG_GROUP_NAME = "/owlbank/lambda/runtime_warmer"


def agentcore_log_group_names(runtime_log_group_name: str) -> tuple[str, ...]:
    """The app log groups in ``agentcore``: the six Lambdas, the warmer and the Runtime's
    group in place of the ECS bridge's, so "Trace a call by call SID" shows a whole call."""
    return LAMBDA_LOG_GROUP_NAMES + (RUNTIME_WARMER_LOG_GROUP_NAME, runtime_log_group_name)

QUERY_NAME_PREFIX = "OwlBank/"
#: Default of the ``call_sid`` query parameter. Parameter values are substituted verbatim,
#: so the default carries its own quotes (``filter call_sid = {{call_sid}}``).
CALL_SID_PLACEHOLDER = '"CA00000000000000000000000000000000"'
AUTH_AUDIT_EVENTS = ("twilio_auth_failed", "verification_failed", "rate_limited",
                     "verification_risk", "otp_sent_untrusted_ani")


@dataclass(frozen=True)
class SavedQuery:
    name: str
    query: str
    #: ``{{parameters}}`` the query text uses, as (name, default value, description).
    parameters: tuple[tuple[str, str, str], ...] = ()


def _in_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(f'"{v}"' for v in values) + "]"


SAVED_QUERIES: tuple[SavedQuery, ...] = (
    SavedQuery(
        name=f"{QUERY_NAME_PREFIX}Trace a call by call SID",
        query=(
            "fields @timestamp, @log, component, event, level, op, outcome, duration_ms, @message\n"
            "| filter call_sid = {{call_sid}}\n"
            "| sort @timestamp asc\n"
            "| limit 10000"),
        parameters=(("call_sid", CALL_SID_PLACEHOLDER,
                     'The call SID to trace, in double quotes, e.g. "CA0123..." (32 hex after CA)'),)),
    SavedQuery(
        name=f"{QUERY_NAME_PREFIX}Errors by component",
        query=(
            "fields @timestamp, component, event, level, outcome\n"
            '| filter level in ["ERROR", "WARNING", "WARN", "CRITICAL"] or outcome = "error"\n'
            "| stats count(*) as occurrences by component, event\n"
            "| sort occurrences desc")),
    SavedQuery(
        name=f"{QUERY_NAME_PREFIX}Slowest invocations and dependency calls",
        query=(
            "fields @timestamp, component, event, op, dependency, operation, duration_ms, call_sid\n"
            '| filter event in ["invocation_end", "dependency_call"]\n'
            "| filter ispresent(duration_ms)\n"
            "| sort duration_ms desc\n"
            "| limit 50")),
    SavedQuery(
        name=f"{QUERY_NAME_PREFIX}Auth failures",
        query=(
            "fields @timestamp, event, component, reason\n"
            f"| filter event in {_in_list(AUTH_AUDIT_EVENTS)}\n"
            "| stats count(*) as occurrences by bin(5m), event, component, reason\n"
            "| sort occurrences desc")),
    SavedQuery(
        name=f"{QUERY_NAME_PREFIX}Escalations by reason",
        query=(
            "fields @timestamp, reason, source\n"
            '| filter event = "escalation"\n'
            "| stats count(*) as escalations by reason, source\n"
            "| sort escalations desc")),
)


def saved_queries(scope: Construct, *, log_group_names: tuple[str, ...] = APP_LOG_GROUP_NAMES) -> list[logs.CfnQueryDefinition]:
    """One ``AWS::Logs::QueryDefinition`` per entry of :data:`SAVED_QUERIES`.

    Instantiate once per account and Region. Uses log-group *names*, so there is no
    dependency on the stacks that own the groups. Parameters use the ``{{name}}`` syntax.
    """
    out = []
    for i, q in enumerate(SAVED_QUERIES):
        params = [logs.CfnQueryDefinition.QueryParameterProperty(name=n, default_value=d, description=desc)
                  for n, d, desc in q.parameters]
        out.append(logs.CfnQueryDefinition(
            scope, f"SavedQuery{i}", name=q.name, query_string=q.query,
            log_group_names=list(log_group_names), parameters=params or None))
    return out
