"""Call observability infrastructure.

Every function here is called by a stack only when ``cfg.call_observability_on``; with the switch
off nothing in this module runs, so every template stays as it was.

* ``OwlBankData``: with ``tracesEnableTransactionSearch=true``, :func:`transaction_search`.
* ``OwlBankBridge``: :func:`bridge_traces` (the traces log group's own CMK ``TracesKey``, the log
  group, its ``spans`` stream, the X-Ray resource policy, the ``TraceReader`` role and the bridge
  role's X-Ray grant; all in one stack, so no cross-stack export of the key) and
  :func:`bridge_env` (the container variables), in both hostings: under the AgentCore hosting option
  (``bridgeHosting=agentcore``) the grant goes on the Runtime's execution role and the spans go
  to the same traces group. :func:`bridge_star_reason` is the Fargate task role's IAM5
  acknowledgement with the X-Ray actions named; the Runtime's is ``runtime.star_reason``.
* ``OwlBankBridge`` (both hostings): :class:`CallAlarms` (metric filters on the bridge's ``turn``,
  ``tool_result`` and ``ws_session_end`` lines, anomaly-detection and fixed-ceiling alarms, error-rate alarms)
  and :func:`call_timeline_query`.
* ``OwlBankCompute``: :func:`observability_layer` (OpenTelemetry in its own layer, on the front
  door only), :func:`frontdoor_env` and :func:`frontdoor_filters` (the ``factor_answer`` timings).

Exporter settings travel as ``OWLBANK_*`` variables, never ``OTEL_EXPORTER_*``: the bridge builds
its exporter in code and refuses any other OTLP endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aws_cdk import CfnResource, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_xray as xray
from constructs import Construct

from .config import ConfigError, InfraConfig
from .grants import PII_DETECT_REASON
from .nag import ack
from .observability import CALL_SID_PLACEHOLDER, QUERY_NAME_PREFIX, Alarms, log_settings

INFRA_DIR = Path(__file__).resolve().parents[1]
#: Built by ``infra/scripts/build_lambda_deps.py`` (``LAYERS["lambda-otel"]``); empty until then.
LAMBDA_OTEL_DIR = INFRA_DIR / "build" / "lambda-otel"

TRACES_KEY_ALIAS = "alias/owlbank-traces"
#: The bridge's traces log group. Under ``/owlbank/``, so :func:`observability.log_group`'s
#: prefix check already allows it; it is encrypted with the TracesKey, not the DataKey.
TRACES_LOG_GROUP_NAME = "/owlbank/traces/bridge"
#: X-Ray can't create this stream under a ``PutLogEvents``-only policy (found by testing), so the stack does.
TRACES_STREAM_NAME = "spans"
#: Transaction Search's groups: X-Ray's default span group and Application Signals' group.
TRANSACTION_SEARCH_LOG_GROUPS = ("aws/spans", "/aws/application-signals/data")
#: CloudWatch Logs accepts only these retention values up to 30 days.
RETENTION_BY_DAYS = {1: logs.RetentionDays.ONE_DAY, 3: logs.RetentionDays.THREE_DAYS,
                     5: logs.RetentionDays.FIVE_DAYS, 7: logs.RetentionDays.ONE_WEEK,
                     14: logs.RetentionDays.TWO_WEEKS, 30: logs.RetentionDays.ONE_MONTH}

#: The X-Ray OTLP write action the bridge's exporter needs. Each was tested with a
#: temporary role: the OTLP traces endpoint authorizes ``xray:PutTraceSegments`` alone (``PutSpans``
#: alone is refused, naming ``PutTraceSegments``). It has no resource-level permissions.
XRAY_WRITE_ACTIONS = ("xray:PutTraceSegments",)
XRAY_OTLP_REASON = ("X-Ray OTLP span export (call observability): xray:PutTraceSegments "
                    "doesn't support resource-level permissions, so its resource must be '*'.")
#: What the TraceReader role may do on the traces log group.
TRACE_READER_LOG_ACTIONS = ("logs:StartQuery", "logs:FilterLogEvents", "logs:GetLogEvents",
                            "logs:DescribeLogStreams")
#: Query follow-ups: they take a query ID and have no resource-level permissions.
TRACE_READER_QUERY_ACTIONS = ("logs:GetQueryResults", "logs:StopQuery")
TRACE_READER_STAR_REASON = ("logs:GetQueryResults and logs:StopQuery take only a query ID and have no "
                            "resource-level permissions, so their resource must be '*'. The queries "
                            "themselves can only be started on the traces log group (logs:StartQuery "
                            "is scoped to it).")


def log_group_arn(stack: Stack, name: str) -> str:
    """A log group's ARN without the trailing ``:*`` (the form KMS's encryption context uses)."""
    return f"arn:{stack.partition}:logs:{stack.region}:{stack.account}:log-group:{name}"


def xray_source_conditions(stack: Stack) -> dict:
    """Confused-deputy conditions for X-Ray writing into this account's log groups."""
    return {"StringEquals": {"aws:SourceAccount": stack.account},
            "ArnLike": {"aws:SourceArn": f"arn:{stack.partition}:xray:{stack.region}:{stack.account}:*"}}


def xray_put_log_events_policy(stack: Stack, scope: Construct, cid: str, *, policy_name: str,
                               group_names: tuple[str, ...], sid: str) -> logs.CfnResourcePolicy:
    """A CloudWatch Logs resource policy: ``xray.amazonaws.com`` may ``logs:PutLogEvents`` into the
    named groups only, from this account's X-Ray. The principal is the service, never ``*``."""
    doc = {"Version": "2012-10-17", "Statement": [{
        "Sid": sid, "Effect": "Allow",
        "Principal": {"Service": "xray.amazonaws.com"},
        "Action": "logs:PutLogEvents",
        "Resource": [f"{log_group_arn(stack, name)}:*" for name in group_names],
        "Condition": xray_source_conditions(stack),
    }]}
    return logs.CfnResourcePolicy(scope, cid, policy_name=policy_name,
                                  policy_document=stack.to_json_string(doc))


# --------------------------------------------------------------------------- OwlBankData

def transaction_search(stack: Stack, cfg: InfraConfig) -> CfnResource:
    """``tracesEnableTransactionSearch=true``: the resource policy X-Ray needs on
    ``aws/spans`` and Application Signals' group (the preflight's ``transaction_search_policy``),
    then ``AWS::XRay::TransactionSearchConfig``. Account-wide in effect (the guide says so)."""
    policy = xray_put_log_events_policy(
        stack, stack, "TransactionSearchPolicy", policy_name=f"{cfg.name_prefix}-transaction-search",
        group_names=TRANSACTION_SEARCH_LOG_GROUPS, sid="TransactionSearchXRayAccess")
    config = xray.CfnTransactionSearchConfig(stack, "TransactionSearchConfig")
    config.add_resource_dependency(policy)
    return config


# --------------------------------------------------------------------------- OwlBankBridge

def traces_retention(cfg: InfraConfig) -> tuple[logs.RetentionDays, RemovalPolicy]:
    """``traceContentRetentionDays`` while text is exported, else the stage's log retention.
    Removal follows the stage (retained at prod)."""
    retention, removal = log_settings(cfg)
    if cfg.trace_content == "off":
        return retention, removal
    days = cfg.trace_content_retention_days
    if days not in RETENTION_BY_DAYS:
        raise ConfigError(f"traceContentRetentionDays={days} is not a CloudWatch Logs retention; "
                          f"use one of {', '.join(map(str, RETENTION_BY_DAYS))}")
    return RETENTION_BY_DAYS[days], removal


def bridge_star_reason(cfg: InfraConfig, base_reason: str) -> str:
    """The bridge role's one ``AwsSolutions-IAM5[Resource::*]`` reason: ``base_reason`` as is when
    off; when on, it also names the X-Ray actions of :func:`bridge_traces` (one ack per rule)."""
    if not cfg.call_observability_on:
        return base_reason
    return PII_DETECT_REASON + " Also " + XRAY_OTLP_REASON + " Nothing else uses '*'."


def traces_key(stack: Stack, cfg: InfraConfig) -> kms.Key:
    """``TracesKey``: the traces log group's own CMK, rotation on. CloudWatch Logs may use
    it only for the traces log group (the encryption-context condition). Created by
    :func:`bridge_traces` in the stack that owns that group (never exported: a cross-stack import
    would block switching ``callObservability`` off)."""
    key = kms.Key(
        stack, "TracesKey",
        alias=TRACES_KEY_ALIAS,
        description="OwlBank call observability: the bridge's traces log group (spans)",
        enable_key_rotation=True,
        removal_policy=RemovalPolicy.RETAIN if cfg.is_prod else RemovalPolicy.DESTROY,
    )
    key.add_to_resource_policy(iam.PolicyStatement(
        sid="AllowCloudWatchLogsTracesGroup",
        principals=[iam.ServicePrincipal(f"logs.{stack.region}.amazonaws.com")],
        actions=["kms:Encrypt*", "kms:Decrypt*", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:Describe*"],
        resources=["*"],  # in a key policy, "*" means this key
        conditions={"ArnEquals": {"kms:EncryptionContext:aws:logs:arn":
                                  log_group_arn(stack, TRACES_LOG_GROUP_NAME)}},
    ))
    return key


def bridge_traces(stack: Stack, cfg: InfraConfig, *, task_role: iam.IRole) -> logs.LogGroup:
    """The bridge's traces log group and what goes with it:

    * ``TracesKey`` (:func:`traces_key`), in this same stack;
    * ``/owlbank/traces/bridge``, encrypted with TracesKey, retention per
      :func:`traces_retention`;
    * its ``spans`` stream;
    * a resource policy: ``xray.amazonaws.com`` may ``logs:PutLogEvents`` into that group only,
      conditioned on ``aws:SourceAccount`` and ``aws:SourceArn``;
    * ``TraceReader`` (``owlbank-<stage>-trace-reader``), assumable from the account: read the
      traces group and decrypt with TracesKey, nothing else;
    * the task role's X-Ray write grant (resource ``*``; its IAM5 reason is
      :func:`bridge_star_reason`).
    """
    key = traces_key(stack, cfg)
    retention, removal = traces_retention(cfg)
    group = logs.LogGroup(stack, "TracesLogGroup", log_group_name=TRACES_LOG_GROUP_NAME,
                          encryption_key=key, retention=retention, removal_policy=removal)
    stream = logs.CfnLogStream(stack, "TracesSpansStream", log_group_name=TRACES_LOG_GROUP_NAME,
                               log_stream_name=TRACES_STREAM_NAME)
    stream.add_resource_dependency(group.node.default_child)
    stream.apply_removal_policy(removal)
    policy = xray_put_log_events_policy(
        stack, stack, "TracesXRayPolicy", policy_name=f"{cfg.name_prefix}-traces-xray",
        group_names=(TRACES_LOG_GROUP_NAME,), sid="XRayPutSpansToTracesGroup")
    policy.add_resource_dependency(group.node.default_child)

    group_arn = f"{log_group_arn(stack, TRACES_LOG_GROUP_NAME)}:*"
    reader = iam.Role(stack, "TraceReader", role_name=f"{cfg.name_prefix}-trace-reader",
                      description="OwlBank call observability: read the bridge's traces log group",
                      assumed_by=iam.AccountRootPrincipal())
    reader.add_to_policy(iam.PolicyStatement(
        sid="ReadTracesGroup", actions=list(TRACE_READER_LOG_ACTIONS), resources=[group_arn]))
    reader.add_to_policy(iam.PolicyStatement(
        sid="TracesQueryResults", actions=list(TRACE_READER_QUERY_ACTIONS), resources=["*"]))
    reader.add_to_policy(iam.PolicyStatement(
        sid="DecryptTraces", actions=["kms:Decrypt"], resources=[key.key_arn]))
    ack(reader, "AwsSolutions-IAM5[Resource::*]", TRACE_READER_STAR_REASON)
    # cdk-nag prints the partition pseudo-parameter as <AWS::Partition>; region and account are the
    # stack's literal env values.
    ack(reader, f"AwsSolutions-IAM5[Resource::arn:<AWS::Partition>:logs:{stack.region}:{stack.account}"
                f":log-group:{TRACES_LOG_GROUP_NAME}:*]",
        "Log stream names under the traces log group are created at runtime, so reading them needs "
        "the group's ARN with the trailing ':*' (that one group only).")

    task_role.add_to_principal_policy(iam.PolicyStatement(
        sid="CallObservabilityXRay", actions=list(XRAY_WRITE_ACTIONS), resources=["*"]))
    return group


def bridge_env(cfg: InfraConfig, *, traces_log_group: str | None) -> dict[str, str]:
    """The bridge's call-observability variables, appended to ``bridge_base_env`` when on.
    ``traces_log_group`` is :data:`TRACES_LOG_GROUP_NAME` in both hostings. Never an
    ``OTEL_EXPORTER_*`` variable."""
    env = {
        "OWLBANK_CALL_OBSERVABILITY": "on",
        "OWLBANK_STAGE": cfg.stage,
        "OWLBANK_TRACE_SAMPLE_RATE": str(cfg.trace_sample_rate),
        "OWLBANK_TRACE_CONTENT": cfg.trace_content,
        **({"OWLBANK_TRACES_LOG_GROUP": traces_log_group} if traces_log_group else {}),
        # Botocore instrumentation: no message content on the Bedrock CLIENT spans.
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "false",
    }
    if cfg.trace_content == "off":
        # Strands' own redaction at the source (an empty unredacted list).
        env["OTEL_SEMCONV_STABILITY_OPT_IN"] = "gen_ai_unredacted_attributes="
    return env


# --------------------------------------------------------------------------- OwlBankCompute

def observability_layer(stack: Stack) -> lambda_.LayerVersion:
    """The front door's OpenTelemetry packages in their own layer (empty until
    ``build_lambda_deps.py`` builds ``lambda-otel``), so the shared deps layer is unchanged."""
    LAMBDA_OTEL_DIR.joinpath("python").mkdir(parents=True, exist_ok=True)
    return lambda_.LayerVersion(
        stack, "ObservabilityLayer",
        code=lambda_.Code.from_asset(str(LAMBDA_OTEL_DIR)),
        compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
        compatible_architectures=[lambda_.Architecture.ARM_64],
        description="OpenTelemetry for the OwlBank front door (call observability; built by "
                    "infra/scripts/build_lambda_deps.py)",
    )


def frontdoor_env(cfg: InfraConfig) -> dict[str, str]:
    """The front door's call-observability variables. Spans go to the Lambda X-Ray daemon over
    loopback UDP, which active tracing already permits: no new IAM."""
    return {
        "OWLBANK_CALL_OBSERVABILITY": "on",
        "OWLBANK_STAGE": cfg.stage,
        "OWLBANK_TRACE_SAMPLE_RATE": str(cfg.trace_sample_rate),
    }


# --------------------------------------------------------------------------- metrics, alarms, query
#
# Two sources:
#
# * Strands' own metrics, written by the bridge as EMF lines on stdout into its log group:
#   namespace ``bedrock-agentcore``, dimensions ``Service`` (``owlbank-<stage>-bridge``) and
#   ``Environment`` (the stage), plus ``tool_name`` on the tool metrics only. Histograms are
#   CloudWatch statistic sets (Count, Sum, Min, Max), so the alarms use ``Average`` or ``Maximum``;
#   percentiles need raw values and don't exist for these metrics. Names and units are Strands
#   1.61's (``strands/telemetry/metrics_constants.py`` and ``metrics.py``).
# * Metric filters for what Strands doesn't measure, in ``OwlBank/<stage>/Calls``, with no
#   dimensions (so no caller data can become one). Their values are raw, so p90 works.
#
# Per-tool anomaly detection ("tool duration (per tool)") would need one detector per tool name,
# known at synth, or a SEARCH expression (which an anomaly detector can't take). The stack watches
# the aggregate over all tools instead (the ``[Service, Environment]`` dimension set the bridge also
# writes for tool metrics): anomaly on its Average, and a ceiling on its Maximum, which fires for
# any single tool call slower than the ceiling whichever tool it was. The per-tool series stay in
# CloudWatch for diagnosis.

EMF_NAMESPACE = "bedrock-agentcore"  # src/owlbank/telemetry/bridge.py METRICS_NAMESPACE
#: Strands 1.61 metric names (histograms unless noted) and their units.
TTFT_METRIC = "strands.model.time_to_first_token"  # ms
MODEL_LATENCY_METRIC = "strands.event_loop.latency"  # ms, the model call's latency per loop cycle
TOOL_DURATION_METRIC = "strands.tool.duration"  # s (Strands records seconds)
#: The bridge's EMF dimension keys; every EMF-based alarm uses a subset of them.
#: ``model_id`` is on the TTFT and latency histograms when the model list has two or more entries
#:; the alarms keep the ``[Service, Environment]`` aggregate.
EMF_DIMENSION_KEYS = frozenset({"Service", "Environment", "tool_name", "model_id"})

CALL_METRIC_NAMESPACE_FORMAT = "OwlBank/{stage}/Calls"
FRONTDOOR_LOG_GROUP_NAME = "/owlbank/lambda/frontdoor"

#: Anomaly detection band width (standard deviations) and the shape of every anomaly and ceiling
#: alarm: 3 periods of 5 minutes in a row, missing data not breaching.
ANOMALY_BAND_STDDEVS = 2
ALARM_PERIOD = Duration.minutes(5)
ALARM_EVALUATION_PERIODS = 3
#: Error rate above 5 % of turns or of tool calls in 15 minutes.
ERROR_RATE_PERCENT = 5
ERROR_RATE_PERIOD = Duration.minutes(15)


@dataclass(frozen=True)
class CallFilter:
    """One metric filter on a JSON log line: ``event`` = ``event``, plus ``field`` >= 0 when the
    value is a field (a missing or null field never matches), else a count of 1 per line.
    ``equals`` and ``not_equals`` add string conditions on other fields."""
    cid: str
    metric_name: str
    event: str
    field: str | None = None  # JSON path under ``$.``; None counts lines
    unit: cw.Unit = cw.Unit.MILLISECONDS
    equals: tuple[tuple[str, str], ...] = ()  # extra string conditions
    not_equals: tuple[tuple[str, str], ...] = ()

    @property
    def value(self) -> str:
        return f"$.{self.field}" if self.field else "1"

    def pattern(self) -> logs.IFilterPattern:
        terms = [logs.FilterPattern.string_value("$.event", "=", self.event)]
        terms += [logs.FilterPattern.string_value(f"$.{k}", "=", v) for k, v in self.equals]
        terms += [logs.FilterPattern.string_value(f"$.{k}", "!=", v) for k, v in self.not_equals]
        if self.field:
            terms.append(logs.FilterPattern.number_value(f"$.{self.field}", ">=", 0))
        return logs.FilterPattern.all(*terms)


#: The bridge's ``turn`` and ``ws_session_end`` lines; and its ``tool_result`` lines (one per
#: inline or Gateway tool call that reaches an outcome; a cancelled call is in neither count).
BRIDGE_FILTERS = (
    CallFilter("CallObsTurnSilence", "TurnSilenceMs", "turn", "silence_ms"),
    CallFilter("CallObsTurnWait", "TurnWaitMs", "turn", "wait_ms"),
    CallFilter("CallObsTurns", "Turns", "turn", unit=cw.Unit.COUNT),
    CallFilter("CallObsTurnErrors", "TurnErrors", "turn", unit=cw.Unit.COUNT, equals=(("outcome", "error"),)),
    CallFilter("CallObsCallInputTokens", "CallInputTokens", "ws_session_end", "input_tokens", unit=cw.Unit.COUNT),
    CallFilter("CallObsCallOutputTokens", "CallOutputTokens", "ws_session_end", "output_tokens",
               unit=cw.Unit.COUNT),
    CallFilter("CallObsToolCalls", "ToolCalls", "tool_result", unit=cw.Unit.COUNT,
               not_equals=(("outcome", "cancelled"),)),
    CallFilter("CallObsToolErrors", "ToolErrors", "tool_result", unit=cw.Unit.COUNT,
               equals=(("outcome", "error"),)),
)
#: The front door's ``factor_answer`` step timings.
FRONTDOOR_FILTERS = (
    CallFilter("CallObsFrontdoorResolve", "FrontdoorResolveMs", "factor_answer", "timings_ms.resolve"),
    CallFilter("CallObsFrontdoorStateLoad", "FrontdoorStateLoadMs", "factor_answer", "timings_ms.state_load"),
    CallFilter("CallObsFrontdoorStateSave", "FrontdoorStateSaveMs", "factor_answer", "timings_ms.state_save"),
)
#: Count filters publish 0 in a period with log lines but no match, so the error rate has a
#: denominator and a zero numerator on a quiet, healthy period. Value filters publish nothing then.
COUNT_METRICS = frozenset({"Turns", "TurnErrors", "ToolCalls", "ToolErrors"})


def call_metric_namespace(cfg: InfraConfig) -> str:
    return CALL_METRIC_NAMESPACE_FORMAT.format(stage=cfg.stage)


def bridge_service_name(cfg: InfraConfig) -> str:
    """The bridge's ``service.name`` and EMF ``Service`` dimension (``owlbank-<stage>-bridge``)."""
    return f"{cfg.name_prefix}-bridge"


def call_filters(scope: Construct, cfg: InfraConfig, group: logs.ILogGroup,
                 filters: tuple[CallFilter, ...]) -> dict[str, cw.Metric]:
    """The metric filters in ``filters`` on ``group``; returns each metric (no statistic set)."""
    out = {}
    for f in filters:
        mf = logs.MetricFilter(scope, f"{f.cid}Filter", log_group=group, filter_pattern=f.pattern(),
                               metric_namespace=call_metric_namespace(cfg), metric_name=f.metric_name,
                               metric_value=f.value, unit=f.unit,
                               default_value=0 if f.metric_name in COUNT_METRICS else None)
        out[f.metric_name] = mf.metric(period=ALARM_PERIOD)
    return out


def frontdoor_filters(scope: Construct, cfg: InfraConfig, group: logs.ILogGroup) -> dict[str, cw.Metric]:
    """``OwlBankCompute``: the front-door step timings. No alarm: they
    are for dashboards and diagnosis."""
    return call_filters(scope, cfg, group, FRONTDOOR_FILTERS)


def emf_dimensions(cfg: InfraConfig) -> dict[str, str]:
    """The ``[Service, Environment]`` dimension set (the aggregate over tools for tool metrics)."""
    return {"Service": bridge_service_name(cfg), "Environment": cfg.stage}


@dataclass(frozen=True)
class _Watched:
    """One metric under anomaly detection plus a fixed ceiling."""
    cid: str
    label: str
    anomaly_stat: str
    ceiling_stat: str
    ceiling: float
    ceiling_text: str


class CallAlarms:
    """The alarms in ``OwlBankBridge``: anomaly-detection alarms (``AWS::CloudWatch::AnomalyDetector``
    plus an alarm on ``ANOMALY_DETECTION_BAND``), their fixed-ceiling backstops and the error-rate
    alarms, all notifying the alarm topic; missing data is not breaching."""

    def __init__(self, scope: Construct, cfg: InfraConfig, topic: sns.ITopic) -> None:
        self.scope, self.cfg, self.topic = scope, cfg, topic
        self.alarms = Alarms(scope, topic, cfg)

    # ---- metric data queries (raw CloudFormation shapes, shared by detector and alarm)

    @staticmethod
    def _stat_query(qid: str, namespace: str, metric_name: str, dims: dict[str, str], stat: str,
                    *, return_data: bool, cls=cw.CfnAlarm) -> object:
        metric = cls.MetricProperty(namespace=namespace, metric_name=metric_name,
                                    dimensions=[cls.DimensionProperty(name=k, value=v) for k, v in dims.items()]
                                    or None)
        return cls.MetricDataQueryProperty(
            id=qid, return_data=return_data,
            metric_stat=cls.MetricStatProperty(metric=metric, period=ALARM_PERIOD.to_seconds(), stat=stat))

    def _anomaly(self, w: _Watched, queries: list[tuple[str, str, dict[str, str]]],
                 expression: str | None = None) -> cw.CfnAlarm:
        """Anomaly detector and alarm on one metric (``queries`` has one entry and ``expression`` is
        None) or on a metric-math ``expression`` over ``queries`` (namespace, name, dims each)."""
        ids = [f"m{i + 1}" for i in range(len(queries))]
        single = expression is None
        if single:
            ((ns, name, dims),) = queries
            detector = cw.CfnAnomalyDetector(
                self.scope, f"{w.cid}AnomalyDetector",
                single_metric_anomaly_detector=cw.CfnAnomalyDetector.SingleMetricAnomalyDetectorProperty(
                    namespace=ns, metric_name=name, stat=w.anomaly_stat,
                    dimensions=[cw.CfnAnomalyDetector.DimensionProperty(name=k, value=v)
                                for k, v in dims.items()] or None))
        else:
            detector = cw.CfnAnomalyDetector(
                self.scope, f"{w.cid}AnomalyDetector",
                metric_math_anomaly_detector=cw.CfnAnomalyDetector.MetricMathAnomalyDetectorProperty(
                    metric_data_queries=[
                        *(self._stat_query(i, ns, name, dims, w.anomaly_stat, return_data=False,
                                           cls=cw.CfnAnomalyDetector)
                          for i, (ns, name, dims) in zip(ids, queries, strict=True)),
                        cw.CfnAnomalyDetector.MetricDataQueryProperty(id="e1", expression=expression,
                                                                      return_data=True)]))
        watched = "m1" if single else "e1"
        metrics = [self._stat_query(i, ns, name, dims, w.anomaly_stat, return_data=single)
                   for i, (ns, name, dims) in zip(ids, queries, strict=True)]
        if not single:
            metrics.append(cw.CfnAlarm.MetricDataQueryProperty(id="e1", expression=expression, label=w.label,
                                                               return_data=True))
        metrics.append(cw.CfnAlarm.MetricDataQueryProperty(
            id="ad1", expression=f"ANOMALY_DETECTION_BAND({watched}, {ANOMALY_BAND_STDDEVS})",
            label=f"{w.label} (expected band)", return_data=True))
        alarm = cw.CfnAlarm(
            self.scope, f"{w.cid}AnomalyAlarm",
            alarm_description=(f"{w.label} ({w.anomaly_stat}) above its learned band for "
                               f"{ALARM_EVALUATION_PERIODS * 5} minutes (anomaly detection, 7.4)"),
            comparison_operator="GreaterThanUpperThreshold", threshold_metric_id="ad1",
            evaluation_periods=ALARM_EVALUATION_PERIODS, datapoints_to_alarm=ALARM_EVALUATION_PERIODS,
            treat_missing_data="notBreaching", metrics=metrics, alarm_actions=[self.topic.topic_arn])
        alarm.add_resource_dependency(detector)
        return alarm

    def _ceiling(self, w: _Watched, metric: cw.IMetric, *, evaluation_periods: int = ALARM_EVALUATION_PERIODS) -> cw.Alarm:
        return self.alarms.alarm(
            f"{w.cid}CeilingAlarm", metric, threshold=w.ceiling,
            comparison=cw.ComparisonOperator.GREATER_THAN_THRESHOLD, evaluation_periods=evaluation_periods,
            description=f"{w.label} ({w.ceiling_stat}) above the fixed ceiling {w.ceiling_text} (7.4 backstop)")

    def _emf(self, metric_name: str, stat: str) -> cw.Metric:
        return cw.Metric(namespace=EMF_NAMESPACE, metric_name=metric_name, statistic=stat,
                         period=ALARM_PERIOD, dimensions_map=emf_dimensions(self.cfg))

    @staticmethod
    def _tool_error_rate(m: dict[str, cw.Metric]) -> cw.MathExpression:
        """The error rate over the tools the agent runs (inline and Gateway),
        from the bridge's own ``tool_result`` lines, not Strands' ``strands.tool.error_count`` (which
        counts every interrupt, so every inline tool call and every hand-off, as an error). Only an
        ``error`` outcome counts as an error; a cancelled call (hang-up, barge-in) is in neither
        count; hand-offs log no ``tool_result``. No calls in the period gives 0 %, not a division by
        zero."""
        return cw.MathExpression(
            expression="IF(calls > 0, 100 * errors / calls, 0)", label="Tool error rate (%)",
            period=ERROR_RATE_PERIOD,
            using_metrics={"errors": m["ToolErrors"].with_(statistic="Sum", period=ERROR_RATE_PERIOD),
                           "calls": m["ToolCalls"].with_(statistic="Sum", period=ERROR_RATE_PERIOD)})

    # ---- the alarms

    def build(self, bridge_group: logs.ILogGroup) -> dict[str, object]:
        cfg = self.cfg
        m = call_filters(self.scope, cfg, bridge_group, BRIDGE_FILTERS)
        ns, dims = call_metric_namespace(cfg), emf_dimensions(cfg)
        wait_ms, silence_ms = cfg.turn_wait_ceiling_seconds * 1000, cfg.silence_ceiling_seconds * 1000
        out: dict[str, object] = {}

        emf = (
            # metric, cid, label, anomaly stat, ceiling stat, ceiling (metric's unit), text
            (TTFT_METRIC, "CallObsTtft", "Model time to first token", "Average", "Average",
             cfg.ttft_ceiling_seconds * 1000, f"{cfg.ttft_ceiling_seconds:g} s (ttftCeilingSeconds)"),
            # A single model call slower than the whole turn's wait ceiling is too slow.
            (MODEL_LATENCY_METRIC, "CallObsModelLatency", "Model latency per loop cycle", "Average", "Average",
             wait_ms, f"{cfg.turn_wait_ceiling_seconds:g} s (turnWaitCeilingSeconds)"),
            # Seconds (Strands' unit). Maximum over all tools: any one tool call over the ceiling.
            (TOOL_DURATION_METRIC, "CallObsToolDuration", "Tool duration (all tools)", "Average", "Maximum",
             cfg.turn_wait_ceiling_seconds, f"{cfg.turn_wait_ceiling_seconds:g} s (turnWaitCeilingSeconds)"),
        )
        for metric_name, cid, label, a_stat, c_stat, ceiling, text in emf:
            w = _Watched(cid, label, a_stat, c_stat, ceiling, text)
            out[f"{cid}Anomaly"] = self._anomaly(w, [(EMF_NAMESPACE, metric_name, dims)])
            out[f"{cid}Ceiling"] = self._ceiling(w, self._emf(metric_name, c_stat))

        logged = (
            ("TurnWaitMs", "CallObsTurnWait", "Caller's wait per turn", wait_ms,
             f"{cfg.turn_wait_ceiling_seconds:g} s (turnWaitCeilingSeconds)"),
            ("TurnSilenceMs", "CallObsTurnSilence", "Silence after the caller stops speaking", silence_ms,
             f"{cfg.silence_ceiling_seconds:g} s (silenceCeilingSeconds)"),
        )
        for metric_name, cid, label, ceiling, text in logged:
            w = _Watched(cid, label, "p90", "p90", ceiling, text)
            out[f"{cid}Anomaly"] = self._anomaly(w, [(ns, metric_name, {})])
            out[f"{cid}Ceiling"] = self._ceiling(w, m[metric_name].with_(statistic="p90"))

        # Tokens per call: input + output from the same ws_session_end line. Anomaly on the sum of
        # the Averages (the average call's total). Ceiling on the sum of the Maximums: any call at
        # or over the ceiling fires; two different calls can together exceed it (an upper bound).
        tokens = _Watched("CallObsTokensPerCall", "Tokens per call", "Average", "Maximum",
                          cfg.tokens_per_call_ceiling, f"{cfg.tokens_per_call_ceiling} (tokensPerCallCeiling)")
        out["CallObsTokensPerCallAnomaly"] = self._anomaly(
            tokens, [(ns, "CallInputTokens", {}), (ns, "CallOutputTokens", {})], expression="m1 + m2")
        out["CallObsTokensPerCallCeiling"] = self._ceiling(tokens, cw.MathExpression(
            expression="m1 + m2", label="Tokens per call (max)", period=ALARM_PERIOD,
            using_metrics={"m1": m["CallInputTokens"].with_(statistic="Maximum"),
                           "m2": m["CallOutputTokens"].with_(statistic="Maximum")}), evaluation_periods=1)

        # Error rates over 15 minutes (no data, or no turns: not breaching).
        rate = ERROR_RATE_PERCENT
        out["CallObsTurnErrorRate"] = self.alarms.alarm(
            "CallObsTurnErrorRateAlarm", cw.MathExpression(
                expression="100 * errors / turns", label="Turn error rate (%)", period=ERROR_RATE_PERIOD,
                using_metrics={"errors": m["TurnErrors"].with_(statistic="Sum", period=ERROR_RATE_PERIOD),
                               "turns": m["Turns"].with_(statistic="Sum", period=ERROR_RATE_PERIOD)}),
            threshold=rate, comparison=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            description=f"More than {rate} % of turns ended in error in 15 minutes")
        out["CallObsToolErrorRate"] = self.alarms.alarm(
            "CallObsToolErrorRateAlarm", self._tool_error_rate(m),
            threshold=rate, comparison=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            description=f"More than {rate} % of calls to tools the agent runs failed in 15 minutes")
        return out


# ---- The call timeline query

CALL_TIMELINE_QUERY_NAME = f"{QUERY_NAME_PREFIX}Call timeline"
#: X-Ray's default span group (front door, Lambda, Gateway, Memory spans).
AWS_SPANS_LOG_GROUP = "aws/spans"
CALL_TIMELINE_QUERY = (
    "# Every span of one call (session.id = CallSid) in start order, from both span groups, with\n"
    "# each span's start and the previous span's end (prev_end_ms). Logs Insights can't compute on\n"
    "# autoregress output, so the gaps between spans, labelled as Twilio-side time (speech\n"
    "# recognition, playback, waiting for the caller), are printed by scripts/call_timeline.py.\n"
    "fields @timestamp, name, `resource.attributes.service.name` as service,\n"
    "  durationNano / 1000000 as duration_ms, startTimeUnixNano / 1000000 as start_ms,\n"
    "  endTimeUnixNano / 1000000 as end_ms, traceId, spanId, parentSpanId, @log\n"
    "| filter `attributes.session.id` = {{session_id}}\n"
    "| sort start_ms asc\n"
    "| autoregress end_ms as prev_end_ms\n"
    "| limit 10000")


def call_timeline_query(scope: Construct, *, traces_log_group_name: str) -> logs.CfnQueryDefinition:
    """"OwlBank/Call timeline" over the traces log group and ``aws/spans``, by ``session.id``."""
    return logs.CfnQueryDefinition(
        scope, "CallTimelineQuery", name=CALL_TIMELINE_QUERY_NAME, query_string=CALL_TIMELINE_QUERY,
        log_group_names=[traces_log_group_name, AWS_SPANS_LOG_GROUP],
        parameters=[logs.CfnQueryDefinition.QueryParameterProperty(
            name="session_id", default_value=CALL_SID_PLACEHOLDER,
            description='The call SID (session.id), in double quotes, e.g. "CA0123..." (32 hex after CA)')])
