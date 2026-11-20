"""Model failover's fleet picture in infra.

Everything here except the table is built only when :attr:`InfraConfig.failover_on`
(``modelFailover=on`` and two or more ``modelIds`` entries). With ``modelFailover=off`` or a list of
one, the Compute and Bridge templates are today's single-model shape and the Data template is
today's plus the table.

* ``OwlBankData``: the ``modelstats`` table (:func:`build_table`), always built: standard, on demand,
  the stack's CMK, TTL ``expires_at``, no Streams, no point-in-time recovery, no index, retained and
  deletion-protected at prod like the other tables; and, only when on, the consumed-write
  alarm against twice the sampling budget.
* ``OwlBankCompute``: the poller Lambda (:data:`POLLER`) on a one-minute schedule, its least-privilege
  role (:func:`grant_poller`), the heartbeat-staleness and every-entry-down alarms.
* ``OwlBankBridge`` (both hostings): the table name (:func:`bridge_env`), the bridge role's two item
  grants (:func:`grant_bridge`), the ``model_unavailable`` and ``ActiveRank`` alarms.

The table has a fixed name (:attr:`InfraConfig.failover_table_name`) and is named by ARN, as a
string, in Compute and Bridge: no CloudFormation export, so turning the feature off never has to
remove an export another stack still imports.

Key layout (``owlbank.failover.fleet``): samples in the fixed partition ``s = samples``,
the verdict in ``s = state``; not the stage, as each stage has its own table, so the bridge needs no
stage setting. Every DynamoDB statement names the one table ARN (no ``/index``) and carries a
``dynamodb:LeadingKeys`` condition on exactly one of those two partitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

from .contract import FAILOVER_TABLES, TTL_ATTRIBUTE
from .nag import ack

if TYPE_CHECKING:
    from .config import InfraConfig
    from .observability import Alarms

#: The table's logical name (``owlbank.store.FAILOVER_SCHEMAS``, ``contract.FAILOVER_TABLES``).
TABLE = "modelstats"
#: The table-name variable, the same convention as every other table (``owlbank.config``).
TABLE_ENV = f"OWLBANK_TABLE_{TABLE.upper()}"
#: The stage, on the poller only, for its EMF metric names (not a key).
STAGE_ENV = "OWLBANK_STAGE"
#: The two fixed partition key values (``owlbank.failover.fleet.SAMPLES_PK``, ``STATE_PK``).
SAMPLES_PARTITION = "samples"
STATE_PARTITION = "state"
#: The poller's logical name; its handler is ``lambdas.model_poller.handler.handler``.
POLLER = "model_poller"
POLLER_MEMORY_MB = 128
#: Two decision cycles 30 s apart; the second can run up to five recovery pings about 8 s
#: apart, each abandoned at its entry's deadline, and the rechecks of unavailable entries.
#: 120 s covers that with room; the next scheduled run may overlap the end of this one, which
#: the conditional state write tolerates.
POLLER_TIMEOUT_S = 120
SCHEDULE = Duration.minutes(1)
#: The fleet's sample-write budget, writes per second (``owlbank.failover.fleet``).
WRITE_BUDGET_PER_S = 5.0
WRITE_ALARM_PERIOD = Duration.minutes(5)
#: Consumed write units per period above twice the budget.
WRITE_ALARM_THRESHOLD = 2 * WRITE_BUDGET_PER_S * WRITE_ALARM_PERIOD.to_seconds()

#: EMF metrics: namespace ``OwlBank/<stage>/Model``, dimension set ``[Service,
#: Environment]`` (``Service`` = :func:`bridge_service` or :func:`poller_service`).
METRIC_NAMESPACE_FORMAT = "OwlBank/{stage}/Model"
#: Poller: 1 per cycle whose heartbeat write of the state record succeeded (none when the table
#: can't be read or written, so the staleness alarm fires).
HEARTBEAT_METRIC = "StateHeartbeat"
#: Poller: per cycle, the number of entries the verdict has neither degraded nor unavailable.
USABLE_METRIC = "UsableEntries"
#: Bridge: each task's active entry rank, 0 = the preferred entry.
ACTIVE_RANK_METRIC = "ActiveRank"
HEARTBEAT_PERIOD = Duration.minutes(1)
HEARTBEAT_MISSING_PERIODS = 3
ACTIVE_RANK_PERIOD = Duration.minutes(5)
ACTIVE_RANK_PERIODS = 3
#: The bridge's log line for an entry found unavailable (4a.4), and its entry field.
UNAVAILABLE_EVENT = "model_unavailable"
UNAVAILABLE_ENTRY_FIELD = "model_id"
UNAVAILABLE_METRIC = "ModelUnavailable"
PREFERRED_UNAVAILABLE_METRIC = "PreferredModelUnavailable"
UNAVAILABLE_PERIOD = Duration.minutes(1)

#: What the bridge does with the table, and what the poller does.
BRIDGE_SAMPLE_ACTIONS = ("dynamodb:UpdateItem",)
BRIDGE_STATE_ACTIONS = ("dynamodb:GetItem",)
POLLER_SAMPLE_ACTIONS = ("dynamodb:Query",)
#: GetItem: the poller reads the previous verdict at the start of an invocation (``fleet_verdict``
#: folds over it; a Lambda instance keeps nothing reliably). PutItem: the conditional state write
#: and the heartbeat.
POLLER_STATE_ACTIONS = ("dynamodb:GetItem", "dynamodb:PutItem")
#: The poller reads the ``model_failover`` policy section (``Policy.from_store``: GetItem of the
#: meta item, BatchGetItem of the rest). No leading-key condition: it reads several fixed items.
POLLER_POLICY_ACTIONS = ("dynamodb:GetItem", "dynamodb:BatchGetItem")
POLICY_TABLE_ENV = "OWLBANK_TABLE_POLICY"
#: The poller's model actions per entry: the synthetic recheck and the ConverseStream pings.
POLLER_MODEL_ACTIONS = ("bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream")

NO_PITR_REASON = (
    "modelstats holds only 10-second latency counters and the fleet verdict, no caller data, and every "
    "item expires by TTL in about 15 minutes: there is nothing to restore, and "
    "the poller rebuilds the verdict within a minute.")


def metric_namespace(cfg: InfraConfig) -> str:
    return METRIC_NAMESPACE_FORMAT.format(stage=cfg.stage)


def bridge_service(cfg: InfraConfig) -> str:
    """The bridge's EMF ``Service`` dimension (the same as call observability's)."""
    return f"{cfg.name_prefix}-bridge"


def poller_service(cfg: InfraConfig) -> str:
    return f"{cfg.name_prefix}-model-poller"


def table_arn(stack: Stack, cfg: InfraConfig) -> str:
    """The table's ARN, built from its fixed name (no reference to the Data stack)."""
    return (f"arn:{stack.partition}:dynamodb:{stack.region}:{stack.account}:table/"
            f"{cfg.failover_table_name}")


def _leading_key(partition: str) -> dict:
    return {"ForAllValues:StringEquals": {"dynamodb:LeadingKeys": [partition]}}


def _item_statement(stack: Stack, cfg: InfraConfig, sid: str, actions: tuple[str, ...],
                    partition: str) -> iam.PolicyStatement:
    return iam.PolicyStatement(sid=sid, actions=list(actions), resources=[table_arn(stack, cfg)],
                               conditions=_leading_key(partition))


# --------------------------------------------------------------------------- OwlBankData

def build_table(stack: Stack, cfg: InfraConfig, key: kms.IKey, alarms: Alarms) -> ddb.Table:
    """``modelstats``, always; its consumed-write alarm only when failover is on
    (nothing writes to the table otherwise)."""
    td = FAILOVER_TABLES[TABLE]
    table = ddb.Table(
        stack, "ModelstatsTable",
        table_name=cfg.failover_table_name,
        partition_key=ddb.Attribute(name=td.pk, type=ddb.AttributeType.STRING),
        sort_key=ddb.Attribute(name=td.sk or "sk", type=ddb.AttributeType.STRING),
        billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
        table_class=ddb.TableClass.STANDARD,
        encryption=ddb.TableEncryption.CUSTOMER_MANAGED,
        encryption_key=key,
        time_to_live_attribute=TTL_ATTRIBUTE,
        deletion_protection=cfg.is_prod,  # Like the other tables
        removal_policy=RemovalPolicy.RETAIN,
    )
    ack(table, "AwsSolutions-DDB3", NO_PITR_REASON)
    if not cfg.failover_on:
        return table
    alarms.alarm(
        "ModelstatsWriteBudgetAlarm",
        table.metric_consumed_write_capacity_units(period=WRITE_ALARM_PERIOD, statistic="Sum"),
        threshold=WRITE_ALARM_THRESHOLD, comparison=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
        description=(f"modelstats consumed write units above twice the sampling budget "
                     f"({WRITE_BUDGET_PER_S:g} writes/s): check the bridges' "
                     "report sampling"))
    return table


# --------------------------------------------------------------------------- OwlBankBridge

def bridge_env(cfg: InfraConfig) -> dict[str, str]:
    """The bridge's table name when on (none when off, so no reporter or reader is built). No
    stage: the partitions are fixed values."""
    if not cfg.failover_on:
        return {}
    return {TABLE_ENV: cfg.failover_table_name}


def grant_bridge(stack: Stack, role: iam.IRole, cfg: InfraConfig) -> None:
    """The bridge updates sample items and reads the state record, nothing else. The CMK use
    through DynamoDB is the bridge's existing ``TablesKmsViaDynamoDB`` statement (same key)."""
    if not cfg.failover_on:
        return
    role.add_to_principal_policy(_item_statement(stack, cfg, "ModelstatsReportSamples",
                                                 BRIDGE_SAMPLE_ACTIONS, SAMPLES_PARTITION))
    role.add_to_principal_policy(_item_statement(stack, cfg, "ModelstatsReadState",
                                                 BRIDGE_STATE_ACTIONS, STATE_PARTITION))


def _metric(cfg: InfraConfig, name: str, service: str, *, statistic: str, period: Duration) -> cw.Metric:
    return cw.Metric(namespace=metric_namespace(cfg), metric_name=name,
                     dimensions_map={"Service": service, "Environment": cfg.stage},
                     statistic=statistic, period=period)


def bridge_alarms(scope: Construct, cfg: InfraConfig, alarms: Alarms, group: logs.ILogGroup) -> None:
    """Alarms on the bridge's log group (either hosting): ``model_unavailable`` at once (any
    entry, and the preferred entry on its own alarm, which names it), and ``ActiveRank > 0`` for 15
    minutes."""
    if not cfg.failover_on:
        return
    preferred = cfg.models[0].id
    event = logs.FilterPattern.string_value("$.event", "=", UNAVAILABLE_EVENT)
    for cid, metric_name, pattern, description in (
        ("ModelUnavailable", UNAVAILABLE_METRIC, event,
         "A model entry is unavailable (model_unavailable on the bridge): the "
         "bridge has moved to the next entry; the log line names the entry and the error code"),
        ("PreferredModelUnavailable", PREFERRED_UNAVAILABLE_METRIC,
         logs.FilterPattern.all(event, logs.FilterPattern.string_value(
             f"$.{UNAVAILABLE_ENTRY_FIELD}", "=", preferred)),
         f"The preferred model entry {preferred} is unavailable (model_unavailable, 4a.4): calls run "
         "on a fallback model"),
    ):
        mf = logs.MetricFilter(scope, f"{cid}Filter", log_group=group, filter_pattern=pattern,
                               metric_namespace=metric_namespace(cfg), metric_name=metric_name,
                               metric_value="1", default_value=0)
        alarms.alarm(f"{cid}Alarm", mf.metric(statistic="Sum", period=UNAVAILABLE_PERIOD),
                     threshold=1, description=description)
    alarms.alarm(
        "ModelActiveRankAlarm",
        _metric(cfg, ACTIVE_RANK_METRIC, bridge_service(cfg), statistic="Maximum",
                period=ACTIVE_RANK_PERIOD),
        threshold=0, comparison=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
        evaluation_periods=ACTIVE_RANK_PERIODS,
        description=(f"A bridge has run off the preferred model ({preferred}) for 15 minutes "
                     "(ActiveRank above 0)"))


# --------------------------------------------------------------------------- OwlBankCompute

def poller_env(cfg: InfraConfig, models_env: dict[str, str], policy_table_name: str) -> dict[str, str]:
    """The poller's environment: the table, the stage (EMF metric names only), the ``policy`` table (its ``model_failover``
    section, cached 60 s), and the bridge's ``OWLBANK_MODELS`` (and ``OWLBANK_FAILOVER_TEST`` when
    set), so it pings with the same entries and parameters."""
    return {TABLE_ENV: cfg.failover_table_name, POLICY_TABLE_ENV: policy_table_name,
            STAGE_ENV: cfg.stage, **models_env}


def grant_poller(stack: Stack, fn: lambda_.IFunction, cfg: InfraConfig, key: kms.IKey,
                 policy_table_arn: str) -> None:
    """The poller queries the sample window and reads and writes the state record only; it
    reads the ``policy`` table (GetItem, BatchGetItem: ``Policy.from_store``, the ``model_failover``
    section); it invokes every listed entry (rechecks
    and pings), each only through its profile. The policy table uses the same data CMK, so
    the one ``kms:ViaService`` statement covers both tables."""
    from .compute_stack import bedrock_invoke_statements, bedrock_nag_ack

    role = fn.role
    if role is None:  # every OwlBank Lambda has its own role (compute_stack.lambda_function)
        raise ValueError("the poller has no role")
    role.add_to_principal_policy(_item_statement(stack, cfg, "ModelstatsQuerySamples",
                                                 POLLER_SAMPLE_ACTIONS, SAMPLES_PARTITION))
    role.add_to_principal_policy(_item_statement(stack, cfg, "ModelstatsWriteState",
                                                 POLLER_STATE_ACTIONS, STATE_PARTITION))
    role.add_to_principal_policy(iam.PolicyStatement(
        sid="PolicyRead", actions=list(POLLER_POLICY_ACTIONS), resources=[policy_table_arn]))
    role.add_to_principal_policy(iam.PolicyStatement(
        sid="ModelstatsKmsViaDynamoDB",
        actions=["kms:Decrypt", "kms:DescribeKey", "kms:Encrypt", "kms:GenerateDataKey"],
        resources=[key.key_arn],
        conditions={"StringEquals": {"kms:ViaService": f"dynamodb.{stack.region}.amazonaws.com"}}))
    for i, entry in enumerate(cfg.models):
        sid = "PollerInvokeModel" if i == 0 else f"PollerInvokeModel{i + 1}"
        for st in bedrock_invoke_statements(stack, entry.id, sid=sid, actions=list(POLLER_MODEL_ACTIONS)):
            role.add_to_principal_policy(st)
        bedrock_nag_ack(role, entry.id)


def schedule(scope: Construct, fn: lambda_.IFunction) -> events.Rule:
    """Once a minute (EventBridge's shortest rate); never retried, the next run comes anyway."""
    rule = events.Rule(scope, "ModelPollerSchedule", schedule=events.Schedule.rate(SCHEDULE),
                       description="Model-failover poller: two decision cycles per minute")
    rule.add_target(targets.LambdaFunction(fn, retry_attempts=0, max_event_age=Duration.minutes(1)))
    return rule


def poller_alarms(cfg: InfraConfig, alarms: Alarms) -> None:
    """The state record not refreshed for 3 minutes (missing heartbeats breach), and every entry
    degraded or unavailable. The poller's Errors and Throttles alarms are the per-function ones."""
    alarms.alarm(
        "ModelStateStaleAlarm",
        _metric(cfg, HEARTBEAT_METRIC, poller_service(cfg), statistic="Sum", period=HEARTBEAT_PERIOD),
        threshold=1, comparison=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
        evaluation_periods=HEARTBEAT_MISSING_PERIODS, missing=cw.TreatMissingData.BREACHING,
        description=("The model-failover state record has not been refreshed for 3 minutes (no poller "
                     "heartbeat): bridges fall back to their own picture and the configured order"))
    alarms.alarm(
        "ModelAllEntriesDownAlarm",
        _metric(cfg, USABLE_METRIC, poller_service(cfg), statistic="Maximum", period=HEARTBEAT_PERIOD),
        threshold=1, comparison=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
        description="Every model entry is degraded or unavailable")
