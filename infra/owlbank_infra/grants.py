"""Explicit least-privilege grants.

CDK's convenience grants add wildcards (``index/*``, ``<fn>:*``, ``kms:ReEncrypt*``,
``kms:GenerateDataKey*``). These helpers name the exact resources and actions, so
cdk-nag IAM5 is fixed rather than suppressed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from aws_cdk import Stack
from aws_cdk import aws_dynamodb as ddb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_

from .contract import TABLES
from .nag import ack

if TYPE_CHECKING:
    from .config import InfraConfig

READ = ["dynamodb:GetItem", "dynamodb:BatchGetItem", "dynamodb:Query", "dynamodb:DescribeTable"]
WRITE = ["dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem", "dynamodb:BatchWriteItem"]
#: The prompt catalog reloads with one paginated ``Scan`` of its table (the
#: table has no index to Query). Its own statement on the catalog table ARN only, never in
#: :data:`READ`, so no other table can be scanned.
CATALOG_TABLE = "catalog"
CATALOG_SCAN_SID = "CatalogScan"
CATALOG_SCAN = ["dynamodb:Scan"]


def catalog_scan_statement(catalog_table_arn: str) -> iam.PolicyStatement:
    """``dynamodb:Scan`` on exactly the catalog table (no ``/index/*``): what a role that loads
    :class:`owlbank.catalog.PromptCatalog` from the store needs beyond :data:`READ`. Same data
    CMK as every table, so the ``kms:ViaService`` statement already covers it."""
    return iam.PolicyStatement(sid=CATALOG_SCAN_SID, actions=list(CATALOG_SCAN),
                               resources=[catalog_table_arn])


def grant_tables(stack: Stack, principal: iam.IGrantable, key: kms.IKey,
                 tables: dict[str, ddb.ITable], read_only: tuple[str, ...] = (),
                 read_write: tuple[str, ...] = ()) -> None:
    pol = principal.grant_principal
    for names, actions, sid in ((read_only, READ, "TablesRead"), (read_write, READ + WRITE, "TablesReadWrite")):
        if not names:
            continue
        resources: list[str] = []
        for n in names:
            arn = tables[n].table_arn
            resources.append(arn)
            resources += [f"{arn}/index/{idx}" for idx, _, _ in TABLES[n].indexes
                          if idx not in TABLES[n].own_grant]
        pol.add_to_principal_policy(iam.PolicyStatement(sid=sid, actions=actions, resources=resources))
    # Every role that reads the catalog loads it through PromptCatalog.from_store (one Scan).
    if CATALOG_TABLE in read_only or CATALOG_TABLE in read_write:
        pol.add_to_principal_policy(catalog_scan_statement(tables[CATALOG_TABLE].table_arn))
    kms_actions = ["kms:Decrypt", "kms:DescribeKey"]
    if read_write:
        kms_actions += ["kms:Encrypt", "kms:GenerateDataKey"]
    if read_only or read_write:
        pol.add_to_principal_policy(iam.PolicyStatement(
            sid="TablesKmsViaDynamoDB", actions=kms_actions, resources=[key.key_arn],
            conditions={"StringEquals": {"kms:ViaService": f"dynamodb.{stack.region}.amazonaws.com"}}))


def grant_table_actions(stack: Stack, principal: iam.IGrantable, key: kms.IKey,
                        tables: dict[str, ddb.ITable], actions: Mapping[str, Sequence[str]]) -> None:
    """Exactly ``actions[name]`` on table ``name`` (its ARN only, no index), one statement per table
    (``Table<Name>``), for a role that uses fewer actions than :data:`READ` and :data:`WRITE` (human-leg
    The sweep). The same ``kms:ViaService`` statement as :func:`grant_tables`, with the write
    actions only when some table is written."""
    pol = principal.grant_principal
    allowed = set(READ) | set(WRITE)
    writes = False
    for name, acts in actions.items():
        unknown = set(acts) - allowed
        if unknown or not acts:
            raise ValueError(f"table {name}: actions must be a non-empty subset of READ + WRITE, got {sorted(unknown)}")
        writes = writes or bool(set(acts) & set(WRITE))
        pol.add_to_principal_policy(iam.PolicyStatement(
            sid=f"Table{name.capitalize()}", actions=list(acts), resources=[tables[name].table_arn]))
    if actions:
        kms_actions = ["kms:Decrypt", "kms:DescribeKey"] + (["kms:Encrypt", "kms:GenerateDataKey"] if writes else [])
        pol.add_to_principal_policy(iam.PolicyStatement(
            sid="TablesKmsViaDynamoDB", actions=kms_actions, resources=[key.key_arn],
            conditions={"StringEquals": {"kms:ViaService": f"dynamodb.{stack.region}.amazonaws.com"}}))


def function_arn(stack: Stack, name: str) -> str:
    """ARN of a Lambda with a deterministic name (``InfraConfig.function_name``), built as a
    string so the stack using it doesn't depend on the stack that owns the function."""
    return f"arn:{stack.partition}:lambda:{stack.region}:{stack.account}:function:{name}"


def grant_invoke(principal: iam.IGrantable, fn: lambda_.IFunction | str, sid: str) -> None:
    """lambda:InvokeFunction on exactly the one ARN given, never ``:*`` and no second ARN.

    Pass an alias ARN (``...:function:<name>:live``) for the three provisioned-concurrency
    functions: invoking an alias is authorized against the alias ARN, and the unqualified
    function ARN does not cover it. Other functions use their unqualified ARN."""
    arn = fn if isinstance(fn, str) else fn.function_arn
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid=sid, actions=["lambda:InvokeFunction"], resources=[arn]))


#: The DEFAULT endpoint of a Runtime ARN (``runtime.ENDPOINT_SUFFIX``, kept equal by a test).
RUNTIME_ENDPOINT_SUFFIX = "/runtime-endpoint/DEFAULT"
RUNTIME_PREFIX_REASON = (
    "The Runtime is named by its fixed name prefix, runtime/<runtime_name>-*, and its DEFAULT "
    "endpoint: AgentCore appends a generated suffix to the name to form the Runtime "
    "ID, and naming it this way means Compute needs no CloudFormation export from the Bridge stack. "
    "The prefix matches only this stage's one bridge Runtime (the name is unique per account and "
    "Region).")


def runtime_invoke_resources(stack: Stack, runtime_name: str) -> list[str]:
    """``[R, E]`` for a Runtime named by its fixed name: ``…:runtime/<runtime_name>-*`` and that
    pattern's DEFAULT endpoint. Strings, not references, so the granting stack imports nothing."""
    r = f"arn:{stack.partition}:bedrock-agentcore:{stack.region}:{stack.account}:runtime/{runtime_name}-*"
    return [r, f"{r}{RUNTIME_ENDPOINT_SUFFIX}"]


def ack_runtime_prefix(stack: Stack, role: iam.IRole, runtime_name: str) -> None:
    """The cdk-nag IAM5 acknowledgements for :func:`runtime_invoke_resources` on ``role``."""
    for r in runtime_invoke_resources(stack, runtime_name):
        ack(role, f"AwsSolutions-IAM5[Resource::{r.replace(f'arn:{stack.partition}:', 'arn:<AWS::Partition>:', 1)}]",
            RUNTIME_PREFIX_REASON)


def relay_parameter_arn(stack: Stack, name: str) -> str:
    """The ARN of an SSM parameter whose name starts with ``/`` (``…:parameter/owlbank/…``)."""
    if not name.startswith("/"):
        raise ValueError("the relay parameter name must start with '/'")
    return f"arn:{stack.partition}:ssm:{stack.region}:{stack.account}:parameter{name}"


def grant_relay_parameter_read(stack: Stack, principal: iam.IGrantable, name: str) -> None:
    """``ssm:GetParameter`` on the one relay parameter. A String parameter: no KMS grant."""
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid="ReadRelayParameter", actions=["ssm:GetParameter"],
        resources=[relay_parameter_arn(stack, name)]))


def grant_key_use(stack: Stack, principal: iam.IGrantable, key: kms.IKey, sid: str, *,
                  via_service: str | None = None) -> None:
    """Encrypt/decrypt with the CMK for AgentCore resources encrypted with it.

    Four named actions on this one key. ``via_service`` (a service name such as
    ``bedrock-agentcore``) adds ``kms:ViaService = <service>.<region>.amazonaws.com``, so the role
    can use the key only through that service, never by calling KMS itself.

    Without it there is no condition, which fits only a role that the service itself assumes: AgentCore
    calls KMS directly as the resource's service role (session ``GenesisMCPTargetTargetEncryption``,
    verified at the first deploy), not through a service endpoint, so a ViaService condition never
    matches there and Gateway creation fails with "not authorized to perform kms:GenerateDataKey".
    Such a role's trust policy allows only ``bedrock-agentcore.amazonaws.com``, with
    ``aws:SourceAccount`` and ``aws:SourceArn``. A role that application code runs as (the bridge's
    Fargate task role, which ``ecs-tasks.amazonaws.com`` assumes) passes ``via_service``.
    """
    conditions = ({"StringEquals": {"kms:ViaService": f"{via_service}.{stack.region}.amazonaws.com"}}
                  if via_service is not None else None)
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid=sid, actions=["kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey"],
        resources=[key.key_arn], conditions=conditions))


#: Comprehend PII detection (the PiiRedactor). DetectPiiEntities has no resource-level
#: permissions (Service Authorization Reference: no resource types), so "*" is required.
PII_DETECT_REASON = ("comprehend:DetectPiiEntities has no resource-level permissions (no resource "
                     "types in the Service Authorization Reference), so its resource must be '*'. "
                     "It reads only the text sent in the request and touches no stored data.")


def grant_pii_detect(principal: iam.IGrantable) -> None:
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid="DetectPiiForRedaction", actions=["comprehend:DetectPiiEntities"], resources=["*"]))


TOKEN_KEY_ENV = "OWLBANK_TOKEN_KMS_KEY"  # noqa: S105 (an environment variable name)


def grant_token_key(principal: iam.IGrantable, key: kms.IKey, actions: list[str], sid: str) -> None:
    """Token-sealing key use, only with a ``call_sid`` key in the encryption context."""
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid=sid, actions=actions, resources=[key.key_arn],
        conditions={"ForAnyValue:StringEquals": {"kms:EncryptionContextKeys": ["call_sid"]}}))


def grant_gateway_invoke(principal: iam.IGrantable, gateway_arn: str, sid: str) -> None:
    """``bedrock-agentcore:InvokeGateway`` on exactly the one Gateway ARN."""
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid=sid, actions=["bedrock-agentcore:InvokeGateway"], resources=[gateway_arn]))


#: The AgentCore Memory data-plane calls the Strands session manager
#: (``bedrock_agentcore.memory.integrations.strands.session_manager``) makes on short-term memory:
#: ``create_event`` (messages, session and agent state), ``list_events`` (load and rehydrate) and
#: ``get_event`` (``read_message``). Not granted: ``DeleteEvent`` (only for legacy-format
#: migration and ``update_message``, i.e. guardrail redaction, neither used here), ``ListSessions``
#: (never called), ``RetrieveMemoryRecords`` (long-term memory only; this Memory has no
#: strategies) and ``GetMemory`` (control plane; the session manager doesn't call it).
STRANDS_MEMORY_ACTIONS = ("bedrock-agentcore:CreateEvent", "bedrock-agentcore:GetEvent",
                          "bedrock-agentcore:ListEvents")


def grant_memory_events(principal: iam.IGrantable, memory_arn: str, sid: str) -> None:
    """Short-term-memory event access on exactly the one Memory ARN."""
    principal.grant_principal.add_to_principal_policy(iam.PolicyStatement(
        sid=sid, actions=list(STRANDS_MEMORY_ACTIONS), resources=[memory_arn]))


#: Tables the bridge reads, and the ones it reads and writes.
BRIDGE_TABLES_RO = ("customers", "policy", "catalog")
BRIDGE_TABLES_RW = ("calls", "segments")
#: The bridge role's one ``Resource::*`` acknowledgement when PII detection is its only ``*``.
BRIDGE_STAR_REASON = PII_DETECT_REASON + " Nothing else uses '*'."


#: The two model actions the agent loop needs per listed model (no wildcard action).
STRANDS_MODEL_ACTIONS = ("bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream")


def strands_model_sid(index: int) -> str:
    """The statement ID prefix of the ``index``-th listed model: ``StrandsInvokeModel`` for the
    first (so a list of one gives today's statements exactly), then ``StrandsInvokeModel2``, ..."""
    return "StrandsInvokeModel" if index == 0 else f"StrandsInvokeModel{index + 1}"


def grant_strands_backend(stack: Stack, role: iam.IRole, *, cfg: InfraConfig, key: kms.IKey,
                          gateway_arn: str, memory_arn: str) -> None:
    """What the bridge role needs to run the agent loop itself: each listed
    model (through its inference profile), the one Gateway, the one Memory,
    and the CMK the Memory is encrypted with."""
    # Imported here: compute_stack imports this module, so a top-level import would be a cycle.
    from .compute_stack import bedrock_invoke_statements, bedrock_nag_ack

    for i, entry in enumerate(cfg.models):
        for st in bedrock_invoke_statements(stack, entry.id, sid=strands_model_sid(i),
                                            actions=list(STRANDS_MODEL_ACTIONS)):
            role.add_to_principal_policy(st)
        bedrock_nag_ack(role, entry.id)
    grant_gateway_invoke(role, gateway_arn, sid="StrandsGateway")
    grant_memory_events(role, memory_arn, sid="StrandsMemory")
    # The Memory is CMK-encrypted. The role runs application code, so only through AgentCore.
    grant_key_use(stack, role, key, sid="StrandsMemoryCmk", via_service="bedrock-agentcore")


def grant_bridge_backend(stack: Stack, role: iam.IRole, *, cfg: InfraConfig, key: kms.IKey,
                         token_key: kms.IKey, tables: dict[str, ddb.ITable], bank_tools_arn: str,
                         gateway_arn: str, memory_arn: str,
                         star_reason: str = BRIDGE_STAR_REASON) -> None:
    """The bridge's application grants, one builder for both hostings: the
    Fargate ``TaskRole`` and the AgentCore Runtime's execution role get exactly these statements.

    ``bank_tools`` (one ARN), token unsealing (``call_sid`` context only), the tables it reads and
    writes, Comprehend PII detection, and the agent loop's model, Gateway and Memory. No secrets:
    Fargate's execution role injects them, and the Runtime role adds its own read (runtime.py).
    ``star_reason`` is the role's ``AwsSolutions-IAM5[Resource::*]`` acknowledgement, so a role
    with more ``*`` resources than PII detection says so in one reason.
    """
    grant_invoke(role, bank_tools_arn, sid="InvokeBankTools")
    grant_token_key(role, token_key, ["kms:Decrypt"], sid="TokenUnseal")
    grant_tables(stack, role, key, tables, read_only=BRIDGE_TABLES_RO, read_write=BRIDGE_TABLES_RW)
    grant_pii_detect(role)  # PiiRedactor (Comprehend)
    ack(role, "AwsSolutions-IAM5[Resource::*]", star_reason)
    grant_strands_backend(stack, role, cfg=cfg, key=key, gateway_arn=gateway_arn,
                          memory_arn=memory_arn)
    # Model failover, only when on: UpdateItem on sample items, GetItem on the state record.
    from .model_failover import grant_bridge  # model_failover imports compute_stack, which imports this
    grant_bridge(stack, role, cfg)
