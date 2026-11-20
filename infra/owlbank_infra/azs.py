"""Which private subnets the AgentCore Runtime gets in VPC mode.

AgentCore VPC mode supports only some AZ **IDs** per Region (``AGENTCORE_SUPPORTED_AZ_IDS``), while
the VPC is built from AZ **names** (``<region>a``/``b``, or ``availabilityZones``), whose IDs differ
per account. ``scripts/preflight.py --agentcore`` reads the account's name -> ID map
(``DescribeAvailabilityZones``, read-only) and caches it in ``infra/cdk.context.json`` under
``owlbank:az-ids:<account>:<region>``; the CDK CLI passes that file to synth as context.

The Runtime gets exactly the private subnets whose AZ ID is supported:

* two (or more) -> all of them;
* one -> that subnet, with a synth warning (no AZ redundancy), and a synth failure at
  ``stage=prod`` unless ``agentcoreSingleAzAck=true``;
* none -> synth fails, naming our AZ IDs, the supported ones and ``availabilityZones`` (the only
  setting that changes the VPC);
* no cached map, or a map without one of our AZ names -> synth fails asking for preflight. Synth
  never guesses an AZ ID.

Nothing here changes the VPC or its subnets; the Fargate templates don't use this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from aws_cdk import Annotations, Token
from constructs import Construct

from .config import ConfigError, InfraConfig

#: Context key prefix of the cached AZ name -> AZ ID map (one key per account and Region).
AZ_IDS_CONTEXT_PREFIX = "owlbank:az-ids"
PREFLIGHT_COMMAND = ("uv run python scripts/preflight.py --profile <profile> --stage <stage> "
                     "--agentcore  (run from infra/; it caches the map in cdk.context.json)")
SINGLE_AZ_WARNING_ID = "owlbank:agentcore-single-az"


def az_ids_context_key(account: str, region: str) -> str:
    """``owlbank:az-ids:<account>:<region>``."""
    return f"{AZ_IDS_CONTEXT_PREFIX}:{account}:{region}"


class AzSelectionError(ConfigError):
    """The Runtime's subnets can't be chosen; the message says what to run or set."""


class _HasAz(Protocol):
    @property
    def availability_zone(self) -> str: ...


@dataclass(frozen=True)
class RuntimeSubnets[S: _HasAz]:
    """The chosen subnets (input order kept) and their AZ IDs; ``warning`` is set when single-AZ."""

    subnets: tuple[S, ...]
    az_ids: tuple[str, ...]
    warning: str | None = None

    @property
    def kind(self) -> Literal["multi_az", "single_az"]:
        return "single_az" if len(self.subnets) == 1 else "multi_az"


def parse_az_map(raw: Any, *, key: str) -> dict[str, str]:
    """The cached map as ``{name: id}``; missing or malformed -> :class:`AzSelectionError` (preflight)."""
    if raw is None:
        raise AzSelectionError(
            f"no AZ name -> AZ ID map in the CDK context ({key}), which AgentCore VPC mode needs to "
            f"choose the Runtime's subnets (synth never guesses AZ IDs). Run preflight first: "
            f"{PREFLIGHT_COMMAND}")
    if (not isinstance(raw, Mapping) or not raw
            or not all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in raw.items())):
        raise AzSelectionError(
            f"the CDK context value {key} is not a non-empty map of AZ name -> AZ ID. Rerun "
            f"preflight to rewrite it: {PREFLIGHT_COMMAND}")
    return dict(raw)


def _describe(names: Sequence[str], az_map: Mapping[str, str]) -> str:
    return ", ".join(f"{n} ({az_map[n]})" for n in names)


def runtime_subnets[S: _HasAz](subnets: Sequence[S], az_map: Mapping[str, str] | None,
                    supported: Sequence[str], stage: str, single_az_ack: bool, *,
                    region: str = "", context_key: str = "") -> RuntimeSubnets[S]:
    """The private ``subnets`` whose AZ ID (via ``az_map``) is in ``supported``. Raises
    :class:`AzSelectionError` for no map, an AZ name missing from the map, no supported subnet, or a
    single supported subnet at ``stage=prod`` without ``single_az_ack``. Never mutates its inputs."""
    key = context_key or f"{AZ_IDS_CONTEXT_PREFIX}:<account>:{region or '<region>'}"
    mapping = parse_az_map(az_map, key=key)
    names = [s.availability_zone for s in subnets]
    if not names:
        raise AzSelectionError("the VPC has no private subnets to place the AgentCore Runtime in")
    unresolved = [n for n in names if not isinstance(n, str) or Token.is_unresolved(n)]
    if unresolved:
        raise AzSelectionError("the private subnets' AZs are not known at synth; the VPC must use "
                               "explicit AZ names (InfraConfig.azs)")
    unknown = sorted({n for n in names if n not in mapping})
    if unknown:
        raise AzSelectionError(
            f"the cached AZ map ({key}) has no AZ ID for {', '.join(unknown)}. Rerun preflight to "
            f"refresh it: {PREFLIGHT_COMMAND}")
    allowed = set(supported)
    chosen = tuple(s for s in subnets if mapping[s.availability_zone] in allowed)
    where = f" in {region}" if region else ""
    supported_text = ", ".join(supported) or (
        f"(none listed{where}; add the Region's list from the AgentCore docs to "
        "AGENTCORE_SUPPORTED_AZ_IDS)")
    if not chosen:
        ordered = list(dict.fromkeys(names))
        candidates = sorted(n for n, i in mapping.items() if i in allowed)
        hint = (f"; in this account those are {_describe(candidates, mapping)}" if candidates else "")
        raise AzSelectionError(
            f"none of the VPC's private subnets is in an AZ that AgentCore VPC mode supports{where}: "
            f"our AZs are {_describe(ordered, mapping)}; the supported AZ IDs are {supported_text}"
            f"{hint}. Set the optional context key availabilityZones to two AZ names whose IDs are "
            "supported (this replaces the VPC and its subnets), or use agentcoreNetworkMode=public.")
    ids = tuple(mapping[s.availability_zone] for s in chosen)
    if len(chosen) > 1:
        return RuntimeSubnets(chosen, ids)
    others = [n for n in dict.fromkeys(names) if n != chosen[0].availability_zone]
    warning = (
        f"the AgentCore Runtime runs in one AZ only ({_describe([chosen[0].availability_zone], mapping)}): "
        f"{_describe(others, mapping)} is not a supported AZ ID ({supported_text}), so a failure of that "
        "AZ stops the bridge. Set availabilityZones to two supported AZ names for AZ redundancy.")
    if stage == "prod" and not single_az_ack:
        raise AzSelectionError(
            f"stage=prod with {warning} Set agentcoreSingleAzAck=true to accept a single-AZ Runtime.")
    return RuntimeSubnets(chosen, ids, warning)


def select_runtime_subnets[S: _HasAz](scope: Construct, cfg: InfraConfig,
                                      subnets: Sequence[S]) -> list[S]:
    """:func:`runtime_subnets` with the map read from ``scope``'s context and the warning added to
    ``scope`` as a synth warning. For the Runtime's ``networkConfiguration`` in VPC mode."""
    key = az_ids_context_key(cfg.account, cfg.region)
    result = runtime_subnets(subnets, scope.node.try_get_context(key), cfg.supported_az_ids,
                             cfg.stage, cfg.agentcore_single_az_ack, region=cfg.region, context_key=key)
    if result.warning:
        Annotations.of(scope).add_warning_v2(SINGLE_AZ_WARNING_ID, result.warning)
    return list(result.subnets)
