"""Network: a VPC across 2 AZs with public subnets (ALB, NAT) and private subnets (Fargate)."""

from __future__ import annotations

from aws_cdk import Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_kms as kms
from constructs import Construct

from .config import InfraConfig
from .observability import log_group


class NetworkStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, key: kms.IKey, **kw) -> None:
        super().__init__(scope, cid, **kw)
        self.vpc = ec2.Vpc(
            self, "Vpc",
            ip_addresses=ec2.IpAddresses.cidr("10.40.0.0/16"),
            availability_zones=cfg.azs,  # explicit: no AZ lookup at synth
            nat_gateways=cfg.nat_gateways,  # TAC calls Twilio at startup: outbound internet needed
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                                        cidr_mask=22),
            ],
            # Left False on purpose (AWS low, skipped): True adds a CDK custom-resource Lambda
            # (Node.js) whose log group is auto-created without the CMK, which
            # breaks "every log group uses the CMK". Nothing uses the default SG. Restrict it
            # in the account baseline (or with Security Hub) instead.
            restrict_default_security_group=False,
        )
        # DynamoDB traffic from the private subnets stays on the AWS network (free).
        self.vpc.add_gateway_endpoint(
            "DynamoDbEndpoint", service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
            subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)])
        flow_logs = log_group(self, "FlowLogs", cfg=cfg, key=key, name="/owlbank/vpc/flow-logs")
        self.vpc.add_flow_log("FlowLog",
                              destination=ec2.FlowLogDestination.to_cloud_watch_logs(flow_logs),
                              traffic_type=ec2.FlowLogTrafficType.REJECT)
