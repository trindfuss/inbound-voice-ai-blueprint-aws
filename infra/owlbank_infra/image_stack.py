"""BridgeImage: the private ECR repository of the bridge image.

One repository, ``cfg.image_repo_name``, holds the image built from ``src/tac_bridge/Dockerfile``
for ``linux/arm64`` by ``infra/scripts/push_bridge_image.py``. The Runtime references the image by
digest (``-c bridgeImageDigest=sha256:...``), never by tag:

* **Immutable tags**: a tag (``git-<sha>``) names one image for ever.
* **Scan on push**: every pushed image gets a basic vulnerability scan.
* **Project CMK**: images are encrypted with the Data stack's key, named through its fixed
  alias (:data:`DATA_KEY_ALIAS`), not through a cross-stack reference. A reference would add an
  export to ``OwlBankData`` that this stack imports, and any later ``OwlBankData`` deploy that
  dropped it while this stack stays would then fail on an export in use.
* **Retained**: ``RETAIN``, so a ``cdk destroy`` of this stack, or a switch back to
  ``fargate``, never deletes the images a Runtime may be running.
* **No repository policy**: only IAM in this account can reach it (the Runtime's execution role
  gets ``ecr:BatchGetImage`` and ``ecr:GetDownloadUrlForLayer`` on this repository's ARN).

Part of the app on both hostings, so a fresh account deploys the repository, pushes
the image, then deploys ``agentcore``. Nothing depends on it with ``fargate``, so the other Fargate
templates are unchanged by it.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_kms as kms
from constructs import Construct

from .config import InfraConfig

#: The fixed alias of the project CMK (``OwlBankData``'s ``DataKey``). A test keeps the two in step.
DATA_KEY_ALIAS = "alias/owlbank-data"


class BridgeImageStack(Stack):
    """``OwlBankBridgeImage``: the ECR repository of the bridge image."""

    def __init__(self, scope: Construct, cid: str, *, cfg: InfraConfig, **kw) -> None:
        super().__init__(scope, cid, **kw)
        key = kms.Alias.from_alias_name(self, "DataKeyAlias", DATA_KEY_ALIAS)
        self.repository = ecr.Repository(
            self, "Repository",
            repository_name=cfg.image_repo_name,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            image_scan_on_push=True,
            encryption=ecr.RepositoryEncryption.KMS,
            encryption_key=key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        CfnOutput(self, "RepositoryUri", value=self.repository.repository_uri,
                  description="Push target of infra/scripts/push_bridge_image.py (--outputs or --repository-uri)")
        CfnOutput(self, "RepositoryArn", value=self.repository.repository_arn)
        CfnOutput(self, "RepositoryName", value=self.repository.repository_name)
