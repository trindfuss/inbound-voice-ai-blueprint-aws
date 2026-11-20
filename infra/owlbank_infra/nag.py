"""cdk-nag (v3) acknowledgements.

cdk-nag v3 uses CDK's native ``Validations.of(scope).acknowledge(id, reason)``.
Every acknowledgement here carries a written reason; ``tests/test_nag.py`` checks
that the synthesized app has no unacknowledged AwsSolutions errors.
"""

from __future__ import annotations

from aws_cdk import Acknowledgment, Validations
from constructs import IConstruct

#: Every acknowledgement made while building the app: (construct path, rule id, reason).
ACKS: list[tuple[str, str, str]] = []


def ack(scope: IConstruct, rule_id: str, reason: str) -> None:
    if not reason or len(reason) < 20:
        raise ValueError(f"suppression of {rule_id} needs a written reason")
    Validations.of(scope).acknowledge(Acknowledgment(id=rule_id, reason=reason))
    ACKS.append((scope.node.path, rule_id, reason))
