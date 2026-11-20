"""Portable secrets: the registry cross-checks.

1. ``data_stack.SECRETS`` (the secrets CDK creates) and ``owlbank.secret_schema.SCHEMAS`` (their
   fields) name exactly the same secrets, so a new secret needs both entries.
2. No allowed context key (``app_builder.CONTEXT_KEYS``) equals, or ends with, a secret field name
   from ``SCHEMAS``, compared case-insensitively with camelCase and snake_case normalised away, so
   no allowed key can carry a secret into a template.

Why here: this suite imports ``owlbank_infra`` (it has ``aws_cdk``; the root suite doesn't), and
``owlbank.secret_schema`` is standard library only, so it imports from ``src/`` as
``test_agent.py`` does for ``owlbank.tools_contract``. No AST reading is needed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owlbank.secret_schema import SCHEMAS  # noqa: E402

from owlbank_infra.app_builder import CONTEXT_KEYS  # noqa: E402
from owlbank_infra.data_stack import SECRETS  # noqa: E402

#: Reviewed (context key, field) pairs that end with a field name but carry no secret. Each needs a
#: reason; adding one is a review decision, like adding a context key.
NOT_SECRET: dict[tuple[str, str], str] = {
    # The Basic auth *user* is a fixed, non-secret name, set by context; only the
    # password is secret, and Secrets Manager generates it.
    ("basicAuthUser", "user"): "the generated Basic auth secret's user name, not secret",
}


def norm(name: str) -> str:
    """``apiSecret``, ``api_secret``, ``API-Secret`` -> ``apisecret``."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def matches(key: str, field: str) -> bool:
    k, f = norm(key), norm(field)
    return k == f or k.endswith(f)


FIELDS = sorted({f for fields in SCHEMAS.values() for f in fields})


def test_data_stack_and_schema_name_the_same_secrets():
    assert set(SECRETS) == set(SCHEMAS)


def test_no_context_key_matches_a_secret_field():
    hits = [(k, f) for k in CONTEXT_KEYS for f in FIELDS if matches(k, f) and (k, f) not in NOT_SECRET]
    assert hits == [], f"context keys that look like secret fields (add to SECRETS, not context): {hits}"


def test_every_exemption_is_still_needed():
    """A stale exemption would hide a future key with the same name."""
    for key, field in NOT_SECRET:
        assert key in CONTEXT_KEYS and field in FIELDS and matches(key, field), (key, field)


def test_the_match_normalises_case_and_separators():
    assert matches("apiSecret", "api_secret")
    assert matches("twilioAuthToken", "auth_token")
    assert matches("TWILIO_AUTH_TOKEN", "auth_token")
    assert matches("basic-auth-password", "password")
    assert matches("accountSid", "account_sid")
    assert not matches("authTokenTtl", "auth_token")
    assert not matches("tokensPerCallCeiling", "auth_token")
