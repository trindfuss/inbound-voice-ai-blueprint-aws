"""Tests for owlbank.config."""

from __future__ import annotations

import pytest

from owlbank.config import TABLE_NAMES, Settings


def test_defaults() -> None:
    s = Settings.from_env({})
    assert s.default_language == "en-US"
    assert s.supported_languages == ("en-US", "es-US")
    assert s.table("tokens") == "owlbank-tokens"
    assert set(s.tables) == set(TABLE_NAMES)
    assert s.secrets.twilio_auth_token is None
    assert s.cache_ttl_s == 60


def test_from_env_overrides() -> None:
    s = Settings.from_env({
        "OWLBANK_TABLE_CALLS": "stack-calls",
        "OWLBANK_SECRET_TWILIO_AUTH_TOKEN": "arn:aws:secretsmanager:us-east-1:000000000000:secret:x",
        "OWLBANK_SUPPORTED_LANGUAGES": "en-US, es-US ,",
        "OWLBANK_TOKEN_MAX_TTL_S": "600",
        "AWS_REGION": "us-east-1",
    })
    assert s.table("calls") == "stack-calls"
    assert s.secrets.twilio_auth_token.endswith(":x")
    assert s.supported_languages == ("en-US", "es-US")
    assert s.token_max_ttl_s == 600 and s.aws_region == "us-east-1"


@pytest.mark.parametrize("env", [
    {"OWLBANK_DEFAULT_LANGUAGE": "fr-FR"},
    {"OWLBANK_CACHE_TTL_S": "0"},
])
def test_invalid_settings_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        Settings.from_env(env)
