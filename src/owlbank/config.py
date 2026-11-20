"""Environment-based settings for every OwlBank component.

Plain dataclasses, read once from environment variables. Importing this module
makes no AWS calls and reads no secrets: secret *ARNs* live here, and the values
are fetched at runtime by ``owlbank.secrets``.

Environment variables (all optional; defaults suit local tests):

==============================  ====================================  =======================
Variable                        Meaning                               Default
==============================  ====================================  =======================
``OWLBANK_TABLE_<LOGICAL>``     Physical DynamoDB table name for a    ``owlbank-<logical>``
                                logical table (see ``TABLE_NAMES``)
``OWLBANK_SECRET_TWILIO_API_KEY``    Secrets Manager ARN              unset
``OWLBANK_SECRET_TWILIO_AUTH_TOKEN`` Secrets Manager ARN              unset
``OWLBANK_SECRET_BASIC_AUTH``        Secrets Manager ARN              unset
``OWLBANK_VERIFY_SERVICE_SID``  Twilio Verify Service SID (OTP)       unset
``OWLBANK_TOKEN_KMS_KEY``       KMS key ID/ARN that seals call tokens unset
                                in call state (``owlbank.token_seal``)
``OWLBANK_DEFAULT_LANGUAGE``    BCP-47 default caller language        ``en-US``
``OWLBANK_SUPPORTED_LANGUAGES`` Comma-separated BCP-47 tags           ``en-US,es-US``
``OWLBANK_TOKEN_MAX_TTL_S``     Maximum call-token lifetime, seconds  ``1800``
``OWLBANK_CACHE_TTL_S``         Policy/catalog read cache, seconds    ``60``
``OWLBANK_RECORD_TTL_S``        TTL for calls/segments records        ``86400``
``AWS_REGION``                  AWS region for clients                unset
==============================  ====================================  =======================
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

#: Logical table names used throughout the code.
TABLE_NAMES: tuple[str, ...] = ("customers", "tokens", "calls", "segments", "policy", "catalog",
                                "risk")

_PREFIX = "OWLBANK_"


def _table_env_var(logical: str) -> str:
    return f"{_PREFIX}TABLE_{logical.upper()}"


@dataclass(frozen=True, slots=True)
class SecretArns:
    """ARNs of the Secrets Manager secrets (values are never held in config)."""

    twilio_api_key: str | None = None
    twilio_auth_token: str | None = None
    basic_auth: str | None = None


@dataclass(frozen=True, slots=True)
class Settings:
    """All runtime settings. Build with :meth:`from_env`; construct directly in tests."""

    tables: Mapping[str, str] = field(
        default_factory=lambda: {name: f"owlbank-{name}" for name in TABLE_NAMES}
    )
    secrets: SecretArns = field(default_factory=SecretArns)
    default_language: str = "en-US"
    supported_languages: tuple[str, ...] = ("en-US", "es-US")
    token_max_ttl_s: int = 1800
    cache_ttl_s: int = 60
    record_ttl_s: int = 86400
    aws_region: str | None = None
    verify_service_sid: str | None = None
    token_kms_key: str | None = None

    def __post_init__(self) -> None:
        missing = [name for name in TABLE_NAMES if name not in self.tables]
        if missing:
            raise ValueError(f"missing table names: {missing}")
        if self.default_language not in self.supported_languages:
            raise ValueError(
                f"default language {self.default_language!r} not in {self.supported_languages}"
            )
        for name, value in (
            ("token_max_ttl_s", self.token_max_ttl_s),
            ("cache_ttl_s", self.cache_ttl_s),
            ("record_ttl_s", self.record_ttl_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    def table(self, logical: str) -> str:
        """Physical table name for a logical table; ``KeyError`` if unknown."""
        return self.tables[logical]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read settings from ``env`` (defaults to ``os.environ``)."""
        env = os.environ if env is None else env
        tables = {name: env.get(_table_env_var(name), f"owlbank-{name}") for name in TABLE_NAMES}
        langs = tuple(
            tag.strip()
            for tag in env.get(f"{_PREFIX}SUPPORTED_LANGUAGES", "en-US,es-US").split(",")
            if tag.strip()
        )
        return cls(
            tables=tables,
            secrets=SecretArns(
                twilio_api_key=env.get(f"{_PREFIX}SECRET_TWILIO_API_KEY") or None,
                twilio_auth_token=env.get(f"{_PREFIX}SECRET_TWILIO_AUTH_TOKEN") or None,
                basic_auth=env.get(f"{_PREFIX}SECRET_BASIC_AUTH") or None,
            ),
            default_language=env.get(f"{_PREFIX}DEFAULT_LANGUAGE", "en-US"),
            supported_languages=langs,
            token_max_ttl_s=int(env.get(f"{_PREFIX}TOKEN_MAX_TTL_S", "1800")),
            cache_ttl_s=int(env.get(f"{_PREFIX}CACHE_TTL_S", "60")),
            record_ttl_s=int(env.get(f"{_PREFIX}RECORD_TTL_S", "86400")),
            aws_region=env.get("AWS_REGION") or None,
            verify_service_sid=env.get(f"{_PREFIX}VERIFY_SERVICE_SID") or None,
            token_kms_key=env.get(f"{_PREFIX}TOKEN_KMS_KEY") or None,
        )
