"""The one-time-passcode factor: a small Twilio Verify client interface.

:class:`VerifyClient` is what the auth service depends on. Tests inject a fake;
:class:`TwilioVerifyClient` wraps the Twilio SDK and is built only by the Lambda
handler, from the scoped API key in Secrets Manager (REST calls use an
``SK…`` key, never the Auth Token). Nothing here logs ``to`` or ``code``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from owlbank.obs import emit, error_code_of, log_caught, timed_call

log = logging.getLogger(__name__)


class VerifyUnavailable(RuntimeError):
    """Twilio Verify could not be reached or rejected the request."""


@runtime_checkable
class VerifyClient(Protocol):
    def send(self, to: str, channel: str, locale: str | None = None) -> None:
        """Start a verification (send a code) to ``to`` over ``channel`` (``sms``, ``call``...).

        ``locale`` is the call's language tag (``en-US``, ``es-US``); the Twilio client
        maps it to a Verify ``Locale`` (see :func:`verify_locale`)."""
        ...

    def check(self, to: str, code: str) -> bool:
        """True if ``code`` is the approved code for ``to``."""
        ...


#: Call language -> Twilio Verify ``Locale`` (Verify takes ``en``, ``es``...; a regional tag
#: not in Verify's list would be rejected, so only the language part is sent).
_VERIFY_LOCALES = {"en-US": "en", "es-US": "es"}


def verify_locale(language: str | None) -> str | None:
    """The Verify ``Locale`` for a call language; ``None`` (Verify's default) if unknown."""
    return _VERIFY_LOCALES.get(language) if isinstance(language, str) else None


class TwilioVerifyClient:
    """:class:`VerifyClient` over ``twilio.rest.Client`` (Verify v2).

    ``client_factory`` returns a configured ``twilio.rest.Client``; it is called
    lazily, so constructing this object makes no network call or secret read.
    """

    def __init__(self, service_sid: str, client_factory: Callable[[], Any]) -> None:
        if not service_sid:
            raise ValueError("service_sid is required")
        self._service_sid = service_sid
        self._factory = client_factory
        self._client: Any | None = None

    def _service(self) -> Any:
        if self._client is None:
            self._client = self._factory()
        return self._client.verify.v2.services(self._service_sid)

    def send(self, to: str, channel: str, locale: str | None = None) -> None:
        params = {"to": to, "channel": channel}
        mapped = verify_locale(locale)
        if mapped is not None:
            params["locale"] = mapped
        try:
            with timed_call("twilio_verify", "verifications.create"):
                self._service().verifications.create(**params)
        except Exception as exc:  # never include the exception text: it can echo `to`
            log_caught(log, "verify_send_failed", exc, operation="verifications.create",
                       decision="raise_verify_unavailable")
            raise VerifyUnavailable("verify send failed") from exc

    def check(self, to: str, code: str) -> bool:
        try:
            with timed_call("twilio_verify", "verification_checks.create", expected_codes=("20404",)):
                result = self._service().verification_checks.create(to=to, code=code)
        except Exception as exc:
            status = getattr(exc, "status", None)
            if status == 404:  # no pending verification (expired or already used)
                emit(log, logging.INFO, "verify_check_no_pending", operation="verification_checks.create",
                     error_code=error_code_of(exc), decision="ignored_because_no_pending_verification")
                return False
            log_caught(log, "verify_check_failed", exc, operation="verification_checks.create",
                       decision="raise_verify_unavailable")
            raise VerifyUnavailable("verify check failed") from exc
        return getattr(result, "status", None) == "approved"


def preload_twilio_sdk() -> None:
    """Import the Twilio SDK now (the slow part of the first OTP call), without creating a client,
    reading a secret or touching the network. A missing SDK is left to fail at first use."""
    try:
        import twilio.rest  # noqa: F401, PLC0415
    except ImportError:  # pragma: no cover - the layer always has it
        log.warning("twilio sdk preload skipped", extra={"event": "twilio_preload_skipped"})


def twilio_client_factory(secret_arn: str) -> Callable[[], Any]:
    """A factory building ``twilio.rest.Client`` from the API-key secret.

    The secret is JSON with ``account_sid``, ``api_key`` (``SK…``) and ``api_secret``.
    """

    def build() -> Any:
        from twilio.rest import Client

        from owlbank.secrets import get_secret

        s = get_secret(secret_arn, logical="twilio_api_key")
        return Client(s["api_key"], s["api_secret"], s["account_sid"])

    return build
