"""API Gateway REQUEST authorizer for the Flex Context API.

The Flex plugin sends the logged-in agent's Flex user token in ``X-Flex-Token``.
This authorizer validates it through a :class:`FlexTokenValidator` and returns an
IAM policy that allows **only** ``GET /context/*`` on the calling API and stage,
with the worker identity in the authorizer context. Every other outcome (no
token, invalid token, validator error, unexpected route, any exception) returns
an explicit Deny. The token is never logged and never placed in the context.

The production validator, :class:`TwilioIamTokenValidator`, calls Twilio's token
validation endpoint ``POST https://iam.twilio.com/v1/Accounts/{AccountSid}/Tokens/validate``
with HTTP Basic credentials read from Secrets Manager (``{"account_sid", "api_key",
"api_secret"}``): the shared ``twilio/api-key`` secret. It needs a Standard key, because
Twilio has no Restricted permission for ``Tokens/validate``. Because the endpoint is
scoped to our account, a token from another Twilio account does not validate.

Environment:

* ``OWLBANK_SECRET_TWILIO_API_KEY``: ARN of that secret.
* ``OWLBANK_FLEX_TOKEN_VALIDATE_URL`` (optional): override of the validation URL
  template, with ``{account_sid}``; defaults to :data:`DEFAULT_VALIDATE_URL`.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from owlbank import obs
from owlbank.obs import timed_call

log = logging.getLogger(__name__)

DEFAULT_VALIDATE_URL = "https://iam.twilio.com/v1/Accounts/{account_sid}/Tokens/validate"
TOKEN_HEADER = "x-flex-token"  # noqa: S105 (an HTTP header name, not a secret)
#: Flex JWE tokens are a few KB; anything larger is rejected before any network call.
MAX_TOKEN_LEN = 8192
#: The only route this authorizer ever allows.
ALLOWED_METHOD = "GET"
ALLOWED_PATH_PREFIX = "context/"

_TOKEN_RE = re.compile(r"^[A-Za-z0-9._~+/=-]+$")
_ACCOUNT_SID_RE = re.compile(r"^AC[0-9a-fA-F]{32}$")
_METHOD_ARN_RE = re.compile(
    r"^arn:(?P<partition>aws[a-z-]*):execute-api:(?P<region>[a-z0-9-]+):(?P<account>\d{12}):"
    r"(?P<api>[A-Za-z0-9]+)/(?P<stage>[A-Za-z0-9_-]+)/(?P<method>[A-Z]+)/(?P<path>.*)$"
)


class InvalidToken(Exception):
    """The token is missing, malformed, expired or rejected by Twilio."""


class ValidatorError(Exception):
    """The validator could not reach a decision (network, secret, bad response)."""


@dataclass(frozen=True, slots=True)
class FlexIdentity:
    """Who the token belongs to (from Twilio's validation response)."""

    worker_sid: str
    identity: str
    roles: tuple[str, ...] = ()


class FlexTokenValidator(Protocol):
    def validate(self, token: str) -> FlexIdentity:
        """Return the identity, or raise :class:`InvalidToken` / :class:`ValidatorError`."""
        ...


# --------------------------------------------------------------------------- Twilio validator

Opener = Callable[[urllib.request.Request, float], Any]


def _require_https(url: str) -> None:
    """Raise unless ``url`` is an ``https`` URL with a host (``urlopen`` also speaks ``file:`` and ``ftp:``).

    An explicit ``raise``, not an ``assert``, so ``python -O`` cannot remove it.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("validation URL must be https")


def _default_opener(req: urllib.request.Request, timeout: float) -> Any:
    _require_https(req.full_url)
    # S310 is satisfied by the check above: the scheme is https, the host comes from the constant
    # DEFAULT_VALIDATE_URL (or the operator-set override, which is also checked at construction).
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  # nosec B310


class TwilioIamTokenValidator:
    """Validates Flex tokens with Twilio's ``Tokens/validate`` endpoint.

    ``credentials`` returns ``(account_sid, username, password)``; the username and
    password are the API key SID and secret. ``opener`` is injectable for tests.
    """

    def __init__(self, credentials: Callable[[], tuple[str, str, str]], *,
                 url_template: str = DEFAULT_VALIDATE_URL, timeout_s: float = 3.0,
                 opener: Opener = _default_opener) -> None:
        _require_https(url_template)
        self._credentials = credentials
        self._url_template = url_template
        self._timeout_s = timeout_s
        self._opener = opener

    def validate(self, token: str) -> FlexIdentity:
        try:
            account_sid, username, password = self._credentials()
        except Exception as exc:  # secret read failed: no decision possible
            obs.log_caught(log, "flex authorizer: credentials unavailable", exc,
                           operation="read_credentials", decision="raise_validator_error")
            raise ValidatorError(f"credentials unavailable: {type(exc).__name__}") from None
        if not _ACCOUNT_SID_RE.match(account_sid or ""):
            raise ValidatorError("account_sid in secret is malformed")
        basic = base64.b64encode(f"{username}:{password}".encode()).decode()
        url = self._url_template.format(account_sid=account_sid)
        _require_https(url)  # the template was checked at construction; the formatted URL is checked too
        req = urllib.request.Request(  # noqa: S310 (https checked just above)
            url,
            data=json.dumps({"token": token}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Basic {basic}"},
            method="POST",
        )
        try:
            with timed_call("twilio_iam", "tokens.validate", expected_codes=("400", "404", "422")), \
                    self._opener(req, self._timeout_s) as resp:
                status = getattr(resp, "status", 200)
                body = resp.read()
        except urllib.error.HTTPError as exc:
            # 4xx with a body can still be a "not valid" answer; 5xx/401 means no decision.
            if 400 <= exc.code < 500 and exc.code not in (401, 403, 429):
                log.info("flex authorizer: token rejected by Twilio", extra={
                    "http_status": exc.code, "decision": "deny_invalid_token"})
                raise InvalidToken(f"token rejected (HTTP {exc.code})") from None
            obs.log_caught(log, "flex authorizer: validation endpoint error", exc,
                           operation="tokens.validate", decision="deny_no_decision")
            raise ValidatorError(f"validation endpoint returned HTTP {exc.code}") from None
        except Exception as exc:
            obs.log_caught(log, "flex authorizer: validation call failed", exc,
                           operation="tokens.validate", decision="deny_no_decision")
            raise ValidatorError(f"validation call failed: {type(exc).__name__}") from None
        if status not in (200, 201):  # the live endpoint answers 201
            raise ValidatorError(f"validation endpoint returned HTTP {status}")
        try:
            data = json.loads(body)
        except ValueError as exc:
            obs.log_caught(log, "flex authorizer: validation response is not JSON", exc,
                           operation="tokens.validate", decision="deny_no_decision")
            raise ValidatorError("validation response is not JSON") from None
        if not isinstance(data, Mapping):
            raise ValidatorError("validation response is not an object")
        if data.get("valid") is not True:
            raise InvalidToken(f"token not valid (code {data.get('code')!r})")
        worker_sid = data.get("worker_sid")
        identity = data.get("identity")
        if not isinstance(worker_sid, str) or not worker_sid.startswith("WK"):
            raise InvalidToken("token has no TaskRouter worker")
        roles = data.get("roles") or []
        return FlexIdentity(
            worker_sid=worker_sid,
            identity=identity if isinstance(identity, str) else "",
            roles=tuple(r for r in roles if isinstance(r, str)) if isinstance(roles, list) else (),
        )


def credentials_from_secret(arn: str | None) -> Callable[[], tuple[str, str, str]]:
    """Read ``(account_sid, api_key, api_secret)`` from the ``twilio/api-key`` secret."""

    def read() -> tuple[str, str, str]:
        from owlbank.secrets import get_secret

        value = get_secret(arn or "", logical="twilio_api_key")
        try:
            return str(value["account_sid"]), str(value["api_key"]), str(value["api_secret"])
        except KeyError as exc:
            obs.log_caught(log, "flex authorizer: api-key secret is incomplete", exc,
                           operation="read_credentials", decision="raise_validator_error")
            raise ValidatorError("twilio/api-key secret lacks account_sid/api_key/api_secret") from None

    return read


# --------------------------------------------------------------------------- authorizer


def _header(event: Mapping[str, Any], name: str) -> str | None:
    headers = event.get("headers") or {}
    found = [v for k, v in headers.items() if isinstance(k, str) and k.lower() == name]
    multi = event.get("multiValueHeaders") or {}
    for k, values in multi.items():
        if isinstance(k, str) and k.lower() == name and isinstance(values, list):
            found.extend(values)
    distinct = {v for v in found if isinstance(v, str)}
    if len(distinct) != 1:
        return None  # missing, or ambiguous duplicates: refuse
    return distinct.pop()


def _policy(principal: str, effect: str, resource: str,
            context: Mapping[str, str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "principalId": principal,
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [{"Action": "execute-api:Invoke", "Effect": effect, "Resource": resource}],
        },
    }
    if context:
        out["context"] = dict(context)
    return out


def deny(event: Mapping[str, Any]) -> dict[str, Any]:
    arn = event.get("methodArn")
    return _policy("unauthorized", "Deny", arn if isinstance(arn, str) and arn else "*")


def context_route_arn(method_arn: str) -> str | None:
    """``GET /context/*`` on the same API and stage, or ``None`` for any other route."""
    m = _METHOD_ARN_RE.match(method_arn or "")
    if not m or m["method"] != ALLOWED_METHOD or not m["path"].startswith(ALLOWED_PATH_PREFIX):
        return None
    return (f"arn:{m['partition']}:execute-api:{m['region']}:{m['account']}:"
            f"{m['api']}/{m['stage']}/{ALLOWED_METHOD}/{ALLOWED_PATH_PREFIX}*")


def authorize(event: Mapping[str, Any], validator: FlexTokenValidator) -> dict[str, Any]:
    """The authorizer decision for one REQUEST event. Never raises."""
    try:
        resource = context_route_arn(str(event.get("methodArn") or ""))
        if resource is None:
            log.warning("flex authorizer: route not allowed")
            return deny(event)
        token = _header(event, TOKEN_HEADER)
        if not token:
            log.info("flex authorizer: no token")
            return deny(event)
        token = token.strip()
        if len(token) > MAX_TOKEN_LEN or not _TOKEN_RE.match(token):
            log.info("flex authorizer: malformed token")
            return deny(event)
        ident = validator.validate(token)
        log.info("flex authorizer: allow", extra={"worker_sid": ident.worker_sid})
        return _policy(ident.worker_sid, "Allow", resource, {
            "worker_sid": ident.worker_sid,
            "identity": ident.identity,
            "roles": ",".join(ident.roles),
        })
    except InvalidToken as exc:
        log.info("flex authorizer: invalid token", extra={"reason": str(exc)[:80], "decision": "deny"})
    except ValidatorError as exc:
        log.warning("flex authorizer: validator error", extra={"reason": str(exc)[:80], "decision": "deny"})
        obs.degraded("validator_error")
    except Exception as exc:  # fail closed on anything unexpected
        obs.log_caught(log, "flex authorizer: unexpected error", exc, level=logging.ERROR, decision="deny")
        obs.degraded("authorizer_error")
    return deny(event)


_default_validator: FlexTokenValidator | None = None


def _validator_from_env() -> FlexTokenValidator:
    global _default_validator
    if _default_validator is None:
        from owlbank.config import Settings

        settings = Settings.from_env()
        _default_validator = TwilioIamTokenValidator(
            credentials_from_secret(settings.secrets.twilio_api_key),
            url_template=os.environ.get("OWLBANK_FLEX_TOKEN_VALIDATE_URL") or DEFAULT_VALIDATE_URL,
        )
    return _default_validator


@obs.instrumented("flex_authorizer", op="authorize")
def handler(event: Mapping[str, Any], context: Any = None) -> dict[str, Any]:
    """Lambda entry point."""
    try:
        validator = _validator_from_env()
    except Exception as exc:
        obs.log_caught(log, "flex authorizer: setup failed", exc, level=logging.ERROR,
                       operation="validator_from_env", decision="deny")
        obs.degraded("setup_failed")
        return deny(event)
    return authorize(event, validator)
