"""Unit tests for lambdas.flex_authorizer (Context API authentication)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from lambdas.flex_authorizer.handler import (
    FlexIdentity,
    InvalidToken,
    TwilioIamTokenValidator,
    ValidatorError,
    authorize,
    context_route_arn,
)

API = "arn:aws:execute-api:us-east-1:123456789012:abc123/prod"
METHOD_ARN = f"{API}/GET/context/ref-1"
ROUTE = f"{API}/GET/context/*"
ACCOUNT = "AC" + "a" * 32
TOKEN = "eyJ6aXAiOiJERUYiLCJraWQiOiJTQVNfUzM.abc-def_ghi"


class FakeValidator:
    def __init__(self, result: FlexIdentity | Exception) -> None:
        self.result = result
        self.calls: list[str] = []

    def validate(self, token: str) -> FlexIdentity:
        self.calls.append(token)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def event(token: str | None = TOKEN, method_arn: str = METHOD_ARN, **headers: str) -> dict:
    h = dict(headers)
    if token is not None:
        h["X-Flex-Token"] = token
    return {"type": "REQUEST", "methodArn": method_arn, "headers": h}


def effect(policy: dict) -> str:
    return policy["policyDocument"]["Statement"][0]["Effect"]


def resource(policy: dict) -> str:
    return policy["policyDocument"]["Statement"][0]["Resource"]


AGENT = FlexIdentity(worker_sid="WK" + "1" * 32, identity="ana", roles=("agent",))


def test_valid_token_allows_context_route_only() -> None:
    v = FakeValidator(AGENT)
    policy = authorize(event(), v)
    assert effect(policy) == "Allow"
    assert resource(policy) == ROUTE
    assert policy["principalId"] == AGENT.worker_sid
    assert policy["context"] == {"worker_sid": AGENT.worker_sid, "identity": "ana", "roles": "agent"}
    assert v.calls == [TOKEN]
    assert TOKEN not in json.dumps(policy)


def test_roles_reach_the_context_and_the_token_never_does() -> None:
    ident = FlexIdentity(worker_sid="WK" + "2" * 32, identity="sue", roles=("agent", "supervisor"))
    policy = authorize(event(), FakeValidator(ident))
    assert effect(policy) == "Allow"
    assert policy["context"]["roles"] == "agent,supervisor"
    assert policy["context"]["worker_sid"] == ident.worker_sid
    assert set(policy["context"]) == {"worker_sid", "identity", "roles"}
    assert TOKEN not in json.dumps(policy)


def test_no_roles_is_an_empty_string_not_a_missing_key() -> None:
    ident = FlexIdentity(worker_sid="WK" + "2" * 32, identity="x", roles=())
    assert authorize(event(), FakeValidator(ident))["context"]["roles"] == ""


def test_the_token_is_never_logged(caplog) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    authorize(event(), FakeValidator(AGENT))
    authorize(event(), FakeValidator(InvalidToken("bad")))
    assert TOKEN not in "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)


def test_header_is_case_insensitive() -> None:
    e = {"methodArn": METHOD_ARN, "headers": {"x-flex-token": TOKEN}}
    assert effect(authorize(e, FakeValidator(AGENT))) == "Allow"


def test_invalid_token_denies() -> None:
    policy = authorize(event(), FakeValidator(InvalidToken("nope")))
    assert effect(policy) == "Deny"
    assert "context" not in policy


@pytest.mark.parametrize("token", [None, "", "   "])
def test_missing_token_denies_without_calling_validator(token) -> None:
    v = FakeValidator(AGENT)
    assert effect(authorize(event(token), v)) == "Deny"
    assert v.calls == []


@pytest.mark.parametrize("token", ["bad token", "x" * 9000, "a<script>"])
def test_malformed_token_denies(token) -> None:
    v = FakeValidator(AGENT)
    assert effect(authorize(event(token), v)) == "Deny"
    assert v.calls == []


@pytest.mark.parametrize("exc", [ValidatorError("down"), RuntimeError("boom")])
def test_validator_error_denies(exc) -> None:
    assert effect(authorize(event(), FakeValidator(exc))) == "Deny"


@pytest.mark.parametrize("arn", [
    f"{API}/POST/context/ref-1", f"{API}/GET/voice/incoming", f"{API}/GET/contextual", "garbage", "",
])
def test_other_routes_denied(arn) -> None:
    v = FakeValidator(AGENT)
    assert effect(authorize(event(method_arn=arn), v)) == "Deny"
    assert v.calls == []


def test_duplicate_conflicting_headers_deny() -> None:
    e = {"methodArn": METHOD_ARN, "headers": {"X-Flex-Token": TOKEN},
         "multiValueHeaders": {"X-Flex-Token": [TOKEN, "other.token"]}}
    assert effect(authorize(e, FakeValidator(AGENT))) == "Deny"


def test_context_route_arn() -> None:
    assert context_route_arn(METHOD_ARN) == ROUTE
    assert context_route_arn(f"{API}/GET/context/a/b") == ROUTE
    assert context_route_arn(f"{API}/DELETE/context/x") is None


# --------------------------------------------------------------------------- Twilio validator


class FakeResp(io.BytesIO):
    def __init__(self, body: dict | bytes, status: int = 200) -> None:
        super().__init__(body if isinstance(body, bytes) else json.dumps(body).encode())
        self.status = status


def make_validator(response):
    seen = {}

    def opener(req, timeout):
        seen["url"], seen["body"], seen["auth"] = req.full_url, req.data, req.get_header("Authorization")
        if isinstance(response, Exception):
            raise response
        return response

    return TwilioIamTokenValidator(lambda: (ACCOUNT, "SKkey", "secret"), opener=opener), seen


@pytest.mark.parametrize("status", [200, 201])  # Twilio answers 201 live
def test_twilio_validator_valid(status) -> None:
    v, seen = make_validator(FakeResp({"valid": True, "worker_sid": "WK" + "2" * 32,
                                       "identity": "ben", "roles": ["agent", "supervisor"]}, status=status))
    ident = v.validate(TOKEN)
    assert ident == FlexIdentity("WK" + "2" * 32, "ben", ("agent", "supervisor"))
    assert seen["url"] == f"https://iam.twilio.com/v1/Accounts/{ACCOUNT}/Tokens/validate"
    assert json.loads(seen["body"]) == {"token": TOKEN}
    assert seen["auth"].startswith("Basic ")


@pytest.mark.parametrize("status", [200, 201])
@pytest.mark.parametrize("body", [{"valid": False, "code": 20101}, {"valid": True}, {"valid": "true"}])
def test_twilio_validator_invalid(body, status) -> None:
    v, _ = make_validator(FakeResp(body, status=status))
    with pytest.raises(InvalidToken):
        v.validate(TOKEN)


@pytest.mark.parametrize("resp", [
    FakeResp(b"not json"), FakeResp({"valid": True}, status=500), FakeResp({"valid": True}, status=202),
    FakeResp({"valid": True}, status=204), TimeoutError(),
    urllib.error.HTTPError("u", 503, "x", {}, None), urllib.error.HTTPError("u", 401, "x", {}, None),
])
def test_twilio_validator_errors(resp) -> None:
    v, _ = make_validator(resp)
    with pytest.raises(ValidatorError):
        v.validate(TOKEN)


def test_twilio_validator_secret_failure() -> None:
    def broken():
        raise KeyError("x")

    with pytest.raises(ValidatorError):
        TwilioIamTokenValidator(broken).validate(TOKEN)


def test_twilio_validator_end_to_end_deny_on_error() -> None:
    v, _ = make_validator(TimeoutError())
    assert effect(authorize(event(), v)) == "Deny"


# --------------------------------------------------------------------------- https-only validation URL


@pytest.mark.parametrize("url", [
    "http://iam.twilio.com/v1/Accounts/{account_sid}/Tokens/validate",
    "file:///etc/passwd?{account_sid}",
    "ftp://iam.twilio.com/{account_sid}",
    "HTTP://iam.twilio.com/{account_sid}",
    "https:///no-host/{account_sid}",
    "//iam.twilio.com/{account_sid}",
])
def test_validator_refuses_a_non_https_url_template(url: str) -> None:
    with pytest.raises(ValueError, match="https"):
        TwilioIamTokenValidator(lambda: (ACCOUNT, "SKkey", "secret"), url_template=url)


def test_default_opener_refuses_non_https_before_opening(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from lambdas.flex_authorizer import handler as h

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("urlopen must not be reached")

    monkeypatch.setattr(h.urllib.request, "urlopen", boom)
    for url in ("file:///etc/passwd", "http://iam.twilio.com/x", "ftp://example.com/x"):
        with pytest.raises(ValueError, match="https"):
            h._default_opener(urllib.request.Request(url), 3.0)  # noqa: S310 (never opened: the opener refuses first)


def test_default_opener_opens_https_with_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    from lambdas.flex_authorizer import handler as h

    seen: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> str:
        seen["url"], seen["timeout"] = req.full_url, timeout
        return "resp"

    monkeypatch.setattr(h.urllib.request, "urlopen", fake_urlopen)
    req = urllib.request.Request(h.DEFAULT_VALIDATE_URL.format(account_sid=ACCOUNT), method="POST")  # noqa: S310 (https, urlopen is faked)
    assert h._default_opener(req, 2.5) == "resp"
    assert seen == {"url": f"https://iam.twilio.com/v1/Accounts/{ACCOUNT}/Tokens/validate", "timeout": 2.5}
