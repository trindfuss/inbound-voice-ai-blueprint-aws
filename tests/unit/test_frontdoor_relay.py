"""Relay URLs: the SigV4 reference verifier, URL shape, PresignError,
credential refresh timing and the CallSid format. No AWS call: fixed fake credentials throughout."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qsl, quote, urlsplit

import botocore.auth
import pytest
from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials
from botocore.exceptions import ClientError

from lambdas.frontdoor import relay
from lambdas.frontdoor.relay import (
    CALL_SID_RE,
    PresignError,
    RuntimeUrlSigner,
    SigningCredentials,
    StaticRelayUrl,
)
from owlbank import ws_nonce
from tests import sigv4_ref
from tests.logutil import lines, text_of

REGION = "us-east-1"
ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/owlbank_dev_bridge-AbC123xyz0"
SID = "CA" + "0123456789abcdef" * 2
NONCE = "nOnCe-_0123456789abcdefghijklmnop"
AKID = "AKIAIOSFODNN7EXAMPLE"
SECRET = "fake/Secret+Key=0123456789abcdefghijklmn"
TOKEN = "fake+Session/Token=0123456789abcdefghij"
T0 = 1_800_000_000.0

# ---------------------------------------------------------------------------- fakes


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeSts:
    """``assume_role`` returning fake credentials that live ``duration_s`` from the clock."""

    def __init__(self, clock: Clock, duration_s: float = 3600, error: Exception | None = None) -> None:
        self.clock, self.duration_s, self.error = clock, duration_s, error
        self.calls: list[dict[str, str]] = []

    def assume_role(self, **kwargs: str) -> dict:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        n = len(self.calls)
        return {"Credentials": {
            "AccessKeyId": f"{AKID[:-2]}{n:02d}", "SecretAccessKey": SECRET, "SessionToken": TOKEN,
            "Expiration": datetime.fromtimestamp(self.clock() + self.duration_s, tz=UTC)}}


def signer(clock: Clock | None = None, sts: FakeSts | None = None, **kw: object) -> RuntimeUrlSigner:
    clock = clock or Clock()
    return RuntimeUrlSigner(runtime_arn=ARN, region=REGION, clock=clock,
                            role_arn="arn:aws:iam::123456789012:role/owlbank-dev-relay-presign",
                            sts=sts or FakeSts(clock), **kw)  # type: ignore[arg-type]


def query(url: str) -> dict[str, str]:
    pairs = parse_qsl(urlsplit(url).query, keep_blank_values=True)
    assert len(pairs) == len({k for k, _ in pairs})
    return dict(pairs)


def verify(url: str, secret: str = SECRET) -> bool:
    return sigv4_ref.verify(url, secret_key=secret, region=REGION, service="bedrock-agentcore")


# ---------------------------------------------------------------------------- reference verifier

FIXED_TIME = datetime(2026, 10, 16, 12, 0, 0)
VECTOR_URL = (f"https://bedrock-agentcore.{REGION}.amazonaws.com/runtimes/{quote(ARN, safe='')}/ws"
              f"?X-Amzn-Bedrock-AgentCore-Runtime-Session-Id={SID}&n={NONCE}")


@pytest.fixture
def vector(monkeypatch: pytest.MonkeyPatch) -> str:
    """One URL signed by botocore itself (not through the signer), at a fixed time."""
    monkeypatch.setattr(botocore.auth, "get_current_datetime", lambda: FIXED_TIME)
    req = AWSRequest(method="GET", url=VECTOR_URL, headers={"host": urlsplit(VECTOR_URL).hostname})
    SigV4QueryAuth(ReadOnlyCredentials(AKID, SECRET, TOKEN), "bedrock-agentcore", REGION,
                   expires=300).add_auth(req)
    return str(req.url).replace("https://", "wss://", 1)


def test_vector_is_stable(vector: str) -> None:
    # Pins the botocore output the verifier was checked against.
    assert query(vector)["X-Amz-Signature"] == (
        "c2d380071e549e3715ab3d37a2e0fdd013c62dca031d06e0a125b91dd64391ab")
    assert query(vector)["X-Amz-Date"] == "20261016T120000Z"


def test_verifier_accepts_the_botocore_vector(vector: str) -> None:
    assert verify(vector)
    assert verify(vector.replace("wss://", "https://", 1))
    signed_at = FIXED_TIME.replace(tzinfo=UTC)
    assert sigv4_ref.verify(vector, secret_key=SECRET, region=REGION, service="bedrock-agentcore",
                            now=signed_at + timedelta(seconds=299))
    assert not sigv4_ref.verify(vector, secret_key=SECRET, region=REGION, service="bedrock-agentcore",
                                now=signed_at + timedelta(seconds=301))


def test_verifier_rejects_wrong_key_region_or_service(vector: str) -> None:
    assert not verify(vector, secret=SECRET + "x")
    assert not sigv4_ref.verify(vector, secret_key=SECRET, region="us-west-2", service="bedrock-agentcore")
    assert not sigv4_ref.verify(vector, secret_key=SECRET, region=REGION, service="bedrock")


def _hex_case_equivalent(a: str, b: str) -> bool:
    """Same request once query escapes are upper-cased (``%2f`` and ``%2F`` decode alike)."""
    import re

    def norm(u: str) -> str:
        head, _, q = u.partition("?")
        return head + "?" + re.sub(r"%[0-9a-fA-F]{2}", lambda m: m.group(0).upper(), q)

    return norm(a) == norm(b)


def test_verifier_rejects_every_single_byte_change(vector: str) -> None:
    start = vector.index("://") + 3  # the scheme isn't signed (wss vs https)
    checked = 0
    for i in range(start, len(vector)):
        for repl in {"x" if vector[i] != "x" else "y", "%", "A" if vector[i] != "A" else "B"}:
            mutated = vector[:i] + repl + vector[i + 1:]
            if _hex_case_equivalent(mutated, vector):
                continue
            assert not verify(mutated), (i, repl)
            checked += 1
        deleted = vector[:i] + vector[i + 1:]
        assert not verify(deleted), i
    assert checked > 2 * (len(vector) - start)


# ---------------------------------------------------------------------------- URL shape


def test_url_shape() -> None:
    url = signer().mint(call_sid=SID, nonce=NONCE)
    parts = urlsplit(url)
    assert parts.scheme == "wss"
    assert parts.netloc == "bedrock-agentcore.us-east-1.amazonaws.com"
    assert parts.path == "/runtimes/" + quote(ARN, safe="") + "/ws"
    q = query(url)
    assert q["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == SID
    assert "n" not in q and NONCE not in url  # the nonce travels as <Parameter name="n">
    assert set(q) == {"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id", "X-Amz-Algorithm",
                      "X-Amz-Credential", "X-Amz-Date", "X-Amz-Expires", "X-Amz-SignedHeaders",
                      "X-Amz-Security-Token", "X-Amz-Signature"}
    assert q["X-Amz-Expires"] == "300"
    assert q["X-Amz-SignedHeaders"] == "host"
    assert q["X-Amz-Security-Token"] == TOKEN
    assert q["X-Amz-Credential"].split("/")[2:] == [REGION, "bedrock-agentcore", "aws4_request"]
    assert verify(url)


@pytest.mark.parametrize("nonce", ["", NONCE, "x" * (ws_nonce.MAX_NONCE_LEN + 1)])
def test_the_nonce_argument_never_reaches_the_url(nonce: str) -> None:
    clock = Clock()
    s = signer(clock)
    url = s.mint(call_sid=SID, nonce=nonce)
    assert "n" not in query(url) and verify(url)
    assert url == s.mint(call_sid=SID, nonce="other")  # same second, same URL: only the nonce differs


def test_shorter_ttl_is_signed() -> None:
    url = signer(ttl_s=120).mint(call_sid=SID, nonce=NONCE)
    assert query(url)["X-Amz-Expires"] == "120"
    assert verify(url)


def test_no_session_token_means_no_token_parameter() -> None:
    creds = SigningCredentials(AKID, SECRET, None, T0 + 3600)
    s = RuntimeUrlSigner(runtime_arn=ARN, region=REGION, clock=Clock(), credential_provider=lambda: creds)
    url = s.mint(call_sid=SID, nonce=NONCE)
    assert "X-Amz-Security-Token" not in query(url)
    assert verify(url)


def test_static_relay_url_is_todays_url() -> None:
    base = "wss://relay.example.com/ws"
    assert StaticRelayUrl(base).mint(call_sid=SID, nonce=NONCE) == ws_nonce.with_nonce(base, NONCE)


@pytest.mark.parametrize("kw", [
    {"ttl_s": 0}, {"ttl_s": 301}, {"ttl_s": True}, {"ttl_s": 300.0},
])
def test_constructor_rejects_bad_settings(kw: dict) -> None:
    with pytest.raises(ValueError):
        signer(**kw)


def test_constructor_rejects_bad_arn_region_and_credential_sources() -> None:
    clock = Clock()
    creds = SigningCredentials(AKID, SECRET, TOKEN, T0 + 3600)
    with pytest.raises(ValueError):
        RuntimeUrlSigner(runtime_arn=ARN, region="us-west-2", clock=clock, credential_provider=lambda: creds)
    with pytest.raises(ValueError):
        RuntimeUrlSigner(runtime_arn="arn:aws:lambda:us-east-1:123456789012:function:x", region=REGION,
                         clock=clock, credential_provider=lambda: creds)
    with pytest.raises(ValueError):
        RuntimeUrlSigner(runtime_arn=ARN, region="US-EAST-1", clock=clock, credential_provider=lambda: creds)
    with pytest.raises(ValueError):
        RuntimeUrlSigner(runtime_arn=ARN, region=REGION, clock=clock)
    with pytest.raises(ValueError):
        RuntimeUrlSigner(runtime_arn=ARN, region=REGION, clock=clock, sts=FakeSts(clock),
                         role_arn="arn:aws:iam::123456789012:role/r", credential_provider=lambda: creds)


@pytest.mark.parametrize("sid", ["", "CA123", "CA" + "0" * 31, "CA" + "0" * 33, "CA" + "A" * 32,
                                 "MZ" + "0" * 32, "CA" + "0" * 32 + "\n", None])
def test_mint_rejects_anything_but_a_callsid(sid: object) -> None:
    sts = FakeSts(Clock())
    with pytest.raises(ValueError):
        signer(sts=sts).mint(call_sid=sid, nonce=NONCE)  # type: ignore[arg-type]
    assert sts.calls == []


# ---------------------------------------------------------------------------- CallSid = session ID

# AgentCore Runtime session IDs: 33 to 256 characters (service quotas page).
SESSION_ID_MIN, SESSION_ID_MAX = 33, 256


def test_callsid_format_meets_session_id_limits() -> None:
    assert CALL_SID_RE.match(SID)
    assert SESSION_ID_MIN <= len(SID) <= SESSION_ID_MAX
    # every string the pattern admits has exactly 34 characters
    assert CALL_SID_RE.pattern == r"^CA[0-9a-f]{32}\Z"
    assert not CALL_SID_RE.match(SID[:-1])
    assert not CALL_SID_RE.match(SID + "0")
    assert not CALL_SID_RE.match(SID + "\n")


# ---------------------------------------------------------------------------- PresignError


def test_sts_failure_raises_presign_error_with_the_class_only() -> None:
    err = ClientError({"Error": {"Code": "AccessDenied", "Message": "secret detail " + TOKEN}},
                      "AssumeRole")
    with pytest.raises(PresignError) as info:
        signer(sts=FakeSts(Clock(), error=err)).mint(call_sid=SID, nonce=NONCE)
    assert info.value.error_class == "ClientError"
    assert str(info.value) == "ClientError"
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_malformed_sts_response_is_a_presign_error() -> None:
    class Broken:
        def assume_role(self, **_: str) -> dict:
            return {"Credentials": {"AccessKeyId": AKID}}

    with pytest.raises(PresignError) as info:
        signer(sts=Broken()).mint(call_sid=SID, nonce=NONCE)  # type: ignore[arg-type]
    assert info.value.error_class == "KeyError"


def test_naive_expiration_is_a_presign_error() -> None:
    class Naive:
        def assume_role(self, **_: str) -> dict:
            return {"Credentials": {"AccessKeyId": AKID, "SecretAccessKey": SECRET, "SessionToken": TOKEN,
                                    "Expiration": datetime(2030, 1, 1)}}

    with pytest.raises(PresignError) as info:
        signer(sts=Naive()).mint(call_sid=SID, nonce=NONCE)  # type: ignore[arg-type]
    assert info.value.error_class == "TypeError"


def test_credentials_that_expire_too_soon_are_refused() -> None:
    clock = Clock()
    sts = FakeSts(clock, duration_s=359)  # under TTL + 60 s even when fresh
    with pytest.raises(PresignError) as info:
        signer(clock, sts).mint(call_sid=SID, nonce=NONCE)
    assert info.value.error_class == "CredentialsExpireTooSoon"


def test_assume_role_names_the_presign_role() -> None:
    clock = Clock()
    sts = FakeSts(clock)
    signer(clock, sts).mint(call_sid=SID, nonce=NONCE)
    assert sts.calls == [{"RoleArn": "arn:aws:iam::123456789012:role/owlbank-dev-relay-presign",
                          "RoleSessionName": relay.ROLE_SESSION_NAME}]


# ---------------------------------------------------------------------------- refresh timing


def test_credentials_are_cached_until_ttl_plus_60s_is_left() -> None:
    clock = Clock()
    sts = FakeSts(clock, duration_s=3600)
    s = signer(clock, sts)
    s.mint(call_sid=SID, nonce=NONCE)
    assert len(sts.calls) == 1
    clock.now = T0 + 3600 - 360  # exactly TTL + 60 s left: still good
    s.mint(call_sid=SID, nonce=NONCE)
    assert len(sts.calls) == 1
    clock.now += 1  # one second less
    url = s.mint(call_sid=SID, nonce=NONCE)
    assert len(sts.calls) == 2
    assert query(url)["X-Amz-Credential"].startswith(f"{AKID[:-2]}02/")


def test_refresh_if_due_refreshes_below_15_minutes_only() -> None:
    clock = Clock()
    sts = FakeSts(clock, duration_s=3600)
    s = signer(clock, sts)
    s.refresh_if_due()  # INIT: nothing cached yet
    assert len(sts.calls) == 1
    clock.now = T0 + 3600 - 900
    s.refresh_if_due()
    assert len(sts.calls) == 1
    clock.now += 1
    s.refresh_if_due()
    assert len(sts.calls) == 2
    s.mint(call_sid=SID, nonce=NONCE)  # the INIT/opportunistic refresh serves the webhook
    assert len(sts.calls) == 2


def test_refresh_if_due_failure_is_logged_and_mint_retries(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    sts = FakeSts(clock, error=ClientError({"Error": {"Code": "Throttling", "Message": TOKEN}}, "AssumeRole"))
    s = signer(clock, sts)
    with caplog.at_level(logging.DEBUG):
        s.refresh_if_due()
    [line] = lines(caplog.records, "lambdas.frontdoor.relay")
    assert line["event"] == "relay_presign_refresh_failed"
    assert line["error_class"] == "PresignError"
    assert TOKEN not in text_of(caplog.records)
    sts.error = None
    assert verify(s.mint(call_sid=SID, nonce=NONCE))
    assert len(sts.calls) == 2


def test_injected_credential_provider_is_used() -> None:
    calls = []

    def provide() -> SigningCredentials:
        calls.append(1)
        return SigningCredentials(AKID, SECRET, TOKEN, T0 + 3600)

    s = RuntimeUrlSigner(runtime_arn=ARN, region=REGION, clock=Clock(), credential_provider=provide)
    assert verify(s.mint(call_sid=SID, nonce=NONCE))
    assert calls == [1]


def test_credentials_repr_hides_the_secret() -> None:
    assert SECRET not in repr(SigningCredentials(AKID, SECRET, TOKEN, T0))
    assert TOKEN not in repr(SigningCredentials(AKID, SECRET, TOKEN, T0))


# ---------------------------------------------------------------------------- logging


def test_mint_writes_nothing_even_with_botocore_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    logging.getLogger("botocore").setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.DEBUG):
            url = signer().mint(call_sid=SID, nonce=NONCE)
            logging.getLogger("botocore.auth").debug("outside signing")
    finally:
        logging.getLogger("botocore").setLevel(logging.NOTSET)
    text = text_of(caplog.records)
    for secret in (query(url)["X-Amz-Signature"], TOKEN, NONCE, AKID[:-2]):
        assert secret not in text
    assert "outside signing" in text  # the filter is scoped to signing
