"""The URL signer for :class:`lambdas.frontdoor.relay.RuntimeUrlSigner`.

URL shape: for any CallSid, nonce, clock, TTL and credential lifetime, a
minted URL has ``X-Amz-Expires`` = TTL <= 300, the CallSid as its session parameter, no nonce and
no ``n`` parameter (adding one, or changing any query byte, makes the reference SigV4 verifier
fail), the Region's host and the ``/runtimes/<encoded ARN>/ws`` path, and was signed with
credentials that had at least TTL + 60 s left; STS is called exactly when the cached credentials
have less than that left.

No leak: over any sequence of mints, signing failures, refreshes and clock moves,
nothing any logger captures (botocore at DEBUG included) contains a signature, a credential, a
session token or a nonce. No AWS call: fake credentials only.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.relay import MAX_TTL_S, PresignError, RuntimeUrlSigner
from tests import sigv4_ref
from tests.logutil import text_of

REGION = "us-east-1"
ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/owlbank_dev_bridge-AbC123xyz0"
ROLE = "arn:aws:iam::123456789012:role/owlbank-dev-relay-presign"
HOST = f"bedrock-agentcore.{REGION}.amazonaws.com"
PATH = f"/runtimes/{quote(ARN, safe='')}/ws"
URLSAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
SECRET_CHARS = URLSAFE + "+/="

call_sids = st.text("0123456789abcdef", min_size=32, max_size=32).map(lambda h: "CA" + h)
nonces = st.text(URLSAFE, min_size=16, max_size=64)  # token_urlsafe(24) is 32 characters
ttls = st.integers(1, MAX_TTL_S)
clocks = st.floats(1.6e9, 2.2e9, allow_nan=False)
secrets_ = st.text(SECRET_CHARS, min_size=24, max_size=48)
tokens = st.text(SECRET_CHARS, min_size=24, max_size=80)
lifetimes = st.floats(0, 4 * 3600, allow_nan=False)  # seconds an issued credential lives


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeSts:
    """Issues numbered credentials, each living the next of ``lifetimes`` (the last repeats)."""

    def __init__(self, clock: Clock, lifetimes: list[float], secret: str, token: str) -> None:
        self.clock, self.lifetimes, self.secret, self.token = clock, lifetimes, secret, token
        self.fail = False
        self.issued: dict[str, float] = {}  # access key -> expiry

    def assume_role(self, **_: str) -> dict:
        if self.fail:
            raise ConnectionError(f"endpoint unreachable {self.token}")
        n = len(self.issued)
        life = self.lifetimes[min(n, len(self.lifetimes) - 1)]
        akid = f"ASIAFAKE{n:012d}"
        expiration = datetime.fromtimestamp(self.clock() + life, tz=UTC)
        # The expiry the signer sees: STS's datetime has microsecond precision, so the oracle uses
        # the same rounded value (an exact float can sit a microsecond past the TTL + 60 s line).
        self.issued[akid] = expiration.timestamp()
        return {"Credentials": {"AccessKeyId": akid, "SecretAccessKey": self.secret,
                                "SessionToken": f"{self.token}{n}", "Expiration": expiration}}


def make(clock: Clock, sts: FakeSts, ttl: int) -> RuntimeUrlSigner:
    return RuntimeUrlSigner(runtime_arn=ARN, region=REGION, clock=clock, role_arn=ROLE, sts=sts,
                            ttl_s=ttl)


def params(url: str) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))


def with_param(url: str, name: str, value: str | None) -> str:
    parts = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != name]
    if value is not None:
        q.insert(1, (name, value))
    return urlunsplit(parts._replace(query=urlencode(q, quote_via=quote, safe="")))


def verify(url: str, secret: str) -> bool:
    return sigv4_ref.verify(url, secret_key=secret, region=REGION, service="bedrock-agentcore")


@settings(max_examples=150, deadline=None)
@given(sid=call_sids, nonce=nonces, other=nonces, ttl=ttls, now=clocks, secret=secrets_,
       token=tokens, lives=st.lists(lifetimes, min_size=1, max_size=3),
       steps=st.lists(st.floats(0, 3600, allow_nan=False), min_size=1, max_size=6))
def test_minted_url_shape(sid: str, nonce: str, other: str, ttl: int, now: float, secret: str,
                          token: str, lives: list[float], steps: list[float]) -> None:
    clock = Clock(now)
    sts = FakeSts(clock, lives, secret, token)
    signer = make(clock, sts, ttl)
    needed = ttl + 60
    for step in steps:
        clock.now += step
        before = dict(sts.issued)
        cached = list(before.values())[-1] if before else None  # the newest replaces the cache
        cached_ok = cached is not None and cached - clock() >= needed
        try:
            url = signer.mint(call_sid=sid, nonce=nonce)
        except PresignError as exc:
            assert exc.error_class == "CredentialsExpireTooSoon"
            newest = list(sts.issued.values())[-1]
            assert newest - clock() < needed
            continue
        # STS is called iff the cached credentials had less than TTL + 60 s left
        assert (len(sts.issued) == len(before)) == cached_ok
        q = params(url)
        parts = urlsplit(url)
        assert parts.scheme == "wss" and parts.netloc == HOST and parts.path == PATH
        assert int(q["X-Amz-Expires"]) == ttl <= 300
        assert q["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == sid
        assert "n" not in q and nonce not in q.values()
        akid = q["X-Amz-Credential"].split("/")[0]
        assert sts.issued[akid] - clock() >= needed
        assert verify(url, secret)
        assert not verify(with_param(url, "n", nonce), secret)  # an added n isn't covered
        assert not verify(with_param(url, "n", other), secret)
        # the nonce argument has no effect on the URL
        # botocore stamps X-Amz-Date from the real clock, so a second mint can land in the next
        # second: compare everything but the date and the signature that depends on it.
        def _stable(u: str) -> dict:
            return {k: v for k, v in params(u).items() if k not in ("X-Amz-Date", "X-Amz-Signature")}
        again = signer.mint(call_sid=sid, nonce=other)
        assert urlsplit(again)[:3] == urlsplit(url)[:3] and _stable(again) == _stable(url)
        assert not verify(with_param(url, "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id",
                                     "CA" + "f" * 32 if sid != "CA" + "f" * 32 else "CA" + "e" * 32),
                          secret)


ops = st.lists(st.one_of(
    st.tuples(st.just("mint"), nonces),
    st.tuples(st.just("fail"), st.booleans()),
    st.tuples(st.just("refresh"), st.integers(0, 3600)),
    st.tuples(st.just("advance"), st.floats(0, 3600, allow_nan=False)),
), min_size=1, max_size=12)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(sid=call_sids, ttl=ttls, now=clocks, secret=secrets_, token=tokens,
       lives=st.lists(lifetimes, min_size=1, max_size=4), seq=ops)
def test_signer_never_logs_url_secrets(caplog: pytest.LogCaptureFixture, sid: str, ttl: int,
                                            now: float, secret: str, token: str, lives: list[float],
                                            seq: list[tuple[str, object]]) -> None:
    caplog.clear()
    clock = Clock(now)
    sts = FakeSts(clock, lives, secret, token)
    signer = make(clock, sts, ttl)
    secrets: set[str] = {secret, token}
    botocore = logging.getLogger("botocore")
    botocore.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.DEBUG):
            for op, arg in seq:
                if op == "mint":
                    secrets.add(str(arg))
                    try:
                        url = signer.mint(call_sid=sid, nonce=str(arg))
                    except PresignError as exc:
                        assert exc.error_class in {"ConnectionError", "CredentialsExpireTooSoon"}
                        assert str(exc) == exc.error_class
                        continue
                    q = params(url)
                    secrets.update({q["X-Amz-Signature"], q["X-Amz-Credential"],
                                    q["X-Amz-Credential"].split("/")[0], q["X-Amz-Security-Token"]})
                elif op == "fail":
                    sts.fail = bool(arg)
                elif op == "refresh":
                    signer.refresh_if_due(margin_s=int(str(arg)))
                else:
                    clock.now += float(str(arg))
    finally:
        botocore.setLevel(logging.NOTSET)
    text = text_of(caplog.records)
    leaked = [s for s in secrets if s in text]
    assert not leaked
