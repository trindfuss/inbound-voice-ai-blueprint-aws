"""Unit and security tests for the auth service Lambda."""

from __future__ import annotations

from tests.logutil import lines as lines_of, text_of

import copy
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.auth_service import handler as auth_handler
from lambdas.auth_service import service
from lambdas.auth_service.service import AuthDeps
from lambdas.auth_service.verify_client import TwilioVerifyClient, VerifyUnavailable
from owlbank import tokens
from owlbank.config import Settings
from owlbank.policy import Policy
from owlbank.store import SCHEMAS, DynamoStore, MemoryStore, Store, create_table_kwargs
from scripts.hash_tool import hash_factor

DATA = Path(__file__).resolve().parents[3] / "data"
FIXTURES = json.loads((DATA / "fixtures.json").read_text(encoding="utf-8"))
FACTORS = json.loads((DATA / "fixtures.test-factors.json").read_text(encoding="utf-8"))["customers"]
POLICY_DOC = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
NOW = 1_700_000_000

THOMAS, SAM, JORDAN, MARIA = "C-1001", "C-1002", "C-1003", "C-1004"
ANI = {THOMAS: "+12025550101", SAM: "+12025550102", JORDAN: "+12025550102",
       MARIA: "+12025550103"}


class FakeVerify:
    """Twilio Verify fake: one valid code per destination."""

    def __init__(self, code: str = "123456", fail: bool = False) -> None:
        self.code, self.fail = code, fail
        self.sent: list[tuple[str, str]] = []
        self.locales: list[str | None] = []
        self.checked: list[str] = []

    def send(self, to: str, channel: str, locale: str | None = None) -> None:
        if self.fail:
            raise VerifyUnavailable("down")
        self.sent.append((to, channel))
        self.locales.append(locale)

    def check(self, to: str, code: str) -> bool:
        if self.fail:
            raise VerifyUnavailable("down")
        self.checked.append(to)
        return code == self.code


class Clock:
    def __init__(self, t: int = NOW) -> None:
        self.t = t

    def __call__(self) -> int:
        return self.t


def seed(store: Store, customers: list[dict[str, Any]] | None = None) -> None:
    for c in customers if customers is not None else FIXTURES["customers"]:
        store.put("customers", copy.deepcopy(c))


def make_deps(store: Store | None = None, *, verify: FakeVerify | None = None,
              clock: Clock | None = None) -> AuthDeps:
    if store is None:
        store = MemoryStore()
        seed(store)
    return AuthDeps(store=store, policy=Policy.from_dict(POLICY_DOC),
                    verify_client=verify if verify is not None else FakeVerify(),
                    clock=clock or Clock(), token_max_ttl_s=1800, record_ttl_s=86400)


@pytest.fixture
def deps() -> AuthDeps:
    return make_deps()


TRUSTED = "TN-Validation-Passed-A"  # StirVerstat of a trusted caller ID


def verify(deps: AuthDeps, cid: str | None = None, *, call_sid: str = "CA1",
           ani: str | None = None, level: int = 1, stir_verstat: str | None = TRUSTED,
           **factors: str) -> dict[str, Any]:
    if cid is not None and not factors:
        factors = {"zip": FACTORS[cid]["zip"]}
        if level >= 2:
            factors["ssn4"] = FACTORS[cid]["ssn4"]
    return service.verify(deps, call_sid=call_sid, ani=ani or (ANI[cid] if cid else None),
                          factors=factors, target_level=level,
                          stir_verstat=stir_verstat)


def check(deps: AuthDeps, token: str, *, call_sid: str = "CA1", min_level: int = 1):
    return tokens.validate(deps.store, token, call_sid=call_sid, min_level=min_level,
                           now=deps.clock())


# --------------------------------------------------------------------------- happy paths


@pytest.mark.parametrize("cid", [THOMAS, SAM, JORDAN, MARIA])
@pytest.mark.parametrize("level", [1, 2])
def test_each_fixture_verifies(deps: AuthDeps, cid: str, level: int) -> None:
    out = verify(deps, cid, level=level)
    assert out["result"] == "verified"
    assert out["level"] == level and out["customer_id"] == cid and out["step_up"] is False
    assert out["factors_verified"] == deps.policy.factors(level)
    claims = check(deps, out["token"], min_level=level)
    assert isinstance(claims, tokens.TokenClaims)
    assert claims.customer_id == cid and claims.level == level and claims.call_sid == "CA1"


def test_token_is_stored_only_as_hash(deps: AuthDeps) -> None:
    out = verify(deps, THOMAS)
    item = deps.store.get("tokens", {"token_hash": tokens.hash_token(out["token"])})
    assert item is not None and out["token"] not in json.dumps(item)
    assert item["expires_at"] == NOW + 1800


# --------------------------------------------------------------------------- outcomes


def test_wrong_zip_not_verified(deps: AuthDeps) -> None:
    out = verify(deps, None, ani=ANI[THOMAS], zip="94106")
    assert out == {"result": "not_verified", "level": 0, "reason": "mismatch",
                   "attempts_remaining": 2}


@pytest.mark.parametrize("ani", ["+12025550199", "anonymous", "", None, "12025550101"])
def test_no_match_or_withheld(deps: AuthDeps, ani: str | None) -> None:
    out = service.verify(deps, stir_verstat=TRUSTED, call_sid="CA1", ani=ani, factors={"zip": "94105"}, target_level=1)
    assert out["result"] == "not_verified" and out["reason"] == "no_candidates"


def test_lee_household_disambiguated_by_zip(deps: AuthDeps) -> None:
    sam = verify(deps, SAM, call_sid="CA-S")
    jordan = verify(deps, JORDAN, call_sid="CA-J")
    assert (sam["customer_id"], jordan["customer_id"]) == (SAM, JORDAN)


def test_lee_household_mixed_factors_fail(deps: AuthDeps) -> None:
    """Sam's ZIP with Jordan's SSN matches neither record: factors never mix customers."""
    out = verify(deps, None, ani=ANI[SAM], level=2, zip=FACTORS[SAM]["zip"],
                 ssn4=FACTORS[JORDAN]["ssn4"])
    assert out["result"] == "not_verified" and out["reason"] == "mismatch"


def test_ambiguous_household_needs_disambiguating_factor() -> None:
    """Two records on one ANI with the same ZIP: ZIP alone is ambiguous; adding SSN4 decides."""
    a = {"customer_id": "C-9001", "phone": "+12025550150", "accounts": [],
         "factors": {"zip": hash_factor("11111"), "ssn4": hash_factor("1234")}}
    b = {"customer_id": "C-9002", "phone": "+12025550150", "accounts": [],
         "factors": {"zip": hash_factor("11111"), "ssn4": hash_factor("5678")}}
    store = MemoryStore()
    seed(store, [a, b])
    deps = make_deps(store)
    out = service.verify(deps, stir_verstat=TRUSTED, call_sid="CA1", ani="+12025550150", factors={"zip": "11111"},
                         target_level=1)
    assert out["result"] == "not_verified" and out["reason"] == "ambiguous"
    assert out["attempts_remaining"] == 3  # an ambiguous answer is not a failure
    assert "C-9001" not in json.dumps(out) and "C-9002" not in json.dumps(out)
    out = service.verify(deps, stir_verstat=TRUSTED, call_sid="CA1", ani="+12025550150",
                         factors={"zip": "11111", "ssn4": "5678"}, target_level=1)
    assert out["result"] == "verified" and out["customer_id"] == "C-9002" and out["level"] == 1


# --------------------------------------------------------------------------- levels


@pytest.mark.parametrize("level", [0, 4, -1, 99])
def test_invalid_levels_rejected(deps: AuthDeps, level: int) -> None:
    out = verify(deps, None, ani=ANI[THOMAS], level=level, zip=FACTORS[THOMAS]["zip"])
    assert out["result"] == "invalid_level"
    assert deps.store.query("tokens", {"call_sid": "CA1"}, index="call_sid-index") == []


@pytest.mark.parametrize("level", ["2", True, None, 1.0])
def test_non_int_levels_rejected(deps: AuthDeps, level: Any) -> None:
    out = service.verify(deps, stir_verstat=TRUSTED, call_sid="CA1", ani=ANI[THOMAS], factors={"zip": "94105"},
                         target_level=level)
    assert out["result"] == "invalid_level"


def test_target_not_above_current_rejected(deps: AuthDeps) -> None:
    verify(deps, THOMAS, level=2)
    out = verify(deps, None, ani=ANI[THOMAS], level=2, ssn4="1111")
    assert out == {"result": "invalid_level", "level": 2}


def test_missing_factor_is_bad_request(deps: AuthDeps) -> None:
    out = verify(deps, None, ani=ANI[THOMAS], level=2, zip=FACTORS[THOMAS]["zip"])
    assert out["result"] == "bad_request" and out["missing"] == ["ssn4"]


@pytest.mark.parametrize("factors", [{"ani_match": "x"}, {"pin": "1234"}, {"zip": 94105},
                                     {"zip": ""}, {"zip": "9" * 33}])
def test_bad_factors(deps: AuthDeps, factors: dict[str, Any]) -> None:
    out = service.verify(deps, stir_verstat=TRUSTED, call_sid="CA1", ani=ANI[THOMAS], factors=factors, target_level=1)
    assert out["result"] == "bad_request"


@pytest.mark.parametrize("call_sid", ["", None, "CA#1", "C" * 65])
def test_bad_call_sid(deps: AuthDeps, call_sid: Any) -> None:
    out = service.verify(deps, stir_verstat=TRUSTED, call_sid=call_sid, ani=ANI[THOMAS], factors={"zip": "94105"},
                         target_level=1)
    assert out["result"] == "bad_request"


# --------------------------------------------------------------------------- step-up


def test_step_up_asks_only_missing_factors_and_upgrades(deps: AuthDeps) -> None:
    first = verify(deps, THOMAS, level=1)
    up = verify(deps, None, ani=ANI[THOMAS], level=2, ssn4=FACTORS[THOMAS]["ssn4"])
    assert up["result"] == "verified" and up["level"] == 2 and up["step_up"] is True
    assert up["customer_id"] == THOMAS and up["token"] != first["token"]
    assert check(deps, first["token"]) == tokens.AuthError(tokens.AuthErrorCode.REVOKED)
    assert isinstance(check(deps, up["token"], min_level=2), tokens.TokenClaims)


def test_step_up_to_level_3_with_otp(deps: AuthDeps) -> None:
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    verify(deps, THOMAS, level=2)
    assert service.send_otp(deps, call_sid="CA1", ani=ANI[THOMAS],
                            language="es-US") == {"result": "sent"}
    assert fake.sent == [(ANI[THOMAS], "sms")] and fake.locales == ["es-US"]
    up = verify(deps, None, ani=ANI[THOMAS], level=3, otp="123456")
    assert up["result"] == "verified" and up["level"] == 3
    assert fake.checked == [ANI[THOMAS]]
    assert isinstance(check(deps, up["token"], min_level=3), tokens.TokenClaims)


def test_step_up_1_to_3_needs_ssn4_and_otp(deps: AuthDeps) -> None:
    verify(deps, THOMAS, level=1)
    assert verify(deps, None, ani=ANI[THOMAS], level=3, otp="123456")["missing"] == ["ssn4"]
    up = verify(deps, None, ani=ANI[THOMAS], level=3, ssn4="1111", otp="123456")
    assert up["result"] == "verified" and up["level"] == 3


def test_wrong_otp_fails_and_keeps_level(deps: AuthDeps) -> None:
    first = verify(deps, THOMAS, level=2)
    out = verify(deps, None, ani=ANI[THOMAS], level=3, otp="000000")
    assert out["result"] == "not_verified" and out["level"] == 2
    assert isinstance(check(deps, first["token"], min_level=2), tokens.TokenClaims)


def test_otp_skipped_when_hashed_factors_fail(deps: AuthDeps) -> None:
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    verify(deps, THOMAS, level=1)
    out = verify(deps, None, ani=ANI[THOMAS], level=3, ssn4="0000", otp="123456")
    assert out["result"] == "not_verified" and fake.checked == []


@pytest.mark.parametrize("client, sent", [
    (None, {"result": "unavailable"}),
    (FakeVerify(fail=True), {"result": "escalate", "reason": "otp_unavailable"}),
])
def test_verify_unavailable(client: FakeVerify | None, sent: dict[str, str]) -> None:
    deps = make_deps()
    deps.verify_client = client
    verify(deps, THOMAS, level=2)
    assert verify(deps, None, ani=ANI[THOMAS], level=3, otp="123456")["result"] == "unavailable"
    assert service.send_otp(deps, call_sid="CA1", ani=ANI[THOMAS]) == sent


def test_step_up_cannot_switch_customer(deps: AuthDeps) -> None:
    """Sam is verified at level 1; Jordan's SSN4 on the shared phone doesn't step up."""
    verify(deps, SAM, level=1)
    out = verify(deps, None, ani=ANI[SAM], level=2, ssn4=FACTORS[JORDAN]["ssn4"])
    assert out["result"] == "not_verified"
    up = verify(deps, None, ani=ANI[SAM], level=2, ssn4=FACTORS[SAM]["ssn4"])
    assert up["customer_id"] == SAM


def test_expired_token_means_initial_verification(deps: AuthDeps) -> None:
    verify(deps, THOMAS, level=1)
    deps.clock.t += 1801  # type: ignore[attr-defined]
    out = verify(deps, None, ani=ANI[THOMAS], level=2, ssn4="1111")
    assert out["result"] == "bad_request" and out["missing"] == ["zip"]


def test_send_otp_needs_verified_call(deps: AuthDeps) -> None:
    assert service.send_otp(deps, call_sid="CA1") == {"result": "not_verified"}
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    assert fake.sent == []


def test_send_otp_rate_limited(deps: AuthDeps) -> None:
    verify(deps, THOMAS, level=2)
    results = [service.send_otp(deps, call_sid="CA1", ani=ANI[THOMAS])["result"]
               for _ in range(4)]
    assert results == ["sent", "sent", "sent", "rate_limited"]


# --------------------------------------------------------------------------- rate limits


def test_per_call_limit_exhaustion(deps: AuthDeps) -> None:
    outs = [verify(deps, None, ani=ANI[THOMAS], zip="00000") for _ in range(3)]
    assert [o["attempts_remaining"] for o in outs] == [2, 1, 0]
    assert verify(deps, THOMAS)["result"] == "rate_limited"  # even the right ZIP
    assert verify(deps, THOMAS, call_sid="CA2")["result"] == "verified"  # other call is fine


def test_success_does_not_consume_attempts(deps: AuthDeps) -> None:
    verify(deps, None, ani=ANI[THOMAS], zip="00000")
    verify(deps, THOMAS, level=1)
    # step-up has its own budget of 2 failures
    outs = [verify(deps, None, ani=ANI[THOMAS], level=2, ssn4="0000")["result"] for _ in range(3)]
    assert outs == ["not_verified", "not_verified", "rate_limited"]


def test_per_customer_limit_across_calls(deps: AuthDeps) -> None:
    for i in range(5):  # 5 failures across 5 calls
        assert verify(deps, None, call_sid=f"CA{i}", ani=ANI[THOMAS], zip="00000")["result"] \
            == "not_verified"
    assert verify(deps, THOMAS, call_sid="CA-new")["result"] == "rate_limited"
    assert verify(deps, MARIA, call_sid="CA-maria")["result"] == "verified"
    deps.clock.t += 3600  # type: ignore[attr-defined]  # next window
    assert verify(deps, THOMAS, call_sid="CA-later")["result"] == "verified"


def test_locked_household_member_does_not_block_the_other(deps: AuthDeps) -> None:
    """Failures on the shared phone count for both; once Sam is locked, Jordan can still verify."""
    store = deps.store
    for n in range(1, 6):
        store.put("calls", {"call_sid": f"ratelimit#customer#{SAM}#w{NOW // 3600}#{n}",
                            "expires_at": NOW + 3600})
    assert verify(deps, SAM)["result"] == "not_verified"  # Sam excluded -> only Jordan left
    assert verify(deps, JORDAN, call_sid="CA2")["result"] == "verified"


def test_policy_limits_are_data(deps: AuthDeps) -> None:
    doc = copy.deepcopy(POLICY_DOC)
    doc["limits"]["verify_attempts_per_call"] = 1
    deps.policy = Policy.from_dict(doc)
    verify(deps, None, ani=ANI[THOMAS], zip="00000")
    assert verify(deps, THOMAS)["result"] == "rate_limited"


# --------------------------------------------------------------------------- audit


def test_audit_log_has_events_but_no_values_or_tokens(deps: AuthDeps,
                                                      caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="owlbank.audit")
    verify(deps, None, ani=ANI[THOMAS], zip="94106")
    ok = verify(deps, THOMAS, level=1)
    up = verify(deps, None, ani=ANI[THOMAS], level=3, ssn4="1111", otp="123456")
    verify(deps, None, ani=ANI[THOMAS], level=9, zip="94105")
    text = text_of(caplog.records)
    events = [ln["audit"] for ln in lines_of(caplog.records, "owlbank.audit")]
    assert events == ["verification_failed", "token_issued", "step_up", "invalid_level"]
    for secret in ("94105", "94106", "1111", "123456", ok["token"], up["token"],
                   tokens.hash_token(ok["token"])):
        assert secret not in text


# --------------------------------------------------------------------------- property


@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(zip_=st.text(alphabet="0123456789", min_size=5, max_size=5),
       ssn4=st.text(alphabet="0123456789", min_size=4, max_size=4))
def test_only_the_real_factors_verify(zip_: str, ssn4: str) -> None:
    """Any ZIP/SSN4 pair verifies Thomas iff it is his; outputs never echo the inputs."""
    deps = make_deps()
    out = service.verify(deps, stir_verstat=TRUSTED, call_sid="CA1", ani=ANI[THOMAS],
                         factors={"zip": zip_, "ssn4": ssn4}, target_level=2)
    real = (zip_, ssn4) == (FACTORS[THOMAS]["zip"], FACTORS[THOMAS]["ssn4"])
    assert (out["result"] == "verified") is real
    if not real:
        assert set(out) == {"result", "level", "reason", "attempts_remaining"}
        assert zip_ not in json.dumps(out) and ssn4 not in json.dumps(out)


# --------------------------------------------------------------------------- event / handler


def test_handle_event_verify_and_unknown_fields(deps: AuthDeps) -> None:
    event = {"op": "verify", "call_sid": "CA1", "ani": ANI[MARIA],
             "factors": {"zip": FACTORS[MARIA]["zip"]}, "target_level": 1,
             "stir_verstat": TRUSTED}
    assert service.handle_event(deps, json.dumps(event))["customer_id"] == MARIA
    assert service.handle_event(deps, {**event, "customer_id": THOMAS})["result"] == "bad_request"
    assert service.handle_event(deps, {"op": "nope"})["result"] == "bad_request"
    assert service.handle_event(deps, "not json")["result"] == "bad_request"
    send = {"op": "send_otp", "call_sid": "CA1", "ani": ANI[MARIA], "language": "en-US",
            "stir_verstat": TRUSTED}
    assert service.handle_event(deps, send) == {"result": "sent"}
    assert service.handle_event(deps, {**send, "customer_id": THOMAS})["result"] == "bad_request"
    # No calling number: no code, and the front door is told to transfer the call.
    assert service.handle_event(deps, {"op": "send_otp", "call_sid": "CA1"}) \
        == {"result": "escalate", "reason": "otp_unavailable"}


def test_store_failure_is_unavailable(deps: AuthDeps) -> None:
    class Broken(MemoryStore):
        def query(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("boom")

    deps.store = Broken()
    assert verify(deps, THOMAS)["result"] == "unavailable"


@pytest.fixture
def dynamo(monkeypatch: pytest.MonkeyPatch) -> Iterator[DynamoStore]:
    for var, val in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing",
                     "AWS_SESSION_TOKEN": "testing", "AWS_DEFAULT_REGION": "us-east-1"}.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    import boto3
    from moto import mock_aws

    cfg = Settings(aws_region="us-east-1")
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        for logical in SCHEMAS:
            client.create_table(**create_table_kwargs(logical, cfg.table(logical)))
        store = DynamoStore.from_settings(cfg)
        seed(store)
        for item in Policy.to_items(POLICY_DOC):
            store.put("policy", item)
        yield store


def test_handler_end_to_end_on_dynamodb(dynamo: DynamoStore, monkeypatch: pytest.MonkeyPatch) -> None:
    deps = auth_handler.build_deps({"AWS_REGION": "us-east-1"})
    assert deps.verify_client is None  # no Verify service configured
    monkeypatch.setattr(auth_handler, "_deps", deps)
    deps.clock = Clock()
    out = auth_handler.handler({"op": "verify", "call_sid": "CA1", "ani": ANI[SAM],
                                "factors": {"zip": FACTORS[SAM]["zip"]}, "target_level": 1,
                                "stir_verstat": TRUSTED})
    assert out["result"] == "verified" and out["customer_id"] == SAM
    up = auth_handler.handler({"op": "verify", "call_sid": "CA1", "ani": ANI[SAM],
                               "factors": {"ssn4": FACTORS[SAM]["ssn4"]}, "target_level": 2,
                               "stir_verstat": TRUSTED})
    assert up["result"] == "verified" and up["step_up"] is True
    assert auth_handler.handler({"op": "verify", "call_sid": "CA1", "ani": ANI[SAM],
                                 "factors": {"otp": "123456"}, "target_level": 3,
                                 "stir_verstat": TRUSTED})["result"] \
        == "unavailable"


def test_build_deps_with_verify_configured(dynamo: DynamoStore) -> None:
    deps = auth_handler.build_deps({"AWS_REGION": "us-east-1",
                                    "OWLBANK_SECRET_TWILIO_API_KEY": "arn:aws:secretsmanager:x",
                                    "OWLBANK_VERIFY_SERVICE_SID": "VA123"})
    assert isinstance(deps.verify_client, TwilioVerifyClient)  # built lazily: no Twilio call


def test_twilio_verify_client_wraps_sdk() -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    class Svc:
        verifications = SimpleNamespace(create=lambda **k: calls.append(("send", k)))
        verification_checks = SimpleNamespace(
            create=lambda **k: SimpleNamespace(status="approved" if k["code"] == "1" else "pending"))

    sdk = SimpleNamespace(verify=SimpleNamespace(v2=SimpleNamespace(services=lambda sid: Svc)))
    client = TwilioVerifyClient("VA1", lambda: sdk)
    client.send("+12025550101", "sms")
    assert calls == [("send", {"to": "+12025550101", "channel": "sms"})]
    assert client.check("+12025550101", "1") is True
    assert client.check("+12025550101", "2") is False

    def boom(**k: Any) -> Any:
        raise RuntimeError("+12025550101 bad")

    Svc.verifications = SimpleNamespace(create=boom)
    with pytest.raises(VerifyUnavailable) as exc:
        client.send("+12025550101", "sms")
    assert "+1202" not in str(exc.value)


def test_alias_ani_verifies_the_fixture_customer(deps: AuthDeps) -> None:
    """A real test number seeded as an alias record (scripts/seed.py) resolves to its fixture."""
    for label, fid in (("A", THOMAS), ("B", SAM), ("B", JORDAN)):
        deps.store.put("customers", {"customer_id": f"ani#{label}#{fid}",
                                     "phone": "+15005550006" if label == "A" else "+15005550007",
                                     "alias_of": fid, "kind": "ani_alias", "label": label})
    out = verify(deps, None, ani="+15005550006", zip=FACTORS[THOMAS]["zip"])
    assert out["result"] == "verified" and out["customer_id"] == THOMAS
    lee = verify(deps, None, call_sid="CA2", ani="+15005550007", zip=FACTORS[JORDAN]["zip"])
    assert lee["customer_id"] == JORDAN
    claims = check(deps, lee["token"], call_sid="CA2")
    assert isinstance(claims, tokens.TokenClaims) and claims.customer_id == JORDAN


# --------------------------------------------------------------------------- OTP destination

ALIAS_NUMBER = "+15005550006"


def _add_alias(deps: AuthDeps, fid: str = THOMAS, phone: str = ALIAS_NUMBER) -> None:
    deps.store.put("customers", {"customer_id": f"ani#A#{fid}", "phone": phone,
                                 "alias_of": fid, "kind": "ani_alias", "label": "A"})


def _otp_round_trip(deps: AuthDeps, ani: str, caplog: pytest.LogCaptureFixture) -> str:
    caplog.set_level(logging.DEBUG)
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    first = verify(deps, None, ani=ani, level=2, zip=FACTORS[THOMAS]["zip"],
                   ssn4=FACTORS[THOMAS]["ssn4"]) if ani != "+12025550199" else verify(
        deps, THOMAS, level=2)
    assert first["result"] == "verified"
    assert service.handle_event(deps, {"op": "send_otp", "call_sid": "CA1", "ani": ani}) \
        == {"result": "sent"}
    up = verify(deps, None, ani=ani, level=3, otp="123456")
    assert up["result"] == "verified"
    (to, channel), = fake.sent
    assert fake.checked == [to] and channel == "sms"  # Verify checks the number it sent to
    text = text_of(caplog.records)
    for value in (to, ani, "123456", ALIAS_NUMBER, ANI[THOMAS]):
        assert value not in text
    return to


def test_otp_goes_to_alias_ani(deps: AuthDeps, caplog: pytest.LogCaptureFixture) -> None:
    _add_alias(deps)
    assert _otp_round_trip(deps, ALIAS_NUMBER, caplog) == ALIAS_NUMBER


def test_otp_goes_to_direct_fixture_ani(deps: AuthDeps, caplog: pytest.LogCaptureFixture) -> None:
    assert _otp_round_trip(deps, ANI[THOMAS], caplog) == ANI[THOMAS]


def _escalates(deps: AuthDeps, ani: Any, caplog: pytest.LogCaptureFixture | None = None) -> None:
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    out = service.send_otp(deps, call_sid="CA1", ani=ani)
    assert out == {"result": "escalate", "reason": "otp_unavailable"}
    assert fake.sent == [] and fake.checked == []
    if caplog is not None:
        text = text_of(caplog.records)
        assert "otp_unavailable" in text and "no_destination" in text
        for value in (ANI[THOMAS], ANI[MARIA], ALIAS_NUMBER, "123456"):
            assert value not in text


def test_otp_unmatched_ani_does_not_fall_back_to_phone_on_file(
        deps: AuthDeps, caplog: pytest.LogCaptureFixture) -> None:
    """Deliberate: a number with no record is not texted, even though the
    customer's own phone is on file."""
    caplog.set_level(logging.DEBUG)
    verify(deps, THOMAS, level=2)
    _escalates(deps, "+12025550199", caplog)


@pytest.mark.parametrize("ani", [None, "", "anonymous", "restricted", "not-a-number",
                                 "12025550101", "+1202555010", "+120255501011234567",
                                 "+1202555010x", " +12025550101", "+12025550101\n"])
def test_otp_withheld_or_malformed_ani_escalates(deps: AuthDeps, ani: Any) -> None:
    verify(deps, THOMAS, level=2)
    _escalates(deps, ani)


def test_otp_never_goes_to_another_customers_number(deps: AuthDeps) -> None:
    """An ANI (or alias) belonging to someone else is not used, and neither is the verified
    customer's own phone on file."""
    _add_alias(deps, fid=MARIA)
    verify(deps, THOMAS, level=2)
    for ani in (ANI[MARIA], ALIAS_NUMBER):
        _escalates(deps, ani)


def test_otp_alias_of_the_customer_is_accepted_and_others_are_not(deps: AuthDeps) -> None:
    _add_alias(deps, fid=THOMAS)
    thomas = deps.store.get("customers", {"customer_id": THOMAS})
    maria = deps.store.get("customers", {"customer_id": MARIA})
    assert service.otp_destination(deps, thomas, ALIAS_NUMBER) == ALIAS_NUMBER
    assert service.otp_destination(deps, maria, ALIAS_NUMBER) is None
    assert service.otp_destination(deps, thomas, ANI[THOMAS]) == ANI[THOMAS]
    assert service.otp_destination(deps, thomas, ANI[MARIA]) is None
    assert service.otp_destination(deps, {}, ANI[THOMAS]) is None


def test_otp_shared_number_goes_to_the_verified_customer_only(deps: AuthDeps) -> None:
    """Sam and Jordan share a phone: either, once verified, may be texted at it."""
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    verify(deps, SAM, level=2)
    assert service.send_otp(deps, call_sid="CA1", ani=ANI[SAM])["result"] == "sent"
    assert fake.sent == [(ANI[SAM], "sms")]


def test_send_failure_audits_the_reason_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    deps = make_deps(verify=FakeVerify(fail=True))
    verify(deps, THOMAS, level=2)
    assert service.send_otp(deps, call_sid="CA1", ani=ANI[THOMAS]) \
        == {"result": "escalate", "reason": "otp_unavailable"}
    text = text_of(caplog.records)
    assert "'audit': 'otp_unavailable'" in text and "verify_send_failed" in text
    assert ANI[THOMAS] not in text


def test_verify_without_a_destination_checks_no_code(deps: AuthDeps) -> None:
    """``verify`` applies the same rule: another number is not a destination for the check."""
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    verify(deps, THOMAS, level=2)
    out = verify(deps, None, ani="+12025550199", level=3, otp="123456")
    assert out["result"] in ("not_verified", "unavailable") and fake.checked == []


# --------------------------------------------------------------------------- locale and untrusted ANI


@pytest.mark.parametrize("language, locale", [("en-US", "en"), ("es-US", "es"), ("fr-FR", None),
                                              (None, None), ("", None), (5, None)])
def test_twilio_client_maps_the_call_language_to_a_verify_locale(language: Any,
                                                                  locale: str | None) -> None:
    calls: list[dict[str, str]] = []

    class Svc:
        verifications = SimpleNamespace(create=lambda **k: calls.append(k))

    sdk = SimpleNamespace(verify=SimpleNamespace(v2=SimpleNamespace(services=lambda sid: Svc)))
    TwilioVerifyClient("VA1", lambda: sdk).send("+12025550101", "sms", language)
    expected = {"to": "+12025550101", "channel": "sms", **({"locale": locale} if locale else {})}
    assert calls == [expected]


def test_send_otp_passes_the_language_through_and_drops_junk(deps: AuthDeps) -> None:
    fake: FakeVerify = deps.verify_client  # type: ignore[assignment]
    verify(deps, THOMAS, level=2)
    for language in ("es-US", "en-US", "bad lang!"):
        service.send_otp(deps, call_sid="CA1", ani=ANI[THOMAS], language=language)  # type: ignore[arg-type]
    assert fake.locales == ["es-US", "en-US", None]  # the 3 sends allowed per call; junk is dropped


UNTRUSTED_EVENT = "otp_sent_untrusted_ani"


@pytest.mark.parametrize("verstat, written", [
    (TRUSTED, False), ("TN-Validation-Failed-C", True), (None, True), ("junk value!", True)])
def test_untrusted_ani_code_send_is_flagged_only_for_untrusted(
        deps: AuthDeps, caplog: pytest.LogCaptureFixture, verstat: str | None, written: bool) -> None:
    caplog.set_level(logging.DEBUG)
    verify(deps, THOMAS, level=2)
    out = service.send_otp(deps, call_sid="CA1", ani=ANI[THOMAS], stir_verstat=verstat)
    assert out == {"result": "sent"}  # a code is sent either way: a warning, not a block
    lines = [r for r in caplog.records if r.name == "owlbank.security"]
    assert [r.getMessage() == UNTRUSTED_EVENT for r in lines] == ([True] if written else [])
    if written:
        assert lines[0].levelno == logging.WARNING
        assert lines_of(lines, "owlbank.security") == [{"event": "otp_sent_untrusted_ani",
                                                        "component": "auth_service"}]
    text = text_of(lines)
    assert ANI[THOMAS] not in text and "123456" not in text and "CA1" not in text


def test_untrusted_flag_is_not_written_when_no_code_is_sent(
        deps: AuthDeps, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    verify(deps, THOMAS, level=2)
    service.send_otp(deps, call_sid="CA1", ani="anonymous", stir_verstat=None)
    assert not [r for r in caplog.records if r.name == "owlbank.security"]


def test_build_deps_reads_verify_sid_from_settings(dynamo: DynamoStore) -> None:
    deps = auth_handler.build_deps({"AWS_REGION": "us-east-1",
                                    "OWLBANK_SECRET_TWILIO_API_KEY": "arn:aws:secretsmanager:x"})
    assert deps.verify_client is None
