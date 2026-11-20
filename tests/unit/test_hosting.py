"""``tac_bridge.hosting`` unit tests."""

from __future__ import annotations

import json
import logging

import pytest

from owlbank import secrets as owl_secrets
from tac_bridge.hosting import (
    HEALTHY,
    HEALTHY_BUSY,
    SECRET_API_KEY_ENV,
    SECRET_AUTH_TOKEN_ENV,
    SocketTracker,
    hosting_from_env,
    load_twilio_secrets,
)
from tests.logutil import text_of

API_ARN = "arn:aws:secretsmanager:us-east-1:111111111111:secret:owlbank/dev/twilio-api-key-AbCdEf"
TOKEN_ARN = "arn:aws:secretsmanager:us-east-1:111111111111:secret:owlbank/dev/twilio-auth-token-GhIjKl"
# Fictional values, distinctive enough to search the logs for.
VALUES = {"account_sid": "AC" + "f" * 32, "api_key": "SK" + "e" * 32,
          "api_secret": "S3cr3tApiValueZz" + "z" * 16, "auth_token": "70ce" * 8}
ENV = {SECRET_API_KEY_ENV: API_ARN, SECRET_AUTH_TOKEN_ENV: TOKEN_ARN}


@pytest.fixture(autouse=True)
def _fresh_secret_cache():
    owl_secrets.clear_cache()
    yield
    owl_secrets.clear_cache()


class FakeSM:
    """A Secrets Manager client: ``docs`` maps ARN -> SecretString (or an exception to raise)."""

    def __init__(self, docs: dict[str, object]) -> None:
        self.docs = docs
        self.calls: list[str] = []

    def get_secret_value(self, *, SecretId: str) -> dict[str, object]:  # noqa: N803 - boto3's name
        self.calls.append(SecretId)
        doc = self.docs[SecretId]
        if isinstance(doc, Exception):
            raise doc
        return {"SecretString": doc}


def good_sm() -> FakeSM:
    return FakeSM({API_ARN: json.dumps({k: VALUES[k] for k in ("account_sid", "api_key", "api_secret")}),
                   TOKEN_ARN: json.dumps({"auth_token": VALUES["auth_token"]})})


# --------------------------------------------------------------------------- hosting_from_env


@pytest.mark.parametrize("env", [{}, {"OWLBANK_HOSTING": ""}, {"OWLBANK_HOSTING": "  "},
                                 {"OWLBANK_HOSTING": "fargate"}])
def test_absent_or_fargate_is_fargate(env: dict[str, str]) -> None:
    assert hosting_from_env(env) == "fargate"


def test_agentcore() -> None:
    assert hosting_from_env({"OWLBANK_HOSTING": "agentcore"}) == "agentcore"


@pytest.mark.parametrize("bad", ["AgentCore", "ecs", "lambda", "agentcore2", "fargate,agentcore"])
def test_bad_value_raises_naming_the_variable_not_the_value(bad: str) -> None:
    with pytest.raises(RuntimeError) as ei:
        hosting_from_env({"OWLBANK_HOSTING": bad})
    assert str(ei.value) == "OWLBANK_HOSTING must be one of: fargate, agentcore"


# --------------------------------------------------------------------------- SocketTracker


def test_tracker_busy_while_a_socket_is_open() -> None:
    now = [1000.0]
    t = SocketTracker(clock=lambda: now[0])
    assert t.ping() == {"status": HEALTHY, "time_of_last_update": 1000}
    now[0] = 1010.5
    t.opened()
    assert t.ping() == {"status": HEALTHY_BUSY, "time_of_last_update": 1010}
    now[0] = 1020.0
    t.opened()  # a second socket (re-entry overlap): no status change, no update
    assert t.ping() == {"status": HEALTHY_BUSY, "time_of_last_update": 1010}
    t.closed()
    assert t.ping()["status"] == HEALTHY_BUSY and t.ping()["time_of_last_update"] == 1010
    now[0] = 1030.0
    t.closed()
    assert t.ping() == {"status": HEALTHY, "time_of_last_update": 1030}


def test_tracker_ignores_an_unmatched_close() -> None:
    t = SocketTracker(clock=lambda: 5.0)
    t.closed()
    assert t.open_count == 0 and t.ping() == {"status": HEALTHY, "time_of_last_update": 5}


# --------------------------------------------------------------------------- load_twilio_secrets


def test_secrets_map_to_tac_names_and_each_secret_is_read_once() -> None:
    sm = good_sm()
    out = load_twilio_secrets(ENV, sm)
    assert out == {"TWILIO_ACCOUNT_SID": VALUES["account_sid"], "TWILIO_API_KEY": VALUES["api_key"],
                   "TWILIO_API_SECRET": VALUES["api_secret"], "TWILIO_AUTH_TOKEN": VALUES["auth_token"]}
    assert sorted(sm.calls) == sorted([API_ARN, TOKEN_ARN])


def test_mapping_matches_the_fargate_task_definition() -> None:
    """The same TAC names and JSON fields as ECS injects on Fargate (``bridge_stack.TASK_SECRETS``)."""
    from tac_bridge.hosting import TWILIO_SECRET_FIELDS

    to_secret = {SECRET_API_KEY_ENV: "twilio_api_key", SECRET_AUTH_TOKEN_ENV: "twilio_auth_token"}
    assert {k: (to_secret[v], f) for k, (v, f) in TWILIO_SECRET_FIELDS.items()} == {
        "TWILIO_ACCOUNT_SID": ("twilio_api_key", "account_sid"),
        "TWILIO_API_KEY": ("twilio_api_key", "api_key"),
        "TWILIO_API_SECRET": ("twilio_api_key", "api_secret"),
        "TWILIO_AUTH_TOKEN": ("twilio_auth_token", "auth_token"),
    }


@pytest.mark.parametrize("missing", [SECRET_API_KEY_ENV, SECRET_AUTH_TOKEN_ENV])
def test_missing_arn_variable_names_the_variable(missing: str) -> None:
    env = {k: v for k, v in ENV.items() if k != missing}
    sm = good_sm()
    with pytest.raises(RuntimeError, match=f"^{missing} is required$"):
        load_twilio_secrets(env, sm)
    assert sm.calls == []  # every ARN variable is checked before any read


def _assert_clean(exc: BaseException, caplog: pytest.LogCaptureFixture, *raw: str) -> None:
    """No secret value in the exception (message, chain, context) or in any log record."""
    seen = [str(exc), repr(exc), str(exc.__cause__), str(exc.__context__)]
    assert exc.__cause__ is None and exc.__suppress_context__
    for value in (*VALUES.values(), *raw):
        assert all(value not in s for s in seen)
        assert value not in text_of(caplog.records)


def test_unreadable_secret_names_the_variable_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    class AccessDenied(Exception):
        response = {"Error": {"Code": "AccessDeniedException"}}

    sm = good_sm()
    sm.docs[TOKEN_ARN] = AccessDenied(f"denied; value was {VALUES['auth_token']}")
    with pytest.raises(RuntimeError) as ei:
        load_twilio_secrets(ENV, sm)
    assert str(ei.value).startswith(SECRET_AUTH_TOKEN_ENV + ":")
    _assert_clean(ei.value, caplog)


_TOKEN = VALUES["auth_token"]


@pytest.mark.parametrize("raw", ['{"auth_token": "' + _TOKEN + '"', f'"{_TOKEN}"', _TOKEN, f'["{_TOKEN}"]'])
def test_non_object_secret_never_echoes_its_text(raw: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    sm = good_sm()
    sm.docs[TOKEN_ARN] = raw
    with pytest.raises(RuntimeError) as ei:
        load_twilio_secrets(ENV, sm)
    assert str(ei.value).startswith(SECRET_AUTH_TOKEN_ENV + ":")
    _assert_clean(ei.value, caplog, raw)


@pytest.mark.parametrize("field", ["account_sid", "api_key", "api_secret"])
@pytest.mark.parametrize("bad", [None, "", 42])
def test_missing_or_empty_field_names_variable_and_field(field: str, bad: object,
                                                         caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    doc = {k: VALUES[k] for k in ("account_sid", "api_key", "api_secret")}
    if bad is None:
        del doc[field]
    else:
        doc[field] = bad  # type: ignore[assignment]
    sm = good_sm()
    sm.docs[API_ARN] = json.dumps(doc)
    with pytest.raises(RuntimeError) as ei:
        load_twilio_secrets(ENV, sm)
    want = "has no " + field if bad is None else f"'s {field} is malformed"
    assert str(ei.value) == f"{SECRET_API_KEY_ENV}: the secret{' ' if bad is None else ''}{want}"
    for value in VALUES.values():
        assert value not in str(ei.value) and value not in text_of(caplog.records)


def test_placeholder_secret_names_the_variable(caplog: pytest.LogCaptureFixture) -> None:
    """CDK's placeholder fails start-up, naming the variable, not the value."""
    caplog.set_level(logging.DEBUG)
    sm = good_sm()
    sm.docs[TOKEN_ARN] = json.dumps({"placeholder": True, "unset": "Xq7Rz2Lp9Wk4Mn8T"})
    with pytest.raises(RuntimeError) as ei:
        load_twilio_secrets(ENV, sm)
    assert str(ei.value) == f"{SECRET_AUTH_TOKEN_ENV}: the secret is still the placeholder"
    _assert_clean(ei.value, caplog, "Xq7Rz2Lp9Wk4Mn8T")
