"""scripts/_common.py Secrets Manager helpers.

``secret_arn`` resolves an ARN without AWS; ``read_secret`` reads through an injected client (a stub
here) and validates with ``owlbank.secret_schema``. Errors name the secret, the ARN, the field and
the IAM action, never a value.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from owlbank.secret_schema import SCHEMAS
from scripts import _common
from scripts._common import (
    GET_SECRET_ACTION,
    SECRET_OUTPUTS,
    ScriptError,
    add_secret_args,
    parse_secret_overrides,
    read_secret,
    secret_arn,
)

ARN = "arn:aws:secretsmanager:us-east-1:111122223333:secret:owlbank/twilio/api-key-AbC"
API_SECRET = "Zq" * 16
GOOD = {
    "twilio_api_key": {"account_sid": "AC" + "0a" * 16, "api_key": "SK" + "1b" * 16, "api_secret": API_SECRET},
    "twilio_auth_token": {"auth_token": "c3" * 16},
    "basic_auth": {"user": "owlbank", "password": "d4" * 24},
}


class ClientError(Exception):
    """Shaped like botocore's: a ``response`` with ``Error.Code``; the message may carry anything."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.response = {"Error": {"Code": code, "Message": message}}


class StubSM:
    """``get_secret_value`` returning a fixed ``SecretString`` or raising; records each call."""

    def __init__(self, value: Any = None, *, raw: str | None = None, error: Exception | None = None,
                 response: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raw = raw if raw is not None else (json.dumps(value) if value is not None else None)
        self._error = error
        self._response = response

    def get_secret_value(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self._error is not None:
            raise self._error
        if self._response is not None:
            return self._response
        return {"ARN": kw["SecretId"], "SecretString": self._raw}


# --------------------------------------------------------------------------- registry and args


def test_secret_outputs_cover_exactly_the_schema_secrets():
    assert set(SECRET_OUTPUTS) == set(SCHEMAS)


def test_secret_outputs_lead_with_the_name_data_stack_emits():
    """``data_stack.py`` emits ``Secret`` + the logical name in CamelCase (``CfnOutput`` id)."""
    for logical, candidates in SECRET_OUTPUTS.items():
        assert candidates[0] == "Secret" + "".join(p.capitalize() for p in logical.split("_"))


def test_populate_secrets_uses_the_moved_map():
    from scripts import populate_secrets as ps

    assert {s.name: s.output_keys for s in ps.SPECS} == dict(SECRET_OUTPUTS)


def test_add_secret_args_parses_outputs_secrets_and_region():
    p = argparse.ArgumentParser()
    add_secret_args(p)
    args = p.parse_args(["--cdk-outputs", "o.json", "--secret", f"basic_auth={ARN}",
                         "--secret", f"twilio_api_key={ARN}", "--region", "eu-west-1"])
    assert args.cdk_outputs == "o.json"
    assert args.region == "eu-west-1"
    assert parse_secret_overrides(args.secret) == {"basic_auth": ARN, "twilio_api_key": ARN}
    empty = p.parse_args([])
    assert (empty.cdk_outputs, empty.secret, empty.region) == (None, None, None)
    assert parse_secret_overrides(empty.secret) == {}


def test_parse_secret_overrides_refuses_unknown_names():
    with pytest.raises(ScriptError, match="unknown secret name.*'basic'"):
        parse_secret_overrides([f"basic={ARN}"])


# --------------------------------------------------------------------------- secret_arn


def test_explicit_arn_wins_over_outputs():
    outputs = {"SecretBasicAuth": "from-outputs"}
    assert secret_arn("basic_auth", outputs, {"basic_auth": ARN}) == ARN


def test_arn_from_cdk_outputs_with_hash_suffix_and_legacy_names():
    assert secret_arn("basic_auth", {"SecretBasicAuthAB12CD": ARN}, {}) == ARN
    assert secret_arn("twilio_auth_token", {"TwilioAuthTokenSecretArn": ARN}, {}) == ARN


def test_missing_arn_names_the_secret_and_the_outputs_tried():
    with pytest.raises(ScriptError) as ei:
        secret_arn("twilio_auth_token", {"SomethingElse": ARN}, {"basic_auth": ARN})
    msg = str(ei.value)
    assert "twilio_auth_token" in msg
    for candidate in SECRET_OUTPUTS["twilio_auth_token"]:
        assert candidate in msg
    assert "--secret twilio_auth_token=ARN" in msg


def test_unknown_logical_name_is_refused():
    with pytest.raises(ScriptError, match="unknown secret"):
        secret_arn("nope", {}, {})


# --------------------------------------------------------------------------- read_secret


@pytest.mark.parametrize("logical", sorted(GOOD))
def test_reads_and_validates_each_secret(logical):
    sm = StubSM(GOOD[logical])
    assert read_secret(logical, ARN, sm) == GOOD[logical]
    assert sm.calls == [{"SecretId": ARN}]


def test_extra_string_fields_pass_through_non_strings_dropped():
    value = {**GOOD["basic_auth"], "note": "x", "n": 3}
    assert read_secret("basic_auth", ARN, StubSM(value)) == {**GOOD["basic_auth"], "note": "x"}


@pytest.mark.parametrize("code", ["AccessDeniedException", "AccessDenied"])
def test_access_denied_names_the_secret_and_the_action(code):
    err = ClientError(code, f"User is not authorized; value was {API_SECRET}")
    with pytest.raises(ScriptError) as ei:
        read_secret("twilio_api_key", ARN, StubSM(error=err))
    msg = str(ei.value)
    assert "twilio_api_key" in msg and ARN in msg and GET_SECRET_ACTION in msg
    assert API_SECRET not in msg
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


def test_other_client_errors_name_the_code_only():
    err = ClientError("ResourceNotFoundException", f"secret {API_SECRET} gone")
    with pytest.raises(ScriptError, match="get_secret_value failed: ResourceNotFoundException") as ei:
        read_secret("twilio_api_key", ARN, StubSM(error=err))
    assert API_SECRET not in str(ei.value)


def test_a_non_aws_exception_is_named_by_class():
    with pytest.raises(ScriptError, match="get_secret_value failed: RuntimeError") as ei:
        read_secret("twilio_api_key", ARN, StubSM(error=RuntimeError(API_SECRET)))
    assert API_SECRET not in str(ei.value)


def test_placeholder_is_named_with_the_hint():
    sm = StubSM({"placeholder": True, "unset": "Qx7Lm2Np9Rt4Vw8Yz1"})
    with pytest.raises(ScriptError) as ei:
        read_secret("twilio_auth_token", ARN, sm)
    msg = str(ei.value)
    assert "placeholder" in msg and "twilio_auth_token" in msg
    assert "Qx7Lm2Np9Rt4Vw8Yz1" not in msg


def test_a_malformed_value_names_the_field_not_the_value():
    bad = {**GOOD["twilio_api_key"], "api_secret": "short-" + "Pp" * 8}
    with pytest.raises(ScriptError) as ei:
        read_secret("twilio_api_key", ARN, StubSM(bad))
    msg = str(ei.value)
    assert "field api_secret: format" in msg
    assert "short-" not in msg and "PpPp" not in msg


def test_missing_fields_are_each_named():
    with pytest.raises(ScriptError) as ei:
        read_secret("basic_auth", ARN, StubSM({"user": "owlbank"}))
    assert "field password: missing" in str(ei.value)


@pytest.mark.parametrize("raw", ["not json {", '"just-a-string"', "[1, 2]"])
def test_a_non_object_or_non_json_value_is_refused_without_echo(raw):
    with pytest.raises(ScriptError) as ei:
        read_secret("basic_auth", ARN, StubSM(raw=raw))
    assert raw not in str(ei.value)


@pytest.mark.parametrize("response", [{"SecretBinary": b"x"}, {"SecretString": None}, "weird"])
def test_no_secret_string_is_refused(response):
    with pytest.raises(ScriptError, match="no SecretString"):
        read_secret("basic_auth", ARN, StubSM(response=response))


def test_unknown_logical_is_refused_before_any_call():
    sm = StubSM(GOOD["basic_auth"])
    with pytest.raises(ScriptError, match="unknown secret"):
        read_secret("nope", ARN, sm)
    assert sm.calls == []


def test_a_successful_read_logs_the_name_only(caplog):
    with caplog.at_level("DEBUG", logger=_common._common_log.name):
        read_secret("twilio_api_key", ARN, StubSM(GOOD["twilio_api_key"]))
    text = caplog.text + " ".join(str(r.__dict__) for r in caplog.records)
    assert "twilio_api_key" in text
    assert API_SECRET not in text


_BAD_PAIRS = [
    {"previous_password": "0f" * 16},                                  # lone
    {"previous_password": "bad", "previous_until": 1_900_000_000},     # bad format
    {"previous_password": "0f" * 16, "previous_until": 1_900_000_000_000},  # ms epoch
]


@pytest.mark.parametrize("pair", _BAD_PAIRS)
def test_a_malformed_optional_pair_does_not_block_a_read(pair):
    value = {**GOOD["basic_auth"], **pair}
    assert read_secret("basic_auth", ARN, StubSM(value))["user"] == GOOD["basic_auth"]["user"]


def test_a_malformed_required_field_still_blocks_with_a_bad_pair():
    value = {**GOOD["basic_auth"], "password": "x", **_BAD_PAIRS[0]}
    with pytest.raises(ScriptError, match="field password: format") as ei:
        read_secret("basic_auth", ARN, StubSM(value))
    assert "previous" not in str(ei.value)
