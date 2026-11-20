"""AgentCore hosting: the relay parameter read at start-up by the front door and the warmer.

Stub SSM clients only: no network, no AWS account. Fail-closed cases, the one log line (never the
value), and that the front door's Fargate URL is the one the template used to carry.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from botocore.exceptions import ClientError

from lambdas.frontdoor import handler as fd_handler
from lambdas.frontdoor.app import FrontDoorConfig
from lambdas.runtime_warmer import handler as warmer
from owlbank import relay_param as rp

from tests.logutil import lines, text_of

NAME = "/owlbank/dev/bridge/relay"
HOST = "d111111abcdef8.cloudfront.net"
ARN = "arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/owlbank_dev_bridge-AbCdEf1234"
LOGGER = "owlbank.relay_param"


class FakeSsm:
    def __init__(self, value: Any = None, *, fail: BaseException | None = None) -> None:
        self.value, self.fail, self.calls = value, fail, []

    def get_parameter(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        if self.fail is not None:
            raise self.fail
        return {"Parameter": {"Name": kw["Name"], "Type": "String", "Value": self.value}}


def _no_ssm() -> Any:
    raise AssertionError("SSM must not be read")


def _unavailable(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [f for f in lines(caplog.records, LOGGER) if f["event"] == "relay_parameter_unavailable"]


# --------------------------------------------------------------------------- resolve_relay_env

def test_fargate_fills_the_relay_url_from_the_host() -> None:
    ssm = FakeSsm(HOST)
    env = rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: NAME, "X": "1"}, hosting="fargate",
                               ssm_factory=lambda: ssm)
    assert env == {rp.RELAY_PARAMETER_ENV: NAME, "X": "1", rp.RELAY_WS_URL_ENV: f"wss://{HOST}/ws"}
    assert ssm.calls == [{"Name": NAME}]  # one read, no WithDecryption (a String parameter)
    assert FrontDoorConfig.from_env(env).relay_ws_url == f"wss://{HOST}/ws"


def test_custom_domain_host_gives_todays_url() -> None:
    """The custom-domain mode's host is ``relay.<domain>``: the URL the template used to carry."""
    env = rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: NAME}, hosting="fargate",
                               ssm_factory=lambda: FakeSsm("relay.owlbank.example.com"))
    assert env[rp.RELAY_WS_URL_ENV] == "wss://relay.owlbank.example.com/ws"


def test_agentcore_fills_the_runtime_arn() -> None:
    env = rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: NAME}, hosting="agentcore",
                               ssm_factory=lambda: FakeSsm(f"  {ARN}\n"))
    assert env[rp.RUNTIME_ARN_ENV] == ARN and rp.RELAY_WS_URL_ENV not in env


def test_without_the_parameter_variable_nothing_is_read() -> None:
    env = {rp.RELAY_WS_URL_ENV: "wss://relay.example/ws"}
    assert rp.resolve_relay_env(env, hosting="fargate", ssm_factory=_no_ssm) == env
    assert rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: "  "}, hosting="agentcore",
                                ssm_factory=_no_ssm) == {rp.RELAY_PARAMETER_ENV: "  "}


def test_unknown_hosting_is_refused() -> None:
    with pytest.raises(ValueError):
        rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: NAME}, hosting="ecs", ssm_factory=_no_ssm)


@pytest.mark.parametrize(("hosting", "ssm", "reason"), [
    ("fargate", FakeSsm(fail=ClientError({"Error": {"Code": "ParameterNotFound", "Message": "m"}},
                                         "GetParameter")), "read_failed"),
    ("agentcore", FakeSsm(fail=ClientError({"Error": {"Code": "AccessDeniedException", "Message": "m"}},
                                           "GetParameter")), "read_failed"),
    ("fargate", FakeSsm(""), "empty"),
    ("agentcore", FakeSsm(None), "empty"),
    ("fargate", FakeSsm(f"wss://{HOST}/ws"), "malformed"),
    ("fargate", FakeSsm(ARN), "malformed"),
    ("agentcore", FakeSsm(HOST), "malformed"),
])
def test_unusable_parameter_fails_closed_with_one_line(hosting: str, ssm: FakeSsm, reason: str,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with pytest.raises(rp.RelayParameterError) as err:
        rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: NAME}, hosting=hosting, ssm_factory=lambda: ssm)
    assert err.value.reason == reason
    [line] = _unavailable(caplog)
    assert line["parameter"] == NAME and line["reason"] == reason and line["decision"] == "fail_closed"
    if reason == "read_failed":
        assert line["error_class"] == "ClientError" and line["code"] in ("ParameterNotFound",
                                                                         "AccessDeniedException")
    text = text_of(caplog.records) + str(err.value)
    if ssm.value:
        assert str(ssm.value) not in text  # never the value


def test_a_variable_set_both_ways_fails_closed(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with pytest.raises(rp.RelayParameterError) as err:
        rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: NAME, rp.RUNTIME_ARN_ENV: ARN}, hosting="agentcore",
                             ssm_factory=_no_ssm)
    assert err.value.reason == "set_twice" and _unavailable(caplog)[0]["reason"] == "set_twice"


# --------------------------------------------------------------------------- front door and warmer wiring

def test_front_door_relay_env_follows_the_hosting_variable() -> None:
    env = fd_handler.relay_env({rp.RELAY_PARAMETER_ENV: NAME}, lambda: FakeSsm(HOST))
    assert env[rp.RELAY_WS_URL_ENV] == f"wss://{HOST}/ws"
    env = fd_handler.relay_env({rp.RELAY_PARAMETER_ENV: NAME, "OWLBANK_HOSTING": "agentcore"},
                               lambda: FakeSsm(ARN))
    assert env[rp.RUNTIME_ARN_ENV] == ARN and rp.RELAY_WS_URL_ENV not in env
    # An unknown hosting is FrontDoorConfig.from_env's to refuse: no read here.
    bad = {rp.RELAY_PARAMETER_ENV: NAME, "OWLBANK_HOSTING": "nope"}
    assert fd_handler.relay_env(bad, _no_ssm) == bad


def test_front_door_ssm_client_settings() -> None:
    assert (fd_handler.SSM_CONNECT_TIMEOUT_S, fd_handler.SSM_READ_TIMEOUT_S) == (2, 3)


def test_warmer_reads_the_runtime_arn_from_the_parameter() -> None:
    ssm = FakeSsm(ARN)
    deps = warmer.build_deps({rp.RELAY_PARAMETER_ENV: NAME}, ssm_factory=lambda: ssm)
    assert deps.runtime_arn == ARN and ssm.calls == [{"Name": NAME}]
    assert deps.client._client is None  # the Runtime client is still built on first use


def test_warmer_fails_closed_and_retries_on_the_next_invoke(monkeypatch: pytest.MonkeyPatch,
                                                           caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with pytest.raises(rp.RelayParameterError):
        warmer.build_deps({rp.RELAY_PARAMETER_ENV: NAME}, ssm_factory=lambda: FakeSsm(""))
    real = warmer.build_deps
    monkeypatch.setattr(warmer, "_deps", None)
    monkeypatch.setattr(warmer, "build_deps", lambda: real({rp.RELAY_PARAMETER_ENV: NAME},
                                                           ssm_factory=lambda: FakeSsm("")))
    assert warmer.handler({"call_sid": "CA" + "0" * 32}) == {"result": "failed"}
    assert warmer._deps is None  # noqa: SLF001 - not cached: the next invoke reads again
