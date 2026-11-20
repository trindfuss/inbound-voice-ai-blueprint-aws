"""AgentCore hosting: properties of the relay parameter read (``owlbank.relay_param``).

* Fargate parity: for any relay host the Bridge stack can write (a DNS host name), the front door's
  relay URL is exactly ``wss://<host>/ws``, the URL the Compute template used to carry, so every
  ``<Connect>`` URL (``StaticRelayUrl``, nonce included) is byte for byte what it was.
* agentcore: any Runtime ARN round-trips unchanged into ``OWLBANK_RUNTIME_ARN``.
* Fail closed: any value that is not of the hosting's form raises :class:`RelayParameterError`
  (reason ``malformed``) and the value is in no log line and not in the error.
* Exactly one ``GetParameter`` when ``OWLBANK_RELAY_PARAMETER`` is set, none otherwise; the rest of
  the environment is untouched.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor.app import FrontDoorConfig
from lambdas.frontdoor.relay import StaticRelayUrl
from owlbank import relay_param as rp
from owlbank import ws_nonce

from tests.logutil import text_of

NAME = "/owlbank/dev/bridge/relay"
SETTINGS = settings(max_examples=200, deadline=None,
                    suppress_health_check=[HealthCheck.function_scoped_fixture])


class FakeSsm:
    def __init__(self, value: str) -> None:
        self.value, self.calls = value, 0

    def get_parameter(self, **kw: Any) -> dict[str, Any]:
        self.calls += 1
        return {"Parameter": {"Name": kw["Name"], "Value": self.value}}


_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
labels = st.builds(lambda first, mid, last: first + mid + last if mid or last else first,
                   st.sampled_from(_ALNUM), st.text(_ALNUM + "-", max_size=20), st.sampled_from(["", *_ALNUM]))
labels = labels.filter(lambda s: not s.endswith("-") and len(s) <= 63)
hosts = st.lists(labels, min_size=2, max_size=5).map(".".join).filter(lambda h: len(h) <= 253)
runtime_arns = st.builds(
    lambda part, region, acct, rid: f"arn:{part}:bedrock-agentcore:{region}:{acct}:runtime/{rid}",
    st.sampled_from(["aws", "aws-us-gov", "aws-cn"]), st.sampled_from(["us-east-1", "us-west-2", "eu-west-1"]),
    st.text("0123456789", min_size=12, max_size=12), st.text(_ALNUM + "_-", min_size=1, max_size=60))
other_env = st.dictionaries(st.sampled_from(["OWLBANK_LOG_LEVEL", "AWS_REGION", "OWLBANK_TABLE_CALLS"]),
                            st.text(_ALNUM, min_size=1, max_size=10), max_size=3)


@SETTINGS
@given(host=hosts, extra=other_env, nonce=st.text(_ALNUM + "-_", min_size=1, max_size=32))
def test_fargate_url_is_todays_for_any_host(host: str, extra: dict[str, str], nonce: str) -> None:
    ssm = FakeSsm(host)
    env = rp.resolve_relay_env({**extra, rp.RELAY_PARAMETER_ENV: NAME}, hosting="fargate",
                               ssm_factory=lambda: ssm)
    url = f"wss://{host}/ws"  # what compute_stack wrote into OWLBANK_RELAY_WS_URL before the relay parameter
    assert env == {**extra, rp.RELAY_PARAMETER_ENV: NAME, rp.RELAY_WS_URL_ENV: url}
    assert ssm.calls == 1
    cfg = FrontDoorConfig.from_env(env)
    assert StaticRelayUrl(cfg.relay_ws_url).mint(call_sid="CA" + "0" * 32, nonce=nonce) == \
        ws_nonce.with_nonce(url, nonce)


@SETTINGS
@given(arn=runtime_arns, extra=other_env)
def test_agentcore_arn_round_trips(arn: str, extra: dict[str, str]) -> None:
    ssm = FakeSsm(arn)
    env = rp.resolve_relay_env({**extra, rp.RELAY_PARAMETER_ENV: NAME}, hosting="agentcore",
                               ssm_factory=lambda: ssm)
    assert env == {**extra, rp.RELAY_PARAMETER_ENV: NAME, rp.RUNTIME_ARN_ENV: arn} and ssm.calls == 1


_MARK = "zq9"  # makes a generated bad value distinctive enough to search the logs for


@SETTINGS
@given(hosting=st.sampled_from(["fargate", "agentcore"]),
       value=st.text(min_size=1, max_size=80).map(lambda s: _MARK + s))
def test_any_malformed_value_fails_closed_without_leaking(hosting: str, value: str,
                                                         caplog: pytest.LogCaptureFixture) -> None:
    good = (rp.HOST_RE if hosting == "fargate" else rp.RUNTIME_ARN_RE).fullmatch(value.strip())
    caplog.clear()
    caplog.set_level(logging.INFO)
    try:
        env = rp.resolve_relay_env({rp.RELAY_PARAMETER_ENV: NAME}, hosting=hosting,
                                   ssm_factory=lambda: FakeSsm(value))
    except rp.RelayParameterError as err:
        assert not good
        assert err.reason in ("malformed", "empty")
        assert value.strip() not in text_of(caplog.records) + str(err)
        return
    assert good and rp.TARGET_ENV[hosting] in env


@SETTINGS
@given(hosting=st.sampled_from(["fargate", "agentcore"]), extra=other_env)
def test_no_parameter_variable_means_no_read(hosting: str, extra: dict[str, str]) -> None:
    def never() -> Any:
        raise AssertionError("no SSM read without OWLBANK_RELAY_PARAMETER")

    assert rp.resolve_relay_env(extra, hosting=hosting, ssm_factory=never) == extra
