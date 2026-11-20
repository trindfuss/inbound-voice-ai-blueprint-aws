"""The bridge's relay parameter.

The Bridge stack writes one SSM String parameter with a fixed name, ``/owlbank/<stage>/bridge/relay``,
on both hostings, and the Compute stack's Lambdas read it at start-up (INIT) instead of importing a
CloudFormation export, so either hosting deploys in either order and a switch is one deploy:

* ``fargate``: the relay WebSocket **host** (the custom relay host, or the CloudFront distribution's
  domain). The front door builds ``wss://<host>/ws`` from it, exactly the URL the template used to
  carry in ``OWLBANK_RELAY_WS_URL``.
* ``agentcore``: the Runtime **ARN** (``OWLBANK_RUNTIME_ARN`` for the URL signer, the session stopper
  and the warmer).

:func:`resolve_relay_env` returns the environment with the hosting's variable filled from the
parameter named by ``OWLBANK_RELAY_PARAMETER``. Without that variable the environment is returned
as it is (the tests set the variable directly). It fails closed: a parameter that
can't be read, is empty or malformed, or a variable set both ways raises :class:`RelayParameterError`
after one ``relay_parameter_unavailable`` log line (parameter name, reason, error class and code; never
the value). The caller then has no app: every request fails until the parameter is right.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from typing import Any

from owlbank import obs

log = logging.getLogger(__name__)

#: Names the SSM parameter (set by the Compute stack).
RELAY_PARAMETER_ENV = "OWLBANK_RELAY_PARAMETER"
#: What the parameter fills, per hosting.
RELAY_WS_URL_ENV = "OWLBANK_RELAY_WS_URL"
RUNTIME_ARN_ENV = "OWLBANK_RUNTIME_ARN"
TARGET_ENV = {"fargate": RELAY_WS_URL_ENV, "agentcore": RUNTIME_ARN_ENV}
#: The path the bridge serves ConversationRelay on (``TWILIO_VOICE_WEBSOCKET_PATH``).
WS_PATH = "/ws"

#: A DNS host name: dot-separated labels of letters, digits and inner hyphens, at least two labels.
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
HOST_RE = re.compile(rf"^(?=.{{1,253}}\Z){_LABEL}(?:\.{_LABEL})+\Z")
#: An AgentCore Runtime ARN (``…:runtime/<RuntimeId>``); the front door's signer checks it too.
RUNTIME_ARN_RE = re.compile(
    r"^arn:aws[a-z-]*:bedrock-agentcore:(?P<region>[a-z0-9-]+):\d{12}:runtime/[A-Za-z0-9_-]+\Z")


class RelayParameterError(RuntimeError):
    """The relay parameter is unusable. The message names the reason, never the value."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"relay parameter unusable: {reason}")
        self.reason = reason


def relay_ws_url(host: str) -> str:
    """``wss://<host>/ws`` for a valid host name; :class:`ValueError` otherwise."""
    if not isinstance(host, str) or not HOST_RE.fullmatch(host):
        raise ValueError("the relay host must be a DNS host name")
    return f"wss://{host}{WS_PATH}"


def runtime_arn(value: str) -> str:
    """``value`` if it is a Runtime ARN; :class:`ValueError` otherwise."""
    if not isinstance(value, str) or not RUNTIME_ARN_RE.fullmatch(value):
        raise ValueError("the relay parameter must be an AgentCore Runtime ARN")
    return value


def _fail(name: str, reason: str, exc: BaseException | None = None) -> RelayParameterError:
    obs.emit(log, logging.ERROR, "relay_parameter_unavailable", parameter=name, reason=reason,
             error_class=type(exc).__name__ if exc is not None else None,
             code=obs.error_code_of(exc) if exc is not None else None, decision="fail_closed")
    return RelayParameterError(reason)


def read_relay_parameter(ssm: Any, name: str) -> str:
    """The parameter's value (``ssm:GetParameter``, no decryption: a String parameter)."""
    try:
        response = ssm.get_parameter(Name=name)
        value = response["Parameter"]["Value"]
    except Exception as exc:  # noqa: BLE001 - logged and re-raised as RelayParameterError
        raise _fail(name, "read_failed", exc) from None
    if not isinstance(value, str) or not value.strip():
        raise _fail(name, "empty")
    return value.strip()


def resolve_relay_env(env: Mapping[str, str], *, hosting: str,
                      ssm_factory: Callable[[], Any]) -> dict[str, str]:
    """``env`` with the hosting's relay variable filled from the parameter (module docs)."""
    if hosting not in TARGET_ENV:
        raise ValueError(f"hosting must be one of {tuple(TARGET_ENV)}")
    out = dict(env)
    name = (env.get(RELAY_PARAMETER_ENV) or "").strip()
    if not name:
        return out
    target = TARGET_ENV[hosting]
    if env.get(target):
        raise _fail(name, "set_twice")  # one source only: no doubt about which one counts
    value = read_relay_parameter(ssm_factory(), name)
    try:
        out[target] = relay_ws_url(value) if hosting == "fargate" else runtime_arn(value)
    except ValueError as exc:
        raise _fail(name, "malformed", exc) from None
    return out
