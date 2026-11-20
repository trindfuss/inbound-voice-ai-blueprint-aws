"""Keep global logging state (root level, handlers, pinned library levels) from leaking across tests.

``owlbank.obs.configure_logging`` changes process-wide logging on purpose (production calls it once at
start-up); a test that calls it, directly or through ``build_app_from_env``, must not change the
levels later tests see.
"""

from __future__ import annotations

import logging
import secrets

import pytest

from owlbank import obs
from tests import leakscan


@pytest.fixture(autouse=True)
def _restore_logging():
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    pinned = {n: logging.getLogger(n).level for n in obs.PINNED_LOGGERS}
    yield
    obs.reset_logging()
    root.setLevel(level)
    for h in list(root.handlers):
        if h not in handlers:
            root.removeHandler(h)
    for n, lv in pinned.items():
        logging.getLogger(n).setLevel(lv)
    obs.unbind_all()


@pytest.fixture(autouse=True, scope="session")
def _record_minted_tokens():
    """Tell the factor leak scan which random tokens the server minted (``tests.leakscan``): a
    nonce or journal reference can hold a short fixture value by chance and is not a leak."""
    originals = {name: getattr(secrets, name) for name in ("token_urlsafe", "token_hex")}

    def recording(fn):
        def mint(*args, **kwargs):
            token = fn(*args, **kwargs)
            leakscan.record_minted(token)
            return token
        return mint

    for name, fn in originals.items():
        setattr(secrets, name, recording(fn))
    yield
    for name, fn in originals.items():
        setattr(secrets, name, fn)


@pytest.fixture(autouse=True)
def _fresh_sensitive_state():
    """A fresh ``owlbank.sensitive`` process state per test: tests reuse
    CallSids (``sid(1)``), and a blob seen, an entry remembered or ``unverifiable`` set by an
    earlier test must not change what a later one sees."""
    from owlbank import sensitive

    sensitive._state = sensitive._ProcessState()
    yield
    sensitive._state = sensitive._ProcessState()


# --------------------------------------------------------------------------- the agent rigs

#: Modules whose bridge tests build an agent rig (``tests.strands_parity``): the component modules
#: that build a bridge rig, and every end-to-end scenario. They run on the real Strands client.
_RIG_NAMES = ("make_rig", "new_agent", "direct")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "protocol_fake(reason): a bridge test whose rig gets the plain scripted agent fake, "
        "for inputs the real agent client never produces")


def _rig_module(module: object) -> bool:
    name = getattr(module, "__name__", "")
    if name.startswith("tests.e2e.test_"):
        return True
    return name.startswith("tests.component.test_") and any(n in vars(module) for n in _RIG_NAMES)


@pytest.fixture(autouse=True)
def _agent_rig(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """In a rig module: a fresh fake Memory per test, and the light Memory client (or the plain
    fake for a ``protocol_fake`` test). Other tests import nothing heavy here."""
    if not _rig_module(request.module):
        yield
        return
    from tests.strands_parity import reset

    plain = request.node.get_closest_marker("protocol_fake") is not None
    reset(plain=plain)
    if not plain:
        from tests.strands_fakes import light_memory_client

        light_memory_client(monkeypatch)
    yield
    reset()
