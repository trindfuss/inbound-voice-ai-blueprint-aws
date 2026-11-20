"""The session-ownership suite on ``agentcore``.

Every test of ``test_bridge_session_ownership`` runs again here with the ConversationRelay
simulator in ``agentcore`` mode: each rig is one process with its own :class:`SessionPin`, each
socket passes the call binding (session ID = ``setup.callSid`` = ``customParameters.callSid``) and
carries its nonce as ``customParameters["n"]``. Ownership must come out exactly as on Fargate
(``new``, ``cached``, ``reload``). The single-conversation guard has no socket in its own suite (it
wraps TAC's Orchestrator client); the property test (``tests/property/test_ws_binding_props.py``)
drives it through the ``agentcore`` socket path.
"""

from __future__ import annotations

import pytest

from tests.component import test_bridge_fakes as fakes
from tests.component.test_bridge_session_ownership import *  # noqa: F403 - re-collected on agentcore


@pytest.fixture(autouse=True)
def _agentcore_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fakes, "SIM_HOSTING", "agentcore")
