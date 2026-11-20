"""Stopgap: the bridge lengthens TAC's conversation polling window."""

from __future__ import annotations

import pytest

from tac.channels.voice.conversation_relay import provider
from tac_bridge import app


@pytest.fixture(autouse=True)
def _restore():
    before = provider._POLL_ATTEMPTS
    yield
    provider._POLL_ATTEMPTS = before


def test_polling_is_set_to_about_thirty_seconds():
    app.set_tac_conversation_polling()
    assert provider._POLL_ATTEMPTS == app.TAC_CONVERSATION_POLL_ATTEMPTS == 20
    # TAC's backoff: base * 2**n capped at the max delay; the waits between tries add up to about 25 s,
    # plus one list call per try.
    waits = sum(min(provider._POLL_BASE_DELAY * (2 ** n), provider._POLL_MAX_DELAY)
                for n in range(provider._POLL_ATTEMPTS - 1))
    assert 20 <= waits <= 35


def test_the_provider_reads_the_module_constant_at_call_time():
    """The stopgap only works if TAC reads ``_POLL_ATTEMPTS`` when it polls, not at import."""
    import inspect

    src = inspect.getsource(provider.ConversationRelayProvider._initialize_conversation) \
        if hasattr(provider, "ConversationRelayProvider") else inspect.getsource(provider)
    assert "range(_POLL_ATTEMPTS)" in src


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_attempts_are_refused(bad):
    with pytest.raises(ValueError):
        app.set_tac_conversation_polling(bad)
