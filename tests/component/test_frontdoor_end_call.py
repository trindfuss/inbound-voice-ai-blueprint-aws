"""End-call tool, front door: the bridge's ``caller_done`` hand-off says
``common.goodbye`` in the call's language and hangs up, like the silence hang-up; the token is
revoked, the journal and log show ``caller_done``, the recap still starts at call end, and
no failure turns the ending into an escalation."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import replace
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import app as fd_app
from lambdas.frontdoor.app import NO_INPUT, NO_INPUT_HANGUP, _hangup_reason
from owlbank import callstate, tokens
from owlbank.segments import clear_sensitive
from owlbank.tools_contract import CALLER_DONE

from tests.component.test_frontdoor_support import (  # noqa: F401  (fd is a fixture)
    THOMAS, Harness, build, fd, says, sid,
)

EN, ES = "en-US", "es-US"
GOODBYE = {EN: "Thank you for calling OwlBank. Goodbye.", ES: "Gracias por llamar a OwlBank. Adiós."}
END_CALL = {"reason": "caller_done", "tool": "end_call", "journal": "jr-e1"}


def login(h: Harness, call: str, lang: str = EN) -> None:
    h.start(call, THOMAS, lang_digit="2" if lang == ES else "")
    h.factor(call, "94105")
    callstate.mutate(h.store, call, lambda s: {"ai_sessions": 1})  # an AI-only call


def events(records: list[logging.LogRecord], name: str) -> list[logging.LogRecord]:
    return [r for r in records if r.getMessage() == name]


@pytest.mark.parametrize("lang", [EN, ES])
def test_caller_done_says_goodbye_and_hangs_up(fd: Harness, caplog, lang: str) -> None:
    call = sid(1)
    login(fd, call, lang)
    assert fd.state(call).level == 1
    root = fd.handoff(call, END_CALL)
    assert [c.tag for c in root] == ["Say", "Hangup"]
    assert says(root) == [GOODBYE[lang]]
    st = fd.state(call)
    assert st.sealed_token is None and st.last_spoken is None
    assert all(t["revoked"] for t in fd.store.query("tokens", {"call_sid": call},
                                                    index="call_sid-index"))
    journal = fd.journal(call)
    assert [e.get("prompt_key") for e in journal if e["type"] == "prompt"][-1] == "common.goodbye"
    (end,) = [e for e in journal if e["type"] == "handoff" and e.get("to") == "frontdoor"]
    assert end["outcome"] == CALLER_DONE  # not a silence hang-up
    (line,) = events(caplog.records, "caller_done_hangup")
    assert line.level_reached == 1
    assert not events(caplog.records, "no_input_hangup")


def test_silence_hangup_keeps_its_own_phrase_and_event(fd: Harness, caplog) -> None:
    call = sid(2)
    login(fd, call)
    root = fd.handoff(call, {"reason": "no_input_hangup"})
    assert says(root) == [fd.app.d.catalog.text("common.goodbye_no_input", EN)]
    assert [e["outcome"] for e in fd.journal(call)
            if e["type"] == "handoff" and e.get("to") == "frontdoor"] == [NO_INPUT_HANGUP]
    assert events(caplog.records, "no_input_hangup") and not events(caplog.records,
                                                                    "caller_done_hangup")


def test_caller_done_wins_over_a_pending_tool(fd: Harness) -> None:
    call = sid(3)
    login(fd, call)
    fd.set_pending_tool(call, "get_balance", {"account": "checking"})
    root = fd.handoff(call, END_CALL)
    assert [c.tag for c in root] == ["Say", "Hangup"]
    assert "get_balance" not in fd.lam.ops()


class RecordingRecap:
    def __init__(self) -> None:
        self.fired: list[str] = []

    def __call__(self, call_sid: str) -> None:
        self.fired.append(call_sid)


def test_the_recap_still_starts_at_call_end(fd: Harness) -> None:
    """The hang-up itself starts nothing; Twilio's ``completed`` status does, once."""
    recap = RecordingRecap()
    fd.app.d = replace(fd.app.d, recap=recap)
    call = sid(4)
    login(fd, call)
    fd.handoff(call, END_CALL)
    assert recap.fired == []
    fd.post("status", call, CallStatus="completed")
    fd.post("status", call, CallStatus="completed")  # Twilio's retry
    assert recap.fired == [call]


def test_a_failed_revoke_still_says_goodbye_and_hangs_up(fd: Harness, caplog) -> None:
    call = sid(5)
    login(fd, call)
    with mock.patch.object(fd_app.tokens, "revoke_call", side_effect=RuntimeError("store down")):
        root = fd.handoff(call, END_CALL)
    assert [c.tag for c in root] == ["Say", "Hangup"] and says(root) == [GOODBYE[EN]]
    (line,) = events(caplog.records, "hangup token revoke failed")
    assert line.decision == "hangup_anyway"


def test_a_missing_goodbye_says_the_generic_error_then_hangs_up(fd: Harness) -> None:
    call = sid(6)
    login(fd, call)
    real = fd.app._say  # noqa: SLF001

    def say(key: str, lang: str, **values: Any):
        if key == "common.goodbye":
            raise KeyError(key)
        return real(key, lang, **values)

    with mock.patch.object(fd.app, "_say", side_effect=say):
        root = fd.handoff(call, END_CALL)
    assert [c.tag for c in root] == ["Say", "Hangup"]
    assert says(root) == [fd.app.d.catalog.text("error.generic", EN)]  # 5.2, never escalated



_REASONS = ["caller_done", "Caller-Done", "CALLER_DONE", "caller done", "no_input",
            "no_input_hangup", "No-Input-Hangup", "escalate", "step_up", "authenticate",
            "get_balance", "end_call", "", None, 3, ["caller_done"]]


def _norm(v: Any) -> str:
    return v.lower().replace("-", "_") if isinstance(v, str) else ""


@settings(max_examples=300, deadline=None)
@given(reason=st.sampled_from(_REASONS), esc=st.one_of(st.none(), st.sampled_from(_REASONS)),
       tool=st.sampled_from([None, "end_call", "escalate", "authenticate"]))
def test_caller_done_is_never_mixed_with_silence_or_escalation(reason, esc, tool) -> None:
    """``caller_done`` only from its own form; never an escalation or a silence hang-up, and
    an escalation or silence form is never ``caller_done``."""
    data: dict[str, Any] = {"reason": reason}
    if esc is not None:
        data["escalationReason"] = esc
    if tool is not None:
        data["tool"] = tool
    got = _hangup_reason(data)
    r, e = _norm(reason), _norm(esc)
    assert (got == CALLER_DONE) == (r == CALLER_DONE)
    if r == "escalate":
        assert got != CALLER_DONE
        assert got == (e if e in (NO_INPUT, NO_INPUT_HANGUP) else None)
    if got in (NO_INPUT, NO_INPUT_HANGUP):
        assert r != CALLER_DONE and got in (r, e)



_FAILURES = ["state", "revoke", "session", "goodbye", "generic", "journal"]


@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(lang=st.sampled_from([EN, ES]), fail=st.sets(st.sampled_from(_FAILURES)),
       reason=st.sampled_from([CALLER_DONE, NO_INPUT_HANGUP]))
def test_every_failure_still_ends_in_one_hangup(lang: str, fail: set[str], reason: str) -> None:
    """For every language and any failure in state clearing, revocation, the session save,
    the catalog or the journal, the response ends in exactly one ``<Hangup/>``, holds no
    ``<Enqueue>`` or ``<Connect>``, and has at most one goodbye phrase."""
    h = build()
    call = sid(7)
    try:
        login(h, call, lang)
        app = h.app
        st_, sess = app._load(call)  # noqa: SLF001
        real_say, real_update = app._say, app._update  # noqa: SLF001
        real_save, real_event = app._save, app._log_event  # noqa: SLF001
        missing = {k for f, k in (("goodbye", fd_app._HANGUP_PHRASES[reason]),  # noqa: SLF001
                                  ("generic", "error.generic")) if f in fail}

        def boom(*_: Any, **__: Any) -> Any:
            raise RuntimeError("injected")

        def say(key: str, lg: str, **values: Any):
            if key in missing:
                raise KeyError(key)
            return real_say(key, lg, **values)

        with mock.patch.object(app, "_say", side_effect=say), \
                mock.patch.object(app, "_update", side_effect=boom if "state" in fail else real_update), \
                mock.patch.object(app, "_save", side_effect=boom if "session" in fail else real_save), \
                mock.patch.object(app, "_log_event",
                                  side_effect=boom if "journal" in fail else real_event), \
                mock.patch.object(tokens, "revoke_call",
                                  side_effect=boom if "revoke" in fail else tokens.revoke_call):
            body = app._hangup(call, st_, sess, reason)  # noqa: SLF001
        root = ET.fromstring(body.encode())
        tags = [c.tag for c in root]
        assert tags[-1] == "Hangup" and tags.count("Hangup") == 1
        assert root.find(".//Enqueue") is None and root.find(".//Connect") is None
        goodbyes = {h.app.d.catalog.text(k, lang) for k in fd_app._HANGUP_PHRASES.values()}  # noqa: SLF001
        assert sum(1 for t in says(root) if t in goodbyes) <= 1
        if not missing & {fd_app._HANGUP_PHRASES[reason]}:  # noqa: SLF001
            assert says(root) == [h.app.d.catalog.text(fd_app._HANGUP_PHRASES[reason], lang)]  # noqa: SLF001
    finally:
        clear_sensitive(call)


CONV = "CH" + "d" * 32


def _boom(*_: Any, **__: Any) -> Any:
    raise RuntimeError("injected")


@pytest.mark.parametrize("reason", [CALLER_DONE, NO_INPUT_HANGUP])
@pytest.mark.parametrize("fail", ["state_load", "session_load", "session_save"])
def test_a_store_failure_on_the_hangup_path_never_escalates(fd: Harness, reason: str, fail: str) -> None:
    """A failed call-state read, session read or session save on the hang-up
    path still ends in the goodbye and ``<Hangup/>`` (never the apology path, which escalates);
    the token is still revoked and the hang-up journaled. Only what failed is lost: the
    language with the call state, the new conversation id with the session."""
    call = sid(9)
    login(fd, call, ES)
    patch = {"state_load": mock.patch.object(fd.app, "_state", side_effect=_boom),
             "session_load": mock.patch.object(fd_app.sessions, "load", side_effect=_boom),
             "session_save": mock.patch.object(fd.app, "_save", side_effect=_boom)}[fail]
    with patch:
        root = fd.handoff(call, {"reason": reason, "conversationId": CONV})
    assert [c.tag for c in root] == ["Say", "Hangup"]
    lang = EN if fail == "state_load" else ES
    assert says(root) == [fd.app.d.catalog.text(fd_app._HANGUP_PHRASES[reason], lang)]  # noqa: SLF001
    assert root.find(".//Enqueue") is None
    assert all(t["revoked"] for t in fd.store.query("tokens", {"call_sid": call},
                                                    index="call_sid-index"))
    (end,) = [e for e in fd.journal(call) if e["type"] == "handoff" and e.get("to") == "frontdoor"]
    assert end["outcome"] == reason
    # the session is read on its own: only a failed session read or save loses the new id
    assert (fd.session(call).conversation_id == CONV) == (fail == "state_load")


def test_the_hangup_saves_the_new_conversation_id(fd: Harness) -> None:
    call = sid(10)
    login(fd, call)
    fd.handoff(call, {**END_CALL, "conversationId": CONV})
    assert fd.session(call).conversation_id == CONV


@pytest.mark.parametrize("missing", [{"common.goodbye"}, {"common.goodbye", "error.generic"}])
def test_without_call_state_the_goodbye_chain_is_unchanged(fd: Harness, missing: set[str]) -> None:
    """Without call state the goodbye chain is the same as with it: the reason's phrase, then
    ``error.generic``, then a bare ``<Hangup/>``."""
    call = sid(11)
    login(fd, call)
    real_say = fd.app._say  # noqa: SLF001

    def say(key: str, lg: str, **values: Any):
        if key in missing:
            raise KeyError(key)
        return real_say(key, lg, **values)

    with mock.patch.object(fd.app, "_state", side_effect=_boom), \
            mock.patch.object(fd.app, "_say", side_effect=say):
        root = fd.handoff(call, END_CALL)
    if "error.generic" in missing:
        assert [c.tag for c in root] == ["Hangup"]
    else:
        assert [c.tag for c in root] == ["Say", "Hangup"]
        assert says(root) == [fd.app.d.catalog.text("error.generic", EN)]


def test_a_missing_call_state_on_the_hangup_path_is_logged(fd: Harness, caplog) -> None:
    call = sid(12)
    login(fd, call)
    with mock.patch.object(fd.app, "_state", return_value=None):
        root = fd.handoff(call, END_CALL)
    assert [c.tag for c in root] == ["Say", "Hangup"]
    assert events(caplog.records, "hangup_without_call_state")
    (line,) = events(caplog.records, "caller_done_hangup")
    assert not hasattr(line, "level_reached")  # unknown without call state: left out


def test_journal_shape_is_unchanged_for_the_plain_path(fd: Harness) -> None:
    call = sid(8)
    login(fd, call)
    fd.handoff(call, END_CALL)
    (end,) = [e for e in fd.journal(call) if e["type"] == "handoff" and e.get("to") == "frontdoor"]
    assert set(end) - {"call_sid", "seq", "ts", "expires_at"} == {"type", "to", "outcome"}
