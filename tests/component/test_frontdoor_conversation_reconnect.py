"""Reconnect ConversationRelay to the call's existing Orchestrator conversation."""

from __future__ import annotations

from lambdas.frontdoor.app import FrontDoorConfig
from tests.component.test_frontdoor_support import build, relay, sid

CONV = "conv_conversation_01m4w5yjaae0nvv2s1ayqbh1fg"


def _balance(fd, call_sid, *, conversation_id=CONV):
    fd.set_pending_tool(call_sid, "get_balance", {"account": "checking"})
    data = {"reason": "tool", "tool": "get_balance", "args": {"account": "checking"}, "journal": "jr-1"}
    if conversation_id:
        data["conversationId"] = conversation_id
    return fd.handoff(call_sid, data)


def test_reconnect_uses_the_calls_conversation_when_enabled():
    fd = build(config_kw={"reuse_conversation": True})
    first = fd.login_thomas(sid(1))
    assert relay(first).get("conversationId") is None  # nothing known yet: grouping rules apply
    assert relay(first).get("conversationConfiguration")
    cr = relay(_balance(fd, sid(1)))
    assert cr.get("conversationId") == CONV
    assert cr.get("conversationConfiguration") is None


def test_default_is_unchanged():
    fd = build()
    fd.login_thomas(sid(1))
    cr = relay(_balance(fd, sid(1)))
    assert cr.get("conversationId") is None
    assert cr.get("conversationConfiguration")


def test_enabled_without_a_known_conversation_falls_back_to_the_configuration():
    fd = build(config_kw={"reuse_conversation": True})
    fd.login_thomas(sid(1))
    cr = relay(_balance(fd, sid(1), conversation_id=None))
    assert cr.get("conversationId") is None and cr.get("conversationConfiguration")


def test_from_env_reads_the_flag():
    base = {"OWLBANK_RELAY_WS_URL": "wss://relay.example.test/ws"}
    assert FrontDoorConfig.from_env(base).reuse_conversation is False
    assert FrontDoorConfig.from_env({**base, "OWLBANK_REUSE_CONVERSATION": "true"}).reuse_conversation is True
