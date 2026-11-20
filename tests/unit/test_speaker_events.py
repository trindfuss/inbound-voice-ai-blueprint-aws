"""ConversationRelay speaker-event frames (``tac_bridge.speaker_events``)."""

from __future__ import annotations

from typing import Any

import pytest

from tac_bridge.silence import AGENT, CALLER
from tac_bridge.speaker_events import SpeakerEvent, is_speaker_frame, parse_speaker_event, shape_of

SECRET = "my secret words 4111"


@pytest.mark.parametrize(("msg", "expected"), [
    ({"type": "info", "name": "clientSpeaking", "value": "on"}, SpeakerEvent("clientSpeaking", CALLER, True)),
    ({"type": "info", "name": "clientSpeaking", "value": "off"}, SpeakerEvent("clientSpeaking", CALLER, False)),
    ({"type": "info", "name": "agentSpeaking", "value": "ON"}, SpeakerEvent("agentSpeaking", AGENT, True)),
    ({"type": "info", "name": "agentSpeaking", "value": False}, SpeakerEvent("agentSpeaking", AGENT, False)),
    ({"type": "clientSpeaking", "value": "on"}, SpeakerEvent("clientSpeaking", CALLER, True)),
    ({"type": "agentSpeaking", "value": " off "}, SpeakerEvent("agentSpeaking", AGENT, False)),
])
def test_recognised_shapes(msg: dict[str, Any], expected: SpeakerEvent) -> None:
    assert is_speaker_frame(msg)
    assert parse_speaker_event(msg) == expected
    assert expected.value in ("on", "off")


@pytest.mark.parametrize("msg", [
    {"type": "info"},
    {"type": "info", "name": "clientSpeaking"},
    {"type": "info", "name": "clientSpeaking", "value": "maybe"},
    {"type": "info", "name": "clientSpeaking", "value": 1},
    {"type": "info", "name": "tokensPlayed", "value": "on"},
    {"type": "info", "name": SECRET, "value": "on"},
    {"type": "clientSpeaking"},
    {"type": "clientSpeaking", "value": SECRET},
])
def test_unrecognised_shapes_are_owned_but_not_parsed(msg: dict[str, Any]) -> None:
    assert is_speaker_frame(msg)
    assert parse_speaker_event(msg) is None


@pytest.mark.parametrize("msg", [
    {"type": "prompt", "voicePrompt": "hi"}, {"type": "interrupt"}, {"type": "dtmf", "digit": "1"},
    {"type": "setup"}, {"type": "error", "description": "x"}, {}, {"name": "clientSpeaking", "value": "on"},
])
def test_other_frames_are_not_speaker_frames(msg: dict[str, Any]) -> None:
    assert not is_speaker_frame(msg)
    assert parse_speaker_event(msg) is None


def test_the_shape_log_has_the_type_and_key_names_only() -> None:
    msg = {"type": "info", "name": SECRET, "value": SECRET, "text": SECRET}
    shape = shape_of(msg)
    assert shape == {"type": "info", "keys": ["name", "text", "type", "value"]}
    assert SECRET not in repr(shape)
    many = {"type": "info", **{f"k{i:02d}" + "x" * 50: SECRET for i in range(30)}}
    shape = shape_of(many)
    assert len(shape["keys"]) <= 12 and all(len(k) <= 32 for k in shape["keys"])
    assert SECRET not in repr(shape)
    assert shape_of({"type": SECRET})["type"] == "other"  # a type it doesn't own is never echoed
