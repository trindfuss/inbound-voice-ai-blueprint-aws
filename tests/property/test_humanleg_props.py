"""Switch semantics: a transcription start exists only when the switch is on and the
conversation id is non-empty. Transcription attributes: every start
carries the name, both track labels, engine, language, speech model and status URL,
and its form encoding round-trips."""

from __future__ import annotations

import urllib.parse

from hypothesis import given
from hypothesis import strategies as st

from owlbank import humanleg

IDS = st.one_of(st.none(), st.text(max_size=40))
TEXT = st.text(min_size=1, max_size=40)


@given(enabled=st.booleans(), conversation_id=IDS)
def test_start_only_when_enabled_and_id_non_empty(
    enabled: bool, conversation_id: str | None
) -> None:
    params = humanleg.transcription_start(conversation_id, enabled=enabled, language="en-US",
                                          status_url="https://h/voice/transcription")
    assert (params is not None) == (enabled and bool(conversation_id))
    if params is not None:
        assert params["ConversationId"] == conversation_id
        assert params["Track"] == "both_tracks"


@given(conversation_id=TEXT, language=TEXT, status_url=TEXT, engine=TEXT, speech_model=TEXT)
def test_start_carries_every_attribute(
    conversation_id: str, language: str, status_url: str, engine: str, speech_model: str
) -> None:
    """Name, both labels, engine, language, model and status URL, for any values."""
    params = humanleg.transcription_start(
        conversation_id, enabled=True, language=language, status_url=status_url,
        engine=engine, speech_model=speech_model)
    assert params == {
        "Name": humanleg.TRANSCRIPTION_NAME, "Track": "both_tracks",
        "InboundTrackLabel": "customer", "OutboundTrackLabel": "agent",
        "TranscriptionEngine": engine, "LanguageCode": language, "SpeechModel": speech_model,
        "StatusCallbackUrl": status_url, "ConversationId": conversation_id}
    # What TwilioCalls.start_transcription sends: the form body decodes to the same values.
    body = urllib.parse.urlencode(dict(params))
    assert dict(urllib.parse.parse_qsl(body, keep_blank_values=True)) == params
