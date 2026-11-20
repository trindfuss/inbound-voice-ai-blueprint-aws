"""Snapshot-style tests for owlbank.twiml."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from owlbank import twiml as t

HEAD = '<?xml version="1.0" encoding="UTF-8"?>'


def test_empty_response() -> None:
    assert t.empty() == f"{HEAD}<Response />"


def test_say_with_voice_and_language() -> None:
    xml = t.render(t.Say("Welcome to OwlBank.", voice="ElevenLabs.abc", language="en-US"))
    assert xml == (
        f'{HEAD}<Response><Say language="en-US" voice="ElevenLabs.abc">'
        "Welcome to OwlBank.</Say></Response>"
    )


def test_say_escapes_markup_ampersand_and_quotes() -> None:
    text = 'Tom & Jerry <script> "quoted" \'single\''
    xml = t.render(t.Say(text, language="en-US"))
    assert "Tom &amp; Jerry &lt;script&gt;" in xml
    assert "<script>" not in xml
    # Parsed back, the text is exactly what went in.
    assert ET.fromstring(xml.removeprefix(HEAD)).find("Say").text == text


def test_attribute_values_escape_quotes() -> None:
    xml = t.render(t.Say("hi", voice='x" onload="y'))
    assert 'voice="x&quot; onload=&quot;y"' in xml
    assert ET.fromstring(xml.removeprefix(HEAD)).find("Say").get("voice") == 'x" onload="y'


def test_play_pause_redirect_hangup() -> None:
    xml = t.render(
        t.Play("https://example.com/a.mp3"), t.Pause(2), t.Redirect("/voice/incoming"), t.Hangup()
    )
    assert xml == (
        f"{HEAD}<Response><Play>https://example.com/a.mp3</Play><Pause length=\"2\" />"
        '<Redirect method="POST">/voice/incoming</Redirect><Hangup /></Response>'
    )


def test_language_gather() -> None:
    xml = t.render(
        t.Gather(
            action="/voice/language",
            input="dtmf",
            num_digits=1,
            timeout=3,
            prompts=[t.Say("Para español, oprima dos.", voice="ElevenLabs.es", language="es-US")],
        ),
        t.Redirect("/voice/language"),
    )
    assert xml == (
        f'{HEAD}<Response><Gather action="/voice/language" input="dtmf" method="POST" '
        'numDigits="1" timeout="3"><Say language="es-US" voice="ElevenLabs.es">'
        "Para español, oprima dos.</Say></Gather>"
        '<Redirect method="POST">/voice/language</Redirect></Response>'
    )


def test_factor_gather_all_attributes() -> None:
    xml = t.render(
        t.Gather(
            action="/voice/factor?step=zip",
            num_digits=5,
            timeout=6,
            language="en-US",
            hints=["checking", "savings"],
            speech_model="deepgram_nova-3",
            finish_on_key="#",
            action_on_empty_result=True,
            prompts=[t.Say("Please enter your five-digit ZIP code.", language="en-US"),
                     t.Play("https://example.com/beep.mp3")],
        )
    )
    assert xml == (
        f'{HEAD}<Response><Gather action="/voice/factor?step=zip" actionOnEmptyResult="true" '
        'finishOnKey="#" hints="checking, savings" input="dtmf speech" language="en-US" '
        'method="POST" numDigits="5" speechModel="deepgram_nova-3" timeout="6">'
        '<Say language="en-US">Please enter your five-digit ZIP code.</Say>'
        "<Play>https://example.com/beep.mp3</Play></Gather></Response>"
    )


def test_enqueue_with_task_json() -> None:
    attrs = {"name": "Thomas & Co", "escalation_reason": "fraud_suspected", "auth_level": 2,
             "ai_summary": 'Said "hi" <ok>'}
    xml = t.render(t.Enqueue("WW123", attrs, wait_url="/wait?lang=es-US", task_priority=100))
    assert xml == (
        f'{HEAD}<Response><Enqueue waitUrl="/wait?lang=es-US" workflowSid="WW123">'
        '<Task priority="100">{"ai_summary":"Said \\"hi\\" &lt;ok&gt;","auth_level":2,'
        '"escalation_reason":"fraud_suspected","name":"Thomas &amp; Co"}</Task>'
        "</Enqueue></Response>"
    )
    body = ET.fromstring(xml.removeprefix(HEAD)).find("Enqueue/Task").text
    assert json.loads(body) == attrs


def test_connect_conversation_relay_full() -> None:
    xml = t.render(
        t.ConversationRelay(
            url="wss://bridge.example.com/ws",
            action="/voice/connect-action",
            welcome_greeting="Thank you. Welcome back to OwlBank, Thomas & Ana.",
            language="en-US",
            tts_provider="ElevenLabs",
            voice="UgBBYS2sOqTuMpoF3BR0-flash_v2",
            transcription_provider="Deepgram",
            transcription_language="en-US",
            speech_model="nova-3-general",
            interruptible="speech",
            welcome_greeting_interruptible="none",
            dtmf_detection=True,
            events="speaker-events",
            hints=["checking", "savings"],
            languages=[
                t.RelayLanguage("es-US", tts_provider="ElevenLabs", voice="CaJslL1xziwefCeTNzHv",
                                transcription_provider="Deepgram",
                                extra_attrs={"transcriptionLanguage": "es"}),
            ],
            parameters={"sessionRef": "ref-1"},
            extra_attrs={"conversationConfiguration": "CFG123", "elevenlabsTextNormalization": "on"},
        )
    )
    assert xml == (
        f'{HEAD}<Response><Connect action="/voice/connect-action"><ConversationRelay '
        'conversationConfiguration="CFG123" dtmfDetection="true" '
        'elevenlabsTextNormalization="on" events="speaker-events" hints="checking, savings" interruptible="speech" '
        'language="en-US" speechModel="nova-3-general" transcriptionLanguage="en-US" '
        'transcriptionProvider="Deepgram" ttsProvider="ElevenLabs" '
        'url="wss://bridge.example.com/ws" voice="UgBBYS2sOqTuMpoF3BR0-flash_v2" '
        'welcomeGreeting="Thank you. Welcome back to OwlBank, Thomas &amp; Ana." '
        'welcomeGreetingInterruptible="none">'
        '<Language code="es-US" transcriptionLanguage="es" transcriptionProvider="Deepgram" '
        'ttsProvider="ElevenLabs" voice="CaJslL1xziwefCeTNzHv" />'
        '<Parameter name="sessionRef" value="ref-1" />'
        "</ConversationRelay></Connect></Response>"
    )


def test_conversation_relay_always_renders_welcome_greeting_even_empty() -> None:
    xml = t.render(t.ConversationRelay(url="wss://x/ws", action="/a", welcome_greeting="",
                                       language="es-US"))
    relay = ET.fromstring(xml.removeprefix(HEAD)).find("Connect/ConversationRelay")
    assert relay.get("welcomeGreeting") == ""


def test_report_input_during_agent_speech_is_rendered_only_when_set() -> None:
    def relay(value: str | None) -> ET.Element:
        xml = t.render(t.ConversationRelay(url="wss://x/ws", action="/a", welcome_greeting="",
                                           language="en-US",
                                           report_input_during_agent_speech=value))
        el = ET.fromstring(xml.removeprefix(HEAD)).find("Connect/ConversationRelay")
        assert el is not None
        return el

    assert "reportInputDuringAgentSpeech" not in relay(None).attrib
    assert relay("dtmf").get("reportInputDuringAgentSpeech") == "dtmf"


@pytest.mark.parametrize("bad", ['x"y', "a b", "1abc", "on-load", "x>y", ""])
def test_extra_attr_names_are_validated(bad: str) -> None:
    with pytest.raises(ValueError):
        t.render(t.ConversationRelay(url="wss://x", action="/a", welcome_greeting="",
                                     language="en-US", extra_attrs={bad: "v"}))


def test_render_rejects_non_verbs() -> None:
    with pytest.raises(TypeError):
        t.render("<Say>raw</Say>")  # type: ignore[arg-type]


def test_transcription_renders_inside_start_with_both_tracks() -> None:
    xml = t.render(t.Transcription("CH123"))
    assert xml == (
        f'{HEAD}<Response><Start><Transcription conversationId="CH123" '
        'track="both_tracks" /></Start></Response>'
    )


def test_transcription_escapes_conversation_id_and_precedes_enqueue() -> None:
    xml = t.render(t.Transcription('a&b"<c>'), t.Enqueue("WW1", {"k": "v"}))
    assert "a&amp;b&quot;&lt;c&gt;" in xml
    root = ET.fromstring(xml.removeprefix(HEAD))
    assert [child.tag for child in root] == ["Start", "Enqueue"]
    el = root.find("Start/Transcription")
    assert el is not None
    assert el.get("conversationId") == 'a&b"<c>'
    assert el.get("track") == "both_tracks"
