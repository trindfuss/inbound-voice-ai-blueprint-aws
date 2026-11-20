"""TwiML builders for the front door and the ``<Connect action>`` handler.

Each verb is a small frozen dataclass; :func:`render` turns a sequence of verbs
into one ``<Response>`` document string. Rendering uses Twilio's
``twilio.twiml.voice_response.VoiceResponse``, which XML-escapes text and
attribute values (``<``, ``&``, quotes) and converts ``snake_case`` keyword
attributes to TwiML's ``lowerCamelCase``.

Attributes Twilio's SDK doesn't yet name (for example a newer ConversationRelay
attribute) go in ``extra_attrs`` with their exact TwiML name. Names are checked
against a strict pattern before being passed through, so no caller can inject
markup through an attribute name; values are escaped by the SDK.

Text passed to :class:`Say` must come from the prompt catalog or server-side
values. These builders don't look anything up.

Note: TAC's own ``generate_twiml`` may replace :class:`ConversationRelay` later
Task checks its options and decides. Until then this builder
produces the identical attribute set.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from twilio.twiml.voice_response import VoiceResponse

_ATTR_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


def _attrs(extra: Mapping[str, str | int | bool] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in (extra or {}).items():
        if not _ATTR_NAME.match(name):
            raise ValueError(f"invalid TwiML attribute name {name!r}")
        out[name] = _fmt(value)
    return out


def _fmt(value: str | int | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _clean(**kwargs: Any) -> dict[str, str]:
    """Drop ``None`` values and stringify the rest (booleans as ``true``/``false``)."""
    return {k: _fmt(v) for k, v in kwargs.items() if v is not None}


# --------------------------------------------------------------------------- verbs


@dataclass(frozen=True, slots=True)
class Say:
    """``<Say voice language>text</Say>``. Voice and language come from the catalog."""

    text: str
    voice: str | None = None
    language: str | None = None


@dataclass(frozen=True, slots=True)
class Play:
    """``<Play>url</Play>``, for catalog entries with an ``audioUrl``."""

    url: str
    loop: int | None = None


@dataclass(frozen=True, slots=True)
class Pause:
    """``<Pause length/>``."""

    length: int = 1


Prompt = Say | Play | Pause


@dataclass(frozen=True, slots=True)
class Gather:
    """``<Gather>`` with nested prompts. Deterministic flows use ``input="dtmf speech"``."""

    action: str
    prompts: Sequence[Prompt] = ()
    input: Literal["dtmf", "speech", "dtmf speech"] = "dtmf speech"
    num_digits: int | None = None
    timeout: int | None = None
    language: str | None = None
    hints: Sequence[str] = ()
    speech_model: str | None = None
    finish_on_key: str | None = None
    #: Seconds of silence that end spoken input (Twilio's default is ``timeout``).
    speech_timeout: int | str | None = None
    method: Literal["GET", "POST"] = "POST"
    action_on_empty_result: bool | None = None
    extra_attrs: Mapping[str, str | int | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Redirect:
    """``<Redirect method>url</Redirect>``."""

    url: str
    method: Literal["GET", "POST"] = "POST"


@dataclass(frozen=True, slots=True)
class Hangup:
    """``<Hangup/>``."""


@dataclass(frozen=True, slots=True)
class Enqueue:
    """``<Enqueue workflowSid waitUrl><Task>{json}</Task></Enqueue>`` for Flex escalation.

    ``task_attributes`` is serialized to compact, key-sorted JSON. Build it from
    server-side state only; it must hold no factor values or tokens.
    """

    workflow_sid: str
    task_attributes: Mapping[str, Any]
    wait_url: str | None = None
    action: str | None = None
    task_priority: int | None = None
    task_timeout: int | None = None


@dataclass(frozen=True, slots=True)
class Transcription:
    """``<Start><Transcription conversationId track="both_tracks"/></Start>``.

    Forks the call leg into a Conversations conversation: the customer (inbound track) and
    everything played to them, such as a human agent (outbound track). Emit it before the
    verb that connects the call; ``conversation_id`` is XML-escaped on render.
    """

    conversation_id: str
    track: Literal["inbound_track", "outbound_track", "both_tracks"] = "both_tracks"


@dataclass(frozen=True, slots=True)
class RelayLanguage:
    """A ConversationRelay ``<Language>`` element (per-language STT/TTS settings)."""

    code: str
    tts_provider: str | None = None
    voice: str | None = None
    transcription_provider: str | None = None
    speech_model: str | None = None
    extra_attrs: Mapping[str, str | int | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConversationRelay:
    """``<Connect action><ConversationRelay ...>`` with nested ``<Language>`` and ``<Parameter>``.

    ``welcome_greeting`` is always rendered, even when empty, because TAC adds its
    own default greeting if the attribute is missing. ``parameters``
    are defence in depth only; the bridge reads call state by ``CallSid``.
    """

    url: str
    action: str
    welcome_greeting: str
    language: str
    tts_provider: str | None = None
    voice: str | None = None
    transcription_provider: str | None = None
    transcription_language: str | None = None
    tts_language: str | None = None
    speech_model: str | None = None
    interruptible: str | None = None
    welcome_greeting_interruptible: str | None = None
    dtmf_detection: bool | None = None
    #: ``reportInputDuringAgentSpeech``: ``none`` (Twilio's default since May 2025), ``dtmf``,
    #: ``speech`` or ``any``. Rendered only when set.
    report_input_during_agent_speech: str | None = None
    #: ``events``: a space-separated list of ConversationRelay event groups
    #: (``speaker-events``).
    events: str | None = None
    hints: Sequence[str] = ()
    intelligence_service: str | None = None
    languages: Sequence[RelayLanguage] = ()
    parameters: Mapping[str, str] = field(default_factory=dict)
    extra_attrs: Mapping[str, str | int | bool] = field(default_factory=dict)


Verb = Say | Play | Pause | Gather | Redirect | Hangup | Enqueue | Transcription | ConversationRelay


# --------------------------------------------------------------------------- rendering


def _add_prompt(parent: Any, prompt: Prompt) -> None:
    if isinstance(prompt, Say):
        parent.say(prompt.text, **_clean(voice=prompt.voice, language=prompt.language))
    elif isinstance(prompt, Play):
        parent.play(prompt.url, **_clean(loop=prompt.loop))
    elif isinstance(prompt, Pause):
        parent.pause(length=prompt.length)
    else:  # pragma: no cover - guarded by typing
        raise TypeError(f"not a prompt: {prompt!r}")


def _add_verb(resp: VoiceResponse, verb: Verb) -> None:
    if isinstance(verb, (Say, Play, Pause)):
        _add_prompt(resp, verb)
    elif isinstance(verb, Gather):
        g = resp.gather(
            **_clean(
                input=verb.input,
                action=verb.action,
                method=verb.method,
                num_digits=verb.num_digits,
                timeout=verb.timeout,
                language=verb.language,
                hints=", ".join(verb.hints) if verb.hints else None,
                speech_model=verb.speech_model,
                finish_on_key=verb.finish_on_key,
                speech_timeout=verb.speech_timeout,
                action_on_empty_result=verb.action_on_empty_result,
            ),
            **_attrs(verb.extra_attrs),
        )
        for prompt in verb.prompts:
            _add_prompt(g, prompt)
    elif isinstance(verb, Redirect):
        resp.redirect(verb.url, method=verb.method)
    elif isinstance(verb, Hangup):
        resp.hangup()
    elif isinstance(verb, Enqueue):
        e = resp.enqueue(
            **_clean(workflow_sid=verb.workflow_sid, wait_url=verb.wait_url, action=verb.action)
        )
        body = json.dumps(verb.task_attributes, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)
        e.task(body, **_clean(priority=verb.task_priority, timeout=verb.task_timeout))
    elif isinstance(verb, Transcription):
        resp.start().transcription(conversation_id=verb.conversation_id, track=verb.track)
    elif isinstance(verb, ConversationRelay):
        connect = resp.connect(action=verb.action)
        cr = connect.conversation_relay(
            **_clean(
                url=verb.url,
                welcome_greeting=verb.welcome_greeting,
                language=verb.language,
                tts_provider=verb.tts_provider,
                voice=verb.voice,
                transcription_provider=verb.transcription_provider,
                transcription_language=verb.transcription_language,
                tts_language=verb.tts_language,
                speech_model=verb.speech_model,
                interruptible=verb.interruptible,
                welcome_greeting_interruptible=verb.welcome_greeting_interruptible,
                dtmf_detection=verb.dtmf_detection,
                report_input_during_agent_speech=verb.report_input_during_agent_speech,
                events=verb.events,
                hints=", ".join(verb.hints) if verb.hints else None,
                intelligence_service=verb.intelligence_service,
            ),
            **_attrs(verb.extra_attrs),
        )
        for lang in verb.languages:
            cr.language(
                **_clean(
                    code=lang.code,
                    tts_provider=lang.tts_provider,
                    voice=lang.voice,
                    transcription_provider=lang.transcription_provider,
                    speech_model=lang.speech_model,
                ),
                **_attrs(lang.extra_attrs),
            )
        for name, value in verb.parameters.items():
            cr.parameter(name=name, value=value)
    else:
        raise TypeError(f"not a TwiML verb: {verb!r}")


def render(*verbs: Verb) -> str:
    """Render verbs, in order, as one ``<Response>`` TwiML document string."""
    resp = VoiceResponse()
    for verb in verbs:
        _add_verb(resp, verb)
    return str(resp.to_xml())


def empty() -> str:
    """An empty ``<Response/>`` (used for status callbacks)."""
    return render()
