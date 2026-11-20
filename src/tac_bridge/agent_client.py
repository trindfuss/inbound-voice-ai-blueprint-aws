"""The agent behind the bridge: the ``AgentClient`` protocol, its events and errors.

The bridge's only implementation is
:class:`~tac_bridge.agent_client_strands.StrandsAgentClient` (a Strands loop calling Bedrock
directly); another agent loop can be added as one more client. One :meth:`AgentClient.turn`
yields

* :class:`TextDelta` as the model speaks (stream these to the caller). How soon depends on the
  client's text release:

  - ``message``: only for text in a model message that has no tool use, all of it when
    the message ends;
  - ``stream``: each chunk at once, so a ``TextDelta`` can come from a message that later calls a
    tool. :class:`TextBeforeTool` then follows, at most once per message, after that message's
    last ``TextDelta``,

* :class:`TextSuppressed` (a length only) for the text of a message that also calls a tool and
  was not released (in ``stream`` mode: only text written after a tool-use block): the caller
  never hears it,
* :class:`ToolCall` for each **inline** tool call the bridge must run,
* :class:`ToolStarted` when a Level-0 Gateway tool starts inside the loop: observation only,
  never a call to run,
* exactly one :class:`TurnEnd` last.

A failure before or during the stream raises :class:`AgentError`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from owlbank.agent_prompt import model_key

#: The marker the model writes for "say nothing" (handover); the bridge never speaks it
#: (:class:`SilentFilter`).
SILENT_MARK = "<silent/>"


def strip_silent(text: str) -> str:
    """``text`` with every :data:`SILENT_MARK` removed, until none remains: removing one can join
    the text around it into another (``"<sil<silent/>ent/>"``), so a single pass is not enough."""
    while SILENT_MARK in text:
        text = text.replace(SILENT_MARK, "")
    return text


#: The most of a trailing run of marker prefixes (``"<<sil"``) :class:`SilentFilter` holds back, in
#: characters. Beyond it the run's oldest pieces are released (cut at a ``"<"``, so no marker is
#: ever in what one feed releases). Only adversarial text builds such a run, and only there can a
#: marker then form around a released piece and be spoken; the bound keeps what is held, and what
#: each feed re-strips and scans, small whatever the model writes.
MAX_HELD = 4 * len(SILENT_MARK)


class SilentFilter:
    """Drops ``<silent/>`` markers from the token stream (handover: the server drops it). The
    bridge runs a turn's ``TextDelta``\\ s through one filter and flushes it at the end of each hop
    (a failed call's held tail is dropped there); the latency bench uses one filter per hop.

    What it releases between flushes, flush included, is exactly :func:`strip_silent` of that
    text, whatever the chunk boundaries, as long as no run of marker prefixes is longer
    than :data:`MAX_HELD`. Markers are removed until none remain, and the held tail is the longest
    run of marker prefixes at the end (``"<<sil"``), up to :data:`MAX_HELD`: each could still
    become a marker once the ones after it complete (``"<<sil" + "ent/>silent/>"``). Past the
    bound, each release is still marker-free and :func:`strip_silent` of all of them is
    :func:`strip_silent` of the text, but a marker can form across two releases."""

    MARK = SILENT_MARK

    def __init__(self) -> None:
        self._buf = ""  # never longer than MAX_HELD

    def feed(self, text: str) -> str:
        out = strip_silent(self._buf + text)
        cut = len(out)
        floor = max(0, cut - MAX_HELD)
        # The marker has one "<", at its start: the tail is held while each piece from a "<" is
        # a proper prefix of it, back to at most MAX_HELD characters from the end.
        while (i := out.rfind("<", floor, cut)) >= 0 and self.MARK.startswith(out[i:cut]):
            cut = i
        self._buf = out[cut:]
        return out[:cut]

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return out


#: No ``<`` or ``>`` reaches text-to-speech, but the
#: :data:`SILENT_MARK` passed on to :class:`SilentFilter`. A model may write its reasoning as text in
#: a tag NAMED ``thinking`` (any case, spacing, attributes, any length); everything from there to
#: :data:`THINKING_CLOSE` is never spoken.
THINKING = "thinking"
#: What ends a reasoning block: ``</thinking`` in any case, then any spaces, then ``>``.
THINKING_CLOSE = "</" + THINKING
#: The name of the silent marker's tag (any case).
SILENT = "silent"
#: The longest tag other than a ``thinking`` one :class:`ReasoningFilter` holds back; a ``"<"`` not
#: closed by then is text (its brackets dropped). A ``thinking`` tag needs no bound: it opens a block
#: as soon as its name ends.
MAX_TAG = 128

_WS = frozenset(" \t\n\r\f\v")


def _name_char(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c in "_-")


def _key_start(c: str) -> bool:
    return c.isascii() and (c.isalpha() or c == "_")


def _key_char(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c in "_.:-")


def _bare_value_char(c: str) -> bool:
    """A character of an unquoted attribute value (``id=1``): anything but space, quotes, ``=``,
    `````` ` `````` and the brackets."""
    return c not in _WS and c not in "\"'=<>`"


class _TagReader:
    """One possible tag, read a character at a time from its ``"<"`` (the only grammar accepted):
    ``<`` ``/``? a name of ASCII letters, digits, ``_`` or ``-``; then, for an opening tag, only
    ``key="value"``, ``key='value'`` or ``key=value`` (no spaces) attributes; spaces; ``/``?
    (opening only), spaces and ``>``. A closing tag takes spaces before its ``>`` but no attributes.
    In an unquoted value, a ``/`` followed only by spaces and ``>`` ends the tag as self-closing
    (``<silent x=1/>``, ``<silent x=/>``; ``<a href=/x/y>`` keeps its ``/``). Spaces right after the ``<`` are taken
    for the silent marker only (``< silent/>``), so prose (``x < 5``) stays prose. Never sees a
    ``"<"`` (no tag holds one)."""

    __slots__ = ("attrs", "closing", "lead", "name", "quote", "state", "text")

    def __init__(self) -> None:
        self.text = "<"
        self.state = "start"
        self.closing = False
        self.lead = False  # spaces after the "<": only the silent marker may follow
        self.name = ""
        self.attrs = False
        self.quote = ""

    @property
    def self_closing(self) -> bool:
        return self.state == "self"

    @property
    def in_attrs(self) -> bool:
        """Past the name of an opening tag: what is held now is attribute text."""
        return self.state not in ("start", "lead_ws", "slash", "name", "close_ws")

    def step(self, c: str) -> str:
        """``"more"``, ``"tag"`` (``c`` ended a whole tag), ``"thinking"`` (``c`` ended the name of an
        opening tag named ``thinking``: a block starts) or ``"text"`` (no tag)."""
        self.text += c
        state = self.state
        if state == "start":
            if c == "/":
                self.closing, self.state = True, "slash"
                return "more"
            if c in _WS:
                self.lead, self.state = True, "lead_ws"
                return "more"
            state = "slash"
        if state == "lead_ws":
            if c in _WS:
                return "more"
            state = "slash"
        if state == "slash":
            if not _name_char(c) or (self.lead and not SILENT.startswith(c.lower())):
                return "text"
            self.name, self.state = c, "name"
            return "more"
        if state == "name":
            if _name_char(c):
                self.name += c
                return "text" if self.lead and not SILENT.startswith(self.name.lower()) else "more"
            if self.lead and self.name.lower() != SILENT:
                return "text"
            if not self.closing and self.name.lower() == THINKING and (c in _WS or c in "/>"):
                return "thinking"
            if c == ">":
                return "text" if self.lead else "tag"
            if c in _WS:
                self.state = "close_ws" if self.closing else "attr_ws"
                return "more"
            if c == "/" and not self.closing:
                self.state = "self"
                return "more"
            return "text"
        if state == "close_ws":
            return "more" if c in _WS else "tag" if c == ">" else "text"
        if state in ("attr_ws", "after", "slash_ws"):
            if c in _WS:
                self.state = "slash_ws" if state == "slash_ws" else "attr_ws"
                return "more"
            if c == "/":
                self.state = "self"
                return "more"
            if c == ">":
                if self.lead and state != "slash_ws":
                    return "text"
                if state == "slash_ws":  # "x=1/ >": the value's "/" made it self-closing
                    self.state = "self"
                return "tag"
            if state != "after" and not self.lead and _key_start(c):
                self.state, self.attrs = "key", True
                return "more"
            return "text"
        if state in ("key", "key_ws"):
            if c == "=":
                self.state = "eq"
                return "more"
            if c in _WS:
                self.state = "key_ws"
                return "more"
            return "more" if state == "key" and _key_char(c) else "text"
        if state == "eq":
            if c in _WS:
                return "more"
            if c in "\"'":
                self.state, self.quote = "val", c
                return "more"
            if c == "/":  # "x=/>": may end the tag self-closing, as after any unquoted value
                self.state = "bare_slash"
                return "more"
            if _bare_value_char(c):
                self.state = "bare"
                return "more"
            return "text"
        if state in ("bare", "bare_slash"):
            if c == "/":
                self.state = "bare_slash"
                return "more"
            if _bare_value_char(c):
                self.state = "bare"
                return "more"
            if c in _WS:
                self.state = "slash_ws" if state == "bare_slash" else "attr_ws"
                return "more"
            if c == ">":
                if state == "bare_slash":  # "x=1/>": self-closing, not a value "1/"
                    self.state = "self"
                return "tag"
            return "text"
        if state == "val":
            if c == self.quote:
                self.state = "after"
                return "more"
            return "text" if c == ">" else "more"
        if state == "self":
            if c in _WS:
                return "more"
            if c == ">":
                return "tag"
        return "text"


def strip_reasoning(text: str) -> str:
    """``text`` as :class:`ReasoningFilter` releases it, fed whole (one message)."""
    f = ReasoningFilter()
    return f.feed(text) + f.flush()


class ReasoningFilter:
    """Makes a model's text fit for speech: no ``<`` or ``>`` is
    released but :data:`SILENT_MARK`.

    * A reasoning block, from an opening tag named ``thinking`` (any case; spaces, ``/`` or ``>``
      after the name; then any attributes, any length) to :data:`THINKING_CLOSE` (any case, spaces
      before the ``>``), is removed with everything in it; an unclosed one to the end of the message
      (fails closed). A ``>`` inside a quoted value (a quote right after ``=``; any other quote is an
      ordinary character) does not end its opening tag. A self-closing
      ``<thinking .../>`` opens nothing and is removed as a tag.
    * A self-closing tag named ``silent`` (any case or spacing: ``<silent />``, ``<Silent/ >``,
      ``< silent/>``) is
      released as exactly :data:`SILENT_MARK`, for :class:`SilentFilter` after this one.
    * Any other tag (:class:`_TagReader`'s grammar) is removed whole, its text kept
      (``"<message>Hi</message>"`` is ``"Hi"``), and counted in ``tags``.
    * Any other ``<`` or ``>`` is dropped and the words kept (``"<Vacation fund>"`` is ``"Vacation
      fund"``); a space after a dropped bracket that follows a space is dropped too
      (``"a <b and c > d"`` is ``"a b and c d"``). A one-word value (``"<Vacation>"``) is a tag.

    Same shape as :class:`SilentFilter`: :meth:`feed` holds back only a possible tag (from its
    ``"<"``, at most :data:`MAX_TAG` characters; a tag that outgrows it in its attributes is dropped
    to its ``>``, a ``"<"`` or the end of the message, and counted in ``tags``) and :meth:`flush`
    ends a message: an unclosed block is dropped, a held ``<thinking`` too (counted as a block), and
    any other held possible tag is released as text. Each character is read once, so what
    one message releases does not depend on the chunk boundaries. Whitespace right after
    removed markup is dropped while the message has released nothing else, so a reply that starts
    with reasoning starts with its first word; and markup removed between two non-space characters
    leaves one space (``"Hello<thinking>x</thinking>world"`` is ``"Hello world"``), between two
    spaces one (``"x <b>y</b> z"``: no double space). The bridge
    flushes at the end of each hop and of each model message in it.

    ``blocks`` and ``chars`` count the reasoning blocks and their characters (tags included),
    ``tags`` the other tags removed, ``skips`` the over-long tags (also in ``tags``) whose drop ran
    to the end of a message (the bridge counts that toward a swallowed answer), over the
    filter's life (a turn). With ``keep``, :attr:`removed`
    is the blocks' text as written (for the turn span, never a log); without it the text is not
    collected."""

    def __init__(self, *, keep: bool = False) -> None:
        self.blocks = 0
        self.chars = 0
        self.tags = 0
        self.skips = 0
        self._keep = keep
        self._removed: list[str] = []
        self._new_message()

    def _new_message(self) -> None:
        self._tag: _TagReader | None = None  # a possible tag being read
        self._in = False  # inside a reasoning block
        self._block: list[str] = []  # with keep: the block's text so far
        self._block_len = 0  # the block's characters so far
        self._head = False  # in a block, still in its opening tag (to its first ">" outside quotes)
        self._head_quote = ""  # in the opening tag, inside this quote
        self._head_slash = False  # the opening tag's last non-space character is "/"
        self._head_eq = False  # the opening tag's last non-space character is "=": a quote opens a value
        self._close = 0  # in a block: how much of THINKING_CLOSE the text ends with
        self._skip = False  # an over-long tag in its attributes: dropped to its ">"
        self._spoke = False  # the message released a non-whitespace character
        self._cut = False  # markup was removed in this message
        self._last = ""  # the last character the message released
        self._gap = False  # markup was removed right after a non-space character: a space may follow
        self._squeeze = False  # a bracket or markup was dropped after a space: a space next is dropped

    @property
    def removed(self) -> str:
        return "".join(self._removed)

    @property
    def held(self) -> str:
        return "" if self._tag is None else self._tag.text

    def feed(self, text: str) -> str:
        out: list[str] = []
        for c in text:
            self._step(c, out)
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        if self._in:  # unclosed: dropped to the end of the message
            self._end_block()
        tag = self._tag
        if tag is not None and tag.state == "name" and not tag.closing and tag.name.lower() == THINKING:
            # "<thinking" with the next character unseen: a block, fail closed
            self._tag = None
            self._open_block(tag.text)
            self._end_block()
        if self._tag is not None:
            self._release(out)
        if self._skip:  # an over-long tag ran to the end of the message: nothing after it spoken
            self.skips += 1
        self._new_message()
        return "".join(out)

    def _step(self, c: str, out: list[str]) -> None:
        if self._in:
            self._in_block(c)
            return
        if self._skip:  # an over-long tag: to its ">"; a "<" may start a tag (or a block)
            if c == ">":
                self._skip = False
                return
            if c != "<":
                return
            self._skip = False
        if self._tag is None:
            if c == "<":
                self._tag = _TagReader()
            else:
                self._text(c, out)
            return
        if c == "<":  # no tag holds a "<": what was held is text, and a tag may start here
            self._release(out)
            self._tag = _TagReader()
            return
        result = self._tag.step(c)
        if result == "more":
            if len(self._tag.text) > MAX_TAG:
                if self._tag.in_attrs:  # never speak attribute text: drop the tag to its ">"
                    self._tag, self._skip = None, True
                    self.tags += 1
                    self._markup_removed()
                else:
                    self._release(out)
            return
        if result == "text":
            self._release(out)
            return
        tag, self._tag = self._tag, None
        if result == "thinking":
            self._markup_removed()
            self._open_block(tag.text)
            self._head, self._head_slash = c != ">", c == "/"
        elif tag.self_closing and not tag.closing and tag.name.lower() == SILENT:
            self._squeeze = False
            for ch in SILENT_MARK:  # for SilentFilter, exactly as it reads it
                self._emit(ch, out)
        else:
            self.tags += 1
            self._markup_removed()

    def _open_block(self, text: str) -> None:
        self._in, self._close, self._head_quote, self._head_eq = True, 0, "", False
        self._block_len = len(text)
        self._block = [text] if self._keep else []

    def _in_block(self, c: str) -> None:
        self._block_len += 1
        if self._keep:
            self._block.append(c)
        if self._head:  # the opening tag, to its first ">" outside quotes (a "<" ends it too)
            if c != "<":
                if self._head_quote:
                    if c == self._head_quote:
                        self._head_quote = ""
                        self._head_slash = False
                    return
                if c == ">":
                    self._head = False
                    if self._head_slash:  # self-closing: no block, a tag
                        self._in = False
                        self._block = []
                        self.tags += 1
                    return
                if c in "\"'" and self._head_eq:  # a quoted value: only right after "=" and spaces
                    self._head_quote, self._head_slash, self._head_eq = c, False, False
                elif c not in _WS:  # a stray quote ("<thinking don't/>") is an ordinary character
                    self._head_slash, self._head_eq = c == "/", c == "="
                return
            self._head = False
        if self._close == len(THINKING_CLOSE):  # "</thinking" read: spaces, then ">"
            if c == ">":
                self._end_block()
                return
            if c in _WS:
                return
            self._close = 0
        if c == "<":
            self._close = 1
        elif self._close and c.isascii() and c.lower() == THINKING_CLOSE[self._close]:
            self._close += 1
        else:
            self._close = 0

    def _markup_removed(self) -> None:
        self._cut = True
        if self._last == " ":
            self._squeeze = True  # removed between two spaces: one space is left
        elif self._last and not self._last.isspace():
            self._gap = True  # removed between words

    def _end_block(self) -> None:
        self.blocks += 1
        self.chars += self._block_len
        if self._keep:
            self._removed.append("".join(self._block))
        self._in = False
        self._block = []
        self._block_len = 0
        self._head = False
        self._close = 0

    def _release(self, out: list[str]) -> None:
        """What was held is no tag: text, its brackets dropped."""
        held, self._tag = self.held, None
        for ch in held:
            self._text(ch, out)

    def _text(self, c: str, out: list[str]) -> None:
        if c in "<>":  # never spoken
            self._squeeze = self._squeeze or self._last == " "
            return
        squeeze, self._squeeze = self._squeeze, False
        if squeeze and c == " ":
            return
        self._emit(c, out)

    def _emit(self, c: str, out: list[str]) -> None:
        if not self._spoke:
            if c.isspace():
                if self._cut:
                    return
            else:
                self._spoke = True
        if self._gap:
            self._gap = False
            if not c.isspace():
                out.append(" ")
        out.append(c)
        self._last = c


# --------------------------------------------------------------------------- events


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool_use_id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextSuppressed:
    """Text the model wrote in a message that also calls a tool. Only its length."""

    chars: int


@dataclass(frozen=True, slots=True)
class TextBeforeTool:
    """Stream mode: a tool-use block started in a model message after its text was already yielded
    as TextDelta (and so spoken). ``chars`` is the length of what the caller hears of it (without
    :data:`SILENT_MARK`); none is sent if that is blank. ``tool`` is the block's tool name.

    At most once per message, after that message's
    last :class:`TextDelta`. Never sent in ``message`` mode."""

    chars: int
    tool: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """A Level-0 Gateway tool (``name`` is its full Gateway name, e.g. ``level0___faq_lookup``)
    has started inside the agent loop, with the model's ``input``.

    **Observation only**, for latency masking: the tool already runs inside the loop and its result
    goes back to the model without the bridge. The bridge must never treat this as a
    :class:`ToolCall` to run or answer. Emitted once per tool use."""

    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MessageEnd:
    """A model message ended (Strands' ``messageStop``): after its last :class:`TextDelta`,
    :class:`TextBeforeTool` and :class:`ToolStarted`; an inline tool's :class:`ToolCall` follows it.
    Once per model message, so a hop of several (a Gateway tool, or a tool name the model made up,
    runs inside the agent loop) has one after each (the bridge flushes its
    :class:`ReasoningFilter` and starts the per-message WARNING again). A client that never sends
    it leaves the hop's end as the only model-message boundary."""


@dataclass(frozen=True, slots=True)
class TurnEnd:
    stop_reason: str | None
    #: Token counts from the stream's ``metadata.usage`` (``input_tokens``, ``output_tokens``), if sent.
    usage: Mapping[str, int] | None = field(default=None, compare=False)
    #: The model (list entry ID) the turn's last model call ran on (the
    #: journal's ``model``); ``None`` when no model was called or the client doesn't say.
    model: str | None = field(default=None, compare=False)


AgentEvent = TextDelta | TextSuppressed | TextBeforeTool | ToolCall | ToolStarted | MessageEnd | TurnEnd


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A result for a pending inline call. ``content`` is a JSON object or a text string."""

    tool_use_id: str
    content: dict[str, Any] | str
    status: Literal["success", "error"] = "success"

    def to_block(self) -> dict[str, Any]:
        # Always JSON *text*: the model reads the JSON from it, and the
        # history keeps one form whatever the content's type.
        text = self.content if isinstance(self.content, str) else json.dumps(self.content, separators=(",", ":"), sort_keys=True)
        body = [{"text": text}]
        return {"toolResult": {"toolUseId": self.tool_use_id, "content": body, "status": self.status}}


class AgentError(RuntimeError):
    """The agent call failed (before or during the stream). ``retryable``."""

    def __init__(self, message: str, *, retryable: bool = True, code: str | None = None,
                 service_message: str | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code
        # AWS's own diagnostic text for a service-side error code (see SERVICE_DIAGNOSTIC_CODES), already
        # truncated. The logger scrubs it again. Never set for anything that could echo what a caller said.
        self.service_message = service_message


#: Stop reasons that mean the turn did not finish normally.
ERROR_STOP_REASONS: frozenset[str] = frozenset({
    "content_filtered", "malformed_model_output", "malformed_tool_use",
    "model_context_window_exceeded", "max_iterations_exceeded", "timeout_exceeded",
    "hook_stopped",
})

_NOT_RETRYABLE = frozenset({"ValidationException", "AccessDeniedException",
                            "ResourceNotFoundException", "validationException"})


# --------------------------------------------------------------------------- protocol

#: The bridge's session-ownership decision at each ConversationRelay session:
#: ``new`` (the call's first AI session), ``cached`` (this process ran the previous one),
#: ``reload`` (another process did, or this one restarted).
HistoryMode = Literal["new", "cached", "reload"]


@runtime_checkable
class AgentClient(Protocol):
    """A turn of the agent loop for one call, and the history writes around it."""

    def turn(
        self,
        session_id: str,
        *,
        user_text: str | None,
        tool_results: Sequence[ToolResult],
        system_prompts: Mapping[str, str] | None,
        system_suffix: str | None = None,
        turn_id: int | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Send the caller's words, results for pending inline calls, or both (the results
        are recorded first, then the model answers the words in one model call), and stream the reply.

        ``system_prompts`` is the stable system prompt per model key
        (:func:`owlbank.agent_prompt.model_key` of each entry's ID, in list order); every model call
        carries its own entry's prompt (:func:`prompt_for`), at the turn's start and on a reissue.

        ``system_suffix`` is the text after the cache point: the per-call part and the clock line
        Kept apart from the stable prompt so a client can cache the prompt and put the
        suffix after the cache point. The
        model sees ``prompt + "\n" + suffix``.

        ``turn_id`` names the bridge turn this call belongs to (the bridge's turn number in the
        session), the same for every hop of the turn and its retries: a turn
        that moved to another entry keeps it for its later hops; a new ``turn_id`` chooses afresh. ``None``:
        no turn is tracked, each call chooses afresh. ``record_results`` takes it
        too (the same meaning).

        An async generator (not a bare iterator): the bridge closes it with ``aclose()`` on barge-in."""
        ...

    def record_results(
        self,
        session_id: str,
        tool_results: Sequence[ToolResult],
        *,
        system_prompts: Mapping[str, str] | None,
        system_suffix: str | None = None,
        turn_id: int | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Add results the caller has already heard (the front door's envelope with
        ``next`` = ``await_caller``) to the call's history, **without** asking the model to answer
        them, and speak nothing for them.

        The stream is just :class:`TurnEnd`, with no model call."""
        ...

    async def record_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Add ``user_text`` then ``assistant_text`` to the call's history as
        one user message and one assistant reply, **without** a model call, and persist them as
        any other turn's messages (the session's welcome, spoken by ConversationRelay, as the
        agent's own turn; the user message is the catalog's unspoken ``agent.call_started``).

        The history must stay valid for the model: never called while a tool use waits for its
        result (the bridge puts the welcome in that result's envelope instead). A client that
        finds its history ending with an unanswered user message adds only the reply, so the
        roles still alternate. Blank text adds nothing."""
        ...

    async def begin_session(self, session_id: str, *, history: HistoryMode = "reload") -> None:
        """The bridge calls it once per ``setup``, in the background, after recording the
        session's owner in call state, with its session-ownership decision; the session's first
        turn awaits it. The client keeps the call's history and loads or reuses it per ``history``."""
        ...


def prompt_for(system_prompts: Mapping[str, str] | None, model_ids: Sequence[str], index: int) -> str | None:
    """The system prompt for entry ``index`` of ``model_ids``: the map's
    entry for its model key, else the preferred (first) entry's (the bridge fills every
    entry, logging ``model_prompt_missing`` where it fell back, so this only serves a caller that
    renders one prompt, such as the bench). ``None`` for no map; a map with neither key is a
    caller's error (:class:`AgentError` ``system_prompt_missing``, before any model call)."""
    if system_prompts is None:
        return None
    for key in (model_key(model_ids[index]), model_key(model_ids[0])):
        if key in system_prompts:
            return system_prompts[key]
    raise AgentError("no system prompt for the turn's model", retryable=False, code="system_prompt_missing")


def join_system(system_prompt: str | None, system_suffix: str | None) -> str | None:
    """The system text the model sees: ``prompt + "\n" + suffix`` (byte-identical to the
    bridge's joined clock prompt). No suffix: the prompt unchanged."""
    if system_suffix is None:
        return system_prompt
    return f"{system_prompt or ''}\n{system_suffix}"


#: Error codes whose AWS message is an infrastructure diagnostic (IAM denial, a model or tool-schema
#: validation failure, throttling, a service-side runtime error) and is logged, scrubbed and truncated.
#: Added after the first real calls: with the message omitted, "runtimeClientError" and "AccessDenied"
#: could only be diagnosed by replaying calls by hand. Every other error keeps the message out of logs.
SERVICE_DIAGNOSTIC_CODES = frozenset({
    "runtimeClientError", "AccessDeniedException", "ValidationException", "ThrottlingException",
    "ServiceQuotaExceededException", "ResourceNotFoundException", "ModelErrorException",
    "ModelTimeoutException", "InternalServerException", "ServiceUnavailableException",
})
SERVICE_MESSAGE_MAX = 400


def agent_error_from_aws(exc: BaseException, *, operation: str) -> AgentError:
    """Map an AWS (botocore) or other exception to :class:`AgentError`: retryable unless
    its error code is in ``_NOT_RETRYABLE``; a ``SERVICE_DIAGNOSTIC_CODES`` message is kept, truncated."""
    if isinstance(exc, AgentError):
        return exc
    code = None
    detail = None
    response = getattr(exc, "response", None)
    if isinstance(response, Mapping):
        error = response.get("Error") or {}
        code = error.get("Code")
        if code in SERVICE_DIAGNOSTIC_CODES and isinstance(error.get("Message"), str):
            detail = error["Message"][:SERVICE_MESSAGE_MAX]
    return AgentError(f"{operation} failed: {type(exc).__name__}", code=code,
                      retryable=code not in _NOT_RETRYABLE, service_message=detail)
