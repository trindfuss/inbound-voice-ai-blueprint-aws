"""Scheduled, pooled, templated tool phrases in the bridge, on both agent backends (parity rig).

Time is real and short (tens to hundreds of milliseconds), as in ``test_bridge_masking``: the
agent fake's ``delay_s`` and the recording backend's per-tool ``delay_s`` set when things arrive,
and each yielded chunk is time-stamped.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from owlbank import callstate as cs
from owlbank.catalog import PromptCatalog
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import CALL_SID, NOW, load_json, run, say, tool, write_call_state
from tests.component.test_bridge_masking import (
    BAL_PHRASE,
    BALANCE,
    BRIDGE_LOG,
    GENERIC,
    TX_PHRASE,
    TXNS,
    RecordingBackend,
    collect,
    context,
    direct,
    mask_lines,
)
from tests.logutil import lines

CATALOG_DOC = load_json("prompt_catalog.json")
OTHER_SID = "CA" + "e" * 32


def skip_lines(caplog: pytest.LogCaptureFixture) -> list[dict[str, Any]]:
    return [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "mask_skipped"]


def policy(*, tool_masks: dict[str, dict[str, Any] | None] | None = None, **masking: Any) -> dict[str, Any]:
    """``data/policy.json`` with ``masking`` overrides and whole ``mask`` objects per tool."""
    doc = masked_policy_doc(**masking)
    for name, mask in (tool_masks or {}).items():
        if mask is None:
            doc["tools"][name].pop("mask", None)
        else:
            doc["tools"][name]["mask"] = mask
    return doc


def catalog(*, drop: tuple[str, ...] = (), texts: dict[tuple[str, str], str] | None = None) -> dict[str, Any]:
    """``data/prompt_catalog.json`` without the ``drop`` keys, with ``texts[(key, lang)]`` set
    (added if missing)."""
    doc = copy.deepcopy(CATALOG_DOC)
    doc["entries"] = [e for e in doc["entries"] if e["key"] not in drop]
    for (key, lang), text in (texts or {}).items():
        hit = [e for e in doc["entries"] if e["key"] == key and e["lang"] == lang]
        if hit:
            hit[0]["text"] = text
        else:
            doc["entries"].append({"key": key, "lang": lang, "text": text, "version": 1,
                                   "status": "published", "audioUrl": None})
    return doc


def tx_backend(delay: float = 0.0) -> RecordingBackend:
    return RecordingBackend({"list_recent_transactions": lambda **_: dict(TXNS)},
                            delay_s={"list_recent_transactions": delay})


def balance_backend() -> RecordingBackend:
    return RecordingBackend({"get_balance": lambda **_: dict(BALANCE)})


def word(key: str, lang: str = "en-US") -> str:
    return PromptCatalog.from_dict(CATALOG_DOC).text(key, lang).strip()


# --------------------------------------------------------------------------- the delay


def test_fast_tool_answer_before_the_deadline_gets_no_tool_phrase(caplog: pytest.LogCaptureFixture) -> None:
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")],
               backend=tx_backend(), policy_doc=policy(after_ms=0, tool_after_ms=400))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "transactions"))
    assert chunks == ["Nothing recent."]
    assert not mask_lines(caplog)
    (skip,) = skip_lines(caplog)
    assert (skip["tool"], skip["key"], skip["reason"]) == (
        "list_recent_transactions", "mask.list_recent_transactions", "answer_started")
    assert (skip["after_ms"], skip["source"]) == (400, "default")
    assert not [e for e in d.events() if e["type"] == "prompt"]
    # the tool itself ran at once: only the phrase waits
    assert d.backend.calls[0]["op"] == "list_recent_transactions"


def test_slow_tool_gets_its_phrase_after_the_delay(caplog: pytest.LogCaptureFixture) -> None:
    backend = tx_backend(delay=0.5)
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")],
               backend=backend, policy_doc=policy(after_ms=0, tool_after_ms=150))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, times = run(collect(d, "transactions"))
    assert chunks == [TX_PHRASE + " ", "Nothing recent."]
    waited = times[TX_PHRASE] - backend.started["list_recent_transactions:1"]
    assert 0.15 - 0.01 <= waited < 0.5  # after the delay, while the tool still ran
    (ln,) = mask_lines(caplog)
    assert (ln["key"], ln["tool"], ln["kind"], ln["after_ms"], ln["source"]) == (
        "mask.list_recent_transactions", "list_recent_transactions", "tool", 150, "default")
    assert [e["prompt_key"] for e in d.events() if e["type"] == "prompt"] == ["mask.list_recent_transactions"]
    assert TX_PHRASE not in d.agent.all_inputs_text() and TX_PHRASE not in d.ai_turn_text()


def test_the_tools_own_after_ms_wins_over_the_default(caplog: pytest.LogCaptureFixture) -> None:
    backend = tx_backend(delay=0.4)
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")],
               backend=backend, policy_doc=policy(after_ms=0, tool_after_ms=5000, tool_masks={
                   "list_recent_transactions": {"phrase_key": "mask.list_recent_transactions", "after_ms": 100}}))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "transactions"))
    assert chunks == [TX_PHRASE + " ", "Nothing recent."]
    (ln,) = mask_lines(caplog)
    assert (ln["after_ms"], ln["source"]) == (100, "tool")


def test_phrase_while_waiting_for_the_model_call_after_the_tool(caplog: pytest.LogCaptureFixture) -> None:
    """"still waiting (for the tool or for the model call after it)": the tool is fast,
    the second model hop is slow, and the phrase falls due while its stream is being read."""
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")],
               backend=tx_backend(), policy_doc=policy(after_ms=0, tool_after_ms=120))
    real_turn = d.agent.turn

    def turn(*a: Any, **kw: Any) -> Any:
        d.agent.delay_s = 0.25 if d.agent.calls else 0.0  # slow from the second hop on
        return real_turn(*a, **kw)

    d.agent.turn = turn  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "transactions"))
    assert chunks == [TX_PHRASE + " ", "Nothing recent."]
    assert d.agent.closed == 2  # neither stream was cancelled by the deadline


# --------------------------------------------------------------------------- after the ack


def test_ack_then_answer_before_the_acks_end_gets_no_tool_phrase(caplog: pytest.LogCaptureFixture) -> None:
    """With tool_after_ms = 0 the phrase would be due at the tool call; anchored to the ack's
    estimated end (0.05 + 0.3 + 11 / 8 s here) it is still pending when the answer comes."""
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")],
               backend=tx_backend(delay=0.3), agent_delay_s=0.06,
               policy_doc=policy(after_ms=50, tool_after_ms=0, chars_per_s=8))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "transactions"))
    assert chunks == [GENERIC + " ", "Nothing recent."]
    (skip,) = skip_lines(caplog)
    assert skip["reason"] == "answer_started"


# --------------------------------------------------------------------------- hand-off


def test_handoff_phrase_at_once_and_held_by_the_tools_own_handoff_wait_ms(
        caplog: pytest.LogCaptureFixture) -> None:
    """The hand-off's phrase ignores after_ms (delay 0), and the hold uses the tool's
    handoff_wait_ms (250 ms), not the policy-wide 5000 ms."""
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=balance_backend(),
               policy_doc=policy(after_ms=0, tool_after_ms=3000, handoff_wait_ms=5000, chars_per_s=8, end_mode="wait",
                                 tool_masks={"get_balance": {"phrase_key": "mask.get_balance",
                                                             "after_ms": 3000, "handoff_wait_ms": 250}}))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, times = run(collect(d, "my checking balance", ctx))
    assert chunks == [BAL_PHRASE + " "]
    held = times["__end__"] - times[BAL_PHRASE]
    assert 0.2 <= held <= 0.25 + 0.1
    (ln,) = mask_lines(caplog)
    assert (ln["after_ms"], ln["source"]) == (0, "handoff")
    (hold,) = [x for x in lines(caplog.records, BRIDGE_LOG) if x["event"] == "handoff_hold"]
    assert hold["wait_ms"] <= 250
    assert ctx.pending_handoff_data is not None and d.call()["prefetched"]["tool_use_id"] == "tu_bal"


# --------------------------------------------------------------------------- pools, templates


def test_two_balance_requests_in_one_call_rotate_and_render_the_account() -> None:
    """The second balance request of the call gets the next phrase of the pool,
    rendered with its own account; another call starts at the pool's first phrase."""
    pol = policy(after_ms=0, handoff_wait_ms=100, chars_per_s=30)  # the prefetch is stored in the hold
    d = direct([tool("get_balance", "tu_b1", account="checking"),
                tool("get_balance", "tu_b2", account="savings"),
                tool("get_balance", "tu_b3", account="savings")],
               backend=RecordingBackend({"get_balance": lambda **_: dict(BALANCE)}), policy_doc=pol)
    write_call_state(d.store, OTHER_SID)
    first, _ = run(collect(d, "checking balance"))
    # The front door read the balance out and reconnected the AI: nothing pending any more.
    cs.mutate(d.store, CALL_SID, lambda _s: {"pending_tool": None, "pending_result": None,
                                             "prefetched": None}, now=lambda: NOW)
    second, _ = run(collect(d, "and savings"))
    other_ctx = context()
    other_ctx.call_sid = OTHER_SID
    other, _ = run(collect(d, "savings balance", other_ctx))
    savings = word("arg.account.savings")
    assert first == [BAL_PHRASE + " "]
    assert second == [PromptCatalog.from_dict(CATALOG_DOC).template("mask.get_balance.1", "en-US").replace("{account}", savings) + " "]
    assert second != first and "{" not in second[0]
    assert other == [BAL_PHRASE + " "]
    keys = [e["prompt_key"] for e in d.events() if e["type"] == "prompt"]
    assert keys == ["mask.get_balance", "mask.get_balance.1"]


@pytest.mark.parametrize(("account", "lang", "want"), [
    ("checking", "en-US", "Getting your checking balance now."),
    ("savings", "en-US", "Getting your savings balance now."),
    ("savings", "es-US", "Obteniendo el saldo de su cuenta de ahorros."),
    ("checking", "es-US", "Obteniendo el saldo de su cuenta de cheques."),
])
def test_account_placeholder_is_rendered_from_the_catalog_words(account: str, lang: str, want: str) -> None:
    d = direct([tool("get_balance", "tu_bal", account=account)], backend=balance_backend(),
               policy_doc=policy(after_ms=0, end_mode="immediate"),
               catalog_doc=catalog(drop=("mask.get_balance",)), language=lang)
    chunks, _ = run(collect(d, "balance"))
    assert chunks == [want + " "]
    (prompt,) = [e for e in d.events() if e["type"] == "prompt"]
    assert (prompt["prompt_key"], prompt["rendered_text"]) == ("mask.get_balance.1", want)


def test_no_renderable_phrase_speaks_nothing_and_logs_an_error(caplog: pytest.LogCaptureFixture) -> None:
    bad = {(f"mask.get_balance.{n}", lang): "Checking your {colour} balance."
           for n in (1, 2) for lang in ("en-US", "es-US")}
    d = direct([tool("get_balance", "tu_bal", account="checking")], backend=balance_backend(),
               policy_doc=policy(after_ms=0), catalog_doc=catalog(drop=("mask.get_balance",), texts=bad))
    ctx = context()
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "balance", ctx))
    assert chunks == []  # never a raw placeholder, never the generic acknowledgement instead
    assert ctx.pending_handoff_data is not None  # the hand-off itself goes on
    assert d.call().get("prefetched") is None  # no phrase, so no prefetch and no hold
    (skip,) = skip_lines(caplog)
    assert (skip["tool"], skip["key"], skip["reason"]) == ("get_balance", "mask.get_balance",
                                                           "no_renderable_phrase")
    errors = [r for r in caplog.records if r.name == BRIDGE_LOG and r.levelno == logging.ERROR]
    assert [getattr(r, "event", None) for r in errors] == ["mask_phrase_unrenderable"]
    assert errors[0].key == "mask.get_balance"  # type: ignore[attr-defined]
    assert not [e for e in d.events() if e["type"] == "prompt"]


def test_tool_off_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    d = direct([tool("list_recent_transactions", "tu_tx", account="checking"), say("Nothing recent.")],
               backend=tx_backend(), policy_doc=policy(after_ms=0, tool_masks={"list_recent_transactions": None}))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "transactions"))
    assert chunks == ["Nothing recent."]
    (skip,) = skip_lines(caplog)
    assert (skip["tool"], skip["key"], skip["reason"]) == ("list_recent_transactions", None, "tool_off")


# --------------------------------------------------------------------------- Gateway tools


FAQ_PHRASE = "Looking that up."
FAQ_CATALOG = {("mask.faq_lookup", "en-US"): FAQ_PHRASE, ("mask.faq_lookup", "es-US"): "Lo estoy buscando."}


def gateway_turn_agent(gateway_s: float) -> Any:
    """One caller turn whose answer needs the Gateway tool ``faq_lookup`` (taking ``gateway_s``): a
    real client whose model calls the tool, which runs inside its loop (the client reports its
    start, ToolStarted), then answers."""
    from tac_bridge.agent_client_strands import StrandsAgentClient
    from tests.strands_fakes import (
        FakeMemoryData,
        FakeModel,
        default_gateway,
        memory_factory,
        settings,
        text_msg,
        tool_msg,
    )

    model = FakeModel([tool_msg(("tg", "level0___faq_lookup", {"q": "fees"})), text_msg("No fee.")])
    return StrandsAgentClient(settings(memory_id="mem-1"), model_factory=lambda _s, _sid: model,
                              gateway_tools=default_gateway(delay=gateway_s),
                              session_manager_factory=memory_factory(FakeMemoryData()))


def test_gateway_tool_is_masked(caplog: pytest.LogCaptureFixture) -> None:
    """The client reports the Gateway tool's start (ToolStarted) and the bridge masks it like an
    inline tool; it is not a tool call for anything else (no tool journal events, not in the
    turn's tools)."""
    pol = policy(after_ms=0, tool_masks={"faq_lookup": {"phrase_key": "mask.faq_lookup", "after_ms": 100}})
    d = direct([], agent=gateway_turn_agent(0.35), policy_doc=pol, catalog_doc=catalog(texts=FAQ_CATALOG))
    with caplog.at_level(logging.INFO, logger=BRIDGE_LOG):
        chunks, _ = run(collect(d, "any fees?"))
    turn_line = [ln for ln in lines(caplog.records, BRIDGE_LOG) if ln["event"] == "turn"][-1]
    assert turn_line["tools"] == []  # never counted as a tool call
    assert not [e for e in d.events() if e["type"] in ("tool_call", "tool_result")]
    assert chunks == [FAQ_PHRASE + " ", "No fee."]
    (ln,) = mask_lines(caplog)
    assert (ln["tool"], ln["key"], ln["after_ms"], ln["source"]) == ("faq_lookup", "mask.faq_lookup",
                                                                     100, "tool")


def test_gateway_tool_start_counts_as_a_tool_call_for_the_acknowledgement() -> None:
    """The Gateway tool's start disarms the acknowledgement (it is "a tool
    call"), so the slow turn hears the tool's phrase only."""
    pol = policy(after_ms=200, tool_masks={"faq_lookup": {"phrase_key": "mask.faq_lookup", "after_ms": 50}})
    d = direct([], agent=gateway_turn_agent(0.4), policy_doc=pol, catalog_doc=catalog(texts=FAQ_CATALOG))
    chunks, _ = run(collect(d, "any fees?"))
    prompts = [e["prompt_key"] for e in d.events() if e["type"] == "prompt"]
    assert chunks == [FAQ_PHRASE + " ", "No fee."]
    assert prompts == ["mask.faq_lookup"]


