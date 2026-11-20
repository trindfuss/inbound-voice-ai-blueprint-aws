"""Model failover: a system prompt per model entry, through the bridge.

The bridge (the component rig, with the simulator) runs a real :class:`StrandsAgentClient` over fake
models, one per entry, that record the system blocks each call receives, with the real guarded
models and their first-token deadlines. The catalog gives every model key a distinct text, so a call
that carries another entry's prompt is seen.

* **Property** (Hypothesis): over random model lists (one to three of the known models, in any order),
  local entry states (a turn may start on a degraded fallback) and per-call behaviours (a missed
  deadline or a hard error before any output moves the call to the next entry: a reissue mid-turn),
  across several caller turns, every model call carries the prompt rendered for its own entry.
  Calls may also be an inline tool use (the bridge runs the tool and calls the agent again: a later
  hop of the turn), and within each caller turn the entries called form one run each: once
  the turn has left an entry it never goes back to it, and it moves only after a call that failed
  before any output (so a hop after a tool result stays on the entry the turn moved to).
* **Missing prompt:** an entry whose key is missing from the catalog gets the preferred entry's prompt, and
  ``model_prompt_missing`` is logged once per process per key and language; the preferred key
  missing is the existing catalog error (no model call), not a fallback.
* **A list of one** sends exactly today's prompt.
* **Agent prompt caching:** the bridge hands the client the static part per model
  key and ``per_call + "\n" + clock`` as the suffix, for an unverified caller, a verified one and
  one with web activity; mid-call verification changes only the suffix; a failover switch gets that
  key's static part and the same suffix; the fallback prompt follows the split; a token in the
  per-call part is refused; ``system_prompt_rendered`` carries sizes and a hash, no text.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, event, example, given, settings
from hypothesis import strategies as st

from owlbank import agent_prompt, untrusted
from owlbank.callstate import WELCOME_OFFER_MADE
from owlbank.catalog import PromptCatalog
from owlbank.failover import UNAVAILABLE, FailoverState, Outcome, Settings
from tac_bridge import bridge as bridge_module
from tac_bridge.agent_client import AgentError, prompt_for
from tac_bridge.agent_client_strands import ModelEntry, StrandsAgentClient, StrandsSettings
from tac_bridge.bridge import BridgeConfig, CallView
from tests.component.test_bridge_fakes import (
    CALL_SID,
    VERIFIED_PROFILE,
    FakeMemory,
    Rig,
    clock_line_at,
    end_frame,
    front_door_update,
    load_json,
    make_rig,
    spoken,
    write_call_state,
)
from tests.component.test_web_activity_bridge import activity
from tests.logutil import fields, lines
from tests.strands_fakes import Call, FakeModel, client_error, text_msg, tool_msg

IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5-5")
KEYS = {m: agent_prompt.model_key(m) for m in IDS}
DEADLINE_S = 0.03
BRIDGE_LOG = "tac_bridge.bridge"
#: Per-call behaviours: ``text`` answers; ``hard`` is a throttle and ``hang`` a stalled first token,
#: both before any output, so the call moves to the next entry; ``tool`` is an inline tool use
#: (output: the bridge runs it and the turn's next hop calls the agent again).
BEHAVIOURS = ("text", "hard", "hang", "tool")
NO_OUTPUT = frozenset({"hard", "hang"})
SHAPES = {IDS[0]: {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000},
          IDS[1]: {"thinking": "off", "text_release": "message", "max_tokens": 400},
          IDS[2]: {"thinking": "adaptive", "text_release": "stream", "max_tokens": 2000}}


def marker(key: str, lang: str) -> str:
    return f"PROMPT-OF-{key}-{lang}"


def catalog_doc(*, drop: Sequence[str] = ()) -> dict[str, Any]:
    """The seed catalog, each model's text opened by its own marker; ``drop`` removes model keys."""
    doc = copy.deepcopy(load_json("prompt_catalog.json"))
    entries = []
    for e in doc["entries"]:
        key = e["key"]
        if key.startswith(agent_prompt.SYSTEM_KEY_PREFIX + "."):
            model = key.removeprefix(agent_prompt.SYSTEM_KEY_PREFIX + ".")
            if model in KEYS.values():
                if model in drop:
                    continue
                e = {**e, "text": f"{marker(model, e['lang'])} {e['text']}"}
        entries.append(e)
    doc["entries"] = entries
    return doc


class PromptModel(FakeModel):
    """An entry's fake model: records each call's system blocks and plays ``behaviours`` in order
    (then answers). ``hang`` sleeps past any deadline (the guard abandons the call)."""

    def __init__(self, model_id: str, behaviours: Sequence[str] = (),
                 log: list[tuple[str, str]] | None = None) -> None:
        super().__init__()
        self.model_id = model_id
        self.behaviours = list(behaviours)
        self.log = log if log is not None else []

    async def stream(self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, *,
                     tool_choice: Any = None, system_prompt_content: Any = None, **kwargs: Any) -> Any:
        self.calls.append(Call(copy.deepcopy(list(messages)), [], copy.deepcopy(system_prompt_content)))
        b = self.behaviours.pop(0) if self.behaviours else "text"
        self.log.append((self.model_id, b))
        if b == "hang":
            await asyncio.sleep(30)
        if b == "hard":
            raise client_error("ThrottlingException")
        chunks = (tool_msg(("list_recent_transactions", {"account": "checking"})) if b == "tool"
                  else text_msg(f"Answer from {KEYS[self.model_id]}."))
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    def system_texts(self) -> list[str]:
        return [json.dumps(c.system, ensure_ascii=False) for c in self.calls]


def failover_state(ids: Sequence[str], marks: Sequence[str]) -> FailoverState:
    base = Settings.for_models(list(ids))
    s = replace(base, entries=tuple(replace(e, deadline_s=None if i == len(ids) - 1 else DEADLINE_S)
                                    for i, e in enumerate(base.entries)))
    state = FailoverState(s)
    for i, mark in enumerate(marks):
        if mark == "unavailable":
            state.record(i, UNAVAILABLE)
        elif mark == "degraded":
            for _ in range(s.miss_run):
                state.record(i, Outcome.ABANDONED)
    return state


def rig_for(ids: Sequence[str], fakes: dict[str, PromptModel], *, marks: Sequence[str] | None = None,
            doc: dict[str, Any] | None = None, config_ids: Sequence[str] | None = None,
            language: str = "en-US") -> Rig:
    """The component rig with a real client over ``fakes`` (one per entry of ``ids``) and the bridge
    told the same list (``config_ids`` overrides it, ``()`` being a bare config)."""
    rig = make_rig(config=BridgeConfig(model_ids=tuple(ids if config_ids is None else config_ids)))
    rig.bridge._catalog = PromptCatalog.from_dict(doc or catalog_doc())  # noqa: SLF001
    client = StrandsAgentClient(
        StrandsSettings(models=tuple(ModelEntry(id=m, **SHAPES[m]) for m in ids),  # type: ignore[arg-type]
                        region="us-east-1"),
        model_factory=lambda one, _sid: fakes[one.primary.id],
        failover_state=failover_state(ids, marks or ["healthy"] * len(ids)))
    rig.bridge._agent = client  # noqa: SLF001 - the real client, over the fake models
    write_call_state(rig.store, language=language)
    return rig


def converse(rig: Rig, prompts: Sequence[str], log: list[Any] | None = None) -> list[str]:
    """What the caller heard per prompt; ``log`` (the fakes' shared call log) gets a ``None`` after
    each turn."""
    async def go() -> list[str]:
        sim = rig.sim()
        await sim.connect()
        said = []
        for p in prompts:
            said.append(spoken(await sim.prompt(p, timeout=5.0)))
            if log is not None:
                log.append(None)
        await sim.hangup()
        return said

    return asyncio.run(go())


def per_turn(log: list[Any]) -> list[list[tuple[str, str]]]:
    turns: list[list[tuple[str, str]]] = [[]]
    for item in log:
        if item is None:
            turns.append([])
        else:
            turns[-1].append(item)
    return turns[:-1]


@pytest.fixture(autouse=True)
def _fresh_missing_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """``model_prompt_missing`` is once per process: each test starts with none logged."""
    monkeypatch.setattr(bridge_module, "_PROMPT_MISSING_LOGGED", set())




@st.composite
def cases(draw: st.DrawFn) -> dict[str, Any]:
    ids = list(draw(st.permutations(IDS)))[:draw(st.integers(1, 3))]
    n = len(ids)
    # The list's last entry never fails and is never unavailable, so every turn ends in an answer
    # (it is always in the rank); the others may fail before output in any order. Tool uses are
    # few enough that no turn reaches the bridge's hop limit.
    scripts = [draw(st.lists(st.sampled_from(BEHAVIOURS), max_size=4)) if i < n - 1
               else draw(st.lists(st.sampled_from(["text", "tool"]), max_size=1)) for i in range(n)]
    assume(sum(b == "tool" for s in scripts for b in s) <= 4)
    marks = [draw(st.sampled_from(["healthy", "degraded", "unavailable"] if i < n - 1 else ["healthy", "degraded"]))
             for i in range(n)]
    turns = draw(st.integers(1, 3))
    return {"ids": ids, "scripts": scripts, "marks": marks, "turns": turns,
            "lang": draw(st.sampled_from(["en-US", "es-US"]))}


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=cases())
# The shape of the live call: a miss on the preferred entry, a tool use on the next, then the hop
# after the tool result; and a second turn that chooses afresh.
@example(case={"ids": list(IDS[:2]), "scripts": [["hang", "text"], ["tool"]], "marks": ["healthy", "healthy"],
               "turns": 2, "lang": "en-US"})
def test_every_model_call_carries_its_own_entrys_prompt(case: dict[str, Any]) -> None:
    ids, lang = case["ids"], case["lang"]
    log: list[Any] = []
    fakes = {m: PromptModel(m, case["scripts"][i], log) for i, m in enumerate(ids)}
    rig = rig_for(ids, fakes, marks=case["marks"], language=lang)
    bridge_module._PROMPT_MISSING_LOGGED.clear()  # noqa: SLF001
    said = converse(rig, [f"question {k}" for k in range(case["turns"])], log)

    assert all("Answer from " in s for s in said), said  # every turn answered
    calls = {m: len(f.calls) for m, f in fakes.items()}
    event(f"entries called: {sum(1 for c in calls.values() if c)} of {len(ids)}")
    event("first call on a fallback" if case["marks"][0] != "healthy" and len(ids) > 1 else "first call preferred")
    if any(len(f.calls) > 1 for f in fakes.values()) or sum(1 for c in calls.values() if c) > 1:
        event("several calls or entries")
    for m, fake in fakes.items():
        own = marker(KEYS[m], lang)
        others = [marker(KEYS[o], lang) for o in ids if o != m]
        for system in fake.system_texts():
            assert system.count(own) == 1, (m, system[:200])
            assert not any(o in system for o in others), (m, system[:200])
    # within a caller turn (all its hops) each entry's calls are one run, never returned to
    # once left, and a move follows only a call that failed before any output.
    for calls_of_turn in per_turn(log):
        if any(b == "tool" for _, b in calls_of_turn):
            event("a turn with a tool hop")
        if any(ba in NO_OUTPUT and bb == "tool" for (_, ba), (_, bb) in zip(calls_of_turn, calls_of_turn[1:],
                                                                           strict=False)):
            event("a tool hop after a move")
        entries = [m for m, _ in calls_of_turn]
        runs = [e for k, e in enumerate(entries) if k == 0 or entries[k - 1] != e]
        assert len(runs) == len(set(runs)), calls_of_turn
        for (a, ba), (b, _) in zip(calls_of_turn, calls_of_turn[1:], strict=False):
            if a != b:
                assert ba in NO_OUTPUT, calls_of_turn




def test_a_missing_key_gets_the_preferred_prompt_and_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=BRIDGE_LOG)
    ids = IDS[:2]
    doc = catalog_doc(drop=[KEYS[IDS[1]]])
    # The preferred entry throttles on every call: each turn is reissued to Haiku 4.5, whose key is
    # missing (a bridge deployed before the reseed).
    fakes = {IDS[0]: PromptModel(IDS[0], ["hard"] * 6), IDS[1]: PromptModel(IDS[1])}
    said = converse(rig_for(ids, fakes, doc=doc), ["first", "second"])
    assert said == ["Answer from haiku-4-5."] * 2
    assert len(fakes[IDS[1]].calls) == 2
    for system in fakes[IDS[1]].system_texts():
        assert marker("haiku-5-5", "en-US") in system and marker("haiku-4-5", "en-US") not in system
    # Another call in the same process (a new rig: the cache is per call): no second line.
    fakes = {IDS[0]: PromptModel(IDS[0], ["hard"]), IDS[1]: PromptModel(IDS[1])}
    converse(rig_for(ids, fakes, doc=doc), ["third"])
    # Another language: its own line, once.
    fakes = {IDS[0]: PromptModel(IDS[0]), IDS[1]: PromptModel(IDS[1])}
    converse(rig_for(ids, fakes, doc=doc, language="es-US"), ["cuarto"])
    missing = [e for e in lines(caplog.records, BRIDGE_LOG) if e["event"] == "model_prompt_missing"]
    assert [(e["model_key"], e["language"], e["key"]) for e in missing] == [
        ("haiku-4-5", "en-US", "agent.system.haiku-4-5"), ("haiku-4-5", "es-US", "agent.system.haiku-4-5")]
    assert all(r.levelno == logging.WARNING for r in caplog.records if getattr(r, "event", "") == "model_prompt_missing")


def test_a_missing_preferred_key_is_the_catalog_error_not_a_fallback(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=BRIDGE_LOG)
    fakes = {m: PromptModel(m) for m in IDS[:2]}
    rig = rig_for(IDS[:2], fakes, doc=catalog_doc(drop=["haiku-5-5"]))

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt_nowait("hello")
        await sim.idle()
        await sim.hangup()

    asyncio.run(go())
    assert all(not f.calls for f in fakes.values())  # no model got another model's prompt
    events = [e["event"] for e in lines(caplog.records, BRIDGE_LOG)]
    assert "model_prompt_missing" not in events


# ------------------------------------------------------------------------------------- a list of one


def test_a_list_of_one_sends_todays_prompt() -> None:
    """One entry: one key, rendered once; the model sees the prompt a bare config renders (today's)."""
    one = (IDS[0],)
    systems = []
    for config_ids in (one, ()):
        fakes = {IDS[0]: PromptModel(IDS[0])}
        rig = rig_for(one, fakes, config_ids=config_ids, doc=load_json("prompt_catalog.json"))
        assert converse(rig, ["hello", "and again"]) == ["Answer from haiku-5-5."] * 2
        meta = rig.bridge._model_keys()  # noqa: SLF001
        assert meta == ("haiku-5-5",)
        systems.append(fakes[IDS[0]].calls)
    assert [c.system for c in systems[0]] == [c.system for c in systems[1]]
    catalog = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
    first_block = systems[0][0].system[0]["text"]
    assert first_block.startswith(catalog.text("agent.system.haiku-5-5", "en-US", firstName="Thomas",
                                               profile="", memory="")[:60])


def test_the_bridge_renders_each_distinct_key_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The data fetch is once per cache miss; the static part is rendered once per distinct
    key (``us.`` and ``global.`` profiles of one model share one), the per-call part once for all."""
    statics: list[str] = []
    per_calls: list[str] = []
    real_static, real_per_call = agent_prompt.render_static, agent_prompt.render_per_call

    def counting_static(catalog: Any, model_key: str, *a: Any, **k: Any) -> str:
        statics.append(model_key)
        return real_static(catalog, model_key, *a, **k)

    def counting_per_call(catalog: Any, lang: str, **k: Any) -> str:
        per_calls.append(lang)
        return real_per_call(catalog, lang, **k)

    monkeypatch.setattr(agent_prompt, "render_static", counting_static)
    monkeypatch.setattr(agent_prompt, "render_per_call", counting_per_call)
    ids = (IDS[0], "global.anthropic.claude-haiku-5-5", IDS[2])
    rig = make_rig(config=BridgeConfig(model_ids=ids))
    write_call_state(rig.store)
    converse(rig, ["hello", "again"])
    assert statics == ["haiku-5-5", "sonnet-5-5"]  # the second turn uses the call's cache
    assert per_calls == ["en-US"]
    (first, second) = rig.agent.calls
    assert list(first["system_prompts"]) == ["haiku-5-5", "sonnet-5-5"]
    assert first["system_prompts"] == second["system_prompts"]


# ------------------------------------------------------------------------------------- prompt_for


def test_prompt_for_reads_the_entrys_key_then_the_preferred_one() -> None:
    ids = (IDS[0], "global.anthropic.claude-haiku-4-5-20251001-v1:0", "test-model")
    prompts = {"haiku-5-5": "A", "haiku-4-5": "B"}
    assert [prompt_for(prompts, ids, i) for i in range(3)] == ["A", "B", "A"]
    assert prompt_for(None, ids, 1) is None
    with pytest.raises(AgentError) as err:
        prompt_for({"sonnet-5-5": "C"}, ids, 1)
    assert err.value.code == "system_prompt_missing" and not err.value.retryable


# ------------------------------------------------------------------ agent prompt caching

SEED_CATALOG = PromptCatalog.from_dict(load_json("prompt_catalog.json"))
ALL_KEYS = tuple(KEYS[m] for m in IDS)
RENDERED_FIELDS = {"model_key", "language", "static_sha", "static_chars", "per_call_chars", "verified", "web_block"}


def statics_for(catalog: PromptCatalog, lang: str) -> dict[str, str]:
    return {k: agent_prompt.render_static(catalog, k, lang) for k in ALL_KEYS}


def split_suffix(rec: dict[str, Any]) -> tuple[str, str]:
    """``(per_call, clock)`` of the suffix the bridge handed the client."""
    per_call, clock = rec["system_suffix"].rsplit("\n", 1)
    return per_call, clock


def call_rig(*, verified: bool = True, lang: str = "en-US", memory: Any = "fake",
             groups: dict[str, Any] | None = None, welcome_offer: str | None = None) -> Rig:
    rig = make_rig(config=BridgeConfig(model_ids=IDS), memory=memory)
    if groups is not None:
        rig.tac.conversation_memory_client.trait_groups[VERIFIED_PROFILE] = groups
    write_call_state(rig.store, verified=verified, language=lang, welcome_offer=welcome_offer)
    return rig


def verified_per_call(rig: Rig, lang: str) -> str:
    """The per-call part the renderer gives the fixture's verified caller with the fake recall."""
    customer = rig.store.get("customers", {"customer_id": "C-1001"})
    view = CallView(call_sid=CALL_SID, language=lang, level=1, verified_customer_id="C-1001")
    facts = agent_prompt.CallerFacts(customer["first_name"], bridge_module._profile_text(customer, view),  # noqa: SLF001
                                     untrusted.wrap("memory", FakeMemory().prompts[VERIFIED_PROFILE]))
    return agent_prompt.render_per_call(SEED_CATALOG, lang, caller=facts, web_block=None)


@pytest.mark.parametrize("lang", ["en-US", "es-US"])
@pytest.mark.parametrize("verified", [False, True], ids=["unverified", "verified"])
def test_the_client_gets_the_static_map_and_the_per_call_suffix(lang: str, verified: bool) -> None:
    rig = call_rig(verified=verified, lang=lang)
    converse(rig, ["Hi"])
    (rec,) = rig.agent.calls
    assert rec["system_prompts"] == statics_for(SEED_CATALOG, lang)
    assert "Thomas" not in json.dumps(rec["system_prompts"])  # no caller data before the cache point
    per_call, clock = split_suffix(rec)
    assert clock == clock_line_at(rig.bridge, lang)
    expected = verified_per_call(rig, lang) if verified else agent_prompt.render_per_call(
        SEED_CATALOG, lang, caller=None, web_block=None)
    assert per_call == expected
    assert ("Thomas" in per_call) is verified


@pytest.mark.parametrize("offered", [False, True], ids=["context", "offer_made"])
@pytest.mark.parametrize("lang", ["en-US", "es-US"])
def test_web_activity_goes_in_the_per_call_part_only(lang: str, offered: bool) -> None:
    rig = call_rig(lang=lang, memory="tac", groups=activity(),
                   welcome_offer=WELCOME_OFFER_MADE if offered else None)
    converse(rig, ["Hi"])
    (rec,) = rig.agent.calls
    assert rec["system_prompts"] == statics_for(SEED_CATALOG, lang)
    assert "HELOC" not in json.dumps(rec["system_prompts"])
    per_call, _clock = split_suffix(rec)
    block = SEED_CATALOG.text(agent_prompt.WEB_ACTIVITY_KEY if offered else agent_prompt.WEB_ACTIVITY_CONTEXT_KEY,
                              lang)
    assert per_call.startswith(block + "\n<caller_context>\n")
    assert "HELOC" in per_call and "Thomas" in per_call


def test_mid_call_verification_changes_only_the_suffix() -> None:
    rig = call_rig(verified=False)

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("Hi")
        front_door_update(rig.store, verified_customer_id="C-1001", level=1, factors_verified=("ani_match", "zip"))
        await sim.prompt("And now?")
        await sim.hangup()

    asyncio.run(go())
    first, second = rig.agent.calls
    assert first["system_prompts"] == second["system_prompts"] == statics_for(SEED_CATALOG, "en-US")
    assert split_suffix(first)[0] == agent_prompt.render_per_call(SEED_CATALOG, "en-US", caller=None, web_block=None)
    assert split_suffix(second)[0] == verified_per_call(rig, "en-US")


def test_a_failover_switch_gets_that_keys_static_part_and_the_same_suffix() -> None:
    doc = catalog_doc()
    fakes = {IDS[0]: PromptModel(IDS[0], ["hard"]), IDS[1]: PromptModel(IDS[1])}
    assert converse(rig_for(IDS[:2], fakes, doc=doc), ["hello"]) == ["Answer from haiku-4-5."]
    (first,), (second,) = fakes[IDS[0]].calls, fakes[IDS[1]].calls
    catalog = PromptCatalog.from_dict(doc)
    assert first.system is not None and second.system is not None
    assert first.system[0] == {"text": agent_prompt.render_static(catalog, "haiku-5-5", "en-US")}
    assert second.system[0] == {"text": agent_prompt.render_static(catalog, "haiku-4-5", "en-US")}
    assert first.system[-1] == second.system[-1]  # the per-call part and clock, unchanged by the move
    assert "Thomas" in first.system[-1]["text"] and "Thomas" not in json.dumps(first.system[:-1])


@pytest.mark.parametrize("lang", ["en-US", "es-US"])
def test_the_fallback_prompt_follows_the_split(lang: str, monkeypatch: pytest.MonkeyPatch,
                                               caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=BRIDGE_LOG)
    rig = call_rig(lang=lang)

    async def failing(*_a: Any) -> Any:
        raise RuntimeError("profile read failed")

    monkeypatch.setattr(rig.bridge, "_system_prompt", failing)
    converse(rig, ["Hi"])
    (rec,) = rig.agent.calls
    assert rec["system_prompts"] == statics_for(SEED_CATALOG, lang)
    per_call, clock = split_suffix(rec)
    assert per_call == agent_prompt.FALLBACK_CONTEXT[lang]  # no verification claim, never a catalog frame
    assert per_call == agent_prompt.render_per_call(SEED_CATALOG, lang, caller=None, web_block=None, fallback=True)
    assert clock == clock_line_at(rig.bridge, lang)
    assert "Thomas" not in rec["system_suffix"]
    rendered = [e for e in lines(caplog.records, BRIDGE_LOG) if e["event"] == "system_prompt_rendered"]
    assert [e["model_key"] for e in rendered] == list(ALL_KEYS)
    assert all(e["per_call_chars"] == len(per_call) and e["web_block"] is None for e in rendered)


def test_a_token_in_the_per_call_part_is_refused() -> None:
    """The sealed token in a caller field reaches only the per-call part (the static part has no
    caller data): the suffix is checked too, so nothing is sent and the call is handed off."""
    rig = call_rig()
    sealed = rig.call()["sealed_token"]
    customer = rig.store.get("customers", {"customer_id": "C-1001"})
    rig.store.put("customers", {**customer, "first_name": f"Tom{sealed}"})

    async def go() -> list[dict[str, Any]]:
        sim = rig.sim()
        await sim.connect()
        frames = await sim.prompt("Hi")
        await sim.hangup()
        return frames

    frames = asyncio.run(go())
    assert rig.agent.calls == []
    end = end_frame(frames)
    assert end is not None and end["data"]["reason"] == "escalate"


def test_system_prompt_rendered_has_sizes_and_a_hash_and_no_text(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=BRIDGE_LOG)
    rig = call_rig(memory="tac", groups=activity())
    converse(rig, ["Hi", "again"])
    first, _second = rig.agent.calls
    per_call, _clock = split_suffix(first)
    records = [r for r in caplog.records if r.name == BRIDGE_LOG and getattr(r, "event", "") == "system_prompt_rendered"]
    events = [fields(r) for r in records]
    assert [e["model_key"] for e in events] == list(ALL_KEYS)  # once per rendered prompt; turn 2 is cached
    for e in events:
        # (``call_sid`` and ``component`` are the JSON handler's context, once a test installed it)
        assert set(e) - {"event", "call_sid", "component"} == RENDERED_FIELDS
        static = first["system_prompts"][e["model_key"]]
        assert e["static_sha"] == hashlib.sha256(static.encode("utf-8")).hexdigest()[:16]
        assert (e["static_chars"], e["per_call_chars"]) == (len(static), len(per_call))
        assert e["language"] == "en-US" and e["verified"] is True
        assert e["web_block"] == agent_prompt.WEB_ACTIVITY_CONTEXT_KEY
    assert all(r.levelno == logging.INFO for r in records)
    logged = json.dumps([[r.getMessage(), fields(r)] for r in records], ensure_ascii=False, default=str)
    customer = rig.store.get("customers", {"customer_id": "C-1001"})
    for value in ("Thomas", customer["last_name"], "HELOC", "<caller_context>", "OwlBank", "Call language"):
        assert value not in logged
