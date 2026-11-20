"""Agent prompt caching: the bridge's static / per-call split, request-level.

* Request-level**: random caller data with secret markers (first and last
  name, account nicknames, memory, web activity, welcome offer, verified or not, a missing record),
  through the bridge's own render (:meth:`Bridge._system_prompt`) and the client's
  :func:`system_blocks`, for every model key and language of the seed catalog and both TTLs:
  the system blocks up to and including the first cache point are byte-identical for any two
  callers and hold no marker. The same with stray ``{firstName}``, ``{profile}`` and ``{memory}``
  in every ``agent.system.*`` text.
* What the bridge sends, joined as the client gives it to the model
  (``join_system(static, per_call + "\\n" + clock)``), is :meth:`RenderedPrompt.joined` of the
  renderer for the same caller; the clock is the last line.
* Over random sequences of view changes in a call (verify, step up, welcome offer
  made or cleared, turns), the static map handed to the client is the same on every turn; only the
  suffix changes.
* Every hop of one turn (tool calls, then the answer) sends byte-identical system
  blocks to the model, even with a clock that would read differently every time a line is built.
  Clock once per call (not rebuilt per turn):
  the line is set at the call's first turn and, the bank-local day and language unchanged, every
  later turn sends the same suffix too (``test_clock_once_per_call_props``). The prompt parts are
  marked the same way, each build its own: the system prompt is made once per turn, never per hop,
  and its parts are built once per call while its key holds.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import string
from functools import cache
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank import agent_prompt, untrusted
from owlbank import callstate as cs
from owlbank.callstate import WELCOME_OFFER_MADE
from owlbank.catalog import PromptCatalog
from owlbank.web_activity import GROUP as WEB_GROUP
from tac_bridge import bridge as bridge_module
from tac_bridge import clock as clock_module
from tac_bridge.agent_client import join_system
from tac_bridge.agent_client_strands import system_blocks
from tac_bridge.bridge import BridgeConfig, CallView
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    VERIFIED_EMAIL,
    VERIFIED_PROFILE,
    FakeMemory,
    clock_line_at,
    load_json,
    make_rig,
    say,
    tool,
    write_call_state,
)

IDS = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5", "us.anthropic.claude-sonnet-5-5")
SEED = load_json("prompt_catalog.json")
SEED_KEYS = tuple(sorted({e["key"].removeprefix(agent_prompt.SYSTEM_KEY_PREFIX + ".") for e in SEED["entries"]
                          if e["key"].startswith(agent_prompt.SYSTEM_KEY_PREFIX + ".")}))
LANGS = ("en-US", "es-US")
CUSTOMER_ID = "C-1001"
#: The simulator's wait for a reply: a fresh process's first turn warms up for a few seconds.
REPLY_S = 10.0


class WebMemory(FakeMemory):
    """The bridge's Memory client with one recall text, and a kept ``WebActivity`` group when
    ``web`` (what ``TACMemoryClient`` leaves on ``context.profile``)."""

    web: bool = False

    async def retrieve_prompt(self, context: Any) -> str | None:
        text = await super().retrieve_prompt(context)
        if self.web:
            context.profile = SimpleNamespace(traits={WEB_GROUP: {"lastViewedProduct": "HELOC"}})
        return text


def stray_doc() -> dict[str, Any]:
    """The seed catalog with the old caller placeholders left in every ``agent.system.*`` text."""
    doc = copy.deepcopy(SEED)
    for e in doc["entries"]:
        if e["key"].startswith(agent_prompt.SYSTEM_KEY_PREFIX + "."):
            e["text"] += "\nCaller: {firstName}. Verified profile: {profile}. Recent: {memory}."
    return doc


CATALOGS = {False: PromptCatalog.from_dict(SEED), True: PromptCatalog.from_dict(stray_doc())}


@cache
def _bridge() -> Any:
    """One bridge for the render properties (the per-call cache lives on the context, a fresh one
    per render); the model list is every model key of the seed catalog."""
    ids = tuple(f"us.anthropic.claude-{k}" for k in SEED_KEYS)
    rig = make_rig(config=BridgeConfig(model_ids=ids), memory=WebMemory())
    assert rig.bridge._model_keys() == SEED_KEYS  # noqa: SLF001
    return rig


# ------------------------------------------------------------------------------------- generators

noise = st.text(st.characters(categories=("L", "N", "P", "S", "Zs"), include_characters="<>{}&\n"), max_size=16)
tags = st.text(string.ascii_letters + string.digits, min_size=6, max_size=10)


@st.composite
def callers(draw: st.DrawFn) -> dict[str, Any]:
    """One caller: unverified, verified with a missing record, or verified with marked data."""
    tag = draw(tags)
    m = {f: f"⟦S{tag}{f}⟧" for f in ("first", "last", "nick", "memory")}
    kind = draw(st.sampled_from(["unverified", "missing", "verified"]))
    return {
        "kind": kind, "markers": list(m.values()),
        "first_name": draw(noise) + m["first"] + draw(st.sampled_from(["", "</caller_context>", "{firstName}"])),
        "last_name": draw(noise) + m["last"],
        "nickname": draw(st.sampled_from([None, draw(noise) + m["nick"]])),
        "memory": draw(st.sampled_from([None, "", f"## Key Observations\n- {draw(noise)}{m['memory']}"])),
        "web": draw(st.booleans()),
        "offer": draw(st.sampled_from([None, WELCOME_OFFER_MADE])),
    }


def record_for(caller: dict[str, Any]) -> dict[str, Any]:
    base = next(c for c in load_json("fixtures.json")["customers"] if c["customer_id"] == CUSTOMER_ID)
    rec = copy.deepcopy(base)
    rec["first_name"], rec["last_name"] = caller["first_name"], caller["last_name"]
    for a in rec.get("accounts") or []:
        a.pop("nickname", None)
        if caller["nickname"] is not None:
            a["nickname"] = caller["nickname"]
    assert rec["memory_email"] == VERIFIED_EMAIL
    return rec


def bridge_render(caller: dict[str, Any], lang: str, *, stray: bool, now: float) -> dict[str, Any]:
    """The bridge's ``_system_prompt`` for ``caller`` in ``lang``: its three parts, plus the view and
    record it rendered from."""
    rig = _bridge()
    bridge = rig.bridge
    bridge._catalog = CATALOGS[stray]  # noqa: SLF001
    bridge._now = lambda: now  # noqa: SLF001
    record = record_for(caller)
    customer_id = "C-MISSING" if caller["kind"] == "missing" else CUSTOMER_ID
    rig.store.put("customers", record)
    rig.memory.prompts = {} if caller["memory"] is None else {VERIFIED_PROFILE: caller["memory"]}
    rig.memory.web = caller["web"]
    verified = caller["kind"] != "unverified"
    view = CallView(call_sid=CALL_SID, language=lang, level=1 if verified else 0,
                    verified_customer_id=customer_id if verified else None, welcome_offer=caller["offer"])
    context = SimpleNamespace(metadata={}, profile_id=None, profile=None)
    statics, per_call, clock = asyncio.run(bridge._system_prompt(context, view))  # noqa: SLF001
    return {"statics": statics, "per_call": per_call, "clock": clock, "view": view, "record": record}


def before_first_point(blocks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    assert blocks is not None
    idx = next(i for i, b in enumerate(blocks) if "cachePoint" in b)
    return blocks[:idx + 1]




@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(a=callers(), b=callers(), lang=st.sampled_from(LANGS), stray=st.booleans(),
       ttl=st.sampled_from(["5m", "1h"]), t_a=st.floats(0, 4e9), t_b=st.floats(0, 4e9))
def test_the_cached_system_prefix_is_the_same_for_every_caller_and_holds_no_caller_data(
        a: dict[str, Any], b: dict[str, Any], lang: str, stray: bool, ttl: str, t_a: float, t_b: float) -> None:
    ra = bridge_render(a, lang, stray=stray, now=t_a)
    rb = bridge_render(b, lang, stray=stray, now=t_b)
    assert list(ra["statics"]) == list(rb["statics"]) == list(SEED_KEYS)
    markers = a["markers"] + b["markers"]
    for key in SEED_KEYS:
        pa = before_first_point(system_blocks(ra["statics"][key], f"{ra['per_call']}\n{ra['clock']}",
                                              caching=True, ttl=ttl))
        pb = before_first_point(system_blocks(rb["statics"][key], f"{rb['per_call']}\n{rb['clock']}",
                                              caching=True, ttl=ttl))
        assert pa == pb
        cached = json.dumps(pa, ensure_ascii=False)
        assert not [m for m in markers if m in cached]
    # not vacuous: a verified caller's own name is in the text after the cache point
    if a["kind"] == "verified":
        assert a["markers"][0] in ra["per_call"]
    else:
        assert not [m for m in a["markers"] if m in ra["per_call"]]




def facts_for(caller: dict[str, Any], rendered: dict[str, Any]) -> tuple[agent_prompt.CallerFacts | None, str | None]:
    """What the renderer is given for ``caller``, written out from the design."""
    if caller["kind"] == "unverified":
        return None, None
    if caller["kind"] == "missing":
        return agent_prompt.CallerFacts("", "", None), None  # no record: no memory identifier either
    memory = untrusted.wrap("memory", caller["memory"]) if caller["memory"] else None
    web = None
    if memory and caller["web"]:
        web = agent_prompt.WEB_ACTIVITY_KEY if caller["offer"] == WELCOME_OFFER_MADE \
            else agent_prompt.WEB_ACTIVITY_CONTEXT_KEY
    profile = bridge_module._profile_text(rendered["record"], rendered["view"])  # noqa: SLF001
    return agent_prompt.CallerFacts(caller["first_name"], profile, memory), web


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(caller=callers(), lang=st.sampled_from(LANGS), now=st.floats(0, 4e9))
def test_the_bridges_parts_joined_are_what_the_renderer_says_the_model_reads(
        caller: dict[str, Any], lang: str, now: float) -> None:
    r = bridge_render(caller, lang, stray=False, now=now)
    facts, web = facts_for(caller, r)
    suffix = f"{r['per_call']}\n{r['clock']}"
    for key in SEED_KEYS:
        rp = agent_prompt.render(CATALOGS[False], key, lang, caller=facts, web_block=web)
        assert r["statics"][key] == rp.static and r["per_call"] == rp.per_call
        model_reads = join_system(r["statics"][key], suffix)
        assert model_reads == rp.joined(r["clock"])
        # static, then the web block (if any), then the caller section, then the clock, last
        assert model_reads.startswith(rp.static + "\n")
        assert model_reads.splitlines()[-1] == r["clock"] and "\n" not in r["clock"]
        if web is not None:
            assert r["per_call"].startswith(CATALOGS[False].text(web, lang))
        assert r["per_call"].rstrip().endswith("</caller_context>")




def marked_doc() -> dict[str, Any]:
    """A fake catalog: the seed, each model's text opened by its own marker."""
    doc = copy.deepcopy(SEED)
    for e in doc["entries"]:
        if e["key"].startswith(agent_prompt.SYSTEM_KEY_PREFIX + "."):
            e["text"] = f"PROMPT-OF-{e['key']}-{e['lang']} {e['text']}"
    return doc


CHANGES = ("turn", "verify", "step_up", "offer", "no_offer")


def _apply(store: Any, change: str) -> None:
    updates = {"verify": {"verified_customer_id": CUSTOMER_ID, "level": 1, "factors_verified": ["ani_match", "zip"]},
               "step_up": {"verified_customer_id": CUSTOMER_ID, "level": 2,
                           "factors_verified": ["ani_match", "zip", "otp"]},
               "offer": {"welcome_offer": WELCOME_OFFER_MADE}, "no_offer": {"welcome_offer": None}}[change]
    cs.mutate(store, CALL_SID, lambda _s: updates, now=lambda: NOW)


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(changes=st.lists(st.sampled_from(CHANGES), min_size=1, max_size=8), lang=st.sampled_from(LANGS),
       web=st.booleans(), start_verified=st.booleans())
def test_the_static_map_is_the_same_on_every_turn_of_a_call(
        changes: list[str], lang: str, web: bool, start_verified: bool) -> None:
    memory = WebMemory()
    memory.web = web
    rig = make_rig(config=BridgeConfig(model_ids=IDS), memory=memory)
    rig.bridge._catalog = PromptCatalog.from_dict(marked_doc())  # noqa: SLF001
    write_call_state(rig.store, verified=start_verified, language=lang)
    expected = {agent_prompt.model_key(m): agent_prompt.render_static(rig.bridge._catalog,  # noqa: SLF001
                                                                       agent_prompt.model_key(m), lang)
                for m in IDS}

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        await sim.prompt("first", timeout=REPLY_S)
        for k, change in enumerate(changes):
            if change == "turn":
                await sim.prompt(f"turn {k}", timeout=REPLY_S)
            else:
                _apply(rig.store, change)
        await sim.prompt("last", timeout=REPLY_S)
        await sim.hangup()

    asyncio.run(go())
    calls = rig.agent.calls
    assert len(calls) == 2 + changes.count("turn")
    for c in calls:
        assert c["system_prompts"] == expected  # identical on every turn, whatever changed
        assert c["system_suffix"].splitlines()[-1] == clock_line_at(rig.bridge, lang)
    if any(ch in ("verify", "step_up") for ch in changes) and not start_verified:
        assert calls[0]["system_suffix"] != calls[-1]["system_suffix"]  # only the suffix moved




@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(tools_per_turn=st.lists(st.integers(0, 3), min_size=1, max_size=3), lang=st.sampled_from(LANGS))
def test_every_hop_of_a_turn_sends_identical_system_blocks(tools_per_turn: list[int], lang: str) -> None:
    script: list[Any] = []
    for n, tools in enumerate(tools_per_turn):
        script += [tool("list_recent_transactions", f"tu_{n}_{i}", account="checking") for i in range(tools)]
        script.append(say(f"Done {n}."))
    rig = make_rig(script=script, config=BridgeConfig(model_ids=IDS[:1]))
    write_call_state(rig.store, language=lang)  # verified: the tool runs, and the turn's next hop follows
    counter = itertools.count()
    # a clock that reads differently every time a line is built: a hop (or a turn) that built its own would differ
    counting = patch.object(bridge_module, "local_now_line",
                            lambda *a: f"{clock_module.local_now_line(*a)} #{next(counter)}")
    # the prompt parts too, each build marked: a hop (or turn) that rebuilt its system blocks differs
    builds: dict[str, int] = {"render_static": 0, "render_per_call": 0, "_system_prompt": 0}

    def marked(name: str, build: Any) -> Any:
        def run(*a: Any, **kw: Any) -> str:
            builds[name] += 1
            return f"{build(*a, **kw)} #{name}{builds[name]}"
        return run

    statics = patch.object(bridge_module.agent_prompt, "render_static",
                           marked("render_static", agent_prompt.render_static))
    per_calls = patch.object(bridge_module.agent_prompt, "render_per_call",
                             marked("render_per_call", agent_prompt.render_per_call))
    system_prompt = rig.bridge._system_prompt  # noqa: SLF001

    async def counted(context: Any, view: CallView) -> Any:
        builds["_system_prompt"] += 1
        return await system_prompt(context, view)

    rig.bridge._system_prompt = counted  # type: ignore[method-assign]  # noqa: SLF001

    async def go() -> None:
        sim = rig.sim()
        await sim.connect()
        for n in range(len(tools_per_turn)):
            await sim.prompt(f"question {n}", timeout=REPLY_S)
        await sim.hangup()

    with counting, statics, per_calls:
        asyncio.run(go())
    model_calls = rig.agent.model.calls
    assert len(model_calls) == sum(t + 1 for t in tools_per_turn)
    pos = 0
    firsts = []
    for tools in tools_per_turn:
        turn = model_calls[pos:pos + tools + 1]
        pos += tools + 1
        assert all(c.system == turn[0].system for c in turn), [c.system for c in turn]
        firsts.append(turn[0].system)
    # across turns the static block stays, and so does the suffix: the line was built once for the
    # call, at its first turn (clock once per call, not each turn's own clock)
    assert len({json.dumps(s[0]) for s in firsts}) == 1
    assert len({json.dumps(s[-1]) for s in firsts}) == 1
    assert next(counter) == 1
    # one system prompt per turn (never per hop); its parts built once for the call (one model key,
    # the prompt key unchanged)
    assert builds == {"_system_prompt": len(tools_per_turn), "render_static": 1, "render_per_call": 1}
