"""Tool-phrase masking properties through the bridge.

Random catalogs for one tool's pool (plain phrases, ``{account}`` templates, broken templates),
random accounts per turn, random delays and tool latencies, several turns in one call. Time is
real and short.

* No chunk the caller hears ever contains a brace; a turn whose pool has no phrase that
  renders for its arguments hears no tool phrase.
* Within the call, the tool never speaks the same pool phrase twice in a row, unless that
  phrase was the only one that rendered for the turn's arguments (a repeat beats silence; see
  :func:`tac_bridge.bridge.rotation_order`).
* A tool phrase is heard before the answer, and no earlier than the tool's start plus the
  delay.
"""

from __future__ import annotations

import asyncio
import copy
import time
from types import SimpleNamespace
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from owlbank.catalog import PromptCatalog
from owlbank.masking_config import pool_keys, render, words_for
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import SegmentLog
from tac_bridge.bridge import Bridge, rotation_order
from tests.component.masking_fixtures import masked_policy_doc
from tests.component.test_bridge_fakes import (
    CALL_SID,
    NOW,
    SEALER,
    FakeAgent,
    _inline_io,
    load_json,
    say,
    seeded_store,
    tool,
    write_call_state,
)
from tests.component.test_bridge_masking import RecordingBackend

TOOL = "list_recent_transactions"
BASE = f"mask.{TOOL}"
CATALOG = load_json("prompt_catalog.json")
TEMPLATES = ["Pulling up your activity.", "Reviewing your {account} activity.", "Getting {account} items.",
             "Checking {colour} things.", "Odd { brace.", "Looking now."]


# --------------------------------------------------------------------------- rotation_order (pure)


@given(st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=0, max_size=5, unique=True),
       st.one_of(st.none(), st.sampled_from(["a", "b", "c", "d", "e", "z"])))
def test_rotation_order_is_the_pool_with_the_last_phrase_last(pool: list[str], last: str | None) -> None:
    order = rotation_order(pool, last)
    assert sorted(order) == sorted(pool)
    if last in pool:
        assert order[-1] == last
        if len(pool) >= 2:
            assert order[0] != last  # The first choice is never the previous phrase
            assert order[0] == pool[(pool.index(last) + 1) % len(pool)]  # in rotation
    else:
        assert order == pool


# --------------------------------------------------------------------------- through the bridge


def catalog_with(pool: dict[str, str]) -> PromptCatalog:
    doc = copy.deepcopy(CATALOG)
    doc["entries"] = [e for e in doc["entries"] if not (e["key"] == BASE or e["key"].startswith(BASE + "."))]
    for key, text in pool.items():
        for lang in ("en-US", "es-US"):
            doc["entries"].append({"key": key, "lang": lang, "text": text, "version": 1,
                                   "status": "published", "audioUrl": None})
    return PromptCatalog.from_dict(doc)


POOLS = st.lists(st.sampled_from(TEMPLATES), min_size=1, max_size=4).flatmap(
    lambda texts: st.booleans().map(lambda with_base: {
        **({BASE: texts[0]} if with_base else {}),
        **{f"{BASE}.{i}": t for i, t in enumerate(texts[1:] if with_base else texts, start=1)}}))


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(pool=POOLS,
       turns=st.lists(st.tuples(st.sampled_from(["checking", "savings"]), st.sampled_from([0.0, 0.08])),
                      min_size=1, max_size=4),
       tool_after_ms=st.sampled_from([0, 40]))
def test_through_the_bridge(pool: dict[str, str], turns: list[tuple[str, float]],
                                            tool_after_ms: int) -> None:
    store = seeded_store()
    write_call_state(store)
    policy = Policy.from_dict(masked_policy_doc(after_ms=0, tool_after_ms=tool_after_ms))
    cat = catalog_with(pool)
    journal = SegmentLog(store, now=lambda: NOW)
    script: list[Any] = []
    for i, (account, _) in enumerate(turns):
        script += [tool(TOOL, f"tu_{i}", account=account), say("Done.")]
    agent = FakeAgent(script)
    backend = RecordingBackend({TOOL: lambda **_: {"tool": TOOL, "data": {"transactions": []}}})
    bridge = Bridge(store=store, agent=agent, backend=backend, memory=None, catalog=cat, policy=policy,
                    pii=PiiGuard.for_policy(FakePiiRedactor(), policy), journal=journal, sealer=SEALER,
                    now=lambda: NOW, io=_inline_io)
    ctx = SimpleNamespace(call_sid=CALL_SID, conversation_id="conv_m6", metadata={}, profile_id=None,
                          profile=None, pending_handoff_data=None)
    words = words_for(TOOL, cat, "en-US")
    keys = pool_keys(BASE, cat.keys)
    rendered = {k: {a: render(TOOL, cat.template(k, "en-US"), {"account": a}, words)
                    for a in ("checking", "savings")} for k in keys}

    async def one_turn(account: str, tool_s: float) -> tuple[list[tuple[str, float]], float]:
        backend.delay_s[TOOL] = tool_s
        n = len(backend.calls)
        chunks = [(c, time.monotonic()) async for c in bridge.run_turn(ctx, f"recent {account}")]
        return chunks, backend.started[f"{TOOL}:{n + 1}"]

    last_key: str | None = None
    seen_prompts = 0
    for account, tool_s in turns:
        chunks, tool_started = asyncio.run(one_turn(account, tool_s))
        texts = [c for c, _ in chunks]
        assert all("{" not in c and "}" not in c for c in texts)
        assert texts[-1] == "Done."
        prompts = [e["prompt_key"] for e in journal.read(CALL_SID) if e["type"] == "prompt"]
        new_prompts, seen_prompts = prompts[seen_prompts:], len(prompts)
        renderable = [k for k in keys if rendered[k][account]]
        heard = texts[:-1]
        if not renderable:
            assert heard == [] and new_prompts == []  # Nothing, never the generic
            continue
        if not heard:
            continue  # the answer came first (then no phrase at all)
        (phrase,) = heard  # at most one tool phrase per turn, and before the answer
        assert chunks[0][1] - tool_started >= tool_after_ms / 1000.0 - 0.01  # Not early
        (key,) = new_prompts
        assert rendered[key][account] + " " == phrase
        if last_key is not None and key == last_key:
            assert renderable == [key]  # A repeat only when nothing else renders
        last_key = key
