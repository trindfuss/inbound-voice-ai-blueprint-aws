"""Slot mode properties.

* For any raw slot, flow and default values (any JSON), any ceiling and any
  ``generative_enabled``, ``configured`` is the first present valid value in the order slot, flow,
  default (a present invalid value gives ``deterministic``); ``effective <= ceiling`` and
  ``<= configured``; ``effective == generative`` exactly when the ceiling, the configuration and
  ``generative_enabled`` allow it and the slot is a non-sensitive choice.
* Any invalid mode value in the policy table gives ``deterministic``, never raises, and
  emits ``slot_mode_invalid`` exactly once per (item, field, version) across repeated
:func:`freeze` calls.
* ``restrict(frozen, c) <= frozen`` and ``<= c``, monotone in ``c``, and
  ``restrict(restrict(f, c1), c2) == restrict(f, min(c1, c2))``.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.slot_policy import FLOWS, MODES, Mode, ResolutionSettings, SlotSettings, mode_rank
from owlbank.slots import modes
from owlbank.slots.modes import INVALID, freeze, mode_decision, restrict
from owlbank.slots.types import SlotDecl, SlotType
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
SEED: dict[str, Any] = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=12),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3),
    max_leaves=6,
)
mode_st = st.sampled_from(MODES)
raw_values = st.one_of(mode_st, st.none(), json_values, st.just(INVALID),
                       st.sampled_from(["Assisted", " assisted", "generative\n", "auto"]))
#: What the table can hold (DynamoDB has no floats).
storable_json = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=12),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3),
    max_leaves=6,
)
storable_invalid = st.one_of(storable_json, st.sampled_from(["Assisted", "auto", ""])).filter(
    lambda v: not (isinstance(v, str) and v in MODES))


def _valid(v: object) -> bool:
    return isinstance(v, str) and v in MODES


@st.composite
def slots(draw: st.DrawFn) -> SlotSettings:
    slot_type: SlotType = draw(st.sampled_from(["choice", "yes_no", "digits"]))
    return SlotSettings(name="s", decl=SlotDecl(slot_type=slot_type), is_factor=False,
                        sensitive=draw(st.booleans()))




@given(s=slots(), raw_mode=raw_values, raw_flow=raw_values, raw_default=raw_values, ceiling=mode_st,
       gen=st.booleans(), flow=st.sampled_from(FLOWS))
def test_precedence_and_ceiling(s: SlotSettings, raw_mode: object, raw_flow: object,
                                     raw_default: object, ceiling: Mode, gen: bool, flow: str) -> None:
    r = ResolutionSettings(generative_enabled=gen)
    d = mode_decision(s, r, flow, ceiling, raw_mode=raw_mode, raw_flow_mode=raw_flow, raw_default=raw_default)
    # configured: the first present value decides (the default always "is present": missing = invalid).
    present = [(v, src) for v, src in ((raw_mode, "slot"), (raw_flow, "flow")) if v is not None]
    first, src = present[0] if present else (raw_default, "default")
    if _valid(first):
        assert (d.configured, d.source) == (first, src)
    else:
        assert (d.configured, d.source, d.effective) == ("deterministic", "invalid", "deterministic")
    assert (d.slot, d.flow, d.ceiling) == ("s", flow, ceiling)
    assert mode_rank(d.effective) <= mode_rank(ceiling)
    assert mode_rank(d.effective) <= mode_rank(d.configured)
    allowed = (ceiling == "generative" and d.configured == "generative" and gen
               and s.decl.slot_type == "choice" and not s.sensitive)
    assert (d.effective == "generative") is allowed
    # capped says why, exactly when the effective mode is lower.
    assert (d.capped is None) is (d.effective == d.configured)
    if d.capped == "ceiling":
        assert d.effective == ceiling
    elif d.capped is not None:
        assert (d.configured, d.effective) == ("generative", "assisted")




@contextmanager
def _captured() -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("owlbank.slots.modes")
    handler, old = _H(level=logging.DEBUG), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)


@st.composite
def bad_tables(draw: st.DrawFn) -> tuple[MemoryStore, str, list[str], tuple[str, str], int]:
    """A seeded table with one invalid mode value; the flow and slots to freeze; the (item, field)
    expected in ``slot_mode_invalid``; the policy version."""
    version = draw(st.integers(0, 10_000))
    doc = copy.deepcopy(SEED)
    doc["version"] = version
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    target = draw(st.sampled_from(["slot", "flow", "default"]))
    flow = draw(st.sampled_from(["login", "stepup"]))
    names = ["zip", "ssn4", "otp"]
    res = store.get("policy", {"pk": "slot_resolution"})
    assert res is not None
    if target == "slot":
        name = draw(st.sampled_from(names))
        slot_item = store.get("policy", {"pk": f"slot#{name}"})
        assert slot_item is not None
        store.put("policy", {**slot_item, "mode": draw(storable_invalid.filter(lambda v: v is not None))})
        expected = (f"slot#{name}", "mode")
    elif target == "flow":
        bad = draw(storable_invalid.filter(lambda v: v is not None and not isinstance(v, dict)))
        entry = draw(st.sampled_from([{"mode": bad}, bad, {"mode": "assisted", "extra": bad}]))
        store.put("policy", {**res, "flows": {flow: entry}})
        expected = ("slot_resolution", f"flows.{flow}.mode")
    else:
        store.put("policy", {**res, "default_mode": draw(storable_invalid)})
        expected = ("slot_resolution", "default_mode")
    return store, flow, names, expected, version


@given(table=bad_tables(), calls=st.integers(1, 4))
def test_invalid_mode_is_deterministic_and_reported_once(
        table: tuple[MemoryStore, str, list[str], tuple[str, str], int], calls: int) -> None:
    store, flow, names, expected, version = table
    modes.reset_invalid_reports()
    policy = Policy.from_store(store)
    with _captured() as records:
        for _ in range(calls):
            decisions = freeze(policy, flow, names, "generative")
    invalid = [d for d in decisions.values() if d.source == "invalid"]
    assert invalid and all(d.effective == d.configured == "deterministic" for d in invalid)
    if expected[0].startswith("slot#"):
        assert [d.slot for d in invalid] == [expected[0].removeprefix("slot#")]
    else:
        assert len(invalid) == len(names)  # every slot inherits the flow's or the default
    events = [(r.item, r.field, r.policy_version) for r in records  # type: ignore[attr-defined]
              if getattr(r, "event", None) == "slot_mode_invalid"]
    assert events == [(*expected, version)]
    assert all(r.levelno == logging.ERROR for r in records if getattr(r, "event", None) == "slot_mode_invalid")


# --------------------------------------------------------------------------- (restrict half)


@given(f=mode_st, c=mode_st, c1=mode_st, c2=mode_st)
def test_restrict_takes_the_lower_mode(f: Mode, c: Mode, c1: Mode, c2: Mode) -> None:
    out = restrict(f, c)
    assert mode_rank(out) <= mode_rank(f) and mode_rank(out) <= mode_rank(c)
    assert out in (f, c)
    if mode_rank(c1) <= mode_rank(c2):
        assert mode_rank(restrict(f, c1)) <= mode_rank(restrict(f, c2))
    lowest = c1 if mode_rank(c1) <= mode_rank(c2) else c2
    assert restrict(restrict(f, c1), c2) == restrict(f, lowest)
    assert restrict(f, "generative") == f and restrict(f, "deterministic") == "deterministic"
