"""Slot settings properties.

* A non-boolean ``sensitive``, ``keypad_only`` or ``barge_in`` is
  refused by:func:`parse_slot` and loads fail-closed from:func:`load_slot`, with exactly one
  error naming the field and never the value.
* Round trip: ``parse_slot(s.to_doc) == s``; a ``slots`` section -> items -> the store loader
  gives the same:meth:`Policy.slot` for every name.
* A ``factors`` entry with no ``slot#`` item loads ``sensitive: true``.
* :func:`gather_input` is keypad-only iff
  ``keypad_only or (is_factor and not accept_spoken_factors)``; ``barge_in`` passes through.
* :func:`alias_conflicts` is non-empty iff some normalized alias or spoken name
  maps to two options in one language; validation refuses exactly those option sets.
* Round trips: ``parse_slot(s.to_doc) == s`` with ``mode``, ``confirm`` and
  ``options``; ``parse_slot_resolution(r.to_doc) == r``; seed section -> items -> the store
  loader gives the same:meth:`Policy.slot` and:attr:`Policy.slot_resolution`.
* Extended: the phase 1 fields keep their fail-closed loading with the phase 2
  fields present, valid or not.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from hypothesis import assume, given
from hypothesis import strategies as st

from owlbank.policy import Policy
from owlbank.slot_policy import (
    CONFIRM_POLICIES,
    FLOWS,
    MODES,
    RESOLVER_BUDGET_MS_RANGE,
    OptionDef,
    SlotPolicyError,
    SlotSettings,
    alias_conflicts,
    gather_input,
    load_slot,
    load_slot_resolution,
    parse_slot,
    parse_slot_resolution,
)
from owlbank.slots.normalize import normalize
from owlbank.store import MemoryStore

DATA = Path(__file__).resolve().parents[2] / "data"
SEED: dict[str, Any] = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
FIELDS = ("sensitive", "keypad_only", "barge_in")
DEFAULTS = {"sensitive": False, "keypad_only": False, "barge_in": True}
ZIP_DEF = SEED["factors"]["zip"]
SENTINEL = "SENTINEL-VALUE-31337"

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(max_size=12),
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3),
    max_leaves=6,
)
non_booleans = json_values.filter(lambda v: not isinstance(v, bool))
slot_names = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)
settings_fields = st.fixed_dictionaries({}, optional={f: st.booleans() for f in FIELDS})


@st.composite
def non_factor_items(draw: st.DrawFn) -> dict[str, Any]:
    item: dict[str, Any] = {"slotType": draw(st.sampled_from(["yes_no", "choice", "digits"]))}
    item.update(draw(settings_fields))
    if draw(st.booleans()):
        item["maxRetries"] = draw(st.integers(0, 5))
    if draw(st.booleans()):
        item["flow"] = draw(st.sampled_from(FLOWS))
    if draw(st.booleans()):
        item["promptKey"] = draw(st.sampled_from(["prompt.a", "prompt.b"]))
    return item


# --------------------------------------------------------------------------- (validation half)


@given(field=st.sampled_from(FIELDS), value=non_booleans, factor=st.booleans())
def test_non_boolean_refused_and_loads_fail_closed(field: str, value: Any, factor: bool) -> None:
    name, fd = ("zip", ZIP_DEF) if factor else ("yes_no", None)
    base: dict[str, Any] = {} if factor else {"slotType": "yes_no"}
    raw = {**base, field: value}
    try:
        parse_slot(name, raw, factor_def=fd)
    except SlotPolicyError as exc:
        assert exc.field == f"slot#{name}.{field}"
    else:
        raise AssertionError("a non-boolean was accepted")
    s, errors = load_slot(name, raw, factor_def=fd)
    assert s is not None and getattr(s, field) is True
    assert all(getattr(s, f) is DEFAULTS[f] for f in FIELDS if f != field)
    assert [e.field for e in errors] == [f"slot#{name}.{field}"]
    # The message is the same whatever the value: it can't carry it.
    _, sentinel_errors = load_slot(name, {**base, field: SENTINEL}, factor_def=fd)
    assert [str(e) for e in errors] == [str(e) for e in sentinel_errors]
    assert SENTINEL not in str(sentinel_errors[0])


@given(raw=settings_fields)
def test_booleans_and_absent_fields(raw: dict[str, bool]) -> None:
    s = parse_slot("zip", raw, factor_def=ZIP_DEF)
    loaded, errors = load_slot("zip", raw, factor_def=ZIP_DEF)
    assert loaded == s and errors == []
    for f in FIELDS:
        assert getattr(s, f) is raw.get(f, DEFAULTS[f])


# --------------------------------------------------------------------------- round trips


@given(raw=settings_fields)
def test_factor_to_doc_round_trip(raw: dict[str, bool]) -> None:
    s = parse_slot("zip", raw, factor_def=ZIP_DEF)
    assert parse_slot("zip", s.to_doc(), factor_def=ZIP_DEF) == s


@given(name=slot_names, item=non_factor_items())
def test_non_factor_to_doc_round_trip(name: str, item: dict[str, Any]) -> None:
    s = parse_slot(name, item, factor_def=None)
    assert parse_slot(name, s.to_doc(), factor_def=None) == s
    loaded, errors = load_slot(name, s.to_doc(), factor_def=None)
    assert loaded == s and errors == []


@st.composite
def slots_sections(draw: st.DrawFn) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for f in SEED["factors"]:
        if draw(st.booleans()):
            section[f] = draw(settings_fields)
    extra = draw(st.lists(slot_names.filter(lambda n: n not in SEED["factors"]), max_size=3, unique=True))
    for n in extra:
        section[n] = draw(non_factor_items())
    return section


@given(section=slots_sections())
def test_section_items_store_round_trip(section: dict[str, Any]) -> None:
    doc = {**copy.deepcopy(SEED), "slots": section}
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    a, b = Policy.from_dict(doc), Policy.from_store(store)
    assert a.slots() == b.slots()
    assert set(a.slots()) == set(SEED["factors"]) | set(section)
    for n in a.slots():
        assert a.slot(n) == b.slot(n)
    assert b.slot_problems == ()




@given(missing=st.sets(st.sampled_from(sorted(SEED["factors"])), min_size=1), keep_meta=st.booleans())
def test_factor_with_no_item_is_sensitive(missing: set[str], keep_meta: bool) -> None:
    doc = copy.deepcopy(SEED)
    for f in SEED["factors"]:  # the others explicitly not sensitive, so only the fail-closed default (no item) can flag them
        doc["slots"][f] = {"sensitive": False}
    store = MemoryStore()
    for item in Policy.to_items(doc):
        if item["pk"] in {f"slot#{m}" for m in missing}:
            continue
        if item["pk"] == "meta" and not keep_meta:
            item = {**item, "slots": [n for n in item["slots"] if n not in missing]}
        store.put("policy", item)
    p = Policy.from_store(store)
    for f in SEED["factors"]:
        assert p.slot(f).sensitive is (f in missing)
    assert {e.field for e in p.slot_problems} == {f"slot#{m}" for m in missing}


# --------------------------------------------------------------------------- (decision half)


@st.composite
def any_settings(draw: st.DrawFn) -> SlotSettings:
    raw = draw(settings_fields)
    if draw(st.booleans()):
        return parse_slot("zip", raw, factor_def=ZIP_DEF)
    return parse_slot(draw(slot_names), {"slotType": "yes_no", **raw}, factor_def=None)


@given(s=any_settings(), asf=st.booleans())
def test_gather_input_follows_settings(s: SlotSettings, asf: bool) -> None:
    g = gather_input(s, asf)
    assert g.keypad_only is (s.keypad_only or (s.is_factor and not asf))
    assert g.barge_in is s.barge_in
    if not asf and s.is_factor:
        assert g.keypad_only is True
    if not s.is_factor:
        assert g.keypad_only is s.keypad_only


# --------------------------------------------------------------------------- phase 2
# The phase 2 round trips and the audit properties, extended.

LANGS = ("en-US", "es-US")
VALUES = ("red", "purple", "rewards", "blue_2")
#: Small, colliding vocabulary: case, accents, punctuation, fillers and spacing all normalize away.
PHRASES = ("red", "Red!", "RED", "rojá", "roja", "purple", "the card", "The  card.", "um card", "card",
           "morada", "rewards", "?!")
CARD_SHAPE: dict[str, Any] = {"slotType": "choice", "promptKey": "prompt.card", "labelKey": "label.card",
                              "maxRetries": 2, "flow": "confirm"}


@st.composite
def option_docs(draw: st.DrawFn, *, valid_phrases: bool = False) -> list[dict[str, Any]]:
    values = draw(st.lists(st.sampled_from(VALUES), min_size=1, max_size=4, unique=True))
    phrases = st.sampled_from([p for p in PHRASES if p != "?!"] if valid_phrases else PHRASES)
    out: list[dict[str, Any]] = []
    for v in values:
        o: dict[str, Any] = {"value": v}
        if draw(st.booleans()):
            o["aliases"] = draw(st.dictionaries(st.sampled_from(LANGS), st.lists(phrases, max_size=3), max_size=2))
        out.append(o)
    return out


def _brute_force_conflict(options: list[OptionDef], names: dict[str, dict[str, str]], langs: tuple[str, ...]) -> bool:
    """Oracle: two different options with a phrase each that normalize to the same words."""
    def phrases(o: OptionDef, lang: str) -> list[str]:
        own = [names[o.value][lang]] if lang in names.get(o.value, {}) else []
        return own + list(o.aliases.get(lang, ()))
    return any(normalize(a, lang) and normalize(a, lang) == normalize(b, lang)
               for lang in langs for i, o1 in enumerate(options) for o2 in options[i + 1:]
               for a in phrases(o1, lang) for b in phrases(o2, lang))


@given(docs=option_docs(valid_phrases=True),
       names=st.dictionaries(st.sampled_from(VALUES), st.dictionaries(st.sampled_from(LANGS),
                                                                     st.sampled_from(PHRASES), max_size=2),
                             max_size=3),
       langs=st.sampled_from([LANGS, ("en-US",), ("es-US",)]))
def test_alias_conflicts_iff_a_phrase_maps_to_two_options(
        docs: list[dict[str, Any]], names: dict[str, dict[str, str]], langs: tuple[str, ...]) -> None:
    options = [OptionDef(d["value"], {k: tuple(v) for k, v in d.get("aliases", {}).items()}) for d in docs]
    conflicts = alias_conflicts(options, names, langs)
    assert bool(conflicts) is _brute_force_conflict(options, names, langs)
    for c in conflicts:
        assert c.lang in langs and len(c.options) >= 2 and list(c.options) == sorted(set(c.options))
    # Validation (aliases only, no spoken names) refuses exactly the conflicting sets ...
    raw = {**CARD_SHAPE, "options": docs}
    try:
        parse_slot("card", raw, factor_def=None, langs=LANGS)
    except SlotPolicyError as exc:
        assert exc.field.startswith("slot#card.options.aliases.")
        assert _brute_force_conflict(options, {}, LANGS)
    else:
        assert not _brute_force_conflict(options, {}, LANGS)
    # ... and the lenient loader leaves none behind.
    loaded, _ = load_slot("card", raw, factor_def=None, langs=LANGS)
    assert loaded is not None and alias_conflicts(loaded.options, {}, LANGS) == []


@st.composite
def phase2_items(draw: st.DrawFn) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """A valid slot item with any of the phase 2 fields: (name, item, factor_def)."""
    kind = draw(st.sampled_from(["factor", "yes_no", "choice"]))
    raw: dict[str, Any] = dict(draw(settings_fields))
    sensitive = raw.get("sensitive", False)
    if kind == "factor":
        name, fd = "zip", ZIP_DEF
    else:
        name, fd = ("card", None) if kind == "choice" else ("yes_no", None)
        raw.update(CARD_SHAPE if kind == "choice" else {"slotType": "yes_no", "flow": "confirm"})
    allowed = [m for m in MODES if m != "generative" or (kind == "choice" and not sensitive)]
    if draw(st.booleans()):
        raw["mode"] = draw(st.sampled_from(allowed))
    if draw(st.booleans()):
        raw["confirm"] = draw(st.sampled_from(CONFIRM_POLICIES))
    if kind == "choice" and draw(st.booleans()):
        docs = draw(option_docs(valid_phrases=True))
        options = [OptionDef(d["value"], {k: tuple(v) for k, v in d.get("aliases", {}).items()}) for d in docs]
        assume(not _brute_force_conflict(options, {}, LANGS))
        for i, d in enumerate(docs):
            if draw(st.booleans()):
                d["dtmf"] = str(i + 1)
        raw["options"] = docs
    return name, raw, fd


@given(case=phase2_items())
def test_phase2_parse_to_doc_round_trip(case: tuple[str, dict[str, Any], dict[str, Any] | None]) -> None:
    name, raw, fd = case
    s = parse_slot(name, raw, factor_def=fd)
    assert parse_slot(name, s.to_doc(), factor_def=fd) == s
    loaded, errors = load_slot(name, s.to_doc(), factor_def=fd)
    assert loaded == s and errors == []


resolution_docs = st.fixed_dictionaries({}, optional={
    "default_mode": st.sampled_from(MODES),
    "flows": st.dictionaries(st.sampled_from(FLOWS), st.builds(lambda m: {"mode": m}, st.sampled_from(MODES))),
    "generative_enabled": st.booleans(),
    "resolver_budget_ms": st.integers(*RESOLVER_BUDGET_MS_RANGE),
})


@given(raw=resolution_docs)
def test_slot_resolution_round_trip(raw: dict[str, Any]) -> None:
    r = parse_slot_resolution(raw)
    assert parse_slot_resolution(r.to_doc()) == r
    assert load_slot_resolution(r.to_doc()) == (r, [])
    assert r.default_mode == raw.get("default_mode", "assisted")
    assert dict(r.flows) == {f: e["mode"] for f, e in raw.get("flows", {}).items()}


@given(case=phase2_items(), res=resolution_docs)
def test_phase2_seed_items_store_round_trip(case: tuple[str, dict[str, Any], dict[str, Any] | None],
                                            res: dict[str, Any]) -> None:
    name, raw, _ = case
    doc = copy.deepcopy(SEED)
    doc["slots"][name] = raw
    doc["slot_resolution"] = res
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    a, b = Policy.from_dict(doc), Policy.from_store(store)
    assert a.slots() == b.slots()
    for n in a.slots():
        assert a.slot(n) == b.slot(n)
    assert a.slot_resolution == b.slot_resolution
    assert b.slot_problems == () and b.slot_resolution_problems == () and b.invalid_mode_fields == frozenset()


phase2_noise = st.fixed_dictionaries({}, optional={
    "mode": st.one_of(st.sampled_from(MODES), json_values),
    "confirm": st.one_of(st.sampled_from(CONFIRM_POLICIES), json_values),
})


@given(field=st.sampled_from(FIELDS), value=non_booleans, factor=st.booleans(), noise=phase2_noise,
       options=st.one_of(st.none(), option_docs(), json_values))
def test_extended_phase1_fields_fail_closed_with_phase2_fields_present(
        field: str, value: Any, factor: bool, noise: dict[str, Any], options: Any) -> None:
    name, fd = ("zip", ZIP_DEF) if factor else ("card", None)
    raw: dict[str, Any] = {} if factor else dict(CARD_SHAPE)
    raw.update(noise)
    if not factor and options is not None:
        raw["options"] = options
    raw[field] = value
    try:
        parse_slot(name, raw, factor_def=fd)
    except SlotPolicyError as exc:
        assert exc.field == f"slot#{name}.{field}"  # the phase 1 field is named first
    else:
        raise AssertionError("a non-boolean was accepted")
    s, errors = load_slot(name, raw, factor_def=fd)
    assert s is not None and getattr(s, field) is True
    assert all(getattr(s, f) is DEFAULTS[f] for f in FIELDS if f != field)
    assert [e.field for e in errors].count(f"slot#{name}.{field}") == 1
    # A bad mode still loads deterministic, whatever else is wrong.
    if "mode" in noise and not (isinstance(noise["mode"], str) and noise["mode"] in MODES) \
            and noise["mode"] is not None:
        assert s.mode == "deterministic"
    if s.mode == "generative":
        assert s.decl.slot_type == "choice" and not s.sensitive
