"""owlbank.policy: the ``web_offers`` section and the escalation reason descriptions."""

from __future__ import annotations

import copy
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.policy import (
    MAX_REASON_DESCRIPTION_CHARS,
    MAX_WEB_OFFERS,
    Policy,
    PolicyError,
    PolicySectionError,
    WebOffer,
    parse_reason_description,
    parse_web_offers,
)
from owlbank.store import MemoryStore
from owlbank.web_activity import MAX_VALUE_CHARS

DATA = Path(__file__).resolve().parents[2] / "data"
SEED: dict[str, Any] = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))


def with_offers(offers: Any) -> dict[str, Any]:
    doc = copy.deepcopy(SEED)
    doc["web_offers"] = {"offers": offers}
    return doc


def without_offers() -> dict[str, Any]:
    doc = copy.deepcopy(SEED)
    doc.pop("web_offers", None)
    return doc


def with_description(reason: str, text: Any) -> dict[str, Any]:
    doc = copy.deepcopy(SEED)
    doc["escalation"]["reasons"][reason]["description"] = text
    return doc


def seeded(doc: dict[str, Any]) -> MemoryStore:
    store = MemoryStore()
    for item in Policy.to_items(doc):
        store.put("policy", item)
    return store


# --------------------------------------------------------------------------- web_offers


def test_seed_offers_parse_in_order() -> None:
    p = Policy.from_dict(SEED)
    assert p.web_offers == (WebOffer("heloc", ("HELOC", "home-equity")),
                            WebOffer("auto_loan", ("Auto loan", "auto-loans")))
    assert isinstance(p.web_offers, tuple)


def test_absent_section_means_no_offers() -> None:
    doc = without_offers()
    assert Policy.from_dict(doc).web_offers == ()
    assert parse_web_offers(None) == ()
    items = Policy.to_items(doc)
    # The item is still written, empty, so a reseed without the section clears a live list.
    assert [i for i in items if i["pk"] == "web_offers"] == [{"pk": "web_offers", "offers": []}]
    assert Policy.from_store(seeded(doc)).web_offers == ()


def test_reseed_without_section_clears_old_offers() -> None:
    store = seeded(SEED)
    assert Policy.from_store(store).web_offers  # the old list is live
    for item in Policy.to_items(without_offers()):  # seed.py only puts, never deletes
        store.put("policy", item)
    assert store.get("policy", {"pk": "web_offers"}) == {"pk": "web_offers", "offers": []}
    assert Policy.from_store(store).web_offers == ()


def test_empty_list_is_allowed() -> None:
    assert parse_web_offers({"offers": []}) == ()


def test_store_round_trip() -> None:
    store = seeded(SEED)
    assert store.get("policy", {"pk": "web_offers"}) == {
        "pk": "web_offers", "offers": SEED["web_offers"]["offers"]}
    assert Policy.from_store(store).web_offers == Policy.from_dict(SEED).web_offers


GOOD = {"id": "heloc", "values": ["HELOC"]}


@pytest.mark.parametrize(("section", "field"), [
    ("x", "web_offers"),
    ({"offers": [], "extra": 1}, "web_offers.extra"),
    ({"offers": "heloc"}, "web_offers.offers"),
    ({"offers": [GOOD] + [{"id": f"o{n}", "values": [f"v{n}"]} for n in range(MAX_WEB_OFFERS)]},
     "web_offers.offers"),
    ({"offers": ["heloc"]}, "web_offers.offers[0]"),
    ({"offers": [{**GOOD, "phrase": "x"}]}, "web_offers.offers[0].phrase"),
    ({"offers": [{"values": ["HELOC"]}]}, "web_offers.offers[0].id"),
    ({"offers": [{"id": "HELOC", "values": ["x"]}]}, "web_offers.offers[0].id"),
    ({"offers": [{"id": "he-loc", "values": ["x"]}]}, "web_offers.offers[0].id"),
    ({"offers": [{"id": "", "values": ["x"]}]}, "web_offers.offers[0].id"),
    ({"offers": [{"id": "a" * 33, "values": ["x"]}]}, "web_offers.offers[0].id"),
    ({"offers": [{"id": 7, "values": ["x"]}]}, "web_offers.offers[0].id"),
    ({"offers": [GOOD, {"id": "heloc", "values": ["other"]}]}, "web_offers.offers[1].id"),
    ({"offers": [{"id": "heloc"}]}, "web_offers.offers[0].values"),
    ({"offers": [{"id": "heloc", "values": []}]}, "web_offers.offers[0].values"),
    ({"offers": [{"id": "heloc", "values": "HELOC"}]}, "web_offers.offers[0].values"),
    ({"offers": [{"id": "heloc", "values": [""]}]}, "web_offers.offers[0].values[0]"),
    ({"offers": [{"id": "heloc", "values": ["   "]}]}, "web_offers.offers[0].values[0]"),
    ({"offers": [{"id": "heloc", "values": [3]}]}, "web_offers.offers[0].values[0]"),
    ({"offers": [{"id": "heloc", "values": ["x" * (MAX_VALUE_CHARS + 1)]}]},
     "web_offers.offers[0].values[0]"),
    ({"offers": [{"id": "heloc", "values": ["HELOC", " heloc "]}]}, "web_offers.offers[0].values[1]"),
    ({"offers": [GOOD, {"id": "auto", "values": ["ＨＥＬＯＣ"]}]}, "web_offers.offers[1].values[0]"),
    ({"offers": [{"id": "a", "values": ["Auto  loan"]}, {"id": "b", "values": ["auto loan"]}]},
     "web_offers.offers[1].values[0]"),
])
def test_each_refusal(section: Any, field: str, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(PolicySectionError) as err:
        parse_web_offers(section)
    assert err.value.field == field
    doc = copy.deepcopy(SEED)
    doc["web_offers"] = section
    with pytest.raises(PolicyError):  # seeding refuses it
        Policy.to_items(doc)
    # At run time an invalid section turns the offers off (today's welcome), with an ERROR
    # naming the field and never a value.
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        assert Policy.from_dict(doc).web_offers == ()
    assert field in caplog.text
    assert "HELOC" not in caplog.text


def test_value_at_the_length_limit_is_allowed() -> None:
    value = "x" * MAX_VALUE_CHARS
    assert parse_web_offers({"offers": [{"id": "x", "values": [value]}]})[0].values == (value,)


def test_exactly_max_offers_is_allowed() -> None:
    offers = [{"id": f"o{n}", "values": [f"v{n}"]} for n in range(MAX_WEB_OFFERS)]
    assert len(parse_web_offers({"offers": offers})) == MAX_WEB_OFFERS


def test_invalid_store_item_turns_offers_off() -> None:
    store = seeded(SEED)
    store.put("policy", {"pk": "web_offers", "offers": [{"id": "BAD", "values": ["x"]}]})
    p = Policy.from_store(store)
    assert p.web_offers == ()
    assert p.reasons == Policy.from_dict(SEED).reasons  # the rest of the policy still loads


# --------------------------------------------------------------------------- validation half

ids = st.from_regex(r"[a-z0-9_]{1,32}", fullmatch=True)
values = st.text(st.characters(codec="utf-8", exclude_categories=("Cs",)), min_size=1,
                 max_size=MAX_VALUE_CHARS).filter(lambda v: v.strip())


def _norm(v: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", v).casefold().split())


@st.composite
def valid_lists(draw: st.DrawFn) -> list[dict[str, Any]]:
    offer_ids = draw(st.lists(ids, max_size=8, unique=True))
    seen: set[str] = set()
    out = []
    for oid in offer_ids:
        vs = draw(st.lists(values.filter(lambda v: _norm(v) and _norm(v) not in seen),
                           min_size=1, max_size=3, unique_by=_norm))
        seen.update(_norm(v) for v in vs)
        out.append({"id": oid, "values": vs})
    return out


@settings(max_examples=200)
@given(valid_lists())
def test_pc3_valid_lists_always_parse(offers: list[dict[str, Any]]) -> None:
    parsed = parse_web_offers({"offers": offers})
    assert parsed == tuple(WebOffer(o["id"], tuple(o["values"])) for o in offers)
    assert Policy.from_store(seeded(with_offers(offers))).web_offers == parsed


def _variant(value: str) -> st.SearchStrategy[str]:
    """Spellings of ``value`` that normalise alike: case, surrounding and repeated whitespace."""
    words = value.split()
    return st.builds(lambda case, left, right, gap: left + gap.join(
        w.upper() if case else w.lower() for w in words) + right,
        st.booleans(), st.sampled_from(["", " ", "\t"]), st.sampled_from(["", "  "]),
        st.sampled_from([" ", "  ", " "]))


@settings(max_examples=200)
@given(valid_lists().filter(bool), st.data())
def test_pc3_a_duplicate_normalised_value_is_always_refused(offers: list[dict[str, Any]],
                                                            data: st.DataObject) -> None:
    src = data.draw(st.sampled_from(offers))
    value = data.draw(st.sampled_from(src["values"]))
    dup = data.draw(_variant(value))
    if _norm(dup) != _norm(value) or len(dup) > MAX_VALUE_CHARS:
        dup = value  # the identical value is a duplicate too
    target = data.draw(st.sampled_from(offers))
    target["values"] = [*target["values"], dup]
    with pytest.raises(PolicySectionError):
        parse_web_offers({"offers": offers})
    with pytest.raises(PolicyError):
        Policy.to_items(with_offers(offers))


bad_ids = st.one_of(
    st.just(""), st.text(min_size=33, max_size=40, alphabet="abc"),
    st.text(min_size=1, max_size=32).filter(lambda s: not all(c in "abcdefghijklmnopqrstuvwxyz0123456789_"
                                                                for c in s)),
    st.integers(), st.none())


@settings(max_examples=200)
@given(valid_lists(), bad_ids, st.data())
def test_pc3_a_bad_id_is_always_refused(offers: list[dict[str, Any]], bad: Any,
                                        data: st.DataObject) -> None:
    at = data.draw(st.integers(0, len(offers)))
    offers.insert(at, {"id": bad, "values": ["☃ unique value not drawn"]})
    if any(_norm(v) == _norm("☃ unique value not drawn") for o in offers[:at] + offers[at + 1:]
           for v in o["values"]):
        return
    with pytest.raises(PolicySectionError) as err:
        parse_web_offers({"offers": offers})
    assert err.value.field == f"web_offers.offers[{at}].id"


# --------------------------------------------------------------------------- reason descriptions


def test_seed_descriptions_and_store_round_trip() -> None:
    a = Policy.from_dict(SEED)
    assert set(a.reason_descriptions) == {"card_dispute", "fraud_suspected"}
    store = seeded(SEED)
    assert store.get("policy", {"pk": "reason#card_dispute"})["description"] == \
        SEED["escalation"]["reasons"]["card_dispute"]["description"]
    assert "description" not in store.get("policy", {"pk": "reason#complaint"})
    assert Policy.from_store(store).reason_descriptions == a.reason_descriptions
    # A description doesn't change the route.
    assert a.escalation_route("card_dispute").queue_key == "disputes"
    assert a.escalation_route("card_dispute").priority == 50
    assert a.escalation_route("card_dispute").phrase_key == "escalate.dispute"
    assert a.queue_name("disputes", "es-US") == "OwlBank Disputes"


def test_description_is_optional() -> None:
    doc = copy.deepcopy(SEED)
    for row in doc["escalation"]["reasons"].values():
        row.pop("description", None)
    assert Policy.from_dict(doc).reason_descriptions == {}
    assert not any("description" in i for i in Policy.to_items(doc))


def test_description_at_the_limit_is_allowed() -> None:
    text = "a" * MAX_REASON_DESCRIPTION_CHARS
    assert parse_reason_description("complaint", text) == text


@pytest.mark.parametrize("text", [
    "", "   ", 7, ["a"],
    "a" * (MAX_REASON_DESCRIPTION_CHARS + 1),
    "two\nlines", "two\rlines", "line sep", " padded", "padded ", "tab\there", "zero​width",
    *[f"has {c} markup" for c in "{}<>[]*#`|~^_"],
])
def test_each_description_refusal(text: Any, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(PolicySectionError) as err:
        parse_reason_description("complaint", text)
    assert err.value.field == "escalation.reasons.complaint.description"
    doc = with_description("complaint", text)
    with pytest.raises(PolicyError):
        Policy.to_items(doc)
    with caplog.at_level(logging.ERROR, logger="owlbank.policy"):
        p = Policy.from_dict(doc)  # at run time: dropped, the rest of the policy loads
    assert "complaint" not in p.reason_descriptions
    assert "card_dispute" in p.reason_descriptions
    assert "escalation.reasons.complaint.description" in caplog.text
