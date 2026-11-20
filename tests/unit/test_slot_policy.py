"""Slot settings."""

from __future__ import annotations

from typing import Any

import pytest

from owlbank.slot_policy import (
    FLOWS,
    AliasConflict,
    GatherInput,
    OptionDef,
    ResolutionSettings,
    SlotPolicyError,
    SlotSettings,
    alias_conflicts,
    gather_input,
    load_slot,
    load_slot_resolution,
    parse_slot,
    parse_slot_resolution,
)
from owlbank.slots.types import SlotDecl

ZIP_DEF: dict[str, Any] = {"slotType": "digits", "constraints": {"length": 5}, "promptKey": "factor.zip",
                           "labelKey": "label.zip", "sensitive": True, "maxRetries": 2, "channel": "dtmf"}
YES_NO: dict[str, Any] = {"slotType": "yes_no", "maxRetries": 1, "flow": "confirm"}
FIELDS = ("sensitive", "keypad_only", "barge_in")
DEFAULTS = {"sensitive": False, "keypad_only": False, "barge_in": True}
NON_BOOLEANS: list[Any] = [None, 0, 1, "true", "false", "", [], {}, [True], {"x": True}, 1.5]


def _factor(raw: dict[str, Any] | None = None) -> SlotSettings:
    return parse_slot("zip", raw or {}, factor_def=ZIP_DEF)


# --------------------------------------------------------------------------- strict


@pytest.mark.parametrize("field", FIELDS)
def test_absent_field_takes_its_default(field: str) -> None:
    assert getattr(_factor(), field) is DEFAULTS[field]
    assert getattr(parse_slot("yes_no", YES_NO, factor_def=None), field) is DEFAULTS[field]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_accepted(field: str, value: bool) -> None:
    assert getattr(_factor({field: value}), field) is value
    assert getattr(parse_slot("yes_no", {**YES_NO, field: value}, factor_def=None), field) is value


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", NON_BOOLEANS)
def test_non_boolean_is_refused_naming_the_field(field: str, value: Any) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        _factor({field: value})
    assert exc.value.field == f"slot#zip.{field}"
    assert repr(value) not in str(exc.value)  # never echoes the value


def test_factor_slot_takes_its_shape_from_the_factor() -> None:
    s = _factor({"sensitive": True})
    assert s.name == "zip" and s.is_factor and s.flow is None
    assert s.decl.slot_type == "digits" and dict(s.decl.constraints) == {"length": 5}
    assert s.decl.prompt_key == "factor.zip" and s.decl.label_key == "label.zip"
    assert s.decl.max_retries == 2


def test_non_factor_slot_carries_its_own_shape() -> None:
    s = parse_slot("yes_no", YES_NO, factor_def=None)
    assert not s.is_factor and s.flow == "confirm"
    assert s.decl == SlotDecl(slot_type="yes_no", max_retries=1)


@pytest.mark.parametrize("raw, field", [
    ({"slotType": "digits"}, "slot#zip.slotType"),          # shape lives in factors
    ({"flow": "confirm"}, "slot#zip.flow"),                 # a factor's flow is its level
    ({"keypadOnly": True}, "slot#zip.keypadOnly"),          # unknown field (a typo)
])
def test_factor_item_refuses_shape_flow_and_unknown_fields(raw: dict[str, Any], field: str) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        _factor(raw)
    assert exc.value.field == field


@pytest.mark.parametrize("raw, field", [
    ({"maxRetries": 1}, "slot#yes_no.slotType"),
    ({"slotType": "nope"}, "slot#yes_no.slotType"),
    ({"slotType": "yes_no", "maxRetries": -1}, "slot#yes_no.maxRetries"),
    ({"slotType": "yes_no", "maxRetries": True}, "slot#yes_no.maxRetries"),
    ({"slotType": "yes_no", "constraints": []}, "slot#yes_no.constraints"),
    ({"slotType": "yes_no", "promptKey": 3}, "slot#yes_no.promptKey"),
    ({"slotType": "yes_no", "flow": "lunch"}, "slot#yes_no.flow"),
])
def test_non_factor_item_validation(raw: dict[str, Any], field: str) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot("yes_no", raw, factor_def=None)
    assert exc.value.field == field


def test_item_must_be_an_object() -> None:
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot("zip", ["sensitive"], factor_def=ZIP_DEF)  # type: ignore[arg-type]
    assert exc.value.field == "slot#zip"


@pytest.mark.parametrize("slot_type", ["yes_no", "choice"])
def test_sensitive_is_allowed_on_yes_no_and_choice(slot_type: str) -> None:
    s = parse_slot("card", {"slotType": slot_type, "sensitive": True}, factor_def=None)
    assert s.sensitive is True


def test_flows_constant() -> None:
    assert FLOWS == ("login", "stepup", "confirm")


@pytest.mark.parametrize("raw", [{}, {"sensitive": True, "keypad_only": True, "barge_in": False}])
def test_to_doc_round_trips(raw: dict[str, Any]) -> None:
    s = _factor(raw)
    assert parse_slot("zip", s.to_doc(), factor_def=ZIP_DEF) == s
    y = parse_slot("yes_no", {**YES_NO, **raw}, factor_def=None)
    assert parse_slot("yes_no", y.to_doc(), factor_def=None) == y


def test_factor_to_doc_has_no_shape() -> None:
    assert _factor().to_doc() == {"sensitive": False, "keypad_only": False, "barge_in": True,
                                  "confirm": "when_inferred"}


# --------------------------------------------------------------------------- declinable


def test_declinable_defaults_to_false_and_is_accepted_on_every_slot() -> None:
    assert _factor().declinable is False
    assert parse_slot("yes_no", YES_NO, factor_def=None).declinable is False
    for value in (True, False):
        assert _factor({"declinable": value}).declinable is value
        assert parse_slot("yes_no", {**YES_NO, "declinable": value}, factor_def=None).declinable is value


@pytest.mark.parametrize("value", NON_BOOLEANS)
def test_non_boolean_declinable_is_refused_strict_and_loads_false(value: Any) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        _factor({"declinable": value})
    assert exc.value.field == "slot#zip.declinable"
    s, errors = load_slot("zip", {"declinable": value}, factor_def=ZIP_DEF)
    assert s is not None and s.declinable is False
    assert [e.field for e in errors] == ["slot#zip.declinable"]
    assert "loaded as false" in errors[0].message


def test_declinable_loads_and_round_trips() -> None:
    s, errors = load_slot("zip", {"declinable": True}, factor_def=ZIP_DEF)
    assert errors == [] and s is not None and s.declinable is True
    assert s.to_doc()["declinable"] is True
    assert parse_slot("zip", s.to_doc(), factor_def=ZIP_DEF) == s
    assert "declinable" not in _factor().to_doc()


# --------------------------------------------------------------------------- lenient


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", NON_BOOLEANS)
def test_load_fails_closed_with_one_error(field: str, value: Any) -> None:
    s, errors = load_slot("zip", {field: value}, factor_def=ZIP_DEF)
    assert s is not None and getattr(s, field) is True  # Fail closed
    for other in FIELDS:
        if other != field:
            assert getattr(s, other) is DEFAULTS[other]
    assert [e.field for e in errors] == [f"slot#zip.{field}"]


def test_load_valid_item_has_no_errors() -> None:
    s, errors = load_slot("zip", {"sensitive": False, "keypad_only": True, "barge_in": False},
                          factor_def=ZIP_DEF)
    assert errors == []
    assert s == _factor({"sensitive": False, "keypad_only": True, "barge_in": False})


def test_load_item_without_sensitive_is_not_sensitive() -> None:
    s, errors = load_slot("ssn4", {"keypad_only": False}, factor_def=ZIP_DEF)
    assert errors == [] and s is not None and s.sensitive is False


def test_load_factor_with_no_item_is_sensitive() -> None:
    s, errors = load_slot("ssn4", None, factor_def=ZIP_DEF)
    assert s is not None and s.is_factor and s.sensitive is True
    assert s.keypad_only is False and s.barge_in is True
    assert [e.field for e in errors] == ["slot#ssn4"]


def test_load_non_factor_with_no_item_cannot_load() -> None:
    s, errors = load_slot("yes_no", None, factor_def=None)
    assert s is None and [e.field for e in errors] == ["slot#yes_no"]


def test_load_non_factor_with_bad_shape_cannot_load() -> None:
    s, errors = load_slot("yes_no", {"slotType": "nope"}, factor_def=None)
    assert s is None and [e.field for e in errors] == ["slot#yes_no.slotType"]


def test_load_ignores_unknown_and_misplaced_fields_with_an_error() -> None:
    s, errors = load_slot("zip", {"keypadOnly": True, "flow": "confirm"}, factor_def=ZIP_DEF)
    assert s == _factor()
    assert sorted(e.field for e in errors) == ["slot#zip.flow", "slot#zip.keypadOnly"]


def test_load_bad_flow_gives_none() -> None:
    s, errors = load_slot("yes_no", {**YES_NO, "flow": 7}, factor_def=None)
    assert s is not None and s.flow is None
    assert [e.field for e in errors] == ["slot#yes_no.flow"]


def test_load_non_object_item_fails_closed() -> None:
    s, errors = load_slot("zip", "oops", factor_def=ZIP_DEF)  # type: ignore[arg-type]
    assert s is not None and (s.sensitive, s.keypad_only, s.barge_in) == (True, True, True)
    assert [e.field for e in errors] == ["slot#zip"]


def test_load_never_echoes_a_value() -> None:
    _, errors = load_slot("zip", {"sensitive": "SECRET-94105"}, factor_def=ZIP_DEF)
    assert errors and all("SECRET" not in str(e) and "94105" not in str(e) for e in errors)


# --------------------------------------------------------------------------- decisions


@pytest.mark.parametrize("is_factor, keypad_only, asf, expected", [
    (True, False, True, False),
    (True, False, False, True),     # the ceiling forces a factor keypad-only
    (True, True, True, True),
    (False, False, False, False),   # non-factor unaffected by the ceiling
    (False, True, True, True),
])
def test_gather_input(is_factor: bool, keypad_only: bool, asf: bool, expected: bool) -> None:
    s = (_factor({"keypad_only": keypad_only, "barge_in": False}) if is_factor
         else parse_slot("yes_no", {**YES_NO, "keypad_only": keypad_only, "barge_in": False}, factor_def=None))
    assert gather_input(s, asf) == GatherInput(keypad_only=expected, barge_in=False)


def test_slot_policy_error_is_a_value_error() -> None:
    err = SlotPolicyError("slot#zip.sensitive", "must be a boolean")
    assert isinstance(err, ValueError) and err.field == "slot#zip.sensitive"


# --------------------------------------------------------------------------- phase 2: mode, confirm, options

CARD: dict[str, Any] = {
    "slotType": "choice", "promptKey": "prompt.card", "labelKey": "label.card", "maxRetries": 2,
    "flow": "confirm",
    "options": [{"value": "red", "aliases": {"en-US": ["red"], "es-US": ["roja"]}},
                {"value": "purple", "aliases": {"en-US": ["purple"], "es-US": ["morada"]}}],
}
BAD_MODES: list[Any] = [None, "", "Assisted", "auto", 0, 1, True, False, [], ["assisted"], {"mode": "assisted"}, 1.5]


def _card(**over: Any) -> dict[str, Any]:
    return {**CARD, **over}


def test_phase2_fields_absent_take_their_defaults() -> None:
    for s in (_factor(), parse_slot("yes_no", YES_NO, factor_def=None), parse_slot("card", CARD, factor_def=None)):
        assert s.mode is None and s.confirm == "when_inferred"
    assert _factor().options == ()


@pytest.mark.parametrize("mode", ["deterministic", "assisted", "generative"])
def test_mode_is_accepted(mode: str) -> None:
    assert parse_slot("card", _card(mode=mode), factor_def=None).mode == mode
    if mode != "generative":
        assert _factor({"mode": mode}).mode == mode
        assert parse_slot("yes_no", {**YES_NO, "mode": mode}, factor_def=None).mode == mode


def test_mode_null_is_absent() -> None:
    assert _factor({"mode": None}).mode is None


@pytest.mark.parametrize("value", [v for v in BAD_MODES if v is not None])
def test_unknown_mode_is_refused_naming_the_field(value: Any) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        _factor({"mode": value})
    assert exc.value.field == "slot#zip.mode" and exc.value.mode_invalid
    # The message is the same for every value, so it can't echo one.
    assert str(exc.value) == "slot#zip.mode: must be one of ['deterministic', 'assisted', 'generative']"


@pytest.mark.parametrize("value", [v for v in BAD_MODES if v is not None])
def test_unknown_mode_loads_deterministic(value: Any) -> None:
    s, errors = load_slot("zip", {"mode": value}, factor_def=ZIP_DEF)
    assert s is not None and s.mode == "deterministic"
    assert [(e.field, e.mode_invalid) for e in errors] == [("slot#zip.mode", True)]


@pytest.mark.parametrize("name, raw, fd", [
    ("ssn4", {"sensitive": True, "mode": "generative"}, ZIP_DEF),               # sensitive digits
    ("zip", {"mode": "generative"}, ZIP_DEF),                                   # digits
    ("yes_no", {**YES_NO, "mode": "generative"}, None),                          # not a choice
    ("card", {**CARD, "sensitive": True, "mode": "generative"}, None),          # sensitive choice
])
def test_generative_is_refused_off_a_non_sensitive_choice(name: str, raw: dict[str, Any],
                                                          fd: dict[str, Any] | None) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot(name, raw, factor_def=fd)
    assert exc.value.field == f"slot#{name}.mode" and not exc.value.mode_invalid
    s, errors = load_slot(name, raw, factor_def=fd)
    assert s is not None and s.mode == "assisted"
    assert [(e.field, e.mode_invalid) for e in errors] == [(f"slot#{name}.mode", False)]


@pytest.mark.parametrize("confirm", ["when_inferred", "always", "never"])
def test_confirm_is_accepted(confirm: str) -> None:
    assert _factor({"confirm": confirm}).confirm == confirm
    assert parse_slot("card", _card(confirm=confirm), factor_def=None).confirm == confirm


@pytest.mark.parametrize("value", [None, "", "sometimes", "Never", True, 0, ["never"]])
def test_bad_confirm_is_refused_and_loads_always(value: Any) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        _factor({"confirm": value})
    assert exc.value.field == "slot#zip.confirm"
    s, errors = load_slot("zip", {"confirm": value}, factor_def=ZIP_DEF)
    assert s is not None and s.confirm == "always"
    assert [e.field for e in errors] == ["slot#zip.confirm"]


def test_options_parse() -> None:
    s = parse_slot("card", _card(options=[*CARD["options"], {"value": "rewards_2", "dtmf": "3"}]),
                   factor_def=None)
    assert s.options == (
        OptionDef("red", {"en-US": ("red",), "es-US": ("roja",)}),
        OptionDef("purple", {"en-US": ("purple",), "es-US": ("morada",)}),
        OptionDef("rewards_2", {}, "3"),
    )


@pytest.mark.parametrize("raw, fd, field", [
    ({"options": []}, ZIP_DEF, "slot#zip.options"),                          # a factor isn't a choice
    ({**YES_NO, "options": []}, None, "slot#yes_no.options"),                 # nor is yes/no
])
def test_options_only_on_a_choice_slot(raw: dict[str, Any], fd: dict[str, Any] | None, field: str) -> None:
    name = field.split("#")[1].split(".")[0]
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot(name, raw, factor_def=fd)
    assert exc.value.field == field


@pytest.mark.parametrize("options, field", [
    ("red", "slot#card.options"),
    (["red"], "slot#card.options[0]"),
    ([{"value": "Red"}], "slot#card.options[0].value"),
    ([{"value": "1red"}], "slot#card.options[0].value"),
    ([{"value": "r" * 33}], "slot#card.options[0].value"),
    ([{"value": 3}], "slot#card.options[0].value"),
    ([{"value": "red"}, {"value": "red"}], "slot#card.options[1].value"),
    ([{"value": "red", "label": "x"}], "slot#card.options[0].label"),
    ([{"value": "red", "dtmf": "12"}], "slot#card.options[0].dtmf"),
    ([{"value": "red", "dtmf": 1}], "slot#card.options[0].dtmf"),
    ([{"value": "red", "dtmf": "1"}, {"value": "blue", "dtmf": "1"}], "slot#card.options[1].dtmf"),
    ([{"value": "red", "aliases": ["red"]}], "slot#card.options[0].aliases"),
    ([{"value": "red", "aliases": {"en-US": "red"}}], "slot#card.options[0].aliases.en-US"),
    ([{"value": "red", "aliases": {"en-US": [""]}}], "slot#card.options[0].aliases.en-US[0]"),
    ([{"value": "red", "aliases": {"en-US": ["?!"]}}], "slot#card.options[0].aliases.en-US[0]"),
    ([{"value": "red", "aliases": {"en-US": [7]}}], "slot#card.options[0].aliases.en-US[0]"),
    ([{"value": "red", "aliases": {"english": ["red"]}}], "slot#card.options[0].aliases"),
])
def test_bad_options_are_refused_naming_the_field(options: Any, field: str) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot("card", _card(options=options), factor_def=None)
    assert exc.value.field == field


@pytest.mark.parametrize("a, b", [("red", "red"), ("Red!", "red"), ("the red one", "The  red, one"),
                                  ("rojá", "roja")])
def test_an_alias_shared_by_two_options_is_refused(a: str, b: str) -> None:
    options = [{"value": "red", "aliases": {"es-US": [a]}}, {"value": "purple", "aliases": {"es-US": [b]}}]
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot("card", _card(options=options), factor_def=None)
    assert exc.value.field == "slot#card.options.aliases.es-US"
    assert "purple" in exc.value.message and "red" in exc.value.message


def test_the_same_alias_in_two_languages_or_twice_in_one_option_is_fine() -> None:
    options = [{"value": "red", "aliases": {"en-US": ["roja", "roja"]}},
               {"value": "purple", "aliases": {"es-US": ["roja"]}}]
    assert len(parse_slot("card", _card(options=options), factor_def=None).options) == 2


def test_langs_restrict_the_alias_languages() -> None:
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot("card", CARD, factor_def=None, langs=("en-US",))
    assert exc.value.field == "slot#card.options[0].aliases"
    assert parse_slot("card", CARD, factor_def=None, langs=("en-US", "es-US")).options


def test_load_drops_bad_options_and_shared_aliases() -> None:
    options = [{"value": "red", "aliases": {"en-US": ["red", "card"]}},
               {"value": "Purple"},
               {"value": "rewards", "aliases": {"en-US": ["Card", "rewards"]}, "dtmf": "x"}]
    s, errors = load_slot("card", _card(options=options), factor_def=None)
    assert s is not None
    assert s.options == (OptionDef("red", {"en-US": ("red",)}), OptionDef("rewards", {"en-US": ("rewards",)}))
    assert [e.field for e in errors] == ["slot#card.options[1].value", "slot#card.options[2].dtmf",
                                         "slot#card.options.aliases.en-US"]


def test_load_ignores_options_on_a_non_choice_slot() -> None:
    s, errors = load_slot("yes_no", {**YES_NO, "options": [{"value": "a"}]}, factor_def=None)
    assert s is not None and s.options == ()
    assert [e.field for e in errors] == ["slot#yes_no.options"]


def test_phase2_errors_never_echo_a_value() -> None:
    secret = "SECRET-94105"
    raw = _card(mode=secret, confirm=secret,
                options=[{"value": secret}, {"value": "red", "dtmf": secret, "aliases": {"en-US": [secret * 9]}}])
    _, errors = load_slot("card", raw, factor_def=None)
    assert len(errors) == 5
    assert all("SECRET" not in str(e) and "94105" not in str(e) for e in errors)


def test_to_doc_round_trips_phase2_fields() -> None:
    s = parse_slot("card", _card(mode="generative", confirm="always",
                                 options=[*CARD["options"], {"value": "rewards", "dtmf": "9"}]), factor_def=None)
    assert parse_slot("card", s.to_doc(), factor_def=None) == s
    assert s.to_doc()["options"][2] == {"value": "rewards", "dtmf": "9"}


# --------------------------------------------------------------------------- alias_conflicts


def test_alias_conflicts_checks_spoken_names() -> None:
    options = [OptionDef("red", {"en-US": ("crimson",)}), OptionDef("purple", {"en-US": ("violet",)})]
    names = {"red": {"en-US": "OwlBank Red Card"}, "purple": {"en-US": "owlbank red card!"}}
    assert alias_conflicts(options, names, ["en-US"]) == [
        AliasConflict("en-US", "owlbank red card", ("purple", "red"))]
    assert alias_conflicts(options, names, ["es-US"]) == []
    clash = {"red": {"en-US": "OwlBank Red Card"}, "purple": {"en-US": "Crimson"}}
    assert [c.options for c in alias_conflicts(options, clash, ["en-US"])] == [("purple", "red")]
    assert alias_conflicts(options, {"red": {"en-US": "OwlBank Red Card"}}, ["en-US"]) == []


# --------------------------------------------------------------------------- slot_resolution


def test_slot_resolution_defaults() -> None:
    assert parse_slot_resolution({}) == ResolutionSettings() == ResolutionSettings(
        default_mode="assisted", flows={}, generative_enabled=False, resolver_budget_ms=2500)


def test_slot_resolution_valid_values() -> None:
    r = parse_slot_resolution({"default_mode": "deterministic", "flows": {"confirm": {"mode": "generative"},
                                                                           "login": {"mode": "assisted"}},
                               "generative_enabled": True, "resolver_budget_ms": 500})
    assert r == ResolutionSettings("deterministic", {"login": "assisted", "confirm": "generative"}, True, 500)
    assert parse_slot_resolution(r.to_doc()) == r
    assert load_slot_resolution(r.to_doc()) == (r, [])


@pytest.mark.parametrize("raw, field", [
    ({"default_mode": "fast"}, "slot_resolution.default_mode"),
    ({"default_mode": None}, "slot_resolution.default_mode"),
    ({"flows": []}, "slot_resolution.flows"),
    ({"flows": {"lunch": {"mode": "assisted"}}}, "slot_resolution.flows"),
    ({"flows": {"login": "assisted"}}, "slot_resolution.flows.login.mode"),
    ({"flows": {"login": {"mode": "x"}}}, "slot_resolution.flows.login.mode"),
    ({"flows": {"login": {"mode": "assisted", "x": 1}}}, "slot_resolution.flows.login.mode"),
    ({"generative_enabled": "yes"}, "slot_resolution.generative_enabled"),
    ({"resolver_budget_ms": 499}, "slot_resolution.resolver_budget_ms"),
    ({"resolver_budget_ms": 4001}, "slot_resolution.resolver_budget_ms"),
    ({"resolver_budget_ms": True}, "slot_resolution.resolver_budget_ms"),
    ({"resolver_budget_ms": 1500.0}, "slot_resolution.resolver_budget_ms"),
    ({"ceiling": "generative"}, "slot_resolution.ceiling"),  # never settable from the table
])
def test_slot_resolution_refuses_naming_the_field(raw: dict[str, Any], field: str) -> None:
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot_resolution(raw)
    assert exc.value.field == field


def test_slot_resolution_must_be_an_object() -> None:
    with pytest.raises(SlotPolicyError):
        parse_slot_resolution(["assisted"])


def test_load_slot_resolution_fails_closed() -> None:
    r, errors = load_slot_resolution(None)
    assert r.default_mode == "deterministic"
    assert [(e.field, e.mode_invalid) for e in errors] == [("slot_resolution.default_mode", True)]
    r, errors = load_slot_resolution({"flows": {"login": {"mode": 3}, "lunch": {"mode": "assisted"}},
                                      "generative_enabled": 1, "resolver_budget_ms": 99})
    assert r == ResolutionSettings("deterministic", {"login": "deterministic"}, False, 2500)
    assert [(e.field, e.mode_invalid) for e in errors] == [
        ("slot_resolution.default_mode", True), ("slot_resolution.flows.login.mode", True),
        ("slot_resolution.flows", False), ("slot_resolution.generative_enabled", False),
        ("slot_resolution.resolver_budget_ms", False)]


@pytest.mark.parametrize("body,field", [
    ({"resolver_budget_ms": 99}, "resolver_budget_ms"),
    ({"generative_enabled": "yes"}, "generative_enabled"),
    ({"flows": {"login": "assisted"}}, "flows.login.mode"),
])
def test_slot_resolution_messages_say_loaded_as_only_when_lenient(body: dict, field: str) -> None:
    """A strict parse refuses the item, so its message never says what it "loaded as"; the
    lenient load does."""
    with pytest.raises(SlotPolicyError) as exc:
        parse_slot_resolution({"default_mode": "assisted", **body})
    assert exc.value.field == f"slot_resolution.{field}" and "loaded as" not in str(exc.value)
    _, errors = load_slot_resolution({"default_mode": "assisted", **body})
    assert [e.field for e in errors] == [f"slot_resolution.{field}"]
    assert "loaded as" in errors[0].message
