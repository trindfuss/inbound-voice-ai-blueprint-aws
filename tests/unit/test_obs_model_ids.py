"""Model IDs are not customer data: the fields that carry them keep a well-formed
ID as it is, so a dated ID survives the scrubber and the PreferredModelUnavailable metric filter can
match it; anything else in those fields is still scrubbed."""

from __future__ import annotations

import io
import json
import logging

import pytest

from owlbank import obs

DATED = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
IDS = [
    DATED,
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-5-5",
    "global.anthropic.claude-sonnet-5-5",
    "eu.anthropic.claude-sonnet-5-5",
]


def test_the_scrubber_alone_does_rewrite_a_dated_id():
    assert obs.scrub(DATED) != DATED  # the premise: the exemption is what keeps it


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize("field", sorted(obs.MODEL_ID_FIELDS - {"models", "global_entries"}))
def test_a_model_id_in_a_model_field_is_kept(field, model_id):
    assert obs.scrub_field(field, model_id) == model_id


def test_a_list_field_is_exempt_element_wise():
    out = obs.scrub_field("models", [*IDS, "call +14155550123 now"])
    assert out[:-1] == IDS
    assert "4155550123" not in out[-1]


@pytest.mark.parametrize("value", ["+14155550123", "caller said 4155550123 ok", "Bearer abcdef0123456789abcdef",
                                   "us.anthropic.claude-haiku-4-5-20251001-v1:0 and 4155550123", ""])
def test_a_non_id_value_in_a_model_field_is_still_scrubbed(value):
    assert obs.scrub_field("model_id", value) == obs.scrub(value)
    assert obs.scrub_field("model_id", value) != value or value == ""


@pytest.mark.parametrize("value", ["415.555.0100", "4111.1111.1111.1111", "10.0.0.1", "us.4111.1111",
                                   "us.anthropic.claude-haiku-5-5-v1:0\n"])
def test_a_dotted_number_is_not_taken_for_a_model_id(value):
    # Only letter-led segments pass, and the whole value must match: these get the normal scrub
    # (including the per-call exact-value one), never the exemption.
    seen: list[str] = []
    obs.scrub_field("model_id", value, lambda text: seen.append(text) or "[scrubbed]")
    assert seen == [value]


def test_the_exemption_is_only_for_the_model_fields():
    assert obs.scrub_field("detail", DATED) == obs.scrub(DATED) != DATED


def _logged(**fields) -> dict:
    stream = io.StringIO()
    obs.configure_logging("unit", stream=stream)
    obs.emit(logging.getLogger("tac_bridge.test"), logging.ERROR, "model_unavailable", **fields)
    return json.loads(stream.getvalue().splitlines()[-1])


def test_a_dated_preferred_id_is_logged_as_configured_so_the_metric_filter_matches(monkeypatch):
    monkeypatch.delenv(obs.LOG_LEVEL_ENV, raising=False)
    rec = _logged(entry=0, model_id=DATED, error_code="ResourceNotFoundException", preferred=True,
                  scope="process")
    # infra/owlbank_infra/model_failover.py: string_value("$.model_id", "=", cfg.models[0].id)
    assert rec["model_id"] == DATED
    assert rec["preferred"] is True


def test_failover_fields_keep_their_ids_through_the_whole_pipeline(monkeypatch):
    monkeypatch.delenv(obs.LOG_LEVEL_ENV, raising=False)
    rec = _logged(from_model="us.anthropic.claude-haiku-5-5", to_model=DATED, models=IDS,
                  preferred=IDS[0], global_entries=[IDS[3]])
    assert rec["to_model"] == DATED and rec["models"] == IDS
    assert rec["preferred"] == IDS[0] and rec["global_entries"] == [IDS[3]]
