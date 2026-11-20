"""Segment-traits, the rendering half, on the real data.

Every offer in ``data/policy.json`` (``web_offers``), rendered with ``data/prompt_catalog.json``
in every catalog language through :func:`owlbank.welcome.render_offer` with a sample first name,
gives a welcome that passes :func:`owlbank.welcome.check_welcome`. A bad phrase or template is
then caught here before a call silently falls back to today's welcome.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from owlbank.catalog import PromptCatalog, render
from owlbank.policy import Policy
from owlbank.welcome import GREETING_KEY, check_welcome, offer_key, render_offer

DATA = Path(__file__).resolve().parents[2] / "data"
CATALOG = PromptCatalog.from_dict(json.loads((DATA / "prompt_catalog.json").read_text()))
POLICY = Policy.from_dict(json.loads((DATA / "policy.json").read_text()))
NAMES = ("Thomas", "María", "Jo", "Anne-Marie")
CASES = [(o.id, lang) for o in POLICY.web_offers for lang in CATALOG.languages]


def test_the_seed_has_offers_and_languages() -> None:
    assert POLICY.web_offers, "data/policy.json lists no web offers"
    assert len(CATALOG.languages) >= 2


@pytest.mark.parametrize(("offer_id", "lang"), CASES, ids=[f"{o}-{lang}" for o, lang in CASES])
@pytest.mark.parametrize("first_name", NAMES)
def test_every_offer_renders_and_passes_check_welcome(offer_id: str, lang: str, first_name: str) -> None:
    text = render_offer(CATALOG, lang, first_name, offer_id)
    assert text is not None, f"{offer_key(offer_id)} in {lang} doesn't render a valid welcome"
    template = CATALOG.template(GREETING_KEY, lang)
    lead = render(template.partition("{offer}")[0], {"firstName": first_name}).strip()
    assert check_welcome(text, lead)
    assert CATALOG.template(offer_key(offer_id), lang).strip() in text
    assert first_name in text
