"""Digit read-back in the front door.

ZIP is not sensitive and ``confirm: when_inferred`` in the seed: an inferred ZIP is read back with
``slot.confirm_digits``, answered as yes/no in the same Gather shape as the ZIP's own (``keypad_only``, ``barge_in`` with Path B). A deterministic yes verifies; a model yes is unclear; no re-asks; a fresh value is a fresh answer. A sensitive slot is never read back.
"""

from __future__ import annotations

import json
from typing import Any

from owlbank.policy import Policy
from owlbank.slots import FakeResolver, ResolverOutput
from tests.component.test_frontdoor_caller_id_trust import Z
from tests.component.test_frontdoor_support import DATA, THOMAS, Harness, build, relay, says, sid

MESSY = f"{'2' * 5}, oh, no actually it is {Z}"
READ_BACK = "Did you mean " + ", ".join(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][int(c)] for c in Z)
ZIP_PROMPT = "Please say or enter your five-digit ZIP code. If you can't, press pound."


def _h(*outs: ResolverOutput, **slot_changes: Any) -> Harness:
    h = build(resolver=FakeResolver(list(outs)))
    if slot_changes:
        raw = json.loads((DATA / "policy.json").read_text())
        raw["slots"]["zip"].update(slot_changes)
        h.app.d.policy = Policy.from_dict(raw)
    return h


def _speak(h: Harness, call: str, text: str):
    return h.xml("factor", call, SpeechResult=text, Confidence="0.9")


def _read_back(h: Harness, call: str):
    h.start(call, THOMAS)
    return _speak(h, call, MESSY)


def verifies(h: Harness) -> int:
    return h.lam.ops().count("verify")


def test_zip_with_a_filler_oh_is_read_back_then_yes_verifies() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z))
    call = sid(1)
    root = _read_back(h, call)
    (text,) = says(root)
    assert text.startswith("Did you mean ") and verifies(h) == 0
    g = root.find("Gather")
    assert g is not None and g.get("input") == "dtmf speech" and g.get("numDigits") is None
    assert g.get("partialResultCallback") is None and g.get("hints") is None
    assert h.session(call).flow["pending"]["kind"] == "confirm_digits"
    assert Z not in json.dumps(h.session(call).flow)          # sealed
    root = _speak(h, call, "yes")
    assert verifies(h) == 1 and relay(root) is not None and h.state(call).level == 1


def test_keypad_one_is_yes_to_the_read_back() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z))
    call = sid(2)
    _read_back(h, call)
    root = h.xml("factor", call, Digits="1")
    assert verifies(h) == 1 and relay(root) is not None


def test_no_re_asks_the_zip_with_one_try_used() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z))
    call = sid(3)
    _read_back(h, call)
    root = _speak(h, call, "no")
    assert says(root) == [ZIP_PROMPT] and verifies(h) == 0
    flow = h.session(call).flow
    assert flow["pending"] is None and flow["tries"]["zip"] == 2  # the read-back and the "no"


def test_a_fresh_value_answers_the_read_back() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z))
    call = sid(4)
    _read_back(h, call)
    root = _speak(h, call, f"no, it's {' '.join(Z)}")
    assert verifies(h) == 1 and relay(root) is not None


def test_a_model_yes_is_unclear_and_the_question_is_asked_again() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z), ResolverOutput(kind="none", reply="yes"))
    call = sid(5)
    _read_back(h, call)
    root = _speak(h, call, "go ahead with that one")
    lines = says(root)
    assert lines[0] == "Please say yes or no." and lines[1].startswith("Did you mean ")
    assert verifies(h) == 0 and h.session(call).flow["pending"]["kind"] == "confirm_digits"


def test_a_read_back_follows_the_zips_barge_in_setting_path_b() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z), barge_in=False)
    call = sid(6)
    root = _read_back(h, call)
    tags = [child.tag for child in root]
    assert tags[0] in ("Say", "Play") and tags[-1] == "Gather"
    assert len(root.find("Gather")) == 0                       # nothing nested (Path B)


def test_confirm_always_reads_back_a_model_exact_zip() -> None:
    h = _h(ResolverOutput(kind="exact", value=Z), confirm="always")
    call = sid(7)
    (text,) = says(_read_back(h, call))
    assert text.startswith("Did you mean ")


def test_confirm_never_accepts_an_inferred_zip() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z), confirm="never")
    call = sid(8)
    root = _read_back(h, call)
    assert verifies(h) == 1 and relay(root) is not None


def test_a_sensitive_zip_is_never_read_back() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z), sensitive=True)
    call = sid(9)
    root = _read_back(h, call)
    assert says(root) == ["Please enter the 5 digits of your ZIP code on your keypad."]
    prompts = [e.get("rendered_text") or "" for e in h.journal(call) if e["type"] == "prompt"]
    assert not any(t.startswith("Did you mean") for t in prompts)


def test_the_read_back_is_journaled_as_asked_then_answered() -> None:
    h = _h(ResolverOutput(kind="inferred", value=Z))
    call = sid(10)
    _read_back(h, call)
    _speak(h, call, "yes")
    events = [e for e in h.journal(call) if e["type"] == "slot_resolution"]
    assert [(e["match"], e["asked"], e["answer"]) for e in events] == [
        ("inferred", "confirm_digits", "none"), ("exact", "confirm_digits", "yes")]
    assert events[0]["value"] == Z and "value" not in events[1]
