"""The Gather shape, the ``yes_no`` slot and ``slot_input``.

**Characterization.** Every factor and confirm ``<Gather>`` the front door renders was
captured as golden TwiML *before* the Gather-shape change, for both languages, with early finish
on and off and with spoken factors on and off. Every seed slot is ``barge_in: true``, so each
response must stay byte-identical to its golden file. Regenerate only when a TwiML change is
intended and reviewed: ``OWLBANK_UPDATE_GOLDEN=1 uv run pytest
tests/component/test_frontdoor_gather_input.py``.

The Gather id is random; :func:`scenario` fixes it with a seeded generator, so the files
hold real ids and the comparison is exact.

**Path B.** ``bargeIn="false"`` doesn't stop a key press, so a ``barge_in: false`` slot plays
its prompt verbs before a ``<Gather>`` that nests none, with the same attributes.

Components: the fake store and ``FakeResolver`` (``test_frontdoor_support.build``)."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lambdas.frontdoor import app as app_module
from lambdas.frontdoor import handler as fd_handler
from lambdas.frontdoor.calls import FakeCallUpdater
from owlbank import callstate
from owlbank.policy import Policy
from owlbank.segments import clear_sensitive
from tests.component.test_frontdoor_caller_id_trust import confirm_handoff
from tests.component.test_frontdoor_support import (
    DATA, OTP, TEST_FACTORS, THOMAS, Harness, build, make_event, sid,
)

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).parent / "golden" / "gather_input"
UPDATE = os.environ.get("OWLBANK_UPDATE_GOLDEN") == "1"
T = TEST_FACTORS["C-1001"]
ZIP_WORDS = " ".join(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                      "nine"][int(c)] for c in T["zip"])

LANGS = {"en": "", "es": "2"}
COMBOS = [(lang, asf, early) for lang in LANGS for asf in (True, False) for early in (True, False)]
#: Which slot each scenario step's Gather asks.
STEP_SLOT = {"zip_ask": "zip", "zip_retry": "zip", "zip_keypad_reminder": "zip",
             "ssn4_ask": "ssn4", "otp_ask": "otp", "confirm_ask": "yes_no", "confirm_retry": "yes_no"}
FACTORS = ("zip", "ssn4", "otp")
PROMPT_TAGS = {"Say", "Play", "Pause"}
SPEECH_ATTRS = ("speechTimeout", "speechModel", "hints", "partialResultCallback", "language")

SlotChanges = Mapping[str, Mapping[str, Any]]


# --------------------------------------------------------------------------- scenario


class _SeededChoice:
    """Stands in for ``secrets`` in the front door module: a reproducible ``choice``."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choice(self, seq: Any) -> Any:
        return self._rng.choice(seq)


def policy_doc(slots: SlotChanges | None = None) -> dict[str, Any]:
    """The seed policy with ``slots`` merged into its ``slots`` section."""
    doc: dict[str, Any] = json.loads((DATA / "policy.json").read_text())
    for name, fields in (slots or {}).items():
        doc["slots"][name].update(fields)
    return doc


def make(asf: bool, early: bool, slots: SlotChanges | None = None) -> Harness:
    h = build(config_kw={"accept_spoken_factors": asf, "early_digits": early},
              calls=FakeCallUpdater() if early else None)
    if slots:
        h.app.d.policy = Policy.from_dict(policy_doc(slots))
    return h


def cleanup(h: Harness) -> None:
    for c in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(c)


def _last_gather(h: Harness, since: int) -> str:
    bodies = [b for b in h.bodies[since:] if "<Gather" in b]
    assert bodies, "no Gather in the responses"
    return bodies[-1]


def scenario(lang: str, asf: bool, early: bool, slots: SlotChanges | None = None,
             harness: list[Harness] | None = None) -> dict[str, str]:
    """Drive one call through every factor and the confirmation (and, when ZIP is keypad-only, a
    second call for the keypad reminder) and return each Gather response body by step name.
    ``harness`` (optional) receives the harness used."""
    h = make(asf, early, slots)
    if harness is not None:
        harness.append(h)
    out: dict[str, str] = {}
    call = sid(1)
    digit = LANGS[lang]
    try:
        with mock.patch.object(app_module, "_stdlib_secrets", _SeededChoice(7)):
            n = len(h.bodies)
            h.start(call, THOMAS, lang_digit=digit)
            out["zip_ask"] = _last_gather(h, n)
            n = len(h.bodies)
            h.factor(call, "12")  # too short: asked again
            out["zip_retry"] = _last_gather(h, n)
            h.factor(call, T["zip"])
            h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
            n = len(h.bodies)
            h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})
            out["ssn4_ask"] = _last_gather(h, n)
            n = len(h.bodies)
            h.factor(call, T["ssn4"])
            out["otp_ask"] = _last_gather(h, n)
            h.factor(call, OTP)
            assert h.state(call).level == 3
            callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})
            n = len(h.bodies)
            confirm_handoff(h, call)
            out["confirm_ask"] = _last_gather(h, n)
            n = len(h.bodies)
            h.xml("factor", call, Digits="9")  # neither yes nor no: asked again
            out["confirm_retry"] = _last_gather(h, n)
            zip_keypad = (slots or {}).get("zip", {}).get("keypad_only", False) or not asf
            if zip_keypad:
                other = sid(2)
                h.start(other, THOMAS, lang_digit=digit)
                n = len(h.bodies)
                h.xml("factor", other, SpeechResult=ZIP_WORDS, Confidence="0.9")
                out["zip_keypad_reminder"] = _last_gather(h, n)
    finally:
        cleanup(h)
    return out


def golden_name(lang: str, asf: bool, early: bool, step: str) -> str:
    return f"{lang}-spoken_{'on' if asf else 'off'}-early_{'on' if early else 'off'}-{step}.xml"


def golden(lang: str, asf: bool, early: bool, step: str) -> str:
    return (GOLDEN / golden_name(lang, asf, early, step)).read_text(encoding="utf-8").rstrip("\n")


def parse(body: str) -> ET.Element:
    return ET.fromstring(body.encode())


def gather_of(root: ET.Element) -> ET.Element:
    g = root.find("Gather")
    assert g is not None, ET.tostring(root)
    return g


def assert_keypad_only(body: str) -> None:
    """``input="dtmf"``, no speech attributes, hints or partial results; no Gather id."""
    g = gather_of(parse(body))
    assert g.get("input") == "dtmf", body
    for attr in SPEECH_ATTRS:
        assert attr not in g.attrib, (attr, body)
    assert "?" not in (g.get("action") or ""), body


def assert_path_b(body: str, reference: str) -> None:
    """Path B: ``body`` is ``reference`` (the ``barge_in: true`` rendering) with the
    Gather's nested prompts moved before it; nothing else changes."""
    root, ref = parse(body), parse(reference)
    ref_gather = gather_of(ref)
    tags = [c.tag for c in root]
    assert tags[-1] == "Gather" and set(tags[:-1]) <= PROMPT_TAGS, tags
    g = root[-1]
    assert len(g) == 0, ET.tostring(g)  # no prompt nested
    assert g.attrib == ref_gather.attrib  # same timeouts, input, action...
    assert "bargeIn" not in g.attrib
    before = [ET.tostring(c) for c in root[:-1]]
    assert before == [ET.tostring(c) for c in ref_gather]
    assert [c.tag for c in ref] == ["Gather"]


# --------------------------------------------------------------------------- characterization


@pytest.mark.parametrize("lang, asf, early", COMBOS)
def test_barge_in_true_gathers_match_the_golden_twiml(lang: str, asf: bool, early: bool) -> None:
    out = scenario(lang, asf, early)
    for step, body in out.items():
        path = GOLDEN / golden_name(lang, asf, early, step)
        if UPDATE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body + "\n", encoding="utf-8")
        assert path.exists(), f"missing golden file {path.name}"
        assert body + "\n" == path.read_text(encoding="utf-8"), path.name


def test_every_golden_file_is_exercised() -> None:
    expected = {golden_name(lang, asf, early, step) for lang, asf, early in COMBOS
                for step in scenario(lang, asf, early)}
    assert {p.name for p in GOLDEN.glob("*.xml")} == expected


# --------------------------------------------------------------------------- keypad-only


@pytest.mark.parametrize("lang", LANGS)
def test_a_keypad_only_slot_renders_a_keypad_only_gather_and_never_finishes_early(lang: str) -> None:
    holder: list[Harness] = []
    out = scenario(lang, True, True, {"zip": {"keypad_only": True}}, harness=holder)
    for step in ("zip_ask", "zip_retry", "zip_keypad_reminder"):
        assert_keypad_only(out[step])
    # The other factors keep their spoken, early-finish Gathers (asf on).
    assert gather_of(parse(out["ssn4_ask"])).get("partialResultCallback")
    h = holder[0]
    assert not h.app._early_eligible(h.app._slot("zip"))  # noqa: SLF001
    assert h.app._early_eligible(h.app._slot("ssn4"))  # noqa: SLF001


def test_a_keypad_only_slot_gets_no_gather_id_so_partials_are_stale(caplog) -> None:
    caplog.set_level(logging.INFO)
    h = make(True, True, {"zip": {"keypad_only": True}})
    try:
        h.start(sid(5), THOMAS)
        assert "gather" not in (h.session(sid(5)).flow or {})
        resp = h.post("partial?g=" + "a" * 16, sid(5), UnstableSpeechResult="nine four",
                      SequenceNumber="1")
        assert resp["statusCode"] == 204
        assert [r.outcome for r in caplog.records  # type: ignore[attr-defined]
                if getattr(r, "event", None) == "factor_partial"] == ["stale"]
        assert h.app.d.calls.updates == []  # type: ignore[union-attr]
    finally:
        cleanup(h)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("early", [True, False])
def test_spoken_factors_off_makes_every_factor_keypad_only(lang: str, early: bool) -> None:
    out = scenario(lang, False, early)
    for step, body in out.items():
        if STEP_SLOT[step] in FACTORS:
            assert_keypad_only(body)
        else:
            assert gather_of(parse(body)).get("input") == "dtmf speech"


def test_a_keypad_only_slot_reminds_a_spoken_answer_to_use_the_keypad() -> None:
    """The keypad reminder follows the effective keypad-only, not ``sensitive``: a
    non-sensitive keypad-only ZIP still asks for the keypad."""
    h = make(True, False, {"zip": {"keypad_only": True, "sensitive": False}})
    try:
        h.start(sid(6), THOMAS)
        root = h.xml("factor", sid(6), SpeechResult=ZIP_WORDS, Confidence="0.9")
        assert [e.text for e in root.iter("Say")][0] == "For your security, please use your keypad."
        assert "verify" not in h.lam.ops()
    finally:
        cleanup(h)


# --------------------------------------------------------------------------- barge_in: false (path B)


@pytest.mark.parametrize("lang, asf, early", COMBOS)
def test_barge_in_false_plays_the_prompts_before_an_empty_gather(lang: str, asf: bool,
                                                                 early: bool) -> None:
    off = {name: {"barge_in": False} for name in (*FACTORS, "yes_no")}
    out = scenario(lang, asf, early, off)
    for step, body in out.items():
        assert_path_b(body, golden(lang, asf, early, step))


# --------------------------------------------------------------------------- yes_no


def _store_policy(h: Harness) -> Policy:
    for item in Policy.to_items(policy_doc()):
        h.store.put("policy", item)
    policy = Policy.from_store(h.store, ttl_s=3600)
    h.app.d.policy = policy
    return policy


def test_yes_no_reads_its_settings_from_the_slot_item() -> None:
    h = make(True, False)
    policy = _store_policy(h)
    call = sid(7)
    try:
        h.start(call, THOMAS)
        h.factor(call, T["zip"])
        h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
        h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})
        h.factor(call, T["ssn4"])
        h.factor(call, OTP)
        callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})
        before = confirm_handoff(h, call)
        assert gather_of(before).get("input") == "dtmf speech"
        item = h.store.get("policy", {"pk": "slot#yes_no"})
        assert item is not None
        h.store.put("policy", {**item, "keypad_only": True, "barge_in": False})
        policy.refresh()
        root = h.xml("factor", call, Digits="9")  # asked again, with the edited settings
        assert [c.tag for c in root][-1] == "Gather" and len(root[-1]) == 0  # path B
        g = root[-1]
        assert g.get("input") == "dtmf" and "speechTimeout" not in g.attrib
        assert policy.slot("yes_no").keypad_only is True
    finally:
        cleanup(h)


def test_a_missing_yes_no_slot_takes_the_error_path() -> None:
    """A table without ``slot#yes_no`` leaves the slot out (lenient load); the confirmation then
    can't be asked and the call gets the apology and fallback, never a crash."""
    doc = policy_doc()
    del doc["slots"]["yes_no"]
    h = make(True, False)
    h.app.d.policy = Policy.from_dict(doc)
    call = sid(8)
    try:
        h.start(call, THOMAS)
        h.factor(call, T["zip"])
        h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
        h.handoff(call, {"reason": "step-up", "tool": "make_payment", "journal": "jr-2"})
        h.factor(call, T["ssn4"])
        h.factor(call, OTP)
        callstate.mutate(h.store, call, lambda s: {"pending_result": None, "pending_tool": None})
        root = confirm_handoff(h, call)
        assert root.find("Gather") is None
        assert root.find("Enqueue") is not None or root.find("Hangup") is not None
    finally:
        cleanup(h)


def test_confirm_slot_constant_is_gone() -> None:
    """``grep CONFIRM_SLOT src`` finds nothing."""
    hits = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*")
            if p.is_file() and p.suffix in (".py", ".js", ".json", ".md")
            and "CONFIRM_SLOT" in p.read_text(encoding="utf-8", errors="replace")]
    assert hits == []


# --------------------------------------------------------------------------- slot_input


@contextmanager
def _cold_start(monkeypatch: pytest.MonkeyPatch, h: Harness) -> Iterator[None]:
    monkeypatch.setattr(fd_handler, "_app", None)
    monkeypatch.setattr(fd_handler, "build_app", lambda env=None: h.app)
    try:
        yield
    finally:
        cleanup(h)


def _slot_input(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == "slot_input"]


def test_slot_input_is_logged_once_per_cold_start(monkeypatch: pytest.MonkeyPatch,
                                                  caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    h = make(False, False, {"yes_no": {"barge_in": False}})
    with _cold_start(monkeypatch, h):
        for n in (1, 2):
            fd_handler.handler(make_event("incoming", {"CallSid": sid(n), "From": THOMAS,
                                                       "AccountSid": "AC" + "b" * 32}), None)
    (rec,) = _slot_input(caplog)
    assert rec.levelno == logging.INFO
    assert rec.acceptSpokenFactors is False  # type: ignore[attr-defined]
    assert rec.slots == {  # type: ignore[attr-defined]
        "zip": {"keypad_only": True, "barge_in": True},
        "ssn4": {"keypad_only": True, "barge_in": True},
        "otp": {"keypad_only": True, "barge_in": True},
        "phone": {"keypad_only": False, "barge_in": True},  # not a factor: speech stays on
        "yes_no": {"keypad_only": False, "barge_in": False},
        "card": {"keypad_only": False, "barge_in": True},
    }


def test_slot_input_carries_names_and_booleans_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    h = make(True, False, {"zip": {"keypad_only": True}})
    try:
        fd_handler.log_slot_input(h.app)
    finally:
        cleanup(h)
    (rec,) = _slot_input(caplog)
    assert rec.acceptSpokenFactors is True  # type: ignore[attr-defined]
    slots = copy.deepcopy(rec.slots)  # type: ignore[attr-defined]
    assert set(slots) == {"zip", "ssn4", "otp", "phone", "yes_no", "card"}
    for fields in slots.values():
        assert set(fields) == {"keypad_only", "barge_in"}
        assert all(isinstance(v, bool) for v in fields.values())
    assert slots["zip"]["keypad_only"] is True and slots["ssn4"]["keypad_only"] is False


def test_slot_input_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    h = make(True, False)

    class Broken:
        def slots(self) -> tuple[str, ...]:
            raise RuntimeError("policy unavailable")

    h.app.d.policy = Broken()  # type: ignore[assignment]
    fd_handler.log_slot_input(h.app)
    assert [getattr(r, "event", None) for r in caplog.records] == ["slot_input_failed"]
