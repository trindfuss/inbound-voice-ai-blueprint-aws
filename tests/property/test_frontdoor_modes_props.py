"""Front-door properties for the resolution modes.

* With every effective mode ``deterministic`` (by the console's ``default_mode`` or by
  ``slot_mode_ceiling``), over ``login``, ``stepup`` and ``confirm`` and any Gather inputs, the
  resolver is never called and the Bedrock client factory is never called.
* For any sequence of policy edits and ceiling changes during a flow, each
  answer's mode equals ``restrict(frozen, ceiling_now)``.
* For any utterance and any ``FakeResolver`` output, a ZIP sent to the auth
  service from a spoken answer is grounded in the words.
* In ``assisted``, no resolver field reaches TwiML or a prompt, and a
  read-back speaks only a grounded, non-sensitive digit value.
* The ``ResolverRequest`` built from a session holds no collected value,
  record field or token; only ``utterance`` carries the caller's words.
* The read-back Gather follows the asked slot's settings, and Path B
  applies for ``barge_in: false``; a keypad-only slot gets no read-back and no model call.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any

from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

import lambdas.frontdoor.slotflow as fd_slotflow
from lambdas.frontdoor import handler
from owlbank import segments
from owlbank.policy import Policy
from owlbank.slots import BedrockResolver, FakeResolver, ResolverOutput, client_for_budget, normalize
from owlbank.slots.grounding import digit_stream, is_ordered_subsequence
from owlbank.slots.modes import restrict
from owlbank.slots.resolver import request_payload
from tests.component.test_frontdoor_caller_id_trust import S, Z, confirm_handoff, to_level3
from tests.component.test_frontdoor_support import DATA, OTP, THOMAS, Harness, build, sid

PROPS = settings(max_examples=60, deadline=None,
                 suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
MODES = ("deterministic", "assisted", "generative")
SEED = json.loads((DATA / "policy.json").read_text())
WORDS = ["um", "yes", "no", "agent", "repeat", "my", "zip", "is", "what's", "balance", "nine",
         "four", "one", "oh", "five", "two", "three", "actually", "please", "the", "code"]

speech = st.lists(st.one_of(st.sampled_from(WORDS), st.from_regex(r"[0-9]{1,6}", fullmatch=True)),
                  min_size=1, max_size=8).map(" ".join)
gather_input = st.one_of(
    st.builds(lambda d: {"Digits": d}, st.from_regex(r"[0-9#]{0,7}", fullmatch=True)),
    st.builds(lambda t: {"SpeechResult": t, "Confidence": "0.9"}, speech),
    st.builds(lambda d, t: {"Digits": d, "SpeechResult": t, "Confidence": "0.9"},
              st.from_regex(r"[0-9]{1,6}", fullmatch=True), speech),
)
resolver_outputs = st.builds(
    ResolverOutput,
    kind=st.sampled_from(["exact", "inferred", "ambiguous", "none", "correction", "intent_switch",
                          "repeat", "escalate"]),
    value=st.one_of(st.none(), st.from_regex(r"[0-9]{3,7}", fullmatch=True), st.just(Z),
                    st.sampled_from(["yes", "no", "red"])),
    slot=st.one_of(st.none(), st.sampled_from(["zip", "ssn4", "otp"])),
    reply=st.sampled_from([None, "yes", "no"]),
)


def _policy(**resolution: Any) -> Policy:
    raw = json.loads(json.dumps(SEED))
    raw["slot_resolution"].update(resolution)
    return Policy.from_dict(raw)


def _ended(h: Harness, call: str) -> bool:
    flow = h.session(call).flow
    return not flow




@PROPS
@given(how=st.sampled_from(["ceiling", "console"]), inputs=st.lists(gather_input, min_size=1, max_size=6),
       flow=st.sampled_from(["login", "stepup", "confirm"]))
def test_deterministic_never_calls_or_builds_the_model(how, inputs, flow):
    built: list[str] = []
    factory = client_for_budget(lambda *a, **k: built.append("client"))
    resolver = BedrockResolver(factory)
    called: list[Any] = []
    real = resolver.resolve

    def spy(*a: Any, **k: Any) -> Any:
        called.append(a)
        return real(*a, **k)

    resolver.resolve = spy  # type: ignore[method-assign]
    if how == "ceiling":
        h = build(resolver=None, config_kw={"slot_mode_ceiling": "deterministic"})
        assert handler.build_resolver(h.app.cfg, lambda *a, **k: built.append("client")) is None
    else:
        h = build(resolver=resolver)  # type: ignore[arg-type]
        h.app.d.policy = _policy(default_mode="deterministic")
    call = sid(1)
    try:
        if flow == "login":
            h.start(call, THOMAS)
        elif flow == "stepup":
            h.login_thomas(call)
            h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
            h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
        else:
            to_level3(h, call)
            confirm_handoff(h, call)
        for params in inputs:
            if _ended(h, call):
                break
            h.post("factor", call, **params)
    finally:
        segments.clear_sensitive(call)
    assert called == [] and built == []




step = st.one_of(
    st.tuples(st.just("edit"), st.sampled_from(MODES)),
    st.tuples(st.just("ceiling"), st.sampled_from(MODES)),
    st.tuples(st.just("answer"), st.sampled_from(["hmm maybe", "I am not sure", "let me think"])),
)


@PROPS
@given(default=st.sampled_from(MODES), ceiling=st.sampled_from(MODES),
       steps=st.lists(step, min_size=1, max_size=8))
def test_each_answer_uses_the_frozen_mode_restricted_by_the_running_ceiling(default, ceiling, steps):
    h = build(resolver=FakeResolver())
    h.app.d.policy = _policy(default_mode=default)
    h.app.cfg = replace(h.app.cfg, slot_mode_ceiling=ceiling)
    frozen = restrict(default, ceiling)   # ZIP is a digit slot: generative caps to assisted
    if frozen == "generative":
        frozen = "assisted"
    used: list[str] = []
    real = fd_slotflow.resolve

    def spy(*a: Any, **k: Any) -> Any:
        used.append(k["mode"])
        return real(*a, **k)

    call = sid(2)
    fd_slotflow.resolve = spy  # type: ignore[assignment]
    try:
        h.start(call, THOMAS)
        now = ceiling
        for kind, arg in steps:
            if kind == "edit":
                h.app.d.policy = _policy(default_mode=arg)
            elif kind == "ceiling":
                now = arg
                h.app.cfg = replace(h.app.cfg, slot_mode_ceiling=arg)
            else:
                if _ended(h, call):
                    break
                n = len(used)
                h.post("factor", call, SpeechResult=arg, Confidence="0.9")
                assert used[n:] == [restrict(frozen, now)], (default, ceiling, steps)
                event(f"mode {used[-1]}")
    finally:
        fd_slotflow.resolve = real  # type: ignore[assignment]
        segments.clear_sensitive(call)




@st.composite
def zip_answers(draw):
    """Speech with the ZIP somewhere in it (or not), and a resolver output that may propose any
    digits: the ZIP, an ordered pick from the words' digit stream, or anything."""
    before = draw(st.lists(st.sampled_from(WORDS), max_size=4))
    after = draw(st.lists(st.sampled_from(WORDS), max_size=4))
    form = draw(st.sampled_from(["digits", "spaced", "absent"]))
    middle = [] if form == "absent" else [Z if form == "digits" else " ".join(Z)]
    text = " ".join([*before, *middle, *after]) or "um"
    out = draw(resolver_outputs)
    stream = digit_stream(normalize(text, "en-US"), "en-US")
    if draw(st.booleans()) and len(stream) >= 5:
        idx = sorted(draw(st.sets(st.integers(0, len(stream) - 1), min_size=5, max_size=5)))
        out = replace(out, value="".join(stream[i] for i in idx))
    return text, out


SENTINEL = "Zq9SENTINELwords"


@PROPS
@given(case=zip_answers(), then=st.sampled_from(["yes", "no", "1", None]))
def test_a_verified_zip_is_grounded_and_no_resolver_text_is_spoken(case, then):
    text, out = case
    out = replace(out, say=SENTINEL, say_lang="en-US", candidates=(SENTINEL,))
    h = build(resolver=FakeResolver([out]))
    call = sid(3)
    try:
        h.start(call, THOMAS)
        h.post("factor", call, SpeechResult=text, Confidence="0.9")
        if then is not None and not _ended(h, call):
            key = "Digits" if then.isdigit() else "SpeechResult"
            h.post("factor", call, **{key: then})
    finally:
        segments.clear_sensitive(call)
    for p in h.lam.payloads:
        if p.get("op") == "verify" and "zip" in p.get("factors", {}):
            v = p["factors"]["zip"]
            assert len(v) == 5 and is_ordered_subsequence(v, digit_stream(normalize(text, "en-US"), "en-US")), (
                text, out, v)
            event("verified")
    for body in h.bodies:  # No resolver field reaches TwiML
        assert SENTINEL not in body
    for e in h.journal(call):
        if e["type"] != "prompt":
            continue
        assert SENTINEL not in (e.get("rendered_text") or "")
        if e.get("prompt_key") == "slot.confirm_digits":
            event("read back")
            spoken = re.findall(r"zero|one|two|three|four|five|six|seven|eight|nine|oh", e["rendered_text"])
            assert spoken, e
            value = digit_stream(normalize(" ".join(spoken), "en-US"), "en-US")
            assert is_ordered_subsequence(value, digit_stream(normalize(text, "en-US"), "en-US"))




@PROPS
@given(words=speech, out=resolver_outputs, at=st.sampled_from(["ssn4", "otp", "confirm"]))
def test_the_request_holds_no_collected_value_record_field_or_token(words, out, at):
    fake = FakeResolver([out], default=ResolverOutput(kind="none"))
    h = build(resolver=fake)
    call = sid(4)
    try:
        if at == "confirm":
            to_level3(h, call)
            confirm_handoff(h, call)
        else:
            h.login_thomas(call)
            h.set_pending_tool(call, "authenticate", {"tool": "make_payment"})
            h.handoff(call, {"reason": "step-up", "tool": "make_payment"})
            if at == "otp":
                h.factor(call, S)
        h.post("factor", call, SpeechResult=f"hmm {words}", Confidence="0.9")
        record = h.store.get("customers", {"customer_id": "C-1001"}) or {}
    finally:
        segments.clear_sensitive(call)
    secrets = {Z, S, OTP, record.get("first_name"), record.get("last_name"),
               (record.get("address") or {}).get("line1"), record.get("phone"), record.get("memory_email")}
    st_ = h.state(call)
    if st_.sealed_token:
        secrets.add(st_.sealed_token)
    for req in fake.requests:
        payload = json.dumps(request_payload(req), ensure_ascii=False)
        for s in secrets - {None}:
            assert s not in payload, (at, s)
        assert not re.search(r"\d{4,}", payload), payload   # no value of any factor
        # The read-back's card is a collected slot of the confirm flow: its name only.
        assert {c.name for c in req.collected} <= {"zip", "ssn4", "otp", "card"}
        event(f"request at {at}")




@PROPS
@given(barge_in=st.booleans(), keypad_only=st.booleans(), spoken=st.booleans())
def test_the_read_back_follows_the_slots_settings(barge_in, keypad_only, spoken):
    raw = json.loads(json.dumps(SEED))
    raw["slots"]["zip"].update(barge_in=barge_in, keypad_only=keypad_only)
    fake = FakeResolver([ResolverOutput(kind="inferred", value=Z)])
    h = build(resolver=fake, config_kw={"accept_spoken_factors": spoken})
    h.app.d.policy = Policy.from_dict(raw)
    call = sid(5)
    try:
        h.start(call, THOMAS)
        resp = h.post("factor", call, SpeechResult=f"22222, oh, no actually it is {Z}", Confidence="0.9")
    finally:
        segments.clear_sensitive(call)
    import xml.etree.ElementTree as ET  # noqa: S405 - our own TwiML

    root = ET.fromstring(resp["body"])  # noqa: S314
    gather = root.find("Gather")
    assert gather is not None
    if keypad_only or not spoken:
        assert fake.calls == [] and gather.get("input") == "dtmf"   # no model, keypad menu
        assert h.session(call).flow["pending"] is None
        return
    assert h.session(call).flow["pending"]["kind"] == "confirm_digits"
    assert gather.get("input") == "dtmf speech" and gather.get("partialResultCallback") is None
    if barge_in:
        assert len(gather) >= 1                             # the question nested, as today
    else:
        assert len(gather) == 0 and root[0].tag in ("Say", "Play")   # Path B
