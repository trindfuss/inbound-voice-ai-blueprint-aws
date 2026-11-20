"""Keyed-phone, P "no cut-off correction": for any interleaving of phone partials and clock
steps, a claim is only ever made from the latest partial of that Gather, and never within
``settle_ms`` of a later partial.

Run end to end through ``/voice/partial`` on concurrent requests under a virtual clock
(:class:`~tests.component.test_frontdoor_phone_settle.VirtualTime`: each partial its own thread,
``Deps.sleep`` parks it, nothing really waits). Checked against the model: the claim, if any, is
made by the first partial with a complete reading whose next partial arrives strictly after its
wait ends (or that has none), at its arrival plus ``settle_ms``, with its value.

Out-of-order delivery: Twilio numbers partials (``SequenceNumber``) but they may arrive in
another order. P: a claim's value is always that of the highest SequenceNumber delivered by then,
and the claim comes at least ``settle_ms`` after that partial arrived.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from lambdas.frontdoor import gathers
from owlbank import sensitive
from owlbank.segments import clear_sensitive
from owlbank.slots import early_phone
from tests.component.test_frontdoor_phone_settle import (
    PHONE, RecordingCalls, VirtualTime, grouped, phone_gather, send_partial, words,
)
from tests.component.test_frontdoor_support import CLOCK, build, sid

NUMBERS = ("2025550101", "4075445555", "3125550199")
#: Complete readings, readings that are never complete, and prefixes.
TEXTS = sorted({
    *(f(n) for n in NUMBERS for f in (words, grouped, lambda n: "one " + words(n), lambda n: "1" + n)),
    *(words(n[:k]) for n in NUMBERS for k in (3, 6, 9)),
    "one " + words("202555010"), "1202555010", "um",
})

partials = st.lists(st.tuples(st.sampled_from(TEXTS), st.integers(min_value=0, max_value=2_500)),
                    min_size=1, max_size=8)


def _keyed(doc: dict[str, Any]) -> None:
    doc["ani_trust"]["ask_phone_on_no_match"] = True


def model(seq: list[tuple[str, int]], settle_ms: int) -> tuple[int, str] | None:
    """(claim offset ms, value) or None: see the module docstring."""
    t, times = 0, []
    for _text, gap in seq:
        t += gap
        times.append(t)
    for i, (text, _) in enumerate(seq):
        outcome, value = early_phone(PHONE, text, "en-US")
        if outcome != "complete" or value is None:
            continue
        if i == len(seq) - 1 or times[i + 1] > times[i] + settle_ms:
            return times[i] + settle_ms, value
    return None


@settings(max_examples=150, deadline=None)
@given(seq=partials, settle_ms=st.sampled_from([300, 1000]))
def test_no_cut_off_correction(seq: list[tuple[str, int]], settle_ms: int) -> None:
    vt = VirtualTime(int(CLOCK * 1000))
    calls = RecordingCalls(vt)
    h = build(policy_edit=_keyed, calls=calls, config_kw={"phone_settle_ms": settle_ms})
    h.app.d.clock, h.app.d.sleep = vt.clock, vt.sleep
    call = sid(90)
    try:
        g, gid = phone_gather(h, call)
        t0 = vt.now_ms
        arrivals: list[tuple[int, str]] = []
        t = t0
        for text, gap in seq:
            t += gap
            arrivals.append((t, text))
            n = len(arrivals)
            vt.at(t, lambda text=text, n=n: send_partial(h, call, g, text, seq=str(n)))
        vt.run()

        item = gathers.load(h.store, call, gid)
        claimed = None
        if calls.updates:
            assert len(calls.updates) == 1 and gathers.state_of(item) == gathers.EARLY
            value = h.sealer.unseal(item["sealed_value"], call_sid=call)
            claimed = (calls.at_ms[0] - t0, value)
            # The claim is from the latest partial delivered by then ...
            latest_ms, latest_text = [a for a in arrivals if a[0] <= calls.at_ms[0]][-1]
            assert early_phone(PHONE, latest_text, "en-US") == ("complete", value)
            # ... and at least settle_ms after it (no partial came in between).
            assert calls.at_ms[0] - latest_ms >= settle_ms
        else:
            assert gathers.state_of(item) == gathers.NONE
        assert claimed == model(seq, settle_ms)
    finally:
        clear_sensitive(call)
        sensitive.forget(call)


@settings(max_examples=150, deadline=None)
@given(seq=partials, delays=st.lists(st.integers(min_value=0, max_value=1_500), min_size=8, max_size=8),
       settle_ms=st.sampled_from([300, 1000]))
def test_out_of_order_delivery_claims_only_the_highest_sequence(
        seq: list[tuple[str, int]], delays: list[int], settle_ms: int) -> None:
    vt = VirtualTime(int(CLOCK * 1000))
    calls = RecordingCalls(vt)
    h = build(policy_edit=_keyed, calls=calls, config_kw={"phone_settle_ms": settle_ms})
    h.app.d.clock, h.app.d.sleep = vt.clock, vt.sleep
    call = sid(91)
    try:
        g, gid = phone_gather(h, call)
        sent = vt.now_ms
        arrivals: list[tuple[int, int, str]] = []          # (arrival ms, SequenceNumber, text)
        for n, ((text, gap), delay) in enumerate(zip(seq, delays[:len(seq)], strict=True), start=1):
            sent += gap
            arrivals.append((sent + delay, n, text))
            vt.at(sent + delay, lambda text=text, n=n: send_partial(h, call, g, text, seq=str(n)))
        vt.run()

        item = gathers.load(h.store, call, gid)
        if not calls.updates:
            assert gathers.state_of(item) == gathers.NONE
            return
        assert len(calls.updates) == 1 and gathers.state_of(item) == gathers.EARLY
        value = h.sealer.unseal(item["sealed_value"], call_sid=call)
        at = calls.at_ms[0]
        delivered = [a for a in arrivals if a[0] <= at]
        top_ms, _, top_text = max(delivered, key=lambda a: a[1])
        assert early_phone(PHONE, top_text, "en-US") == ("complete", value)
        assert at - top_ms >= settle_ms
    finally:
        clear_sensitive(call)
        sensitive.forget(call)
