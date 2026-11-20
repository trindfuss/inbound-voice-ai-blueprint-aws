"""Keyed-phone: the phone slot finishes early only after a short settle.

A partial whose reading is a complete phone number (ten digits starting 2-9, or eleven with a
leading 1) is claimed only if no newer partial for the Gather arrives within ``phone_settle_ms``.
Every live partial records a fresh ``settle_id`` on the Gather item before it is parsed; the
settling one sleeps (``Deps.sleep``), reloads and claims only while the id is still its own.
Other digit slots and choice slots are unchanged. The no-leak scan covers the keyed number at
teardown; logs are shape only.

:class:`VirtualTime` runs each request in its own thread under a virtual clock, one thread at a
time: ``Deps.sleep`` parks the thread until the driver moves the clock past its wake time, so
requests interleave as they would on concurrent Lambdas and nothing really waits.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from lambdas.frontdoor import gathers
from lambdas.frontdoor.app import PHONE_SETTLE_ENV, FrontDoorConfig
from lambdas.frontdoor.calls import FakeCallUpdater
from owlbank.slots import SlotDecl, early_phone
from owlbank.store import MemoryStore
from tests.component.test_frontdoor_login import (
    PHONE_Q, THOMAS_KEYED, UNKNOWN_KEYED, ZIP_Q, auth_payloads, keyed_harness, keyed_teardown,
)
from tests.component.test_frontdoor_support import CLOCK, THOMAS, Harness, says, sid

EN = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
PHONE = SlotDecl(slot_type="digits", constraints={"length": 10})


def words(digits: str) -> str:
    return " ".join("oh" if c == "0" else EN[int(c)] for c in digits)


def grouped(digits: str) -> str:
    return f"{digits[:3]} {digits[3:6]} {digits[6:]}"


class VirtualTime:
    """A virtual clock (integer ms) and a scheduler for front-door requests (see module docstring).

    ``at(ms, fn)`` schedules a request; :meth:`run` delivers them in time order, requests before
    sleepers waking at the same ms (a partial that arrives exactly as a wait ends is newer)."""

    def __init__(self, start_ms: int) -> None:
        self.now_ms = start_ms
        self._cv = threading.Condition()
        self._active = 0
        self._sleepers: list[tuple[int, int, list[bool]]] = []
        self._events: list[tuple[int, int, Callable[[], Any]]] = []
        self._seq = itertools.count()
        self._errors: list[BaseException] = []

    def clock(self) -> float:
        return self.now_ms / 1000

    def sleep(self, seconds: float) -> None:
        with self._cv:
            flag = [False]
            heapq.heappush(self._sleepers, (self.now_ms + round(seconds * 1000), next(self._seq), flag))
            self._active -= 1
            self._cv.notify_all()
            self._cv.wait_for(lambda: flag[0])

    def at(self, ms: int, fn: Callable[[], Any]) -> None:
        heapq.heappush(self._events, (ms, next(self._seq), fn))

    def _settle(self) -> None:
        self._cv.wait_for(lambda: self._active == 0)

    def _spawn(self, fn: Callable[[], Any]) -> threading.Thread:
        def body() -> None:
            try:
                fn()
            except BaseException as exc:  # surfaced by run()
                self._errors.append(exc)
            finally:
                with self._cv:
                    self._active -= 1
                    self._cv.notify_all()
        self._active += 1
        t = threading.Thread(target=body, daemon=True)
        t.start()
        return t

    def run(self) -> None:
        threads = []
        with self._cv:
            while self._events or self._sleepers:
                if self._events and (not self._sleepers or self._events[0][0] <= self._sleepers[0][0]):
                    ms, _, fn = heapq.heappop(self._events)
                    self.now_ms = max(self.now_ms, ms)
                    threads.append(self._spawn(fn))
                else:
                    ms, _, flag = heapq.heappop(self._sleepers)
                    self.now_ms = max(self.now_ms, ms)
                    self._active += 1
                    flag[0] = True
                    self._cv.notify_all()
                self._settle()
        for t in threads:
            t.join(timeout=5)
        if self._errors:
            raise self._errors[0]


class RecordingCalls(FakeCallUpdater):
    """A call updater that also records the virtual time of each update."""

    def __init__(self, vt: VirtualTime | None = None) -> None:
        super().__init__()
        self.vt = vt
        self.at_ms: list[int] = []

    def update_twiml(self, call_sid: str, twiml: str) -> None:
        super().update_twiml(call_sid, twiml)
        self.at_ms.append(self.vt.now_ms if self.vt else -1)


def phone_gather(h: Harness, call: str) -> tuple[ET.Element, str]:
    root = h.start(call, "anonymous")
    assert says(root) == [PHONE_Q]
    g = root.find("Gather")
    assert g is not None
    return g, parse_qs(urlsplit(g.get("action") or "").query)["g"][0]


#: The next SequenceNumber per (harness, call): Twilio numbers a Gather's partials in order.
_SEQ: dict[tuple[int, str], int] = {}
#: ``send_partial(..., seq=NO_SEQ)`` sends no SequenceNumber at all.
NO_SEQ = object()


def send_partial(h: Harness, call: str, g: ET.Element, text: str, *, seq: Any = None) -> None:
    """One partial callback; ``seq``: its SequenceNumber (default: the next in order; a string
    is sent as is; :data:`NO_SEQ`: none)."""
    if seq is None:
        n = _SEQ[(id(h), call)] = _SEQ.get((id(h), call), 0) + 1
        seq = str(n)
    extra = {} if seq is NO_SEQ else {"SequenceNumber": str(seq)}
    resp = h.post((g.get("partialResultCallback") or "").rsplit("/voice/", 1)[1], call,
                  UnstableSpeechResult=text, **extra)
    assert resp["statusCode"] == 204


def follow_redirect(h: Harness, calls: FakeCallUpdater, call: str) -> ET.Element:
    (_, twiml), = calls.updates
    url = ET.fromstring(twiml.encode()).find("Redirect").text or ""
    return ET.fromstring(h.post(url.rsplit("/voice/", 1)[1], call)["body"].encode())


def partial_outcomes(records) -> list[str]:
    return [r.outcome for r in records if getattr(r, "event", None) == "factor_partial"]


def make(caplog, *, settle_ms: int | None = 1000, vt: VirtualTime | None = None):
    """``settle_ms`` None: the config's default; the timing tests pin 1000."""
    calls = RecordingCalls(vt)
    kw: dict[str, Any] = {} if settle_ms is None else {"phone_settle_ms": settle_ms}
    h = keyed_harness(caplog, calls=calls, config_kw=kw)
    h.sent_factors.update({THOMAS_KEYED, *UNKNOWN_KEYED})
    if vt is not None:
        h.app.d.clock = vt.clock
        h.app.d.sleep = vt.sleep
    return h, calls


# --------------------------------------------------------------------------- the parse


@pytest.mark.parametrize(("text", "want"), [
    ("202 555 0101", ("complete", "2025550101")),
    (words("2025550101"), ("complete", "2025550101")),
    ("one " + words("2025550101"), ("complete", "2025550101")),     # eleven with a leading one
    ("12025550101", ("complete", "2025550101")),
    ("1 202 555 0101", ("complete", "2025550101")),
    ("one " + words("202555010"), ("incomplete", None)),            # "one" + nine: never complete
    ("1202555010", ("incomplete", None)),
    ("0202555010", ("incomplete", None)),
    ("10202555010", ("incomplete", None)),                           # leading 1 then a 0 area code
    ("202 555", ("incomplete", None)),
])
def test_early_phone_reading(text, want):
    assert early_phone(PHONE, text, "en-US") == want


# --------------------------------------------------------------------------- unit: the route


def test_the_wait_opens_the_twilio_connection_and_counts_its_time(caplog):
    """The settling partial warms the updater inside the wait; the warm-up's time is part
    of the wait, not added to it."""
    vt = VirtualTime(int(CLOCK * 1000))
    h, calls = make(caplog, vt=vt)

    def slow_warm() -> None:
        calls.warms += 1
        vt.now_ms += 300

    calls.ensure_warm = slow_warm  # type: ignore[method-assign]
    try:
        g, _ = phone_gather(h, sid(80))
        t0 = vt.now_ms
        vt.at(t0, lambda: send_partial(h, sid(80), g, words(THOMAS_KEYED)))
        vt.run()
        assert calls.warms == 1 and calls.at_ms == [t0 + 1000]
    finally:
        keyed_teardown(h, caplog)


def test_a_failed_warm_up_still_claims(caplog):
    h, calls = make(caplog)

    def broken() -> None:
        raise RuntimeError("twilio unavailable")

    calls.ensure_warm = broken  # type: ignore[method-assign]
    try:
        g, _ = phone_gather(h, sid(81))
        send_partial(h, sid(81), g, words(THOMAS_KEYED))
        assert h.sleeps == [1.0] and len(calls.updates) == 1
        assert partial_outcomes(caplog.records) == ["settling", "complete"]
        assert any(getattr(r, "event", None) == "twilio_connection_warm_failed" for r in caplog.records)
    finally:
        keyed_teardown(h, caplog)


def test_a_settling_phone_answer_warms_once(caplog):
    h, calls = make(caplog)
    try:
        g, _ = phone_gather(h, sid(82))
        send_partial(h, sid(82), g, words(THOMAS_KEYED))
        assert calls.warms == 1
    finally:
        keyed_teardown(h, caplog)


def test_complete_partial_with_no_newer_one_claims_after_the_wait(caplog):
    h, calls = make(caplog)
    try:
        g, gid = phone_gather(h, sid(60))
        send_partial(h, sid(60), g, words(THOMAS_KEYED))
        assert h.sleeps == [1.0] and len(calls.updates) == 1
        assert partial_outcomes(caplog.records) == ["settling", "complete"]
        item = gathers.load(h.store, sid(60), gid)
        assert gathers.state_of(item) == gathers.EARLY
        assert not set(gathers.SETTLE_FIELDS) & set(item)       # the claim drops the settle fields
        assert says(follow_redirect(h, calls, sid(60))) == [ZIP_Q]
        assert auth_payloads(h, "ani_lookup")[0]["args"]["ani"] == THOMAS
    finally:
        keyed_teardown(h, caplog)


def test_a_correction_within_the_window_supersedes_and_the_claim_uses_it(caplog):
    vt = VirtualTime(int(CLOCK * 1000))
    h, calls = make(caplog, vt=vt)
    try:
        g, _ = phone_gather(h, sid(61))
        t0 = vt.now_ms
        vt.at(t0, lambda: send_partial(h, sid(61), g, words(UNKNOWN_KEYED[0])))
        vt.at(t0 + 400, lambda: send_partial(h, sid(61), g, words(THOMAS_KEYED)))   # the correction
        vt.run()
        assert len(calls.updates) == 1 and calls.at_ms == [t0 + 1400]
        assert sorted(partial_outcomes(caplog.records)) == ["complete", "settling", "settling",
                                                            "superseded"]
        assert says(follow_redirect(h, calls, sid(61))) == [ZIP_Q]
        assert [p["args"]["ani"] for p in auth_payloads(h, "ani_lookup")] == [THOMAS]
    finally:
        keyed_teardown(h, caplog)


def test_an_incomplete_newer_partial_also_supersedes(caplog):
    vt = VirtualTime(int(CLOCK * 1000))
    h, calls = make(caplog, vt=vt)
    try:
        g, _ = phone_gather(h, sid(62))
        t0 = vt.now_ms
        vt.at(t0, lambda: send_partial(h, sid(62), g, words(THOMAS_KEYED)))
        vt.at(t0 + 999, lambda: send_partial(h, sid(62), g, words(THOMAS_KEYED[:6])))   # a shorter revision
        vt.run()
        assert calls.updates == []
        assert partial_outcomes(caplog.records) == ["settling", "incomplete", "superseded"]
    finally:
        keyed_teardown(h, caplog)


def test_ten_digits_starting_with_one_never_claim(caplog):
    h, calls = make(caplog)
    try:
        g, gid = phone_gather(h, sid(63))
        send_partial(h, sid(63), g, "one " + words(THOMAS_KEYED[:9]))
        assert h.sleeps == [] and calls.updates == []
        assert partial_outcomes(caplog.records) == ["incomplete"]
        assert gathers.load(h.store, sid(63), gid)["settle_seq"] == 1   # still counts as newer
    finally:
        keyed_teardown(h, caplog)


def test_eleven_with_a_leading_one_claims_the_ten(caplog):
    h, calls = make(caplog)
    try:
        g, _ = phone_gather(h, sid(64))
        send_partial(h, sid(64), g, "1" + THOMAS_KEYED)
        assert h.sleeps == [1.0] and len(calls.updates) == 1
        assert says(follow_redirect(h, calls, sid(64))) == [ZIP_Q]
        assert auth_payloads(h, "ani_lookup")[0]["args"]["ani"] == THOMAS
    finally:
        keyed_teardown(h, caplog)


def test_the_final_webhook_during_the_wait_wins(caplog):
    vt = VirtualTime(int(CLOCK * 1000))
    h, calls = make(caplog, vt=vt)
    try:
        g, _ = phone_gather(h, sid(65))
        t0 = vt.now_ms
        roots: list[ET.Element] = []
        vt.at(t0, lambda: send_partial(h, sid(65), g, words(THOMAS_KEYED)))
        vt.at(t0 + 500, lambda: roots.append(h.xml(
            (g.get("action") or "").rsplit("/voice/", 1)[1], sid(65),
            SpeechResult=words(THOMAS_KEYED), Confidence="0.9")))
        vt.run()
        assert calls.updates == [] and says(roots[0]) == [ZIP_Q]
        assert partial_outcomes(caplog.records) == ["settling", "already_claimed"]
    finally:
        keyed_teardown(h, caplog)


def test_settle_zero_claims_at_once(caplog):
    h, calls = make(caplog, settle_ms=0)
    try:
        g, gid = phone_gather(h, sid(66))
        send_partial(h, sid(66), g, words(THOMAS_KEYED))
        assert h.sleeps == [] and len(calls.updates) == 1
        assert partial_outcomes(caplog.records) == ["complete"]
        assert "settle_seq" not in gathers.load(h.store, sid(66), gid)
    finally:
        keyed_teardown(h, caplog)


def test_settling_logs_are_shape_only(caplog):
    h, _ = make(caplog)
    try:
        g, _ = phone_gather(h, sid(67))
        send_partial(h, sid(67), g, words(THOMAS_KEYED))
        lines = [r for r in caplog.records if getattr(r, "event", None) == "factor_partial"]
        assert {r.outcome for r in lines} == {"settling", "complete"}
        for r in lines:
            blob = r.getMessage() + " " + " ".join(str(v) for v in r.__dict__.values())
            assert words(THOMAS_KEYED) not in blob and THOMAS_KEYED not in blob
        assert lines[0].settle_ms == 1000
    finally:
        keyed_teardown(h, caplog)   # no digits or E.164 anywhere


def test_zip_partials_are_unchanged(caplog):
    """Other digit slots: no settle field, no wait, the first complete partial claims."""
    from tests.component.test_frontdoor_early_digits import (
        ZIP_WORDS, gather_of, gid_of, partial,
    )
    from tests.component.test_frontdoor_early_digits import make as make_zip
    h, calls = make_zip()
    try:
        g = gather_of(h.start(sid(68), THOMAS))
        partial(h, sid(68), g, ZIP_WORDS)
        assert h.sleeps == [] and len(calls.updates) == 1
        assert not set(gathers.SETTLE_FIELDS) & set(gathers.load(h.store, sid(68), gid_of(g)))
    finally:
        from tests.component.test_frontdoor_early_digits import _cleanup
        _cleanup(h)


# --------------------------------------------------------------------------- the settle fields


def test_take_drops_the_settle_fields_and_a_settle_claim_needs_its_id():
    store, call, gid = MemoryStore(), "CA" + "1" * 32, "g" * 16
    assert gathers.mark_partial(store, call, gid, settle_id="a" * 16, seq=1, expires_at=9)
    assert gathers.mark_partial(store, call, gid, settle_id="b" * 16, seq=2, expires_at=9)
    item = gathers.load(store, call, gid)
    assert item["settle_seq"] == 2 and gathers.settle_id_of(item) == "b" * 16
    assert not gathers.cas(store, call, gid, gathers.NONE, {"state": gathers.EARLY}, settle_id="a" * 16)
    won, _ = gathers.take(store, call, gid, allowed=(gathers.NONE,),
                          changes={"state": gathers.PROCESSING})
    assert won and not set(gathers.SETTLE_FIELDS) & set(gathers.load(store, call, gid))
    # A partial that lands after the take can't claim the processing item with its id.
    gathers.mark_partial(store, call, gid, settle_id="c" * 16, seq=3, expires_at=9)
    reloaded = gathers.load(store, call, gid)
    assert gathers.state_of(reloaded) == gathers.PROCESSING
    with pytest.raises(ValueError):
        gathers.cas(store, call, gid, gathers.PROCESSING, {}, settle_id="c" * 16)


# --------------------------------------------------------------------------- partial order


def test_mark_partial_only_moves_forward_and_unsequenced_only_replaces_the_id():
    store, call, gid = MemoryStore(), "CA" + "2" * 32, "h" * 16
    assert gathers.mark_partial(store, call, gid, settle_id="a" * 16, seq=5, expires_at=9)
    assert not gathers.mark_partial(store, call, gid, settle_id="b" * 16, seq=4, expires_at=9)
    assert not gathers.mark_partial(store, call, gid, settle_id="c" * 16, seq=5, expires_at=9)
    item = gathers.load(store, call, gid)
    assert (item["settle_seq"], gathers.settle_id_of(item)) == (5, "a" * 16)
    assert not gathers.mark_partial(store, call, gid, settle_id="d" * 16, seq=None, expires_at=9)
    item = gathers.load(store, call, gid)
    assert (item["settle_seq"], gathers.settle_id_of(item), item["settle_unseq"]) == (5, "d" * 16, 1)
    assert gathers.mark_partial(store, call, gid, settle_id="e" * 16, seq=6, expires_at=9)


def test_an_older_partial_delivered_late_neither_supersedes_nor_claims(caplog):
    """The correction (seq 2) arrives first; the original reading (seq 1) arrives during its wait:
    the correction still claims, with its value; the late one writes nothing."""
    vt = VirtualTime(int(CLOCK * 1000))
    h, calls = make(caplog, vt=vt)
    try:
        g, gid = phone_gather(h, sid(70))
        t0 = vt.now_ms
        vt.at(t0, lambda: send_partial(h, sid(70), g, words(THOMAS_KEYED), seq="2"))
        vt.at(t0 + 300, lambda: send_partial(h, sid(70), g, words(UNKNOWN_KEYED[0]), seq="1"))
        vt.run()
        assert len(calls.updates) == 1 and calls.at_ms == [t0 + 1000]
        assert partial_outcomes(caplog.records) == ["settling", "superseded", "complete"]
        assert says(follow_redirect(h, calls, sid(70))) == [ZIP_Q]
        assert [p["args"]["ani"] for p in auth_payloads(h, "ani_lookup")] == [THOMAS]
    finally:
        keyed_teardown(h, caplog)


def test_an_older_complete_partial_after_a_newer_one_never_claims(caplog):
    """seq 3 (incomplete) then seq 2 (complete) delivered late: no claim at all."""
    h, calls = make(caplog)
    try:
        g, _ = phone_gather(h, sid(71))
        send_partial(h, sid(71), g, words(THOMAS_KEYED[:6]), seq="3")
        send_partial(h, sid(71), g, words(THOMAS_KEYED), seq="2")
        assert h.sleeps == [] and calls.updates == []
        assert partial_outcomes(caplog.records) == ["incomplete", "superseded"]
    finally:
        keyed_teardown(h, caplog)


@pytest.mark.parametrize("seq", [NO_SEQ, "", "x", "-1", "1.5", "1e3", "１", "1234567890"])
def test_a_partial_without_a_valid_sequence_number_never_claims_but_blocks_older(caplog, seq):
    vt = VirtualTime(int(CLOCK * 1000))
    h, calls = make(caplog, vt=vt)
    try:
        g, gid = phone_gather(h, sid(72))
        t0 = vt.now_ms
        vt.at(t0, lambda: send_partial(h, sid(72), g, words(THOMAS_KEYED), seq="1"))
        vt.at(t0 + 200, lambda: send_partial(h, sid(72), g, words(THOMAS_KEYED), seq=seq))
        vt.run()
        assert calls.updates == []
        assert sorted(partial_outcomes(caplog.records)) == ["settling", "superseded", "unsequenced"]
        assert gathers.state_of(gathers.load(h.store, sid(72), gid)) == gathers.NONE
    finally:
        keyed_teardown(h, caplog)


# --------------------------------------------------------------------------- the setting


def test_settle_setting_from_env():
    base = {"OWLBANK_RELAY_WS_URL": "wss://x.example/ws"}
    assert FrontDoorConfig.from_env(base).phone_settle_ms == 0
    assert FrontDoorConfig.from_env({**base, PHONE_SETTLE_ENV: ""}).phone_settle_ms == 0
    assert FrontDoorConfig.from_env({**base, PHONE_SETTLE_ENV: "750"}).phone_settle_ms == 750
    assert FrontDoorConfig.from_env({**base, PHONE_SETTLE_ENV: "0"}).phone_settle_ms == 0
    for bad in ("fast", "-1", "3001", "1.5"):
        with pytest.raises(ValueError, match=PHONE_SETTLE_ENV) as err:
            FrontDoorConfig.from_env({**base, PHONE_SETTLE_ENV: bad})
        assert bad not in str(err.value)
