"""Properties of the AgentCore hosting adapter.

* Over any open/close sequence, ``/ping`` says ``HealthyBusy`` iff the open count is
  above zero, and ``time_of_last_update`` changes only when the status changes (and then to the
  clock's time).
* For any ``/invocations`` payload, ``prewarm`` and unknown actions make zero
  calls on TAC, the voice channel or the bridge (so no store, Memory, Bedrock or Gateway call);
  ``prewarm`` answers 200 ``{"status": "ready"}`` and anything else 400.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from tac_bridge.app import build_app
from tac_bridge.hosting import HEALTHY, HEALTHY_BUSY, SocketTracker



@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.booleans(), st.floats(0, 100, allow_nan=False)), max_size=60))
def test_busy_iff_open_and_update_only_on_change(steps: list[tuple[bool, float]]) -> None:
    now = [1_800_000_000.0]
    tracker = SocketTracker(clock=lambda: now[0])
    count = 0
    last = tracker.ping()
    assert last == {"status": HEALTHY, "time_of_last_update": int(now[0])}
    for is_open, dt in steps:
        now[0] += dt
        if is_open:
            tracker.opened()
            count += 1
        else:
            tracker.closed()
            count = max(0, count - 1)
        ping = tracker.ping()
        assert ping["status"] == (HEALTHY_BUSY if count > 0 else HEALTHY)
        assert tracker.open_count == count
        if ping["status"] == last["status"]:
            assert ping["time_of_last_update"] == last["time_of_last_update"]
        else:
            assert ping["time_of_last_update"] == int(now[0])
        last = ping




class Recorder:
    """Records every attribute read, write and call made through it (and its children)."""

    def __init__(self, name: str, log: list[str], **attrs: Any) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_attrs", attrs)

    def __getattr__(self, attr: str) -> Any:
        self._log.append(f"get {self._name}.{attr}")
        if attr in self._attrs:
            return self._attrs[attr]
        return Recorder(f"{self._name}.{attr}", self._log)

    def __setattr__(self, attr: str, value: Any) -> None:
        self._log.append(f"set {self._name}.{attr}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._log.append(f"call {self._name}")
        return Recorder(f"{self._name}()", self._log)


LOG: list[str] = []
_TAC = Recorder("tac", LOG, config=SimpleNamespace(voice_websocket_path="/ws"))
CLIENT = TestClient(build_app(_TAC, Recorder("voice", LOG), Recorder("bridge", LOG),  # type: ignore[arg-type]
                              hosting="agentcore"))

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=12), inner, max_size=4),
    max_leaves=12)
actions = st.sampled_from(["prewarm", "PREWARM", "prewarm ", "orchestrator_webhook", "", "stop"]) | st.text()
payloads = st.one_of(
    json_values,
    st.builds(lambda a, rest: {**rest, "action": a}, actions | json_values,
              st.dictionaries(st.text(max_size=12), json_values, max_size=3)),
)


def _expect_ready(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("action") == "prewarm"


@settings(max_examples=300, deadline=None)
@given(payloads)
def test_invocations_touch_nothing(payload: Any) -> None:
    before = len(LOG)
    r = CLIENT.post("/invocations", content=json.dumps(payload).encode(),
                    headers={"content-type": "application/json"})
    if _expect_ready(payload):
        assert r.status_code == 200 and r.json() == {"status": "ready"}
    else:
        assert r.status_code == 400 and r.json() == {"status": "error"}
    assert LOG[before:] == []


@settings(max_examples=200, deadline=None)
@given(st.binary(max_size=64))
def test_raw_bodies_touch_nothing(body: bytes) -> None:
    before = len(LOG)
    r = CLIENT.post("/invocations", content=body, headers={"content-type": "application/json"})
    try:
        parsed: Any = json.loads(body)
    except ValueError:
        parsed = None
    assert r.status_code == (200 if _expect_ready(parsed) else 400)
    assert LOG[before:] == []


def test_building_the_app_touched_only_the_wiring() -> None:
    """Sanity check for the recorder: build-time wiring is recorded, so the empty slices above mean something."""
    assert "call tac.on_message_ready" in LOG and "set bridge.voice" in LOG
