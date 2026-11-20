"""The rehearsal leaves nothing open, whatever fails (scripts/rehearse_human_leg.py).

The fake Twilio (:class:`tests.unit.scripts.test_rehearse_human_leg.FakeWorld`) is a state machine of calls,
tasks, conferences and conversations with Twilio's transitions; it is strict (ending one call ends no other,
a call that ends leaves its task open) and its task filter is leaky (the queue's other task is always listed).

* **No leftovers.** Over any mode, transfer or not, a long wait or not, a slow reservation, and up to two
  failures injected at any requests (each either before its effect, or after it with the answer lost), no
  call, task or conversation of the run is left open, and the queue's other task is never read or written.
* **Honest leftovers.** If one operation fails every time, the run exits 3 and names every call, task and
  conversation it left open (nothing open goes unreported).
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from scripts import rehearse_human_leg as rh
from tests.unit.scripts.test_rehearse_human_leg import OTHER_CALL, OTHER_TASK, STT, Failing, FakeWorld, argv


class FailingAt(Failing):
    """:class:`Failing` with a set of failing request indices, each with its own kind."""

    def __init__(self, world: FakeWorld, failures: dict[int, str]) -> None:
        super().__init__(world)
        self.failures = failures

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.world, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            index = self.count
            self.count += 1
            self.world.log.append((name, args, kwargs))
            kind = self.failures.get(index)
            if kind is None:
                return method(*args, **kwargs)
            if kind == "after":
                method(*args, **kwargs)
            if name in ("close_conversation", "start_transcription", "stop_transcription"):
                return False
            if name == "resolve_conversation":
                return None
            raise rh.ApiError(f"{name}: HTTP 503")

        return call


def run(world: FakeWorld, args: list[str], api: Failing) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = rh.main(args, api_factory=lambda cfg: api, environ={}, clock=world.now, sleep=world.sleep,
                   stt_loader=lambda lang: STT, out=lines.append, run_id="p1")
    return code, lines


scenario = st.fixed_dictionaries({
    "mode": st.sampled_from(["standalone", "frontdoor"]),
    "transfer": st.booleans(),
    "long_wait": st.sampled_from([0, 6, 11]),
    "reservation_delay": st.integers(0, 3),
})
failures = st.dictionaries(st.integers(0, 90), st.sampled_from(["before", "after"]), max_size=2)


def extra_args(s: dict[str, Any]) -> list[str]:
    out = ["--transfer"] if s["transfer"] else []
    if s["long_wait"]:
        out += ["--long-wait", str(s["long_wait"]), "--confirm-long-wait", str(s["long_wait"])]
    return out


@settings(max_examples=400, deadline=None)
@given(scenario, failures)
def test_no_call_task_or_conversation_is_left_open(s: dict[str, Any], fail: dict[int, str]) -> None:
    world = FakeWorld(mode=s["mode"], reservation_delay=s["reservation_delay"])
    code, lines = run(world, argv(s["mode"], *extra_args(s)), FailingAt(world, fail))
    assert world.open_things() == {"calls": [], "tasks": [], "conversations": []}, (fail, lines)
    assert code in (0, 1, 2), lines
    assert OTHER_TASK not in world.touched_tasks
    assert world.tasks[OTHER_TASK]["status"] == "pending" and world.calls[OTHER_CALL]["status"] == "in-progress"
    placed = [n for n, _a, _k in world.log].count("create_call")
    assert placed <= rh.MAX_CALLS
    assert [n for n, _a, _k in world.log].count("add_participant") <= 1  # with the agent: 2 legs at most


ALWAYS = ["end_call", "set_task_status", "task_status", "close_conversation", "find_calls", "find_tasks",
          "find_conversation", "call_status"]


@settings(max_examples=150, deadline=None)
@given(scenario, st.sampled_from(ALWAYS))
def test_whatever_is_left_open_is_reported(s: dict[str, Any], always: str) -> None:
    world = FakeWorld(mode=s["mode"], reservation_delay=s["reservation_delay"])
    code, lines = run(world, argv(s["mode"], *extra_args(s)), Failing(world, always=always))
    left = world.open_things()
    if any(left.values()):
        assert code == 3, lines
        text = "\n".join(lines)
        for things in left.values():
            for thing in things:
                assert thing in text or "could not be" in text, (thing, lines)
    assert OTHER_TASK not in world.touched_tasks
