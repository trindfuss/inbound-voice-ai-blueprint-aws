"""scripts/twilio_setup.py ``--taskrouter-events``.

The Flex workspace's event callback is workspace-wide, so the step sets it only when it is empty or
already the front door's, and refuses to touch anyone else's. The recording fake from
``tests/unit/scripts/test_twilio_setup.py`` stands in for Twilio; nothing here reaches it.
No PBT: flag handling and one find-or-set step.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import pytest

from scripts import twilio_setup as ts
from tests.unit.scripts.test_twilio_setup import (
    BASE, PASSWORD, USER, FakeTwilio, NoSM, run_apply,
)
from tests.unit.scripts.test_twilio_setup import (  # noqa: F401 (autouse: no real Secrets Manager client)
    _no_real_secrets_manager,
)

FLAG = ["--with-flex", "--taskrouter-events"]
OURS = f"https://{quote(USER)}:{quote(PASSWORD)}@api.owlbank.example/voice/taskrouter"
FILTER = "reservation.accepted,task.canceled,workflow.timeout,task.updated"


class EventsFake(FakeTwilio):
    """``FakeTwilio`` whose workspace also has ``fetch`` and ``update`` (event callback fields)."""

    def __init__(self, *, callback: str | None = None, events_filter: str | None = None,
                 **kw: Any) -> None:
        super().__init__(**kw)
        for w in self.workspaces:
            w.event_callback_url = callback
            w.events_filter = events_filter

    def _workspace(self, ws: str) -> Any:
        base = super()._workspace(ws)
        target = next(w for w in self.workspaces if w.sid == ws)

        def fetch() -> Any:
            self.calls.append(("workspaces.fetch", ws))
            return SimpleNamespace(**vars(target))

        def update(**kw: Any) -> Any:
            self.calls.append(("workspaces.update", kw))
            for k, v in kw.items():
                setattr(target, k, v)
            return target

        return SimpleNamespace(task_queues=base.task_queues, workflows=base.workflows,
                               fetch=fetch, update=update)


def events_writes(fake: FakeTwilio) -> list[tuple[str, Any]]:
    return [c for c in fake.calls if c[0] == "workspaces.update"]


def ws(fake: FakeTwilio) -> Any:
    return fake.workspaces[0]


# --------------------------------------------------------------------------- plan


def test_without_the_flag_the_plan_has_no_events_step() -> None:
    cfg = ts.SetupConfig(phone_number="+12065550007", api_domain="api.owlbank.example",
                         relay_domain="relay.owlbank.example",
                         queue_names={k: k for k, *_ in ts.QUEUES})
    assert "ensure_taskrouter_events" not in [s.kind for s in ts.build_plan(cfg)]


@pytest.mark.parametrize("phase", ["post", "all"])
def test_flag_appends_the_step_last_in_post_and_all(phase: str) -> None:
    cfg = ts.SetupConfig(phone_number="+12065550007", api_domain="api.owlbank.example",
                         relay_domain="relay.owlbank.example", phase=phase, with_flex=True, taskrouter_events=True,
                         queue_names={k: k for k, *_ in ts.QUEUES}, api_base_path="/v1")
    step = ts.build_plan(cfg)[-1]
    assert step.kind == "ensure_taskrouter_events"
    assert str(step.params["event_callback_url"]) == "https://***:***@api.owlbank.example/v1/voice/taskrouter"
    assert step.params["events_filter"] == FILTER


def test_with_flex_alone_has_no_events_step() -> None:
    cfg = ts.SetupConfig(phone_number="+12065550007", api_domain="api.owlbank.example",
                         relay_domain="relay.owlbank.example", with_flex=True,
                         queue_names={k: k for k, *_ in ts.QUEUES})
    assert "ensure_taskrouter_events" not in [s.kind for s in ts.build_plan(cfg)]


def test_flag_without_with_flex_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--taskrouter-events"]) == 2
    assert "--taskrouter-events needs --with-flex" in capsys.readouterr().err


def test_flag_with_phase_pre_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main(["--phase", "pre", *FLAG]) == 2
    assert "--taskrouter-events applies to --phase post or all" in capsys.readouterr().err


def test_dry_run_shows_the_step_without_credentials_and_makes_no_calls(
        capsys: pytest.CaptureFixture[str]) -> None:
    def no_client(*a: Any) -> Any:
        raise AssertionError("Twilio client in dry run")

    assert ts.main([*BASE, *FLAG], sm_client=NoSM(), client_factory=no_client) == 0
    out = capsys.readouterr().out
    line = next(x for x in out.splitlines() if "ensure_taskrouter_events" in x)
    assert "https://***:***@api.owlbank.example/voice/taskrouter" in line
    assert FILTER in line and "empty or already this front door's" in line
    assert USER not in out and PASSWORD not in out


# --------------------------------------------------------------------------- apply


def test_sets_the_callback_and_filter_when_unset(tmp_path: Path) -> None:
    fake = EventsFake()
    assert run_apply(tmp_path, fake, FLAG) == 0
    assert events_writes(fake) == [("workspaces.update",
                                    {"event_callback_url": OURS, "events_filter": FILTER})]
    assert (ws(fake).event_callback_url, ws(fake).events_filter) == (OURS, FILTER)


def test_no_op_when_already_ours(tmp_path: Path) -> None:
    fake = EventsFake(callback=OURS, events_filter=FILTER)
    assert run_apply(tmp_path, fake, FLAG) == 0
    assert events_writes(fake) == []
    assert ("workspaces.fetch", ws(fake).sid) in fake.calls


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    fake = EventsFake()
    assert run_apply(tmp_path, fake, FLAG) == 0
    assert run_apply(tmp_path, fake, FLAG) == 0
    assert len(events_writes(fake)) == 1


@pytest.mark.parametrize("callback, events_filter", [
    ("https://old:creds@api.owlbank.example/voice/taskrouter", FILTER),   # rotated credentials
    (OURS, "task.created"),                                              # a different filter
])
def test_ours_with_drift_is_brought_up_to_date(tmp_path: Path, callback: str, events_filter: str) -> None:
    fake = EventsFake(callback=callback, events_filter=events_filter)
    assert run_apply(tmp_path, fake, FLAG) == 0
    assert events_writes(fake) == [("workspaces.update",
                                    {"event_callback_url": OURS, "events_filter": FILTER})]


@pytest.mark.parametrize("foreign", [
    "https://flex-plugins.example.com/taskrouter-events",
    "https://u:secret-pw@api.owlbank.example/voice/other",
    "https://api.other.example/voice/taskrouter",
])
def test_refuses_to_overwrite_a_foreign_callback(tmp_path: Path, foreign: str,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    fake = EventsFake(callback=foreign, events_filter="task.created")
    assert run_apply(tmp_path, fake, FLAG) == 3
    err = capsys.readouterr().err
    assert "already sends its TaskRouter events to" in err and "left as it is" in err
    assert "secret-pw" not in err and PASSWORD not in err
    assert events_writes(fake) == []
    assert ws(fake).event_callback_url == foreign


def test_without_the_flag_the_workspace_callback_is_never_read(tmp_path: Path) -> None:
    fake = EventsFake(callback="https://flex-plugins.example.com/x")
    assert run_apply(tmp_path, fake) == 0
    assert not [c for c in fake.calls if c[0] in ("workspaces.fetch", "workspaces.update")]


def test_the_number_status_callback_stays_voice_status(tmp_path: Path) -> None:
    """The ``call.end`` span comes from the number's status callback, which the setup
    already points at ``/voice/status`` (no change for call observability)."""
    fake = EventsFake()
    assert run_apply(tmp_path, fake, FLAG) == 0
    assert fake.numbers[0].status_callback == \
        f"https://{quote(USER)}:{quote(PASSWORD)}@api.owlbank.example/voice/status"


def test_filter_matches_the_front_doors_outcome_map() -> None:
    """The events the workspace sends are the ones the front door turns into spans plus
    ``task.updated`` (the hand-off's attribute changes). Twilio
    rejects an unknown event type (HTTP 400, 20001), so a name edited on one side only fails here."""
    from lambdas.frontdoor.app import TASKROUTER_OUTCOMES

    assert set(ts.TASKROUTER_EVENTS_FILTER.split(",")) == set(TASKROUTER_OUTCOMES) | {"task.updated"}
    assert "task.timeout" not in TASKROUTER_OUTCOMES  # not a TaskRouter event
