"""scripts/twilio_setup.py --phase pre|post|all: the phased first deploy.

``pre`` needs no domain and writes the IDs the stacks read; ``post`` sets the webhooks after the
deploy. Same fake client as the main test module; nothing here reaches Twilio.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts import twilio_setup as ts

from .test_twilio_setup import (
    API_KEY_ARN,
    API_SECRET,
    AUTH_TOKEN,
    BASIC_ARN,
    NUMBER,
    PARENT,
    PASSWORD,
    SECRET_ARGS,
    TOKEN_ARN,
    SUB,
    USER,
    FakeTwilio,
    NoSM,
    StubSM,
    _no_real_secrets_manager,  # noqa: F401 (autouse fixture: no real Secrets Manager client)
)

DOMAINS = ["--api-domain", "api.owlbank.example", "--relay-domain", "relay.owlbank.example"]
PHONE = ["--phone-number", NUMBER]
WEBHOOK = "https://relay.owlbank.example/webhook"  # no credentials: the bridge never challenges


def run(tmp_path: Path, fake: FakeTwilio, phase: str, *, extra: Sequence[str] = (),
        sm: Any = None) -> int:
    args = ["--phase", phase, "--account-sid", SUB, "--apply", "--output", str(tmp_path / "t.json"),
            *SECRET_ARGS, *extra]
    if phase != "pre":
        args += [*PHONE, *DOMAINS]
    return ts.main(args, sm_client=sm or StubSM(), client_factory=fake.factory)


def out_file(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "t.json").read_text())


# --------------------------------------------------------------------------- dry run


def test_default_phase_is_all() -> None:
    assert ts.build_parser().parse_args(["--phone-number", NUMBER]).phase == "all"


@pytest.mark.parametrize("with_flex", [False, True])
def test_pre_dry_run_needs_no_domain_or_phone_and_makes_no_calls(capsys: pytest.CaptureFixture[str],
                                                                 with_flex: bool) -> None:
    def boom(*a: Any) -> Any:
        raise AssertionError("network use in a dry run")

    assert ts.main(["--phase", "pre", *(["--with-flex"] if with_flex else [])],
                   sm_client=NoSM(), client_factory=boom) == 0
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN") and "phase pre" in out
    for text in ("check_account", "ensure_memory_store",
                 "ensure_conversation_configuration", "ensure_verify_service", "omitted until --phase post"):
        assert text in out, text
    assert ("ensure_queue" in out) is with_flex
    for text in ("configure_number", "voice/incoming", "/webhook", "statusCallbacks", "***:***"):
        assert text not in out, text


def test_post_dry_run_lists_only_the_webhook_steps(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main(["--phase", "post", *PHONE, *DOMAINS]) == 0
    out = capsys.readouterr().out
    assert "configure_number" in out and "ensure_status_callbacks" in out and "verify_setup" in out
    assert "https://relay.owlbank.example/webhook" in out and "@relay.owlbank.example" not in out
    assert "ensure_queue" not in out and "ensure_memory_store" not in out and "ensure_verify_service" not in out


def test_plan_kinds_per_phase() -> None:
    from .test_twilio_setup import cfg

    assert [s.kind for s in ts.build_plan(cfg(phase="pre"))] == [
        "check_account", "ensure_queue", "ensure_queue", "ensure_queue", "ensure_queue", "ensure_workflow",
        "ensure_flex_skills", "ensure_memory_store", "ensure_conversation_configuration", "ensure_verify_service",
        "ensure_trait_group"]
    assert [s.kind for s in ts.build_plan(cfg(phase="pre", with_flex=False))] == [
        "check_account", "ensure_memory_store", "ensure_conversation_configuration", "ensure_verify_service",
        "ensure_trait_group"]
    for with_flex in (False, True):
        assert [s.kind for s in ts.build_plan(cfg(phase="post", with_flex=with_flex))] == [
            "check_account", "configure_number", "ensure_status_callbacks", "verify_setup"]
    assert len(ts.build_plan(cfg(phase="all"))) == 12
    assert len(ts.build_plan(cfg(phase="all", with_flex=False))) == 6


@pytest.mark.parametrize("phase", ["pre", "post", "all"])
def test_apply_without_with_flex_calls_no_workspace_or_taskrouter_endpoint(tmp_path: Path, phase: str) -> None:
    """No Flex API call. The fake's TaskRouter and Flex HTTP endpoints raise on any use, and the
    outputs file has no Flex key."""
    class Tripwire:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"TaskRouter used without --with-flex: {name}")

        def __call__(self, *a: Any, **kw: Any) -> Any:
            raise AssertionError("TaskRouter used without --with-flex")

    class NoFlexTwilio(FakeTwilio):
        def __init__(self) -> None:
            super().__init__(workspace=False)
            self.taskrouter = Tripwire()  # type: ignore[assignment]

        def request(self, *a: Any, **kw: Any) -> Any:
            raise AssertionError("Flex HTTP endpoint used without --with-flex")

    fake = NoFlexTwilio()
    if phase == "post":  # post reads the configuration that pre made
        assert run(tmp_path, fake, "pre") == 0
    assert run(tmp_path, fake, phase) == 0
    out = out_file(tmp_path)
    assert not {"workspaceSid", "workflowSid", "queueSids", "flexInstanceSid"} & set(out)
    assert not [c for c in fake.calls if c[0].split(".")[0] in ("workspaces", "queues", "workflows", "workers", "flex")]
    assert phase == "post" or out["memoryStoreId"]


def test_with_flex_apply_writes_the_flex_keys(tmp_path: Path) -> None:
    assert run(tmp_path, FakeTwilio(), "pre", extra=["--with-flex"]) == 0
    out = out_file(tmp_path)
    assert out["workspaceSid"].startswith("WS") and out["workflowSid"].startswith("WW")
    assert set(out["queueSids"]) == {"support-en", "support-es", "fraud", "disputes"}


def test_post_and_all_need_the_phone_number(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main(["--phase", "post", *DOMAINS]) == 2
    assert "--phone-number is required" in capsys.readouterr().err
    assert ts.main(DOMAINS) == 2


def test_unknown_phase_is_refused() -> None:
    with pytest.raises(SystemExit):
        ts.main(["--phase", "middle"])


# --------------------------------------------------------------------------- pre


def test_pre_creates_the_configuration_without_callbacks_and_never_touches_the_number(tmp_path: Path) -> None:
    fake = FakeTwilio()
    sm = StubSM()
    assert run(tmp_path, fake, "pre", sm=sm) == 0
    assert sm.reads == [API_KEY_ARN, TOKEN_ARN]  # no webhook URLs, so no Basic credentials
    assert not [c for c in fake.calls if c[0].startswith("numbers")]
    (conf,) = fake.configs
    assert conf.status_callbacks is None
    create = next(c for c in fake.calls if c[0] == "configs.create")
    assert not create[1].get("statusCallbacks")
    out = out_file(tmp_path)
    assert out["conversationConfigurationId"] == conf.id
    assert out["verifyServiceSid"].startswith("VA") and out["memoryStoreId"].startswith("mem_store_")
    assert "phoneNumberSid" not in out
    assert fake.numbers[0].voice_url is None


def test_pre_is_idempotent(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    first = out_file(tmp_path)
    fake.calls.clear()
    assert run(tmp_path, fake, "pre") == 0
    assert fake.writes() == [] and out_file(tmp_path) == first


def test_pre_prints_what_comes_next(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Pre runs after OwlBankData and the Twilio secrets: its hint is the table's, and nothing else."""
    assert run(tmp_path, FakeTwilio(), "pre") == 0
    assert ts.NEXT_STEPS["pre"] in capsys.readouterr().out.splitlines()


# --------------------------------------------------------------------------- post


def test_post_sets_webhooks_and_merges_into_the_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    pre = out_file(tmp_path)
    fake.calls.clear()
    assert run(tmp_path, fake, "post") == 0
    num = fake.numbers[0]
    assert num.voice_url == f"https://{USER}:{PASSWORD}@api.owlbank.example/voice/incoming"
    assert num.voice_method == "POST"
    assert fake.configs[0].status_callbacks == [{"url": WEBHOOK, "method": "POST"}]
    assert sorted(c[0] for c in fake.writes()) == ["configs.update", "numbers.update"]
    assert not [c for c in fake.calls if c[0].split(".")[0] in ("queues", "workflows", "stores", "verify")]
    out = out_file(tmp_path)
    assert out["phoneNumberSid"] == num.sid
    assert {k: out[k] for k in pre} == pre  # nothing from pre is lost
    assert out["conversationConfigurationId"] == pre["conversationConfigurationId"]
    cap = capsys.readouterr()
    assert PASSWORD not in cap.out + cap.err + (tmp_path / "t.json").read_text()
    assert API_SECRET not in cap.out + cap.err


def test_post_replaces_a_status_callback_that_carries_credentials(tmp_path: Path,
                                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """A configuration set up before has ``https://user:pass@relay/webhook``; a re-run of post
    rewrites it without credentials, and the read-back compares against the credential-free URL."""
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    assert run(tmp_path, fake, "post") == 0
    fake.configs[0].status_callbacks = [
        {"url": f"https://{USER}:{PASSWORD}@relay.owlbank.example/webhook", "method": "POST"}]
    fake.calls.clear()
    assert run(tmp_path, fake, "post") == 0
    assert [c[0] for c in fake.writes()] == ["configs.update"]
    assert fake.configs[0].status_callbacks == [{"url": WEBHOOK, "method": "POST"}]
    cap = capsys.readouterr()
    assert PASSWORD not in cap.out + cap.err


def test_post_is_idempotent(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    assert run(tmp_path, fake, "post") == 0
    first = out_file(tmp_path)
    fake.calls.clear()
    assert run(tmp_path, fake, "post") == 0
    assert fake.writes() == [] and out_file(tmp_path) == first


def test_post_without_pre_says_to_run_pre(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, FakeTwilio(), "post") == 3
    assert "--phase pre" in capsys.readouterr().err


def test_post_refuses_a_configuration_that_is_not_the_one_the_bridge_got(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    data = out_file(tmp_path)
    data["conversationConfigurationId"] = "conv_configuration_other"
    (tmp_path / "t.json").write_text(json.dumps(data))
    assert run(tmp_path, fake, "post") == 3
    err = capsys.readouterr().err
    assert "differs" in err and "twilioOutputs" in err
    assert PASSWORD not in err


def test_post_read_back_catches_a_webhook_that_did_not_stick(tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    original = fake._number

    def stubborn(sid: str) -> Any:  # accepts the update and ignores it
        inst = original(sid)
        inst.update = lambda **kw: fake.calls.append(("numbers.update", kw))
        return inst

    fake.incoming_phone_numbers._ctx = stubborn
    assert run(tmp_path, fake, "post") == 3
    assert "not what was set" in capsys.readouterr().err


def test_pre_after_post_keeps_the_status_callbacks(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    assert run(tmp_path, fake, "post") == 0
    fake.calls.clear()
    assert run(tmp_path, fake, "pre") == 0
    assert fake.writes() == []
    assert fake.configs[0].status_callbacks == [{"url": WEBHOOK, "method": "POST"}]
    # Even when pre has to update something else, the callbacks are sent back unchanged.
    fake.configs[0].channel_settings = {"VOICE": {"statusTimeouts": {"inactive": 99, "closed": 9}}}
    assert run(tmp_path, fake, "pre") == 0
    update = next(c for c in fake.calls if c[0] == "configs.update")
    assert update[1]["statusCallbacks"] == [{"url": WEBHOOK, "method": "POST"}]
    assert fake.configs[0].status_callbacks == [{"url": WEBHOOK, "method": "POST"}]


def test_all_matches_pre_then_post(tmp_path: Path) -> None:
    one, two = FakeTwilio(), FakeTwilio()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert run(tmp_path / "a", one, "all") == 0
    assert run(tmp_path / "b", two, "pre") == 0 and run(tmp_path / "b", two, "post") == 0

    def state(f: FakeTwilio) -> Any:
        c = f.configs[0]
        return (c.status_callbacks, c.channel_settings, c.memory_extraction_enabled,
                f.numbers[0].voice_url, f.numbers[0].status_callback, len(f.queues), len(f.workflows),
                len(f.verify_services))

    assert state(one) == state(two)
    a, b = out_file(tmp_path / "a"), out_file(tmp_path / "b")
    assert set(a) == set(b)


@settings(max_examples=25, deadline=None)
@given(st.lists(st.sampled_from(["pre", "post", "all"]), min_size=1, max_size=6))
def test_property_any_phase_order_converges_and_rerun_is_a_no_op(phases: list[str]) -> None:
    """Idempotence: whatever order the phases run in, the last phase run again changes nothing, and
    once ``post`` or ``all`` has run the webhooks are set (``pre`` never clears them)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fake = FakeTwilio()
        executed: list[str] = []
        for ph in phases:
            if ph == "post" and not fake.configs:
                continue  # post before pre is refused (tested above)
            assert run(tmp, fake, ph) == 0
            executed.append(ph)
        if not executed:
            return
        fake.calls.clear()
        assert run(tmp, fake, executed[-1]) == 0
        assert fake.writes() == []
        if any(p in ("post", "all") for p in executed):
            assert fake.configs[0].status_callbacks == [{"url": WEBHOOK, "method": "POST"}]
        else:
            assert not fake.configs[0].status_callbacks


# --------------------------------------------------------------------------- status callback by hosting

EXEC_API = "abc123def4.execute-api.us-east-1.amazonaws.com"
SINK = f"https://{USER}:{PASSWORD}@api.owlbank.example/voice/orchestrator"


def outputs_file(tmp_path: Path, **extra: str) -> list[str]:
    """A ``cdk deploy --outputs-file`` with the given outputs per stack; returns the CLI arguments."""
    path = tmp_path / "cdk-outputs.json"
    path.write_text(json.dumps({"OwlBankBridge": extra}))
    return ["--cdk-outputs", str(path)]


def test_status_callback_url_per_hosting() -> None:
    from .test_twilio_setup import cfg

    assert ts.status_callback_url(cfg()) == "https://relay.x/webhook"
    sink = ts.status_callback_url(cfg(bridge_hosting="agentcore", api_base_path="/v1"))
    assert isinstance(sink, ts.CredUrl)
    assert str(sink) == "https://***:***@api.x/v1/voice/orchestrator"
    assert ts.render_webhook(sink, ts.BasicCreds("u ser", "p@ss/w:rd")) == \
        "https://u%20ser:p%40ss%2Fw%3Ard@api.x/v1/voice/orchestrator"
    with pytest.raises(ts.ScriptError, match="bridge hosting"):
        cfg(bridge_hosting="lambda")


def test_agentcore_needs_no_relay_domain_but_fargate_does() -> None:
    from .test_twilio_setup import cfg

    cfg(phase="post", bridge_hosting="agentcore", relay_domain="")
    with pytest.raises(ts.ScriptError, match="relay-domain"):
        cfg(phase="post", relay_domain="")


@pytest.mark.parametrize("phase", ["post", "all"])
def test_agentcore_dry_run_shows_the_sink_and_never_the_password(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], phase: str) -> None:
    def boom(*a: Any) -> Any:
        raise AssertionError("network use in a dry run")

    args = ["--phase", phase, *PHONE,
            *outputs_file(tmp_path, BridgeHosting="agentcore", ApiDomain=EXEC_API, ApiStage="v1")]
    assert ts.main(args, sm_client=NoSM(), client_factory=boom) == 0
    out = capsys.readouterr().out
    assert f"https://***:***@{EXEC_API}/v1/voice/orchestrator" in out
    assert "/webhook" not in out and PASSWORD not in out and USER not in out


def test_missing_bridge_hosting_output_is_fargate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--phase", "post", *PHONE,
            *outputs_file(tmp_path, ApiDomain="api.owlbank.example", RelayDomain="relay.owlbank.example")]
    assert ts.main(args) == 0
    out = capsys.readouterr().out
    assert WEBHOOK in out and "/voice/orchestrator" not in out


def test_fargate_without_a_relay_domain_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--phase", "post", *PHONE, *outputs_file(tmp_path, ApiDomain="api.owlbank.example")]
    assert ts.main(args) == 2
    assert "--relay-domain" in capsys.readouterr().err


def test_an_unknown_bridge_hosting_output_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--phase", "post", *PHONE, *DOMAINS, *outputs_file(tmp_path, BridgeHosting="ecs")]
    assert ts.main(args) == 2
    assert "BridgeHosting" in capsys.readouterr().err


def test_agentcore_post_sets_the_sink_with_basic_and_never_prints_the_password(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    agentcore = outputs_file(tmp_path, BridgeHosting="agentcore")
    assert run(tmp_path, fake, "pre") == 0
    assert run(tmp_path, fake, "post", extra=agentcore) == 0
    assert fake.configs[0].status_callbacks == [{"url": SINK, "method": "POST"}]
    assert fake.numbers[0].voice_url == f"https://{USER}:{PASSWORD}@api.owlbank.example/voice/incoming"
    fake.calls.clear()
    assert run(tmp_path, fake, "post", extra=agentcore) == 0  # idempotent, read-back included
    assert fake.writes() == []
    cap = capsys.readouterr()
    assert PASSWORD not in cap.out + cap.err + (tmp_path / "t.json").read_text()


def test_switching_hosting_moves_the_status_callback_both_ways(tmp_path: Path) -> None:
    """After a hosting switch, ``--phase post`` points the callback at the new place."""
    fake = FakeTwilio()
    agentcore = outputs_file(tmp_path, BridgeHosting="agentcore")
    assert run(tmp_path, fake, "pre") == 0
    assert run(tmp_path, fake, "post") == 0
    assert fake.configs[0].status_callbacks == [{"url": WEBHOOK, "method": "POST"}]
    assert run(tmp_path, fake, "post", extra=agentcore) == 0
    assert fake.configs[0].status_callbacks == [{"url": SINK, "method": "POST"}]
    fake.calls.clear()
    assert run(tmp_path, fake, "post") == 0  # back to fargate: no BridgeHosting output
    assert [c[0] for c in fake.writes()] == ["configs.update"]
    assert fake.configs[0].status_callbacks == [{"url": WEBHOOK, "method": "POST"}]


def test_agentcore_all_creates_the_configuration_with_the_sink(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "all", extra=outputs_file(tmp_path, BridgeHosting="agentcore")) == 0
    assert fake.configs[0].status_callbacks == [{"url": SINK, "method": "POST"}]


def test_status_callbacks_compare_after_percent_decoding():
    """Twilio can return a sink URL's Basic credentials with different percent-encoding; the
    read-back check compares decoded parts, and still catches a real difference."""
    from scripts.twilio_setup import same_callbacks, same_url

    sent = "https://us%2Br:p%40ss%3Aword@api.example.com/v1/voice/orchestrator"
    back = "https://us+r:p%40ss:word@API.example.com/v1/voice/orchestrator"
    want = [{"url": sent, "method": "POST"}]
    assert same_url(sent, back)
    assert same_callbacks([{"url": back, "method": "POST"}], want)
    assert not same_callbacks([{"url": back, "method": "GET"}], want)
    assert not same_callbacks([{"url": back.replace("word@", "w0rd@"), "method": "POST"}], want)
    assert not same_callbacks([{"url": back.replace("/orchestrator", "/status"), "method": "POST"}], want)
    assert not same_callbacks([], want) and not same_callbacks(None, want)
    assert not same_url("https://api.example.com/x", "https://u:p@api.example.com/x")


# --------------------------------------------------------------------------- --check, labels, hints

REPO = Path(__file__).resolve().parents[3]
LISTED_KEY_SECRET = "ListedKeySecretValue0123456789ab"
CRED_VOICE_URL = "https://hook-user:" + PASSWORD + "@api.owlbank.example/voice/incoming?token=QueryTokenValue99"
_WRITE_NAMES = {"create", "update", "delete", "remove", "post", "put"}


class ReadOnly:
    """Wraps a fake so any write-shaped call raises; ``request`` is allowed for GET only."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in _WRITE_NAMES:
            raise AssertionError(f"write call {name} in --check")
        value = getattr(self._inner, name)
        if name == "request":
            def get_only(method: str, *a: Any, **kw: Any) -> Any:
                assert method == "GET", f"{method} request in --check"
                return value(method, *a, **kw)
            return get_only
        return self._wrap(value)

    def __call__(self, *a: Any, **kw: Any) -> Any:
        return self._wrap(self._inner(*a, **kw))

    @staticmethod
    def _wrap(value: Any) -> Any:
        return ReadOnly(value) if callable(value) or type(value).__name__ == "SimpleNamespace" else value


class CheckClients:
    """``client_factory`` for --check: read-only views; the Auth Token client also lists keys."""

    def __init__(self, fake: FakeTwilio) -> None:
        self.fake = fake
        self.made: list[tuple[str, str, str]] = []

    def __call__(self, user: str, password: str, account: str) -> Any:
        self.made.append((user, password, account))
        view = ReadOnly(self.fake.factory(user, password, account))
        if user.startswith("SK"):
            return view
        keys = {"keys": [{"sid": "SK" + "e5" * 16, "friendly_name": ts.RESTRICTED_KEYS["context_api_key"].friendly_name,
                          "secret": LISTED_KEY_SECRET}]}

        def request(method: str, uri: str, **_kw: Any) -> Any:
            assert method == "GET", method
            return SimpleNamespace(status_code=200, text=json.dumps(keys))
        return SimpleNamespace(api=view.api, request=request)


def run_check(fake: FakeTwilio, phase: str = "all", extra: Sequence[str] = ()) -> tuple[int, str, StubSM]:
    out, err, sm = io.StringIO(), io.StringIO(), StubSM()
    args = ["--check", "--phase", phase, "--account-sid", SUB, *SECRET_ARGS,
            *(["--phone-number", NUMBER, *DOMAINS] if phase != "pre" else []), *extra]
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = ts.main(args, sm_client=sm, client_factory=CheckClients(fake))
    return rc, out.getvalue() + err.getvalue(), sm


def test_check_prints_the_account_resources_number_and_application() -> None:
    fake = FakeTwilio()
    fake.numbers[0].voice_url = CRED_VOICE_URL
    fake.numbers[0].voice_application_sid = "AP" + "7" * 32
    fake.numbers[0].account_sid = SUB
    fake.application_rows["AP" + "7" * 32] = SimpleNamespace(voice_url="https://x.example/app", account_sid=SUB)
    rc, out, sm = run_check(fake, "all", ["--with-flex"])
    assert rc == 0, out
    assert "active subaccount" in out and "voice_application_sid AP" + "7" * 32 in out
    assert "https://api.owlbank.example/voice/incoming" in out
    assert "Flex TaskRouter workspace" in out and "queue OwlBank Fraud" in out
    assert BASIC_ARN not in sm.reads


def test_check_reports_a_standard_key_from_another_account() -> None:
    rc, out, _ = run_check(FakeTwilio(key_account="AC" + "9" * 32))
    assert rc == 3 and "can't list Calls" in out and "different accounts" in out


def test_check_reports_a_number_or_application_of_another_account() -> None:
    fake = FakeTwilio()
    other = "AC" + "9" * 32
    fake.numbers[0].account_sid = other
    fake.numbers[0].voice_application_sid = "AP" + "7" * 32
    fake.application_rows["AP" + "7" * 32] = SimpleNamespace(voice_url=None, account_sid=other)
    rc, out, _ = run_check(fake)
    assert rc == 3 and out.count("PROBLEM") == 2 and other in out


def test_check_without_with_flex_calls_no_taskrouter_endpoint() -> None:
    fake = FakeTwilio()
    assert run_check(fake)[0] == 0
    assert not [c for c in fake.calls if c[0].split(".")[0] in ("workspaces", "queues", "workflows")]


def test_check_cannot_be_combined_with_apply_and_needs_the_account(tmp_path: Path) -> None:
    for args in (["--check", "--apply", "--account-sid", SUB], ["--check"]):
        assert ts.main([*args, *SECRET_ARGS, *PHONE, *DOMAINS], sm_client=NoSM(),
                       client_factory=lambda *a: pytest.fail("client built")) == 2


@given(phase=st.sampled_from(["pre", "post", "all"]), with_flex=st.booleans(), main_account=st.booleans(),
       dedicated=st.booleans(), app=st.booleans())
@settings(max_examples=60, deadline=None)
def test_property_check_makes_no_write_call_and_prints_no_secret(
        phase: str, with_flex: bool, main_account: bool, dedicated: bool, app: bool) -> None:
    fake = FakeTwilio(owner=SUB if main_account else PARENT)
    fake.numbers[0].voice_url = CRED_VOICE_URL
    fake.numbers[0].status_callback = CRED_VOICE_URL
    if app:
        fake.numbers[0].voice_application_sid = "AP" + "7" * 32
        fake.application_rows["AP" + "7" * 32] = SimpleNamespace(voice_url=CRED_VOICE_URL, account_sid=SUB)
    extra = [*(["--with-flex"] if with_flex else []), *(["--dedicated-main-account"] if dedicated else [])]
    rc, out, _ = run_check(fake, phase, extra)
    assert rc in (0, 3), out
    assert fake.writes() == [] and fake.refused == []
    for secret in (AUTH_TOKEN, API_SECRET, PASSWORD, LISTED_KEY_SECRET, "QueryTokenValue99"):
        assert secret not in out, secret


def test_apply_prints_the_previous_webhooks_masked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    fake.numbers[0].voice_url = CRED_VOICE_URL
    fake.numbers[0].voice_application_sid = "AP" + "7" * 32
    assert run(tmp_path, fake, "post") == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("previous number"))
    assert "https://api.owlbank.example/voice/incoming" in line and "AP" + "7" * 32 in line
    for secret in (PASSWORD, "hook-user", "QueryTokenValue99", "@"):
        assert secret not in line


@pytest.mark.parametrize("phase", ["pre", "post", "all"])
@pytest.mark.parametrize("with_flex", [False, True])
def test_dry_run_makes_no_twilio_call_and_labels_every_step(
        capsys: pytest.CaptureFixture[str], phase: str, with_flex: bool) -> None:
    def refuse(*_a: Any) -> Any:
        raise AssertionError("Twilio used in a dry run")

    args = ["--phase", phase, *(PHONE + DOMAINS if phase != "pre" else []),
            *(["--with-flex", *(["--create-flex"] if phase != "post" else []),
               *(["--taskrouter-events"] if phase != "pre" else [])] if with_flex else [])]
    assert ts.main(args, sm_client=NoSM(), client_factory=refuse) == 0
    steps = [ln for ln in capsys.readouterr().out.splitlines() if re.match(r"\s+\d+\. ", ln)]
    assert steps and all(re.match(r"\s+\d+\. \w+ \[[^\]]+\]: ", ln) for ln in steps)
    assert ts.ACCOUNT_REQUIRE in steps[0]


def _readme_script_mentions() -> list[str]:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    first = text[text.index("### First deploy"):]
    first = first[:first.index("**Test calls")]
    steps = re.split(r"^\d+\. ", first, flags=re.M)[1:]
    return [m for step in steps for m in re.findall(r"scripts/(\w+\.py)", step)]


def _script_path(name: str) -> Path | None:
    return next((p for p in (REPO / "scripts" / name, REPO / "infra" / "scripts" / name) if p.is_file()), None)


def test_post_hint_names_the_flex_steps_only_with_flex() -> None:
    assert ts.next_steps("post", True) == ts.NEXT_STEPS["post"]
    without = ts.next_steps("post", False)
    assert without == ts.NEXT_STEPS["post_no_flex"] and without is not None
    assert "scripts/" not in without and "test calls" in without
    assert ts.next_steps("pre", False) == ts.NEXT_STEPS["pre"] == ts.next_steps("pre", True)
    assert ts.next_steps("all", True) is None


def test_post_prints_the_hint_for_its_flex_choice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run(tmp_path, fake, "pre") == 0
    assert run(tmp_path, fake, "post") == 0
    lines = capsys.readouterr().out.splitlines()
    assert ts.NEXT_STEPS["post_no_flex"] in lines and ts.NEXT_STEPS["post"] not in lines


def test_check_reports_duplicate_workspaces_by_their_message() -> None:
    fake = FakeTwilio()
    fake.workspaces = [SimpleNamespace(sid="WS" + str(i) * 32, friendly_name=f"{ts.FLEX_WORKSPACE} {n}")
                       for i, n in ((1, "A"), (2, "B"))]
    rc, out, _ = run_check(fake, "all", ["--with-flex"])
    assert rc == 3 and "more than one TaskRouter workspace" in out
    assert "ScriptError (status" not in out


@pytest.mark.parametrize("phase", ["pre", "post"])
def test_next_steps_name_existing_scripts_in_the_readme_order(phase: str) -> None:
    hint = ts.NEXT_STEPS[phase]
    named = re.findall(r"scripts/(\w+\.py)", hint)
    assert named
    for name in named:
        path = _script_path(name)
        assert path is not None, name
        for flag in re.findall(r"(--[a-z][a-z-]+)", hint[hint.index(name):].split(", then")[0]):
            source = path.read_text(encoding="utf-8") + (REPO / "scripts" / "_common.py").read_text("utf-8")
            assert f'"{flag}"' in source, (name, flag)
    # The Flex steps of the post hint live in the Add Flex recipe, which follows the README's first deploy.
    add_flex = re.findall(r"scripts/(\w+\.py)", (REPO / "docs" / "add-flex.md").read_text(encoding="utf-8"))
    mentions = iter([*_readme_script_mentions(), *add_flex])
    assert all(name in mentions for name in named)  # an ordered subsequence of the README's mentions
