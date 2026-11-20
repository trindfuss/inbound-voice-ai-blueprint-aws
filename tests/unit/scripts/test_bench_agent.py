"""scripts/bench/agent_bench.py: the Strands agent, offline.

No AWS: the Gateway and Memory factories are monkeypatched, the stack outputs and the synthesized
bridge template are temporary files, and the end-to-end run uses the real ``StrandsAgentClient`` on a
scripted ``FakeModel``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

import tac_bridge.agent_client_strands as acs
from scripts.bench import agent_bench as ab
from tac_bridge.agent_client import TextBeforeTool, TextDelta, ToolCall, TurnEnd
from tests.strands_fakes import FakeGatewayProvider, FakeModel, settings, text_msg, tool_msg, usage

BASELINE = Path(__file__).resolve().parents[2] / "fixtures" / "bench" / "baseline-harness-20261014T132906Z.json"


@pytest.fixture(autouse=True)
def _no_capture(monkeypatch):
    """Each test starts without the log capture; whatever a test installs is removed after it."""
    monkeypatch.setattr(ab, "CAPTURE", None)
    logger = logging.getLogger(ab.STRANDS_LOGGER)
    level, handlers = logger.level, list(logger.handlers)
    yield
    for h in list(logger.handlers):
        if h not in handlers:
            logger.removeHandler(h)
    logger.setLevel(level)


def _template(path: Path, *, thinking: Any = "adaptive", model_id: Any = "us.anthropic.claude-haiku-5-5",
              max_tokens: Any = 2000, containers: int = 1, text_release: Any = "stream",
              fallbacks: int = 0) -> Path:
    """A synthesized bridge template: the task definition's bridge container with the agent's
    model list in ``OWLBANK_MODELS`` (the given preferred entry, then ``fallbacks`` more), and,
    beside it, a container without it."""
    entries = [{"id": model_id, "thinking": thinking, "max_tokens": max_tokens, "text_release": text_release,
                "prompt_caching": False}]
    entries += [{"id": f"us.anthropic.claude-fallback-{i}", "thinking": "off", "max_tokens": 400,
                 "text_release": "message", "prompt_caching": False} for i in range(fallbacks)]
    env = [{"Name": "OWLBANK_MODELS", "Value": json.dumps(entries)}, {"Name": "AWS_REGION", "Value": "us-east-1"}]
    defs = [{"Name": "bridge", "Environment": env} for _ in range(containers)]
    defs.append({"Name": "sidecar", "Environment": [{"Name": "X", "Value": "y"}]})
    path.write_text(json.dumps({"Resources": {
        "Cluster": {"Type": "AWS::ECS::Cluster", "Properties": {}},
        "TaskDef": {"Type": "AWS::ECS::TaskDefinition", "Properties": {"ContainerDefinitions": defs}},
    }}))
    return path


@pytest.fixture
def stack(tmp_path, monkeypatch):
    """Stack outputs as after the deploy (the agent stack's own GatewayUrl and MemoryId, next
    to the cross-stack export that also contains ``MemoryId``), and the synthesized template."""
    outputs = tmp_path / "cdk-outputs.json"
    outputs.write_text(json.dumps({"OwlBankAgent": {
        "ExportsOutputFnGetAttMemoryMemoryId5865BD41": "export-memory",
        "GatewayUrl": "https://gw.example.invalid/mcp",
        "MemoryId": "mem-bench-1",
    }}))
    monkeypatch.setattr(ab, "OUTPUTS", outputs)
    template = _template(tmp_path / "OwlBankBridge.template.json")
    monkeypatch.setattr(ab.deployed_model_settings, "__defaults__", (template,))
    calls: dict[str, Any] = {}

    def gateway(url, region, **kw):
        calls["gateway"] = (url, region)
        return FakeGatewayProvider([])

    def memory(memory_id, region, **kw):
        calls["memory"] = (memory_id, region)
        return lambda session_id, mode: None

    monkeypatch.setattr(acs, "make_gateway_tool_provider", gateway)
    monkeypatch.setattr(acs, "memory_session_manager_factory", memory)
    return calls


def _args(*argv: str):
    return ab.build_parser().parse_args(["--agent", "strands", *argv])


# ----------------------------------------------------------------------------- make_strands wiring


def test_strands_is_the_only_agent_and_the_default():
    assert {"strands": ab.make_strands} == ab.AGENTS
    assert ab.build_parser().parse_args([]).agent == "strands"
    with pytest.raises(SystemExit):  # the harness agent is gone
        ab.build_parser().parse_args(["--agent", "harness"])


@pytest.mark.parametrize(("flag", "caching"), [((), False), (("--prompt-caching", "off"), False),
                                               (("--prompt-caching", "on"), True)])
def test_make_strands_builds_the_client_from_outputs_and_bridge_settings(stack, flag, caching):
    client, warm, info = ab.make_strands(_args(*flag))
    assert isinstance(client, acs.StrandsAgentClient)
    s = client.settings
    assert (s.primary.id, s.primary.max_tokens, s.primary.thinking) == ("us.anthropic.claude-haiku-5-5", 2000, "adaptive")
    assert s.primary.prompt_caching is caching
    assert (s.gateway_url, s.memory_id, s.region) == ("https://gw.example.invalid/mcp", "mem-bench-1", "us-east-1")
    assert stack == {"gateway": ("https://gw.example.invalid/mcp", "us-east-1"), "memory": ("mem-bench-1", "us-east-1")}
    assert info["backend"] == "strands" and info["prompt_caching"] is caching
    assert (info["model_id"], info["max_tokens"], info["thinking"]) == (s.primary.id, s.primary.max_tokens, s.primary.thinking)
    assert info["model_settings_source"].endswith("OwlBankBridge.template.json")
    assert info["session_prefix"] == "bench-" and info["memory_actor_id"] == acs.MEMORY_ACTOR_ID
    assert info["packages"]["strands-agents"] == version("strands-agents")
    assert info["packages"]["bedrock-agentcore"] == version("bedrock-agentcore")
    assert ab.CAPTURE is not None
    assert asyncio.iscoroutinefunction(warm)


def test_strands_warm_is_the_bridges_session_start_prebuild(stack, monkeypatch):
    client, warm, _ = ab.make_strands(_args())
    seen = []

    async def begin_session(session_id, *, history="reload"):
        seen.append((session_id, history))

    monkeypatch.setattr(client, "begin_session", begin_session)
    asyncio.run(warm("bench-abc"))
    assert seen == [("bench-abc", "new")]


def test_explicit_settings_override_and_are_recorded(stack):
    client, _, info = ab.make_strands(_args("--model-id", "us.anthropic.other", "--thinking", "off",
                                            "--max-tokens", "400", "--gateway-url", "https://g2.invalid",
                                            "--memory-id", "mem-2", "--text-release", "message"))
    s = client.settings
    assert (s.primary.id, s.primary.thinking, s.primary.max_tokens, s.gateway_url, s.memory_id, s.primary.text_release) == (
        "us.anthropic.other", "off", 400, "https://g2.invalid", "mem-2", "message")
    assert info["model_settings_source"] == "command line"


def test_one_override_keeps_the_rest_from_the_template(stack):
    client, _, info = ab.make_strands(_args("--thinking", "default"))
    s = client.settings
    assert (s.primary.id, s.primary.thinking, s.primary.max_tokens, s.primary.text_release) == (
        "us.anthropic.claude-haiku-5-5", "default", 2000, "stream")
    assert info["model_settings_source"].endswith("OwlBankBridge.template.json (overridden: thinking)")


def test_all_three_settings_need_no_template(stack, tmp_path, monkeypatch):
    monkeypatch.setattr(ab.deployed_model_settings, "__defaults__", (tmp_path / "absent.json",))
    client, _, _ = ab.make_strands(_args("--model-id", "m", "--thinking", "adaptive", "--max-tokens", "400"))
    assert (client.settings.primary.id, client.settings.primary.max_tokens) == ("m", 400)
    assert client.settings.primary.text_release == "stream"  # no template: the bridge's default


def test_missing_gateway_output_is_a_clear_error(stack, tmp_path, monkeypatch):
    (tmp_path / "cdk-outputs.json").write_text(json.dumps({"OwlBankAgent": {"MemoryId": "m"}}))
    with pytest.raises(SystemExit, match="GatewayUrl"):
        ab.make_strands(_args())


@pytest.mark.parametrize("mode", ["adaptive", "off", "default"])
def test_model_settings_from_the_synthesized_bridge(tmp_path, mode):
    t = _template(tmp_path / "t.json", thinking=mode, max_tokens=777)
    m = ab.deployed_model_settings(t)
    assert (m["model_id"], m["max_tokens"], m["thinking"]) == ("us.anthropic.claude-haiku-5-5", 777, mode)


def test_model_settings_are_the_preferred_entry_of_the_list(tmp_path):
    """Model failover: ``OWLBANK_MODELS`` holds the list; the benchmark runs its first entry."""
    m = ab.deployed_model_settings(_template(tmp_path / "t.json", fallbacks=2))
    assert m["model_id"] == "us.anthropic.claude-haiku-5-5"
    assert m["models"] == ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-fallback-0",
                           "us.anthropic.claude-fallback-1"]


def test_a_template_with_the_old_single_model_variables_stops_the_run(tmp_path):
    path = tmp_path / "old.json"
    env = [{"Name": "OWLBANK_MODEL_ID", "Value": "us.anthropic.claude-haiku-5-5"}]
    path.write_text(json.dumps({"Resources": {"TaskDef": {"Type": "AWS::ECS::TaskDefinition", "Properties": {
        "ContainerDefinitions": [{"Name": "bridge", "Environment": env}]}}}}))
    with pytest.raises(SystemExit, match="no single bridge container with OWLBANK_MODELS"):
        ab.deployed_model_settings(path)


@pytest.mark.parametrize("bad", [{"thinking": "deep"}, {"max_tokens": "many"}, {"model_id": {"Ref": "X"}},
                                 {"text_release": "sentence"}, {"text_release": None}])
def test_unrecognised_model_settings_stop_the_run(tmp_path, bad):
    with pytest.raises(SystemExit, match="unrecognised"):
        ab.deployed_model_settings(_template(tmp_path / "t.json", **bad))


def test_an_unrecognised_text_release_names_the_text_release_flag(tmp_path):
    with pytest.raises(SystemExit, match=r"text_release 'Stream' in OWLBANK_MODELS; pass --text-release \(message or stream\)"):
        ab.deployed_model_settings(_template(tmp_path / "t.json", text_release="Stream"))


def test_no_template_or_no_single_bridge_container_stops_the_run(tmp_path):
    with pytest.raises(SystemExit, match="--model-id, --max-tokens and --thinking"):
        ab.deployed_model_settings(tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="no single bridge container"):
        ab.deployed_model_settings(_template(tmp_path / "t.json", containers=2))


@pytest.mark.parametrize(("argv", "label"), [
    (["--agent", "strands"], "strands-nocache"),
    (["--agent", "strands", "--prompt-caching", "on"], "strands-cache"),
    (["--agent", "strands", "--cold"], "strands-nocache-cold"),
    ([], "strands-nocache"),
])
def test_default_labels(argv, label):
    assert ab.default_label(ab.build_parser().parse_args(argv)) == label


# ----------------------------------------------------------------------------- a fake run


def _fake_strands_run(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    model = FakeModel([
        tool_msg(("list_recent_transactions", {"accountType": "checking"}),
                 use=usage(100, 20, cache_read=0, cache_write=900)),
        text_msg("Your last four transactions were a coffee and some groceries.",
                 use=usage(150, 30, cache_read=900, cache_write=0)),
    ])
    client = acs.StrandsAgentClient(settings(), model_factory=lambda _s, _sid: model,
                                    session_manager_factory=lambda sid, mode: None)
    ab.install_capture()

    async def warm(session_id: str) -> None:
        await client.begin_session(session_id, history="new")

    async def go() -> dict[str, Any]:
        try:
            return await ab.run_scenario(client, warm, "transactions_inline", ab.SCENARIOS["transactions_inline"],
                                         cold=False, session_prefix="bench-", settle_s=0)
        finally:
            await client.aclose()

    run = asyncio.run(go())
    args = ab.build_parser().parse_args(["--agent", "strands", "--runs", "1"])
    path = ab.write_result(tmp_path, label=ab.default_label(args), args=args,
                           backend={"backend": "strands", "prompt_caching": False}, runs=[run])
    return run, path


def test_fake_run_records_usage_model_calls_and_history(tmp_path):
    run, path = _fake_strands_run(tmp_path)
    assert run["session"].startswith("bench-")
    hops = run["steps"][0]["hops"]
    assert [h["tools"] for h in hops] == [["list_recent_transactions"], []]
    assert hops[0]["usage"]["cache_write_tokens"] == 900
    assert hops[1]["usage"]["cache_read_tokens"] == 900
    assert [len(h["model_calls"]) for h in hops] == [1, 1]
    assert all(c["model_latency_s"] is not None and c["model_latency_s"] >= 0 for h in hops for c in h["model_calls"])
    assert hops[1]["model_calls"][0]["cache_read_tokens"] == 900
    # the warm-up's pre-build (history="new"): one empty build, no Memory read
    assert [(b["event"], b["outcome"]) for b in run["history_builds"]] == [("agent_history_new", "ok")]

    data = json.loads(path.read_text())
    assert path.name.startswith("strands-nocache-") and path.suffix == ".json"
    assert data["backend"] == {"backend": "strands", "prompt_caching": False}
    o = data["overall"]
    assert o["hops"] == 2 and o["model_calls"] == 2
    assert o["usage"] == {"input_tokens": 250, "output_tokens": 50, "cache_read_tokens": 900, "cache_write_tokens": 900}
    assert o["cache_hit_ratio"] == round(900 / (250 + 900 + 900), 3)
    assert o["overhead_per_model_call_p50"] is not None
    step = data["summary"]["transactions_inline#1"]
    assert step["cache_read_tokens_mean"] == 900 and step["cache_write_tokens_mean"] == 900


# ----------------------------------------------------------------------------- compare


def _old_format() -> dict[str, Any]:
    """A result file from an older benchmark: no backend, no overall, hops without usage or model calls,
    and a summary without the masking fields."""
    hop = {"first_event_s": 2.0, "first_text_s": 2.0, "tool_call_s": None, "end_s": 3.0, "tools": [],
           "text": "", "suppressed_chars": 0, "stop": "end_turn"}
    return {"label": "baseline-harness", "agent": "harness", "cold": False, "runs": 1, "errors": 0,
            "summary": {"greeting#1": {"n": 1, "response_p50": 3.0, "response_p90": 3.2, "hop_end_p50": 3.0}},
            "raw": [{"scenario": "greeting", "steps": [{"say": "hi", "hops": [hop], "response_s": 3.0}]}]}


def _new_format(caching: bool, response: float) -> dict[str, Any]:
    hop = {"end_s": response, "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 80 if caching else 0,
                                         "cache_write_tokens": 0},
           "model_calls": [{"model_latency_s": response - 0.2, "model_ttfb_s": 0.5}]}
    runs = [{"scenario": "greeting", "steps": [{"hops": [hop], "response_s": response}]}]
    return {"label": "strands-cache" if caching else "strands-nocache", "agent": "strands", "cold": False,
            "backend": {"backend": "strands", "prompt_caching": caching},
            "summary": {"greeting#1": {"n": 1, "response_p50": response, "response_p90": response,
                                       "first_audio_p50": response, "max_silence_p50": response}},
            "overall": ab.overall(runs), "raw": runs}


def test_compare_two_files_old_against_new():
    lines = ab.compare_lines([_old_format(), _new_format(False, 1.5)])
    text = "\n".join(lines)
    assert "A: baseline-harness [harness]" in text and "B: strands-nocache [strands, cache off]" in text
    row = next(x for x in lines if x.startswith("greeting#1") and "response_p50" in x)
    assert "3.00" in row and "1.50" in row and "-50%" in row
    fa = next(x for x in lines if "first_audio_p50" in x)
    assert "-" in fa.split()[2:3]  # missing in the old file
    assert any(x.startswith("overall") and "hop_end_p50" in x for x in lines)  # computed from the old raw


def test_compare_three_way_reports_each_step_and_the_total():
    lines = ab.compare_lines([_old_format(), _new_format(False, 2.0), _new_format(True, 1.5)])
    head = next(x for x in lines if x.startswith("step"))
    assert "B vs A" in head and "C vs B" in head and "C vs A" in head
    row = next(x for x in lines if x.startswith("greeting#1") and "response_p50" in x)
    assert row.split()[-3:] == ["-33%", "-25%", "-50%"]
    ratio = next(x for x in lines if "cache_hit_ratio" in x)
    assert "0.00" in ratio and "0.89" in ratio


@pytest.mark.parametrize("n", [1, 4])
def test_compare_takes_two_or_three(n):
    with pytest.raises(ValueError):
        ab.compare_lines([_old_format()] * n)


def test_compare_cli_rejects_one_file():
    with pytest.raises(SystemExit):
        asyncio.run(ab.main(["--compare", str(BASELINE)]))


def test_compare_cli_against_the_real_baseline(tmp_path, capsys):
    new = tmp_path / "strands-nocache-x.json"
    new.write_text(json.dumps(_new_format(False, 1.5)))
    asyncio.run(ab.main(["--compare", str(BASELINE), str(new), str(new)]))
    out = capsys.readouterr().out
    assert "greeting#1" in out and "strands-nocache [strands, cache off]" in out


# ----------------------------------------------------------------------------- --tracing on|off


class _Handle:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.tracer_provider = self
        self.flushed_ms: int | None = None

    def force_flush(self, timeout_ms: int) -> bool:
        self.events.append("force_flush")
        self.flushed_ms = timeout_ms
        return True

    def shutdown(self) -> bool:
        self.events.append("shutdown")
        return True


def test_tracing_defaults_to_off_and_labels_record_it():
    assert _args().tracing == "off" and _args().traces_log_group == "/owlbank/traces/bridge"
    assert ab.default_label(_args("--tracing", "on")) == "strands-nocache-tracing-on"
    assert ab.default_label(_args("--tracing", "on", "--prompt-caching", "on")) == "strands-cache-tracing-on"
    assert ab.default_label(_args("--tracing", "on", "--cold")) == "strands-nocache-cold-tracing-on"
    assert ab.default_label(_args("--tracing", "off")) == "strands-nocache"
    with pytest.raises(SystemExit):
        _args("--tracing", "maybe")


def test_tracing_off_never_imports_owlbank_telemetry():
    """A fresh interpreter: loading the bench, parsing the default arguments and building the label
    must leave ``owlbank.telemetry`` unimported (the default run is the untraced baseline)."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r}); sys.path.insert(0, {str(root / 'src')!r})\n"
        "from scripts.bench import agent_bench as ab\n"
        "args = ab.build_parser().parse_args([])\n"
        "ab.default_label(args)\n"
        "bad = [m for m in sys.modules if m == 'owlbank.telemetry' or m.startswith('owlbank.telemetry.')]\n"
        "print('IMPORTED' if bad else 'CLEAN', bad)\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,  # noqa: S603
                         check=True, timeout=120)
    assert out.stdout.startswith("CLEAN"), out.stdout + out.stderr


def test_main_with_tracing_off_does_not_start_tracing(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "start_tracing", lambda *_a, **_k: pytest.fail("tracing started while off"))
    monkeypatch.setattr(ab, "flush_tracing", lambda *_a, **_k: pytest.fail("flush while off"))
    monkeypatch.setitem(ab.AGENTS, "strands", lambda _args: (_FakeAgent(), None, {"backend": "strands"}))
    asyncio.run(ab.main(["--runs", "0", "--out", str(tmp_path)]))
    (path,) = tmp_path.glob("*.json")
    saved = json.loads(path.read_text())
    assert saved["tracing"] == "off" and saved["label"] == "strands-nocache"


class _FakeAgent:
    async def aclose(self) -> None:
        pass


def test_main_with_tracing_on_installs_before_the_client_and_flushes_at_the_end(tmp_path, monkeypatch):
    order: list[str] = []
    handle = _Handle()
    monkeypatch.setattr(ab, "start_tracing", lambda args: order.append("start") or handle)
    monkeypatch.setattr(ab, "flush_tracing", lambda h: order.append("flush") or h.shutdown())
    monkeypatch.setitem(ab.AGENTS, "strands", lambda _args: (order.append("client"), (_FakeAgent(), None))[1])
    asyncio.run(ab.main(["--runs", "0", "--tracing", "on", "--out", str(tmp_path)]))
    assert order == ["start", "client", "flush"] and handle.events == ["shutdown"]
    (path,) = tmp_path.glob("*.json")
    saved = json.loads(path.read_text())
    assert saved["tracing"] == "on" and saved["label"] == "strands-nocache-tracing-on"
    assert "tracing on" in ab.describe(saved)


def test_start_tracing_installs_the_bridge_telemetry_with_the_expected_settings(capsys):
    seen: dict[str, Any] = {}
    handle = _Handle()

    def install(settings, **kw):
        seen["settings"], seen["kw"] = settings, kw
        return handle

    got = ab.start_tracing(_args("--tracing", "on", "--region", "us-west-2"), install=install)
    assert got is handle
    s = seen["settings"]
    assert (s.service_name, s.stage, s.hosting) == ("owlbank-dev-bridge", "dev", "fargate")
    assert (s.sample_rate, s.content) == (1.0, "off")
    assert (s.traces_log_group, s.region) == ("/owlbank/traces/bridge", "us-west-2")
    assert seen["kw"]["environ"] == ab.tracing_environment("us-west-2")  # the same variables the bridge reads
    assert "redact" not in seen["kw"]  # content off: no redactor, as the bridge builds none
    assert "/owlbank/traces/bridge" in capsys.readouterr().out


def test_start_tracing_uses_a_given_log_group():
    seen = {}
    ab.start_tracing(_args("--tracing", "on", "--traces-log-group", "/x/traces"),
                     install=lambda s, **_k: seen.setdefault("s", s) and _Handle())
    assert seen["s"].traces_log_group == "/x/traces"


def test_start_tracing_stops_the_run_when_the_install_is_refused():
    with pytest.raises(SystemExit, match="refused or failed"):
        ab.start_tracing(_args("--tracing", "on"), install=lambda *_a, **_k: None)


def test_flush_tracing_flushes_then_stops_and_never_raises(capsys):
    h = _Handle()
    ab.flush_tracing(h)
    assert h.events == ["force_flush", "shutdown"] and h.flushed_ms == 15_000

    class Broken:
        class tracer_provider:  # noqa: N801
            @staticmethod
            def force_flush(_ms):
                raise RuntimeError("boom")

        def shutdown(self):
            raise AssertionError("not reached")

    ab.flush_tracing(Broken())
    assert "tracing flush failed: RuntimeError" in capsys.readouterr().err


# ----------------------------------------------------------------------------- --text-release


@pytest.mark.parametrize(("flag", "mode"), [((), "stream"), (("--text-release", "stream"), "stream"),
                                            (("--text-release", "message"), "message")])
def test_text_release_reaches_the_settings_and_the_backend_info(stack, flag, mode):
    client, _warm, info = ab.make_strands(_args(*flag))
    assert client.settings.primary.text_release == mode
    assert info["text_release"] == mode


def test_text_release_rejects_other_values():
    with pytest.raises(SystemExit):
        _args("--text-release", "sentence")


@pytest.mark.parametrize(("argv", "label"), [
    ([], "strands-nocache"),
    (["--text-release", "stream"], "strands-nocache"),
    (["--text-release", "message"], "strands-nocache-message"),
    (["--text-release", "message", "--prompt-caching", "on", "--cold"], "strands-cache-cold-message"),
])
def test_default_label_includes_a_non_default_text_release(argv, label):
    assert ab.default_label(_args(*argv)) == label


@pytest.mark.parametrize(("flag", "mode"), [((), "stream"), (("--text-release", "message"), "message")])
def test_result_file_records_text_release(tmp_path, flag, mode):
    # The run's text release as main resolved it into args (the flag, else the
    # synthesized bridge's), not a field the backend info may lack.
    args = _args("--runs", "1", *flag)
    args.text_release = mode
    path = ab.write_result(tmp_path, label=ab.default_label(args), args=args,
                           backend={"backend": "strands"}, runs=[])
    data = json.loads(path.read_text())
    assert data["text_release"] == mode
    assert f"text {mode}" in ab.describe(data)


def test_old_result_file_without_text_release_still_loads_and_compares(tmp_path, capsys):
    old = _new_format(False, 1.5)
    assert "text_release" not in old
    assert "text" not in ab.describe(old).split("[")[1]
    new = {**_new_format(False, 1.2), "text_release": "message", "label": "strands-nocache-message"}
    paths = []
    for i, d in enumerate((old, new)):
        paths.append(tmp_path / f"r{i}.json")
        paths[-1].write_text(json.dumps(d))
    asyncio.run(ab.main(["--compare", *map(str, paths)]))
    out = capsys.readouterr().out
    assert "strands-nocache [strands, cache off]" in out and "text message" in out
    saved = json.loads(BASELINE.read_text())
    assert "text_release" not in saved  # a real file from before the flag
    asyncio.run(ab.main(["--resummarize", str(BASELINE)]))
    assert "re-summarized" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["stream", "message"])
def test_text_release_defaults_to_the_synthesized_bridges(stack, tmp_path, monkeypatch, mode):
    template = _template(tmp_path / "OwlBankBridge.template.json", text_release=mode)
    monkeypatch.setattr(ab.deployed_model_settings, "__defaults__", (template,))
    assert ab.deployed_model_settings(template)["text_release"] == mode
    client, _, info = ab.make_strands(_args())
    assert client.settings.primary.text_release == info["text_release"] == mode
    assert info["model_settings_source"].endswith("OwlBankBridge.template.json")
    other = "message" if mode == "stream" else "stream"
    client, _, info = ab.make_strands(_args("--text-release", other))
    assert client.settings.primary.text_release == other
    assert info["model_settings_source"].endswith("(overridden: text_release)")


@pytest.mark.parametrize("argv", [
    ("--thinking", "off"),  # the template's stream release
    ("--thinking", "off", "--text-release", "stream"),
    ("--model-id", "m", "--thinking", "off", "--max-tokens", "400"),  # no template: stream
])
def test_stream_release_with_thinking_off_stops_the_run(stack, argv):
    with pytest.raises(SystemExit, match="text_release=stream needs thinking adaptive or default.*--text-release message"):
        ab.make_strands(_args(*argv))


def test_stream_release_with_thinking_off_in_the_template_stops_the_run(stack, tmp_path, monkeypatch):
    template = _template(tmp_path / "OwlBankBridge.template.json", thinking="off")
    monkeypatch.setattr(ab.deployed_model_settings, "__defaults__", (template,))
    with pytest.raises(SystemExit, match="text_release=stream needs thinking"):
        ab.make_strands(_args())
    assert ab.make_strands(_args("--text-release", "message"))[0].settings.primary.thinking == "off"


def test_main_records_the_bridges_text_release_in_the_label_and_file(stack, tmp_path, monkeypatch):
    template = _template(tmp_path / "OwlBankBridge.template.json", text_release="message")
    monkeypatch.setattr(ab.deployed_model_settings, "__defaults__", (template,))

    async def no_run(*_a, **_k):
        return {"scenario": "greeting", "session": "bench-x", "steps": []}

    monkeypatch.setattr(ab, "run_scenario", no_run)
    asyncio.run(ab.main(["--runs", "1", "--scenarios", "greeting", "--out", str(tmp_path / "out")]))
    (path,) = (tmp_path / "out").iterdir()
    assert path.name.startswith("strands-nocache-message-")
    assert json.loads(path.read_text())["text_release"] == "message"


# ----------------------------------------------------------------------------- narration and <silent/> per hop (5)


class _Scripted:
    """An agent client whose one turn yields fixed events."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events

    async def turn(self, _sid, **_kw):
        for ev in self.events:
            yield ev


def _hop(events: list[Any]) -> dict[str, Any]:
    return asyncio.run(ab.run_hop(_Scripted(events), "bench-x", {"haiku-5-5": "p"}, user_text="hi", results=[]))


def test_hop_records_text_before_tool_count_and_chars():
    hop = _hop([TextDelta("Let me check. "), TextBeforeTool(14, "list_recent_transactions"),
                ToolCall("tu1", "list_recent_transactions", {}), TurnEnd("tool_use")])
    assert (hop["text_before_tool"], hop["text_before_tool_chars"]) == (1, 14)
    assert hop["text"] == "Let me check. " and hop["first_text_s"] is not None
    plain = _hop([TextDelta("Hello."), TurnEnd("end_turn")])
    assert (plain["text_before_tool"], plain["text_before_tool_chars"]) == (0, 0)


@pytest.mark.parametrize("chunks", [["<sil", "ent/>"], ["<", "silent", "/>"], ["  ", "<silent/>", "\n"],
                                    ["<sil<silent/>ent/>"]])
def test_a_silent_marker_split_across_chunks_is_no_first_text(chunks):
    hop = _hop([TextDelta(c) for c in chunks] + [TurnEnd("end_turn")])
    assert hop["first_text_s"] is None
    assert "<" not in hop["text"] and not hop["text"].strip()


def test_text_after_a_split_marker_times_the_first_text():
    hop = _hop([TextDelta("<sil"), TextDelta("ent/>"), TextDelta("Okay."), TurnEnd("end_turn")])
    assert hop["text"] == "Okay." and hop["first_text_s"] is not None


def test_overall_counts_hops_with_text_before_tool_and_old_files_count_none():
    def run(*hops):
        return {"scenario": "s", "steps": [{"hops": [{"end_s": 1.0, **h} for h in hops]}]}

    o = ab.overall([run({"text_before_tool": 1, "text_before_tool_chars": 14}, {"text_before_tool": 0,
                                                                                "text_before_tool_chars": 0}),
                    run({"text_before_tool": 1, "text_before_tool_chars": 6})])
    assert (o["text_before_tool_hops"], o["text_before_tool_chars"]) == (2, 20)
    old = ab.overall(json.loads(BASELINE.read_text())["raw"])  # a real file from before the field
    assert (old["text_before_tool_hops"], old["text_before_tool_chars"]) == (0, 0)


# ----------------------------------------------------------------------------- the split system prompt


def _bridge_inputs(lang: str, key: str, now):
    """What the bridge sends for the bench's verified customer with no memory: the rendered parts
    (``agent_prompt.render``) and the clock line (``Bridge._clock_line``)."""
    from owlbank import agent_prompt
    from owlbank.catalog import PromptCatalog
    from tac_bridge.clock import local_now_line, zone
    catalog = PromptCatalog.from_dict(json.loads((ab.ROOT / "data" / "prompt_catalog.json").read_text()))
    rendered = agent_prompt.render(catalog, key, lang, caller=agent_prompt.CallerFacts("Thomas", ab.BENCH_PROFILE, None),
                                   web_block=None)
    return rendered, local_now_line(now, zone("America/New_York"), lang)


@pytest.mark.parametrize("lang", ["en-US", "es-US"])
def test_the_bench_request_is_the_bridges_static_part_and_suffix(lang):
    import datetime
    now = datetime.datetime(2026, 10, 10, 15, 0, tzinfo=datetime.UTC)
    ids = ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-sonnet-5-5")
    prompts, suffix = ab.system_prompts(ids, lang), ab.system_suffix(lang, now)
    assert set(prompts) == {"haiku-5-5", "sonnet-5-5"}
    for key, static in prompts.items():
        rendered, clock = _bridge_inputs(lang, key, now)
        assert static == rendered.static
        assert suffix == f"{rendered.per_call}\n{clock}"
        assert ab.system_prompt(lang, key) == rendered


def test_run_hop_sends_the_suffix_after_the_cache_point():
    seen: dict[str, Any] = {}

    class Spy:
        async def turn(self, _sid, **kw):
            seen.update(kw)
            yield TurnEnd("end_turn")

    asyncio.run(ab.run_hop(Spy(), "bench-x", {"haiku-5-5": "static"}, user_text="hi", results=[], suffix="tail"))
    assert (seen["system_prompts"], seen["system_suffix"]) == ({"haiku-5-5": "static"}, "tail")
