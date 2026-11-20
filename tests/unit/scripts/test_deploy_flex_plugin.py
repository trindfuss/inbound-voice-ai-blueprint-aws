"""scripts/deploy_flex_plugin.py: a fake runner records every argument list.

No property-based tests for the new behaviour: it is I/O glue over the Twilio CLI, so the tests are
example-based with an injected command runner and ``exists``. The URL validator's property tests below pre-date it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts import deploy_flex_plugin as d
from scripts._common import ScriptError

URL = "https://abc123.execute-api.us-east-1.amazonaws.com/v1"
SID = "AC" + "a" * 32
PKG = {"name": "plugin-owlbank-context", "version": "0.1.0"}
SECRET = "SECRET-TOKEN-VALUE"  # noqa: S105


class FakeRunner:
    def __init__(self, plugin_dir: Path, *, fail: dict[str, d.CommandFailed] | None = None,
                 plugins: str = "@twilio-labs/plugin-flex 6.0.0",
                 bump_to: str | None = "1.0.0", profiles: list | None = None, fetched: str = SID,
                 versions: list[str] | None = None, released: list[str] | None = None) -> None:
        self.calls: list[tuple[list[str], Path | None]] = []
        self.dir = plugin_dir
        self.fail = fail or {}
        self.plugins = plugins
        self.bump_to = bump_to
        self.profiles = [{"id": "dev", "accountSid": SID, "active": False}] if profiles is None else profiles
        self.fetched = fetched
        self.versions = versions or []
        self.released = released or []

    def __call__(self, args, cwd=None) -> str:
        args = list(args)
        self.calls.append((args, cwd))
        key = args[1] if len(args) > 1 else args[0]
        if key in self.fail:
            raise self.fail[key]
        if args == ["twilio", "--version"]:
            return f"twilio-cli/6.0.0 token={SECRET}\n"
        if args == ["twilio", "plugins"]:
            return self.plugins
        if key == "profiles:list":
            return json.dumps(self.profiles)
        if key == "api:flex:v1:configuration:fetch":
            return json.dumps([{"accountSid": self.fetched, "status": "ok"}])
        if key == "flex:plugins:list:plugin-versions":
            return json.dumps([{"name": PKG["name"], "version": v} for v in self.versions])
        if key == "flex:plugins:list:releases":
            return json.dumps([{"sid": "FK1", "plugins": [{"name": PKG["name"], "version": v}]}
                               for v in self.released])
        if key == "flex:plugins:deploy" and self.bump_to:
            self.versions = [*self.versions, self.bump_to]
            (self.dir / "package.json").write_text(json.dumps({**PKG, "version": self.bump_to}), encoding="utf-8")
        return ""

    @property
    def argv(self) -> list[list[str]]:
        return [c[0] for c in self.calls]


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    (tmp_path / "package.json").write_text(json.dumps(PKG), encoding="utf-8")
    return tmp_path


def run_main(plugin_dir: Path, runner: FakeRunner, *extra: str, outputs=None, env_writes=None, exists=None) -> int:
    def writer(path: Path, url: str) -> None:
        (env_writes if env_writes is not None else []).append((path, url))

    return d.main(["--profile", "dev", *extra], runner=runner, plugin_dir=plugin_dir, env_writer=writer,
                  exists=exists or (lambda path: True),
                  stack_outputs=outputs or (lambda name, region: {"ApiDomain": "abc123.execute-api.us-east-1.amazonaws.com",
                                                                  "ApiStage": "v1"}))


LIST = ["twilio", "flex:plugins:list:plugin-versions", "--name", "plugin-owlbank-context", "--profile", "dev", "--json"]
RELEASES = ["twilio", "flex:plugins:list:releases", "--profile", "dev", "--json"]
PREFLIGHT = [
    ["twilio", "--version"],
    ["twilio", "plugins"],
    ["twilio", "profiles:list", "-o", "json"],
    ["twilio", "api:flex:v1:configuration:fetch", "--profile", "dev", "-o", "json", "--properties", "accountSid"],
]


def test_full_run_exact_argument_lists(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir)
    writes: list = []
    assert run_main(plugin_dir, runner, "--api-url", URL, env_writes=writes) == 0
    assert runner.argv == PREFLIGHT + [
        LIST, RELEASES,
        ["npm", "ci"],
        ["twilio", "flex:plugins:build", "--profile", "dev"],
        ["twilio", "flex:plugins:deploy", "--major", "--changelog", "OwlBank screen pop",
         "--description", "OwlBank AI escalation context", "--profile", "dev"],
        LIST,
        ["twilio", "flex:plugins:release", "--plugin", "plugin-owlbank-context@1.0.0", "--name",
         "OwlBank screen pop", "--description", "OwlBank AI escalation context", "--profile", "dev"],
    ]
    assert all(c[1] == plugin_dir for c in runner.calls)
    assert writes == [(plugin_dir / ".env", URL)]
    out = capsys.readouterr().out
    assert "plugin-owlbank-context" in out and "1.0.0" in out and SID in out


def test_url_from_stack_outputs(plugin_dir) -> None:
    runner = FakeRunner(plugin_dir)
    writes: list = []
    assert run_main(plugin_dir, runner, env_writes=writes) == 0
    assert writes[0][1] == URL


def test_url_from_custom_domain_has_no_stage() -> None:
    assert d.api_url_from_outputs({"ApiDomain": "api.owlbank.example"}) == "https://api.owlbank.example"


def test_stack_lookup_failure_asks_for_api_url(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir)

    def boom(name: str, region: str):
        raise RuntimeError("AccessDenied")

    assert run_main(plugin_dir, runner, outputs=boom) == 1
    assert "--api-url" in capsys.readouterr().err
    assert runner.calls == []


def test_dry_run_runs_only_preflight(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir)
    writes: list = []
    assert run_main(plugin_dir, runner, "--api-url", URL, "--dry-run", env_writes=writes) == 0
    assert runner.argv == PREFLIGHT
    assert writes == []
    out = capsys.readouterr().out
    for cmd in ("npm ci", "twilio flex:plugins:deploy --major",
                "twilio flex:plugins:release --plugin plugin-owlbank-context@<version assigned by Flex>"):
        assert cmd in out
    assert "@0.1.0" not in out and "Flex CLI, not this script" in out and "nothing was built" in out


@pytest.mark.parametrize("fail_key, text", [
    ("--version", "Twilio CLI is not installed"),
    ("plugins", "could not list"),
    ("profiles:list", "profile 'dev' is not logged in"),
    ("api:flex:v1:configuration:fetch", "profile 'dev' could not read the account's Flex configuration"),
])
def test_preflight_failures_are_named(plugin_dir, capsys, fail_key, text) -> None:
    runner = FakeRunner(plugin_dir, fail={fail_key: d.CommandFailed("x", missing=True)})
    assert run_main(plugin_dir, runner, "--api-url", URL) == 1
    assert text in capsys.readouterr().err
    assert not any(c[1] == "flex:plugins:deploy" for c in runner.argv)


@pytest.mark.parametrize("profiles", [
    [], [{"id": "other", "accountSid": SID}], [{"id": "dev"}], {"id": "dev", "accountSid": SID},
])
def test_profile_missing_from_the_listing_is_named(plugin_dir, capsys, profiles) -> None:
    runner = FakeRunner(plugin_dir, profiles=profiles)
    assert run_main(plugin_dir, runner, "--api-url", URL) == 1
    assert "profile 'dev' is not logged in" in capsys.readouterr().err
    assert not any(c[1] == "api:flex:v1:configuration:fetch" for c in runner.argv)


def test_account_read_must_return_the_profiles_account(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, fetched="AC" + "b" * 32)
    assert run_main(plugin_dir, runner, "--api-url", URL) == 1
    assert "did not return its account's Flex configuration" in capsys.readouterr().err
    assert not any(c[1] == "flex:plugins:deploy" for c in runner.argv)


def test_missing_flex_plugin_is_named(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, plugins="@oclif/plugin-help 1.0.0")
    assert run_main(plugin_dir, runner, "--api-url", URL) == 1
    assert "Flex plugin is not installed" in capsys.readouterr().err
    assert not any(c[0] == "npm" for c in runner.argv)


def test_already_released_is_reported_not_an_error(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, versions=["1"], released=["1"])
    assert run_main(plugin_dir, runner, "--api-url", URL) == 0
    assert runner.argv == PREFLIGHT + [LIST, RELEASES]
    out = capsys.readouterr().out
    assert "Already released" in out and "plugin-owlbank-context 1" in out and SID in out


def test_already_released_message_names_force(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, versions=["1"], released=["1"])
    assert run_main(plugin_dir, runner, "--api-url", URL) == 0
    assert "--force" in capsys.readouterr().out


def test_force_builds_deploys_and_releases_when_already_released(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, versions=["1"], released=["1"])
    writes: list = []
    assert run_main(plugin_dir, runner, "--api-url", URL, "--force", env_writes=writes) == 0
    assert runner.argv[len(PREFLIGHT):] == [
        LIST, RELEASES,
        ["npm", "ci"],
        ["twilio", "flex:plugins:build", "--profile", "dev"],
        ["twilio", "flex:plugins:deploy", "--major", "--changelog", "OwlBank screen pop",
         "--description", "OwlBank AI escalation context", "--profile", "dev"],
        LIST,
        ["twilio", "flex:plugins:release", "--plugin", "plugin-owlbank-context@1.0.0", "--name",
         "OwlBank screen pop", "--description", "OwlBank AI escalation context", "--profile", "dev"],
    ]
    assert writes == [(plugin_dir / ".env", URL)]
    assert "nothing to do" not in capsys.readouterr().out


def test_latest_version_not_in_a_release_is_released_even_if_an_older_one_was(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, versions=["1", "2"], released=["1"])
    assert run_main(plugin_dir, runner, "--api-url", URL) == 0
    assert "Already released" not in capsys.readouterr().out


def test_package_json_version_does_not_decide_released(plugin_dir, capsys) -> None:
    # a fresh clone says 0.1.0 (4.0.0 in the real repo) while Flex's own count is 1
    runner = FakeRunner(plugin_dir, versions=["1"], released=["1"])
    assert run_main(plugin_dir, runner, "--api-url", URL) == 0
    assert "Already released" in capsys.readouterr().out
    assert not any(c[0] == "npm" for c in runner.argv)


@pytest.mark.parametrize("command", ["flex:plugins:list:plugin-versions", "flex:plugins:list:releases"])
def test_empty_stdout_is_an_error_naming_the_command(plugin_dir, capsys, command) -> None:
    class Empty(FakeRunner):
        def __call__(self, args, cwd=None):
            if list(args)[1:2] == [command]:
                return ""
            return super().__call__(args, cwd)

    runner = Empty(plugin_dir)
    assert run_main(plugin_dir, runner, "--api-url", URL) == 1
    err = capsys.readouterr().err
    assert f"the Twilio Flex CLI printed nothing for {command}" in err
    assert "may not exist in this plugin version" in err
    assert not any(c[0] == "npm" for c in runner.argv)


def test_list_commands_use_the_real_flex_command_names() -> None:
    assert d.list_args("dev", "plugin-x") == ["twilio", "flex:plugins:list:plugin-versions", "--name", "plugin-x",
                                              "--profile", "dev", "--json"]
    assert d.list_releases_args("dev") == ["twilio", "flex:plugins:list:releases", "--profile", "dev", "--json"]


def test_genuine_empty_json_list_is_a_fresh_account(plugin_dir) -> None:
    runner = FakeRunner(plugin_dir)  # no versions, no releases: both print []
    assert run_main(plugin_dir, runner, "--api-url", URL) == 0
    assert ["npm", "ci"] in runner.argv


def test_plugin_state_ignores_other_plugins_and_orders_numerically() -> None:
    plugins = json.dumps([{"name": "other", "version": "99"}, {"name": "p", "version": "9"},
                          {"name": "p", "version": "10"}])
    releases = json.dumps([{"plugins": [{"name": "p", "version": "9"}, {"name": "other", "version": "10"}]}])
    assert d.plugin_state(plugins, releases, "p") == ("10", False)
    assert d.plugin_state("[]", "[]", "p") == (None, False)


def test_incomplete_install_fails_naming_the_install(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir)
    seen: list[Path] = []

    def missing(path: Path) -> bool:
        seen.append(path)
        return False

    assert run_main(plugin_dir, runner, "--api-url", URL, exists=missing) == 1
    err = capsys.readouterr().err
    assert "dependency install failed" in err and "package-lock.json" in err
    assert seen == [plugin_dir / "node_modules" / "react" / "package.json"]
    assert runner.argv[-1] == ["npm", "ci"]  # no build, deploy or release after it


def test_preflight_reports_every_failure_at_once(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, plugins="@oclif/plugin-help 1.0.0", profiles=[])
    assert run_main(plugin_dir, runner, "--api-url", URL) == 1
    err = capsys.readouterr().err
    assert "Flex plugin is not installed" in err and "profile 'dev' is not logged in" in err
    assert not any(c[0] == "npm" for c in runner.argv)


def test_missing_cli_is_reported_once_without_running_the_rest(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, fail={"--version": d.CommandFailed("x", missing=True)})
    assert run_main(plugin_dir, runner, "--api-url", URL) == 1
    assert "Twilio CLI is not installed" in capsys.readouterr().err
    assert runner.argv == [["twilio", "--version"]]


def test_profile_with_a_space_has_a_clear_error(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir)
    assert d.main(["--profile=my dev", "--api-url", URL], runner=runner, plugin_dir=plugin_dir) == 1
    assert "contains a space" in capsys.readouterr().err
    assert runner.calls == []


def test_aws_profile_note_is_in_the_help_and_docstring(capsys) -> None:
    with pytest.raises(SystemExit):
        d.main(["--help"])
    assert "AWS_PROFILE" in capsys.readouterr().out
    assert "AWS_PROFILE" in (d.__doc__ or "")


def test_force_when_not_released_is_the_normal_run(plugin_dir) -> None:
    plain = FakeRunner(plugin_dir)
    assert run_main(plugin_dir, plain, "--api-url", URL) == 0
    (plugin_dir / "package.json").write_text(json.dumps(PKG), encoding="utf-8")
    forced = FakeRunner(plugin_dir)
    assert run_main(plugin_dir, forced, "--api-url", URL, "--force") == 0
    assert forced.argv == plain.argv


def test_deployed_not_released_only_releases(plugin_dir, capsys) -> None:
    runner = FakeRunner(plugin_dir, versions=["1", "2"], released=["1"])
    assert run_main(plugin_dir, runner, "--api-url", URL) == 0
    assert runner.argv[-1][:4] == ["twilio", "flex:plugins:release", "--plugin", "plugin-owlbank-context@2"]
    assert not any(c[0] == "npm" for c in runner.argv)
    assert "Already deployed" in capsys.readouterr().out


def test_output_has_no_secret_or_environment(plugin_dir, capsys, monkeypatch) -> None:
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", SECRET)
    runner = FakeRunner(plugin_dir)
    run_main(plugin_dir, runner, "--api-url", URL)
    run_main(plugin_dir, FakeRunner(plugin_dir), "--api-url", URL, "--dry-run")
    cap = capsys.readouterr()
    assert SECRET not in cap.out + cap.err


@pytest.mark.parametrize("profile", ["", "-x", "a b", "a;b", "$(x)"])
def test_bad_profile_rejected(plugin_dir, profile) -> None:
    runner = FakeRunner(plugin_dir)
    assert d.main([f"--profile={profile}", "--api-url", URL], runner=runner, plugin_dir=plugin_dir) == 1
    assert runner.calls == []


def test_env_file_ignored_by_git() -> None:
    lines = (d.PLUGIN_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines


def test_default_env_writer_writes_only_the_url(tmp_path) -> None:
    d.default_env_writer(tmp_path / ".env", URL)
    assert (tmp_path / ".env").read_text(encoding="utf-8") == f"{d.ENV_VAR}={URL}\n"


# --------------------------------------------------------------------------- property


def _no_calls_and_rejected(url: str) -> None:
    runner = FakeRunner(Path("."))
    def no_aws(name: str, region: str):
        raise AssertionError("an explicit --api-url must not read the stack")

    code = d.main(["--profile", "dev", f"--api-url={url}"], runner=runner, plugin_dir=Path("."), stack_outputs=no_aws)
    assert code == 1
    assert runner.calls == []


@settings(max_examples=200)
@given(st.text().filter(lambda s: not s.startswith("https://")))
def test_non_https_is_rejected_before_any_command(url: str) -> None:
    _no_calls_and_rejected(url)


@settings(max_examples=100)
@given(st.sampled_from(["http", "ftp", "file", "javascript", "HTTP", "ws", ""]),
       st.from_regex(r"[a-z0-9.]{1,20}", fullmatch=True))
def test_other_schemes_with_a_host_are_rejected(scheme: str, host: str) -> None:
    _no_calls_and_rejected(f"{scheme}://{host}" if scheme else host)


@settings(max_examples=100)
@given(st.sampled_from(["https://", "https:///x", "https://:443", "https://?q=1", "https:// host", "https://u:p@h.example",
                        "https://h.example/x?y", "https://h.example/#f", "https://h.example\n"]))
def test_https_without_a_usable_host_is_rejected(url: str) -> None:
    _no_calls_and_rejected(url)


@given(st.from_regex(r"[a-z][a-z0-9-]{0,20}\.[a-z]{2,6}", fullmatch=True), st.from_regex(r"(/[a-z0-9_-]{1,8}){0,3}", fullmatch=True))
def test_https_with_host_is_accepted(host: str, path: str) -> None:
    assert d.validate_api_url(f"https://{host}{path}") == f"https://{host}{path}"
    with pytest.raises(ScriptError):
        d.validate_api_url(f"http://{host}{path}")


def test_default_runner_closes_stdin(monkeypatch) -> None:
    seen: dict = {}

    def fake_run(args, **kw):
        seen.update(kw)
        return type("P", (), {"returncode": 0, "stdout": "ok"})()

    monkeypatch.setattr(d.subprocess, "run", fake_run)
    assert d.default_runner(["twilio", "--version"]) == "ok"
    assert seen["stdin"] is d.subprocess.DEVNULL
