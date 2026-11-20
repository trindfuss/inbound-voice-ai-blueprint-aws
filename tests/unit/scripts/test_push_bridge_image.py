"""AgentCore hosting: ``infra/scripts/push_bridge_image.py`` with a fake runner.

Command lines (arm64 buildx build of ``src/tac_bridge/Dockerfile``, ``git-<sha>`` tag, ECR login
through stdin), digest parsing, the dry-run default (no AWS or Docker command), refusal on a dirty
tree unless ``--allow-dirty``, and the ECR password never printed. The Finch path (``--cli finch``,
default from ``CDK_DOCKER``): ``finch build`` / ``finch push`` and the digest from
``aws ecr describe-images``. No Docker, Finch, AWS or network. No PBT (command glue; assertions
instead).
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "infra" / "scripts" / "push_bridge_image.py"
_spec = importlib.util.spec_from_file_location("push_bridge_image", SCRIPT)
assert _spec and _spec.loader
pbi = importlib.util.module_from_spec(_spec)
sys.modules["push_bridge_image"] = pbi  # dataclasses need the module registered
_spec.loader.exec_module(pbi)

URI = "111111111111.dkr.ecr.us-east-1.amazonaws.com/owlbank-dev-bridge"
REGISTRY = "111111111111.dkr.ecr.us-east-1.amazonaws.com"
SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "sha256:" + "ab" * 32
PASSWORD = "ecr-login-password-NOT-A-REAL-ONE"  # noqa: S105 (a fake for the tests)


class FakeRunner:
    def __init__(self, *, dirty: bool = False, digest: str | None = DIGEST, metadata: str | None = None,
                 fail_on: str | None = None, described: str | None = None) -> None:
        self.calls: list[dict] = []
        self.dirty, self.digest, self.metadata, self.fail_on = dirty, digest, metadata, fail_on
        self.described = DIGEST + "\n" if described is None else described

    def __call__(self, args: Sequence[str], *, cwd: Path | None = None, stdin: str | None = None,
                 stream: bool = False) -> str:
        args = list(args)
        self.calls.append({"args": args, "cwd": cwd, "stdin": stdin, "stream": stream})
        if self.fail_on and self.fail_on in args:
            raise pbi.PushError(f"{' '.join(args[:2])} exited with status 1")
        if args[:2] == ["git", "status"]:
            return " M src/tac_bridge/app.py\n" if self.dirty else ""
        if args[:2] == ["git", "rev-parse"]:
            return SHA + "\n"
        if args[:3] == ["aws", "ecr", "get-login-password"]:
            return PASSWORD + "\n"
        if args[:3] == ["aws", "ecr", "describe-images"]:
            return self.described
        if args[:3] == ["docker", "buildx", "build"]:
            path = Path(args[args.index("--metadata-file") + 1])
            if self.metadata is not None:
                path.write_text(self.metadata)
            elif self.digest is not None:
                path.write_text(json.dumps({"containerimage.digest": self.digest,
                                            "image.name": f"{URI}:git-{SHA[:12]}"}))
        return ""

    def commands(self) -> list[list[str]]:
        return [c["args"] for c in self.calls]

    def first(self, *prefix: str) -> dict:
        return next(c for c in self.calls if c["args"][:len(prefix)] == list(prefix))


@pytest.fixture(autouse=True)
def _no_cdk_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The docker tests must not depend on an exported ``CDK_DOCKER=finch``."""
    monkeypatch.delenv("CDK_DOCKER", raising=False)


def run(argv: list[str], runner: FakeRunner) -> tuple[int, str]:
    out = io.StringIO()
    return pbi.run(argv, runner=runner, out=out), out.getvalue()


# ---- dry run (the default) -------------------------------------------------------------------

def test_dry_run_is_the_default_and_runs_only_git(capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner()
    code, out = run(["--repository-uri", URI], r)
    assert code == 0
    assert {c[0] for c in r.commands()} == {"git"}
    assert f"{URI}:git-{SHA[:12]}" in out
    assert "--platform linux/arm64" in out and "-f src/tac_bridge/Dockerfile" in out
    assert "--password-stdin" in out and PASSWORD not in out + capsys.readouterr().err


def test_explicit_dry_run_flag(capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner()
    assert run(["--repository-uri", URI, "--dry-run"], r)[0] == 0
    assert {c[0] for c in r.commands()} == {"git"}


def test_apply_and_dry_run_are_exclusive() -> None:
    with pytest.raises(SystemExit):
        run(["--repository-uri", URI, "--apply", "--dry-run"], FakeRunner())


def test_a_repository_source_is_required() -> None:
    with pytest.raises(SystemExit):
        run([], FakeRunner())


# ---- apply: command lines ------------------------------------------------------------------

def test_apply_command_lines() -> None:
    r = FakeRunner()
    code, out = run(["--repository-uri", URI, "--profile", "sandbox", "--apply"], r)
    assert code == 0
    cmds = r.commands()
    assert cmds[0] == ["git", "status", "--porcelain"]
    assert cmds[1] == ["git", "rev-parse", "HEAD"]
    assert cmds[2] == ["aws", "ecr", "get-login-password", "--region", "us-east-1", "--profile", "sandbox"]
    assert cmds[3] == ["docker", "login", "--username", "AWS", "--password-stdin", REGISTRY]
    build = cmds[4]
    meta = build[build.index("--metadata-file") + 1]
    assert build == ["docker", "buildx", "build", "--platform", "linux/arm64",
                     "-f", "src/tac_bridge/Dockerfile", "-t", f"{URI}:git-{SHA[:12]}",
                     "--provenance=false", "--sbom=false", "--metadata-file", meta, "--push", "."]
    assert len(cmds) == 5


def test_git_and_build_run_from_the_repository_root() -> None:
    r = FakeRunner()
    run(["--repository-uri", URI, "--apply"], r)
    assert r.first("git", "status")["cwd"] == REPO
    assert r.first("git", "rev-parse")["cwd"] == REPO
    build = r.first("docker", "buildx")
    assert build["cwd"] == REPO and build["stream"] is True
    assert (REPO / "src" / "tac_bridge" / "Dockerfile").is_file()


def test_no_profile_means_no_profile_flag() -> None:
    r = FakeRunner()
    run(["--repository-uri", URI, "--apply"], r)
    assert "--profile" not in r.first("aws", "ecr")["args"]


def test_region_comes_from_the_repository_uri() -> None:
    r = FakeRunner()
    run(["--repository-uri", URI.replace("us-east-1", "us-west-2"), "--apply"], r)
    pw = r.first("aws", "ecr")["args"]
    assert pw[pw.index("--region") + 1] == "us-west-2"


def test_password_goes_only_through_stdin(capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner()
    code, out = run(["--repository-uri", URI, "--apply"], r)
    assert code == 0
    login = r.first("docker", "login")
    assert login["stdin"] == PASSWORD and PASSWORD not in login["args"]
    assert all(PASSWORD not in " ".join(c) for c in r.commands())
    captured = capsys.readouterr()
    assert PASSWORD not in out + captured.out + captured.err


# ---- digest ------------------------------------------------------------------------------------

def test_apply_prints_the_digest_and_the_deploy_flag() -> None:
    code, out = run(["--repository-uri", URI, "--apply"], FakeRunner())
    assert code == 0
    assert f"digest {DIGEST}" in out
    assert f"-c bridgeImageDigest={DIGEST}" in out


@pytest.mark.parametrize("metadata", [
    "not json",
    "[]",
    "{}",
    json.dumps({"containerimage.digest": "sha256:" + "AB" * 32}),
    json.dumps({"containerimage.digest": "sha256:" + "ab" * 31}),
    json.dumps({"containerimage.digest": "sha512:" + "ab" * 32}),
    json.dumps({"containerimage.digest": 7}),
])
def test_bad_metadata_fails_without_printing_a_digest(metadata: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(["--repository-uri", URI, "--apply"], FakeRunner(metadata=metadata))
    assert code == 1
    assert "digest" not in out and "bridgeImageDigest=" not in out
    assert "error:" in capsys.readouterr().err


def test_missing_metadata_file_fails(capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run(["--repository-uri", URI, "--apply"], FakeRunner(digest=None))
    assert code == 1 and "metadata" in capsys.readouterr().err


def test_parse_digest_accepts_a_valid_digest() -> None:
    assert pbi.parse_digest(json.dumps({"containerimage.digest": DIGEST})) == DIGEST


def test_digest_pattern_matches_the_config_parser() -> None:
    """The script's check is the one ``bridgeImageDigest`` applies (``config.py`` is stdlib-only)."""
    spec = importlib.util.spec_from_file_location("owlbank_infra_config_s3",
                                                  REPO / "infra" / "owlbank_infra" / "config.py")
    assert spec and spec.loader
    config = importlib.util.module_from_spec(spec)
    sys.modules["owlbank_infra_config_s3"] = config
    spec.loader.exec_module(config)
    assert pbi.DIGEST_RE.pattern == config._IMAGE_DIGEST.pattern


# ---- dirty tree ----------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [[], ["--apply"]])
def test_dirty_tree_is_refused(mode: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner(dirty=True)
    code, out = run(["--repository-uri", URI, *mode], r)
    assert code == 1
    assert r.commands() == [["git", "status", "--porcelain"]]
    assert "--allow-dirty" in capsys.readouterr().err


def test_allow_dirty_builds_with_a_dirty_tag() -> None:
    r = FakeRunner(dirty=True)
    code, out = run(["--repository-uri", URI, "--allow-dirty", "--apply"], r)
    assert code == 0
    build = r.first("docker", "buildx")["args"]
    assert build[build.index("-t") + 1] == f"{URI}:git-{SHA[:12]}-dirty"


def test_tag_is_git_and_twelve_hex() -> None:
    r = FakeRunner()
    run(["--repository-uri", URI, "--apply"], r)
    build = r.first("docker", "buildx")["args"]
    assert re.fullmatch(r".*:git-[0-9a-f]{12}", build[build.index("-t") + 1])


# ---- repository URI and outputs ---------------------------------------------------------------

def test_outputs_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    path.write_text(json.dumps({"OwlBankBridgeImage": {"RepositoryUri": URI, "RepositoryArn": "x"},
                                "OwlBankData": {"RepositoryUri": "not-this-one"}}))
    r = FakeRunner()
    code, out = run(["--outputs", str(path), "--apply"], r)
    assert code == 0
    assert r.first("docker", "login")["args"][-1] == REGISTRY


@pytest.mark.parametrize("doc", [{}, {"OwlBankBridgeImage": {}}, {"OwlBankData": {"RepositoryUri": URI}}, []])
def test_outputs_without_the_uri_fail(tmp_path: Path, doc: object, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "out.json"
    path.write_text(json.dumps(doc))
    r = FakeRunner()
    assert run(["--outputs", str(path)], r)[0] == 1
    assert r.calls == [] and "RepositoryUri" in capsys.readouterr().err


def test_unreadable_outputs_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--outputs", str(tmp_path / "missing.json")], FakeRunner())[0] == 1
    assert "can't read CDK outputs" in capsys.readouterr().err


@pytest.mark.parametrize("uri", [
    "owlbank-dev-bridge",
    "public.ecr.aws/owlbank/bridge",
    "11111111111.dkr.ecr.us-east-1.amazonaws.com/owlbank-dev-bridge",
    "111111111111.dkr.ecr.us-east-1.amazonaws.com/OwlBank",
    "111111111111.dkr.ecr.us-east-1.amazonaws.com/owlbank-dev-bridge:latest",
    "111111111111.dkr.ecr.us-east-1.amazonaws.com/owlbank-dev-bridge@" + DIGEST,
])
def test_bad_repository_uri_is_refused(uri: str) -> None:
    r = FakeRunner()
    assert run(["--repository-uri", uri, "--apply"], r)[0] == 1
    assert r.calls == []


# ---- failures ----------------------------------------------------------------------------------

@pytest.mark.parametrize("step", ["get-login-password", "login", "buildx"])
def test_a_failing_command_stops_and_prints_no_digest(step: str, capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner(fail_on=step)
    code, out = run(["--repository-uri", URI, "--apply"], r)
    assert code == 1 and "digest" not in out
    err = capsys.readouterr().err
    assert "exited with status" in err and PASSWORD not in err


def test_default_runner_never_echoes_output(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(pbi.PushError) as exc:
        pbi.default_runner([sys.executable, "-c", f"import sys; print({PASSWORD!r}); sys.exit(3)"])
    assert "status 3" in str(exc.value) and PASSWORD not in str(exc.value)


def test_default_runner_returns_stdout_and_feeds_stdin() -> None:
    assert pbi.default_runner([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
                              stdin="abc").strip() == "ABC"


def test_default_runner_missing_program() -> None:
    with pytest.raises(pbi.PushError, match="could not run"):
        pbi.default_runner(["owlbank-no-such-program-xyz"])


# ---- finch ---------------------------------------------------------------------------------------

FINCH = ["--repository-uri", URI, "--cli", "finch"]
REF = f"{URI}:git-{SHA[:12]}"
DESCRIBE = ["aws", "ecr", "describe-images", "--repository-name", "owlbank-dev-bridge",
            "--image-ids", f"imageTag=git-{SHA[:12]}", "--region", "us-east-1",
            "--query", "imageDetails[0].imageDigest", "--output", "text"]


def test_finch_apply_command_lines() -> None:
    r = FakeRunner()
    code, out = run([*FINCH, "--apply"], r)
    assert code == 0
    assert r.commands() == [
        ["git", "status", "--porcelain"],
        ["git", "rev-parse", "HEAD"],
        ["aws", "ecr", "get-login-password", "--region", "us-east-1"],
        ["finch", "login", "--username", "AWS", "--password-stdin", REGISTRY],
        ["finch", "build", "--platform", "linux/arm64", "-f", "src/tac_bridge/Dockerfile", "-t", REF, "."],
        ["finch", "push", REF],
        DESCRIBE,
    ]
    assert not any(c[0] == "docker" for c in r.commands())


def test_finch_build_and_push_run_from_the_repository_root() -> None:
    r = FakeRunner()
    run([*FINCH, "--apply"], r)
    for prefix in (("finch", "build"), ("finch", "push")):
        call = r.first(*prefix)
        assert call["cwd"] == REPO and call["stream"] is True
    assert r.first("aws", "ecr", "describe-images")["stream"] is False


def test_finch_profile_goes_to_both_aws_calls() -> None:
    r = FakeRunner()
    run([*FINCH, "--profile", "sandbox", "--apply"], r)
    for sub in ("get-login-password", "describe-images"):
        args = r.first("aws", "ecr", sub)["args"]
        assert args[args.index("--profile") + 1] == "sandbox"


def test_finch_password_goes_only_through_stdin(capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner()
    code, out = run([*FINCH, "--apply"], r)
    assert code == 0
    login = r.first("finch", "login")
    assert login["stdin"] == PASSWORD and PASSWORD not in login["args"]
    assert all(PASSWORD not in " ".join(c["args"]) for c in r.calls)
    assert [c["stdin"] for c in r.calls if c["stdin"] is not None] == [PASSWORD]
    captured = capsys.readouterr()
    assert PASSWORD not in out + captured.out + captured.err


def test_finch_digest_comes_from_describe_images() -> None:
    code, out = run([*FINCH, "--apply"], FakeRunner())
    assert code == 0
    assert f"pushed {REF}" in out and f"digest {DIGEST}" in out
    assert f"-c bridgeImageDigest={DIGEST}" in out


@pytest.mark.parametrize("described", [
    "", "None\n", "sha256:" + "AB" * 32, "sha256:" + "ab" * 31, "sha512:" + "ab" * 32,
    DIGEST + "\t" + DIGEST, "ab" * 32,
])
def test_finch_bad_described_digest_fails(described: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run([*FINCH, "--apply"], FakeRunner(described=described))
    assert code == 1
    assert "digest" not in out and "bridgeImageDigest=" not in out
    assert "describe-images" in capsys.readouterr().err


@pytest.mark.parametrize("step", ["get-login-password", "login", "build", "push", "describe-images"])
def test_finch_a_failing_command_stops(step: str, capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner(fail_on=step)
    code, out = run([*FINCH, "--apply"], r)
    assert code == 1 and "digest" not in out
    assert r.commands()[-1].count(step) == 1  # nothing ran after the failing step
    err = capsys.readouterr().err
    assert "exited with status" in err and PASSWORD not in err


def test_finch_dry_run_prints_finch_commands_and_runs_only_git(capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner()
    code, out = run(FINCH, r)
    assert code == 0
    assert {c[0] for c in r.commands()} == {"git"}
    assert f"finch login --username AWS --password-stdin {REGISTRY}" in out
    assert f"finch build --platform linux/arm64 -f src/tac_bridge/Dockerfile -t {REF} ." in out
    assert f"finch push {REF}" in out
    assert " ".join(DESCRIBE) in out
    assert "docker" not in out and "buildx" not in out
    assert PASSWORD not in out + capsys.readouterr().err


def test_finch_dirty_tree_refused_and_allow_dirty_tag() -> None:
    r = FakeRunner(dirty=True)
    assert run([*FINCH, "--apply"], r)[0] == 1
    assert r.commands() == [["git", "status", "--porcelain"]]
    r = FakeRunner(dirty=True)
    assert run([*FINCH, "--allow-dirty", "--apply"], r)[0] == 0
    assert r.first("finch", "push")["args"] == ["finch", "push", f"{REF}-dirty"]
    assert f"imageTag=git-{SHA[:12]}-dirty" in r.first("aws", "ecr", "describe-images")["args"]


# ---- --cli selection -----------------------------------------------------------------------------

@pytest.mark.parametrize(("env", "cli"), [
    (None, "docker"), ("finch", "finch"), ("docker", "docker"), ("", "docker"),
    ("podman", "docker"), ("FINCH", "docker"), ("/usr/local/bin/finch", "docker"),
])
def test_cli_default_from_cdk_docker(env: str | None, cli: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if env is not None:
        monkeypatch.setenv("CDK_DOCKER", env)
    r = FakeRunner()
    assert run(["--repository-uri", URI, "--apply"], r)[0] == 0
    assert r.commands()[3][:2] == [cli, "login"]


def test_explicit_cli_overrides_cdk_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDK_DOCKER", "finch")
    r = FakeRunner()
    assert run(["--repository-uri", URI, "--cli", "docker", "--apply"], r)[0] == 0
    assert r.first("docker", "buildx") and not any(c[0] == "finch" for c in r.commands())


@pytest.mark.parametrize("cli", ["podman", "Finch", ""])
def test_invalid_cli_is_refused(cli: str, capsys: pytest.CaptureFixture[str]) -> None:
    r = FakeRunner()
    with pytest.raises(SystemExit) as exc:
        run(["--repository-uri", URI, "--cli", cli], r)
    assert exc.value.code == 2 and r.calls == []
    assert "--cli" in capsys.readouterr().err


def test_default_cli_helper() -> None:
    assert pbi.default_cli({}) == "docker"
    assert pbi.default_cli({"CDK_DOCKER": "finch"}) == "finch"
    assert pbi.default_cli({"CDK_DOCKER": "nerdctl"}) == "docker"
