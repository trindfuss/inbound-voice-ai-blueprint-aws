"""Install the OwlBank Flex plugin in the customer's Flex account.

The AWS stacks do not install the plugin; this script does, with the Twilio CLI. Usage::

    # plan only: runs the read-only preflight, prints the commands it would run
    uv run python scripts/deploy_flex_plugin.py --profile owlbank-dev --dry-run

    # install: the Context API URL comes from the API stack's outputs, or --api-url
    uv run python scripts/deploy_flex_plugin.py --profile owlbank-dev [--api-url https://api.example.com]

Steps: (1) resolve the Context API URL (``--api-url``, else the ``OwlBankApi`` stack's ``ApiDomain`` and
``ApiStage`` outputs through a read-only ``DescribeStacks``) and reject anything that is not ``https://`` with a
host, before any command runs; (2) preflight: ``twilio --version``, the Flex plugin, and the named CLI profile;
(3) in ``flex-plugin/``: ``npm ci`` (then ``node_modules/react/package.json`` must exist), ``.env`` (the URL
only), build, deploy, release. The plugin's latest version is read with ``flex:plugins:list --json`` and the
releases with ``flex:plugins:list-releases --json`` (the plain listing is empty when run non-interactively); a
latest version that is already in a release is reported, not an error, so a re-run is safe. The Flex CLI, not
this script, assigns the version on ``deploy --major`` and rewrites ``package.json``; the script reads it back.

The URL is baked into the build, so each environment is its own deploy. ``--profile`` names a Twilio CLI profile:
credentials stay in the CLI's own store and never reach this script, a file or its output. It is not the AWS
profile: the stack lookup (no ``--api-url``) uses the usual AWS credential chain, so set ``AWS_PROFILE`` for it. Commands are argument
lists (no shell). The script never prints the environment, the CLI's config or a command's raw output.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import (
    REPO_ROOT,
    ScriptError,
    resolve_region,
    script_error,
    script_start,
    script_step,
    start_logging,
)

slog = logging.getLogger("deploy_flex_plugin")

PLUGIN_DIR = REPO_ROOT / "flex-plugin"
ENV_VAR = "FLEX_APP_OWLBANK_CONTEXT_API_URL"
DEFAULT_STACK = "OwlBankApi"
CHANGELOG = "OwlBank screen pop"
DESCRIPTION = "OwlBank AI escalation context"

_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_VERSION = re.compile(r"\d+(?:\.\d+){0,2}")
_NAME_KEYS = ("name", "uniqueName", "pluginName", "plugin")
_VERSION_KEYS = ("version", "latestVersion")
VERSION_PLACEHOLDER = "<version assigned by Flex>"
_BAD_CHARS = re.compile(r"[\s\x00-\x1f\x7f\\\"'`$]")

#: Runs an argument list (in ``cwd`` if given) and returns stdout; raises :class:`CommandFailed`.
Runner = Callable[..., str]
#: Reads a stack's outputs: ``(stack_name, region) -> {OutputKey: value}``.
StackOutputs = Callable[[str, str], Mapping[str, str]]
EnvWriter = Callable[[Path, str], None]
#: Whether a path exists (injected so tests need no ``node_modules``).
Exists = Callable[[Path], bool]


class CommandFailed(ScriptError):
    """A command did not run or exited non-zero. Names the command, never its output."""

    def __init__(self, message: str, *, missing: bool = False) -> None:
        super().__init__(message)
        self.missing = missing


def default_runner(args: Sequence[str], cwd: Path | None = None) -> str:
    try:
        # stdin is closed so a CLI prompt fails fast instead of hanging an unattended run.
        # TODO: confirm `flex:plugins:deploy --bypass-validation` (and any release flag) against
        # `twilio flex:plugins:deploy --help` at the first live install; not added unverified.
        proc = subprocess.run(list(args), capture_output=True, text=True, check=False, cwd=cwd, timeout=900,
                              stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise CommandFailed(f"{args[0]} not found", missing=True) from None
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommandFailed(f"could not run {args[0]}: {type(exc).__name__}") from None
    if proc.returncode != 0:
        raise CommandFailed(f"{' '.join(args[:2])} exited with status {proc.returncode}")
    return proc.stdout


def default_stack_outputs(stack_name: str, region: str) -> dict[str, str]:
    import boto3  # imported late: only needed when --api-url is not given

    stacks = boto3.client("cloudformation", region_name=region).describe_stacks(StackName=stack_name)["Stacks"]
    return {o["OutputKey"]: o["OutputValue"] for o in stacks[0].get("Outputs", [])}


def default_exists(path: Path) -> bool:
    return path.exists()


def default_env_writer(path: Path, url: str) -> None:
    path.write_text(f"{ENV_VAR}={url}\n", encoding="utf-8")


# --------------------------------------------------------------------------- pure helpers


def validate_api_url(url: str) -> str:
    """The URL as given if it is ``https://`` with a host; else :class:`ScriptError`."""
    if not isinstance(url, str) or _BAD_CHARS.search(url):
        raise ScriptError("the Context API URL must be an https:// URL with a host")
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        raise ScriptError("the Context API URL must be an https:// URL with a host") from None
    if parts.scheme != "https" or not host or parts.username or parts.password or parts.query or parts.fragment:
        raise ScriptError("the Context API URL must be an https:// URL with a host "
                          "(no credentials, query or fragment)")
    return url.rstrip("/")


def api_url_from_outputs(outputs: Mapping[str, str]) -> str:
    """``https://<ApiDomain>``, plus ``/<ApiStage>`` when the stack has that output (default endpoint)."""
    domain = outputs.get("ApiDomain")
    if not domain:
        raise ScriptError("the API stack has no ApiDomain output: pass --api-url")
    stage = outputs.get("ApiStage")
    return f"https://{domain}" + (f"/{stage}" if stage else "")


def read_package(plugin_dir: Path) -> tuple[str, str]:
    doc = json.loads((plugin_dir / "package.json").read_text(encoding="utf-8"))
    return str(doc["name"]), str(doc["version"])


VERSIONS_COMMAND = "flex:plugins:list:plugin-versions"
RELEASES_COMMAND = "flex:plugins:list:releases"


def _parse_json(text: str, what: str, command: str) -> object:
    if not text.strip():
        # Exit 0 with nothing printed is what a command that does not exist in this plugin version does.
        raise ScriptError(f"the Twilio Flex CLI printed nothing for {command}; "
                          "the command may not exist in this plugin version")
    try:
        return json.loads(text)
    except ValueError:
        raise ScriptError(f"could not read the {what} (--json output was empty or not JSON)") from None


def _plugin_versions(doc: object, name: str) -> list[tuple[int, ...]]:
    """Every version of plugin ``name`` found anywhere in a ``--json`` document (list or nested releases)."""
    found: list[tuple[int, ...]] = []
    stack = [doc]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
        elif isinstance(node, dict):
            if name in (node.get(k) for k in _NAME_KEYS):
                for key in _VERSION_KEYS:
                    value = node.get(key)
                    if isinstance(value, str) and _VERSION.fullmatch(value):
                        found.append(tuple(int(p) for p in value.split(".")))
            stack.extend(v for v in node.values() if isinstance(v, (list, dict)))
    return found


def _fmt(version: tuple[int, ...]) -> str:
    return ".".join(str(p) for p in version)


def plugin_state(plugins_json: str, releases_json: str, name: str) -> tuple[str | None, bool]:
    """(latest deployed version, whether a release contains it) for ``name``.

    Decided from Flex's own records, never from ``package.json`` (a fresh clone says 4.0.0 while Flex's count
    starts at 1). ``None`` when the plugin has no deployed version."""
    versions = _plugin_versions(_parse_json(plugins_json, "plugin listing", VERSIONS_COMMAND), name)
    released = _plugin_versions(_parse_json(releases_json, "release listing", RELEASES_COMMAND), name)
    if not versions:
        return None, False
    latest = max(versions)
    return _fmt(latest), latest in released


def list_args(profile: str, name: str) -> list[str]:
    # TODO: the command names are verified against plugin-flex 7.1.3 --help, but the --json output shape is
    # not: it is confirmed during the dev deploy check (the parser below accepts any nesting).
    return ["twilio", VERSIONS_COMMAND, "--name", name, "--profile", profile, "--json"]


def list_releases_args(profile: str) -> list[str]:
    return ["twilio", RELEASES_COMMAND, "--profile", profile, "--json"]


def deploy_args(profile: str) -> list[str]:
    return ["twilio", "flex:plugins:deploy", "--major", "--changelog", CHANGELOG,
            "--description", DESCRIPTION, "--profile", profile]


def release_args(profile: str, name: str, version: str) -> list[str]:
    return ["twilio", "flex:plugins:release", "--plugin", f"{name}@{version}", "--name", CHANGELOG,
            "--description", DESCRIPTION, "--profile", profile]


def preflight_commands(profile: str) -> list[tuple[str, list[str]]]:
    """The checks before the account read; the profile's SID comes from ``profiles:list`` because
    ``api:core:accounts:fetch`` requires ``--sid`` (Twilio CLI 7.2.0)."""
    return [("cli", ["twilio", "--version"]),
            ("flex_plugin", ["twilio", "plugins"]),
            ("profile", ["twilio", "profiles:list", "-o", "json"])]


def flex_config_args(profile: str) -> list[str]:
    """One authenticated read: the account's Flex configuration. Proves the profile's credentials work and
    that the account has Flex. Not ``api:core:accounts:fetch``: the Standard API key ``twilio login``
    creates gets 70004 on the Accounts resource."""
    return ["twilio", "api:flex:v1:configuration:fetch", "--profile", profile, "-o", "json",
            "--properties", "accountSid"]


def profile_sid(listing: str, profile: str) -> str:
    """The account SID of ``profile`` in ``twilio profiles:list -o json``; ``""`` if it isn't there."""
    try:
        rows = json.loads(listing)
    except ValueError:
        return ""
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("id") == profile and isinstance(row.get("accountSid"), str):
            return row["accountSid"]
    return ""


def build_commands(profile: str, name: str, version: str) -> list[tuple[str, list[str]]]:
    return [("npm_ci", ["npm", "ci"]),
            ("build", ["twilio", "flex:plugins:build", "--profile", profile]),
            ("deploy", deploy_args(profile)),
            ("release", release_args(profile, name, version))]


# --------------------------------------------------------------------------- steps


def preflight(profile: str, runner: Runner, cwd: Path) -> str:
    """Runs the read-only checks and reports every failure in one message; returns the Flex account SID."""
    sid = ""
    failures: list[str] = []
    cli_ok = True
    for step, args in preflight_commands(profile):
        if step != "cli" and not cli_ok:
            continue  # every later check runs the CLI; it can't be checked until the CLI is installed
        try:
            out = runner(args, cwd=cwd)
        except CommandFailed:
            if step == "cli":
                cli_ok = False
                failures.append("the Twilio CLI is not installed or not on PATH "
                                "(install it, then: twilio plugins:install @twilio-labs/plugin-flex)")
            elif step == "profile":
                failures.append(f"Twilio CLI profile '{profile}' is not logged in or does not exist "
                                f"(run: twilio login, then pass --profile)")
            else:
                failures.append("could not list the Twilio CLI plugins")
            continue
        if step == "flex_plugin" and "plugin-flex" not in out:
            failures.append("the Flex plugin is not installed in the Twilio CLI "
                            "(run: twilio plugins:install @twilio-labs/plugin-flex)")
            continue
        if step == "profile":
            sid = profile_sid(out, profile)
            if not sid:
                failures.append(f"Twilio CLI profile '{profile}' is not logged in or does not exist "
                                f"(run: twilio login, then pass --profile)")
                continue
        script_step(slog, "deploy_flex_plugin", f"preflight_{step}", "ok")
    if sid and cli_ok:
        try:
            out = runner(flex_config_args(profile), cwd=cwd)
            if str(json.loads(out)[0]["accountSid"]) != sid:
                raise ValueError
            script_step(slog, "deploy_flex_plugin", "preflight_flex_account", "ok")
        except CommandFailed:
            failures.append(f"Twilio CLI profile '{profile}' could not read the account's Flex configuration "
                            f"(no Flex on this account, or credentials revoked: twilio login --profile {profile})")
        except (ValueError, KeyError, IndexError, TypeError):
            failures.append(f"Twilio CLI profile '{profile}' did not return its account's Flex configuration")
    if failures:
        raise ScriptError("preflight failed:\n  - " + "\n  - ".join(failures)
                          if len(failures) > 1 else failures[0])
    return sid


def resolve_url(args: argparse.Namespace, stack_outputs: StackOutputs) -> str:
    if args.api_url is not None:
        return validate_api_url(args.api_url)
    region = resolve_region(args.region, logger=slog)
    try:
        outputs = stack_outputs(args.stack_name, region)
    except ScriptError:
        raise
    except Exception as exc:
        raise ScriptError(f"could not read the {args.stack_name} stack outputs ({type(exc).__name__}): "
                          "pass --api-url") from None
    return validate_api_url(api_url_from_outputs(outputs))


def run(args: argparse.Namespace, runner: Runner, stack_outputs: StackOutputs, env_writer: EnvWriter,
        plugin_dir: Path, exists: Exists = default_exists) -> int:
    url = resolve_url(args, stack_outputs)  # https-only, before any command runs
    profile = args.profile
    name, version = read_package(plugin_dir)
    sid = preflight(profile, runner, plugin_dir)

    if args.dry_run:
        print("Dry run: preflight passed; nothing was built, deployed or released. "
              "Commands that would run in flex-plugin/:")
        for _step, cmd in build_commands(profile, name, VERSION_PLACEHOLDER):
            print("  " + " ".join(cmd))
        print(f"  (writes .env with {ENV_VAR}={url} before the build)")
        print("  (the Flex CLI, not this script, sets the new version on 'deploy --major' and rewrites "
              "package.json; the release then uses that version)")
        print(f"Flex account SID: {sid}")
        return 0

    latest, released = plugin_state(runner(list_args(profile, name), cwd=plugin_dir),
                                    runner(list_releases_args(profile), cwd=plugin_dir), name)
    if released and not args.force:
        script_step(slog, "deploy_flex_plugin", "release", "skipped", reason="already_released")
        print(f"Already released: {name} {latest}; nothing to do. "
              "If the code or the API URL changed, run again with --force.")
        version = str(latest)
    else:
        if args.force or latest is None:
            for step, cmd in build_commands(profile, name, version)[:2]:
                if step == "build":
                    env_writer(plugin_dir / ".env", url)
                runner(cmd, cwd=plugin_dir)
                script_step(slog, "deploy_flex_plugin", step, "changed")
                if step == "npm_ci" and not exists(plugin_dir / "node_modules" / "react" / "package.json"):
                    raise ScriptError("dependency install failed: node_modules is incomplete "
                                      "(node_modules/react/package.json is missing); "
                                      "check the registry in package-lock.json")
            runner(deploy_args(profile), cwd=plugin_dir)
            script_step(slog, "deploy_flex_plugin", "deploy", "changed")
            version = _read_back(runner, profile, plugin_dir, name)
        else:
            version = str(latest)
            print(f"Already deployed: {name} {version}; releasing it.")
        runner(release_args(profile, name, version), cwd=plugin_dir)
        script_step(slog, "deploy_flex_plugin", "release", "changed")
    print(f"Plugin: {name}\nReleased version: {version}\nFlex account SID: {sid}")
    print("The API URL is baked into this build: deploy again for each environment.")
    return 0


def _read_back(runner: Runner, profile: str, plugin_dir: Path, name: str) -> str:
    """The version Flex assigned on deploy, from ``list --json``; else the one the CLI wrote to package.json."""
    listing = runner(list_args(profile, name), cwd=plugin_dir)
    versions = _plugin_versions(_parse_json(listing, "plugin listing", VERSIONS_COMMAND), name)
    return _fmt(max(versions)) if versions else read_package(plugin_dir)[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Install the OwlBank Flex plugin with the Twilio CLI.")
    p.add_argument("--profile", required=True, help="Twilio CLI profile name (credentials stay in the CLI); not an AWS profile")
    p.add_argument("--api-url", help="Context API https base URL; default: from the API stack's outputs "
                                         "(read with your AWS credentials: set AWS_PROFILE for that)")
    p.add_argument("--stack-name", default=DEFAULT_STACK, help=f"API stack name (default {DEFAULT_STACK})")
    p.add_argument("--region", help="AWS Region for the stack lookup")
    p.add_argument("--force", action="store_true",
                   help="build, deploy and release a new version even if the current one is already released")
    p.add_argument("--dry-run", action="store_true", help="run the read-only preflight and print the commands")
    return p


def main(argv: list[str] | None = None, *, runner: Runner = default_runner,
         stack_outputs: StackOutputs = default_stack_outputs, env_writer: EnvWriter = default_env_writer,
         plugin_dir: Path = PLUGIN_DIR, exists: Exists = default_exists) -> int:
    args = build_parser().parse_args(argv)
    script_start(slog, "deploy_flex_plugin", mode="dry_run" if args.dry_run else "apply")
    try:
        if re.search(r"\s", args.profile):
            raise ScriptError(f"--profile {args.profile!r} contains a space: the Twilio CLI profile name must be a "
                              "single word (letters, digits, . _ -); log in again under a name "
                              "without spaces: twilio login")
        if not _PROFILE.fullmatch(args.profile):
            raise ScriptError("--profile must be a Twilio CLI profile name (letters, digits, . _ -)")
        return run(args, runner, stack_outputs, env_writer, plugin_dir, exists)
    except ScriptError as exc:
        script_error(slog, "deploy_flex_plugin", exc, decision="abort")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    start_logging("deploy_flex_plugin")
    sys.exit(main())
