#!/usr/bin/env python3
"""Build the bridge image for ``linux/arm64``, push it to our ECR repository, print its digest.

Part of the AgentCore hosting option. The repository is ``OwlBankBridgeImage``'s
(immutable tags, scan on push, the project CMK). The Runtime references the image by digest, so the
next deploy takes ``-c bridgeImageDigest=<the printed digest>``::

    # dry run (the default): checks the git tree, prints the commands, runs nothing else
    uv run python infra/scripts/push_bridge_image.py --outputs cdk-outputs.json
    # build and push
    uv run python infra/scripts/push_bridge_image.py --outputs cdk-outputs.json --profile <profile> --apply

``--outputs`` is a ``cdk deploy --outputs-file`` JSON holding ``OwlBankBridgeImage``'s
``RepositoryUri``; ``--repository-uri`` names it directly instead.

* ``--cli docker|finch`` picks the container CLI; the default is ``$CDK_DOCKER`` when that is
  ``docker`` or ``finch`` (the project's CDK deploys use ``CDK_DOCKER=finch``), else ``docker``.
* The image is built from the repository root with ``src/tac_bridge/Dockerfile`` (the same
  Dockerfile and image as Fargate) for ``linux/arm64`` (the AgentCore Runtime contract).
  Docker: ``docker buildx build --push`` without provenance or SBOM attestations, so the pushed
  digest is a single image manifest. Finch: ``finch build`` then ``finch push``.
* The tag is ``git-<12-hex commit>``. A dirty working tree is refused, since the image would not
  be that commit; ``--allow-dirty`` builds it anyway, tagged ``git-<commit>-dirty``. Tags are
  immutable, so pushing the same tag twice fails at ECR: commit, then push.
* ECR login: ``aws ecr get-login-password`` piped to ``<cli> login --password-stdin``. The password
  goes from one process to the other through this script's memory; it is never printed or logged.
* The digest comes from buildx's ``--metadata-file`` (``containerimage.digest``) with Docker, or from
  ``aws ecr describe-images`` for the pushed tag with Finch, and is checked to be ``sha256:<64 hex>``
  (the format ``bridgeImageDigest`` accepts, ``infra/owlbank_infra/config.py``).

No AWS SDK: the commands are the AWS CLI and Docker or Finch, run through an injectable runner, so the tests
use a fake one and need no Docker, AWS account or network. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = "src/tac_bridge/Dockerfile"
PLATFORM = "linux/arm64"
#: The image digest ``bridgeImageDigest`` accepts (same pattern as ``config._IMAGE_DIGEST``).
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
#: A private ECR repository URI: ``<account>.dkr.ecr.<region>.amazonaws.com[.cn]/<name>``.
REPO_URI_RE = re.compile(
    r"(?P<registry>(?P<account>\d{12})\.dkr\.ecr\.(?P<region>[a-z]{2}(?:-[a-z]+)+-\d)"
    r"\.amazonaws\.com(?:\.cn)?)/(?P<name>[a-z0-9]+(?:[._/-][a-z0-9]+)*)")
#: ``RepositoryUri`` of ``OwlBankBridgeImage`` in a CDK outputs file.
OUTPUT_KEY = "RepositoryUri"
IMAGE_STACK = "OwlBankBridgeImage"
METADATA_DIGEST_KEY = "containerimage.digest"
CLIS = ("docker", "finch")


class PushError(Exception):
    """A user-facing error. Messages never include a password or a command's output."""


class Runner(Protocol):
    def __call__(self, args: Sequence[str], *, cwd: Path | None = None, stdin: str | None = None,
                 stream: bool = False) -> str: ...


def default_runner(args: Sequence[str], *, cwd: Path | None = None, stdin: str | None = None,
                   stream: bool = False) -> str:
    """Run ``args``; return stdout (``""`` when ``stream``, whose output goes to the terminal).
    A failure names the command and its status, never its output (it could hold the password)."""
    try:
        # The commands are this script's own fixed git, aws, docker and finch lines; no shell.
        proc = subprocess.run(list(args), cwd=cwd, input=stdin, text=True, check=False,  # noqa: S603
                              capture_output=not stream)
    except OSError as exc:
        raise PushError(f"could not run {args[0]}: {type(exc).__name__}") from None
    if proc.returncode != 0:
        raise PushError(f"{' '.join(args[:2])} exited with status {proc.returncode}")
    return "" if stream else proc.stdout


@dataclass(frozen=True)
class Repo:
    uri: str
    registry: str
    region: str
    name: str


def parse_repository_uri(uri: str) -> Repo:
    m = REPO_URI_RE.fullmatch(uri.strip())
    if not m:
        raise PushError("repository URI: expected <account>.dkr.ecr.<region>.amazonaws.com/<name> "
                        f"(the {IMAGE_STACK} output {OUTPUT_KEY})")
    return Repo(uri=uri.strip(), registry=m["registry"], region=m["region"], name=m["name"])


def repository_uri_from_outputs(path: Path) -> str:
    """``RepositoryUri`` of ``OwlBankBridgeImage`` in a ``cdk deploy --outputs-file`` JSON."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PushError(f"can't read CDK outputs {path}: {type(exc).__name__}") from None
    stack = doc.get(IMAGE_STACK) if isinstance(doc, dict) else None
    value = stack.get(OUTPUT_KEY) if isinstance(stack, dict) else None
    if not isinstance(value, str):
        raise PushError(f"CDK outputs {path}: no {IMAGE_STACK}.{OUTPUT_KEY} (deploy {IMAGE_STACK} "
                        "with --outputs-file, or pass --repository-uri)")
    return value


def git_tag(runner: Runner, *, allow_dirty: bool) -> str:
    """``git-<12-hex commit>``; refuses a dirty tree unless ``allow_dirty`` (then ``-dirty``)."""
    dirty = bool(runner(["git", "status", "--porcelain"], cwd=REPO_ROOT).strip())
    if dirty and not allow_dirty:
        raise PushError("the working tree has uncommitted changes, so the image would not be the "
                        "tagged commit: commit or stash them, or pass --allow-dirty")
    sha = runner(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise PushError("git rev-parse HEAD did not return a commit")
    return f"git-{sha[:12]}" + ("-dirty" if dirty else "")


def aws_args(*args: str, region: str, profile: str | None) -> list[str]:
    return ["aws", *args, "--region", region, *(["--profile", profile] if profile else [])]


def build_args(image_ref: str, metadata_file: str) -> list[str]:
    return ["docker", "buildx", "build", "--platform", PLATFORM, "-f", DOCKERFILE,
            "-t", image_ref, "--provenance=false", "--sbom=false",
            "--metadata-file", metadata_file, "--push", "."]


def finch_build_args(image_ref: str) -> list[str]:
    return ["finch", "build", "--platform", PLATFORM, "-f", DOCKERFILE, "-t", image_ref, "."]


def describe_digest_args(repo: Repo, tag: str, profile: str | None) -> list[str]:
    return aws_args("ecr", "describe-images", "--repository-name", repo.name,
                    "--image-ids", f"imageTag={tag}", region=repo.region, profile=profile) + [
        "--query", "imageDetails[0].imageDigest", "--output", "text"]


def check_digest(value: str, source: str) -> str:
    digest = value.strip()
    if not DIGEST_RE.fullmatch(digest):
        raise PushError(f"{source} gave no valid image digest (expected sha256:<64 hex>)")
    return digest


def default_cli(environ: dict[str, str] | None = None) -> str:
    """``$CDK_DOCKER`` when it names a supported CLI, else ``docker``."""
    value = (os.environ if environ is None else environ).get("CDK_DOCKER", "")
    return value if value in CLIS else "docker"


def parse_digest(metadata: str) -> str:
    """The pushed image's digest from buildx's metadata JSON, checked to be ``sha256:<64 hex>``."""
    try:
        doc = json.loads(metadata)
    except ValueError:
        raise PushError("buildx metadata is not JSON") from None
    digest = doc.get(METADATA_DIGEST_KEY) if isinstance(doc, dict) else None
    if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
        raise PushError(f"buildx metadata has no valid {METADATA_DIGEST_KEY} (expected sha256:<64 hex>)")
    return digest


def run(argv: Sequence[str], *, runner: Runner = default_runner, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--outputs", type=Path, help=f"cdk outputs file with {IMAGE_STACK}.{OUTPUT_KEY}")
    src.add_argument("--repository-uri", help="the ECR repository URI")
    p.add_argument("--profile", help="AWS CLI profile for the ECR login")
    p.add_argument("--cli", choices=CLIS, default=default_cli(),
                   help="container CLI (default: $CDK_DOCKER if docker or finch, else docker)")
    p.add_argument("--allow-dirty", action="store_true",
                   help="build from a tree with uncommitted changes (tag gets -dirty)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="apply", action="store_false",
                      help="print the commands and run none but the read-only git checks (default)")
    mode.add_argument("--apply", dest="apply", action="store_true", help="build, push, print the digest")
    p.set_defaults(apply=False)
    a = p.parse_args(list(argv))
    try:
        uri = a.repository_uri if a.repository_uri is not None else repository_uri_from_outputs(a.outputs)
        repo = parse_repository_uri(uri)
        tag = git_tag(runner, allow_dirty=a.allow_dirty)
        ref = f"{repo.uri}:{tag}"
        password_cmd = aws_args("ecr", "get-login-password", region=repo.region, profile=a.profile)
        login_cmd = [a.cli, "login", "--username", "AWS", "--password-stdin", repo.registry]
        if a.cli == "finch":
            steps = [finch_build_args(ref), ["finch", "push", ref],
                     describe_digest_args(repo, tag, a.profile)]
        else:
            steps = [build_args(ref, "<metadata-file>")]
        if not a.apply:
            print("dry run: would run, from the repository root:", file=out)
            print(f"  {' '.join(password_cmd)} | {' '.join(login_cmd)}", file=out)
            for step in steps:
                print(f"  {' '.join(step)}", file=out)
            print(f"image: {ref}  (rerun with --apply to build and push)", file=out)
            return 0
        password = runner(password_cmd).strip()
        if not password:
            raise PushError("aws ecr get-login-password returned nothing")
        runner(login_cmd, stdin=password)
        if a.cli == "finch":
            build_cmd, push_cmd, describe_cmd = steps
            runner(build_cmd, cwd=REPO_ROOT, stream=True)
            runner(push_cmd, cwd=REPO_ROOT, stream=True)
            digest = check_digest(runner(describe_cmd), "aws ecr describe-images")
        else:
            digest = _docker_build_push(runner, ref)
    except PushError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"pushed {ref}", file=out)
    print(f"digest {digest}", file=out)
    print(f"next: deploy with -c bridgeImageDigest={digest}", file=out)
    return 0


def _docker_build_push(runner: Runner, ref: str) -> str:
    """``docker buildx build --push``; the digest from its metadata file."""
    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp) / "buildx-metadata.json"
        runner(build_args(ref, str(meta)), cwd=REPO_ROOT, stream=True)
        try:
            metadata = meta.read_text(encoding="utf-8")
        except OSError:
            raise PushError("buildx wrote no metadata file") from None
    return parse_digest(metadata)


def main(argv: Sequence[str] | None = None) -> int:
    return run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
