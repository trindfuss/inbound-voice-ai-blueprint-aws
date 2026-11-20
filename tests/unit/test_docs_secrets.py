""" (docs part): the documents in scope describe Secrets Manager and name only real things.

Structural checks, not phrase matching on behaviour: no document refers to the macOS Keychain, no
``-c`` context key is a secret field name, every ``scripts/<name>.py`` exists, every ``--flag``
shown for the three secrets scripts is in that script's ``--help``, every ``owlbank/twilio/<x>``
secret name is in ``data_stack.SECRETS`` (read statically).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from owlbank.secret_schema import SCHEMAS

REPO = Path(__file__).resolve().parents[2]
DOCS = sorted([REPO / "README.md", REPO / "infra" / "README.md", *(REPO / "docs").rglob("*.md")])
SCRIPT_DIRS = (REPO / "scripts", REPO / "infra" / "scripts")
FLAG_SCRIPTS = ("check_secrets", "populate_secrets", "twilio_setup")
KEYCHAIN_NEEDLES = ("keychain", "find-generic-password", "add-generic-password")

_SCRIPT_REF = re.compile(r"\bscripts/([A-Za-z0-9_]+)\.py\b")
_FLAG_USE = re.compile(r"(?:scripts/)?\b(" + "|".join(FLAG_SCRIPTS) + r")\.py([^`\n]*)")
_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
_CONTEXT_KEY = re.compile(r"(?<!\S)-c\s+([A-Za-z][A-Za-z0-9_.]*)=")
_SECRET_NAME = re.compile(r"\bowlbank/twilio/[a-z-]+")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _script_exists(name: str) -> bool:
    return any((d / f"{name}.py").is_file() for d in SCRIPT_DIRS)


def _help_flags(name: str) -> set[str]:
    out = subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), "--help"],  # noqa: S603 (our own script)
                         capture_output=True, text=True, cwd=REPO, check=True).stdout
    return set(_FLAG.findall(out))


def _secret_names(registry: str = "SECRETS") -> set[str]:
    tree = ast.parse(_text(REPO / "infra" / "owlbank_infra" / "data_stack.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == registry:
            return {"owlbank/" + v.elts[0].value for v in node.value.values}
    raise AssertionError(f"{registry} not found in data_stack.py")


def test_scope_is_found() -> None:
    names = {p.name for p in DOCS}
    assert {"README.md", "production-guide.md", "runbook.md", "solution-guide.md"} <= names


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_keychain_in_docs(path: Path) -> None:
    text = _text(path).lower()
    assert not [n for n in KEYCHAIN_NEEDLES if n in text]


def test_no_context_key_carries_a_secret_value() -> None:
    fields = {_norm(f) for schema in SCHEMAS.values() for f in schema}
    bad = []
    for path in DOCS:
        for key in _CONTEXT_KEY.findall(_text(path)):
            if any(_norm(key).endswith(f) for f in fields):
                bad.append(f"{path.relative_to(REPO)}: -c {key}")
    assert not bad, bad


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_named_scripts_exist(path: Path) -> None:
    missing = {n for n in _SCRIPT_REF.findall(_text(path)) if not _script_exists(n)}
    assert not missing, missing


def test_secrets_scripts_exist() -> None:
    assert all((REPO / "scripts" / f"{n}.py").is_file() for n in FLAG_SCRIPTS)


@pytest.mark.parametrize("script", FLAG_SCRIPTS)
def test_flags_shown_exist_in_the_scripts_help(script: str) -> None:
    known = _help_flags(script)
    assert "--help" in known
    bad = []
    for path in DOCS:
        for name, rest in _FLAG_USE.findall(_text(path)):
            if name == script:
                bad += [f"{path.relative_to(REPO)}: {f}" for f in _FLAG.findall(rest) if f not in known]
    assert not bad, bad


def test_secret_names_are_in_the_registry() -> None:
    known = _secret_names()
    assert {"owlbank/twilio/api-key", "owlbank/twilio/auth-token", "owlbank/twilio/basic-auth"} == known
    # Secrets only some configurations create (human-leg ``contextApiRestrictedKey=on``).
    known |= _secret_names("OPTIONAL_SECRETS")
    bad = [f"{p.relative_to(REPO)}: {n}" for p in DOCS for n in _SECRET_NAME.findall(_text(p))
           if n not in known]
    assert not bad, bad
