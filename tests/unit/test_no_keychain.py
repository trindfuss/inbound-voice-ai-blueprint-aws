"""No file under scripts/, src/ or infra/owlbank_infra/ refers to the macOS Keychain.

Secrets live only in Secrets Manager (portable-secrets feature). This is a structural tree scan,
not a phrase check on behaviour. Nothing is excluded except compiled caches and node_modules.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = ("scripts", "src", "infra/owlbank_infra")
SKIP_DIRS = {"__pycache__", "node_modules", ".venv", "cdk.out"}
NEEDLES = ("keychain", "security find-generic-password")


def _files() -> list[Path]:
    out: list[Path] = []
    for root in ROOTS:
        for path in (REPO / root).rglob("*"):
            if path.is_file() and not SKIP_DIRS.intersection(path.parts) and path.suffix != ".pyc":
                out.append(path)
    return out


def test_scan_covers_the_tree() -> None:
    assert len(_files()) > 50


def test_no_keychain_reference_in_scripts_src_or_infra() -> None:
    hits = []
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue  # binary asset
        hits += [f"{path.relative_to(REPO)}: {n!r}" for n in NEEDLES if n in text]
    assert not hits, hits
