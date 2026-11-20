"""The Fargate templates stay identical except asset hashes.

For each covered context set (default; custom domain; ``s3PublicAccessBlockMode=account``;
``stage=prod`` with a domain), every stack's template, normalized by
:mod:`snapshot_util` (64-hex asset hashes and ``aws:asset:*`` metadata only), must
equal the stored snapshot under ``snapshots/<context-set>/<stack>.json``. The context sets
are the existing fictional fixtures of ``conftest.py`` (``synth_default``, ``synth_domain``,
``synth_bpa_account``, ``synth_prod``); nothing is looked up or deployed.

Baseline: the design fixes the snapshots at commit ``ea6d28c``. When they were captured
 ``git diff ea6d28c HEAD -- infra src scripts data`` was empty, so
capturing at HEAD equalled capturing at ``ea6d28c``.

Regenerating (only for an approved, intended Fargate change, never to make this pass):
``OWLBANK_UPDATE_SNAPSHOTS=1 uv run --directory infra pytest -q tests/test_fargate_snapshot.py``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import snapshot_util as su
from conftest import INFRA, REPO

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"
CONTEXT_SETS = {
    "default": "synth_default",
    "custom-domain": "synth_domain",
    "bpa-account": "synth_bpa_account",
    "prod-domain": "synth_prod",
}
UPDATE = os.environ.get("OWLBANK_UPDATE_SNAPSHOTS") == "1"


def _normalized(synth) -> dict[str, Any]:
    return {name: su.normalize(tpl) for name, tpl in synth.templates.items()}


@pytest.mark.parametrize("context_set", sorted(CONTEXT_SETS))
def test_fargate_templates_match_snapshots(context_set: str, request: pytest.FixtureRequest) -> None:
    synth = request.getfixturevalue(CONTEXT_SETS[context_set])
    actual = _normalized(synth)
    folder = SNAPSHOTS / context_set
    if UPDATE:
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.json"):
            old.unlink()
        for name, tpl in actual.items():
            (folder / f"{name}.json").write_text(su.dump(tpl))
    stored = {p.stem: su.load(p) for p in sorted(folder.glob("*.json"))}
    assert stored, f"no snapshots in {folder}"
    assert sorted(actual) == sorted(stored), "the set of stacks changed"
    problems = [f"{name}: {line}" for name in sorted(stored)
                for line in su.diff(stored[name], actual[name])]
    assert not problems, (f"{context_set}: Fargate templates differ from the snapshots"
                          f":\n" + "\n".join(problems[:50]))


def test_one_property_change_fails_with_a_readable_diff() -> None:
    """A deliberate one-property change in a scratch copy of a stored template is caught, and
    the report names the path and both values."""
    path = SNAPSHOTS / "default" / "OwlBankBridge.json"
    expected = su.load(path)
    scratch = copy.deepcopy(expected)
    lid, res = next((lid, r) for lid, r in sorted(scratch["Resources"].items())
                    if r["Type"] == "AWS::ECS::Service")
    before = res["Properties"]["DesiredCount"]
    res["Properties"]["DesiredCount"] = before + 7
    lines = su.diff(expected, su.normalize(scratch))
    assert lines == [f"$.Resources.{lid}.Properties.DesiredCount: {before} -> {before + 7}"]


def test_snapshots_hold_only_normalized_hashes() -> None:
    for path in SNAPSHOTS.rglob("*.json"):
        tpl = su.load(path)
        assert su.normalize(tpl) == tpl, path


# --------------------------------------------------------------------------- leak scan

def _leakscan():
    spec = importlib.util.spec_from_file_location("owlbank_leakscan", REPO / "tests" / "leakscan.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in (k, *_strings(v))]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


FICTIONAL_PHONE = re.compile(r"^\+1\d{3}55501\d{2}$")  # NANP 555-0100..0199
PHONE_LIKE = re.compile(r"(?<![\w.])\+\d{10,15}(?!\d)")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
RESERVED_EMAIL_DOMAINS = {"example.com", "example.net", "example.org"}
TWILIO_SID = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}[0-9a-f]{32}(?![0-9a-f])")
ACCOUNT_IN_ARN = re.compile(r"arn:aws[\w-]*:[\w-]*:[\w-]*:(\d{12}):")


def _local_account_values() -> list[str]:
    """Account-specific identifiers from this checkout's git-ignored deploy outputs, if any
    (Twilio SIDs from ``twilio.context.json``, AWS account IDs from ``cdk-outputs*.json``). They
    are only compared against, never written anywhere."""
    values: set[str] = set()
    twilio = INFRA / "twilio.context.json"
    if twilio.is_file():
        values.update(s for s in _strings(json.loads(twilio.read_text())) if TWILIO_SID.fullmatch(s))
    for out in INFRA.glob("cdk-outputs*.json"):
        values.update(ACCOUNT_IN_ARN.findall(out.read_text()))
    values.discard("111111111111")  # the fake account the tests synthesize with
    return sorted(values)


def test_snapshots_carry_no_phone_number_secret_or_email() -> None:
    files = sorted(SNAPSHOTS.rglob("*.json"))
    assert files
    leakscan = _leakscan()
    local = leakscan.compile_values(_local_account_values())
    for path in files:
        text = path.read_text()
        strings = _strings(json.loads(text))
        for s in strings:
            for m in PHONE_LIKE.findall(s):
                assert FICTIONAL_PHONE.match(m), f"{path}: non-fictional phone number"
            for m in EMAIL.finditer(s):
                assert m.group(1).lower() in RESERVED_EMAIL_DOMAINS, f"{path}: real-looking email"
            for m in TWILIO_SID.findall(s):
                assert len(set(m[2:])) == 1, f"{path}: Twilio SID is not a fictional placeholder"
        assert set(ACCOUNT_IN_ARN.findall(text)) <= {"111111111111"}, f"{path}: real account ID"
        assert not leakscan.leaks(local, text), f"{path}: carries an account-specific value"
