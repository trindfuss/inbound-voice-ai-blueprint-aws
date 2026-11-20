"""Template normalizer and differ for the Fargate snapshot test.

Pure Python, no CDK import, so the root suite's property test (tests/property/) can load it.

Normalization replaces exactly two things, nothing else:

- every 64-hex run (a content-addressed asset hash: S3 object keys, ECR image tags, asset
  directory names), in string values and keys, by :data:`HASH_PLACEHOLDER`;
- the value of every ``aws:asset:*`` key (CDK's asset metadata) by :data:`ASSET_PLACEHOLDER`.

A 64-hex run must stand alone: a longer run of hex characters is left as it is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

HASH_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
HASH_PLACEHOLDER = "<asset-hash>"
#: Lambda ``currentVersion`` logical IDs embed a 32-hex hash of the function's code and
#: configuration (``…CurrentVersion<8 hex><32 hex>``), so new code changes them like an asset
#: hash does (identical except asset hashes).
VERSION_ID_RE = re.compile(r"(CurrentVersion[0-9A-F]{8})[0-9a-f]{32}(?![0-9a-fA-F])")
VERSION_PLACEHOLDER = r"\1<code-hash>"
ASSET_KEY_PREFIX = "aws:asset:"
ASSET_PLACEHOLDER = "<aws-asset-metadata>"


def _norm_str(s: str) -> str:
    return VERSION_ID_RE.sub(VERSION_PLACEHOLDER, HASH_RE.sub(HASH_PLACEHOLDER, s))


def normalize(obj: Any) -> Any:
    """Return a normalized copy of a JSON value (the input is not modified)."""
    if isinstance(obj, str):
        return _norm_str(obj)
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            nkey = _norm_str(key)
            out[nkey] = ASSET_PLACEHOLDER if key.startswith(ASSET_KEY_PREFIX) else normalize(value)
        return out
    return obj


def _show(value: Any) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 200 else text[:197] + "..."


def diff(expected: Any, actual: Any, path: str = "$") -> list[str]:
    """Readable differences between two JSON values, one line per differing path."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            sub = f"{path}.{key}"
            if key not in actual:
                lines.append(f"{sub}: removed (was {_show(expected[key])})")
            elif key not in expected:
                lines.append(f"{sub}: added ({_show(actual[key])})")
            else:
                lines.extend(diff(expected[key], actual[key], sub))
        return lines
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: list length {len(expected)} -> {len(actual)}: "
                    f"{_show(expected)} -> {_show(actual)}"]
        return [line for i, (e, a) in enumerate(zip(expected, actual, strict=True))
                for line in diff(e, a, f"{path}[{i}]")]
    if type(expected) is not type(actual) or expected != actual:
        return [f"{path}: {_show(expected)} -> {_show(actual)}"]
    return []


def dump(template: Any) -> str:
    """The stored form of a snapshot: sorted keys, stable indentation, trailing newline."""
    return json.dumps(template, indent=1, sort_keys=True) + "\n"


def load(path: Path) -> Any:
    return json.loads(path.read_text())
