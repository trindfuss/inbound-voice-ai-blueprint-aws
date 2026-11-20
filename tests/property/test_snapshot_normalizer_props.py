"""The Fargate snapshot normalizer (``infra/tests/snapshot_util.py``).

- Idempotent: normalizing twice equals normalizing once, for any JSON tree, including trees
  with asset hashes and ``aws:asset:*`` keys planted anywhere.
- Changes nothing in a tree that has no standalone 64-hex run and no ``aws:asset:*`` key.
- Replaces every standalone 64-hex run and every ``aws:asset:*`` value.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

_PATH = Path(__file__).resolve().parents[2] / "infra" / "tests" / "snapshot_util.py"
_spec = importlib.util.spec_from_file_location("owlbank_snapshot_util", _PATH)
assert _spec and _spec.loader
su = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(su)

HEX = "0123456789abcdefABCDEF"
hashes = st.text(HEX, min_size=64, max_size=64)
# Text that mixes hex-heavy runs with separators, so near-miss lengths (63, 65) come up.
texts = st.one_of(st.text(max_size=30), st.text(HEX + ".:/-_ ", max_size=140))
keys = st.one_of(st.text(max_size=12), st.sampled_from(["Metadata", "aws:cdk:path", "Properties"]))
scalars = st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), texts)


def trees(leaf: st.SearchStrategy[Any], key: st.SearchStrategy[str]) -> st.SearchStrategy[Any]:
    return st.recursive(leaf, lambda inner: st.one_of(
        st.lists(inner, max_size=4), st.dictionaries(key, inner, max_size=4)), max_leaves=25)


with_hash_text = st.tuples(texts, hashes, texts).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}")
asset_keys = st.text(max_size=12).map(lambda s: "aws:asset:" + s)
dirty_trees = trees(st.one_of(scalars, with_hash_text), st.one_of(keys, asset_keys, with_hash_text))


def _clean(obj: Any) -> bool:
    if isinstance(obj, str):
        return su.HASH_RE.search(obj) is None
    if isinstance(obj, list):
        return all(_clean(v) for v in obj)
    if isinstance(obj, dict):
        return all(_clean(k) and not k.startswith(su.ASSET_KEY_PREFIX) and _clean(v)
                   for k, v in obj.items())
    return True


clean_trees = trees(scalars, keys).filter(_clean)


@given(dirty_trees)
def test_normalize_is_idempotent(tree: Any) -> None:
    once = su.normalize(tree)
    assert su.normalize(once) == once


@given(clean_trees)
def test_normalize_changes_nothing_without_hashes_or_asset_keys(tree: Any) -> None:
    before = json.dumps(tree, sort_keys=True)
    assert su.normalize(tree) == tree
    assert json.dumps(tree, sort_keys=True) == before  # the input is not modified


@given(dirty_trees)
def test_normalized_tree_has_no_hash_and_no_asset_value(tree: Any) -> None:
    out = su.normalize(tree)

    def check(obj: Any) -> None:
        if isinstance(obj, str):
            assert su.HASH_RE.search(obj) is None
        elif isinstance(obj, list):
            for v in obj:
                check(v)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                check(k)
                if k.startswith(su.ASSET_KEY_PREFIX):
                    assert v == su.ASSET_PLACEHOLDER
                else:
                    check(v)

    check(out)


@given(texts, hashes, texts)
def test_hash_is_replaced_only_at_its_place(left: str, digest: str, right: str) -> None:
    out = su.normalize(f"{left}/{digest}/{right}")
    assert f"/{su.HASH_PLACEHOLDER}/" in out


def test_longer_hex_run_is_left_alone() -> None:
    assert su.normalize("a" * 65) == "a" * 65
    assert su.normalize("a" * 63) == "a" * 63


@given(prefix=st.text(alphabet="0123456789ABCDEF", min_size=8, max_size=8),
       code=st.text(alphabet="0123456789abcdef", min_size=32, max_size=32))
def test_lambda_version_ids_lose_only_their_code_hash(prefix: str, code: str) -> None:
    """A ``currentVersion`` logical ID keeps its name and 8-hex prefix; only the 32-hex code hash
    goes, in keys and in string values (``Fn::GetAtt``), so new Lambda code is not a template diff."""
    lid = f"AuthServiceFnCurrentVersion{prefix}{code}"
    tree = {"Resources": {lid: {"Type": "AWS::Lambda::Version"}},
            "Ref": {"Fn::GetAtt": [lid, "Version"]}}
    out = su.normalize(tree)
    want = f"AuthServiceFnCurrentVersion{prefix}<code-hash>"
    assert list(out["Resources"]) == [want]
    assert out["Ref"]["Fn::GetAtt"][0] == want
    assert su.normalize(out) == out
