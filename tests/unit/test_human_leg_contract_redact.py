"""The live contract test's redaction: no value under a token/secret/password key survives, at any depth."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.live._redact import REDACTED, SENSITIVE_KEY_PARTS, redact

_SECRET_VALUE = "s3cr3t-value-" + "x" * 8
_KEYS = st.one_of(
    st.text(max_size=8),
    st.sampled_from(["auth_token", "AuthToken", "api_SECRET", "Password", "token", "client_secret_2", "name", "sid"]),
)
_LEAVES = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=12))
_JSON = st.recursive(_LEAVES, lambda kids: st.one_of(st.lists(kids, max_size=4),
                                                     st.dictionaries(_KEYS, kids, max_size=4)), max_leaves=25)


def _sensitive(key: object) -> bool:
    return any(p in str(key).lower() for p in SENSITIVE_KEY_PARTS)


def _no_value_under_sensitive_key(node: Any) -> bool:
    if isinstance(node, dict):
        return all((v == REDACTED) if _sensitive(k) else _no_value_under_sensitive_key(v) for k, v in node.items())
    if isinstance(node, list):
        return all(_no_value_under_sensitive_key(v) for v in node)
    return True


def test_a_fake_account_response_is_redacted() -> None:
    body = {"sid": "AC" + "0" * 32, "auth_token": _SECRET_VALUE, "owner": {"Password": _SECRET_VALUE},
            "subresource_uris": [{"api_secret": {"k": _SECRET_VALUE}, "ok": "kept"}]}
    out = redact(body)
    assert _SECRET_VALUE not in repr(out)
    assert out["sid"] == body["sid"] and out["subresource_uris"][0]["ok"] == "kept"


def test_the_input_is_not_modified() -> None:
    body = {"auth_token": _SECRET_VALUE}
    redact(body)
    assert body == {"auth_token": _SECRET_VALUE}


@settings(max_examples=200, deadline=None)
@given(_JSON)
def test_no_value_under_a_token_secret_or_password_key_survives(doc: Any) -> None:
    assert _no_value_under_sensitive_key(redact(doc))


@settings(max_examples=100, deadline=None)
@given(_JSON, st.sampled_from(["token", "Secret", "PASSWORD", "x_auth_token_y"]))
def test_a_planted_secret_never_survives_at_any_depth(doc: Any, key: str) -> None:
    wrapped = {"a": [doc, {"b": {key: {"deep": [_SECRET_VALUE]}}}], key: _SECRET_VALUE}
    assert _SECRET_VALUE not in repr(redact(wrapped))
