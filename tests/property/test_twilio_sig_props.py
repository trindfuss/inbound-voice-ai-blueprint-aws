"""Property tests for owlbank.twilio_sig.

Signatures are produced by Twilio's own ``RequestValidator.compute_signature``
over arbitrary URLs and parameter dicts. Valid ones must pass; any single
character change (substitution, insertion or deletion) in the URL, one
parameter name or value, the JSON body, or the signature must fail.
"""

from __future__ import annotations

import hashlib

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from twilio.request_validator import RequestValidator

from owlbank.twilio_sig import validate

host_chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
url_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~%!$&'()*+,;=:@/"

hosts = st.lists(st.text(alphabet=host_chars, min_size=1, max_size=10), min_size=1,
                 max_size=3).map(".".join)
ports = st.one_of(st.none(), st.integers(min_value=1, max_value=65535))
paths = st.text(alphabet=url_chars, max_size=30).map(lambda p: "/" + p)
queries = st.one_of(st.none(), st.text(alphabet=url_chars + "?", min_size=1, max_size=30))


@st.composite
def urls(draw: st.DrawFn) -> str:
    scheme = draw(st.sampled_from(["https", "http"]))
    port = draw(ports)
    netloc = draw(hosts) + (f":{port}" if port is not None else "")
    query = draw(queries)
    return f"{scheme}://{netloc}{draw(paths)}" + (f"?{query}" if query is not None else "")


params_strategy = st.dictionaries(st.text(min_size=1, max_size=15), st.text(max_size=30),
                                  max_size=8)
tokens = st.text(min_size=1, max_size=40)
any_char = st.characters(codec="utf-8")


@st.composite
def single_char_edit(draw: st.DrawFn, s: str) -> str:
    """Return ``s`` with exactly one character substituted, inserted or deleted."""
    ops = ["insert"] + (["substitute", "delete"] if s else [])
    op = draw(st.sampled_from(ops))
    if op == "insert":
        i = draw(st.integers(min_value=0, max_value=len(s)))
        return s[:i] + draw(any_char) + s[i:]
    i = draw(st.integers(min_value=0, max_value=len(s) - 1))
    if op == "delete":
        return s[:i] + s[i + 1 :]
    c = draw(any_char.filter(lambda ch: ch != s[i]))
    return s[:i] + c + s[i + 1 :]


def sign(url: str, params: dict[str, str], token: str) -> str:
    return RequestValidator(token).compute_signature(url, params)


@settings(max_examples=300)
@given(url=urls(), params=params_strategy, token=tokens)
def test_valid_form_signatures_pass(url: str, params: dict[str, str], token: str) -> None:
    assert validate(url, params, sign(url, params, token), token)


@settings(max_examples=300)
@given(url=urls(), params=params_strategy, token=tokens, data=st.data())
def test_url_edit_fails(url: str, params: dict[str, str], token: str, data: st.DataObject) -> None:
    sig = sign(url, params, token)
    mutated = data.draw(single_char_edit(url))
    assert not validate(mutated, params, sig, token)


@settings(max_examples=300)
@given(url=urls(), params=params_strategy.filter(bool), token=tokens, data=st.data(),
       edit_name=st.booleans())
def test_param_edit_fails(url: str, params: dict[str, str], token: str, data: st.DataObject,
                          edit_name: bool) -> None:
    sig = sign(url, params, token)
    name = data.draw(st.sampled_from(sorted(params)))
    mutated = dict(params)
    if edit_name:
        new_name = data.draw(single_char_edit(name))
        # an edit that merges two parameters, or empties a name, is not a single-char
        # change to the parameter set
        assume(new_name and new_name not in params)
        mutated[new_name] = mutated.pop(name)
    else:
        mutated[name] = data.draw(single_char_edit(params[name]))
    assert not validate(url, mutated, sig, token)


@settings(max_examples=300)
@given(url=urls(), params=params_strategy, token=tokens, data=st.data())
def test_signature_edit_fails(url: str, params: dict[str, str], token: str,
                              data: st.DataObject) -> None:
    sig = sign(url, params, token)
    mutated = data.draw(single_char_edit(sig))
    assert not validate(url, params, mutated, token)


@settings(max_examples=200)
@given(url=urls(), params=params_strategy, token=tokens, other=tokens)
def test_other_auth_token_fails(url: str, params: dict[str, str], token: str, other: str) -> None:
    assume(other != token)
    assert not validate(url, params, sign(url, params, token), other)


# ----------------------------------------------------------------------- JSON bodies


def json_url(base: str, body: str) -> str:
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}bodySHA256={hashlib.sha256(body.encode('utf-8')).hexdigest()}"


bases = urls().filter(lambda u: "bodySHA256" not in u and "#" not in u)


@settings(max_examples=200)
@given(base=bases, body=st.text(max_size=200), token=tokens, data=st.data())
def test_json_body_valid_and_edits_fail(base: str, body: str, token: str,
                                        data: st.DataObject) -> None:
    url = json_url(base, body)
    sig = sign(url, {}, token)
    assert validate(url, None, sig, token, body=body)
    assert not validate(url, None, sig, token, body=data.draw(single_char_edit(body)))
    assert not validate(data.draw(single_char_edit(url)), None, sig, token, body=body)
    assert not validate(url, None, data.draw(single_char_edit(sig)), token, body=body)
