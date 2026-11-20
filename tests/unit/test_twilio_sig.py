"""Unit tests for owlbank.twilio_sig."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from twilio.request_validator import RequestValidator

from hypothesis import given
from hypothesis import strategies as st

from owlbank.twilio_sig import (
    FORM_CREDENTIALS,
    FORM_PLAIN,
    reconstruct_url,
    signed_url_form,
    validate,
    with_credentials,
)

TOKEN = "12345"
URL = "https://mycompany.com/myapp.php?foo=1&bar=2"
PARAMS = {
    "CallSid": "CA1234567890ABCDE",
    "Caller": "+12349013030",
    "Digits": "1234",
    "From": "+12349013030",
    "To": "+18005551212",
}


def spec_signature(url: str, params: dict[str, str], token: str) -> str:
    """Twilio's documented algorithm, written out independently of the SDK."""
    data = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()


EXPECTED = spec_signature(URL, PARAMS, TOKEN)


def sign(url: str, params: dict[str, str] | None = None, token: str = TOKEN) -> str:
    return RequestValidator(token).compute_signature(url, params or {})


def test_matches_documented_algorithm() -> None:
    assert sign(URL, PARAMS) == EXPECTED
    assert validate(URL, PARAMS, EXPECTED, TOKEN)


def test_agrees_with_sdk_validate_on_normal_input() -> None:
    sdk = RequestValidator(TOKEN)
    for url in (URL, "https://h.example/voice", "http://h.example:8080/x?y=1"):
        sig = sign(url, PARAMS)
        assert validate(url, PARAMS, sig, TOKEN) is True
        assert sdk.validate(url, PARAMS, sig) is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda u, p, s, t: (u + "x", p, s, t),
        lambda u, p, s, t: (u, {**p, "Digits": "1235"}, s, t),
        lambda u, p, s, t: (u, {**p, "Extra": "1"}, s, t),
        lambda u, p, s, t: (u, {k: v for k, v in p.items() if k != "To"}, s, t),
        lambda u, p, s, t: (u, p, s[:-2] + "x=", t),
        lambda u, p, s, t: (u, p, s, "54321"),
        lambda u, p, s, t: (u.replace("https", "http"), p, s, t),
        lambda u, p, s, t: (u.split("?")[0], p, s, t),  # dropped query string
    ],
)
def test_any_tampering_fails(mutate) -> None:
    assert not validate(*mutate(URL, PARAMS, EXPECTED, TOKEN))


def test_port_variants_accepted_like_sdk() -> None:
    no_port = "https://h.example/voice"
    with_port = "https://h.example:443/voice"
    assert validate(with_port, PARAMS, sign(no_port, PARAMS), TOKEN)
    assert validate(no_port, PARAMS, sign(with_port, PARAMS), TOKEN)
    # a non-default port must match what Twilio signed
    assert not validate("https://h.example:8443/voice", PARAMS, sign(with_port, PARAMS), TOKEN)


def test_no_url_normalization() -> None:
    url = "https://h.example/voice"
    sig = sign(url, PARAMS)
    assert not validate("HTTPS://h.example/voice", PARAMS, sig, TOKEN)
    assert not validate(url + "?", PARAMS, sig, TOKEN)
    assert not validate(url + "#", PARAMS, sig, TOKEN)


@pytest.mark.parametrize(
    "url",
    ["ftp://h/x", "https://", "https://u:p@h/x", "https://h:/x", "https://h:44x/x",
     "https://a:b:c/x", "https://[::1/x", "/relative"],
)
def test_malformed_urls_fail_without_raising(url: str) -> None:
    assert not validate(url, PARAMS, sign(url, PARAMS), TOKEN)


def test_ipv6_host() -> None:
    url = "https://[::1]:8443/voice"
    assert validate(url, PARAMS, sign(url, PARAMS), TOKEN)


@pytest.mark.parametrize(
    "url,sig,token",
    [("", EXPECTED, TOKEN), (URL, "", TOKEN), (URL, EXPECTED, ""), (URL, None, TOKEN),
     (URL, EXPECTED, None), (None, EXPECTED, TOKEN)],
)
def test_empty_or_missing_inputs_fail(url, sig, token) -> None:
    assert not validate(url, PARAMS, sig, token)


def test_non_ascii_signature_fails_without_raising() -> None:
    assert not validate(URL, PARAMS, "é" * 28, TOKEN)


def test_none_params_equals_empty() -> None:
    url = "https://h.example/status"
    assert validate(url, None, sign(url), TOKEN)


# ----------------------------------------------------------------------- JSON bodies


def json_request(body: str) -> tuple[str, str]:
    url = "https://h.example/hook?bodySHA256=" + hashlib.sha256(body.encode()).hexdigest()
    return url, sign(url)


def test_json_body_passes() -> None:
    body = json.dumps({"CallSid": "CA1", "Status": "completed"})
    url, sig = json_request(body)
    assert validate(url, None, sig, TOKEN, body=body)
    assert validate(url, {}, sig, TOKEN, body=body)


def test_json_body_tampering_fails() -> None:
    body = json.dumps({"CallSid": "CA1"})
    url, sig = json_request(body)
    assert not validate(url, None, sig, TOKEN, body=body.replace("CA1", "CA2"))


def test_json_body_requires_hash_param_and_vice_versa() -> None:
    body = "{}"
    url, sig = json_request(body)
    assert not validate(url, None, sig, TOKEN)  # hash in URL but no body
    plain = "https://h.example/hook"
    assert not validate(plain, None, sign(plain), TOKEN, body=body)  # body but no hash
    assert not validate(url, {"a": "b"}, sig, TOKEN, body=body)  # both params and body


# ----------------------------------------------------------------------- reconstruct_url


def test_reconstruct_plain() -> None:
    assert reconstruct_url("https", "h.example", "/voice/incoming", None) == (
        "https://h.example/voice/incoming"
    )


def test_reconstruct_keeps_raw_query() -> None:
    q = "b=2&a=%20x&a=1"
    assert reconstruct_url("https", "h.example", "/v", q) == "https://h.example/v?" + q
    assert reconstruct_url("https", "h.example", "/v", "?" + q) == "https://h.example/v?" + q
    assert reconstruct_url("https", "h.example", "/v", "") == "https://h.example/v"


def test_reconstruct_behind_proxy() -> None:
    # ALB/API Gateway: internal http, public https on the default port
    url = reconstruct_url("http", "10.0.1.5:8080", "/ws", None, forwarded_proto="https",
                          forwarded_host="voice.owlbank.example", forwarded_port="443")
    assert url == "https://voice.owlbank.example/ws"


def test_reconstruct_forwarded_lists_use_first() -> None:
    url = reconstruct_url("http", "x", "/v", None, forwarded_proto="https, http",
                          forwarded_host="a.example, b.internal")
    assert url == "https://a.example/v"


def test_reconstruct_non_default_port() -> None:
    assert reconstruct_url("https", "h.example", "/v", None, forwarded_port=8443) == (
        "https://h.example:8443/v"
    )
    assert reconstruct_url("http", "h.example", "/v", None, forwarded_port="80") == (
        "http://h.example/v"
    )
    # a port already on the host wins
    assert reconstruct_url("https", "h.example:9000", "/v", None, forwarded_port=8443) == (
        "https://h.example:9000/v"
    )


def test_reconstruct_normalizes_path_and_scheme_case() -> None:
    assert reconstruct_url("HTTPS", "h.example", "v", None) == "https://h.example/v"
    assert reconstruct_url("https", "h.example", "", None) == "https://h.example/"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scheme": "ftp", "host": "h"},
        {"scheme": "https", "host": ""},
        {"scheme": "https", "host": "u:p@h"},
        {"scheme": "https", "host": "h/x"},
        {"scheme": "https", "host": "h", "forwarded_port": "44x"},
    ],
)
def test_reconstruct_rejects_bad_input(kwargs) -> None:
    with pytest.raises(ValueError):
        reconstruct_url(path="/v", query=None, **kwargs)


def test_reconstructed_url_validates() -> None:
    public = "https://voice.owlbank.example/voice/factor?step=zip"
    sig = sign(public, PARAMS)
    url = reconstruct_url("http", "internal:8080", "/voice/factor", "step=zip",
                          forwarded_proto="https", forwarded_host="voice.owlbank.example",
                          forwarded_port="443")
    assert validate(url, PARAMS, sig, TOKEN)


# --------------------------------------------------------------------------- user info forms

CREDS = ("owlbank-webhook", "0123456789abcdef" * 3)
BARE = "https://api.example.com/v1/voice/incoming"
WITH_CREDS = f"https://{CREDS[0]}:{CREDS[1]}@api.example.com/v1/voice/incoming"


def test_with_credentials_inserts_user_info_before_the_host() -> None:
    assert with_credentials(BARE, *CREDS) == WITH_CREDS
    assert with_credentials("http://h.example:8080/x?y=1", "u", "p") == "http://u:p@h.example:8080/x?y=1"


@pytest.mark.parametrize("user,password", [("", "p"), ("u", ""), ("u:v", "p"), ("u@x", "p"), ("u", "p@q"),
                                           ("u", "p/q"), ("u", "p?q"), ("u", "p#q"), ("u", "p q")])
def test_with_credentials_refuses_credentials_that_change_the_parse(user: str, password: str) -> None:
    assert with_credentials(BARE, user, password) is None


def test_with_credentials_refuses_urls_that_already_have_user_info_or_are_malformed() -> None:
    assert with_credentials(WITH_CREDS, *CREDS) is None
    assert with_credentials("ftp://h.example/x", *CREDS) is None
    assert with_credentials("not a url", *CREDS) is None


def test_signed_url_form_names_the_form_that_matched() -> None:
    plain = sign(BARE, PARAMS)
    creds = sign(WITH_CREDS, PARAMS)
    assert signed_url_form(BARE, PARAMS, plain, TOKEN, credentials=CREDS) == FORM_PLAIN
    assert signed_url_form(BARE, PARAMS, creds, TOKEN, credentials=CREDS) == FORM_CREDENTIALS
    assert signed_url_form(BARE, PARAMS, plain, TOKEN) == FORM_PLAIN


def test_the_credentials_form_needs_the_credentials_to_be_passed() -> None:
    """Without the verified credentials the URL-with-user-info signature is not accepted."""
    creds = sign(WITH_CREDS, PARAMS)
    assert signed_url_form(BARE, PARAMS, creds, TOKEN) is None
    assert validate(BARE, PARAMS, creds, TOKEN) is False
    assert validate(BARE, PARAMS, creds, TOKEN, credentials=CREDS) is True


def test_only_the_given_credentials_are_accepted() -> None:
    other = sign("https://someone:else@api.example.com/v1/voice/incoming", PARAMS)
    assert signed_url_form(BARE, PARAMS, other, TOKEN, credentials=CREDS) is None
    wrong_pw = sign(f"https://{CREDS[0]}:nope@api.example.com/v1/voice/incoming", PARAMS)
    assert signed_url_form(BARE, PARAMS, wrong_pw, TOKEN, credentials=CREDS) is None


def test_both_forms_also_accept_the_port_variant() -> None:
    for signed in ("https://api.example.com:443/v1/voice/incoming",
                   f"https://{CREDS[0]}:{CREDS[1]}@api.example.com:443/v1/voice/incoming"):
        assert validate(BARE, PARAMS, sign(signed, PARAMS), TOKEN, credentials=CREDS)
    assert not validate(BARE, PARAMS, sign("https://api.example.com:444/v1/voice/incoming", PARAMS),
                        TOKEN, credentials=CREDS)


def test_credentials_never_rescue_a_wrong_token_or_tampered_params() -> None:
    sig = sign(WITH_CREDS, PARAMS)
    assert signed_url_form(BARE, PARAMS, sig, "54321", credentials=CREDS) is None
    assert signed_url_form(BARE, {**PARAMS, "Digits": "9999"}, sig, TOKEN, credentials=CREDS) is None
    assert signed_url_form(BARE, PARAMS, "", TOKEN, credentials=CREDS) is None
    assert signed_url_form("not a url", PARAMS, sig, TOKEN, credentials=CREDS) is None


def test_url_given_to_validate_still_rejects_user_info() -> None:
    assert signed_url_form(WITH_CREDS, PARAMS, sign(WITH_CREDS, PARAMS), TOKEN, credentials=CREDS) is None


_cred = st.text(alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E, exclude_characters="/?#@:%"),
                min_size=1, max_size=20)


@given(user=_cred, password=_cred, signed_with=st.sampled_from([FORM_PLAIN, FORM_CREDENTIALS]))
def test_property_a_signature_matches_exactly_the_form_it_was_computed_over(user, password, signed_with) -> None:
    """For any credentials, a signature over a form validates as that form and as no other, and
    other credentials never validate the credentials form."""
    url = with_credentials(BARE, user, password) if signed_with == FORM_CREDENTIALS else BARE
    assert url is not None
    sig = sign(url, PARAMS)
    assert signed_url_form(BARE, PARAMS, sig, TOKEN, credentials=(user, password)) == signed_with
    if signed_with == FORM_CREDENTIALS:
        assert signed_url_form(BARE, PARAMS, sig, TOKEN, credentials=(user, password + "x")) is None
        assert signed_url_form(BARE, PARAMS, sig, TOKEN) is None
