"""The log scrubber. For any text containing a phone number, a run of six or more
digits, a token-like string, an Authorization/Basic value or a URL query, none of those survives;
and scrubbing twice equals scrubbing once (idempotence), for any text at all."""

from __future__ import annotations

import base64
import io
import json
import logging
import string
import time
from urllib.parse import urlencode

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from owlbank import obs, segments
from owlbank.obs import scrub

# Surrounding prose: no '=', '&', '?', '+', '/', ':' or digits, so a surviving substring can only
# come from the generated sensitive value, never from the filler.
FILLER = st.text(alphabet=string.ascii_letters + " .,;!'()", max_size=30)
ANY_TEXT = st.text(max_size=300)

phones = st.from_regex(r"\+[1-9][0-9]{7,14}", fullmatch=True)
digit_runs = st.from_regex(r"[0-9]{6,24}", fullmatch=True)
b64_tokens = st.binary(min_size=18, max_size=48).map(
    lambda b: base64.urlsafe_b64encode(b).decode().rstrip("="))
hex_tokens = st.binary(min_size=10, max_size=40).map(lambda b: b.hex())
tokens = st.one_of(b64_tokens, hex_tokens)
basic_values = st.tuples(st.text(string.ascii_letters + string.digits, min_size=1, max_size=12),
                         st.text(string.ascii_letters + string.digits, min_size=1, max_size=16)).map(
    lambda up: base64.b64encode(f"{up[0]}:{up[1]}".encode()).decode())
queries = st.dictionaries(st.text(string.ascii_lowercase, min_size=1, max_size=6),
                          st.text(string.ascii_letters + string.digits, min_size=1, max_size=20),
                          min_size=1, max_size=4).map(urlencode)

CASES = settings(max_examples=200, deadline=None)


def around(filler_a: str, value: str, filler_b: str) -> str:
    return f"{filler_a} {value} {filler_b}"


@CASES
@given(FILLER, phones, FILLER)
def test_no_phone_number_survives(a: str, phone: str, b: str) -> None:
    out = scrub(around(a, phone, b))
    assert phone not in out
    assert phone[-4:] in out and "+*" in out  # the last four are kept (+*******1234)


@CASES
@given(FILLER, digit_runs, FILLER)
def test_no_run_of_six_digits_survives(a: str, digits: str, b: str) -> None:
    out = scrub(around(a, digits, b))
    assert digits not in out
    assert not any(len(run) >= 6 for run in "".join(c if c.isdigit() else " " for c in out).split())


@CASES
@given(FILLER, tokens, FILLER)
def test_no_token_like_string_survives(a: str, token: str, b: str) -> None:
    assert token not in scrub(around(a, token, b))


@CASES
@given(FILLER, basic_values, FILLER)
def test_no_basic_header_value_survives(a: str, value: str, b: str) -> None:
    for text in (f"Authorization: Basic {value}", f"authorization={value}", f'{{"Authorization": "Basic {value}"}}'):
        assert value not in scrub(around(a, text, b))


@CASES
@given(FILLER, queries, FILLER, st.sampled_from(["https://relay.example.com/ws", "wss://h.example/ws/x", "/ws", "GET /a/b"]))
def test_no_url_query_survives(a: str, query: str, b: str, base: str) -> None:
    out = scrub(around(a, f"{base}?{query}", b))
    assert query not in out
    assert "?[redacted]" in out


@CASES
@given(FILLER, phones, FILLER, digit_runs, FILLER, tokens, FILLER, basic_values, FILLER, queries)
def test_all_classes_together_and_idempotence(a, phone, b, digits, c, token, d, basic, e, query) -> None:
    text = f"{a} {phone} {b} {digits} {c} {token} {d} Authorization: Basic {basic} {e} https://x.example/p?{query}"
    once = scrub(text)
    for secret in (phone, digits, token, basic, query):
        assert secret not in once
    assert scrub(once) == once


@settings(max_examples=400, deadline=None)
@given(ANY_TEXT)
def test_scrubbing_is_idempotent_and_deterministic_for_any_text(text: str) -> None:
    once = scrub(text)
    assert scrub(once) == once
    assert scrub(text) == once


@settings(max_examples=200, deadline=None)
@given(st.lists(st.sampled_from(["+", "?", "=", "&", "/", ":", "-", "_", "0", "1", "9", "a", "Z", " ",
                                 "Basic ", "Authorization:", "https://", "[redacted]", "***", "CA"]),
                max_size=60).map("".join))
def test_idempotent_on_adversarial_fragments(text: str) -> None:
    once = scrub(text)
    assert scrub(once) == once


def test_twilio_sids_uuids_and_identifiers_are_kept() -> None:
    sid = "CA" + "0123456789abcdef" * 2
    text = (f"call {sid} request 8f507cfc-1234-4697-b07a-ac58fc914c95 code AUTH_LEVEL_INSUFFICIENT "
            "error ProvisionedThroughputExceededException event otp_sent_untrusted_ani")
    assert scrub(text) == text


def test_degenerate_tokens_are_masked() -> None:
    for token in ("A" * 24, "a" * 30, "0" * 24, "Zq9secretNonceValue_abcdefgh123456"):
        assert token not in scrub(f"x {token} y")


def test_scrubbing_is_fast() -> None:
    text = ("GET /health 200 +12025550123 call CA" + "ab" * 16 + " done in 12 ms " * 3 + "\n") * 2000  # ~200 KB
    started = time.perf_counter()
    scrub(text)
    assert time.perf_counter() - started < 1.0


# --------------------------------------------------------------------------- The exact-value step

_SID = "CA" + "0123456789abcdef" * 2
_VALUES = st.text(alphabet=string.ascii_letters + string.digits, min_size=4, max_size=12)
_SITES = st.sampled_from(["template", "arg", "extra", "nested", "exception"])
_SEPS = st.sampled_from(["", "", " ", "-", " - ", "  ", "--"])


@st.composite
def _spoken(draw: st.DrawFn, value: str) -> str:
    """``value`` with arbitrary case per character and spaces or hyphens between characters."""
    chars = [c.upper() if draw(st.booleans()) else c.lower() for c in value]
    return "".join(c + (draw(_SEPS) if i < len(chars) - 1 else "") for i, c in enumerate(chars))


def _log_once(site: str, text: str) -> dict:
    """Log ``text`` at ``site`` through ScrubFilter + JsonFormatter; the JSON line without ``ts``."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(obs.JsonFormatter())
    handler.addFilter(obs.ScrubFilter())
    log = logging.getLogger("props.cpm12")
    log.propagate = False
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    try:
        if site == "template":
            log.warning("heard " + text.replace("%", "%%"))
        elif site == "arg":
            log.warning("heard %s", text)
        elif site == "extra":
            log.warning("heard", extra={"field": text})
        elif site == "nested":
            log.warning("heard", extra={"outer": {"inner": [text, {"k": text}]}})
        else:
            try:
                raise ValueError(text)
            except ValueError:
                log.warning("heard", exc_info=True)
    finally:
        log.handlers.clear()
    line = json.loads(stream.getvalue())
    line.pop("ts", None)
    return line


@settings(max_examples=300, deadline=None)
@given(value=_VALUES, a=FILLER, b=FILLER, site=_SITES, data=st.data())
def test_logger_half_no_registered_value_survives(value: str, a: str, b: str, site: str,
                                                         data: st.DataObject) -> None:
    """Logger half: a registered value, embedded with any case, spaces and hyphens in any
    part of a log call, is not in the formatted line under ``redact_text``'s normalization."""
    norm = segments._norm_text(value)
    spoken = data.draw(_spoken(value))
    segments.register_sensitive(_SID, [value])
    try:
        with obs.call_context(_SID):
            line = _log_once(site, f"{a} {spoken} {b}")
            skeleton = _log_once(site, f"{a}  {b}")
    finally:
        segments.clear_sensitive(_SID)
    # A value that the line's fixed parts already hold ("WARNING", the SID, "[redacted]") isn't a leak.
    assume(norm not in segments._norm_text(json.dumps(skeleton, ensure_ascii=False)))
    assert norm not in segments._norm_text(json.dumps(line, ensure_ascii=False))


@settings(max_examples=200, deadline=None)
@given(value=_VALUES, text=st.text(max_size=80), site=_SITES)
def test_cpm13b_logger_half_the_callers_text_is_unchanged(value: str, text: str, site: str) -> None:
    """Logger half: logging never changes the caller's objects; only the record is scrubbed."""
    original = f"{text} {value} {text}"
    payload = {"inner": [original]}
    snapshot = (original, json.dumps(payload))
    segments.register_sensitive(_SID, [value])
    try:
        with obs.call_context(_SID):
            if site == "nested":
                _log_once("extra", payload)  # type: ignore[arg-type]
            else:
                _log_once(site, original)
    finally:
        segments.clear_sensitive(_SID)
    assert (original, json.dumps(payload)) == snapshot
