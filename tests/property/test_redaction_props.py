"""No segment, call-state or task-attribute payload carries a factor value.

Generated factor values and payloads: the guard rejects every leak (raw, spaced,
hyphenated, any case, at any depth, or behind a denied key) and accepts every
clean payload. A leak is matched on word boundaries and only in values,
so the generated leaks sit on word boundaries.

AgentCore hosting (the bridge and log-redaction part; the signer's own part is in
``test_relay_signer_props.py``): the pre-signed ``wss://`` URL shape. For any sequence of mint,
failure, connect and reject events, and any way a library or our code might log the URL (message,
argument, bytes, ``extra`` field, nested field, path and query, the query alone), the JSON lines the
handler writes contain none of the URL's ``X-Amz-Signature``, ``X-Amz-Credential`` (or its access
key), ``X-Amz-Security-Token`` values, and no relay nonce (``n``). Benign text is kept as it was.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from starlette.websockets import WebSocketDisconnect

from lambdas.frontdoor.relay import PresignError, RuntimeUrlSigner, SigningCredentials
from owlbank import callstate as cs
from owlbank import obs, ws_nonce
from owlbank import segments as sg
from owlbank.security_log import auth_failed, strip_queries
from owlbank.store import MemoryStore
from owlbank.task_attributes import build_task_attributes

SID = "CA" + "e" * 32
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'"

factor_values = st.one_of(
    st.text("0123456789", min_size=4, max_size=9),  # zip, ssn4, otp, ...
    st.text("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ", min_size=6, max_size=10)
    .filter(lambda s: any(c.isdigit() for c in s)),  # alphanumeric codes
)

safe_keys = st.text("abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12).filter(
    lambda k: not sg.is_denied_key(k) and k not in sg.ID_KEYS
    and k not in {"type", "call_sid", "seq", "ts", "expires_at"})

clean_scalars = st.one_of(st.text(LETTERS, max_size=40), st.booleans(), st.none(),
                          st.integers(-10**6, 10**6))
clean_payloads = st.recursive(
    clean_scalars,
    lambda inner: st.one_of(st.lists(inner, max_size=4),
                            st.dictionaries(safe_keys, inner, max_size=4)),
    max_leaves=12,
)
event_types = st.sampled_from(sorted(sg.EVENT_TYPES))
separators = st.sampled_from(["", " ", "-", "  ", " - "])


def _no_leak(payload: Any, factors: list[str]) -> bool:
    norm = sg._norm_text(json.dumps(payload, ensure_ascii=False))
    return not any(sg._norm_text(f) in norm for f in factors)


#: What may stand between a leak and the text around it: the guard matches on word boundaries.
boundaries = st.sampled_from([" ", ".", ",", "!", "?", "'", " - ", "_"])


@st.composite
def disguised(draw: st.DrawFn, value: str) -> str:
    sep = draw(separators)
    body = sep.join(value) if draw(st.booleans()) else value
    if draw(st.booleans()):
        body = body.lower()
    pre, post = draw(st.text(LETTERS, max_size=10)), draw(st.text(LETTERS, max_size=10))
    if pre and pre[-1].isalnum():
        pre += draw(boundaries)
    if post and post[0].isalnum():
        post = draw(boundaries) + post
    return pre + body + post


def _insert(data: st.DataObject, payload: Any, key: str, leak: Any) -> Any:
    """Put ``leak`` under ``key`` in a randomly chosen dict inside ``payload``."""
    dicts: list[dict] = []

    def collect(v: Any) -> None:
        if isinstance(v, dict):
            dicts.append(v)
            for x in v.values():
                collect(x)
        elif isinstance(v, list):
            for x in v:
                collect(x)

    collect(payload)
    target = data.draw(st.sampled_from(dicts)) if dicts else payload
    if data.draw(st.booleans()):
        target[key] = leak
    else:
        target[key] = [leak]
    return payload


@pytest.fixture(autouse=True)
def _clear() -> None:
    sg.clear_sensitive(SID)
    yield
    sg.clear_sensitive(SID)


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(factors=st.lists(factor_values, min_size=1, max_size=3), body=clean_payloads,
       etype=event_types)
def test_clean_payloads_accepted(factors: list[str], body: Any, etype: str) -> None:
    sg.clear_sensitive(SID)
    event = {"type": etype, "body": body}
    assume(_no_leak(event, factors))
    sg.register_sensitive(SID, factors)
    log = sg.SegmentLog(MemoryStore(), now=lambda: 1)
    log.append(SID, event)
    assert log.read(SID)[0]["body"] == body


@settings(max_examples=300, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data(), factors=st.lists(factor_values, min_size=1, max_size=3),
       body=st.dictionaries(safe_keys, clean_payloads, max_size=4), key=safe_keys,
       etype=event_types)
def test_leaked_factor_rejected_anywhere(data: st.DataObject, factors: list[str], body: dict,
                                         key: str, etype: str) -> None:
    sg.clear_sensitive(SID)
    sg.register_sensitive(SID, factors)
    factor = data.draw(st.sampled_from(factors))
    leak: Any = data.draw(disguised(factor))
    if factor.isdigit() and factor[0] != "0" and data.draw(st.booleans()):
        leak = int(factor)
    event = {"type": etype, **_insert(data, body, key, leak)}
    store = MemoryStore()
    with pytest.raises(sg.RedactionError) as err:
        sg.SegmentLog(store, now=lambda: 1).append(SID, event)
    assert factor not in str(err.value)
    assert store.query("segments", {"call_sid": SID}) == []


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(factor=factor_values, glue=st.text("abcxyzABCXYZ0123456789", min_size=1, max_size=3),
       before=st.booleans(), etype=event_types)
def test_values_match_on_word_boundaries_and_keys_never(factor: str, glue: str, before: bool,
                                                            etype: str) -> None:
    """A registered value glued to a letter or digit isn't a match (``lang`` in ``language``),
    except that an all-digit value glued to a letter still is (``acct1234``);
    a key holding the value never blocks a write; the same value as a separate word is caught."""
    sg.clear_sensitive(SID)
    sg.register_sensitive(SID, [factor])
    glued = glue + factor if before else factor + glue
    touching = glue[-1] if before else glue[0]
    still_matches = factor.isdigit() and not touching.isdigit()
    assert sg.contains_sensitive(SID, glued) is still_matches
    log = sg.SegmentLog(MemoryStore(), now=lambda: 1)
    if still_matches:
        assert factor not in sg.redact_text(SID, glued)
        with pytest.raises(sg.RedactionError):
            log.append(SID, {"type": etype, "body": {"text": glued}})
    else:
        assert sg.redact_text(SID, glued) == glued
        log.append(SID, {"type": etype, "body": {f"k{factor}": "ok", "text": glued}})
    spaced = f"{glued} {factor}." if before else f"{factor}, {glued}"
    assert sg.contains_sensitive(SID, spaced)
    with pytest.raises(sg.RedactionError):
        log.append(SID, {"type": etype, "body": {"text": spaced}})


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data(), body=st.dictionaries(safe_keys, clean_payloads, max_size=4),
       denied=st.sampled_from(sg.DENY_KEYS), suffix=st.sampled_from(["", "4", "_code", "Value"]),
       upper=st.booleans(), etype=event_types)
def test_denied_keys_rejected_at_any_depth(data: st.DataObject, body: dict, denied: str,
                                           suffix: str, upper: bool, etype: str) -> None:
    key = (denied.upper() if upper else denied) + suffix
    event = {"type": etype, **_insert(data, body, key, "x")}
    with pytest.raises(sg.RedactionError):
        sg.SegmentLog(MemoryStore()).append(SID, event)


@settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data(), factors=st.lists(factor_values, min_size=1, max_size=3),
       args=st.dictionaries(safe_keys, clean_payloads, max_size=3), key=safe_keys,
       leak_it=st.booleans())
def test_call_state_guard(data: st.DataObject, factors: list[str], args: dict, key: str,
                          leak_it: bool) -> None:
    sg.clear_sensitive(SID)
    assume(_no_leak(args, factors))
    sg.register_sensitive(SID, factors)
    store = MemoryStore()
    v1 = cs.create(store, cs.CallState(call_sid=SID, expires_at=10**9))
    if leak_it:
        args = _insert(data, args, key, data.draw(disguised(data.draw(st.sampled_from(factors)))))
        with pytest.raises(sg.RedactionError):
            cs.update(store, v1, now=lambda: 1, pending_tool=cs.PendingTool("t", "x", args))
        assert cs.load(store, SID).version == 1
    else:
        v2 = cs.update(store, v1, now=lambda: 1, pending_tool=cs.PendingTool("t", "x", args))
        assert cs.load(store, SID) == v2


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data(), factors=st.lists(factor_values, min_size=1, max_size=3),
       pre=st.text(max_size=300), post=st.text(max_size=300))
def test_task_attributes_never_carry_factor(data: st.DataObject, factors: list[str],
                                            pre: str, post: str) -> None:
    sg.clear_sensitive(SID)
    sg.register_sensitive(SID, factors)
    # The leak sits on word boundaries; pre and post hold no factor of their own.
    assume(_no_leak([pre, post], factors))
    summary = pre + " " + data.draw(disguised(data.draw(st.sampled_from(factors)))) + " " + post
    attrs = build_task_attributes(cs.CallState(call_sid=SID, expires_at=1, level=1), {},
                                  "complaint", summary, now=1)
    scanned = {k: v for k, v in attrs.items() if k not in sg.ID_KEYS}
    assert _no_leak(scanned, factors)


# --------------------------------------------------------------------------- The pre-signed URL shape

REGION = "us-east-1"
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/owlbank_dev_bridge-AbC123xyz0"
CALL = "CA" + "1" * 32
WORDS = "abcdefghijklmnopqrstuvwxyz"
#: Loggers a URL could reach: libraries (the record factory and filters), ours, and an unknown one.
LEAK_LOGGERS = ("tac.server.signature_validation", "uvicorn.access", "uvicorn.error", "starlette",
                "fastapi", "tac_bridge.app", "lambdas.frontdoor.app", "some.library")
LEAK_FORMS = ("arg", "inline", "bytes", "extra", "emit", "nested", "path", "query", "sentence")

akids = st.text("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", min_size=16, max_size=16).map(lambda s: "ASIA" + s)
aws_secrets = st.binary(min_size=30, max_size=30).map(lambda b: base64.b64encode(b).decode())
session_tokens = st.binary(min_size=96, max_size=600).map(lambda b: base64.b64encode(b).decode())
relay_nonces = st.binary(min_size=24, max_size=24).map(
    lambda b: base64.urlsafe_b64encode(b).decode().rstrip("="))  # secrets.token_urlsafe(24)
benign = st.lists(st.text(WORDS, min_size=1, max_size=10), min_size=1, max_size=8).map(" ".join)


def _signer(akid: str, secret: str, token: str) -> RuntimeUrlSigner:
    creds = SigningCredentials(access_key=akid, secret_key=secret, token=token, expires_at=2.0e9 + 7200)
    return RuntimeUrlSigner(runtime_arn=RUNTIME_ARN, region=REGION, clock=lambda: 2.0e9,
                            credential_provider=lambda: creds, ttl_s=300)


def _failing_provider(token: str, akid: str) -> Any:
    def provide() -> SigningCredentials:
        raise ConnectionError(f"no route {token} {akid}")  # an error message carrying secrets
    return provide


def _url_secrets(url: str) -> set[str]:
    q = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    raw = {q["X-Amz-Signature"], q["X-Amz-Credential"], q["X-Amz-Credential"].split("/")[0],
           q["X-Amz-Security-Token"]}
    return raw | {quote(v, safe="") for v in raw}


def _leak(form: str, logger: logging.Logger, url: str, words: str) -> None:
    """Log ``url`` the way ``form`` says (what a library or a careless line might do). At WARNING:
    TAC sets its own loggers to WARNING, and the line must be written to be checked."""
    path_query = url.split("://", 1)[1].split("/", 1)[1]
    query = url.split("?", 1)[1]
    if form == "arg":
        logger.warning("connecting to %s", url)
    elif form == "inline":
        logger.warning(f"validation failed. URL used: {url}")  # noqa: G004 - the shape under test
    elif form == "bytes":
        logger.warning("raw %s", url.encode())
    elif form == "extra":
        logger.warning("upgrade", extra={"target": url})
    elif form == "emit":
        obs.emit(logger, logging.WARNING, "relay_event", url=url, note=words)
    elif form == "nested":
        obs.emit(logger, logging.WARNING, "relay_event", detail={"request": {"url": url}, "list": [url]})
    elif form == "path":
        logger.warning('"GET /%s HTTP/1.1" 101', path_query)
    elif form == "query":
        obs.emit(logger, logging.WARNING, "relay_event", query=query)
    else:
        logger.warning("%s then %s and %s", words, url, words)


@pytest.fixture
def plain_agent() -> Iterator[None]:
    """The rigs' plain scripted agent (no Strands client): these tests never reach a turn."""
    from tests.strands_parity import reset

    reset(plain=True)
    yield
    reset()


class Output:
    """Everything any handler writes: our JSON handler on the root logger, and every other plain
    stream handler a library installed on its own logger (TAC's ``tac`` handler, which doesn't
    propagate). Streams are put back after the test."""

    def __init__(self) -> None:
        self.buf = io.StringIO()
        self._moved: list[tuple[logging.StreamHandler[Any], Any]] = []

    def start(self) -> None:
        """Before each example: an empty buffer, and every handler pointed at it."""
        self.buf.seek(0)
        self.buf.truncate()
        obs.configure_logging("bridge", level=logging.DEBUG, stream=self.buf)
        self.redirect()

    def redirect(self) -> None:
        """Point library handlers (made since the last call too) at the buffer."""
        loggers = [lg for lg in logging.root.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
        for lg in loggers:
            for h in lg.handlers:
                if type(h) is logging.StreamHandler and h.stream is not self.buf:
                    self._moved.append((h, h.stream))
                    h.setStream(self.buf)

    def text(self) -> str:
        return self.buf.getvalue()

    def restore(self) -> None:
        for h, stream in reversed(self._moved):
            h.setStream(stream)


@pytest.fixture
def output() -> Iterator[Output]:
    out = Output()
    yield out
    out.restore()


def _agentcore_client() -> tuple[TestClient, Any]:
    from tac_bridge.app import build_app
    from tests.component.test_bridge_fakes import make_rig, write_call_state

    rig = make_rig()
    write_call_state(rig.store, CALL)
    return TestClient(build_app(rig.tac, rig.voice, rig.bridge, hosting="agentcore")), rig


def _socket(client: TestClient, session_id: str, params: dict[str, Any]) -> None:
    """One relay socket on the ``agentcore`` app: ``setup`` with ``params``, a key press, close."""
    from tac_bridge.hosting import SESSION_ID_HEADER

    try:
        with client.websocket_connect("/ws", headers={SESSION_ID_HEADER: session_id}) as ws:
            ws.send_json({"type": "setup", "callSid": CALL, "from": "+12025550101", "to": "+12025550199",
                          "customParameters": {"callSid": CALL, **params}})
            ws.send_json({"type": "dtmf", "digit": "1"})
    except WebSocketDisconnect:
        pass


event_kinds = st.sampled_from(("mint", "failure", "connect", "reject_nonce", "reject_binding", "leak"))
event_steps = st.lists(st.tuples(event_kinds, st.sampled_from(LEAK_LOGGERS), st.sampled_from(LEAK_FORMS),
                                 relay_nonces, benign), min_size=1, max_size=8)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(akid=akids, secret=aws_secrets, token=session_tokens, events=event_steps)
def test_no_presigned_url_secret_or_nonce_in_any_log_line(plain_agent: None, output: Output, akid: str,
                                                               secret: str, token: str, events: list[Any]) -> None:
    output.start()
    signer = _signer(akid, secret, token)
    failing = RuntimeUrlSigner(runtime_arn=RUNTIME_ARN, region=REGION, clock=lambda: 2.0e9,
                               credential_provider=_failing_provider(token, akid))
    client, rig = _agentcore_client()
    output.redirect()  # TAC's own handler, made by the rig
    forbidden: set[str] = {secret, token, quote(token, safe=""), akid}
    url = signer.mint(call_sid=CALL, nonce="unused")
    forbidden |= _url_secrets(url)
    with client:
        for kind, logger_name, form, nonce, words in events:
            forbidden.add(nonce)
            logger = logging.getLogger(logger_name)
            if kind == "mint":
                url = signer.mint(call_sid=CALL, nonce=nonce)
                forbidden |= _url_secrets(url)
                _leak(form, logger, url, words)
            elif kind == "failure":
                with pytest.raises(PresignError):
                    failing.mint(call_sid=CALL, nonce=nonce)
                obs.emit(logger, logging.WARNING, "relay_presign_failed", error_class="ConnectionError")
            elif kind == "connect":  # the front door's fresh nonce, then a socket carrying it
                cs.mutate(rig.store, CALL, lambda _s, n=nonce: {"ws_nonce_hash": ws_nonce.nonce_hash(n)},
                          now=lambda: 1_800_000_000)
                _socket(client, CALL, {"n": nonce})
            elif kind == "reject_nonce":
                _socket(client, CALL, {"n": nonce})  # not the stored nonce (or already used)
            elif kind == "reject_binding":
                _socket(client, "CA" + "2" * 32, {"n": nonce})
            else:
                _leak(form, logger, url, words)
                auth_failed("bridge", "nonce")
    out = output.text()
    assert out, "nothing was captured"
    assert sorted(s for s in forbidden if s and s in out) == []


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(akid=akids, secret=aws_secrets, token=session_tokens, nonce=relay_nonces,
       form=st.sampled_from(LEAK_FORMS), logger_name=st.sampled_from(LEAK_LOGGERS), words=benign)
def test_presigned_shape_is_scrubbed_from_any_field_and_benign_text_kept(
        output: Output, akid: str, secret: str, token: str, nonce: str, form: str, logger_name: str,
        words: str) -> None:
    """One URL (with an ``n``, as a misbuilt URL might carry one) logged one way: no secret of it
    survives, and the benign words logged with it are still there."""
    output.start()
    url = _signer(akid, secret, token).mint(call_sid=CALL, nonce=nonce) + f"&n={nonce}"
    _leak(form, logging.getLogger(logger_name), url, words)
    out = output.text()
    assert out
    assert sorted(s for s in _url_secrets(url) | {nonce, akid} if s in out) == []
    if form in ("emit", "sentence"):
        assert words in out


@settings(max_examples=200, deadline=None)
@given(text=benign)
def test_benign_text_passes_redaction_unchanged(text: str) -> None:
    assert strip_queries(text) == text
    assert obs.scrub(text) == text
    host_path = f"wss://bedrock-agentcore.{REGION}.amazonaws.com/runtimes/{text.replace(' ', '-')}/ws"
    assert strip_queries(host_path) == host_path  # no query: nothing to strip
    assert strip_queries(host_path + "?X-Amz-Signature=abc&n=def") == host_path + "?[redacted]"
