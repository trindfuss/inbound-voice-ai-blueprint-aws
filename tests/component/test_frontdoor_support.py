"""Shared harness for the front-door component tests (signed fake Twilio webhooks).

Every test that uses :func:`fd` also gets the no-leak boundary check at teardown: no
response body, log record or stored item may contain any factor value the test
sent (or any fixture factor value).
"""

from __future__ import annotations

import base64
import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode

import pytest
from twilio.request_validator import RequestValidator

from owlbank.basic_auth import BasicSecret

from lambdas.auth_service import service as auth_service
from lambdas.auth_service.verify_client import VerifyUnavailable
from lambdas.bank_tools import service as bank_service
from lambdas.frontdoor import state as sessions
from lambdas.frontdoor.app import Deps, FrontDoor, FrontDoorConfig
from lambdas.frontdoor.backend import LambdaBackendClient
from lambdas.frontdoor.fakes import FakeLambdaClient
from owlbank import callstate
from owlbank.callstate import PendingTool
from owlbank.catalog import PromptCatalog
from owlbank.envelope import ResultEnvelope
from owlbank.pii import FakePiiRedactor, PiiGuard
from owlbank.policy import Policy
from owlbank.segments import ID_KEYS, clear_sensitive
from owlbank.slots import FakeResolver
from owlbank.store import MemoryStore
from owlbank.token_seal import FakeSealer
from tests.leakscan import compile_values, leaks

DATA = Path(__file__).resolve().parents[2] / "data"
AUTH_TOKEN = "test-auth-token-0123456789abcdef"
BASIC = ("twilio-hook", "s3cret-pass-word")
HOST = "voice.owlbank.example"
STAGE = "/prod"
WORKFLOW = "WW" + "a" * 32
#: The ``<Enqueue>`` ``action``: the front door's own route (no ``waitUrl``).
ENQUEUE_ACTION_URL = f"{STAGE}/voice/enqueue-action"
CLOCK = 1_760_000_000.0

THOMAS = "+12025550101"
LEES = "+12025550102"
MARIA = "+12025550103"
NOBODY = "+12025550199"

TEST_FACTORS: dict[str, dict[str, str]] = json.loads(
    (DATA / "fixtures.test-factors.json").read_text())["customers"]
OTP = "246813"
#: ``verify.failed_ai`` (en-US): the welcome after a failed or declined login.
FAILED_AI = ("I'm sorry, I couldn't verify your identity. I can still help with general questions, "
             "like our opening hours, or connect you with a specialist.")

# Server-generated identifiers and times are not caller input (same rule as
# owlbank.segments.ID_KEYS); random digits in them could collide with a factor.
_NS_SKIP_KEYS = set(ID_KEYS) | {"token_hash", "claimed_at", "ended_at", "at", "issued_at",
                                 "started_ms"}
_STD_LOG_ATTRS = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}


def server_noise() -> str:
    """Digits the server itself emits at the fixed test clock (timestamps, TTLs)."""
    from datetime import UTC, datetime
    t = int(CLOCK)
    iso = datetime.fromtimestamp(t, tz=UTC).strftime("%Y%m%d%H%M%S")
    return " ".join(str(x) for x in (t, t + 86400, t + 1800, iso))


def all_fixture_factor_values() -> set[str]:
    return {v for f in TEST_FACTORS.values() for v in f.values()} | {OTP}


def sign(url: str, params: dict[str, str], token: str = AUTH_TOKEN) -> str:
    return RequestValidator(token).compute_signature(url, params)


def basic_header(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


#: ``make_event(auth=NO_AUTH)``: leave the ``Authorization`` header out, as Twilio's first request does.
NO_AUTH = ""
#: Signed-URL forms: without credentials, or with ``user:password@`` in the URL.
URL_FORMS = ("plain", "credentials")


def signed_url(path: str, form: str = "plain") -> str:
    """The URL Twilio signs: the public URL, with ``BASIC``'s credentials inserted for ``credentials``."""
    assert form in URL_FORMS, form
    userinfo = f"{BASIC[0]}:{BASIC[1]}@" if form == "credentials" else ""
    return f"https://{userinfo}{HOST}{path}"


def make_event(route: str, params: dict[str, str], *, signature: str | None = None,
               auth: str | None = None, url_params: dict[str, str] | None = None,
               form: str = "plain") -> dict[str, Any]:
    """An API Gateway proxy event for a Twilio request. ``auth=None`` carries the right Basic
    credentials (the repeat after the challenge), :data:`NO_AUTH` none (the first request), any
    other string as the header. ``form``: which URL the signature is computed over. ``route`` may
    carry a query (``factor?g=...``); the signature covers it, as Twilio's does."""
    route, _, qs = route.partition("?")
    query = dict(parse_qsl(qs)) if qs else None
    path = f"{STAGE}/voice/{route}"
    tail = f"?{urlencode(query)}" if query else ""
    sig = signature if signature is not None else sign(signed_url(path, form) + tail,
                                                       url_params or params)
    headers = {
        "Host": HOST, "X-Forwarded-Proto": "https", "X-Forwarded-Port": "443",
        "Content-Type": "application/x-www-form-urlencoded",
        "X-Twilio-Signature": sig,
    }
    if auth != NO_AUTH:
        headers["Authorization"] = auth if auth is not None else basic_header(*BASIC)
    return {
        "resource": f"/voice/{route}",
        "path": f"/voice/{route}",
        "httpMethod": "POST",
        "headers": headers,
        "multiValueHeaders": None,
        "queryStringParameters": query,
        "multiValueQueryStringParameters": {k: [v] for k, v in query.items()} if query else None,
        "requestContext": {"path": path, "stage": "prod", "domainName": HOST},
        "body": urlencode(params),
        "isBase64Encoded": False,
    }


def twilio_post(handle: Callable[[dict[str, Any]], dict[str, Any]], route: str, params: dict[str, str],
                *, form: str = "plain") -> dict[str, Any]:
    """One Twilio webhook the way Twilio sends it to a password-protected URL: first with no
    ``Authorization`` header (answered 401 with ``WWW-Authenticate: Basic``, nothing else run),
    then the same request again with ``Authorization: Basic`` and the signature over ``form``'s URL.
    Returns the second response."""
    first = handle(make_event(route, params, auth=NO_AUTH, form=form))
    assert first == {"statusCode": 401, "headers": {"WWW-Authenticate": 'Basic realm="owlbank"'},
                     "body": ""}, first
    return handle(make_event(route, params, form=form))


class FakeVerify:
    """Twilio Verify stand-in for the real auth service: accepts :data:`OTP` at the number it sent to."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.locales: list[str | None] = []
        self.fail = False  # Verify can't deliver (raises like TwilioVerifyClient)

    def send(self, to: str, channel: str, locale: str | None = None) -> None:
        if self.fail:
            raise VerifyUnavailable("verify send failed")
        self.sent.append((to, channel))
        self.locales.append(locale)

    def check(self, to: str, code: str) -> bool:
        return code == OTP and any(t == to for t, _ in self.sent)


class RecordingCloser:
    def __init__(self, fail: bool = False) -> None:
        self.closed: list[tuple[str, str | None]] = []
        self.transcriptions: list[tuple[str, str | None, str]] = []
        self.legs: list[str | None] = []
        self.fail = fail

    def close(self, *, call_sid: str, conversation_id: str | None) -> None:
        if self.fail:
            raise RuntimeError("orchestrator down")
        self.closed.append((call_sid, conversation_id))

    def transcription_ended(self, *, call_sid: str, conversation_id: str | None, event: str,
                            leg: str | None = None) -> None:
        if self.fail:
            raise RuntimeError("orchestrator down")
        self.transcriptions.append((call_sid, conversation_id, event))
        self.legs.append(leg)


@dataclass
class Harness:
    store: MemoryStore
    app: FrontDoor
    lam: FakeLambdaClient
    verify: FakeVerify
    sealer: FakeSealer
    resolver: FakeResolver
    closer: RecordingCloser
    bodies: list[str] = field(default_factory=list)
    sent_factors: set[str] = field(default_factory=set)
    form: str = "plain"  # which URL form the fake Twilio signs (URL_FORMS)
    #: Every ``Deps.sleep`` the front door asked for (seconds); nothing really waits.
    sleeps: list[float] = field(default_factory=list)

    def post(self, route: str, call_sid: str, **params: str) -> dict[str, Any]:
        p = {"CallSid": call_sid, "AccountSid": "AC" + "b" * 32, **params}
        resp = twilio_post(self.app.handle, route, p, form=self.form)
        self.bodies.append(resp["body"])
        return resp

    def xml(self, route: str, call_sid: str, *, follow: bool = True, **params: str) -> ET.Element:
        """The TwiML for one webhook. ``follow`` (the default) does what Twilio does with a
        response that ends in ``<Redirect>`` and has nothing that waits for the caller (the
        "Got it." while the code is sent): it POSTs the redirect URL with the call's parameters
        only, and returns one ``<Response>`` with what the caller hears, in order: the verbs
        before the ``<Redirect>``, then the redirected response."""
        resp = self.post(route, call_sid, **params)
        assert resp["statusCode"] == 200, resp
        root = ET.fromstring(resp["body"].encode())
        if not follow:
            return root
        for _ in range(5):
            redirect = redirect_target(root)
            if redirect is None:
                return root
            nxt = self.post(redirect, call_sid)
            assert nxt["statusCode"] == 200, nxt
            merged = ET.Element(root.tag, root.attrib)
            merged.extend([c for c in root if c.tag != "Redirect"])
            merged.extend(list(ET.fromstring(nxt["body"].encode())))
            root = merged
        raise AssertionError("redirect loop")

    def factor(self, call_sid: str, value: str, *, follow: bool = True, **extra: str) -> ET.Element:
        self.sent_factors.add(value)
        return self.xml("factor", call_sid, follow=follow, Digits=value, **extra)

    # -- journeys
    def start(self, call_sid: str, ani: str, *, lang_digit: str = "") -> ET.Element:
        self.xml("incoming", call_sid, From=ani, To="+12025550100", CallStatus="ringing",
                 StirVerstat="TN-Validation-Passed-A")
        return self.xml("language", call_sid, Digits=lang_digit)

    def login_thomas(self, call_sid: str) -> ET.Element:
        self.start(call_sid, THOMAS)
        return self.factor(call_sid, TEST_FACTORS["C-1001"]["zip"])

    def set_pending_tool(self, call_sid: str, name: str, args: dict[str, Any]) -> None:
        callstate.mutate(self.store, call_sid, lambda s: {
            "pending_tool": PendingTool(tool_use_id="tooluse_1", name=name, args=args)})

    def handoff(self, call_sid: str, data: dict[str, Any]) -> ET.Element:
        return self.xml("connect-action", call_sid, CallStatus="in-progress",
                        SessionStatus="ended", SessionId="VX" + "c" * 32,
                        SessionDuration="12", HandoffData=json.dumps(data))

    def state(self, call_sid: str) -> callstate.CallState:
        st = callstate.load(self.store, call_sid)
        assert st is not None
        return st

    def session(self, call_sid: str) -> sessions.Session:
        s = sessions.load(self.store, call_sid)
        assert s is not None
        return s

    def pending_result(self, call_sid: str) -> ResultEnvelope:
        raw = self.state(call_sid).pending_result
        assert raw is not None
        return ResultEnvelope.from_dict(dict(raw))

    def journal(self, call_sid: str) -> list[dict[str, Any]]:
        return self.store.query("segments", {"call_sid": call_sid})

    # -- no-leak check
    def nonsensitive_slots(self) -> frozenset[str]:
        """The slots whose value a ``slot_resolution`` event or a ``factor_answer`` line may carry
        the ones not flagged sensitive."""
        policy = self.app.d.policy
        return frozenset(n for n in policy.slots() if not policy.slot(n).sensitive)

    def stored_strings(self) -> list[str]:
        out: list[str] = []
        open_slots = self.nonsensitive_slots()

        def walk(v: Any) -> None:
            if isinstance(v, dict):
                # a non-sensitive slot's value is journaled in its slot_resolution event only.
                sanctioned = v.get("type") == "slot_resolution" and v.get("slot") in open_slots
                for k, x in v.items():
                    # A sealed field is ciphertext (the sealer's own tests show it never holds the
                    # plaintext); random base64 can hold any four digits between letters.
                    if str(k).startswith("sealed"):
                        continue
                    if k not in _NS_SKIP_KEYS and not (sanctioned and k == "value"):
                        out.append(str(k))
                        walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif v is not None:
                out.append(str(v))
        for table in ("calls", "segments", "tokens"):
            for item in self.store._tables[table].values():  # noqa: SLF001 - test introspection
                walk(item)
        return out


#: Verbs after which Twilio waits for the caller or leaves the call (a ``<Redirect>`` after them
#: is not reached at once: a ``<Gather>`` posts its action; the language menu relies on that).
_WAITING_VERBS = frozenset({"Gather", "Connect", "Enqueue", "Dial", "Hangup", "Pay", "Record"})


def redirect_target(root: ET.Element) -> str | None:
    """The ``/voice/<route>`` a response redirects to straight away, else ``None``."""
    tags = [c.tag for c in root]
    if "Redirect" not in tags or _WAITING_VERBS & set(tags[: tags.index("Redirect")]):
        return None
    url = root[tags.index("Redirect")].text or ""
    assert "/voice/" in url, url
    return url.rsplit("/voice/", 1)[1]


def assert_no_leak(h: Harness, records: list[logging.LogRecord], values: set[str]) -> None:
    """No factor value, as a whole value (:mod:`tests.leakscan`), in any response, log or
    stored string. A random server value that merely contains one (an 8-digit confirmation
    number holding a test SSN's last four) is not a leak. The one sanctioned place for a value is
    a non-sensitive slot's own ``value`` field in ``factor_answer`` and ``slot_resolution``
    ; a sensitive slot's never is. A non-sensitive slot's
    ``factor_answer.speech`` may hold the non-sensitive fixture values (the caller said
    their ZIP), and is still scanned for every other value."""
    patterns = compile_values(v for v in values if len(v) >= 4)
    open_slots = h.nonsensitive_slots() if hasattr(h, "nonsensitive_slots") else frozenset()
    open_values = {f[n] for f in TEST_FACTORS.values() for n in f if n in open_slots}
    speech_patterns = compile_values(v for v in values if len(v) >= 4 and v not in open_values)
    logs = []
    for r in records:
        logs.append(r.getMessage())
        sanctioned = getattr(r, "event", None) == "factor_answer" and getattr(r, "factor", None) in open_slots
        logs.extend(str(v) for k, v in r.__dict__.items()
                    if k not in _STD_LOG_ATTRS and k not in _NS_SKIP_KEYS
                    and not (sanctioned and k in ("value", "speech")))
        if sanctioned and isinstance(getattr(r, "speech", None), str):
            assert not leaks(speech_patterns, r.speech), "sensitive value leaked into speech"
    for where, texts in (("response", h.bodies), ("log", logs), ("store", h.stored_strings())):
        for t in texts:
            assert not leaks(patterns, t), f"factor value leaked into {where}"


def build(*, resolver: FakeResolver | None = None, config_kw: dict[str, Any] | None = None,
          closer: RecordingCloser | None = None, calls: Any = None, human_leg: Any = None,
          policy_edit: Callable[[dict[str, Any]], None] | None = None,
          catalog_edit: Callable[[dict[str, Any]], None] | None = None,
          flex: bool = True, briefing_model: Any = None) -> Harness:
    """Front door + the real auth service and bank tools, in process, over one MemoryStore.
    ``policy_edit`` changes the seed policy document (in place) before every service loads it;
    ``catalog_edit`` the seed prompt catalog before the front door loads it. ``flex`` (the
    default) configures the workflow SID, so an escalation enqueues; ``flex=False`` leaves it
    out, so an escalation reads the briefing (``briefing_model`` is its model, ``None`` for none)."""
    store = MemoryStore()
    for c in json.loads((DATA / "fixtures.json").read_text())["customers"]:
        store.put("customers", c)
    catalog_doc = json.loads((DATA / "prompt_catalog.json").read_text())
    if catalog_edit is not None:
        catalog_edit(catalog_doc)
    catalog = PromptCatalog.from_dict(catalog_doc)
    policy_doc = json.loads((DATA / "policy.json").read_text())
    if policy_edit is not None:
        policy_edit(policy_doc)
    policy = Policy.from_dict(policy_doc)
    verify = FakeVerify()
    auth_deps = auth_service.AuthDeps(store=store, policy=policy, verify_client=verify,
                                      clock=lambda: int(CLOCK))
    bank_deps = bank_service.BankDeps(store=store, policy=policy, clock=lambda: int(CLOCK))
    lam = FakeLambdaClient({
        "owlbank-auth": lambda e: auth_service.handle_event(auth_deps, e),
        "owlbank-bank": lambda e: bank_service.handle_event(bank_deps, e),
    })
    backend = LambdaBackendClient(lam, auth_function="owlbank-auth", bank_function="owlbank-bank")
    clf = resolver or FakeResolver()
    closer = closer or RecordingCloser()
    sealer = FakeSealer()
    sleeps: list[float] = []
    deps = Deps(store=store, catalog=catalog, policy=policy, backend=backend,
                auth_token=lambda: AUTH_TOKEN, basic_credentials=lambda: BASIC,
                basic_secret=lambda force=False: BasicSecret(*BASIC), sealer=sealer,
                pii=PiiGuard.for_policy(FakePiiRedactor(), policy), resolver=clf, clock=lambda: CLOCK, closer=closer,
                calls=calls, human_leg=human_leg, sleep=sleeps.append,
                briefing_model=briefing_model)
    cfg = FrontDoorConfig(relay_ws_url="wss://bridge.owlbank.example/ws",
                          conversation_configuration="conv_configuration_" + "d" * 26,
                          workflow_sid=WORKFLOW if flex else None, **(config_kw or {}))
    return Harness(store=store, app=FrontDoor(deps, cfg), lam=lam, verify=verify, sealer=sealer,
                   resolver=clf, closer=closer, sleeps=sleeps)


@pytest.fixture
def fd(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    h = build()
    yield h
    assert_no_leak(h, caplog.records, all_fixture_factor_values() | h.sent_factors)
    for sid in {k[0].split("#")[0] for k in h.store._tables["calls"]}:  # noqa: SLF001
        clear_sensitive(sid)


def sid(n: int) -> str:
    """A CallSid with no digit runs that could look like a factor."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "CA" + "e" * 30 + letters[n // 26 % 26] + letters[n % 26]


def says(root: ET.Element) -> list[str]:
    return [e.text or "" for e in root.iter("Say")]


def relay(root: ET.Element) -> ET.Element:
    found = root.find("./Connect/ConversationRelay")
    assert found is not None, ET.tostring(root)
    return found


def greeting(root: ET.Element) -> str:
    """The ConversationRelay ``welcomeGreeting`` (scripted read-outs are spoken here)."""
    return relay(root).get("welcomeGreeting") or ""


def assert_scripted(root: ET.Element, *texts: str, interruptible: str = "any") -> None:
    """A return to the AI after a scripted step is ONLY ``<Connect><ConversationRelay>``,
    with the scripted ``texts`` (in order) joined by single spaces as the interruptible greeting."""
    assert [c.tag for c in root] == ["Connect"], ET.tostring(root)
    assert says(root) == [] and root.find(".//Play") is None
    cr = relay(root)
    assert cr.get("welcomeGreeting") == " ".join(texts)
    assert cr.get("welcomeGreetingInterruptible") == interruptible


def assert_outcome(root: ET.Element, *outcome: str, follow: str,
                   interruptible: str = "any") -> None:
    """An outcome read-out is plain ``<Say>`` before ``<Connect>`` (it can't
    be cut off), and only the follow-up ``follow`` is the interruptible welcome greeting."""
    assert [c.tag for c in root] == ["Say"] * len(outcome) + ["Connect"], ET.tostring(root)
    assert says(root) == list(outcome)
    cr = relay(root)
    assert cr.get("welcomeGreeting") == follow
    assert cr.get("welcomeGreetingInterruptible") == interruptible
