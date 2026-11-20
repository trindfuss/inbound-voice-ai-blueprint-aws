"""scripts/twilio_setup.py: plan contents, dry run, idempotent apply, Basic credentials hidden.

A recording fake stands in for ``twilio.rest.Client`` and a stub for the Secrets Manager client.
Nothing here reaches Twilio or AWS.
"""

from __future__ import annotations

import contextlib
import inspect
import itertools
import json
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from twilio.base.exceptions import TwilioRestException

from owlbank.web_activity import TRAITS as WEB_TRAITS
from scripts import twilio_setup as ts
from scripts._common import DATA_DIR, ScriptError

from .test_memory_traits import MemoryStub

SUB = "AC" + "1" * 32
PARENT = "AC" + "2" * 32
NUMBER = "+12065550007"
API_SID = "SK" + "b2" * 16
API_SECRET = "TwilioApiSecret0123456789abcdefX"  # the format: 32 letters and digits
USER = "hook-user"
PASSWORD = "9f3e" * 10  # lower-case hex, 40 characters
BASE = ["--phone-number", NUMBER, "--api-domain", "api.owlbank.example",
        "--relay-domain", "relay.owlbank.example"]

API_KEY_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:owlbank/twilio/api-key-AbCdEf"
BASIC_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:owlbank/twilio/basic-auth-GhIjKl"
TOKEN_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:owlbank/twilio/auth-token-MnOpQr"
AUTH_TOKEN = "0123456789abcdef" * 2
#: The Restricted keys have their own tests (test_twilio_setup_restricted_keys.py); here they are skipped.
SECRET_ARGS = ["--secret", f"twilio_api_key={API_KEY_ARN}", "--secret", f"basic_auth={BASIC_ARN}",
               "--secret", f"twilio_auth_token={TOKEN_ARN}", "--skip-context-api-key"]
SECRET_VALUES: dict[str, dict[str, Any]] = {
    API_KEY_ARN: {"account_sid": SUB, "api_key": API_SID, "api_secret": API_SECRET},
    BASIC_ARN: {"user": USER, "password": PASSWORD},
    TOKEN_ARN: {"auth_token": AUTH_TOKEN},
}


class StubSM:
    """The Secrets Manager client's ``get_secret_value``: values by ARN (default the two above);
    records each ``SecretId`` read."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(SECRET_VALUES if values is None else values)
        self.reads: list[str] = []

    def get_secret_value(self, *, SecretId: str) -> dict[str, str]:  # noqa: N803 (boto3's name)
        self.reads.append(SecretId)
        return {"ARN": SecretId, "SecretString": json.dumps(self.values[SecretId])}


class NoSM:
    """A Secrets Manager client that must not be used (dry runs)."""

    def get_secret_value(self, **_kw: Any) -> Any:
        raise AssertionError("secret read in a dry run")


@pytest.fixture(autouse=True)
def _no_real_secrets_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that forgets ``sm_client`` fails instead of building a real boto3 client."""
    def refuse(region: str) -> Any:
        raise AssertionError(f"real Secrets Manager client requested ({region})")

    monkeypatch.setattr(ts, "default_sm_client", refuse)


# --------------------------------------------------------------------------- fake Twilio client


class _AsCredentials:
    """The fake as seen by one credential pair, the way Twilio answers: the Accounts resource refuses a
    Standard API key (an ``SK`` user) with error 70004. Everything else passes through to the fake."""

    def __init__(self, fake: FakeTwilio, user: str) -> None:
        self._fake, self._standard_key = fake, user.startswith("SK")

    def __getattr__(self, name: str) -> Any:
        if name == "api" and self._standard_key:
            return SimpleNamespace(v2010=SimpleNamespace(accounts=self._fake._refused_accounts))
        return getattr(self._fake, name)


class FakeTwilio:
    """The subset of ``twilio.rest.Client`` the script uses, with state and a call log."""

    def __init__(self, *, owner: str = PARENT, status: str = "active", workspace: bool = True,
                 numbers: int = 1, async_create: bool = False, key_account: str = SUB) -> None:
        #: The account the Standard key belongs to: Calls under any other account are refused.
        self.key_account = key_account
        #: Voice applications by SID (the number's ``voice_application_sid`` points at one).
        self.application_rows: dict[str, SimpleNamespace] = {}
        self.calls: list[tuple[str, Any]] = []
        #: Calls a Standard API key made that Twilio refuses (70004): a test fails on any.
        self.refused: list[tuple[str, Any]] = []
        self.ids = itertools.count(1)
        self.async_create = async_create
        self.account = SimpleNamespace(sid=SUB, owner_account_sid=owner, status=status)
        self.numbers = [SimpleNamespace(sid=f"PN{i:032d}", phone_number=NUMBER, voice_url=None,
                                        voice_method="GET", status_callback=None,
                                        status_callback_method="POST") for i in range(numbers)]
        self.workspaces = [SimpleNamespace(sid="WS" + "0" * 32, friendly_name=ts.FLEX_WORKSPACE)] \
            if workspace else []
        self.queues: list[SimpleNamespace] = []
        self.workflows: list[SimpleNamespace] = []
        self.stores: dict[str, SimpleNamespace] = {}
        self.configs: list[SimpleNamespace] = []
        self.verify_services: list[SimpleNamespace] = []
        #: Flex's Configuration (the skill definitions) and the workspace's workers.
        self.flex_skills: list[dict[str, Any]] = []
        self.workers: list[SimpleNamespace] = []
        self.config_http: list[tuple[str, Any]] = []
        self.config_status = 200
        # Memory's HTTP layer (trait group calls): the stub answers GET/POST of the WebActivity group
        self.memory_http = MemoryStub()
        self.http_client = self.memory_http.http_client
        self.username, self.password = API_SID, API_SECRET
        self.api = SimpleNamespace(v2010=SimpleNamespace(accounts=self._accounts))
        self.applications = _Callable(self._application)
        self.incoming_phone_numbers = _Callable(self._number, list=self._list_numbers)
        self.taskrouter = SimpleNamespace(v1=SimpleNamespace(workspaces=_Callable(
            self._workspace, list=self._list_workspaces)))
        self.memory = SimpleNamespace(v1=SimpleNamespace(stores=_Callable(
            self._store, list=self._list_stores, create=self._create_store)))
        self.conversations = SimpleNamespace(v2=SimpleNamespace(configurations=_Callable(
            self._config, list=self._list_configs, create=self._create_config)))
        self.verify = SimpleNamespace(v2=SimpleNamespace(services=_Callable(
            self._verify_service, list=self._list_verify, create=self._create_verify)))

    def request(self, method: str, uri: str, params: Any = None, data: Any = None,
                headers: Any = None, **kw: Any) -> Any:
        """Flex's Configuration endpoint (the skill definitions)."""
        assert uri == ts.FLEX_CONFIGURATION_URL, uri
        self.config_http.append((method, data))
        self.calls.append(("flex.configuration", method))
        if self.config_status >= 400:
            return SimpleNamespace(status_code=self.config_status,
                                   text=f'{{"code": 20003, "message": "{SECRET_BODY}"}}')
        if method == "POST":
            self.flex_skills = list(data["taskrouter_skills"])
        return SimpleNamespace(status_code=200, text=json.dumps(
            {"account_sid": SUB, "taskrouter_skills": self.flex_skills}))

    def writes(self) -> list[tuple[str, Any]]:
        return [c for c in self.calls if c[0].split(".")[-1] in ("create", "update")]

    def _sid(self, prefix: str) -> str:
        return f"{prefix}{next(self.ids):032d}"

    def factory(self, user: str, secret: str, account: str) -> Any:
        """A ``client_factory``: the same fake, answering as the credentials given would be answered."""
        return _AsCredentials(self, user)

    def _refused_accounts(self, sid: str) -> Any:
        def fetch() -> Any:
            self.refused.append(("account.fetch", sid))
            raise TwilioRestException(403, f"/Accounts/{sid}", "Resource not accessible with API Key",
                                      code=70004)
        return SimpleNamespace(fetch=fetch, calls=self._calls_of(sid))

    def _calls_of(self, sid: str) -> Any:
        def list_calls(limit: int | None = None) -> list[Any]:
            self.calls.append(("calls.list", sid))
            if sid != self.key_account:
                raise TwilioRestException(401, f"/Accounts/{sid}/Calls", "Authenticate", code=20003)
            return []
        return SimpleNamespace(list=list_calls)

    def _application(self, sid: str) -> Any:
        def fetch() -> Any:
            self.calls.append(("applications.fetch", sid))
            return self.application_rows[sid]
        return SimpleNamespace(fetch=fetch)

    # account
    def _accounts(self, sid: str) -> Any:
        def fetch() -> Any:
            self.calls.append(("account.fetch", sid))
            return self.account
        return SimpleNamespace(fetch=fetch, calls=self._calls_of(sid))

    # numbers
    def _list_numbers(self, phone_number: str) -> list[Any]:
        self.calls.append(("numbers.list", phone_number))
        return [n for n in self.numbers if n.phone_number == phone_number]

    def _number(self, sid: str) -> Any:
        def update(**kw: Any) -> Any:
            self.calls.append(("numbers.update", kw))
            n = next(n for n in self.numbers if n.sid == sid)
            for k, v in kw.items():
                setattr(n, k, v)
            return n
        return SimpleNamespace(update=update)

    # taskrouter
    def _list_workspaces(self, friendly_name: str | None = None) -> list[Any]:
        self.calls.append(("workspaces.list", friendly_name))
        return [w for w in self.workspaces if friendly_name in (None, w.friendly_name)]

    def _workspace(self, ws: str) -> Any:
        def list_q(friendly_name: str) -> list[Any]:
            self.calls.append(("queues.list", friendly_name))
            return [q for q in self.queues if q.friendly_name == friendly_name]

        def create_q(friendly_name: str, target_workers: str) -> Any:
            self.calls.append(("queues.create", {"friendly_name": friendly_name,
                                                 "target_workers": target_workers}))
            q = SimpleNamespace(sid=self._sid("WQ"), friendly_name=friendly_name,
                                target_workers=target_workers)
            self.queues.append(q)
            return q

        def queue(sid: str) -> Any:
            def update(**kw: Any) -> Any:
                self.calls.append(("queues.update", kw))
                q = next(q for q in self.queues if q.sid == sid)
                q.target_workers = kw["target_workers"]
                return q
            return SimpleNamespace(update=update)

        def list_w(friendly_name: str) -> list[Any]:
            self.calls.append(("workflows.list", friendly_name))
            return [w for w in self.workflows if w.friendly_name == friendly_name]

        def create_w(friendly_name: str, configuration: str) -> Any:
            self.calls.append(("workflows.create", {"friendly_name": friendly_name,
                                                    "configuration": configuration}))
            w = SimpleNamespace(sid=self._sid("WW"), friendly_name=friendly_name,
                                configuration=configuration)
            self.workflows.append(w)
            return w

        def workflow(sid: str) -> Any:
            def update(**kw: Any) -> Any:
                self.calls.append(("workflows.update", kw))
                w = next(w for w in self.workflows if w.sid == sid)
                w.configuration = kw["configuration"]
                return w
            return SimpleNamespace(update=update)

        def list_workers() -> list[Any]:
            self.calls.append(("workers.list", None))
            return list(self.workers)

        def worker(sid: str) -> Any:
            def update(**kw: Any) -> Any:
                self.calls.append(("workers.update", kw))
                w = next(w for w in self.workers if w.sid == sid)
                w.attributes = kw["attributes"]
                return w
            return SimpleNamespace(update=update)

        return SimpleNamespace(task_queues=_Callable(queue, list=list_q, create=create_q),
                               workflows=_Callable(workflow, list=list_w, create=create_w),
                               workers=_Callable(worker, list=list_workers))

    # memory
    def _list_stores(self) -> list[str]:
        self.calls.append(("stores.list", None))
        return list(self.stores)

    def _store(self, store_id: str) -> Any:
        def fetch() -> Any:
            self.calls.append(("stores.fetch", store_id))
            return self.stores[store_id]
        return SimpleNamespace(fetch=fetch)

    def _create_store(self, req: Any) -> Any:
        body = req.to_dict()
        self.calls.append(("stores.create", body))
        sid = f"mem_store_{next(self.ids):026d}"
        self.stores[sid] = SimpleNamespace(id=sid, display_name=body["displayName"])
        return SimpleNamespace(id=None if self.async_create else sid)

    # conversation configurations
    def _list_configs(self) -> list[Any]:
        self.calls.append(("configs.list", None))
        return list(self.configs)

    def _instance(self, cid: str, body: dict[str, Any]) -> Any:
        return SimpleNamespace(
            id=cid, display_name=body["displayName"],
            conversation_grouping_type=body["conversationGroupingType"],
            memory_store_id=body["memoryStoreId"], channel_settings=body["channelSettings"],
            status_callbacks=body.get("statusCallbacks"),
            memory_extraction_enabled=body["memoryExtractionEnabled"],
            intelligence_configuration_ids=body.get("intelligenceConfigurationIds"))

    def _create_config(self, create_configuration_request: Any) -> Any:
        body = create_configuration_request.to_dict()
        self.calls.append(("configs.create", body))
        inst = self._instance(f"conv_configuration_{next(self.ids):026d}", body)
        self.configs.append(inst)
        return SimpleNamespace(id=None if self.async_create else inst.id)

    def _config(self, cid: str) -> Any:
        def update(update_configuration_request: Any) -> Any:
            body = update_configuration_request.to_dict()
            self.calls.append(("configs.update", body))
            i = next(i for i, c in enumerate(self.configs) if c.id == cid)
            self.configs[i] = self._instance(cid, body)
            return self.configs[i]
        return SimpleNamespace(update=update)


    # verify services
    def _list_verify(self) -> list[Any]:
        self.calls.append(("verify.list", None))
        return list(self.verify_services)

    def _create_verify(self, **kw: Any) -> Any:
        self.calls.append(("verify.create", kw))
        svc = SimpleNamespace(sid="VA" + f"{next(self.ids):032x}", friendly_name=kw["friendly_name"],
                              code_length=kw.get("code_length", 6),
                              do_not_share_warning_enabled=kw.get("do_not_share_warning_enabled", False),
                              default_template_sid=kw.get("default_template_sid"))
        self.verify_services.append(svc)
        return svc

    def _verify_service(self, sid: str) -> Any:
        def update(**kw: Any) -> Any:
            self.calls.append(("verify.update", kw))
            svc = next(v for v in self.verify_services if v.sid == sid)
            for k, v in kw.items():
                setattr(svc, k, v)
            return svc
        return SimpleNamespace(update=update)


class _Callable:
    """``list_resource(sid)`` plus attribute methods, like the SDK's list resources."""

    def __init__(self, ctx: Any, **methods: Any) -> None:
        self._ctx = ctx
        self.__dict__.update(methods)

    def __call__(self, sid: str) -> Any:
        return self._ctx(sid)


def apply_args(tmp_path: Path, *, with_flex: bool = True) -> list[str]:
    """Apply arguments; ``--with-flex`` by default (most tests exercise the Flex steps)."""
    return [*BASE, "--account-sid", SUB, "--apply", "--output", str(tmp_path / "twilio.json"), *SECRET_ARGS,
            *(["--with-flex"] if with_flex else [])]


def run_apply(tmp_path: Path, fake: FakeTwilio, extra: list[str] | None = None, *,
              with_flex: bool = True) -> int:
    made: list[tuple[str, str, str]] = []

    def factory(key: str, secret: str, account: str) -> Any:
        made.append((key, secret, account))
        return fake.factory(key, secret, account)

    rc = ts.main(apply_args(tmp_path, with_flex=with_flex) + (extra or []), sm_client=StubSM(),
                 client_factory=factory)
    if rc == 0:
        assert made == [(API_SID, API_SECRET, SUB), (SUB, AUTH_TOKEN, SUB)]
    assert fake.refused == []
    return rc


# --------------------------------------------------------------------------- dry run


FLEX_ONLY_TEXT = ("OwlBank Support EN", "OwlBank Support ES", "OwlBank Fraud", "OwlBank Disputes",
                  ts.WORKFLOW_NAME, "WQ<fraud>", "WQ<disputes>", "ensure_queue", "ensure_workflow",
                  "ensure_flex_skills")


@pytest.mark.parametrize("with_flex", [False, True])
def test_dry_run_is_default_and_makes_zero_calls(capsys: pytest.CaptureFixture[str], with_flex: bool) -> None:
    def no_client(*a: Any) -> Any:
        raise AssertionError("Twilio client built in dry run")

    assert ts.main([*BASE, *(["--with-flex"] if with_flex else [])], sm_client=NoSM(),
                   client_factory=no_client) == 0
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN")
    for text in ("check_account", ts.MEMORY_STORE_NAME, "https://***:***@api.owlbank.example/voice/incoming",
                 "https://***:***@api.owlbank.example/voice/status",
                 "https://relay.owlbank.example/webhook", "Disputes", '"inactive": 15'):
        if text == "Disputes" and not with_flex:
            continue
        assert text in out, text
    for text in FLEX_ONLY_TEXT:
        assert (text in out) is with_flex, text
    assert "captureRules" not in out


# --------------------------------------------------------------------------- secrets


def test_dry_run_reads_no_secret_even_with_the_arns_known(tmp_path: Path) -> None:
    outputs = tmp_path / "cdk-outputs.json"
    outputs.write_text(json.dumps({"OwlBankData": {"SecretTwilioApiKey": API_KEY_ARN,
                                                   "SecretBasicAuth": BASIC_ARN}}))

    def no_client(*a: Any) -> Any:
        raise AssertionError("Twilio client built in dry run")

    for phase in ("pre", "post", "all"):
        args = ["--phase", phase, "--cdk-outputs", str(outputs), *SECRET_ARGS]
        if phase != "pre":
            args += BASE
        assert ts.main(args, sm_client=NoSM(), client_factory=no_client) == 0


@pytest.mark.parametrize("phase, read", [
    ("pre", [API_KEY_ARN, TOKEN_ARN]),                # no webhook URL, so no Basic credentials
    ("post", [API_KEY_ARN, BASIC_ARN, TOKEN_ARN]),
    ("all", [API_KEY_ARN, BASIC_ARN, TOKEN_ARN]),
])
def test_apply_reads_the_api_key_and_token_always_and_basic_auth_outside_pre(
        tmp_path: Path, phase: str, read: list[str]) -> None:
    fake, sm = FakeTwilio(), StubSM()
    made: list[tuple[str, str, str]] = []

    def factory(key: str, secret: str, account: str) -> Any:
        made.append((key, secret, account))
        return fake.factory(key, secret, account)

    if phase == "post":  # post needs pre's configuration
        assert ts.main([*apply_args(tmp_path), "--phase", "pre"], sm_client=StubSM(),
                       client_factory=fake.factory) == 0
    assert ts.main([*apply_args(tmp_path), "--phase", phase], sm_client=sm, client_factory=factory) == 0
    assert sm.reads == read
    assert made == [(API_SID, API_SECRET, SUB), (SUB, AUTH_TOKEN, SUB)]
    assert fake.refused == []
    if phase != "pre":
        assert fake.numbers[0].voice_url == f"https://{USER}:{PASSWORD}@api.owlbank.example/voice/incoming"


def test_pre_resolves_the_api_key_from_the_data_stack_outputs_alone(tmp_path: Path) -> None:
    """Phase pre runs when only OwlBankData is deployed; its outputs name the secrets."""
    outputs = tmp_path / "cdk-outputs.json"
    outputs.write_text(json.dumps({"OwlBankData": {"SecretTwilioApiKey": API_KEY_ARN,
                                                   "SecretTwilioAuthToken": TOKEN_ARN,
                                                   "SecretBasicAuth": BASIC_ARN}}))
    sm = StubSM()
    args = ["--phase", "pre", "--account-sid", SUB, "--apply", "--output", str(tmp_path / "t.json"),
            "--cdk-outputs", str(outputs), "--skip-context-api-key"]
    assert ts.main(args, sm_client=sm, client_factory=FakeTwilio().factory) == 0
    assert sm.reads == [API_KEY_ARN, TOKEN_ARN]


def test_secret_override_wins_over_the_outputs(tmp_path: Path) -> None:
    other = API_KEY_ARN.replace("AbCdEf", "ZzZzZz")
    outputs = tmp_path / "cdk-outputs.json"
    outputs.write_text(json.dumps({"OwlBankData": {"SecretTwilioApiKey": API_KEY_ARN,
                                                   "SecretTwilioAuthToken": TOKEN_ARN}}))
    sm = StubSM({other: SECRET_VALUES[API_KEY_ARN], TOKEN_ARN: SECRET_VALUES[TOKEN_ARN]})
    args = ["--phase", "pre", "--account-sid", SUB, "--apply", "--output", str(tmp_path / "t.json"),
            "--cdk-outputs", str(outputs), "--secret", f"twilio_api_key={other}", "--skip-context-api-key"]
    assert ts.main(args, sm_client=sm, client_factory=FakeTwilio().factory) == 0
    assert sm.reads == [other, TOKEN_ARN]


@pytest.mark.parametrize("phase, missing", [("pre", "twilio_api_key"), ("post", "basic_auth"),
                                            ("pre", "twilio_auth_token"), ("post", "twilio_auth_token")])
def test_a_missing_arn_stops_before_any_read(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                             phase: str, missing: str) -> None:
    args = ["--phase", phase, "--account-sid", SUB, "--apply", "--output", str(tmp_path / "t.json")]
    if phase != "pre":
        args += BASE
    if missing != "twilio_api_key":
        args += ["--secret", f"twilio_api_key={API_KEY_ARN}"]
    if missing == "basic_auth":
        args += ["--secret", f"twilio_auth_token={TOKEN_ARN}"]
    elif missing == "twilio_auth_token" and phase != "pre":
        args += ["--secret", f"basic_auth={BASIC_ARN}"]

    def no_client(*a: Any) -> Any:
        raise AssertionError("Twilio client built without the secrets")

    assert ts.main(args, sm_client=NoSM(), client_factory=no_client) == 2
    err = capsys.readouterr().err
    assert f"no ARN for secret {missing}" in err and "--cdk-outputs" in err


def test_an_unknown_secret_name_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--secret", "twilio_password=arn:x"], sm_client=NoSM()) == 2
    assert "unknown secret name" in capsys.readouterr().err


@pytest.mark.parametrize("arn, value, field", [
    (BASIC_ARN, {"placeholder": True, "unset": "x" * 20}, "placeholder"),
    (API_KEY_ARN, {"placeholder": True, "unset": "x" * 20}, "placeholder"),
    (API_KEY_ARN, {**SECRET_VALUES[API_KEY_ARN], "api_secret": "short-" + API_SECRET}, "api_secret"),
    (API_KEY_ARN, {"account_sid": SUB, "api_secret": API_SECRET}, "api_key"),
    (BASIC_ARN, {"user": USER, "password": PASSWORD.upper()}, "password"),
])
def test_a_placeholder_or_malformed_secret_stops_it_naming_the_field_never_the_value(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
        arn: str, value: dict[str, Any], field: str) -> None:
    caplog.set_level("DEBUG")
    sm = StubSM({**SECRET_VALUES, arn: value})

    def no_client(*a: Any) -> Any:
        raise AssertionError("Twilio client built with an unusable secret")

    assert ts.main(apply_args(tmp_path), sm_client=sm, client_factory=no_client) == 3
    err = capsys.readouterr().err
    logical = "basic_auth" if arn == BASIC_ARN else "twilio_api_key"
    assert logical in err and field in err
    logged = caplog.text
    for v in (API_SECRET, PASSWORD, PASSWORD.upper(), "short-" + API_SECRET):
        assert v not in err and v not in logged
    assert not (tmp_path / "twilio.json").exists()


def test_without_an_injected_client_the_region_comes_from_the_arn(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    regions: list[str] = []
    sm = StubSM({k.replace("us-east-1", "eu-west-1"): v for k, v in SECRET_VALUES.items()})

    def factory(region: str) -> StubSM:
        regions.append(region)
        return sm

    monkeypatch.setattr(ts, "default_sm_client", factory)
    args = [*BASE, "--account-sid", SUB, "--apply", "--output", str(tmp_path / "twilio.json"),
            *(a.replace("us-east-1", "eu-west-1") for a in SECRET_ARGS)]
    assert ts.main(args, client_factory=FakeTwilio().factory) == 0
    assert regions == ["eu-west-1"] and len(sm.reads) == 3
    regions.clear()
    assert ts.main([*args, "--region", "us-west-2"], client_factory=FakeTwilio().factory) == 0
    assert regions == ["us-west-2"]


@settings(max_examples=200, deadline=None)
@given(user=st.from_regex(r"[A-Za-z0-9._-]{1,64}", fullmatch=True),
       password=st.from_regex(r"[0-9a-f]{32,96}", fullmatch=True))
def test_property_password_webhook_url_round_trips_unescaped(user: str, password: str) -> None:
    """A webhook URL built with any password (lower-case hex, 32 or more) and a valid user
    needs no percent-escaping and round-trips through ``urllib.parse`` unchanged."""
    from urllib.parse import quote, unquote, urlsplit, urlunsplit

    from owlbank.secret_schema import validate

    assert validate("basic_auth", {"user": user, "password": password}) == []
    for path in ("/voice/incoming", "/v1/voice/status", ts.ORCHESTRATOR_CALLBACK_PATH):
        url = ts.CredUrl("api.owlbank.example", path).render(ts.BasicCreds(user, password))
        assert url == f"https://{user}:{password}@api.owlbank.example{path}"  # nothing escaped
        assert "%" not in url and quote(password, safe="") == password and quote(user, safe="") == user
        parts = urlsplit(url)
        assert (parts.username, parts.password, parts.hostname, parts.path) == (
            user, password, "api.owlbank.example", path)
        assert unquote(parts.password or "") == password and urlunsplit(parts) == url
        assert ts.same_url(url, urlunsplit(parts))


def test_the_bridge_status_callback_url_has_no_credentials() -> None:
    """The bridge's /webhook is signature-only and never challenges for Basic, so Twilio would
    never send credentials to it (and a signed URL carrying them would fail TAC's check)."""
    from urllib.parse import urlsplit

    assert ts.bridge_webhook_url(cfg(relay_domain="relay.owlbank.example")) == "https://relay.owlbank.example/webhook"
    for phase in ("post", "all"):
        plan = ts.build_plan(cfg(phase=phase))
        (step,) = [s for s in plan if s.kind in ("ensure_status_callbacks", "ensure_conversation_configuration")]
        url = step.params["webhook"]
        assert isinstance(url, str) and urlsplit(url).username is None and "@" not in url
        assert "***" not in str(step.describe())
        if phase == "all":
            assert step.params["body"]["statusCallbacks"] == [{"url": url, "method": "POST"}]
    # The number's voice and status URLs on the API keep their credentials (the API challenges).
    (number,) = [s for s in ts.build_plan(cfg(phase="post")) if s.kind == "configure_number"]
    assert str(number.params["voice_url"]).startswith("https://***:***@")
    assert str(number.params["status_callback"]).startswith("https://***:***@")


def test_dry_run_reads_domains_from_cdk_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"S": {"ApiDomain": "a.example", "RelayDomain": "r.example"}}))
    assert ts.main(["--phone-number", NUMBER, "--cdk-outputs", str(p)]) == 0
    assert "r.example/webhook" in capsys.readouterr().out


EXEC_HOST = "abc123def4.execute-api.us-east-1.amazonaws.com"
CF_HOST = "d1234abcd.cloudfront.net"


def test_dry_run_cloudfront_mode_adds_the_stage_to_the_webhook_urls(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """No domain: ApiDomain is the execute-api host and the URLs carry the stage; RelayDomain is the
    CloudFront domain (the bridge webhook has no stage)."""
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"OwlBankApi": {"ApiDomain": EXEC_HOST, "ApiStage": "v1"},
                             "OwlBankBridge": {"RelayDomain": CF_HOST}}))
    assert ts.main(["--phone-number", NUMBER, "--cdk-outputs", str(p)]) == 0
    out = capsys.readouterr().out
    assert f"https://***:***@{EXEC_HOST}/v1/voice/incoming" in out
    assert f"https://***:***@{EXEC_HOST}/v1/voice/status" in out
    assert f"https://{CF_HOST}/webhook" in out  # the bridge callback carries no credentials
    assert f"***@{CF_HOST}" not in out
    assert f"{EXEC_HOST}/voice/" not in out


def test_execute_api_host_defaults_to_stage_v1_and_the_output_wins(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main(["--phone-number", NUMBER, "--api-domain", EXEC_HOST, "--relay-domain", CF_HOST]) == 0
    assert f"{EXEC_HOST}/v1/voice/incoming" in capsys.readouterr().out
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"ApiStage": "prod2"}))
    assert ts.main(["--phone-number", NUMBER, "--api-domain", EXEC_HOST, "--relay-domain", CF_HOST,
                    "--cdk-outputs", str(p)]) == 0
    assert f"{EXEC_HOST}/prod2/voice/incoming" in capsys.readouterr().out
    assert ts.main(["--phone-number", NUMBER, "--api-domain", EXEC_HOST, "--relay-domain", CF_HOST,
                    "--cdk-outputs", str(p), "--api-stage", "v9"]) == 0
    assert f"{EXEC_HOST}/v9/voice/incoming" in capsys.readouterr().out


def test_custom_domain_urls_have_no_stage_even_with_a_stale_stage_output(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"ApiStage": "v1"}))  # left over from a CloudFront-mode deploy
    assert ts.main([*BASE, "--cdk-outputs", str(p)]) == 0
    out = capsys.readouterr().out
    assert "https://***:***@api.owlbank.example/voice/incoming" in out
    assert "api.owlbank.example/v1/" not in out


def test_apply_puts_the_stage_in_the_number_urls(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake, ["--api-domain", EXEC_HOST, "--relay-domain", CF_HOST]) == 0
    num = fake.numbers[0]
    assert num.voice_url == f"https://{USER}:{PASSWORD}@{EXEC_HOST}/v1/voice/incoming"
    assert num.status_callback == f"https://{USER}:{PASSWORD}@{EXEC_HOST}/v1/voice/status"
    assert fake.configs[0].status_callbacks == [
        {"url": f"https://{CF_HOST}/webhook", "method": "POST"}]  # no credentials


@pytest.mark.parametrize("stage", ["v 1", "v1/x", "x" * 129])
def test_bad_stage_is_refused(stage: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = ts.main(["--phone-number", NUMBER, "--api-domain", EXEC_HOST, "--relay-domain", CF_HOST,
                    "--api-stage", stage])
    assert code == 2 and "stage" in capsys.readouterr().err


# --------------------------------------------------------------------------- plan contents


def cfg(**kw: Any) -> ts.SetupConfig:
    policy = json.loads((DATA_DIR / "policy.json").read_text(encoding="utf-8"))
    base = {"phone_number": NUMBER, "api_domain": "api.x", "relay_domain": "relay.x", "account_sid": SUB,
            "with_flex": True, "queue_names": ts.queue_names_from_policy(policy)}
    return ts.SetupConfig(**{**base, **kw})


def test_plan_order_without_flex() -> None:
    assert [s.kind for s in ts.build_plan(cfg(with_flex=False))] == [
        "check_account", "configure_number", "ensure_memory_store", "ensure_conversation_configuration",
        "ensure_verify_service", "ensure_trait_group"]


def test_plan_order_and_queues() -> None:
    plan = ts.build_plan(cfg())
    assert [s.kind for s in plan] == ["check_account", "configure_number", "ensure_queue", "ensure_queue",
                                      "ensure_queue", "ensure_queue", "ensure_workflow", "ensure_flex_skills",
                                      "ensure_memory_store", "ensure_conversation_configuration", "ensure_verify_service",
                                      "ensure_trait_group"]
    queues = {s.name: s.params["target_workers"] for s in plan if s.kind == "ensure_queue"}
    assert queues == {"OwlBank Support EN": 'routing.skills HAS "en"',
                      "OwlBank Support ES": 'routing.skills HAS "es"',
                      "OwlBank Fraud": 'routing.skills HAS "fraud"',
                      "OwlBank Disputes": 'routing.skills HAS "disputes"'}


def test_workflow_matches_handover() -> None:
    wf = ts.workflow_configuration({"support-en": "WQen", "support-es": "WQes", "fraud": "WQfr",
                                   "disputes": "WQdi"})
    filters = wf["task_routing"]["filters"]
    assert [f["filter_friendly_name"] for f in filters] == [
        "Fraud", "Disputes", "Vulnerable customer", "Vulnerable customer EN", "Spanish"]
    assert filters[0]["expression"] == "escalation_reason IN ['fraud_suspected', 'verification_risk']"
    assert filters[0]["targets"] == [{"queue": "WQfr", "priority": 100}]
    assert filters[1]["expression"] == "escalation_reason IN ['card_dispute']"
    assert filters[1]["targets"] == [{"queue": "WQdi", "priority": 50}]
    assert filters[2]["expression"] == "escalation_reason == 'vulnerable_customer' AND language == 'es-US'"
    assert wf["task_routing"]["default_filter"] == {"queue": "WQen"}


def _targets(routing: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for f in routing["filters"] for x in f["targets"]] + [routing["default_filter"]]


def test_task_timeout_option_is_gone() -> None:
    """Callers may wait an hour or two, so nothing may cancel the task on a timer."""
    assert not hasattr(ts, "DEFAULT_TASK_TIMEOUT_S") and not hasattr(ts, "MAX_TASK_TIMEOUT_S")
    assert "task_timeout_s" not in inspect.signature(ts.SetupConfig).parameters
    assert "--task-timeout-s" not in ts.build_parser().format_help()
    with pytest.raises(SystemExit):
        ts.build_parser().parse_args([*BASE, "--task-timeout-s", "300"])


QUEUE_SID = st.text(alphabet="WQabcdef0123456789", min_size=1, max_size=34)


@settings(max_examples=100, deadline=None)
@given(sids=st.fixed_dictionaries({k: QUEUE_SID for k, _, _, _ in ts.QUEUES}),
       phase=st.sampled_from(ts.PHASES), hosting=st.sampled_from(ts.HOSTINGS),
       events=st.booleans())
def test_pbt_no_target_or_catch_all_has_a_timeout(sids: dict[str, str], phase: str, hosting: str,
                                                  events: bool) -> None:
    """For any queue SIDs and any setup input, no workflow target and not the catch-all carries a
    ``timeout``; the catch-all is kept and goes to the English queue."""
    routing = ts.workflow_configuration(sids)["task_routing"]
    assert routing["default_filter"] == {"queue": sids["support-en"]}
    assert all("timeout" not in x for x in _targets(routing))
    c = cfg(phase=phase, bridge_hosting=hosting, taskrouter_events=events and phase != "pre")
    steps = [s for s in ts.build_plan(c) if s.kind == "ensure_workflow"]
    assert len(steps) == (0 if phase == "post" else 1)  # ``post`` doesn't touch the workflow
    for step in steps:
        assert all("timeout" not in x for x in _targets(step.params["configuration"]["task_routing"]))


def test_intelligence_ids_are_carried_into_the_body_and_never_added() -> None:
    c = cfg()
    assert "intelligenceConfigurationIds" not in ts.conversation_configuration(c, "m", None)
    assert ts.conversation_configuration(c, "m", None, ["intelligence_configuration_a"])[
        "intelligenceConfigurationIds"] == ["intelligence_configuration_a"]


def test_fraud_filter_covers_every_fraud_queue_reason_in_the_policy() -> None:
    policy = json.loads((DATA_DIR / "policy.json").read_text())
    fraud = {r for r, v in policy["escalation"]["reasons"].items() if v["queue"] == "fraud"}
    assert set(ts.FRAUD_REASONS) == fraud


def test_every_disputes_queue_reason_in_the_policy_routes_to_the_disputes_queue() -> None:
    policy = json.loads((DATA_DIR / "policy.json").read_text())
    disputes = {r for r, v in policy["escalation"]["reasons"].items() if v["queue"] == "disputes"}
    assert disputes and set(ts.DISPUTES_REASONS) == disputes
    wf = ts.workflow_configuration({"support-en": "WQen", "support-es": "WQes", "fraud": "WQfr",
                                    "disputes": "WQdi"})
    to_disputes = [f for f in wf["task_routing"]["filters"] if f["targets"][0]["queue"] == "WQdi"]
    assert len(to_disputes) == 1
    for reason in disputes:
        assert f"'{reason}'" in to_disputes[0]["expression"]


def test_conversation_configuration_body() -> None:
    body = ts.conversation_configuration(cfg(inactive_timeout_min=20, closed_timeout_min=21), "mem_1",
                                         "https://relay.x/webhook")
    assert body["channelSettings"] == {"VOICE": {"statusTimeouts": {"inactive": 20, "closed": 21}}}
    assert body["statusCallbacks"] == [{"url": "https://relay.x/webhook", "method": "POST"}]
    assert body["memoryStoreId"] == "mem_1" and body["memoryExtractionEnabled"] is False
    assert "intelligenceConfigurationIds" not in body
    assert len(body["displayName"]) <= 32


@pytest.mark.parametrize("kw, msg", [
    ({"inactive_timeout_min": ts.LONGEST_DETERMINISTIC_SEGMENT_MIN}, "longest deterministic segment"),
    ({"closed_timeout_min": 0}, "closed"),
    ({"phone_number": "2065550007"}, "E.164"),
    ({"api_domain": "https://api.x"}, "bare host"),
    ({"relay_domain": "u:p@relay.x"}, "bare host"),
    ({"account_sid": "SK" + "0" * 32}, "Account SID"),
])
def test_config_guards(kw: dict[str, Any], msg: str) -> None:
    with pytest.raises(ScriptError, match=msg):
        cfg(**kw)


# --------------------------------------------------------------------------- apply


def test_apply_creates_everything_and_writes_outputs(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    out = json.loads((tmp_path / "twilio.json").read_text())
    assert out["accountSid"] == SUB and out["phoneNumberSid"] == fake.numbers[0].sid
    assert set(out["queueSids"]) == {"support-en", "support-es", "fraud", "disputes"}
    assert out["workflowSid"].startswith("WW") and out["memoryStoreId"].startswith("mem_store_")
    assert out["conversationConfigurationId"].startswith("conv_configuration_")

    num = fake.numbers[0]
    assert num.voice_url == f"https://{USER}:{PASSWORD}@api.owlbank.example/voice/incoming"
    assert num.status_callback == f"https://{USER}:{PASSWORD}@api.owlbank.example/voice/status"
    assert num.voice_method == "POST"
    wf = json.loads(fake.workflows[0].configuration)
    assert wf["task_routing"]["filters"][0]["targets"][0]["queue"] == out["queueSids"]["fraud"]
    conf = fake.configs[0]
    assert conf.memory_store_id == out["memoryStoreId"]
    assert conf.status_callbacks == [{"url": "https://relay.owlbank.example/webhook", "method": "POST"}]
    assert not conf.channel_settings["VOICE"].get("captureRules")

    cap = capsys.readouterr()
    assert PASSWORD not in cap.out + cap.err and API_SECRET not in cap.out + cap.err
    assert PASSWORD not in (tmp_path / "twilio.json").read_text()


def test_apply_is_idempotent(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    first = json.loads((tmp_path / "twilio.json").read_text())
    fake.calls.clear()
    assert run_apply(tmp_path, fake) == 0
    assert fake.writes() == []
    assert json.loads((tmp_path / "twilio.json").read_text()) == first
    assert len(fake.queues) == 4 and len(fake.workflows) == 1 and len(fake.configs) == 1


def test_apply_creates_the_trait_group_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    (_, _, body), = fake.memory_http.calls("POST")
    assert body["displayName"] == "WebActivity" and set(body["traits"]) == set(WEB_TRAITS)
    assert "create  trait group WebActivity" in capsys.readouterr().out
    fake.memory_http.log.clear()
    assert run_apply(tmp_path, fake) == 0
    assert fake.memory_http.calls("POST") == []
    second = capsys.readouterr().out
    assert "ok      trait group WebActivity" in second and API_SECRET not in second


@pytest.mark.parametrize("failure", ["refused", "network"])
def test_a_trait_group_failure_fails_open(failure: str, tmp_path: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    """The trait group is last and fails open; everything a call needs is made,
    the output file is written, and the warning names no secret."""
    fake = FakeTwilio()
    if failure == "refused":
        fake.memory_http.post_status = 403
    else:
        def down(*_a: Any, **_k: Any) -> Any:
            raise ConnectionError(f"memory unreachable {API_SECRET}")

        fake.http_client = SimpleNamespace(request=down)
    assert run_apply(tmp_path, fake) == 0
    out = json.loads((tmp_path / "twilio.json").read_text())
    assert out["conversationConfigurationId"] and out["verifyServiceSid"]
    printed = capsys.readouterr()
    assert "warn    trait group WebActivity not made" in printed.out
    assert API_SECRET not in printed.out + printed.err


def test_dry_run_plans_the_trait_group_and_calls_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert ts.main(BASE, sm_client=StubSM(), client_factory=fake.factory) == 0
    out = capsys.readouterr().out
    assert "ensure_trait_group" in out and "create trait group WebActivity" in out
    assert fake.memory_http.log == []


def test_apply_fixes_drift(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    fake.queues[0].target_workers = "1==1"
    fake.workflows[0].configuration = "{}"
    fake.configs[0].status_callbacks = []
    fake.calls.clear()
    assert run_apply(tmp_path, fake) == 0
    assert sorted(c[0] for c in fake.writes()) == ["configs.update", "queues.update", "workflows.update"]


def test_apply_turns_extraction_off_and_keeps_ci_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    assert fake.configs[0].memory_extraction_enabled is False
    fake.configs[0].memory_extraction_enabled = True
    fake.configs[0].intelligence_configuration_ids = ["intelligence_configuration_x", "intelligence_configuration_y"]
    fake.calls.clear()
    assert run_apply(tmp_path, fake) == 0
    updates = [c[1] for c in fake.calls if c[0] == "configs.update"]
    assert len(updates) == 1 and updates[0]["memoryExtractionEnabled"] is False
    assert updates[0]["intelligenceConfigurationIds"] == ["intelligence_configuration_x",
                                                          "intelligence_configuration_y"]
    assert fake.configs[0].intelligence_configuration_ids == ["intelligence_configuration_x",
                                                              "intelligence_configuration_y"]
    # a fresh configuration never gets any
    fresh = FakeTwilio()
    assert run_apply(tmp_path, fresh) == 0
    assert not fresh.configs[0].intelligence_configuration_ids
    out = capsys.readouterr()
    assert API_SECRET not in out.out + out.err


def test_apply_writes_a_workflow_without_any_timeout(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    routing = json.loads(fake.workflows[0].configuration)["task_routing"]
    assert routing["default_filter"] == {"queue": routing["default_filter"]["queue"]}
    assert all("timeout" not in x for x in _targets(routing))
    assert '"timeout"' not in fake.workflows[0].configuration


@settings(max_examples=50, deadline=None)
@given(ids=st.lists(st.text(alphabet="abcdefghij0123456789_", min_size=1, max_size=40), max_size=8),
       extraction=st.booleans())
def test_pbt_update_keeps_every_intelligence_id(ids: list[str], extraction: bool, tmp_path_factory: Any) -> None:
    """For any existing configuration, an update keeps every intelligenceConfigurationIds entry."""
    tmp = tmp_path_factory.mktemp("ci")
    fake = FakeTwilio()
    assert run_apply(tmp, fake) == 0
    fake.configs[0].intelligence_configuration_ids = list(ids)
    fake.configs[0].memory_extraction_enabled = extraction
    fake.configs[0].status_callbacks = []
    fake.calls.clear()
    assert run_apply(tmp, fake) == 0
    assert fake.configs[0].intelligence_configuration_ids == (ids or None)
    assert fake.configs[0].memory_extraction_enabled is False
    for body in (c[1] for c in fake.calls if c[0] == "configs.update"):
        assert (body.get("intelligenceConfigurationIds") or []) == ids


def test_async_creates_are_found_by_name(tmp_path: Path) -> None:
    fake = FakeTwilio(async_create=True)
    assert run_apply(tmp_path, fake) == 0
    out = json.loads((tmp_path / "twilio.json").read_text())
    assert out["memoryStoreId"] in fake.stores and out["conversationConfigurationId"] == fake.configs[0].id


@pytest.mark.parametrize("fake, msg", [
    (FakeTwilio(owner=SUB), "main account"),
    (FakeTwilio(status="suspended"), "not active"),
    (FakeTwilio(workspace=False), "Flex"),
    (FakeTwilio(numbers=0), "found 0"),
])
def test_apply_refusals(tmp_path: Path, fake: FakeTwilio, msg: str,
                        capsys: pytest.CaptureFixture[str]) -> None:
    assert run_apply(tmp_path, fake) == 3
    assert msg in capsys.readouterr().err
    assert not (tmp_path / "twilio.json").exists()


def test_main_account_refused_before_any_write(tmp_path: Path) -> None:
    fake = FakeTwilio(owner=SUB)
    run_apply(tmp_path, fake)
    assert fake.writes() == [] and [c[0] for c in fake.calls] == ["account.fetch"]


def test_apply_needs_account_sid(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--apply"], sm_client=StubSM()) == 2
    assert "--account-sid" in capsys.readouterr().err


def test_twilio_exception_is_reported_without_details(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    class Boom(Exception):
        status, code = 401, 20003

    fake = FakeTwilio()

    def fail(phone_number: str) -> Any:
        raise Boom(f"https://{USER}:{PASSWORD}@x unauthorized")

    fake.incoming_phone_numbers.list = fail
    assert run_apply(tmp_path, fake) == 1
    err = capsys.readouterr().err
    assert "Boom (status 401, code 20003)" in err and PASSWORD not in err


# --------------------------------------------------------------------------- credentials never shown


creds = st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=6, max_size=30)


@settings(max_examples=100, deadline=None)
@given(user=creds, password=creds)
def test_property_cred_url_hides_credentials(user: str, password: str) -> None:
    url = ts.CredUrl("relay.x", "/webhook")
    c = ts.BasicCreds(user, password)
    # The printable forms are constants: they can't depend on the credentials.
    assert str(url) == repr(url) == "https://***:***@relay.x/webhook"
    assert str(c) == repr(c) == "BasicCreds(***)"
    from urllib.parse import unquote, urlsplit

    parts = urlsplit(url.render(c))
    assert parts.scheme == "https" and parts.hostname == "relay.x" and parts.path == "/webhook"
    assert (unquote(parts.username or ""), unquote(parts.password or "")) == (user, password)


def test_dedicated_main_account_accepted_with_flag(tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio(owner=SUB)
    assert run_apply(tmp_path, fake, ["--dedicated-main-account"]) == 0
    assert "main account, accepted as dedicated" in capsys.readouterr().out
    assert (tmp_path / "twilio.json").exists()


def test_dedicated_flag_does_not_bypass_inactive(tmp_path: Path) -> None:
    fake = FakeTwilio(owner=SUB, status="suspended")
    assert run_apply(tmp_path, fake, ["--dedicated-main-account"]) == 3


# --------------------------------------------------------------------------- Verify service

TEMPLATE = "HJ" + "ab" * 16


def written(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "twilio.json").read_text())


def test_dry_run_lists_the_verify_service_step_and_makes_no_calls(
        capsys: pytest.CaptureFixture[str]) -> None:
    def no_client(*a: Any) -> Any:
        raise AssertionError("Twilio client built in dry run")

    assert ts.main([*BASE, "--verify-friendly-name", "Acme Bank", "--verify-template-sid", TEMPLATE],
                   sm_client=NoSM(), client_factory=no_client) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if "ensure_verify_service" in ln)
    assert "Acme Bank" in line and '"code_length": 6' in line
    assert '"do_not_share_warning_enabled": true' in line and TEMPLATE in line


def test_verify_step_defaults() -> None:
    step, = (s for s in ts.build_plan(cfg()) if s.kind == "ensure_verify_service")
    assert step.name == "OwlBank"
    assert step.params["code_length"] == 6 and step.params["do_not_share_warning_enabled"] is True
    assert "lifetime" not in json.dumps(step.params).lower()  # Verify's default 10-minute validity


def test_apply_creates_the_verify_service_and_writes_its_sid(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    (svc,) = fake.verify_services
    (create,) = [kw for name, kw in fake.calls if name == "verify.create"]
    assert create == {"friendly_name": "OwlBank", "code_length": 6,
                      "do_not_share_warning_enabled": True}  # no template, no custom validity
    out = written(tmp_path)
    assert out["verifyServiceSid"] == svc.sid
    assert re.fullmatch(r"VA[0-9a-f]{32}", out["verifyServiceSid"])  # what infra's config accepts
    infra_config = (Path(__file__).resolve().parents[3] / "infra/owlbank_infra/config.py").read_text()
    assert 'twilio_outputs.get("verifyServiceSid")' in infra_config  # the key CDK reads


def test_verify_service_is_found_by_friendly_name_before_creating(tmp_path: Path) -> None:
    fake = FakeTwilio()
    other = SimpleNamespace(sid="VA" + "1" * 32, friendly_name="Something else", code_length=6,
                            do_not_share_warning_enabled=True, default_template_sid=None)
    mine = SimpleNamespace(sid="VA" + "2" * 32, friendly_name="Acme Bank", code_length=6,
                           do_not_share_warning_enabled=True, default_template_sid=None)
    fake.verify_services = [other, mine]
    assert run_apply(tmp_path, fake, ["--verify-friendly-name", "Acme Bank"]) == 0
    assert [n for n, _ in fake.writes() if n.startswith("verify")] == []
    assert written(tmp_path)["verifyServiceSid"] == mine.sid
    assert len(fake.verify_services) == 2


def test_other_friendly_name_creates_a_new_service(tmp_path: Path) -> None:
    fake = FakeTwilio()
    fake.verify_services = [SimpleNamespace(sid="VA" + "1" * 32, friendly_name="Something else",
                                            code_length=6, do_not_share_warning_enabled=True,
                                            default_template_sid=None)]
    assert run_apply(tmp_path, fake) == 0
    assert len(fake.verify_services) == 2 and fake.verify_services[1].friendly_name == "OwlBank"
    assert written(tmp_path)["verifyServiceSid"] == fake.verify_services[1].sid


def test_existing_service_with_the_warning_off_or_wrong_code_length_is_updated(tmp_path: Path) -> None:
    fake = FakeTwilio()
    svc = SimpleNamespace(sid="VA" + "3" * 32, friendly_name="OwlBank", code_length=4,
                          do_not_share_warning_enabled=False, default_template_sid="HJ" + "9" * 32)
    fake.verify_services = [svc]
    assert run_apply(tmp_path, fake) == 0
    (update,) = [kw for n, kw in fake.calls if n == "verify.update"]
    assert update == {"code_length": 6, "do_not_share_warning_enabled": True}
    assert not [n for n, _ in fake.calls if n == "verify.create"]
    assert svc.default_template_sid == "HJ" + "9" * 32  # no --verify-template-sid: left alone
    fake.calls.clear()
    assert run_apply(tmp_path, fake) == 0  # idempotent
    assert fake.writes() == []


def test_warning_flag_alone_triggers_the_update(tmp_path: Path) -> None:
    fake = FakeTwilio()
    fake.verify_services = [SimpleNamespace(sid="VA" + "4" * 32, friendly_name="OwlBank", code_length=6,
                                            do_not_share_warning_enabled=False,
                                            default_template_sid=None)]
    assert run_apply(tmp_path, fake) == 0
    assert [kw for n, kw in fake.calls if n == "verify.update"] == [
        {"do_not_share_warning_enabled": True}]


def test_template_sid_is_set_on_create_and_on_update_and_is_idempotent(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake, ["--verify-template-sid", TEMPLATE]) == 0
    (create,) = [kw for n, kw in fake.calls if n == "verify.create"]
    assert create["default_template_sid"] == TEMPLATE
    fake.calls.clear()
    assert run_apply(tmp_path, fake, ["--verify-template-sid", TEMPLATE]) == 0
    assert fake.writes() == []
    other = "HJ" + "cd" * 16
    assert run_apply(tmp_path, fake, ["--verify-template-sid", other]) == 0
    assert [kw for n, kw in fake.calls if n == "verify.update"] == [{"default_template_sid": other}]


@pytest.mark.parametrize("sid", ["", "HJ123", "HX" + "a" * 32, "HJ" + "g" * 32, "hj" + "a" * 32,
                                 TEMPLATE + "0"])
def test_template_sid_must_be_hj_plus_32_hex(sid: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--verify-template-sid", sid]) == 2
    assert "HJ + 32 hex" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["", "   ", "x" * 31])
def test_verify_friendly_name_is_bounded(name: str) -> None:
    with pytest.raises(ScriptError, match="verify-friendly-name"):
        cfg(verify_friendly_name=name)


# --------------------------------------------------------------------------- --create-flex

GO_SID = "GO" + "1" * 32
SECRET_BODY = "TOPSECRET-response-body-4242"


class FakeClock:
    """A clock that moves only when ``sleep`` is called, so nothing waits."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FlexFake(FakeTwilio):
    """FakeTwilio without the Flex workspace, plus ``client.request`` for the Flex Instances API.

    ``statuses`` are the ``status`` values the GETs return in turn (the last one repeats); the
    workspace appears ``lag`` lookups after the first ``ready`` was served."""

    def __init__(self, *, statuses: Sequence[str] = ("creation-in-progress", "ready"),
                 post_status: int = 201, post_text: str | None = None, post_state: str =
                 "creation-in-progress", lag: int = 0, get_status: int = 200,
                 workspace: bool = False, ws_name: str = f"{ts.FLEX_WORKSPACE} for {GO_SID}",
                 early_after: int | None = None, **kw: Any) -> None:
        super().__init__(workspace=workspace, **kw)
        self.ws_name, self.early_after, self.lookups = ws_name, early_after, 0
        self.statuses, self.post_status, self.post_state = list(statuses), post_status, post_state
        self.post_text, self.lag, self.get_status = post_text, lag, get_status
        self.ready = False
        self.http: list[tuple[str, str, Any, Any]] = []

    def posts(self) -> list[tuple[str, str, Any, Any]]:
        return [h for h in self.http if h[0] == "POST"]

    def request(self, method: str, uri: str, params: Any = None, data: Any = None,
                headers: Any = None, **kw: Any) -> Any:
        if uri == ts.FLEX_CONFIGURATION_URL:
            return super().request(method, uri, params, data, headers, **kw)
        self.http.append((method, uri, data, headers))
        self.calls.append(("flex.request", method))
        if method == "POST":
            self.ready = self.post_state == "ready"
            text = self.post_text if self.post_text is not None else json.dumps(
                {"flex_instance_sid": GO_SID, "account_sid": SUB, "status": self.post_state})
            return SimpleNamespace(status_code=self.post_status, text=text)
        if self.get_status >= 400:
            return SimpleNamespace(status_code=self.get_status, text=f'{{"code": 20404, "message": "{SECRET_BODY}"}}')
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        self.ready = self.ready or status == "ready"
        return SimpleNamespace(status_code=200, text=json.dumps(
            {"flex_instance_sid": GO_SID, "account_sid": SUB, "status": status}))

    def _list_workspaces(self, friendly_name: str | None = None) -> list[Any]:
        self.lookups += 1
        early = self.early_after is not None and self.lookups > self.early_after
        if (self.ready or early) and not self.workspaces:
            if self.lag > 0 and not early:
                self.lag -= 1
            else:
                self.workspaces.append(SimpleNamespace(sid="WS" + "0" * 32, friendly_name=self.ws_name))
        return super()._list_workspaces(friendly_name)


def run_flex(tmp_path: Path, fake: FakeTwilio, clock: FakeClock, extra: list[str] | None = None) -> int:
    return ts.main([*apply_args(tmp_path), "--create-flex", *(extra or [])], sm_client=StubSM(),
                   client_factory=fake.factory, sleep=clock.sleep, clock=clock.now)


def flex_out(tmp_path: Path) -> dict[str, Any]:
    return dict(json.loads((tmp_path / "twilio.json").read_text()))


FLEX_KINDS = {"ensure_flex", "ensure_queue", "ensure_workflow", "ensure_flex_skills",
              "ensure_taskrouter_events"}
FLEX_QUEUES = {"support-en": "A", "support-es": "B", "fraud": "C", "disputes": "D"}


def test_create_flex_alone_leaves_the_plan_without_flex_steps() -> None:
    """Without ``--with-flex`` no plan has a Flex step; with it and without ``--create-flex`` no ensure_flex."""
    for phase in ("pre", "post", "all"):
        base = {"phase": phase, "phone_number": NUMBER, "api_domain": "a.example", "relay_domain": "r.example",
                "queue_names": FLEX_QUEUES}
        assert not FLEX_KINDS & {s.kind for s in ts.build_plan(ts.SetupConfig(**base))}
        assert "ensure_flex" not in [s.kind for s in ts.build_plan(ts.SetupConfig(with_flex=True, **base))]
    pre = [s.kind for s in ts.build_plan(ts.SetupConfig(phase="pre", with_flex=True, phone_number=NUMBER,
                                                         api_domain="a.example", relay_domain="r.example",
                                                         queue_names=FLEX_QUEUES))]
    assert pre == ["check_account", "ensure_queue", "ensure_queue", "ensure_queue", "ensure_queue",
                   "ensure_workflow", "ensure_flex_skills", "ensure_memory_store", "ensure_conversation_configuration",
                   "ensure_verify_service", "ensure_trait_group"]
    default_pre = [s.kind for s in ts.build_plan(ts.SetupConfig(phase="pre", phone_number=NUMBER,
                                                                 api_domain="a.example", relay_domain="r.example",
                                                                 queue_names=FLEX_QUEUES))]
    assert default_pre == ["check_account", "ensure_memory_store", "ensure_conversation_configuration",
                           "ensure_verify_service", "ensure_trait_group"]


@pytest.mark.parametrize("phase", ["pre", "all"])
def test_flag_puts_ensure_flex_right_after_check_account(phase: str) -> None:
    cfg = ts.SetupConfig(phase=phase, phone_number=NUMBER, api_domain="a.example", relay_domain="r.example",
                         queue_names=FLEX_QUEUES, with_flex=True, create_flex=True)
    kinds = [s.kind for s in ts.build_plan(cfg)]
    assert kinds[:2] == ["check_account", "ensure_flex"] and kinds.count("ensure_flex") == 1
    without = ts.SetupConfig(phase=phase, phone_number=NUMBER, api_domain="a.example",
                             relay_domain="r.example", queue_names=FLEX_QUEUES, with_flex=True)
    assert [k for k in kinds if k != "ensure_flex"] == [s.kind for s in ts.build_plan(without)]


@pytest.mark.parametrize("flag", ["--create-flex", "--taskrouter-events"])
def test_flex_sub_flags_need_with_flex(flag: str, capsys: pytest.CaptureFixture[str]) -> None:
    phase = ["--phase", "post"] if flag == "--taskrouter-events" else []
    assert ts.main([*BASE, *phase, flag]) == 2
    err = capsys.readouterr().err
    assert flag in err and "--with-flex" in err
    # And with the opt-in the same flag is accepted.
    assert ts.main([*BASE, *phase, flag, "--with-flex"]) == 0


@settings(max_examples=60, deadline=None)
@given(phase=st.sampled_from(["pre", "post", "all"]), create=st.booleans(), events=st.booleans(),
       with_flex=st.booleans())
def test_p1_without_with_flex_no_flex_step_and_no_flex_key(phase: str, create: bool, events: bool,
                                                           with_flex: bool) -> None:
    """Any combination without ``--with-flex`` has no Flex-workspace step and writes no Flex
    key; the sub-flags without it are refused."""
    kw: dict[str, Any] = {"phase": phase, "with_flex": with_flex, "create_flex": create, "taskrouter_events": events}
    refused = ((create and (not with_flex or phase == "post")) or (events and (not with_flex or phase == "pre")))
    if refused:
        with pytest.raises(ts.ScriptError):
            cfg(**kw)
        return
    config = cfg(**kw)
    kinds = {s.kind for s in ts.build_plan(config)}
    if not with_flex:
        assert not FLEX_KINDS & kinds
        assert set(ts.Applier(FakeTwilio(), config, None).out) == {"accountSid"}
    else:
        assert ({"ensure_queue", "ensure_workflow", "ensure_flex_skills"} <= kinds) is (phase != "post")


def test_flag_with_phase_post_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--phase", "post", "--with-flex", "--create-flex"]) == 2
    assert "--create-flex applies to --phase pre or all" in capsys.readouterr().err


@pytest.mark.parametrize("minutes", ["0", "-3", str(ts.MAX_FLEX_TIMEOUT_MIN + 1)])
def test_flex_timeout_is_bounded(minutes: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--with-flex", "--create-flex", "--flex-timeout-min", minutes]) == 2
    assert "--flex-timeout-min" in capsys.readouterr().err


def test_workspace_exists_means_no_post_and_no_flex_call(tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    fake = FlexFake(workspace=True)
    clock = FakeClock()
    assert run_flex(tmp_path, fake, clock) == 0
    assert fake.http == [] and clock.sleeps == []
    assert "Flex already provisioned" in capsys.readouterr().out
    assert "flexInstanceSid" not in flex_out(tmp_path)  # not known: nothing was created or fetched
    assert len(fake.queues) == 4


def test_creates_polls_to_ready_then_finds_the_workspace(tmp_path: Path) -> None:
    fake = FlexFake(statuses=("creation-in-progress", "creation-in-progress", "ready"), lag=2)
    clock = FakeClock()
    assert run_flex(tmp_path, fake, clock) == 0
    (post,) = fake.posts()
    assert post[1] == "https://flex-api.twilio.com/v2/Instances"
    assert post[2] == {"conversations": {"default": False}}
    assert post[3] == {"Content-Type": "application/json"}
    gets = [h for h in fake.http if h[0] == "GET"]
    assert [g[1] for g in gets] == [f"https://flex-api.twilio.com/v2/Instances/{GO_SID}"] * 3
    assert clock.sleeps == [10] * 5  # three polls, then two workspace re-checks
    assert fake.workspaces and len(fake.queues) == 4
    assert flex_out(tmp_path)["flexInstanceSid"] == GO_SID
    first_flex = next(i for i, c in enumerate(fake.calls) if c[0] == "flex.request")
    assert next(i for i, c in enumerate(fake.calls) if c[0] == "queues.create") > first_flex


def test_created_already_ready_skips_polling(tmp_path: Path) -> None:
    fake = FlexFake(post_state="ready", statuses=("ready",))
    clock = FakeClock()
    assert run_flex(tmp_path, fake, clock) == 0
    assert [h[0] for h in fake.http] == ["POST"] and clock.sleeps == []


def test_second_run_with_the_flag_makes_no_post(tmp_path: Path) -> None:
    fake = FlexFake()
    assert run_flex(tmp_path, fake, FakeClock()) == 0
    fake.http.clear()
    fake.calls.clear()
    assert run_flex(tmp_path, fake, FakeClock()) == 0
    assert fake.http == [] and fake.writes() == []
    assert flex_out(tmp_path)["flexInstanceSid"] == GO_SID  # kept from the first run


def test_http_error_logs_status_and_code_never_the_body(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                        caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    fake = FlexFake(post_status=403, post_text=json.dumps({"code": 20003, "message": SECRET_BODY}))
    assert run_flex(tmp_path, fake, FakeClock()) == 3
    err = capsys.readouterr().err
    assert "HTTP 403" in err and "20003" in err and "Console" in err and "--create-flex" in err
    assert SECRET_BODY not in err
    assert SECRET_BODY not in "\n".join(str(r.__dict__) for r in caplog.records)
    flat = [r.__dict__ for r in caplog.records]
    assert any(d.get("http_status") == 403 and d.get("error_code") == "20003" for d in flat)
    assert fake.queues == [] and not (tmp_path / "twilio.json").exists()


def test_http_error_with_an_unreadable_body_still_reports_the_status(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FlexFake(post_status=500, post_text=f"<html>{SECRET_BODY}</html>")
    assert run_flex(tmp_path, fake, FakeClock()) == 3
    err = capsys.readouterr().err
    assert "HTTP 500" in err and SECRET_BODY not in err


def test_timeout_stops_clearly_and_the_rerun_resumes_the_same_instance(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FlexFake(statuses=("creation-in-progress",))
    clock = FakeClock()
    assert run_flex(tmp_path, fake, clock, ["--flex-timeout-min", "1"]) == 3
    err = capsys.readouterr().err
    assert "not ready after 1 minutes" in err and "re-run" in err and GO_SID in err
    assert clock.sleeps == [10] * 6 and len(fake.posts()) == 1
    assert fake.queues == []
    # the SID was kept (and nothing else written), so the re-run resumes it: no second POST
    assert json.loads((tmp_path / "twilio.json").read_text()) == {"flexInstanceSid": GO_SID}
    fake.statuses = ["creation-in-progress", "ready"]
    assert run_flex(tmp_path, fake, FakeClock()) == 0
    assert len(fake.posts()) == 1 and len(fake.queues) == 4
    assert flex_out(tmp_path)["flexInstanceSid"] == GO_SID


@pytest.mark.parametrize("where", ["post", "poll"])
@pytest.mark.parametrize("state", ["failed", "deleted", ""])
def test_unexpected_status_stops_with_the_status_shown(where: str, state: str, tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    fake = (FlexFake(post_state=state) if where == "post"
            else FlexFake(statuses=("creation-in-progress", state)))
    assert run_flex(tmp_path, fake, FakeClock()) == 3
    err = capsys.readouterr().err
    assert f"has status {state!r}" in err and "Console" in err
    assert fake.queues == []


def test_ready_but_workspace_never_appears_is_a_bounded_error(tmp_path: Path,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    fake = FlexFake(statuses=("ready",), lag=10**6)
    clock = FakeClock()
    assert run_flex(tmp_path, fake, clock) == 3
    assert "isn't visible yet" in capsys.readouterr().err
    assert clock.sleeps == [10] * (1 + ts.FLEX_WORKSPACE_TRIES - 1) and fake.queues == []  # 1 poll wait


def test_response_without_a_valid_instance_sid_is_refused(tmp_path: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    fake = FlexFake(post_text=json.dumps({"flex_instance_sid": "../x", "status": "ready"}))
    assert run_flex(tmp_path, fake, FakeClock()) == 3
    assert "valid flex_instance_sid" in capsys.readouterr().err


def test_a_stale_kept_sid_that_is_gone_creates_a_new_instance(tmp_path: Path) -> None:
    (tmp_path / "twilio.json").write_text(json.dumps({"flexInstanceSid": "GO" + "9" * 32}))
    fake = FlexFake(get_status=404)
    fake.get_status = 404
    # the first GET (the kept SID) is a 404; then the POST works and polling GETs must succeed
    real = fake.request

    def request(method: str, uri: str, **kw: Any) -> Any:
        if method == "GET" and GO_SID in uri:
            fake.get_status = 200
        return real(method, uri, **kw)

    fake.request = request  # type: ignore[method-assign]
    assert run_flex(tmp_path, fake, FakeClock()) == 0
    assert len(fake.posts()) == 1 and flex_out(tmp_path)["flexInstanceSid"] == GO_SID


def test_dry_run_with_the_flag_makes_zero_calls_and_shows_the_flex_steps(
        capsys: pytest.CaptureFixture[str]) -> None:
    def no_client(*a: Any) -> Any:
        raise AssertionError("Twilio client built in dry run")

    assert ts.main([*BASE, "--with-flex", "--create-flex"], sm_client=NoSM(), client_factory=no_client) == 0
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN")
    assert "ensure_flex" in out and "Flex already provisioned" in out
    assert "POST https://flex-api.twilio.com/v2/Instances" in out and r'\"default\": false' in out
    assert "GET https://flex-api.twilio.com/v2/Instances/<flex_instance_sid> every 10 s" in out
    assert "up to 15 min" in out and "billable" in out
    assert ts.main([*BASE, "--with-flex", "--create-flex", "--flex-timeout-min", "30"], sm_client=NoSM(),
                   client_factory=no_client) == 0
    assert "up to 30 min" in capsys.readouterr().out
    assert ts.main([*BASE, "--with-flex"], sm_client=NoSM(), client_factory=no_client) == 0
    assert "ensure_flex:" not in capsys.readouterr().out


def test_flex_step_is_logged_like_the_others(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    assert run_flex(tmp_path, FlexFake(), FakeClock()) == 0
    flat = [r.__dict__ for r in caplog.records]
    assert any(d.get("event") == "script_step" and d.get("step") == "ensure_flex"
               and d.get("outcome") == "ok" for d in flat)
    ops = {d.get("operation") for d in flat if d.get("event") == "dependency_call"}
    assert {"flex_instance_create", "flex_instance_fetch"} <= ops


# --------------------------------------------------------------------------- the workspace resolver


def ws(name: str, n: int) -> SimpleNamespace:
    return SimpleNamespace(sid=f"WS{n:032d}", friendly_name=name)


def run_resolver(tmp_path: Path, workspaces: list[SimpleNamespace], *, known: str | None = None,
                 extra: list[str] | None = None) -> tuple[int, FlexFake]:
    """Flag on, the workspaces as given, an optional kept instance SID in the output file."""
    if known:
        (tmp_path / "twilio.json").write_text(json.dumps({"flexInstanceSid": known}))
    fake = FlexFake()
    fake.workspaces = list(workspaces)
    return run_flex(tmp_path, fake, FakeClock(), extra), fake


def test_resolver_a_known_instance_sid_picks_the_api_named_workspace_over_the_exact_name(
        tmp_path: Path) -> None:
    mine, other = ws(f"{ts.FLEX_WORKSPACE} for {GO_SID}", 1), ws(ts.FLEX_WORKSPACE, 2)
    rc, fake = run_resolver(tmp_path, [other, mine], known=GO_SID)
    assert rc == 0 and fake.http == []
    out = flex_out(tmp_path)
    assert out["workspaceSid"] == mine.sid and out["flexInstanceSid"] == GO_SID
    assert out["queueSids"] and len(fake.queues) == 4  # the later steps used the same workspace


def test_resolver_b_exact_console_name(tmp_path: Path) -> None:
    rc, fake = run_resolver(tmp_path, [ws("Other", 1), ws(ts.FLEX_WORKSPACE, 2)])
    assert rc == 0 and fake.http == []
    assert flex_out(tmp_path)["workspaceSid"] == ws("", 2).sid


@pytest.mark.parametrize("known", [None, "GO" + "9" * 32])
def test_resolver_c_single_prefix_match(known: str | None, tmp_path: Path) -> None:
    only = ws(f"{ts.FLEX_WORKSPACE} for GO{'7' * 32}", 3)
    rc, fake = run_resolver(tmp_path, [ws("Other", 1), only], known=known)
    assert rc == 0 and fake.http == []  # a stale kept SID doesn't matter: no POST, no GET
    assert flex_out(tmp_path)["workspaceSid"] == only.sid


def test_resolver_ambiguous_prefix_stops_listing_names_and_sids_and_never_guesses(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = ws(f"{ts.FLEX_WORKSPACE} for GO{'7' * 32}", 3), ws(f"{ts.FLEX_WORKSPACE} for GO{'8' * 32}", 4)
    rc, fake = run_resolver(tmp_path, [a, b])
    assert rc == 3 and fake.http == [] and fake.queues == []
    err = capsys.readouterr().err
    for w in (a, b):
        assert w.friendly_name in err and w.sid in err
    assert not (tmp_path / "twilio.json").exists()


def test_resolver_ambiguity_is_also_an_error_without_the_flag(tmp_path: Path) -> None:
    fake = FlexFake()
    fake.workspaces = [ws(f"{ts.FLEX_WORKSPACE} A", 1), ws(f"{ts.FLEX_WORKSPACE} B", 2)]
    assert run_apply(tmp_path, fake) == 3 and fake.queues == []


def test_without_a_flex_workspace_the_queue_step_still_says_set_up_flex(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FlexFake()
    fake.workspaces = [ws("Unrelated", 1)]
    assert run_apply(tmp_path, fake) == 3
    assert "set up Flex first" in capsys.readouterr().err


def test_created_instance_with_the_api_style_workspace_name_works_end_to_end(tmp_path: Path) -> None:
    fake = FlexFake(lag=1)
    assert run_flex(tmp_path, fake, FakeClock()) == 0
    out = flex_out(tmp_path)
    assert fake.workspaces[0].friendly_name == f"{ts.FLEX_WORKSPACE} for {GO_SID}"
    assert out["workspaceSid"] == fake.workspaces[0].sid and len(fake.queues) == 4
    assert out["flexInstanceSid"] == GO_SID


def test_ensure_flex_stops_waiting_as_soon_as_the_resolver_finds_the_workspace(tmp_path: Path) -> None:
    fake = FlexFake(statuses=("creation-in-progress",), early_after=3)
    clock = FakeClock()
    assert run_flex(tmp_path, fake, clock, ["--flex-timeout-min", "1"]) == 0
    assert len(fake.posts()) == 1
    assert 0 < len(clock.sleeps) < 6 and clock.t < 60  # nowhere near the 1-minute timeout
    assert len(fake.queues) == 4 and flex_out(tmp_path)["flexInstanceSid"] == GO_SID


def test_dry_run_names_the_workspace_generically(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--with-flex", "--create-flex"]) == 0
    out = capsys.readouterr().out
    assert "the Flex TaskRouter workspace" in out and "Flex Task Assignment" not in out


# --------------------------------------------------------------------------- timeouts: inactive < closed


def test_default_timeouts_pass_and_closed_exceeds_inactive(capsys: pytest.CaptureFixture[str]) -> None:
    assert (ts.DEFAULT_INACTIVE_MIN, ts.DEFAULT_CLOSED_MIN) == (15, 30)
    assert ts.DEFAULT_INACTIVE_MIN > ts.LONGEST_DETERMINISTIC_SEGMENT_MIN
    plan = ts.build_plan(cfg())
    step = next(s for s in plan if s.kind == "ensure_conversation_configuration")
    assert step.params["body"]["channelSettings"]["VOICE"]["statusTimeouts"] == {"inactive": 15, "closed": 30}
    assert ts.main(BASE) == 0
    assert '"closed": 30' in capsys.readouterr().out


@pytest.mark.parametrize("closed", [1, 14, 15])
def test_closed_must_exceed_inactive(closed: int) -> None:
    with pytest.raises(ScriptError, match=r"must be greater than the inactive timeout.*less than closed"):
        cfg(closed_timeout_min=closed)


def test_closed_one_above_inactive_is_the_boundary_that_passes() -> None:
    assert cfg(inactive_timeout_min=20, closed_timeout_min=21).closed_timeout_min == 21


def test_cli_refuses_inverted_timeouts(capsys: pytest.CaptureFixture[str]) -> None:
    assert ts.main([*BASE, "--inactive-timeout-min", "40"]) == 2  # default closed 30 < 40
    assert "must be greater than the inactive timeout" in capsys.readouterr().err


# --------------------------------------------------------------------------- Twilio 4xx messages


def twilio_400(msg: str, code: int | None = 400) -> Exception:
    from twilio.base.exceptions import TwilioRestException

    return TwilioRestException(400, f"https://{USER}:{PASSWORD}@conversations.twilio.com/v2/Configurations",
                               msg, code)


def test_twilio_4xx_message_reaches_the_error_and_the_log(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                          caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    fake = FakeTwilio()

    def reject(**kw: Any) -> Any:
        raise twilio_400("channelSettings[VOICE].statusTimeouts inactive timeout must be less than "
                         "closed timeout")

    fake.conversations.v2.configurations.create = reject
    assert run_apply(tmp_path, fake) == 3
    err = capsys.readouterr().err
    assert ("Twilio rejected ensure_conversation_configuration (HTTP 400, code 400): channelSettings"
            "[VOICE].statusTimeouts inactive timeout must be less than closed timeout") in err
    assert PASSWORD not in err and "@conversations" not in err  # the exception's URL is not used
    logged = [r.__dict__ for r in caplog.records if r.__dict__.get("event") == "script_error"]
    assert logged and "must be less than closed timeout" in logged[0]["error_message"]
    assert logged[0]["http_status"] == 400 and logged[0]["error_code"] == "400"


def test_twilio_4xx_message_is_scrubbed_and_truncated(tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    secret = "sEcReTtOkEn_" + "x" * 30
    fake = FakeTwilio()

    def reject(**kw: Any) -> Any:
        raise twilio_400(f"bad value {secret} for +12065550123 " + "long " * 200, code=None)

    fake.conversations.v2.configurations.create = reject
    assert run_apply(tmp_path, fake) == 3
    err = capsys.readouterr().err.strip()
    assert secret not in err and "+12065550123" not in err and "[token]" in err
    detail = err.split("code None): ", 1)[1]
    assert len(detail) <= ts.TWILIO_MESSAGE_MAX + len("...") and detail.endswith("...")


def test_a_non_4xx_or_message_less_exception_keeps_the_old_report(tmp_path: Path,
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    from twilio.base.exceptions import TwilioRestException

    fake = FakeTwilio()

    def boom(**kw: Any) -> Any:
        raise TwilioRestException(503, "https://x", "upstream " + "m" * 5, 20500)

    fake.conversations.v2.configurations.create = boom
    assert run_apply(tmp_path, fake) == 1
    assert "TwilioRestException (status 503, code 20500)" in capsys.readouterr().err


def test_verify_rereads_a_configuration_that_lags_after_the_update(tmp_path: Path) -> None:
    """Right after an update the read-back can still show the previous statusCallbacks;
    verify reads again (sleeping between reads) and passes once the new value shows."""
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    fake.configs[0].status_callbacks = []
    real_list = fake.conversations.v2.configurations.list
    stale = {"reads": 0}

    def lagging_list(*a: Any, **kw: Any) -> list[SimpleNamespace]:
        out = real_list(*a, **kw)
        in_verify = any(f.function == "verify_setup" for f in inspect.stack()[1:6])
        if in_verify and stale["reads"] < 2:
            stale["reads"] += 1
            return [SimpleNamespace(**{**vars(c), "status_callbacks": []}) for c in out]
        return out

    fake.conversations.v2.configurations.list = lagging_list
    slept: list[float] = []
    rc = ts.main([*apply_args(tmp_path), "--phase", "post"], sm_client=StubSM(), client_factory=fake.factory,
                 sleep=slept.append)
    assert rc == 0
    assert stale["reads"] == 2 and slept == [ts.VERIFY_READ_WAIT_S] * 2


# --------------------------------------------------------------------------- Flex skills


def _skill_entry(name: str) -> dict[str, Any]:
    return {"name": name, "multivalue": False, "minimum": None, "maximum": None}


def test_agent_skills_match_the_queue_targets_exactly() -> None:
    used = []
    for *_, target in ts.QUEUES:
        match = re.fullmatch(r'routing\.skills HAS "([^"]+)"', target)
        assert match is not None
        used.append(match.group(1))
    assert list(ts.AGENT_SKILLS) == used


@pytest.mark.parametrize("phase", ["pre", "all"])
def test_skills_step_comes_after_the_workflow_and_before_the_memory_store(phase: str) -> None:
    kinds = [s.kind for s in ts.build_plan(cfg(phase=phase))]
    assert kinds.index("ensure_workflow") + 1 == kinds.index("ensure_flex_skills")
    assert kinds.index("ensure_flex_skills") + 1 == kinds.index("ensure_memory_store")


def test_dry_run_plan_says_it_adds_names_only_and_assigns_nothing() -> None:
    step = next(s for s in ts.build_plan(cfg()) if s.kind == "ensure_flex_skills")
    assert "taskrouter_skills" in step.params["define"]
    assert step.params["assign"].startswith("nothing")


def test_apply_adds_missing_names_keeps_existing_and_touches_no_worker(tmp_path: Path) -> None:
    fake = FakeTwilio()
    fake.flex_skills = [_skill_entry("billing"), _skill_entry("en")]
    fake.workers = [SimpleNamespace(sid="WK" + "1" * 32, friendly_name="a", attributes="{}")]
    assert run_apply(tmp_path, fake) == 0
    assert [e["name"] for e in fake.flex_skills] == ["billing", "en", "es", "fraud", "disputes"]
    assert [m for m, _ in fake.config_http] == ["GET", "POST"]
    assert fake.config_http[1][1]["account_sid"] == SUB
    assert not [c for c in fake.calls if c[0].startswith("workers.")]


def test_second_run_posts_nothing(tmp_path: Path) -> None:
    fake = FakeTwilio()
    assert run_apply(tmp_path, fake) == 0
    fake.config_http.clear()
    assert run_apply(tmp_path, fake) == 0
    assert [m for m, _ in fake.config_http] == ["GET"]


def test_a_flex_configuration_http_error_is_a_clean_script_error_without_the_body(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeTwilio()
    fake.config_status = 403
    assert run_apply(tmp_path, fake) == 3
    err = capsys.readouterr().err
    assert "HTTP 403" in err and "20003" in err and SECRET_BODY not in err
    assert "--create-flex" not in err
    assert issubclass(ts.FlexApiError, ScriptError)
    assert not [c for c in fake.calls if c[0] == "stores.create"]  # stops: nothing after it runs


# --------------------------------------------------------------------------- the account check's client


@pytest.mark.parametrize("phase", ["pre", "post", "all"])
def test_the_auth_token_secret_missing_changes_nothing(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], phase: str) -> None:
    """No Auth Token secret (no ARN, or only the placeholder): a clear error before any Twilio call."""
    fake = FakeTwilio()
    base = [*apply_args(tmp_path), "--phase", phase]
    no_token = [a for a in base if a != f"twilio_auth_token={TOKEN_ARN}"]
    del no_token[no_token.index("--secret", no_token.index(f"basic_auth={BASIC_ARN}") + 1)]  # its flag
    assert ts.main(no_token, sm_client=StubSM(), client_factory=fake.factory) == 2
    assert "no ARN for secret twilio_auth_token" in capsys.readouterr().err
    placeholder = StubSM({**SECRET_VALUES, TOKEN_ARN: {"placeholder": True, "unset": "x" * 20}})
    assert ts.main(base, sm_client=placeholder, client_factory=fake.factory) == 3
    err = capsys.readouterr().err
    assert "twilio_auth_token" in err and AUTH_TOKEN not in err
    assert fake.calls == [] and fake.refused == []


@pytest.mark.parametrize("phase", ["pre", "post", "all"])
def test_the_account_check_runs_on_the_auth_token_client_and_a_standard_key_is_refused(
        tmp_path: Path, phase: str) -> None:
    fake = FakeTwilio()
    if phase == "post":
        assert run_apply(tmp_path, fake, ["--phase", "pre"]) == 0
    assert run_apply(tmp_path, fake, ["--phase", phase]) == 0
    assert ("account.fetch", SUB) in fake.calls and fake.refused == []
    with pytest.raises(TwilioRestException) as raised:  # the fake refuses the Standard key like Twilio
        fake.factory(API_SID, API_SECRET, SUB).api.v2010.accounts(SUB).fetch()
    assert raised.value.code == 70004


@settings(max_examples=40, deadline=None)
@given(phase=st.sampled_from(["pre", "post", "all"]), with_flex=st.booleans(),
       dedicated=st.booleans(), owner=st.sampled_from([PARENT, SUB]))
def test_check_account_never_makes_a_call_a_standard_key_is_refused(
        phase: str, with_flex: bool, dedicated: bool, owner: str) -> None:
    """For any phase and flag combination, a check on the account client makes no call the
    Standard-key fake refuses (Twilio answers 70004 to a Standard key on the Accounts resource)."""
    fake = FakeTwilio(owner=owner)
    cfg = ts.SetupConfig(account_sid=SUB, phase=phase, with_flex=with_flex, phone_number=NUMBER,
                         api_domain="api.owlbank.example", relay_domain="relay.owlbank.example",
                         dedicated_main_account=dedicated, queue_names={k: k for k, *_ in ts.QUEUES})
    applier = ts.Applier(fake.factory(API_SID, API_SECRET, SUB), cfg, None, log=lambda _m: None,
                         account_client=fake.factory(SUB, AUTH_TOKEN, SUB))
    step = next(s for s in ts.build_plan(cfg) if s.kind == "check_account")
    with contextlib.suppress(ScriptError):  # a main account without --dedicated-main-account
        applier.check_account(step)
    assert fake.refused == [] and fake.calls == [("account.fetch", SUB)]


def test_check_account_without_the_auth_token_client_is_refused_not_run_on_the_standard_key() -> None:
    fake = FakeTwilio()
    cfg = ts.SetupConfig(account_sid=SUB, phase="pre", phone_number=NUMBER, api_domain="api.owlbank.example",
                         relay_domain="relay.owlbank.example", queue_names={k: k for k, *_ in ts.QUEUES})
    applier = ts.Applier(fake.factory(API_SID, API_SECRET, SUB), cfg, None, log=lambda _m: None)
    step = next(s for s in ts.build_plan(cfg) if s.kind == "check_account")
    with pytest.raises(ScriptError, match="account_client"):
        applier.check_account(step)
    assert fake.calls == [] and fake.refused == []
