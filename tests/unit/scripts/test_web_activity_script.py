"""scripts/web_activity.py against a stateful Memory stub: the plan,
``--apply`` (PATCH on an existing profile, upsert on a missing one), a second run writing nothing,
``--clear``, every ``--check`` reason and the recall text, the read-back wait and its timeout, a 308
re-lookup, the prod and non-fixture refusals, and no credential in any output. No network. The properties are in
tests/property/test_web_activity_script_props.py."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from owlbank.web_activity import GROUP, TRAITS
from scripts import web_activity as wa

STORE = "mem_store_" + "1" * 26
ACCOUNT = "AC" + "3" * 32
API_SID = "SK" + "c4" * 16
API_SECRET = "apisecretMEMORY9876543210abcdefg"
EMAIL = "thomas.doe@owlbank-demo.example"  # C-1001's memory_email in data/fixtures.json
NOW = datetime(2026, 10, 28, 15, 0, 0, tzinfo=UTC)
NOW_TEXT = "2026-10-28T15:00:00Z"


ARN = "arn:aws:secretsmanager:us-east-1:111122223333:secret:owlbank/twilio-api-key-AbCdEf"


class SecretsStub:
    """A Secrets Manager client holding the Twilio API key secret (no AWS call)."""

    def __init__(self, fields: dict[str, str] | None = None) -> None:
        self.calls: list[str] = []
        self.fields = fields if fields is not None else {
            "account_sid": ACCOUNT, "api_key": API_SID, "api_secret": API_SECRET}

    def get_secret_value(self, SecretId: str) -> dict[str, str]:  # noqa: N803 (boto3's name)
        self.calls.append(SecretId)
        return {"SecretString": json.dumps(self.fields)}


def sm_client() -> SecretsStub:
    return SecretsStub()


def _resp(status: int, doc: Any) -> Any:
    return SimpleNamespace(status_code=status, text=json.dumps(doc))


class ProfileStub:
    """A Twilio client whose HTTP layer answers the Memory calls of the script, with state.

    ``profiles``: profile ID -> traits by group; ``emails``: email -> profile IDs (two IDs model two
    profiles for one email). ``lag``: profile reads before a queued write shows (Memory queues trait
    writes). ``merged``: profile ID -> the ID it was merged into (reads and PATCHes of the old ID
    answer 308; the lookup already names the new one). ``drop_empty``: a group whose traits are all
    removed disappears (else it reads as ``{}``). ``create_lag``: reads of a profile the upsert just
    created that answer 404 (``hidden``: profile ID -> 404 reads left)."""

    def __init__(self, *, group: bool = True, lag: int = 0, drop_empty: bool = True, create_lag: int = 0) -> None:
        self.group, self.lag, self.drop_empty, self.create_lag = group, lag, drop_empty, create_lag
        self.hidden: dict[str, int] = {}
        self.profiles: dict[str, dict[str, dict[str, Any]]] = {}
        self.emails: dict[str, list[str]] = {}
        self.merged: dict[str, str] = {}
        self.pending: list[list[Any]] = []  # [reads left, profile ID, traits]
        self.log: list[tuple[str, str, Any]] = []
        self.auths: set[Any] = set()
        self.next_id = 1
        self.username, self.password = API_SID, API_SECRET
        self.http_client = SimpleNamespace(request=self.request)

    # -- state helpers for tests

    def add(self, email: str = EMAIL, traits: dict[str, dict[str, Any]] | None = None) -> str:
        pid = f"mem_profile_{self.next_id:026d}"
        self.next_id += 1
        self.profiles[pid] = {"Contact": {"email": email}, **(traits or {})}
        self.emails.setdefault(email, []).append(pid)
        return pid

    def writes(self) -> list[tuple[str, str, Any]]:
        return [c for c in self.log if c[0] in ("PATCH", "POST") and not c[1].endswith("/Lookup")]

    def settled(self, pid: str) -> dict[str, dict[str, Any]]:
        """The profile once every queued write has landed."""
        while self.pending:
            self._land(self.pending.pop(0))
        return self.profiles[pid]

    # -- Memory

    def _merge(self, pid: str, traits: dict[str, dict[str, Any]]) -> None:
        if self.lag:
            self.pending.append([self.lag, pid, traits])
        else:
            self._land([0, pid, traits])

    def _land(self, item: list[Any]) -> None:
        _, pid, traits = item
        profile = self.profiles[pid]
        for group, values in traits.items():
            cur = profile.setdefault(group, {})
            for k, v in values.items():
                if v is None:
                    cur.pop(k, None)
                else:
                    cur[k] = v
            if not cur and self.drop_empty:
                profile.pop(group)

    def _tick(self) -> None:
        for item in list(self.pending):
            item[0] -= 1
            if item[0] <= 0:
                self.pending.remove(item)
                self._land(item)

    def request(self, method: str, url: str, params: Any = None, data: Any = None, headers: Any = None,
                auth: Any = None, timeout: Any = None, allow_redirects: bool = False) -> Any:
        self.log.append((method, url, data))
        self.auths.add(auth)
        assert not allow_redirects and timeout
        path = unquote(urlsplit(url).path)
        prefix = f"/v1/Stores/{STORE}/Profiles"
        if path.endswith(f"/TraitGroups/{GROUP}") and method == "GET":
            return _resp(200, {"displayName": GROUP, "traits": {}}) if self.group else _resp(404, {"code": 520047})
        if method == "POST" and path.endswith("/TraitGroups"):
            self.group = True
            return _resp(202, {"statusUrl": None})
        if method == "POST" and path == prefix + "/Lookup":
            return _resp(200, {"normalizedValue": data["value"], "profiles": self.emails.get(data["value"], [])})
        if method == "POST" and path == prefix:
            email = data["traits"]["Contact"]["email"]
            pid = (self.emails.get(email) or [None])[0]
            if pid is None:
                pid = self.add(email)
                if self.create_lag:
                    self.hidden[pid] = self.create_lag
            self._merge(pid, {g: t for g, t in data["traits"].items() if g != "Contact"})
            return _resp(202, {"id": pid})
        if path.startswith(prefix + "/") and method in ("GET", "PATCH"):
            pid = path.rsplit("/", 1)[1]
            if pid in self.merged:
                return _resp(308, {"code": 520308, "message": "merged"})
            if pid not in self.profiles:
                return _resp(404, {"code": 20404})
            if self.hidden.get(pid, 0) > 0:
                self.hidden[pid] -= 1
                return _resp(404, {"code": 20404})
            if method == "PATCH":
                self._merge(pid, data["traits"])
                return _resp(202, {"message": "accepted for processing"})
            self._tick()
            return _resp(200, {"id": pid, "createdAt": "2026-10-01T00:00:00Z", "traits": self.profiles[pid]})
        raise AssertionError(f"unexpected {method} {url}")


class Clock:
    def __init__(self) -> None:
        self.t, self.sleeps = 0.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def setup_file(tmp_path: Path) -> Path:
    path = tmp_path / "twilio.context.json"
    path.write_text(json.dumps({"memoryStoreId": STORE, "accountSid": ACCOUNT}), encoding="utf-8")
    return path


def run(tmp_path: Path, stub: ProfileStub, *argv: str, clock: Clock | None = None,
        now: datetime = NOW) -> int:
    clock = clock or Clock()
    made: list[Any] = []

    def factory(key: str, secret: str, account: str | None) -> Any:
        made.append((key, secret, account))
        return stub

    rc = wa.main(["--setup-output", str(setup_file(tmp_path)), "--secret", f"twilio_api_key={ARN}", *argv],
                 sm_client=sm_client(), client_factory=factory,
                 sleep=clock.sleep, clock=clock.now, now=lambda: now)
    assert all(m == (API_SID, API_SECRET, ACCOUNT) for m in made)
    return rc


def heloc(at: str = NOW_TEXT, product: str = "HELOC", views: int = 3) -> dict[str, Any]:
    return {"lastViewedProduct": product, "lastViewedAt": at, "viewCount": views}


@pytest.fixture(autouse=True)
def _no_secret_anywhere(capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> Any:
    """No credential (API key SID or secret) in any output or log record."""
    caplog.set_level(logging.DEBUG)
    yield
    out = capsys.readouterr()
    logs = "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    for secret in (API_SECRET, API_SID):
        assert secret not in out.out + out.err + logs


def output(capsys: pytest.CaptureFixture[str]) -> tuple[str, str]:
    out = capsys.readouterr()
    # Printed again so the autouse credential check sees it too.
    print(out.out, end="")
    print(out.err, end="", file=sys.stderr)
    return out.out, out.err


# --------------------------------------------------------------------------- plan


def test_plan_names_customer_group_and_traits_and_calls_nothing(tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()

    class NoRead:
        def get_secret_value(self, SecretId: str) -> Any:  # noqa: N803
            raise AssertionError("the plan reads no credential")

    rc = wa.main(["--setup-output", str(setup_file(tmp_path))], sm_client=NoRead(),
                 client_factory=lambda *a: pytest.fail("no client in the plan"), now=lambda: NOW)
    out, _ = output(capsys)
    assert rc == 0 and stub.log == []
    assert out.splitlines()[0] == wa.BANNER
    assert "C-1001" in out and EMAIL in out and STORE in out and "DRY RUN" in out
    assert f"group {GROUP}, traits {', '.join(TRAITS)}" in out
    assert "'HELOC'" in out and NOW_TEXT in out


def test_clear_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, ProfileStub(), "--clear") == 0
    out, _ = output(capsys)
    assert out.startswith(wa.BANNER + "\n") and "null" in out and "Planned (clear)" in out


@pytest.mark.parametrize("stage", ["prod", "PROD", "staging"])
def test_prod_and_unknown_stages_are_refused(stage: str, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    assert run(tmp_path, stub, "--stage", stage, "--apply") == wa.EXIT_USAGE
    out, err = output(capsys)
    assert out.startswith(wa.BANNER) and stub.log == []
    assert ("refused" in err) if stage.lower() == "prod" else ("must be one of" in err)


def test_test_stage_is_accepted(tmp_path: Path) -> None:
    assert run(tmp_path, ProfileStub(), "--stage", "test") == 0


def test_a_customer_not_in_the_fixtures_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    assert run(tmp_path, stub, "--customer", "C-9999", "--apply") == wa.EXIT_USAGE
    assert "not a customer in data/fixtures.json" in output(capsys)[1] and stub.log == []


def test_another_fixture_customer_is_used_by_its_memory_email(tmp_path: Path,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    fixtures = json.loads((wa.DATA_DIR / "fixtures.json").read_text(encoding="utf-8"))
    other = next(c for c in fixtures["customers"] if c["customer_id"] != "C-1001")
    assert run(tmp_path, ProfileStub(), "--customer", other["customer_id"]) == 0
    assert other["memory_email"] in output(capsys)[0]


@pytest.mark.parametrize("bad", [["--viewed-at", "2026-10-28"], ["--viewed-at", "yesterday"],
                                 ["--product", "  "], ["--views", "-1"], ["--wait", "-1"],
                                 ["--check", "--apply"], ["--trait-groups", " , "],
                                 ["--trait-groups", "Contact,Contact"], ["--trait-groups", "bad name"],
                                 ["--trait-groups", ",".join(f"G{i}" for i in range(11))],
                                 ["--max-age-hours", "0"], ["--max-age-hours", "8761"]])
def test_bad_options_exit_2_before_any_call(bad: list[str], tmp_path: Path) -> None:
    stub = ProfileStub()
    assert run(tmp_path, stub, *bad) == wa.EXIT_USAGE and stub.log == []


def test_missing_setup_output_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = wa.main(["--setup-output", str(tmp_path / "none.json")], now=lambda: NOW)
    assert rc == wa.EXIT_USAGE and "twilio_setup.py --apply" in output(capsys)[1]


def test_viewed_at_is_normalised_to_utc(tmp_path: Path) -> None:
    stub = ProfileStub()
    pid = stub.add()
    assert run(tmp_path, stub, "--viewed-at", "2026-10-28T10:00:00-04:00", "--apply") == 0
    assert stub.profiles[pid][GROUP]["lastViewedAt"] == "2026-10-28T14:00:00Z"


# --------------------------------------------------------------------------- --apply


def test_apply_patches_an_existing_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    pid = stub.add()
    assert run(tmp_path, stub, "--apply") == 0
    (method, url, body), = stub.writes()
    assert method == "PATCH" and url.endswith(f"/Profiles/{pid}")
    assert body == {"traits": {GROUP: heloc()}}
    assert stub.profiles[pid] == {"Contact": {"email": EMAIL}, GROUP: heloc()}
    out, _ = output(capsys)
    assert out.startswith(wa.BANNER + "\n") and "visible after 0.0 s" in out
    assert stub.auths == {(API_SID, API_SECRET)}


def test_apply_upserts_a_missing_profile_keyed_by_email(tmp_path: Path) -> None:
    stub = ProfileStub()
    assert run(tmp_path, stub, "--product", "auto loan", "--views", "1", "--apply") == 0
    (method, url, body), = stub.writes()
    assert method == "POST" and urlsplit(url).path == f"/v1/Stores/{STORE}/Profiles"
    assert body == {"traits": {"Contact": {"email": EMAIL}, GROUP: heloc(product="auto loan", views=1)}}
    (pid,) = stub.emails[EMAIL]
    assert stub.profiles[pid][GROUP] == heloc(product="auto loan", views=1)


def test_read_back_after_an_upsert_waits_while_the_new_profile_reads_404(tmp_path: Path) -> None:
    stub, clock = ProfileStub(create_lag=2), Clock()
    assert run(tmp_path, stub, "--apply", clock=clock) == 0
    assert clock.sleeps == [wa.READ_POLL_S] * 2
    (pid,) = stub.emails[EMAIL]
    assert stub.profiles[pid][GROUP] == heloc()


def test_a_404_reading_an_existing_profile_is_still_an_error(tmp_path: Path) -> None:
    stub = ProfileStub()
    pid = stub.add(traits={GROUP: heloc(views=1)})
    stub.hidden[pid] = 10**6  # the lookup names it, but it reads 404 (deleted meanwhile)
    assert run(tmp_path, stub, "--apply") == wa.EXIT_MEMORY


def test_apply_creates_a_missing_trait_group_first(tmp_path: Path) -> None:
    stub = ProfileStub(group=False)
    stub.add()
    assert run(tmp_path, stub, "--apply") == 0
    methods = [(m, urlsplit(u).path.rsplit("/", 1)[1]) for m, u, _ in stub.writes()]
    assert methods[0] == ("POST", "TraitGroups") and methods[1][0] == "PATCH"


def test_a_second_apply_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    pid = stub.add()
    assert run(tmp_path, stub, "--apply") == 0
    first = json.loads(json.dumps(stub.profiles))
    stub.log.clear()
    assert run(tmp_path, stub, "--apply") == 0
    assert stub.writes() == [] and stub.profiles == first
    assert "already holds these values; nothing written" in output(capsys)[0]
    assert stub.emails[EMAIL] == [pid]


def test_apply_changes_only_what_differs(tmp_path: Path) -> None:
    stub = ProfileStub()
    pid = stub.add(traits={GROUP: heloc(views=1)})
    assert run(tmp_path, stub, "--apply") == 0
    assert stub.profiles[pid][GROUP] == heloc()


def test_read_back_waits_and_reports_the_time(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub, clock = ProfileStub(lag=3), Clock()
    stub.add()
    assert run(tmp_path, stub, "--apply", clock=clock) == 0
    # The write shows on the third read after it: two waits of READ_POLL_S.
    assert clock.sleeps == [wa.READ_POLL_S] * 2
    assert "visible after 2.0 s" in output(capsys)[0]


def test_read_back_timeout_is_an_error_with_a_non_zero_exit(tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    stub, clock = ProfileStub(lag=10**6), Clock()
    stub.add()
    assert run(tmp_path, stub, "--apply", "--wait", "5", clock=clock) == wa.EXIT_MEMORY
    assert "not visible after 5 s" in output(capsys)[1]
    assert 5 <= clock.t <= 5 + wa.READ_POLL_S


def test_two_profiles_for_one_email_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    stub.add()
    stub.add()
    assert run(tmp_path, stub, "--apply") == wa.EXIT_MEMORY
    assert "2 profiles share" in output(capsys)[1] and stub.writes() == []


def test_a_308_re_looks_the_profile_up_and_retries_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    old = stub.add()
    new = stub.add(email="merged@owlbank-demo.example")
    stub.merged[old] = new
    lookups = iter([[old], [new]])  # the first lookup still names the old ID, the next the new one
    real = stub.request

    def request(method: str, url: str, **kw: Any) -> Any:
        if url.endswith("/Lookup"):
            stub.log.append((method, url, kw.get("data")))
            return _resp(200, {"profiles": next(lookups)})
        return real(method, url, **kw)

    stub.http_client = SimpleNamespace(request=request)
    assert run(tmp_path, stub, "--apply") == 0
    assert stub.profiles[new][GROUP] == heloc()
    gets_of_old = [c for c in stub.log if c[0] == "GET" and c[1].endswith(old)]
    assert len(gets_of_old) == 1 and "profile was merged" in output(capsys)[0]


def test_a_second_308_after_the_re_lookup_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    old = stub.add()
    stub.merged[old] = "mem_profile_elsewhere"
    assert run(tmp_path, stub, "--apply") == wa.EXIT_MEMORY
    assert "HTTP 308" in output(capsys)[1]
    assert len([c for c in stub.log if c[0] == "GET" and c[1].endswith(old)]) == 2


def test_a_memory_error_names_status_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    stub.add()

    def refuse(method: str, url: str, **kw: Any) -> Any:
        return _resp(403, {"code": 20003, "message": f"denied {API_SECRET}"})

    stub.http_client = SimpleNamespace(request=refuse)
    assert run(tmp_path, stub, "--apply") == wa.EXIT_MEMORY
    err = output(capsys)[1]
    assert "HTTP 403" in err and "denied" not in err


def test_a_network_error_prints_its_type_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()

    def down(*_a: Any, **_k: Any) -> Any:
        raise ConnectionError(f"https://{API_SID}:{API_SECRET}@memory.twilio.com unreachable")

    stub.http_client = SimpleNamespace(request=down)
    assert run(tmp_path, stub, "--apply") == wa.EXIT_UNEXPECTED
    assert output(capsys)[1].strip() == "web_activity: ConnectionError"


def test_a_secrets_manager_failure_is_named_without_values(tmp_path: Path,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    class Broken:
        def get_secret_value(self, SecretId: str) -> Any:  # noqa: N803
            raise RuntimeError(API_SECRET)

    rc = wa.main(["--setup-output", str(setup_file(tmp_path)), "--secret", f"twilio_api_key={ARN}", "--apply"],
                 sm_client=Broken(), client_factory=lambda *a: pytest.fail("no client without a key"),
                 now=lambda: NOW)
    err = output(capsys)[1]
    assert rc == wa.EXIT_MEMORY and "twilio_api_key" in err and "get_secret_value failed" in err


def test_a_secret_field_fault_names_the_field_not_the_value(tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    bad = SecretsStub({"account_sid": ACCOUNT, "api_key": API_SID, "api_secret": "short-" + API_SECRET})
    rc = wa.main(["--setup-output", str(setup_file(tmp_path)), "--secret", f"twilio_api_key={ARN}", "--apply"],
                 sm_client=bad, client_factory=lambda *a: pytest.fail("no client without a key"),
                 now=lambda: NOW)
    assert rc == wa.EXIT_MEMORY and "api_secret" in output(capsys)[1]


def test_no_arn_is_a_setup_error_naming_the_secret(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = wa.main(["--setup-output", str(setup_file(tmp_path)), "--apply"], sm_client=SecretsStub(),
                 client_factory=lambda *a: pytest.fail("no client without a key"), now=lambda: NOW)
    assert rc == wa.EXIT_USAGE and "twilio_api_key" in output(capsys)[1]  # missing setup


# --------------------------------------------------------------------------- --clear


@pytest.mark.parametrize("drop_empty", [True, False])
def test_clear_removes_web_activity_and_keeps_contact(drop_empty: bool, tmp_path: Path,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub(drop_empty=drop_empty, lag=2)
    pid = stub.add(traits={GROUP: heloc()})
    assert run(tmp_path, stub, "--clear", "--apply") == 0
    (method, _, body), = stub.writes()
    assert method == "PATCH" and body == {"traits": {GROUP: dict.fromkeys(TRAITS)}}
    assert stub.profiles[pid].get(GROUP, {}) == {} and stub.profiles[pid]["Contact"] == {"email": EMAIL}
    assert f"{GROUP} gone after" in output(capsys)[0]


def test_clear_with_nothing_to_clear_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    assert run(tmp_path, stub, "--clear", "--apply") == 0
    stub.add()
    assert run(tmp_path, stub, "--clear", "--apply") == 0
    assert stub.writes() == []
    out = output(capsys)[0]
    assert "no profile for" in out and "nothing to clear" in out


# --------------------------------------------------------------------------- --check


def check(tmp_path: Path, stub: ProfileStub, capsys: pytest.CaptureFixture[str], *argv: str,
          now: datetime = NOW) -> tuple[int, str]:
    rc = run(tmp_path, stub, "--check", *argv, now=now)
    out, _ = output(capsys)
    assert out.startswith(wa.BANNER + "\n") and stub.writes() == []
    return rc, out


def test_check_prints_the_recall_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    stub.add(traits={GROUP: heloc(product="auto loan"), "Other": {"x": "hidden-value"}})
    rc, out = check(tmp_path, stub, capsys)
    assert rc == 0
    text = out.split("recall text the agent would get:\n", 1)[1]
    assert "auto loan" in text and NOW_TEXT in text and EMAIL in text
    assert "hidden-value" not in out  # not on the allow-list
    assert "observations and summaries are not shown" in out


def test_check_text_is_what_tac_builds_after_the_bridge_filter(tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    from tac.adapters import AdapterOptions, MemoryPromptBuilder
    from tac.models.memory import ProfileResponse
    from tac.models.session import ConversationSession

    from owlbank.web_activity import bound_profile_traits

    stub = ProfileStub()
    pid = stub.add(traits={GROUP: heloc(product="HELOC​ with   spaces " + "x" * 200)})
    _, out = check(tmp_path, stub, capsys)
    allow = ["Contact", GROUP]
    kept = bound_profile_traits(stub.profiles[pid], allow=allow, now=NOW, max_age_hours=24)
    ctx = ConversationSession(conversation_id="c", channel="VOICE", profile_id=pid,
                              profile=ProfileResponse(id=pid, created_at="x", traits=kept))
    expected = MemoryPromptBuilder.build(None, ctx, AdapterOptions(profile_traits=allow))
    assert expected and out.split("recall text the agent would get:\n", 1)[1].rstrip("\n") == expected


@pytest.mark.parametrize(("case", "reason"), [
    ("no_profile", "no profile for"),
    ("no_traits", "has no WebActivity traits"),
    ("cleared_empty_group", "has no WebActivity traits"),
    ("not_allowed", "not on the trait-group allow-list"),
    ("too_old", "too old"),
    ("future", "in the future"),
    ("unparseable", "not an ISO-8601 date and time"),
    ("missing_at", "missing or not an ISO-8601"),
])
def test_check_names_each_reason_nothing_comes_back(case: str, reason: str, tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    argv: list[str] = []
    if case == "no_traits":
        stub.add()
    elif case == "cleared_empty_group":
        stub.add(traits={GROUP: dict.fromkeys(TRAITS)})
    elif case == "not_allowed":
        stub.add(traits={GROUP: heloc()})
        argv = ["--trait-groups", "Contact"]
    elif case == "too_old":
        stub.add(traits={GROUP: heloc(at=(NOW - timedelta(hours=25)).strftime("%Y-%m-%dT%H:%M:%SZ"))})
    elif case == "future":
        stub.add(traits={GROUP: heloc(at=(NOW + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"))})
    elif case == "unparseable":
        stub.add(traits={GROUP: heloc(at="last tuesday")})
    elif case == "missing_at":
        stub.add(traits={GROUP: {"lastViewedProduct": "HELOC", "viewCount": 3}})
    rc, out = check(tmp_path, stub, capsys, *argv)
    assert rc == wa.EXIT_NOT_RETRIEVED
    assert f"no {GROUP} comes back: " in out and reason in out
    assert "recall text" not in out


def test_check_window_override_keeps_an_older_view(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    stub.add(traits={GROUP: heloc(at=(NOW - timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ"))})
    rc, out = check(tmp_path, stub, capsys, "--max-age-hours", "48")
    assert rc == 0 and "window 48 h" in out


def test_check_with_two_profiles_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = ProfileStub()
    stub.add()
    stub.add()
    assert run(tmp_path, stub, "--check") == wa.EXIT_MEMORY
    assert "2 profiles share" in output(capsys)[1]


def test_apply_then_check_then_clear_then_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The demo cycle of the runbook against one stub."""
    stub = ProfileStub()
    assert run(tmp_path, stub, "--apply") == 0
    assert run(tmp_path, stub, "--check") == 0
    assert run(tmp_path, stub, "--clear", "--apply") == 0
    assert run(tmp_path, stub, "--check") == wa.EXIT_NOT_RETRIEVED
    assert "has no WebActivity traits" in output(capsys)[0]


def test_a_trait_the_script_does_not_write_is_kept_by_apply_and_removed_by_clear(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A group holding another trait (one a feed added) made the read-back never match."""
    stub = ProfileStub()
    pid = stub.add(traits={GROUP: {**heloc(views=1), "referrer": "newsletter"}})
    assert run(tmp_path, stub, "--apply") == 0
    assert stub.profiles[pid][GROUP] == {**heloc(), "referrer": "newsletter"}
    assert run(tmp_path, stub, "--clear", "--apply") == 0
    assert GROUP not in stub.profiles[pid]
    *_, (_, _, body) = stub.writes()
    assert body == {"traits": {GROUP: {**dict.fromkeys(TRAITS), "referrer": None}}}


# --------------------------------------------------------------------------- the shared helpers


def test_upsert_without_a_profile_id_is_an_error() -> None:
    from scripts import _memory_traits as mt
    from scripts._common import ScriptError

    stub = ProfileStub()
    stub.http_client = SimpleNamespace(request=lambda *a, **k: _resp(202, {"message": "accepted"}))
    with pytest.raises(ScriptError, match="without a profile ID"):
        mt.upsert_profile(stub, STORE, {"Contact": {"email": EMAIL}})


def test_reading_a_missing_profile_raises_not_empty() -> None:
    from scripts import _memory_traits as mt

    with pytest.raises(mt.MemoryApiError) as err:
        mt.get_profile_traits(ProfileStub(), STORE, "mem_profile_gone")
    assert err.value.status == 404


@pytest.mark.parametrize("mode", ["--apply", "--check"])
@pytest.mark.parametrize("where", [[], ["--cdk-outputs", "missing-outputs.json"]])
def test_no_secret_arn_is_missing_setup(mode: str, where: list[str], tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    """No ARN for the API key secret (no --secret, no or a missing outputs file) exits 2,
    before any AWS or Memory call."""
    argv = ["--setup-output", str(setup_file(tmp_path)), mode,
            *[str(tmp_path / a) if a.endswith(".json") else a for a in where]]
    rc = wa.main(argv, sm_client=None, client_factory=lambda *a: pytest.fail("no Memory call expected"),
                 now=lambda: NOW)
    assert rc == wa.EXIT_USAGE
    assert "web_activity:" in output(capsys)[1]
