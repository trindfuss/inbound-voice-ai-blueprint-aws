"""scripts/memory_history.py against a stateful Memory stub: the
plan, ``--apply`` with the preset built from the fixture and with custom text, a repeat writing
nothing, ``--clear`` leaving a real item, the read-back wait and its timeout, the prod, non-fixture and
no-profile refusals, and no credential in any output. No network. The property test is in
tests/property/test_memory_history_props.py."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from scripts import memory_history as mh

STORE = "mem_store_" + "1" * 26
ACCOUNT = "AC" + "3" * 32
API_SID = "SK" + "c4" * 16
API_SECRET = "apisecretHISTORY9876543210abcdef"
EMAIL = "thomas.doe@owlbank-demo.example"  # C-1001's memory_email in data/fixtures.json
NOW = datetime(2026, 10, 29, 15, 0, 0, tzinfo=UTC)
YESTERDAY = "2026-10-28T15:00:00Z"
PRESET_SUMMARY = ("WhatsApp chat: the customer said they don't recognise a $412.40 charge from Example "
                  "Airlines on their OwlBank Red Card dated 2026-10-05. The chat assistant logged it for "
                  "the disputes team and said they would follow up.")
PRESET_OBSERVATION = ("Customer has an open question about an unrecognised Example Airlines charge on the "
                      "OwlBank Red Card.")


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


class HistoryStub:
    """A Twilio client whose HTTP layer answers the Memory calls of the script, with state.

    ``summaries`` / ``observations``: profile ID -> list of stored items. ``lag``: lists before a
    created summary shows (Memory accepts summaries asynchronously). ``page``: the page size the stub
    serves (exercises the token pagination)."""

    def __init__(self, *, lag: int = 0, page: int = 2) -> None:
        self.lag, self.page = lag, page
        self.emails: dict[str, list[str]] = {}
        self.summaries: dict[str, list[dict[str, Any]]] = {}
        self.observations: dict[str, list[dict[str, Any]]] = {}
        self.pending: list[list[Any]] = []  # [lists left, profile ID, summary]
        self.log: list[tuple[str, str, Any]] = []
        self.next_id = 1
        self.username, self.password = API_SID, API_SECRET
        self.http_client = SimpleNamespace(request=self.request)

    def add_profile(self, email: str = EMAIL) -> str:
        pid = f"mem_profile_{self.next_id:026d}"
        self.next_id += 1
        self.emails.setdefault(email, []).append(pid)
        self.summaries[pid], self.observations[pid] = [], []
        return pid

    def item(self, content: str, source: str) -> dict[str, Any]:
        self.next_id += 1
        return {"id": f"mem_item_{self.next_id:026d}", "content": content, "source": source}

    def writes(self) -> list[tuple[str, str, Any]]:
        return [c for c in self.log if c[0] in ("POST", "DELETE") and not c[1].endswith("/Lookup")]

    def _tick(self) -> None:
        for item in list(self.pending):
            item[0] -= 1
            if item[0] <= 0:
                self.pending.remove(item)
                self.summaries[item[1]].append(item[2])

    def request(self, method: str, url: str, params: Any = None, data: Any = None, headers: Any = None,
                auth: Any = None, timeout: Any = None, allow_redirects: bool = False) -> Any:
        self.log.append((method, url, data))
        assert auth == (API_SID, API_SECRET) and not allow_redirects and timeout
        path = unquote(urlsplit(url).path)
        prefix = f"/v1/Stores/{STORE}/Profiles"
        if method == "POST" and path == prefix + "/Lookup":
            return _resp(200, {"profiles": self.emails.get(data["value"], [])})
        parts = path[len(prefix) + 1:].split("/")
        pid, coll = parts[0], parts[1]
        if pid not in self.summaries:
            return _resp(404, {"code": 20404})
        items = self.summaries if coll == "ConversationSummaries" else self.observations
        key = "summaries" if coll == "ConversationSummaries" else "observations"
        if method == "POST":
            for it in data[key]:
                made = {**it, "id": self.item("", "")["id"]}
                if key == "summaries" and self.lag:
                    self.pending.append([self.lag, pid, made])
                else:
                    items[pid].append(made)
            return _resp(202, {"message": "accepted"})
        if method == "DELETE":
            before = len(items[pid])
            items[pid] = [i for i in items[pid] if i["id"] != parts[2]]
            return _resp(204, None) if len(items[pid]) < before else _resp(404, {"code": 20404})
        assert method == "GET"
        self._tick()
        start = int((params or {}).get("pageToken", 0))
        end = start + min(self.page, int(params["pageSize"]))
        meta = {"key": key, "nextToken": str(end) if end < len(items[pid]) else None}
        return _resp(200, {key: items[pid][start:end], "meta": meta})


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


def run(tmp_path: Path, stub: HistoryStub, *argv: str, clock: Clock | None = None) -> int:
    clock = clock or Clock()
    return mh.main(["--setup-output", str(setup_file(tmp_path)), "--secret", f"twilio_api_key={ARN}", *argv],
                   sm_client=sm_client(),
                   client_factory=lambda *a: stub if a == (API_SID, API_SECRET, ACCOUNT) else pytest.fail("wrong key"),
                   sleep=clock.sleep, clock=clock.now, now=lambda: NOW)


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
    return out.out, out.err


def test_plan_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = HistoryStub()
    stub.add_profile()
    assert run(tmp_path, stub) == 0
    out, _ = output(capsys)
    assert out.startswith("Demo data: ") and "DRY RUN" in out and "C-1001" in out
    assert PRESET_SUMMARY in out and PRESET_OBSERVATION in out and YESTERDAY in out
    assert stub.log == []  # not even a lookup


def test_apply_writes_the_preset_from_the_fixture(tmp_path: Path) -> None:
    stub = HistoryStub()
    (pid,) = [stub.add_profile()]
    assert run(tmp_path, stub, "--apply") == 0
    (s,), (o,) = stub.summaries[pid], stub.observations[pid]
    assert s["content"] == PRESET_SUMMARY and s["source"] == "owlbank-demo"
    assert s["occurredAt"] == YESTERDAY and s["conversationId"].startswith("conv_conversation_")
    assert len(s["conversationId"]) == len("conv_conversation_") + 26
    # Memory's own pattern (HTTP 400, Twilio error 20001 otherwise).
    assert re.fullmatch(r"conv_conversation_[0-7][0-9a-z]{25}", s["conversationId"])
    assert o["content"] == PRESET_OBSERVATION and o["source"] == "owlbank-demo"
    assert o["occurredAt"] == YESTERDAY and o["conversationIds"] == [s["conversationId"]]
    assert not re.search(r"\b(he|him|his|she|her|hers)\b", s["content"] + o["content"], re.IGNORECASE)


def test_channel_names_the_channel(tmp_path: Path) -> None:
    stub = HistoryStub()
    pid = stub.add_profile()
    assert run(tmp_path, stub, "--channel", "sms", "--apply") == 0
    assert stub.summaries[pid][0]["content"].startswith("SMS chat: the customer said")


def test_second_apply_writes_nothing(tmp_path: Path) -> None:
    stub = HistoryStub()
    stub.add_profile()
    assert run(tmp_path, stub, "--apply") == 0
    before = len(stub.writes())
    assert run(tmp_path, stub, "--apply") == 0
    assert len(stub.writes()) == before


def test_custom_text_writes_only_what_is_given(tmp_path: Path) -> None:
    stub = HistoryStub()
    pid = stub.add_profile()
    assert run(tmp_path, stub, "--summary", " Custom summary. ", "--apply") == 0
    assert [s["content"] for s in stub.summaries[pid]] == ["Custom summary."]
    assert stub.observations[pid] == []
    assert run(tmp_path, stub, "--observation", "Custom observation.", "--apply") == 0
    assert [o["content"] for o in stub.observations[pid]] == ["Custom observation."]


def test_custom_text_needs_no_airline_transaction(tmp_path: Path) -> None:
    stub = HistoryStub()
    pid = stub.add_profile("sam.doe@owlbank-demo.example")
    assert run(tmp_path, stub, "--customer", "C-1002", "--observation", "Likes email.", "--apply") == 0
    assert len(stub.observations[pid]) == 1


def test_clear_deletes_only_demo_items(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = HistoryStub()
    pid = stub.add_profile()
    real_s, real_o = stub.item("A real chat.", "whatsapp-bot"), stub.item("A real note.", "crm")
    stub.summaries[pid].append(real_s)
    stub.observations[pid].append(real_o)
    assert run(tmp_path, stub, "--apply") == 0
    for _ in range(3):  # more than one page of demo items
        stub.summaries[pid].append(stub.item("demo", "owlbank-demo"))
    capsys.readouterr()
    assert run(tmp_path, stub, "--clear") == 0  # a plan: nothing deleted
    assert len(stub.summaries[pid]) == 5
    assert run(tmp_path, stub, "--clear", "--apply") == 0
    assert stub.summaries[pid] == [real_s] and stub.observations[pid] == [real_o]
    assert run(tmp_path, stub, "--clear", "--apply") == 0  # nothing left to clear
    assert "nothing to clear" in output(capsys)[0]


def test_read_back_waits_for_the_asynchronous_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub, clock = HistoryStub(lag=3), Clock()
    pid = stub.add_profile()
    assert run(tmp_path, stub, "--apply", clock=clock) == 0
    assert clock.sleeps and len(stub.summaries[pid]) == 1
    assert "items visible after" in output(capsys)[0]


def test_read_back_timeout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = HistoryStub(lag=10**6)
    stub.add_profile()
    assert run(tmp_path, stub, "--apply", "--wait", "5") == 3
    assert "not visible after 5 s" in output(capsys)[1]


@pytest.mark.parametrize("stage", ["prod", "PROD", " prod "])
def test_prod_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str], stage: str) -> None:
    stub = HistoryStub()
    assert run(tmp_path, stub, "--stage", stage, "--apply") == 2
    assert "prod is refused" in output(capsys)[1] and stub.log == []


def test_unknown_stage_is_refused(tmp_path: Path) -> None:
    assert run(tmp_path, HistoryStub(), "--stage", "qa", "--apply") == 2


def test_non_fixture_customer_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = HistoryStub()
    assert run(tmp_path, stub, "--customer", "C-9999", "--apply") == 2
    assert "not a customer in data/fixtures.json" in output(capsys)[1] and stub.log == []


def test_preset_refused_without_the_transaction(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stub = HistoryStub()
    stub.add_profile("sam.doe@owlbank-demo.example")
    assert run(tmp_path, stub, "--customer", "C-1002", "--apply") == 2
    assert "Example Airlines" in output(capsys)[1] and stub.log == []


def test_no_profile_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(tmp_path, HistoryStub(), "--apply") == 3
    assert "no profile for" in output(capsys)[1]


def test_two_profiles_for_one_email_is_an_error(tmp_path: Path) -> None:
    stub = HistoryStub()
    stub.add_profile()
    stub.add_profile()
    assert run(tmp_path, stub, "--apply") == 3
    assert stub.writes() == []


def test_bad_text_and_wait_are_usage_errors(tmp_path: Path) -> None:
    assert run(tmp_path, HistoryStub(), "--summary", "   ", "--apply") == 2
    assert run(tmp_path, HistoryStub(), "--wait", "-1") == 2


def test_missing_setup_output_is_a_usage_error(tmp_path: Path) -> None:
    rc = mh.main(["--setup-output", str(tmp_path / "none.json")])
    assert rc == 2


def test_a_secrets_manager_failure_is_named_without_values(tmp_path: Path,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    class Broken:
        def get_secret_value(self, SecretId: str) -> Any:  # noqa: N803
            raise RuntimeError(API_SECRET)

    rc = mh.main(["--setup-output", str(setup_file(tmp_path)), "--secret", f"twilio_api_key={ARN}", "--apply"],
                 sm_client=Broken(), client_factory=lambda *a: pytest.fail("no client without a key"),
                 now=lambda: NOW)
    err = capsys.readouterr().err
    print(err, end="")
    assert rc == mh.EXIT_MEMORY and "twilio_api_key" in err and "get_secret_value failed" in err


def test_a_secret_field_fault_names_the_field_not_the_value(tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    bad = SecretsStub({"account_sid": ACCOUNT, "api_key": API_SID, "api_secret": "short-" + API_SECRET})
    rc = mh.main(["--setup-output", str(setup_file(tmp_path)), "--secret", f"twilio_api_key={ARN}", "--apply"],
                 sm_client=bad, client_factory=lambda *a: pytest.fail("no client without a key"),
                 now=lambda: NOW)
    err = capsys.readouterr().err
    print(err, end="")
    assert rc == mh.EXIT_MEMORY and "api_secret" in err


def test_no_arn_is_a_setup_error_naming_the_secret(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = mh.main(["--setup-output", str(setup_file(tmp_path)), "--apply"], sm_client=SecretsStub(),
                 client_factory=lambda *a: pytest.fail("no client without a key"), now=lambda: NOW)
    assert rc == mh.EXIT_USAGE and "twilio_api_key" in capsys.readouterr().err  # missing setup


def test_the_plan_reads_no_secret(tmp_path: Path) -> None:
    sm = SecretsStub()
    rc = mh.main(["--setup-output", str(setup_file(tmp_path))], sm_client=sm,
                 client_factory=lambda *a: pytest.fail("no client in the plan"), now=lambda: NOW)
    assert rc == 0 and sm.calls == []


def test_unexpected_error_prints_the_type_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    class Boom(HistoryStub):
        def request(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError(f"https://x.example/?key={API_SECRET}")

    assert run(tmp_path, Boom(), "--apply") == 1
    assert output(capsys)[1].strip() == "memory_history: ConnectionError"



def test_missing_outputs_file_is_missing_setup(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A --cdk-outputs file that doesn't exist exits 2, before any AWS or Memory call."""
    argv = ["--setup-output", str(setup_file(tmp_path)), "--apply", "--cdk-outputs", str(tmp_path / "none.json")]
    rc = mh.main(argv, sm_client=None, client_factory=lambda *a: pytest.fail("no Memory call expected"),
                 now=lambda: NOW)
    assert rc == mh.EXIT_USAGE
    assert "memory_history:" in capsys.readouterr().err
