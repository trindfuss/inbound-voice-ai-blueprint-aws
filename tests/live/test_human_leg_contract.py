"""Human-leg: Orchestrator contract test against the dev account's real API, no call (``-m human_leg_contract``).

Answers (participant body, worker identity, auto-created human participant, hand-off marker) and
(does a ``PUT`` keep omitted fields, is it the right verb) without a phone call, and probes the API key's
scope. Skipped by the default ``pytest`` run (``addopts``). It **creates one conversation** in the dev
account and closes it again, so check that the account is a sandbox before the first run::

    export OWLBANK_EXPECT_ACCOUNT_SID=AC...                    # must match the account in use, else the test refuses
    export OWLBANK_TWILIO_ACCOUNT_SID=AC...                    # account of the key (an identifier, not a secret)
    export OWLBANK_CONVERSATION_CONFIGURATION_ID=conv_configuration_...   # dev Orchestrator configuration
    # optional: OWLBANK_CONTRACT_WORKER_SID=WK...  a real Flex worker (else a made-up one tests the validation)
    # optional: OWLBANK_CONTRACT_KEY_SID / OWLBANK_CONTRACT_KEY_SECRET (default: the ``twilio_api_key``
    #           secret in Secrets Manager, its ARN from OWLBANK_CDK_OUTPUTS, default infra/cdk-outputs.json;
    #           the secret is never printed or recorded)
    uv run pytest -m human_leg_contract -rA -s tests/live/test_human_leg_contract.py

Every HTTP exchange (method, URL, request body, status, response body; the ``Authorization`` header is never
recorded) is printed as one ``CONTRACT {json}`` line, and the full list is written to
``OWLBANK_CONTRACT_OUT`` (default: none) as JSON. Conversation names carry
the ``humanleg-contract-`` prefix; the conversation is closed in a ``finally`` even if a step fails.

No property-based test (a live contract check, not a logic invariant).
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
import uuid
from typing import Any

import pytest

from owlbank.humanleg_client import DEFAULT_BASE_URL, OrchestratorHumanLeg, _send
from scripts._common import ScriptError, load_cdk_outputs, read_secret, resolve_region, secret_arn
from tests.live._redact import redact

pytestmark = pytest.mark.human_leg_contract

NAME_PREFIX = "humanleg-contract-"
#: Valid in shape (``WK`` + 32 hex), so the client sends it; not a real worker unless you pass one.
FAKE_WORKER_SID = "WK" + "0" * 32
DEFAULT_CDK_OUTPUTS = "infra/cdk-outputs.json"


def _key_and_secret() -> tuple[str, str] | None:
    sid, secret = os.environ.get("OWLBANK_CONTRACT_KEY_SID"), os.environ.get("OWLBANK_CONTRACT_KEY_SECRET")
    if sid and secret:
        return sid, secret
    try:
        import boto3

        arn = secret_arn("twilio_api_key",
                         load_cdk_outputs(os.environ.get("OWLBANK_CDK_OUTPUTS", DEFAULT_CDK_OUTPUTS)), {})
        client = boto3.client("secretsmanager", region_name=resolve_region(arns=[arn]))
        creds = read_secret("twilio_api_key", arn, client)
    except (ImportError, ScriptError):
        return None
    return creds["api_key"], creds["api_secret"]


class Recorder:
    """Wraps the client's opener: records every exchange, keeps HTTP error bodies the client discards."""

    def __init__(self) -> None:
        self.exchanges: list[dict[str, Any]] = []
        self.step = ""

    def __call__(self, req: urllib.request.Request, timeout: float) -> Any:
        entry: dict[str, Any] = {"step": self.step, "method": req.get_method(), "url": req.full_url,
                                 "request": _json_or_text(req.data)}
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310  # https only: the client checked
        except urllib.error.HTTPError as exc:
            body = exc.read()
            entry.update(status=exc.code, response=_json_or_text(body))
            self._emit(entry)
            raise urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.headers, io.BytesIO(body)) from None
        raw = resp.read()
        entry.update(status=int(getattr(resp, "status", 200)), response=_json_or_text(raw))
        self._emit(entry)
        return _Replay(entry["status"], raw)

    def _emit(self, entry: dict[str, Any]) -> None:
        """Print and keep a redacted copy (keys with token/secret/password, any depth); the key-scope probes
 keep the status only, no response body. The raw entry is never printed or saved."""
        safe = redact(entry)
        if self.step.startswith("probe_"):
            safe["response"] = None
        self.exchanges.append(safe)
        print("CONTRACT " + json.dumps(safe, sort_keys=True))  # noqa: T201


class _Replay:
    """The already-read response, for the client (a context manager with ``status`` and ``read``)."""

    def __init__(self, status: int, raw: bytes) -> None:
        self.status = status
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Replay:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _json_or_text(raw: bytes | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")[:2000]


def _note(recorder: Recorder, step: str, **facts: Any) -> None:
    recorder.step = step
    print("CONTRACT_STEP " + json.dumps({"step": step, **facts}, sort_keys=True, default=str))  # noqa: T201


def test_orchestrator_contract() -> None:
    pair = _key_and_secret()
    account_sid = os.environ.get("OWLBANK_TWILIO_ACCOUNT_SID")
    config_id = os.environ.get("OWLBANK_CONVERSATION_CONFIGURATION_ID")
    if not pair or not account_sid or not config_id:
        pytest.skip("no Twilio credentials, account SID or conversation configuration id for the dev account")
    expected = os.environ.get("OWLBANK_EXPECT_ACCOUNT_SID")
    if expected != account_sid:
        pytest.fail("refusing to run: OWLBANK_EXPECT_ACCOUNT_SID must equal the account in use "
                    f"(expected {expected!r}, in use {account_sid!r})")
    key_sid, key_secret = pair
    worker_sid = os.environ.get("OWLBANK_CONTRACT_WORKER_SID", FAKE_WORKER_SID)

    recorder = Recorder()
    client = OrchestratorHumanLeg(lambda: (account_sid, key_sid, key_secret), opener=recorder)
    call = client._request  # a small test-only helper over the client's HTTP: raw calls, not client methods
    conversation_id: str | None = None
    closed_by_client = False
    try:
        # 0. Create. The client never creates conversations, so the test does (body shape is itself a finding).
        name = NAME_PREFIX + uuid.uuid4().hex[:8]
        _note(recorder, "create_conversation", name=name)
        for body in ({"configurationId": config_id, "name": name}, {"configurationId": config_id}):
            made = call("POST", "/v2/Conversations", "contract.create", body=body)
            if made.ok and made.data and made.data.get("id"):
                conversation_id = str(made.data["id"])
                break
        assert conversation_id, "could not create a test conversation (see the CONTRACT lines for the responses)"

        # 1. Read back: status, participants Orchestrator made itself (auto-created human participant?).
        _note(recorder, "get_conversation")
        call("GET", f"/v2/Conversations/{conversation_id}", "contract.get")
        _note(recorder, "list_participants_initial")
        call("GET", f"/v2/Conversations/{conversation_id}/Participants", "contract.participants")

        # 2. Add a human participant with a worker identity, then again (the client treats it as a no-op).
        _note(recorder, "add_human_participant", worker_sid=worker_sid)
        first = client.add_human_participant(conversation_id, worker_sid, "contract-agent")
        _note(recorder, "list_participants_after_add", client_returned=first)
        call("GET", f"/v2/Conversations/{conversation_id}/Participants", "contract.participants")
        _note(recorder, "add_human_participant_again")
        second = client.add_human_participant(conversation_id, worker_sid, "contract-agent")
        _note(recorder, "add_again_result", client_returned=second)

        # 3. Post a communication (shape unverified: the body below is a first guess; a refusal is a finding).
        _note(recorder, "post_communication")
        call("POST", f"/v2/Conversations/{conversation_id}/Communications", "contract.communication",
             body={"author": {"address": "humanleg-contract", "channel": "API"},
                   "content": {"type": "TEXT", "text": "humanleg contract test"},
                   "recipients": []})

        # 4.: seed metadata, then PUT with status and a DIFFERENT key only. Does the seed survive?
        _note(recorder, "put_seed_metadata")
        call("PUT", f"/v2/Conversations/{conversation_id}", "contract.put", body={"metadata": {"seed": "a"}})
        _note(recorder, "put_status_active_without_seed_key")
        call("PUT", f"/v2/Conversations/{conversation_id}", "contract.put",
             body={"status": "ACTIVE", "metadata": {"probe": "b"}})
        _note(recorder, "get_after_partial_put")
        call("GET", f"/v2/Conversations/{conversation_id}", "contract.get")
        _note(recorder, "mark_handoff")
        marked = client.mark_handoff(conversation_id, {"humanLeg": "1"})
        _note(recorder, "get_after_mark_handoff", client_returned=marked)
        call("GET", f"/v2/Conversations/{conversation_id}", "contract.get")

        # 5. Key scope: the same key against two other APIs; only the statuses matter.
        key = (key_sid, key_secret)
        for step, url in (
            ("probe_calls_list", f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json?PageSize=1"),
            ("probe_taskrouter", "https://taskrouter.twilio.com/v1/Workspaces?PageSize=1"),
            ("probe_account_fetch", f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}.json"),
        ):
            _note(recorder, step)
            _send(recorder, 10.0, "GET", url, "twilio_contract_probe", step, key, None, "application/json")

        # 6. Close, then close again (client: second sends nothing) and a raw PUT CLOSED on the closed one.
        _note(recorder, "close")
        closed_by_client = client.close(conversation_id)
        assert closed_by_client is True
        sent_before = len(recorder.exchanges)
        _note(recorder, "close_again_via_client")
        assert client.close(conversation_id) is True
        assert len(recorder.exchanges) == sent_before, "a repeated close must send no request"
        _note(recorder, "raw_put_closed_on_closed")
        call("PUT", f"/v2/Conversations/{conversation_id}", "contract.put", body={"status": "CLOSED"})
        _note(recorder, "raw_put_active_on_closed")
        call("PUT", f"/v2/Conversations/{conversation_id}", "contract.put", body={"status": "ACTIVE"})
    finally:
        if conversation_id:
            # Whatever ``closed_by_client`` says: a step may have reopened the conversation (raw PUT ACTIVE).
            _note(recorder, "final_status")
            final = client._request("GET", f"/v2/Conversations/{conversation_id}", "contract.get")
            if (final.data or {}).get("status") != "CLOSED":
                _note(recorder, "cleanup_close")
                # A fresh client: this one may have a step half done; closing must not depend on its state.
                cleanup = OrchestratorHumanLeg(lambda: (account_sid, key_sid, key_secret), opener=recorder)
                cleanup.close(conversation_id)
                _note(recorder, "final_status_after_cleanup")
                final = client._request("GET", f"/v2/Conversations/{conversation_id}", "contract.get")
            print("CONTRACT_FINAL " + json.dumps({"conversation_id": conversation_id,  # noqa: T201
                                                  "status": (final.data or {}).get("status")}))
        out = os.environ.get("OWLBANK_CONTRACT_OUT")
        if out:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(recorder.exchanges, fh, indent=2, sort_keys=True)
    assert (final.data or {}).get("status") == "CLOSED", "the test conversation was left open"


def test_base_url_is_https() -> None:
    """Guard: the recorder would send credentials wherever the client points; it must be Twilio's https host."""
    assert DEFAULT_BASE_URL.startswith("https://conversations.twilio.com")
