"""INIT-phase wiring and lazy clients, and what each Lambda does at import time."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from owlbank.coldstart import LazyClient, build_at_init, init_type, provisioned
from tests.logutil import fields

INIT = {"AWS_LAMBDA_INITIALIZATION_TYPE": "on-demand"}
PC = {"AWS_LAMBDA_INITIALIZATION_TYPE": "provisioned-concurrency"}


# --------------------------------------------------------------------------- build_at_init


def test_outside_lambda_nothing_is_built():
    calls: list[int] = []
    assert build_at_init(lambda: calls.append(1), {}) is None
    assert calls == []
    assert init_type({}) is None and not provisioned({})


def test_inside_lambda_it_builds_once_and_returns_the_value():
    calls: list[int] = []

    def build() -> str:
        calls.append(1)
        return "deps"

    assert build_at_init(build, INIT) == "deps" and calls == [1]


@pytest.mark.parametrize("env,want", [(INIT, False), (PC, True), ({}, False),
                                      ({"AWS_LAMBDA_INITIALIZATION_TYPE": "snap-start"}, False)])
def test_provisioned_detection(env, want):
    assert provisioned(env) is want


def test_a_failing_build_is_swallowed_and_logged_so_the_first_request_retries(caplog):
    def build() -> None:
        raise RuntimeError("secret detail")

    with caplog.at_level(logging.DEBUG):
        assert build_at_init(build, INIT) is None
    (rec,) = [r for r in caplog.records if getattr(r, "event", "") == "init_build_failed"]
    assert fields(rec)["decision"] == "build_on_first_request" and fields(rec)["error_class"] == "RuntimeError"
    assert "secret detail" not in caplog.text


def test_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE", raising=False)
    assert init_type() is None
    monkeypatch.setenv("AWS_LAMBDA_INITIALIZATION_TYPE", "provisioned-concurrency")
    assert provisioned()


# --------------------------------------------------------------------------- LazyClient


def test_lazy_client_builds_nothing_until_used_and_then_once():
    built: list[str] = []

    class Client:
        def encrypt(self, **kw: Any) -> str:
            return "sealed"

    def factory() -> Client:
        built.append("kms")
        return Client()

    lazy = LazyClient(factory)
    assert built == []
    assert lazy.encrypt(KeyId="k") == "sealed" and lazy.encrypt() == "sealed"
    assert built == ["kms"]


def test_lazy_client_does_not_pretend_to_have_private_or_dunder_attributes():
    lazy = LazyClient(lambda: (_ for _ in ()).throw(AssertionError("must not build")))
    assert not hasattr(lazy, "__wrapped__") and not hasattr(lazy, "_private")


# --------------------------------------------------------------------------- front door wiring


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch):
    """Offline boto3: fake credentials and Region; every client creation is recorded."""
    import boto3

    for k, v in {"AWS_ACCESS_KEY_ID": "AKIAEXAMPLE", "AWS_SECRET_ACCESS_KEY": "x", "AWS_REGION": "us-east-1",
                 "AWS_DEFAULT_REGION": "us-east-1", "OWLBANK_RELAY_WS_URL": "wss://relay.example/ws",
                 "OWLBANK_AUTH_SERVICE_FUNCTION": "auth:live", "OWLBANK_BANK_TOOLS_FUNCTION": "bank:live",
                 "OWLBANK_TOKEN_KMS_KEY": "alias/x"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("OWLBANK_PREWARM_FUNCTION", raising=False)
    # build() reads the policy and catalog tables once; serve them from the seed files, offline.
    from owlbank.catalog import PromptCatalog
    from owlbank.policy import Policy

    data = Path(__file__).resolve().parents[2] / "data"
    policy_doc = json.loads((data / "policy.json").read_text())
    catalog_doc = json.loads((data / "prompt_catalog.json").read_text())
    monkeypatch.setattr(Policy, "from_store", classmethod(lambda cls, store, **kw: cls.from_dict(policy_doc)))
    monkeypatch.setattr(PromptCatalog, "from_store",
                        classmethod(lambda cls, store, **kw: cls.from_dict(catalog_doc)))
    created: list[str] = []
    real = boto3.client

    def spy(service: str, *a: Any, **kw: Any) -> Any:
        created.append(service)
        return real(service, *a, **kw)

    monkeypatch.setattr(boto3, "client", spy)
    return created


def test_frontdoor_build_app_creates_only_the_clients_incoming_needs(aws_env):
    from lambdas.frontdoor import handler

    handler.build_app()
    assert aws_env == ["lambda"]  # KMS, Bedrock and Comprehend wait for first use


def test_frontdoor_quick_clients_make_one_attempt(aws_env):
    """The front door's backend Lambda client makes exactly one attempt (botocore's
    ``max_attempts`` counts retries; ``total_max_attempts`` counts attempts), so a slow
    dependency takes the front door's own apology path well inside Twilio's 15 s limit."""
    from lambdas.frontdoor import handler

    app = handler.build_app()
    cfg = app.d.backend._client.meta.config  # noqa: SLF001 - the built client's resolved config
    assert cfg.retries.get("total_max_attempts") == 1
    assert "max_attempts" not in cfg.retries
    assert (cfg.connect_timeout, cfg.read_timeout) == (2, 4)


def test_frontdoor_ignores_the_removed_harness_prewarm_setting(aws_env, monkeypatch):
    """The harness warm-up is gone; a stale ``OWLBANK_PREWARM_FUNCTION`` builds nothing."""
    from lambdas.frontdoor import handler

    monkeypatch.setenv("OWLBANK_PREWARM_FUNCTION", "arn:aws:lambda:us-east-1:1:function:warmer")
    app = handler.build_app()
    assert aws_env == ["lambda"]  # the backend client only
    assert not hasattr(app.d, "prewarm")


def test_frontdoor_lazy_clients_are_built_on_first_use(aws_env):
    from lambdas.frontdoor import handler

    app = handler.build_app()
    assert aws_env == ["lambda"]  # the resolver's client too waits for its first use
    app.d.resolver._client_for(1500).converse  # noqa: B018, SLF001 - first use creates the client
    app.d.sealer._kms.encrypt  # noqa: B018, SLF001
    assert sorted(aws_env) == ["bedrock-runtime", "kms", "lambda"]


def test_frontdoor_deterministic_ceiling_builds_no_resolver(aws_env, monkeypatch):
    """With ``OWLBANK_SLOT_MODE_CEILING=deterministic`` there is no resolver, so no
    Bedrock client can ever be built."""
    from lambdas.frontdoor import handler

    monkeypatch.setenv("OWLBANK_SLOT_MODE_CEILING", "deterministic")
    app = handler.build_app()
    assert app.cfg.slot_mode_ceiling == "deterministic" and app.d.resolver is None
    assert "bedrock-runtime" not in aws_env


def test_frontdoor_invalid_ceiling_fails_init(aws_env, monkeypatch):
    """Anything but the three modes fails INIT (synth already refuses it), without echoing it."""
    from lambdas.frontdoor import handler

    monkeypatch.setenv("OWLBANK_SLOT_MODE_CEILING", "bogus-value")
    with pytest.raises(ValueError, match="OWLBANK_SLOT_MODE_CEILING") as exc:
        handler.build_app()
    assert "bogus-value" not in str(exc.value)


# --------------------------------------------------------------------------- slot preload


def _catalog_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Every ``option.*``/``slot.*`` template read, recorded (key, lang)."""
    from owlbank.catalog import PromptCatalog

    reads: list[tuple[str, str]] = []
    real = PromptCatalog.template

    def spy(self: PromptCatalog, key: str, lang: str) -> str:
        if key.startswith(("option.", "slot.")):
            reads.append((key, lang))
        return real(self, key, lang)

    monkeypatch.setattr(PromptCatalog, "template", spy)
    return reads


def test_the_slot_modules_are_imported_with_the_handler():
    import sys

    from lambdas.frontdoor import handler

    names = {m.__name__ for m in handler.SLOT_MODULES}
    assert names == {f"owlbank.slots.{n}" for n in ("dialog", "generative", "grounding", "modes", "resolver")}
    assert names <= set(sys.modules)


def test_init_preloads_the_slot_texts_once_and_requests_do_not_again(monkeypatch, aws_env, caplog):
    """At cold start every ``option.*``/``slot.*`` text is read once per language it has; the
    handler then reuses the app, so a request reads none of them up front."""
    reads = _catalog_reads(monkeypatch)
    with caplog.at_level(logging.INFO):
        mod = _reload(monkeypatch, "lambdas.frontdoor.handler", INIT)
    try:
        app = mod._app  # noqa: SLF001
        assert app is not None
        cat = app.d.catalog
        want = sorted((k, lang) for k in cat.keys if k.startswith(("option.", "slot."))
                      for lang in cat.languages if cat.has(k, lang))
        assert want and sorted(reads) == want  # each exactly once
        assert any(k.startswith("option.card.") for k, _ in reads)
        (rec,) = [r for r in caplog.records if getattr(r, "event", "") == "slot_preload"]
        f = fields(rec)
        assert f["texts"] == len(want) and f["slots"] == len(app.d.policy.slots())
        assert set(f) >= {"slots", "texts", "languages", "duration_ms"}
        reads.clear()
        monkeypatch.setattr(mod, "new_app", lambda: pytest.fail("a request rebuilt the app"))
        monkeypatch.setattr(app, "handle", lambda event, **_kw: {"statusCode": 200})
        assert mod.handler({}, None) == {"statusCode": 200}
        assert reads == [] and mod._app is app  # noqa: SLF001
    finally:
        monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE")
        importlib.reload(mod)


def test_preload_runs_once_per_new_app(monkeypatch, aws_env):
    from lambdas.frontdoor import handler

    calls: list[object] = []
    monkeypatch.setattr(handler, "preload_slots", lambda app: calls.append(app))
    app = handler.new_app()
    assert calls == [app]


def test_a_catalog_read_failure_does_not_break_init(monkeypatch, aws_env, caplog):
    """The preload is best effort. A failing read is logged (class only, no message) and
    INIT still builds the app."""
    from owlbank.catalog import PromptCatalog

    def boom(self: PromptCatalog, key: str, lang: str) -> str:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(PromptCatalog, "template", boom)
    with caplog.at_level(logging.DEBUG):
        mod = _reload(monkeypatch, "lambdas.frontdoor.handler", INIT)
    try:
        assert mod._app is not None  # noqa: SLF001
        (rec,) = [r for r in caplog.records if getattr(r, "event", "") == "slot_preload_failed"]
        assert fields(rec)["decision"] == "ignored" and fields(rec)["error_class"] == "RuntimeError"
        assert "secret detail" not in caplog.text
        assert not [r for r in caplog.records if getattr(r, "event", "") == "slot_preload"]
    finally:
        monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE")
        importlib.reload(mod)


# --------------------------------------------------------------------------- import-time behaviour


def _reload(monkeypatch: pytest.MonkeyPatch, module_name: str, env: dict[str, str]):
    """Import ``module_name`` afresh as the Lambda runtime would, with ``env`` set; returns the module."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    mod = importlib.import_module(module_name)
    return importlib.reload(mod)


@pytest.mark.parametrize("module_name,attr", [("lambdas.frontdoor.handler", "_app"),
                                              ("lambdas.auth_service.handler", "_deps"),
                                              ("lambdas.bank_tools.handler", "_deps")])
def test_import_builds_nothing_outside_lambda(monkeypatch, aws_env, module_name, attr):
    monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE", raising=False)
    mod = _reload(monkeypatch, module_name, {})
    try:
        assert getattr(mod, attr) is None and aws_env == []
    finally:
        importlib.reload(mod)


@pytest.mark.parametrize("module_name,attr", [("lambdas.frontdoor.handler", "_app"),
                                              ("lambdas.auth_service.handler", "_deps"),
                                              ("lambdas.bank_tools.handler", "_deps")])
def test_import_inside_lambda_wires_up_during_init_without_calling_aws(monkeypatch, aws_env, module_name, attr):
    """INIT builds the dependencies (no secret is read), so the first request starts with them in place."""
    from owlbank import secrets

    monkeypatch.setattr(secrets, "get_secret", lambda *a, **k: pytest.fail("a secret was read at import"))
    mod = _reload(monkeypatch, module_name, INIT)
    try:
        assert getattr(mod, attr) is not None
    finally:
        monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE")
        importlib.reload(mod)
        assert getattr(mod, attr) is None


def test_auth_service_preloads_twilio_only_with_provisioned_concurrency(monkeypatch, aws_env):
    from lambdas.auth_service import verify_client

    loads: list[int] = []
    monkeypatch.setattr(verify_client, "preload_twilio_sdk", lambda: loads.append(1))
    mod = _reload(monkeypatch, "lambdas.auth_service.handler", INIT)
    assert loads == []  # on-demand: still lazy
    mod = _reload(monkeypatch, "lambdas.auth_service.handler", PC)
    assert loads == [1]
    monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE")
    importlib.reload(mod)


def test_a_failing_init_leaves_the_handler_to_build_on_the_first_request(monkeypatch, aws_env):
    monkeypatch.delenv("OWLBANK_RELAY_WS_URL")  # build_app needs it: INIT build fails
    mod = _reload(monkeypatch, "lambdas.frontdoor.handler", INIT)
    try:
        assert mod._app is None  # noqa: SLF001 - the first request retries (and raises the real error)
    finally:
        monkeypatch.delenv("AWS_LAMBDA_INITIALIZATION_TYPE")
        importlib.reload(mod)


def test_twilio_sdk_preload_is_imports_only(monkeypatch):
    import sys

    from lambdas.auth_service.verify_client import preload_twilio_sdk
    from owlbank import secrets

    monkeypatch.setattr(secrets, "get_secret", lambda *a, **k: pytest.fail("a secret was read"))
    preload_twilio_sdk()  # no client, no secret, no network: it must simply return
    assert "twilio.rest" in sys.modules


def test_preload_builds_the_resolver_client_for_the_policy_budget(aws_env):
    """A cold first resolver call took about 2.6 s against the 2.5 s budget (real call): the preload now builds the bedrock-runtime client at INIT, for the budget the
    policy holds, so the caller's first answer doesn't pay for it."""
    from lambdas.frontdoor import handler

    app = handler.build_app()
    assert "bedrock-runtime" not in aws_env
    handler.preload_slots(app)
    assert aws_env.count("bedrock-runtime") == 1
    budget = app.d.policy.slot_resolution.resolver_budget_ms
    app.d.resolver._client_for(budget).converse  # noqa: B018, SLF001 - the same, already built
    assert aws_env.count("bedrock-runtime") == 1


def test_preload_builds_no_bedrock_client_with_the_deterministic_ceiling(aws_env, monkeypatch):
    """With the deterministic ceiling there is no resolver, so the preload builds nothing."""
    from lambdas.frontdoor import handler

    monkeypatch.setenv("OWLBANK_SLOT_MODE_CEILING", "deterministic")
    app = handler.build_app()
    handler.preload_slots(app)
    assert "bedrock-runtime" not in aws_env
