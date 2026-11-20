"""The copies kept outside ``src/`` equal the bridge's and ``owlbank.failover``'s.

``infra/`` is its own uv project and can't import ``src/``; the preflight resolver
(``infra/scripts/resolve_models.py``) stays standalone. Both keep copies, pinned here:

* the resolver's error classification equals :func:`owlbank.failover.classify`, over the
  table of error codes and over generated codes and messages;
* the infra default list is the three models with measured thresholds in
  ``owlbank.failover.settings``, and its per-model defaults (thinking, text release, max tokens)
  equal the resolver's;
* the names infra alarms on equal what the bridge writes: the EMF namespace, dimensions and
  ``ActiveRank``, the ``model_unavailable`` line and its ``model_id`` field, the table and its
  partitions, the write budget, and the staleness limit against the heartbeat alarm;
* preflight's copies of ``owlbank.agent_prompt`` (the placeholder sentence, the
  call-language line, the catalog's language fallback, the prompt parts and their JSON encoding and
  hash, the model key) and of ``owlbank.prompt_validation`` (``suite_hash``, ``params_hash`` and its
  keys, the default ``runs``; the property test is in ``tests/property/test_preflight_prompts_props.py``);
* preflight's ``entry_params`` (the parameters it hashes for an entry of the list to deploy)
  equal the ``OWLBANK_MODELS`` entry synth builds from ``modelIds``, ``modelParams`` and
  ``promptCaching`` (but its caching, not hashed), and the runner's ``entry_params`` of that entry;
* agent prompt caching: preflight's copy of the placeholder allow-list (``AGENT_PREFIX``,
  ``ALLOWED_PLACEHOLDERS``, the placeholder pattern, ``allowed_placeholders``, ``placeholder_problems``).

``infra/owlbank_infra/config.py`` and the resolver import only the standard library and are loaded
by file path; ``model_failover.py`` needs CDK, so its constants are read from its syntax tree.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from lambdas.model_poller import service as poller
from owlbank import agent_prompt
from owlbank.catalog import PromptCatalog
from owlbank.failover import UNAVAILABLE, Outcome, classify
from owlbank.failover import fleet as failover_fleet
from owlbank.failover import settings as failover_settings
from owlbank.failover.classify import INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX
from owlbank.prompt_validation import DEFAULT_RUNS, PARAM_KEYS, RECORDS_VERSION
from owlbank.prompt_validation import params_hash as owl_params_hash
from owlbank.prompt_validation import suite_hash as owl_suite_hash
from tac_bridge import fleet as bridge_fleet
from tac_bridge import model_metrics
from tac_bridge.app import FLEET_TABLE_ENV
from tests.prompt_gate_strategies import LANGS, MODEL_KEYS, catalogs, current_hash

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra"


def _load(path: Path, name: str) -> Iterator[ModuleType]:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while building the class
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def infra_config() -> Iterator[ModuleType]:
    yield from _load(INFRA / "owlbank_infra" / "config.py", "_infra_config_for_failover_parity")


@pytest.fixture(scope="module")
def resolver() -> Iterator[ModuleType]:
    yield from _load(INFRA / "scripts" / "resolve_models.py", "_resolver_for_failover_parity")


@pytest.fixture(scope="module")
def preflight() -> Iterator[ModuleType]:
    yield from _load(INFRA / "scripts" / "preflight.py", "_preflight_for_failover_parity")


def _constants(path: Path) -> dict[str, Any]:
    """Module-level ``NAME = <literal>`` (and ``Duration.minutes(<n>)``, in seconds) of a module."""
    out: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Name) and value.func.value.id == "Duration"):
            unit = {"seconds": 1, "minutes": 60, "hours": 3600}[value.func.attr]
            out[target.id] = ast.literal_eval(value.args[0]) * unit
            continue
        try:
            out[target.id] = ast.literal_eval(value)
        except ValueError:
            continue
    return out


def _returned_format(path: Path, function: str) -> str:
    """The f-string a one-line function returns, as source (``f'{cfg.name_prefix}-bridge'``)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            (ret,) = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            assert ret.value is not None
            return ast.unparse(ret.value)
    raise AssertionError(f"no function {function}")


MODEL_FAILOVER = _constants(INFRA / "owlbank_infra" / "model_failover.py")


# --------------------------------------------------------------------------- classification


def _class_of(outcome: Any) -> str:
    """``classify``'s answer in the resolver's three classes."""
    if outcome == UNAVAILABLE:
        return "unavailable"
    if outcome is Outcome.REQUEST_ERROR:
        return "request_error"
    return "hard_error"


#: The error table (and the codes ``classify`` knows), one row per signal.
SECTION_5 = [
    ("ValidationException", INVALID_MODEL_MESSAGE),
    ("ResourceNotFoundException", "Model not found"),
    ("AccessDeniedException", "You don't have access to the model"),
    ("ValidationException", REQUEST_ERROR_PREFIX + " messages.1: invalid signature"),
    ("ThrottlingException", "Too many requests"),
    ("ServiceUnavailableException", ""),
    ("ModelTimeoutException", ""),
    ("ModelErrorException", ""),
    ("ModelStreamErrorException", ""),
    ("ModelNotReadyException", ""),
    ("InternalServerException", ""),
    ("EndpointConnectionError", ""),
    ("ConnectionClosedError", ""),
    ("ReadTimeoutError", ""),
    ("ContextWindowOverflowException", ""),
    ("throttlingException", "event-stream style"),
    ("", ""),
    ("SomethingNew", "unknown"),
]


@pytest.mark.parametrize(("code", "message"), SECTION_5)
def test_the_resolvers_classification_equals_classify_on_the_error_table(
        resolver: ModuleType, code: str, message: str) -> None:
    assert resolver.classify_error(code, message) == _class_of(classify(code, message))


def test_the_resolvers_fixed_texts_are_classifys(resolver: ModuleType) -> None:
    assert resolver.INVALID_ID_MESSAGE == INVALID_MODEL_MESSAGE
    assert resolver.REQUEST_ERROR_PREFIX == REQUEST_ERROR_PREFIX


_CODES = st.one_of(
    st.sampled_from([c for c, _ in SECTION_5]),
    st.sampled_from([c for c, _ in SECTION_5]).map(lambda c: c[:1].lower() + c[1:]),
    st.sampled_from([c for c, _ in SECTION_5]).map(lambda c: f" {c} "),
    st.text(max_size=40),
)
_MESSAGES = st.one_of(
    st.sampled_from([INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX, ""]),
    st.tuples(st.sampled_from([INVALID_MODEL_MESSAGE, REQUEST_ERROR_PREFIX, ""]), st.text(max_size=40),
              st.text(alphabet=" \t\n", max_size=3)).map(lambda t: t[2] + t[0] + t[1]),
    st.text(max_size=80),
)


@given(_CODES, _MESSAGES)
def test_the_resolvers_classification_equals_classify_on_any_error(resolver: ModuleType, code: str,
                                                                     message: str) -> None:
    """PBT: for any code and message the resolver and the bridge agree on the class."""
    assert resolver.classify_error(code, message) == _class_of(classify(code, message))


# --------------------------------------------------------------------------- default entries


def test_the_infra_default_list_is_the_models_with_measured_thresholds(infra_config: ModuleType) -> None:
    """Every default entry has its own trip, recover and deadline in ``owlbank.failover``;
    none falls back to the baseline-derived defaults."""
    ids = infra_config.default_model_ids("us-east-1")
    assert len(ids) == 3
    measured = [failover_settings.series_parts(m) for m in ids]
    assert measured == list(failover_settings._KNOWN)  # noqa: SLF001
    s = failover_settings.Settings.for_models(ids)
    for entry, key in zip(s.entries, measured, strict=True):
        trip, recover, deadline = failover_settings._KNOWN[key]  # noqa: SLF001
        assert (entry.trip_ttft_s, entry.recover_ttft_s) == (trip, recover)
    assert [e.deadline_s for e in s.entries] == [failover_settings._KNOWN[measured[0]][2],  # noqa: SLF001
                                                failover_settings._KNOWN[measured[1]][2], None]  # noqa: SLF001


@pytest.mark.parametrize("model_id", [
    "us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-5-5", "global.anthropic.claude-sonnet-5-5",
    "eu.anthropic.claude-sonnet-4-6", "us.anthropic.claude-3-haiku-20240307-v1:0",
])
def test_the_infra_and_resolver_entry_defaults_agree(infra_config: ModuleType, resolver: ModuleType,
                                                     model_id: str) -> None:
    entry = infra_config._model_entry(model_id, {}, False)  # noqa: SLF001
    assert resolver.default_params(model_id) == {"thinking": entry.thinking, "text_release": entry.text_release,
                                                 "max_tokens": entry.max_tokens}


@pytest.mark.parametrize("thinking", ["adaptive", "off", "default", "between_tools", "enabled:1024"])
def test_the_resolvers_thinking_fields_are_the_bridges(resolver: ModuleType, thinking: str) -> None:
    from tac_bridge.agent_client_strands import THINKING_PARAMS, thinking_request_fields

    assert set(resolver.THINKING_FIELDS) == set(THINKING_PARAMS)
    assert resolver.thinking_request_fields(thinking) == thinking_request_fields(thinking)


def test_the_infra_and_resolver_geographies_and_default_shape_agree(infra_config: ModuleType,
                                                                    resolver: ModuleType) -> None:
    assert tuple(resolver.REGION_GEOS) == tuple(infra_config.REGION_GEOS)
    families = [b.split("claude-")[1].split("-")[0] for b in infra_config.DEFAULT_MODEL_BASES]
    assert [f for f, _ in resolver.DEFAULT_SHAPE] == families


# --------------------------------------------------------------------------- the fleet and the alarms


def test_the_table_partitions_and_budget_agree() -> None:
    mf = MODEL_FAILOVER
    assert mf["TABLE"] == bridge_fleet.TABLE == poller.TABLE == "modelstats"
    assert bridge_fleet.TABLE_ENV == FLEET_TABLE_ENV == "OWLBANK_TABLE_MODELSTATS"
    assert (mf["SAMPLES_PARTITION"], mf["STATE_PARTITION"]) == (failover_fleet.SAMPLES_PK, failover_fleet.STATE_PK)
    assert mf["WRITE_BUDGET_PER_S"] == failover_fleet.WRITE_BUDGET_PER_S


def test_the_staleness_alarm_matches_the_bridges_staleness_limit() -> None:
    """Bridges stop using a record older than the limit; the alarm fires after as long without a
    heartbeat."""
    mf = MODEL_FAILOVER
    assert mf["HEARTBEAT_PERIOD"] * mf["HEARTBEAT_MISSING_PERIODS"] == bridge_fleet.STALE_AFTER_S


def test_the_bridges_metrics_are_the_ones_infra_alarms_on(infra_config: ModuleType) -> None:
    mf = MODEL_FAILOVER
    assert mf["METRIC_NAMESPACE_FORMAT"] == model_metrics.NAMESPACE_FORMAT
    assert poller.metric_namespace("dev") == model_metrics.NAMESPACE_FORMAT.format(stage="dev")
    assert mf["ACTIVE_RANK_METRIC"] == model_metrics.ACTIVE_RANK
    assert _returned_format(INFRA / "owlbank_infra" / "model_failover.py", "bridge_service") \
        == "f'{cfg.name_prefix}-bridge'"
    for stage in ("dev", "prod", "load-test"):
        cfg = infra_config.InfraConfig(stage=stage)
        metrics = model_metrics.ModelMetrics(stage)
        assert metrics.service == f"{cfg.name_prefix}-bridge"
        rec = metrics.record(model_metrics.ACTIVE_RANK, 1)
        assert rec["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [["Service", "Environment"]]
        assert (rec["Service"], rec["Environment"]) == (f"{cfg.name_prefix}-bridge", cfg.stage)
        # The bridge has no stage setting: it reads it from the fleet table's name.
        assert model_metrics.metric_stage({FLEET_TABLE_ENV: cfg.failover_table_name}) == stage


def test_the_unavailable_alarm_filters_on_the_bridges_line() -> None:
    """4a.4: the log metric filters match ``$.event = "model_unavailable"`` and ``$.model_id``; the
    client's line carries both (``test_strands_failover_reporting``:
    ``test_an_unavailable_entry_is_logged_for_the_alarm``)."""
    assert MODEL_FAILOVER["UNAVAILABLE_EVENT"] == "model_unavailable"
    assert MODEL_FAILOVER["UNAVAILABLE_ENTRY_FIELD"] == "model_id"


def test_the_metric_filter_field_is_exempt_from_scrubbing():
    """The PreferredModelUnavailable filter compares this log field with the configured ID, so the
    scrubber must leave a (dated) model ID in it alone."""
    from owlbank import obs

    assert MODEL_FAILOVER["UNAVAILABLE_ENTRY_FIELD"] in obs.MODEL_ID_FIELDS


# --------------------------------------------------------------------------- per-model prompts


def test_preflights_prompt_constants_are_agent_prompts(preflight: ModuleType) -> None:
    assert dict(agent_prompt.PLACEHOLDER_SENTENCE) == preflight.PLACEHOLDER_SENTENCE
    assert preflight.DEFAULT_PLACEHOLDER_LANG == agent_prompt.DEFAULT_PLACEHOLDER_LANG
    assert preflight.CALL_LANGUAGE_LINE == agent_prompt.CALL_LANGUAGE_LINE
    assert preflight.SYSTEM_KEY_PREFIX == agent_prompt.SYSTEM_KEY_PREFIX
    assert (preflight.WEB_ACTIVITY_KEY, preflight.WEB_ACTIVITY_CONTEXT_KEY) == agent_prompt.WEB_BLOCK_KEYS
    assert (preflight.CALLER_CONTEXT_KEY, preflight.CALLER_CONTEXT_UNVERIFIED_KEY) == agent_prompt.CALLER_CONTEXT_KEYS
    assert dict(agent_prompt.UNKNOWN_NAME) == preflight.UNKNOWN_NAME
    assert dict(agent_prompt.EMPTY_VALUE) == preflight.EMPTY_VALUE
    assert dict(agent_prompt.FALLBACK_CONTEXT) == preflight.FALLBACK_CONTEXT
    assert preflight.RECORDS_VERSION == RECORDS_VERSION
    assert preflight.DEFAULT_SUITE_RUNS == DEFAULT_RUNS
    assert preflight.PARAM_KEYS == PARAM_KEYS


def test_preflights_placeholder_allow_list_is_agent_prompts(preflight: ModuleType) -> None:
    """Agent prompt caching: the allow-list copy, its prefix and the placeholder pattern."""
    from owlbank.catalog import PLACEHOLDER

    assert preflight.AGENT_PREFIX == agent_prompt.AGENT_PREFIX
    assert dict(agent_prompt.ALLOWED_PLACEHOLDERS) == preflight.ALLOWED_PLACEHOLDERS
    assert preflight.PLACEHOLDER.pattern == PLACEHOLDER.pattern
    assert preflight.PLACEHOLDER.flags == PLACEHOLDER.flags


_D5_KEYS = st.one_of(
    st.sampled_from(["agent.system.haiku-5-5", "agent.system.opus-5-5", "agent.web_activity",
                     *agent_prompt.CALLER_CONTEXT_KEYS, "agent.", "agent", "agentx", "greeting.welcome"]),
    st.text(max_size=20))
_D5_TEXTS = st.one_of(
    st.lists(st.sampled_from(["x ", "{firstName}", "{profile}", "{memory}", "{other}", "{", "}", "{{}}", "{1a}"]),
             max_size=6).map("".join),
    st.text(max_size=20))


@given(_D5_KEYS, _D5_TEXTS)
def test_preflights_placeholder_problems_equal_agent_prompts(preflight: ModuleType, key: str, text: str) -> None:
    """Agent prompt caching: the standalone ``placeholder_problems`` and
    ``allowed_placeholders`` give the same verdict and reason for any key and text."""
    assert preflight.allowed_placeholders(key) == agent_prompt.allowed_placeholders(key)
    assert preflight.placeholder_problems(key, text) == agent_prompt.placeholder_problems(key, text)


def test_preflight_reads_the_repositorys_files_as_the_seed_does(preflight: ModuleType) -> None:
    """The default ``--data-dir`` is the repository's ``data/`` and the file names are the seed's."""
    from scripts import seed
    from scripts._common import DATA_DIR

    assert preflight.DATA_DIR == DATA_DIR
    assert (preflight.PROMPT_VALIDATION_FILE, preflight.PROMPT_SUITE_FILE) == (
        seed.PROMPT_VALIDATION_FILE, seed.PROMPT_SUITE_FILE)
    assert preflight.PROMPT_CATALOG_FILE == "prompt_catalog.json"


@given(catalogs(), st.sampled_from([*MODEL_KEYS, "opus-4-1", "us.anthropic.claude-opus-4"]),
       st.sampled_from([*LANGS, "xx-XX"]))
def test_preflights_prompt_hash_equals_agent_prompts(preflight: ModuleType, doc: dict[str, Any],
                                                     model_key: str, lang: str) -> None:
    """PBT: over catalogs with missing languages, single-language keys, drafts, versions, missing
    web blocks or caller-context frames and non-ASCII text, the copy serves the same 10 parts and
    hashes them alike (``None`` where the bridge raises ``CatalogKeyError``)."""
    catalog = PromptCatalog.from_dict(doc)
    copy = preflight.PromptCatalogCopy(doc)
    assert preflight.prompt_hash(copy, model_key, lang) == current_hash(catalog, model_key, lang)
    want = current_hash(catalog, model_key, lang)
    if want is not None:
        parts = agent_prompt.prompt_parts(catalog, model_key, lang)
        assert preflight.prompt_parts(copy, model_key, lang) == list(dataclasses.astuple(parts))


def test_preflights_prompt_hash_equals_agent_prompts_on_the_repositorys_catalog(preflight: ModuleType) -> None:
    doc = json.loads((ROOT / "data" / "prompt_catalog.json").read_text(encoding="utf-8"))
    catalog, copy = PromptCatalog.from_dict(doc), preflight.PromptCatalogCopy(doc)
    for model_key in MODEL_KEYS:
        for lang in catalog.languages:
            assert preflight.prompt_hash(copy, model_key, lang) == agent_prompt.prompt_hash(catalog, model_key, lang)


_IDS = st.one_of(
    st.builds(lambda pre, fam, ma, mi, tail: f"{pre}anthropic.claude-{fam}-{ma}-{mi}{tail}",
              st.sampled_from(["", "us.", "global.", "eu.", "US.", "us-gov."]),
              st.sampled_from(["haiku", "sonnet", "opus", "Haiku"]),
              st.integers(0, 12), st.integers(0, 120),
              st.sampled_from(["", "-20251001", "-20251001-v1:0", "-v2", "-v1:0", "-x", "-2025"])),
    st.sampled_from(["us.anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-sonnet-5",
                     "us.anthropic.claude-haiku-5-5", "meta.llama3"]),
    st.text(max_size=30),
)


@given(_IDS)
def test_preflights_model_key_equals_agent_prompts(preflight: ModuleType, model_id: str) -> None:
    assert preflight.prompt_model_key(model_id) == agent_prompt.model_key(model_id)


_JSON = st.recursive(st.none() | st.booleans() | st.integers() | st.text(max_size=8),
                     lambda inner: st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner,
                                                                                 max_size=3),
                     max_leaves=12)


@given(_JSON)
def test_preflights_suite_hash_equals_prompt_validations(preflight: ModuleType, doc: Any) -> None:
    assert preflight.suite_hash(doc) == owl_suite_hash(doc)


_PARAMS = st.fixed_dictionaries({
    "thinking": st.sampled_from(["adaptive", "off", "default", "enabled:2048"]) | st.text(max_size=8),
    "text_release": st.sampled_from(["stream", "message"]) | st.text(max_size=8),
    "max_tokens": st.integers(1, 9000),
}, optional={"id": st.text(max_size=12), "prompt_caching": st.booleans(), "other": _JSON})


@given(_PARAMS)
def test_preflights_params_hash_equals_prompt_validations(preflight: ModuleType, params: dict[str, Any]) -> None:
    assert preflight.params_hash(params) == owl_params_hash(params)


_ENTRY_IDS = ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0",
              "us.anthropic.claude-sonnet-5-5", "us.anthropic.claude-sonnet-4-6"]
_OVERRIDES = st.fixed_dictionaries({}, optional={
    "thinking": st.sampled_from(["adaptive", "off", "enabled:2048", " Adaptive ", "OFF"]),
    "text_release": st.sampled_from(["stream", "message", "Message"]),
    "max_tokens": st.sampled_from([64, 400, 1500, 4096]),
    "prompt_caching": st.sampled_from([True, False, "on", "off", " ON "]),
})


@given(st.lists(st.sampled_from(_ENTRY_IDS), min_size=1, max_size=3, unique=True), st.data(), st.booleans())
def test_preflights_entry_params_are_synths_owlbank_models_entry(infra_config: ModuleType, preflight: ModuleType,
                                                                 ids: list[str], data: st.DataObject,
                                                                 caching: bool) -> None:
    """PBT: for any list, ``modelParams`` and ``promptCaching`` synth accepts, the parameters
    preflight hashes per entry are the ``OWLBANK_MODELS`` entry's (``ModelEntry.as_env``) fields of
    ``PARAM_KEYS``, which the runner reads (``--models``) and hashes with
    :func:`scripts.prompt_suite.entry_params`. Preflight's input is ``modelParams``; ``promptCaching``
    (and a ``prompt_caching`` override) is not hashed, so the same parameters result whatever
    synth's caching."""
    from scripts import prompt_suite as ps

    overrides = {mid: data.draw(_OVERRIDES) for mid in ids if data.draw(st.booleans())}
    ctx = {"modelIds": ids, "modelParams": overrides}
    try:
        entries = infra_config._models(ctx, region="us-east-1",  # noqa: SLF001
                                       prompt_caching=caching)
    except infra_config.ConfigError:
        assume(False)  # a parameter set synth refuses is never deployed
        return
    env = ps.parse_models(json.dumps([e.as_env() for e in entries]))
    for e, bridge_entry in zip(entries, env, strict=True):
        synth = {k: v for k, v in e.as_env().items() if k in PARAM_KEYS}
        assert set(e.as_env()) - set(PARAM_KEYS) == {"id", "prompt_caching"}
        assert preflight.entry_params(e.id, overrides) == synth
        assert ps.entry_params(bridge_entry) == synth
