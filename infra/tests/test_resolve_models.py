"""scripts/resolve_models.py against fake Bedrock control-plane and runtime clients: no AWS account,
no network. The version ordering has a property-based test (the ordering is a total order consistent
with (family, major, minor, date)); the rest is I/O glue around it, covered by the cases below."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from hypothesis import given
from hypothesis import strategies as st

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("resolve_models", SCRIPTS / "resolve_models.py")
rm = importlib.util.module_from_spec(spec)
sys.modules["resolve_models"] = rm  # dataclasses need the module registered
spec.loader.exec_module(rm)

pspec = importlib.util.spec_from_file_location("preflight_for_models", SCRIPTS / "preflight.py")
preflight = importlib.util.module_from_spec(pspec)
sys.modules["preflight_for_models"] = preflight
pspec.loader.exec_module(preflight)

HAIKU55 = "anthropic.claude-haiku-5-5"
HAIKU45 = "anthropic.claude-haiku-4-5-20251001-v1:0"
HAIKU3 = "anthropic.claude-3-haiku-20240307-v1:0"
SONNET55 = "anthropic.claude-sonnet-5-5"
SONNET5 = "anthropic.claude-sonnet-5"
SONNET46 = "anthropic.claude-sonnet-4-6"
SONNET45 = "anthropic.claude-sonnet-4-5-20250929-v1:0"
SONNET35 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
EOL = dt.datetime(2027, 4, 8, tzinfo=dt.UTC)

#: Foundation model -> lifecycle, as ListFoundationModels reports them in this account.
CATALOG = {
    HAIKU55: {"status": "ACTIVE"}, HAIKU45: {"status": "ACTIVE"}, HAIKU3: {"status": "ACTIVE"},
    SONNET55: {"status": "ACTIVE"}, SONNET5: {"status": "ACTIVE"}, SONNET46: {"status": "ACTIVE"},
    SONNET45: {"status": "LEGACY", "endOfLifeTime": EOL}, SONNET35: {"status": "ACTIVE"},
    "anthropic.claude-opus-4-1-20250805-v1:0": {"status": "ACTIVE"},
}


def profile_ids(*prefixes: str) -> list[str]:
    return [f"{p}.{m}" for p in prefixes for m in CATALOG]


def client_error(code: str, message: str, op: str = "Converse") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, op)


class FakeBedrock:
    """The control plane: two list calls (profiles over two pages), and the availability call the
    resolver must never rely on (the request is the test)."""

    def __init__(self, catalog=None, profiles=None):
        self.catalog = CATALOG if catalog is None else catalog
        self.profiles = profile_ids("us", "eu", "global") if profiles is None else profiles
        self.calls: list[str] = []

    def list_foundation_models(self, byProvider):
        assert byProvider == "Anthropic"
        self.calls.append("list_foundation_models")
        return {"modelSummaries": [{"modelId": m, "modelLifecycle": lc} for m, lc in self.catalog.items()]}

    def list_inference_profiles(self, typeEquals, nextToken=None):
        assert typeEquals == "SYSTEM_DEFINED"
        self.calls.append("list_inference_profiles")
        half = len(self.profiles) // 2
        chunk = self.profiles[:half] if nextToken is None else self.profiles[half:]
        page = {"inferenceProfileSummaries": [{"inferenceProfileId": p, "status": "ACTIVE"} for p in chunk]}
        if nextToken is None:
            page["nextToken"] = "page2"
        return page

    def get_foundation_model_availability(self, modelId):
        self.calls.append("get_foundation_model_availability")
        return {"agreementAvailability": {"status": "NOT_AVAILABLE"}}


class FakeRuntime:
    """Bedrock runtime: ``errors`` maps a model ID to the error its request gets."""

    def __init__(self, errors=None):
        self.errors = errors or {}
        self.requests: list[dict] = []

    def converse(self, **kw):
        self.requests.append(kw)
        if kw["modelId"] in self.errors:
            raise self.errors[kw["modelId"]]
        return {"output": {"message": {"role": "assistant", "content": [{"text": "k"}]}}}

    @property
    def tried(self) -> list[str]:
        return [r["modelId"] for r in self.requests]


INVALID_ID = client_error("ValidationException", "The provided model identifier is invalid.")


def resolve(runtime=None, bedrock=None, **kw):
    kw.setdefault("region", "us-east-1")
    return rm.resolve(bedrock=bedrock or FakeBedrock(), runtime=runtime or FakeRuntime(), **kw)


def lines(res, level):
    return [f for f in res.findings if f.level == level]


# --------------------------------------------------------------------------- the default list


def test_default_list_is_highest_haiku_next_haiku_highest_sonnet_in_the_geography():
    rt = FakeRuntime()
    res = resolve(rt)
    assert res.model_ids == ["us.anthropic.claude-haiku-5-5", f"us.{HAIKU45}", "us.anthropic.claude-sonnet-5-5"]
    assert res.ok
    assert not lines(res, "FAIL") and not lines(res, "WARN")
    assert rt.tried == res.model_ids  # each entry got the synthetic request, no other model did
    assert all(r["inferenceConfig"] == {"maxTokens": 1} for r in rt.requests)
    assert res.deploy_args() == ("-c 'modelIds=[\"us.anthropic.claude-haiku-5-5\",\"us." + HAIKU45
                                 + "\",\"us.anthropic.claude-sonnet-5-5\"]'")


def test_entry_defaults_follow_the_known_models():
    res = resolve()
    haiku55, haiku45, sonnet55 = res.entries
    assert (haiku55.thinking, haiku55.text_release, haiku55.max_tokens) == ("adaptive", "stream", 2000)
    assert (haiku45.thinking, haiku45.text_release, haiku45.max_tokens) == ("off", "message", 1000)
    assert (sonnet55.thinking, sonnet55.text_release) == ("adaptive", "stream")


def test_profiles_are_read_across_pages_and_only_the_geography_is_a_candidate():
    res = resolve(region="eu-west-1")
    assert all(m.startswith("eu.") for m in res.model_ids)


def test_unknown_region_geography_is_a_fail():
    res = resolve(region="sa-east-1")
    assert not res.ok and "sa-east-1" in lines(res, "FAIL")[0].detail


@pytest.mark.parametrize("failing", ["list_foundation_models", "list_inference_profiles"])
@pytest.mark.parametrize("code", ["AccessDeniedException", "ThrottlingException"])
def test_a_failing_catalog_listing_is_a_fail_naming_the_call_and_code_not_a_traceback(failing, code):
    class Denied(FakeBedrock):
        def __getattribute__(self, name):
            if name == failing:
                def boom(**kw):
                    raise client_error(code, "nope", op=failing)
                return boom
            return super().__getattribute__(name)

    res = resolve(bedrock=Denied())
    fails = lines(res, "FAIL")
    assert not res.ok and len(fails) == 1
    assert failing in fails[0].detail and code in fails[0].detail


def test_preflight_check_models_returns_a_fail_result_when_the_catalog_listing_fails():
    class Denied(FakeBedrock):
        def list_foundation_models(self, byProvider):
            raise client_error("AccessDeniedException", "no", op="ListFoundationModels")

    results = preflight.check_models(Denied(), FakeRuntime(), "us-east-1")
    assert [r.ok for r in results] == [False]
    assert "AccessDeniedException" in results[0].detail


# --------------------------------------------------------------------------- replacement


def test_missing_model_is_replaced_by_the_next_of_its_series():
    rt = FakeRuntime({f"us.{HAIKU45}": INVALID_ID})
    res = resolve(rt)
    assert res.model_ids == ["us.anthropic.claude-haiku-5-5", f"us.{HAIKU3}", "us.anthropic.claude-sonnet-5-5"]
    (warn,) = lines(res, "WARN")
    assert f"us.{HAIKU45}" in warn.name and f"us.{HAIKU3}" in warn.detail and "unavailable" in warn.detail
    assert res.ok


def test_replacement_never_duplicates_a_listed_entry():
    rt = FakeRuntime({"us.anthropic.claude-haiku-5-5": INVALID_ID})
    res = resolve(rt)
    # Haiku 4.5 is already the second entry, so the first entry's replacement is the next one down.
    assert res.model_ids == [f"us.{HAIKU3}", f"us.{HAIKU45}", "us.anthropic.claude-sonnet-5-5"]
    assert rt.tried.count("us.anthropic.claude-haiku-5-5") == 1  # the failed entry is never retried
    assert len(set(res.model_ids)) == 3


def test_a_replacement_that_fails_too_is_skipped_for_the_next():
    rt = FakeRuntime({f"us.{HAIKU45}": INVALID_ID, f"us.{HAIKU3}": INVALID_ID})
    res = resolve(rt)
    assert not res.ok
    assert res.model_ids == ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-sonnet-5-5"]
    (fail,) = lines(res, "FAIL")
    assert f"us.{HAIKU45}" in fail.name and "no haiku replacement" in fail.detail


def test_a_hard_error_also_replaces_but_a_request_error_does_not():
    throttled = client_error("ThrottlingException", "Too many requests")
    res = resolve(FakeRuntime({"us.anthropic.claude-sonnet-5-5": throttled}))
    assert res.model_ids[-1] == "us.anthropic.claude-sonnet-5"
    assert "hard_error: ThrottlingException" in lines(res, "WARN")[0].detail

    bad = client_error("ValidationException", "The model returned the following errors: nope")
    res = resolve(FakeRuntime({"us.anthropic.claude-sonnet-5-5": bad}))
    assert not res.ok and "us.anthropic.claude-sonnet-5-5" not in res.model_ids
    assert "us.anthropic.claude-sonnet-5" not in res.model_ids  # no replacement tried


@pytest.mark.parametrize(("thinking", "fields", "max_tokens"), [
    ("adaptive", rm.THINKING_FIELDS["adaptive"], 1),
    ("off", {"thinking": {"type": "disabled"}}, 1),
    ("between_tools", {"thinking": {"type": "between_tools"}}, 1),
    ("enabled:1024", {"thinking": {"type": "enabled", "budget_tokens": 1024}}, 1025),
    ("default", None, 1),
])
def test_the_synthetic_request_carries_the_entrys_thinking_fields(thinking, fields, max_tokens):
    rt = FakeRuntime()
    sonnet = "us.anthropic.claude-sonnet-5-5"
    params = {sonnet: {"thinking": thinking, "text_release": "message", "max_tokens": 2000}}
    res = resolve(rt, model_ids=[sonnet], params=params)
    assert res.ok
    (req,) = rt.requests
    assert req.get("additionalModelRequestFields") == fields
    assert req["inferenceConfig"] == {"maxTokens": max_tokens}


def test_a_thinking_setting_the_model_refuses_fails_naming_it():
    # Sonnet 5.5 refuses ``disabled``, and the refusal is a request error.
    refused = client_error("ValidationException", "The model returned the following errors: To turn thinking "
                           "off on this model, send \"thinking\": {\"type\": \"between_tools\"} instead")

    class RefusesOff(FakeRuntime):
        def converse(self, **kw):
            self.requests.append(kw)
            if kw.get("additionalModelRequestFields") == {"thinking": {"type": "disabled"}}:
                raise refused
            return {"output": {"message": {"role": "assistant", "content": [{"text": "k"}]}}}

    sonnet = "us.anthropic.claude-sonnet-5-5"
    res = resolve(RefusesOff(), model_ids=[sonnet],
                  params={sonnet: {"thinking": "off", "text_release": "message", "max_tokens": 400}})
    assert not res.ok
    assert any("thinking=off" in f.detail for f in lines(res, "FAIL"))
    res = resolve(RefusesOff(), model_ids=[sonnet],
                  params={sonnet: {"thinking": "between_tools", "text_release": "message", "max_tokens": 400}})
    assert res.ok and res.model_ids == [sonnet]


def test_fail_when_no_entry_works():
    rt = FakeRuntime(dict.fromkeys((f"us.{m}" for m in CATALOG), INVALID_ID))
    res = resolve(rt)
    assert not res.ok and res.entries == []
    assert any(f.name == "Model list" and "no entry answered" in f.detail for f in lines(res, "FAIL"))


def test_sonnet_55_is_kept_although_the_control_plane_says_the_agreement_is_unavailable():
    bedrock = FakeBedrock()
    res = resolve(FakeRuntime(), bedrock, model_ids=["us.anthropic.claude-sonnet-5-5"])
    assert res.model_ids == ["us.anthropic.claude-sonnet-5-5"] and res.ok
    assert "get_foundation_model_availability" not in bedrock.calls  # never consulted: the request is the test
    assert not lines(res, "WARN")


# --------------------------------------------------------------------------- residency and warnings


def test_another_geographys_profile_is_refused_from_us_east_1():
    rt = FakeRuntime()
    res = resolve(rt, model_ids=["eu.anthropic.claude-haiku-5-5", "us.anthropic.claude-sonnet-5-5"])
    assert not res.ok
    (fail,) = lines(res, "FAIL")
    assert "eu.anthropic.claude-haiku-5-5" in fail.name and "another geography" in fail.detail
    assert rt.tried == ["us.anthropic.claude-sonnet-5-5"]  # never called, never replaced
    assert res.model_ids == ["us.anthropic.claude-sonnet-5-5"]


def test_bare_model_id_is_refused():
    res = resolve(model_ids=[HAIKU55])
    assert not res.ok and "inference profile" in lines(res, "FAIL")[0].detail


def test_global_entry_is_warned_and_kept():
    res = resolve(model_ids=["global.anthropic.claude-haiku-5-5", "us.anthropic.claude-sonnet-5-5"])
    assert res.ok and res.model_ids[0] == "global.anthropic.claude-haiku-5-5"
    (warn,) = lines(res, "WARN")
    assert "global.anthropic.claude-haiku-5-5" in warn.name and "any commercial Region" in warn.detail


def test_a_failing_global_entry_is_replaced_by_a_geographic_one():
    rt = FakeRuntime({"global.anthropic.claude-haiku-5-5": INVALID_ID})
    res = resolve(rt, model_ids=["global.anthropic.claude-haiku-5-5"])
    assert res.model_ids == [f"us.{HAIKU45}"]  # the proposal stays inside the geography


def test_legacy_model_is_warned_with_its_end_of_life_date():
    res = resolve(model_ids=[f"us.{SONNET45}"])
    assert res.ok
    (warn,) = lines(res, "WARN")
    assert "LEGACY" in warn.detail and "2027-04-08" in warn.detail


def test_legacy_model_without_a_published_date_says_so():
    catalog = {**CATALOG, SONNET45: {"status": "LEGACY"}}
    res = resolve(bedrock=FakeBedrock(catalog), model_ids=[f"us.{SONNET45}"])
    assert "no end-of-life date published" in lines(res, "WARN")[0].detail


def test_legacy_models_are_never_proposed_as_replacements():
    rt = FakeRuntime({"us.anthropic.claude-sonnet-5-5": INVALID_ID, "us.anthropic.claude-sonnet-5": INVALID_ID,
                      "us.anthropic.claude-sonnet-4-6": INVALID_ID})
    res = resolve(rt, model_ids=["us.anthropic.claude-sonnet-5-5"])
    assert res.model_ids == [f"us.{SONNET35}"]  # skips the LEGACY Sonnet 4.5 between 4.6 and 3.5
    assert f"us.{SONNET45}" not in rt.tried


# --------------------------------------------------------------------------- parameters


def test_streaming_text_without_reasoning_is_refused_naming_the_entry():
    rt = FakeRuntime()
    res = resolve(rt, model_ids=[f"us.{HAIKU45}", "us.anthropic.claude-haiku-5-5"],
                  params={f"us.{HAIKU45}": {"text_release": "stream"}})
    assert not res.ok
    (fail,) = lines(res, "FAIL")
    assert f"us.{HAIKU45}" in fail.name and "text_release=stream" in fail.detail
    assert rt.tried == ["us.anthropic.claude-haiku-5-5"]  # nothing sent for the invalid entry
    assert res.deploy_args().count("modelParams") == 1


@pytest.mark.parametrize("override", [
    {"thinking": "turbo"}, {"thinking": "enabled:abc"}, {"thinking": "enabled:10"},
    {"thinking": "enabled:5000", "max_tokens": 2000}, {"text_release": "chunks"},
    {"max_tokens": 0}, {"max_tokens": True},
])
def test_invalid_parameter_sets_are_refused(override):
    res = resolve(model_ids=["us.anthropic.claude-haiku-5-5"], params={"us.anthropic.claude-haiku-5-5": override})
    assert not res.ok and "invalid parameters" in lines(res, "FAIL")[0].detail


def test_valid_override_is_used_and_echoed():
    mid = "us.anthropic.claude-haiku-5-5"
    res = resolve(model_ids=[mid], params={mid: {"thinking": "enabled:1500", "max_tokens": 4000}})
    assert res.entries[0].thinking == "enabled:1500" and res.entries[0].max_tokens == 4000
    assert res.deploy_args().endswith("'modelParams=" + json.dumps({mid: res.params[mid]}, separators=(",", ":")) + "'")


@pytest.mark.parametrize(("mid", "thinking", "max_tokens"), [
    ("us.anthropic.claude-haiku-5-5", "off", 1000),           # reasoning default turned off
    ("us.anthropic.claude-haiku-4-5", "enabled:1500", 2000),  # off default turned on (1000 would refuse)
])
def test_max_tokens_follows_an_overridden_thinking_as_synth(mid, thinking, max_tokens):
    res = resolve(model_ids=[mid], params={mid: {"thinking": thinking, "text_release": "message"}})
    assert res.ok and res.entries[0].max_tokens == max_tokens


def test_list_length_is_one_to_three():
    res = resolve(model_ids=[f"us.{m}" for m in (HAIKU55, HAIKU45, HAIKU3, SONNET55)])
    assert not res.ok and "one to three" in lines(res, "FAIL")[0].detail


def test_model_that_cannot_be_ordered_is_refused():
    res = resolve(model_ids=["us.anthropic.claude-opus-4-1-20250805-v1:0"])
    assert res.ok  # an Opus ID parses; it only has no series to draw replacements from
    res = resolve(model_ids=["us.anthropic.titan-text"])
    assert not res.ok and "can order" in lines(res, "FAIL")[0].detail


# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize(("code", "message", "expected"), [
    ("ValidationException", "The provided model identifier is invalid.", rm.UNAVAILABLE),
    ("ResourceNotFoundException", "anything", rm.UNAVAILABLE),
    ("AccessDeniedException", "anything", rm.UNAVAILABLE),
    ("ValidationException", "The model returned the following errors: bad field", rm.REQUEST_ERROR),
    ("ValidationException", "The provided model identifier is invalid. Extra", rm.REQUEST_ERROR),
    ("ValidationException", "any other validation message", rm.REQUEST_ERROR),
    ("validationException", "The provided model identifier is invalid.", rm.UNAVAILABLE),
    ("ContextWindowOverflowException", "x", rm.REQUEST_ERROR),
    ("ThrottlingException", "x", rm.HARD_ERROR),
    ("ServiceUnavailableException", "x", rm.HARD_ERROR),
    ("ModelTimeoutException", "x", rm.HARD_ERROR),
    ("ModelErrorException", "x", rm.HARD_ERROR),
    ("ModelStreamErrorException", "x", rm.HARD_ERROR),
    ("ModelNotReadyException", "x", rm.HARD_ERROR),
    ("EndpointConnectionError", "x", rm.HARD_ERROR),
])
def test_classification_table(code, message, expected):
    assert rm.classify_error(code, message) == expected


@given(st.text(), st.text())
def test_classify_never_raises(code, message):
    assert rm.classify_error(code, message) in (rm.UNAVAILABLE, rm.REQUEST_ERROR, rm.HARD_ERROR)


@given(st.text())
def test_request_errors_are_never_unavailable(tail):
    msg = rm.REQUEST_ERROR_PREFIX + tail
    assert rm.classify_error("ValidationException", msg) == rm.REQUEST_ERROR


# --------------------------------------------------------------------------- version ordering

#: Every model ID seen in this account, highest version first within each family.
SEEN_IN_ORDER = {
    "haiku": ["anthropic.claude-haiku-5-5", HAIKU45, HAIKU3],
    "sonnet": [SONNET55, SONNET5, SONNET46, SONNET45, SONNET35],
}


@pytest.mark.parametrize("family", ["haiku", "sonnet"])
def test_version_ordering_over_the_ids_seen_in_this_account(family):
    ids = SEEN_IN_ORDER[family]
    for prefix in ("", "us.", "eu.", "global."):
        shuffled = [prefix + i for i in reversed(ids)]
        assert sorted(shuffled, key=rm.parse_version, reverse=True) == [prefix + i for i in ids]


def test_both_id_styles_and_the_old_style_parse():
    assert rm.parse_version("us.anthropic.claude-haiku-4-5-20251001-v1:0") == ("haiku", 4, 5, 20251001, (1, 0))
    assert rm.parse_version("anthropic.claude-sonnet-5") == ("sonnet", 5, 0, 0, (0, 0))
    assert rm.parse_version("anthropic.claude-sonnet-4-20250514-v1:0") == ("sonnet", 4, 0, 20250514, (1, 0))
    assert rm.parse_version("anthropic.claude-3-haiku-20240307-v1:0") == ("haiku", 3, 0, 20240307, (1, 0))
    assert rm.parse_version("anthropic.claude-3-5-sonnet-20241022-v2:0") == ("sonnet", 3, 5, 20241022, (2, 0))
    assert rm.parse_version("amazon.titan-text-v1") is None
    assert rm.parse_version("anthropic.claude-instant-v1") is None


_dates = st.one_of(st.just(0), st.dates(min_value=dt.date(2023, 1, 1), max_value=dt.date(2030, 1, 1))
                   .map(lambda d: int(d.strftime("%Y%m%d"))))
_keys = st.tuples(st.sampled_from(["haiku", "sonnet", "opus"]), st.integers(1, 12), st.integers(0, 9), _dates)


def render(key, style: str, prefix: str) -> str:
    family, major, minor, date = key
    tail = f"-{date}-v1:0" if date else ""
    if style == "old":  # claude-3-5-sonnet-<date>
        mid = f"{major}-{minor}-{family}" if minor else f"{major}-{family}"
    else:  # claude-sonnet-5-5-<date>
        mid = f"{family}-{major}-{minor}" if minor or style == "new_explicit_minor" else f"{family}-{major}"
    return f"{prefix}anthropic.claude-{mid}{tail}"


@given(st.lists(_keys, min_size=2, max_size=12, unique=True), st.sampled_from(["old", "new", "new_explicit_minor"]),
       st.sampled_from(["", "us.", "global."]), st.randoms())
def test_ordering_is_a_total_order_consistent_with_the_version_tuples(keys, style, prefix, rnd):
    # Property: parsing recovers the tuple, and sorting by the parsed key sorts the tuples (a total
    # order: any two distinct tuples compare, antisymmetrically and transitively, as the tuples do).
    parsed = {}
    for k in keys:
        v = rm.parse_version(render(k, style, prefix))
        assert v is not None
        assert (v.family, v.major, v.minor, v.date) == k
        parsed[k] = v
    shuffled = list(keys)
    rnd.shuffle(shuffled)
    by_key = sorted(shuffled, key=lambda k: parsed[k])
    by_family: dict[str, list] = {}
    for k in by_key:
        by_family.setdefault(k[0], []).append(k)
    for family, ordered in by_family.items():
        assert ordered == sorted(k for k in keys if k[0] == family), family
    for a in keys:
        for b in keys:
            if a[0] == b[0] and a != b:
                assert (parsed[a] < parsed[b]) != (parsed[b] < parsed[a])
                assert (parsed[a] < parsed[b]) == (a[1:] < b[1:])


# --------------------------------------------------------------------------- preflight wiring


def test_preflight_check_models_maps_findings_to_results_and_prints_the_deploy_args(capsys):
    rt = FakeRuntime({f"us.{HAIKU45}": INVALID_ID})
    results = preflight.check_models(FakeBedrock(), rt, "us-east-1")
    out = capsys.readouterr().out
    assert "-c 'modelIds=[" in out
    assert all(r.ok for r in results)
    warns = [r for r in results if r.warn]
    assert len(warns) == 1 and f"us.{HAIKU45}" in warns[0].name


def test_preflight_check_models_reports_a_fail_as_not_ok():
    results = preflight.check_models(FakeBedrock(), FakeRuntime(), "us-east-1", model_ids=["eu.anthropic.claude-haiku-5-5"])
    assert [r.ok for r in results] == [False, False]  # the refused entry, then "no entry answered"
    assert "another geography" in results[0].detail


def test_preflight_report_prints_fail_and_warn_lines_in_its_format(capsys):
    results = preflight.check_models(FakeBedrock(), FakeRuntime({f"us.{HAIKU45}": INVALID_ID}), "us-east-1",
                                     model_ids=[f"us.{HAIKU45}", "eu.anthropic.claude-haiku-5-5"])
    assert preflight.report(results) == 1
    out = capsys.readouterr().out
    assert "[WARN] Model entry us." in out and "[FAIL] Model entry eu.anthropic.claude-haiku-5-5:" in out
    assert "       fix (run it yourself):" in out


# --------------------------------------------------------------------------- validated prompts


def test_a_replacement_without_validated_prompts_is_skipped_for_the_next():
    rt = FakeRuntime({"us.anthropic.claude-sonnet-5-5": INVALID_ID})
    asked = []

    def has_prompt(model_id):
        asked.append(model_id)
        return model_id != "us.anthropic.claude-sonnet-5"

    res = resolve(rt, has_prompt=has_prompt)
    assert res.model_ids[-1] == f"us.{SONNET46}"
    assert "us.anthropic.claude-sonnet-5" not in rt.tried  # skipped before its request
    assert asked[:2] == ["us.anthropic.claude-sonnet-5", f"us.{SONNET46}"]
    (warn,) = lines(res, "WARN")
    assert f"us.{SONNET46}" in warn.detail and res.ok


def test_fail_when_no_replacement_has_validated_prompts():
    rt = FakeRuntime({f"us.{HAIKU45}": INVALID_ID})
    res = resolve(rt, has_prompt=lambda model_id: False)
    assert not res.ok
    assert res.model_ids == ["us.anthropic.claude-haiku-5-5", "us.anthropic.claude-sonnet-5-5"]
    (fail,) = lines(res, "FAIL")
    assert f"us.{HAIKU45}" in fail.name and f"us.{HAIKU3}" in fail.detail and "validated prompt" in fail.detail
    assert "prompt_suite" in fail.fix
    assert rt.tried == res.model_ids[:1] + [f"us.{HAIKU45}"] + res.model_ids[1:]  # no replacement was probed


def test_a_replacement_with_prompts_that_fails_the_request_is_not_a_prompt_fail():
    rt = FakeRuntime({f"us.{HAIKU45}": INVALID_ID, f"us.{HAIKU3}": INVALID_ID})
    res = resolve(rt, has_prompt=lambda model_id: True)
    (fail,) = lines(res, "FAIL")
    assert "no haiku replacement works" in fail.detail and "validated prompt" not in fail.detail


def test_has_prompt_never_judges_the_requested_entries():
    rt = FakeRuntime()
    res = resolve(rt, has_prompt=lambda model_id: False)
    assert res.ok and len(res.model_ids) == 3
