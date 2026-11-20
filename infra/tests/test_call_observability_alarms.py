"""Call observability metrics, alarms and the call timeline query.

Template assertions over offline synths with ``callObservability=on``: Fargate dev (CloudFront
mode), Fargate prod (custom domain), dev with every ceiling setting overridden, and the AgentCore hosting
(``bridgeHosting=agentcore``, VPC mode, dev). The switch-off side is ``synth_any`` and the
``agentcore`` VPC dev synth: none of these resources exist there, and the snapshot tests
(``test_fargate_snapshot``) hold the templates unchanged.

* 7.3: metric filters in ``OwlBank/<stage>/Calls`` (pattern, metric name, value field, no
  dimensions): ``TurnSilenceMs``, ``TurnWaitMs``, ``Turns``, ``TurnErrors``, ``CallInputTokens``,
  ``CallOutputTokens``, and for 7.5 ``ToolCalls`` and ``ToolErrors``, on the bridge's log group; ``FrontdoorResolveMs``, ``FrontdoorStateLoadMs``,
  ``FrontdoorStateSaveMs`` on the front door's.
* 7.4: an ``AWS::CloudWatch::AnomalyDetector`` and an ``ANOMALY_DETECTION_BAND`` alarm per watched
  metric (time to first token, model latency, tool duration over all tools, wait, silence, tokens
  per call), each with a fixed-ceiling alarm whose threshold comes from the settings.
* 7.5: turn and tool error rates as metric math over 15 minutes, above 5 %; the tool rate from the
  bridge's ``tool_result`` lines, not Strands' tool metrics.
* EMF-based alarms use only ``Service``, ``Environment`` and ``tool_name``.
* 9.2: "OwlBank/Call timeline" over the traces log group and ``aws/spans`` by ``session.id``.

No property-based tests: these are fixed template assertions over a small example matrix of
contexts, with no generated input space ("PBT: none").
"""

from __future__ import annotations

import json
import warnings

import pytest
from conftest import FAKE_CERT, _synth
from test_bridge_agentcore import _ctx
from test_synth_and_nag import _nag_report

from owlbank_infra import call_observability as co
from owlbank_infra.observability import BRIDGE_LOG_GROUP_NAME, RUNTIME_LOG_GROUP_PREFIX

BRIDGE, COMPUTE = "OwlBankBridge", "OwlBankCompute"
ALARM, FILTER, QUERY = "AWS::CloudWatch::Alarm", "AWS::Logs::MetricFilter", "AWS::Logs::QueryDefinition"
DETECTOR = "AWS::CloudWatch::AnomalyDetector"

CEILINGS = {"ttftCeilingSeconds": 2.5, "turnWaitCeilingSeconds": 7, "silenceCeilingSeconds": 3.5,
            "tokensPerCallCeiling": 12345}
CONTEXTS = {
    "on-dev": {"callObservability": "on"},
    "on-prod": {"callObservability": "on", "stage": "prod", "domainName": "owlbank.example.com",
                "certificateArn": FAKE_CERT, "bridgeDesiredCount": 2},
    "on-ceilings": {"callObservability": "on", **CEILINGS},
    "on-agentcore": {**_ctx("vpc", "dev"), "callObservability": "on"},
}

#: (metric name, event, value field or None for a count, extra string conditions; a ``!`` prefix on
#: the key is a not-equals condition)
BRIDGE_FILTERS = [
    ("TurnSilenceMs", "turn", "silence_ms", {}),
    ("TurnWaitMs", "turn", "wait_ms", {}),
    ("Turns", "turn", None, {}),
    ("TurnErrors", "turn", None, {"outcome": "error"}),
    ("CallInputTokens", "ws_session_end", "input_tokens", {}),
    ("CallOutputTokens", "ws_session_end", "output_tokens", {}),
    ("ToolCalls", "tool_result", None, {"!outcome": "cancelled"}),
    ("ToolErrors", "tool_result", None, {"outcome": "error"}),
]
FRONTDOOR_FILTERS = [
    ("FrontdoorResolveMs", "factor_answer", "timings_ms.resolve", {}),
    ("FrontdoorStateLoadMs", "factor_answer", "timings_ms.state_load", {}),
    ("FrontdoorStateSaveMs", "factor_answer", "timings_ms.state_save", {}),
]
#: Anomaly-watched metrics: (construct-ID stem, namespace kind, metric, stat).
ANOMALY = [
    ("CallObsTtft", "emf", "strands.model.time_to_first_token", "Average"),
    ("CallObsModelLatency", "emf", "strands.event_loop.latency", "Average"),
    ("CallObsToolDuration", "emf", "strands.tool.duration", "Average"),
    ("CallObsTurnWait", "calls", "TurnWaitMs", "p90"),
    ("CallObsTurnSilence", "calls", "TurnSilenceMs", "p90"),
]


def _quiet(factory, name, ctx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _synth(factory, name, ctx)


@pytest.fixture(scope="module")
def obs(tmp_path_factory) -> dict:
    return {name: _quiet(tmp_path_factory, f"callobs-alarms-{name}", ctx) for name, ctx in CONTEXTS.items()}


@pytest.fixture(params=sorted(CONTEXTS))
def on(request, obs):
    return obs[request.param]


@pytest.fixture(params=["on-dev", "on-prod", "on-agentcore"])
def on_default_ceilings(request, obs):
    return obs[request.param]


# --------------------------------------------------------------------------- helpers

def _res(synth, stack, rtype):
    return {lid: r for lid, r in synth.templates[stack]["Resources"].items() if r["Type"] == rtype}


def _by_lid_prefix(synth, stack, rtype, prefix):
    (r,) = [r["Properties"] for lid, r in _res(synth, stack, rtype).items() if lid.startswith(prefix)]
    return r


def _filter(synth, stack, metric_name):
    (mf,) = [r["Properties"] for r in _res(synth, stack, FILTER).values()
             if r["Properties"]["MetricTransformations"][0]["MetricName"] == metric_name]
    return mf


def _calls_ns(synth):
    return f"OwlBank/{synth.cfg.stage}/Calls"


def _emf_dims(synth):
    return {"Service": f"owlbank-{synth.cfg.stage}-bridge", "Environment": synth.cfg.stage}


def _dims(raw):
    return {d["Name"]: d["Value"] for d in raw or []}


def _topic_ref(synth):
    """The alarm topic reference every existing bridge alarm uses (AWS-M4)."""
    refs = {json.dumps(r["Properties"]["AlarmActions"]) for lid, r in _res(synth, BRIDGE, ALARM).items()
            if not lid.startswith("CallObs")}
    (ref,) = refs
    return json.loads(ref)


def _bridge_group_ref(synth):
    if synth.cfg.is_agentcore:
        (lid,) = [lid for lid, r in _res(synth, BRIDGE, "AWS::Logs::LogGroup").items()
                  if RUNTIME_LOG_GROUP_PREFIX in json.dumps(r["Properties"]["LogGroupName"])]
    else:
        (lid,) = [lid for lid, r in _res(synth, BRIDGE, "AWS::Logs::LogGroup").items()
                  if r["Properties"]["LogGroupName"] == BRIDGE_LOG_GROUP_NAME]
    return {"Ref": lid}


def _frontdoor_group_ref(synth):
    (lid,) = [lid for lid, r in _res(synth, COMPUTE, "AWS::Logs::LogGroup").items()
              if r["Properties"]["LogGroupName"] == co.FRONTDOOR_LOG_GROUP_NAME]
    return {"Ref": lid}


def _all_metric_stats(props):
    """Every (namespace, metric, dims) an alarm or detector watches, in any of the CFN shapes."""
    out = []
    if "MetricName" in props:
        out.append((props["Namespace"], props["MetricName"], _dims(props.get("Dimensions"))))
    if "SingleMetricAnomalyDetector" in props:
        d = props["SingleMetricAnomalyDetector"]
        out.append((d["Namespace"], d["MetricName"], _dims(d.get("Dimensions"))))
    queries = props.get("Metrics") or props.get("MetricMathAnomalyDetector", {}).get("MetricDataQueries", [])
    for q in queries:
        if "MetricStat" in q:
            m = q["MetricStat"]["Metric"]
            out.append((m["Namespace"], m["MetricName"], _dims(m.get("Dimensions"))))
    return out


# --------------------------------------------------------------------------- 7.3 metric filters

@pytest.mark.parametrize("metric, event, field, equals", BRIDGE_FILTERS, ids=[f[0] for f in BRIDGE_FILTERS])
def test_bridge_filter(on, metric, event, field, equals):
    mf = _filter(on, BRIDGE, metric)
    assert mf["LogGroupName"] == _bridge_group_ref(on)
    (t,) = mf["MetricTransformations"]
    assert t["MetricNamespace"] == _calls_ns(on)
    assert t["MetricValue"] == (f"$.{field}" if field else "1")
    assert "Dimensions" not in t  # no caller data can become a dimension
    pattern = mf["FilterPattern"]
    assert f'$.event = "{event}"' in pattern
    for k, v in equals.items():
        if k.startswith("!"):
            assert f'$.{k[1:]} != "{v}"' in pattern
        else:
            assert f'$.{k} = "{v}"' in pattern
    if field:
        assert f"$.{field} >= 0" in pattern  # a null or missing field never publishes
        assert "DefaultValue" not in t
    else:
        assert t["DefaultValue"] == 0  # counts: a zero numerator on a healthy period
        assert t["Unit"] == "Count"


@pytest.mark.parametrize("metric, event, field, equals", FRONTDOOR_FILTERS, ids=[f[0] for f in FRONTDOOR_FILTERS])
def test_frontdoor_filter(on, metric, event, field, equals):
    mf = _filter(on, COMPUTE, metric)
    assert mf["LogGroupName"] == _frontdoor_group_ref(on)
    (t,) = mf["MetricTransformations"]
    assert (t["MetricNamespace"], t["MetricValue"], t["Unit"]) == (_calls_ns(on), f"$.{field}", "Milliseconds")
    assert "Dimensions" not in t
    assert mf["FilterPattern"] == f'{{ ($.event = "{event}") && ($.{field} >= 0) }}'


def test_filters_are_not_on_the_traces_log_group(on):
    """7.3: filters go on the existing log groups, not the traces log group."""
    for _s, _l, r in on.resources(FILTER):
        assert r["Properties"]["LogGroupName"] != co.TRACES_LOG_GROUP_NAME


# --------------------------------------------------------------------------- 7.4 anomaly alarms

def _anomaly_alarm(synth, stem):
    return _by_lid_prefix(synth, BRIDGE, ALARM, f"{stem}AnomalyAlarm")


@pytest.mark.parametrize("stem, kind, metric, stat", ANOMALY, ids=[a[0] for a in ANOMALY])
def test_anomaly_detector_and_alarm(on, stem, kind, metric, stat):
    ns, dims = (co.EMF_NAMESPACE, _emf_dims(on)) if kind == "emf" else (_calls_ns(on), {})
    det = _by_lid_prefix(on, BRIDGE, DETECTOR, f"{stem}AnomalyDetector")["SingleMetricAnomalyDetector"]
    assert (det["Namespace"], det["MetricName"], det["Stat"]) == (ns, metric, stat)
    assert _dims(det.get("Dimensions")) == dims

    alarm = _anomaly_alarm(on, stem)
    assert alarm["ComparisonOperator"] == "GreaterThanUpperThreshold"
    assert alarm["ThresholdMetricId"] == "ad1"
    assert "Threshold" not in alarm
    assert alarm["TreatMissingData"] == "notBreaching"
    assert alarm["AlarmActions"] == _topic_ref(on)
    assert (alarm["EvaluationPeriods"], alarm["DatapointsToAlarm"]) == (3, 3)
    queries = {q["Id"]: q for q in alarm["Metrics"]}
    assert queries["ad1"]["Expression"] == "ANOMALY_DETECTION_BAND(m1, 2)"
    m1 = queries["m1"]["MetricStat"]
    assert (m1["Metric"]["Namespace"], m1["Metric"]["MetricName"], m1["Stat"], m1["Period"]) == (ns, metric, stat, 300)
    assert _dims(m1["Metric"].get("Dimensions")) == dims
    assert queries["m1"]["ReturnData"] is True


def test_anomaly_alarm_depends_on_its_detector(on):
    res = on.templates[BRIDGE]["Resources"]
    for lid, r in res.items():
        if r["Type"] == ALARM and "AnomalyAlarm" in lid:
            stem = lid.split("AnomalyAlarm")[0]
            assert f"{stem}AnomalyDetector" in r.get("DependsOn", []), lid


def test_tokens_per_call_anomaly_is_metric_math(on):
    det = _by_lid_prefix(on, BRIDGE, DETECTOR, "CallObsTokensPerCallAnomalyDetector")["MetricMathAnomalyDetector"]
    q = {x["Id"]: x for x in det["MetricDataQueries"]}
    assert q["e1"] == {"Id": "e1", "Expression": "m1 + m2", "ReturnData": True}
    assert {q[i]["MetricStat"]["Metric"]["MetricName"] for i in ("m1", "m2")} == {"CallInputTokens", "CallOutputTokens"}
    assert {q[i]["MetricStat"]["Stat"] for i in ("m1", "m2")} == {"Average"}
    alarm = _anomaly_alarm(on, "CallObsTokensPerCall")
    aq = {x["Id"]: x for x in alarm["Metrics"]}
    assert aq["e1"]["Expression"] == "m1 + m2"
    assert aq["ad1"]["Expression"] == "ANOMALY_DETECTION_BAND(e1, 2)"
    assert (aq["m1"]["ReturnData"], aq["m2"]["ReturnData"]) == (False, False)
    assert alarm["ThresholdMetricId"] == "ad1"
    assert alarm["ComparisonOperator"] == "GreaterThanUpperThreshold"
    assert alarm["AlarmActions"] == _topic_ref(on)


def test_six_anomaly_detectors_and_alarms(on):
    assert len(_res(on, BRIDGE, DETECTOR)) == 6
    assert len([lid for lid in _res(on, BRIDGE, ALARM) if "AnomalyAlarm" in lid]) == 6


# --------------------------------------------------------------------------- 7.4 fixed ceilings

def _ceilings(cfg):
    """(stem, metric, stat or extended stat, threshold) from the config's settings."""
    return [
        ("CallObsTtft", "strands.model.time_to_first_token", "Average", cfg["ttftCeilingSeconds"] * 1000),
        ("CallObsModelLatency", "strands.event_loop.latency", "Average", cfg["turnWaitCeilingSeconds"] * 1000),
        ("CallObsToolDuration", "strands.tool.duration", "Maximum", cfg["turnWaitCeilingSeconds"]),  # seconds
        ("CallObsTurnWait", "TurnWaitMs", "p90", cfg["turnWaitCeilingSeconds"] * 1000),
        ("CallObsTurnSilence", "TurnSilenceMs", "p90", cfg["silenceCeilingSeconds"] * 1000),
    ]


DEFAULTS = {"ttftCeilingSeconds": 3, "turnWaitCeilingSeconds": 5, "silenceCeilingSeconds": 4,
            "tokensPerCallCeiling": 60000}


@pytest.mark.parametrize("which, settings", [("default", DEFAULTS), ("override", CEILINGS)])
def test_ceiling_alarms_use_the_settings(obs, on_default_ceilings, which, settings):
    synth = on_default_ceilings if which == "default" else obs["on-ceilings"]
    for stem, metric, stat, threshold in _ceilings(settings):
        a = _by_lid_prefix(synth, BRIDGE, ALARM, f"{stem}CeilingAlarm")
        assert a["MetricName"] == metric
        assert a.get("Statistic", a.get("ExtendedStatistic")) == stat
        assert a["Threshold"] == pytest.approx(threshold), stem
        assert a["ComparisonOperator"] == "GreaterThanThreshold"
        assert (a["Period"], a["EvaluationPeriods"]) == (300, 3)
        assert a["TreatMissingData"] == "notBreaching"
        assert a["AlarmActions"] == _topic_ref(synth)
    tokens = _by_lid_prefix(synth, BRIDGE, ALARM, "CallObsTokensPerCallCeilingAlarm")
    assert tokens["Threshold"] == settings["tokensPerCallCeiling"]
    q = {x["Id"]: x for x in tokens["Metrics"]}
    (expr,) = [x for x in q.values() if "Expression" in x]
    assert expr["Expression"] == "m1 + m2"
    assert {q[i]["MetricStat"]["Metric"]["MetricName"] for i in ("m1", "m2")} == {"CallInputTokens", "CallOutputTokens"}
    assert {q[i]["MetricStat"]["Stat"] for i in ("m1", "m2")} == {"Maximum"}
    assert tokens["AlarmActions"] == _topic_ref(synth)


# --------------------------------------------------------------------------- 7.5 error rates

@pytest.mark.parametrize("stem", ["CallObsTurnErrorRateAlarm", "CallObsToolErrorRateAlarm"])
def test_error_rate_alarm_shape(on, stem):
    a = _by_lid_prefix(on, BRIDGE, ALARM, stem)
    assert (a["Threshold"], a["ComparisonOperator"], a["EvaluationPeriods"]) == (5, "GreaterThanThreshold", 1)
    assert a["TreatMissingData"] == "notBreaching"
    assert a["AlarmActions"] == _topic_ref(on)
    assert sum(1 for x in a["Metrics"] if x.get("ReturnData", True)) == 1  # one watched expression
    for x in a["Metrics"]:
        if "MetricStat" in x:
            assert (x["MetricStat"]["Stat"], x["MetricStat"]["Period"]) == ("Sum", 900), stem  # 15 minutes


def test_turn_error_rate_alarm(on):
    a = _by_lid_prefix(on, BRIDGE, ALARM, "CallObsTurnErrorRateAlarm")
    q = {x["Id"]: x for x in a["Metrics"]}
    (expr,) = [x for x in q.values() if "Expression" in x]
    assert expr["Expression"] == "100 * errors / turns" and expr["ReturnData"] is True
    for qid, metric in (("errors", "TurnErrors"), ("turns", "Turns")):
        ms = q[qid]["MetricStat"]
        assert (ms["Metric"]["Namespace"], ms["Metric"]["MetricName"]) == (_calls_ns(on), metric)
        assert _dims(ms["Metric"].get("Dimensions")) == {}


def _tool_alarm(synth):
    a = _by_lid_prefix(synth, BRIDGE, ALARM, "CallObsToolErrorRateAlarm")
    return {x["Id"]: x for x in a["Metrics"]}


def test_tool_error_rate_alarm_uses_the_tool_result_lines(on):
    """Errors over calls from the bridge's ``tool_result`` filters, not Strands'
    tool metrics (which count every interrupt and hand-off as an error); no calls gives 0 %."""
    q = _tool_alarm(on)
    (ret,) = [x for x in q.values() if x.get("ReturnData", True)]
    assert ret["Expression"] == "IF(calls > 0, 100 * errors / calls, 0)"
    assert set(q) == {"errors", "calls", ret["Id"]}
    for qid, metric in (("errors", "ToolErrors"), ("calls", "ToolCalls")):
        assert q[qid]["ReturnData"] is False
        ms = q[qid]["MetricStat"]
        assert (ms["Metric"]["Namespace"], ms["Metric"]["MetricName"]) == (_calls_ns(on), metric)
        assert _dims(ms["Metric"].get("Dimensions")) == {}
        assert (ms["Stat"], ms["Period"]) == ("Sum", 900)
    # Same group as the other bridge filters (the Runtime's under the AgentCore hosting option).
    for metric in ("ToolErrors", "ToolCalls"):
        assert _filter(on, BRIDGE, metric)["LogGroupName"] == _bridge_group_ref(on)


def test_no_alarm_reads_strands_tool_counters(on):
    """Strands' ``strands.tool.call_count`` and ``strands.tool.error_count`` stay in EMF for
    dashboards only."""
    for lid, r in _res(on, BRIDGE, ALARM).items():
        if lid.startswith("CallObs"):
            for _ns, metric, _dims_ in _all_metric_stats(r["Properties"]):
                assert metric not in {"strands.tool.call_count", "strands.tool.error_count"}, lid


# --------------------------------------------------------------------------- Dimensions, topic

def test_emf_alarm_dimensions_are_bounded(on):
    """Anything in the EMF namespace uses only Service/Environment/tool_name,
    with this deployment's values; the Calls metrics have no dimensions at all."""
    seen = 0
    for rtype in (ALARM, DETECTOR):
        for lid, r in _res(on, BRIDGE, rtype).items():
            if not lid.startswith("CallObs"):
                continue
            for ns, _metric, dims in _all_metric_stats(r["Properties"]):
                seen += 1
                if ns == co.EMF_NAMESPACE:
                    assert set(dims) <= co.EMF_DIMENSION_KEYS, lid
                    assert {k: dims[k] for k in ("Service", "Environment")} == _emf_dims(on), lid
                else:
                    assert ns == _calls_ns(on) and dims == {}, lid
    assert seen > 20


def test_every_call_alarm_notifies_the_alarm_topic(on):
    alarms = {lid: r for lid, r in _res(on, BRIDGE, ALARM).items() if lid.startswith("CallObs")}
    assert len(alarms) == 6 + 6 + 2  # anomaly, ceilings, error rates
    for lid, r in alarms.items():
        assert r["Properties"]["AlarmActions"] == _topic_ref(on), lid
        assert r["Properties"]["TreatMissingData"] == "notBreaching", lid


def test_no_call_alarms_in_compute(on):
    assert not [lid for lid in _res(on, COMPUTE, ALARM) if lid.startswith("CallObs")]
    assert not _res(on, COMPUTE, DETECTOR)


# --------------------------------------------------------------------------- 9.2 call timeline

def test_call_timeline_query(on):
    (q,) = [r["Properties"] for r in _res(on, BRIDGE, QUERY).values()
            if r["Properties"]["Name"] == "OwlBank/Call timeline"]
    groups = q["LogGroupNames"]
    assert groups[1] == "aws/spans" and len(groups) == 2
    assert groups[0] == co.TRACES_LOG_GROUP_NAME  # both hostings (6.4, option A)
    text = q["QueryString"]
    assert "`attributes.session.id` = {{session_id}}" in text
    assert "| sort start_ms asc" in text
    for needle in ("name", "service", "duration_ms", "start_ms", "end_ms", "autoregress end_ms as prev_end_ms"):
        assert needle in text
    # Logs Insights can't compute on autoregress output: no arithmetic on it here;
    # scripts/call_timeline.py prints the gaps, labelled as Twilio-side time.
    for gone in ("gap_ms", "twilio_side_ms", "greatest(", "- prev_end_ms"):
        assert gone not in text
    assert "scripts/call_timeline.py" in text and "Twilio-side" in text
    assert text.rstrip().endswith("| limit 10000")
    (p,) = q["Parameters"]
    assert p["Name"] == "session_id" and p["DefaultValue"].startswith('"CA')


def test_call_timeline_query_is_created_once(on):
    names = [r["Properties"]["Name"] for _s, _l, r in on.resources(QUERY)]
    assert names.count("OwlBank/Call timeline") == 1


# --------------------------------------------------------------------------- cdk-nag

def test_cdk_nag_clean(on):
    report = _nag_report(on)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [(v["ruleName"], v.get("violatingResources")) for v in errors]
        assert report["conclusion"] == "success"


# --------------------------------------------------------------------------- off: none of it

@pytest.fixture(params=["synth_default", "synth_domain", "synth_prod", "synth_agentcore_vpc_dev"])
def off(request):
    return request.getfixturevalue(request.param)


def test_off_creates_none_of_it(off):
    assert not list(off.resources(DETECTOR))
    for _s, lid, r in off.resources():
        assert not lid.startswith(("CallObs", "CallTimeline")), lid
    for _s, _l, r in off.resources(FILTER):
        assert not r["Properties"]["MetricTransformations"][0]["MetricNamespace"].endswith("/Calls")
    for _s, _l, r in off.resources(QUERY):
        assert r["Properties"]["Name"] != "OwlBank/Call timeline"
    for _s, _l, r in off.resources(ALARM):
        assert r["Properties"].get("Namespace") != co.EMF_NAMESPACE
        for x in r["Properties"].get("Metrics") or []:
            assert x.get("MetricStat", {}).get("Metric", {}).get("Namespace") != co.EMF_NAMESPACE
        assert "ThresholdMetricId" not in r["Properties"]
