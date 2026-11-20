"""Admin API redaction route properties:

* For any sequence of ``PUT /redaction`` with random bodies, callers and revs,
  each successful change writes exactly one audit entry with ``security_change: true``, the
  item's ``rev`` goes up by one, and the item's ``rev`` equals the number of entries; a stale or
  ahead ``rev``, a refused body and a non-admin caller change neither the item nor the audit
  table.
* For any stored config and any body, the body is accepted iff
  :func:`owlbank.pii.parse_redaction_config` accepts the section it makes (the stored fields with
  the body's over them), and then the stored item round-trips: the parser reads it back as that
  config, and the policy the components load holds it.
"""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.admin_api import audit
from lambdas.admin_api.app import AdminApi, Deps
from lambdas.admin_api.redaction_routes import FIELDS
from owlbank.pii import (
    COMPREHEND_PII_ENTITY_TYPES,
    ON_FAILURE,
    RedactionConfig,
    RedactionConfigError,
    parse_redaction_config,
)
from owlbank.policy import Policy
from owlbank.store import MemoryStore
from tests.admin_api_fakes import POOL_ID, Clock, FakeCognito, call, claims, seeded_store

_SEED = seeded_store()
ADMIN = claims(["admin"], sub="sub-admin")
CALLERS = {"admin": ADMIN, "editor": claims(["admin-editor"]), "viewer": claims(["admin-viewer"])}
TYPES = sorted(COMPREHEND_PII_ENTITY_TYPES)


def fresh_store() -> MemoryStore:
    store = MemoryStore()
    store._tables = copy.deepcopy(_SEED._tables)
    return store


def api_for(store: MemoryStore) -> AdminApi:
    clock = Clock()
    return AdminApi(Deps(store=store, cognito=FakeCognito(), user_pool_id=POOL_ID, retention_days=90,
                         clock=clock, suffix=clock.suffix))


def item(store: MemoryStore) -> dict[str, Any] | None:
    return copy.deepcopy(store.get("policy", {"pk": "redaction"}))


def n_audits(store: MemoryStore) -> int:
    return len(store._tables.get("admin_audit", {}))


_junk = st.one_of(st.none(), st.booleans(), st.integers(-2, 2), st.text(max_size=4),
                  st.lists(st.integers(0, 1), max_size=1), st.dictionaries(st.just("k"), st.just(1), max_size=1))
_types = st.lists(st.sampled_from(TYPES), max_size=6)
_conf = st.floats(0, 1, allow_nan=False, allow_infinity=False)
_mask = st.sampled_from(("[REDACTED:{type}]", "***", "[{type}]", "<pii>", "", "{name}", "{type", "a}"))
#: Mostly valid values, so most writes succeed; some invalid ones in every field.
_values: dict[str, st.SearchStrategy[Any]] = {
    "enabled": st.one_of(st.booleans(), st.booleans(), _junk),
    "entity_types": st.one_of(_types, _types, st.just([]), st.lists(st.sampled_from([*TYPES[:3], "NOT_A_TYPE"]),
                                                                     max_size=3), _junk),
    "min_confidence": st.one_of(_conf, _conf.map(repr), st.floats(-1, 2, allow_nan=False), _junk),
    "mask": st.one_of(_mask, _junk),
    "on_failure": st.one_of(st.sampled_from(sorted(ON_FAILURE)), st.text(max_size=8), _junk),
}
_valid_values: dict[str, st.SearchStrategy[Any]] = {
    "enabled": st.booleans(), "entity_types": _types, "min_confidence": st.one_of(_conf, _conf.map(repr)),
    "mask": st.sampled_from(("[REDACTED:{type}]", "***", "[{type}]")),
    "on_failure": st.sampled_from(sorted(ON_FAILURE)),
}
#: Valid field values three times in four (the section may still be refused: enabled, no types).
_body = st.one_of(
    st.fixed_dictionaries({}, optional=_valid_values), st.fixed_dictionaries({}, optional=_valid_values),
    st.fixed_dictionaries({}, optional=_valid_values),
    st.fixed_dictionaries({}, optional={**_values, "apply_to": st.just(["journal"]), "bogus": st.just(1)}))
#: A valid stored config (the item as the seed or an earlier write leaves it).
_stored = st.builds(
    lambda enabled, types, conf, mask, on_failure: RedactionConfig(
        enabled=enabled and bool(types), entity_types=tuple(dict.fromkeys(types)), min_confidence=conf,
        mask=mask, on_failure=on_failure),
    st.booleans(), _types, _conf, st.sampled_from(("[REDACTED:{type}]", "***", "[{type}]")),
    st.sampled_from(sorted(ON_FAILURE)))


def _expected(stored: RedactionConfig, body: dict[str, Any]) -> RedactionConfig | None:
    """The config a PUT of ``body`` over ``stored`` must store, or ``None`` if it must be refused."""
    if set(body) - set(FIELDS):
        return None
    updates = {f: body[f] for f in FIELDS if f in body}
    if not updates:
        return None
    try:
        return parse_redaction_config({**stored.to_doc(), **updates})
    except RedactionConfigError:
        return None


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_stored, _body)
def test_accepted_iff_the_parser_accepts_and_the_item_round_trips(stored: RedactionConfig,
                                                                        body: dict[str, Any]) -> None:
    store = fresh_store()
    store.put("policy", {"pk": "redaction", **stored.to_doc(), "rev": 3})
    api = api_for(store)
    before, audits_before = item(store), n_audits(store)
    want = _expected(stored, body)
    status, out = call(api, "PUT", "/redaction", {**body, "rev": 3}, who=ADMIN)
    if want is None:
        assert status == 400, (status, out)
        assert item(store) == before and n_audits(store) == audits_before
        return
    assert status == 200, (status, out)
    after = item(store)
    assert after is not None
    fields = {k: v for k, v in after.items() if k not in ("pk", "rev")}
    assert parse_redaction_config(fields) == want  # round-trips
    assert fields == want.to_doc()
    assert Policy.from_store(store).redaction == want  # what the components load
    assert out["effective"] == want.to_doc() and out["problems"] == []
    assert out["changed"] is (want != stored)
    assert ("redaction_off" in [w["code"] for w in out["warnings"]]) is (not want.enabled)


_ops = st.lists(st.tuples(st.one_of(_body.filter(bool), _body), st.sampled_from(("current", "stale", "ahead")),
                          st.sampled_from(("admin", "admin", "editor", "viewer"))),
                min_size=1, max_size=12)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_ops)
def test_redaction_one_security_entry_per_change(ops: list[tuple[dict[str, Any], str, str]]) -> None:
    store = fresh_store()
    api = api_for(store)
    for body, rev_choice, who in ops:
        before, audits_before = item(store), n_audits(store)
        rev = (before or {}).get("rev", 0)
        sent = {"current": rev, "stale": rev - 1, "ahead": rev + 1}[rev_choice]
        if sent < 0:
            sent = rev + 2
        status, out = call(api, "PUT", "/redaction", {**body, "rev": sent}, who=CALLERS[who])
        if who != "admin":
            assert status == 403, (status, out)
        elif rev_choice != "current":
            # A stale or ahead rev never writes: a 409 (or a 400 for a malformed body, checked first).
            assert status in (400, 409), (status, out)
        if status != 200 or not out["changed"]:
            assert status in (200, 400, 403, 409), (status, out)
            assert item(store) == before and n_audits(store) == audits_before
            continue
        assert who == "admin" and rev_choice == "current"
        assert n_audits(store) == audits_before + 1
        after = item(store)
        assert after is not None and after["rev"] == rev + 1
        (entry,) = [e for e in store.query("admin_audit", {"pk": audit.REDACTION_PK})
                    if e["sk"] == out["audit"]["sk"]]
        assert entry["changes"] and entry["security_change"] is True is out["security_change"]
        assert entry["rev"] == rev + 1 and entry["actor"]["sub"] == "sub-admin"
        for c in entry["changes"]:
            assert c["field"] in audit.SECURITY_FIELDS
            assert after.get(c["field"].removeprefix("redaction.")) == c["new"]
    n = len(store.query("admin_audit", {"pk": audit.REDACTION_PK}))
    assert (item(store) or {}).get("rev", 0) == n
    assert n_audits(store) == n  # no other item was audited
    Policy.from_store(store)  # every accepted write leaves a table the components load
