"""scripts/seed.py: moto DynamoDB, the real-number and stage guards, masking."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank.config import TABLE_NAMES
from owlbank.store import DynamoStore, MemoryStore, create_table_kwargs
from scripts import seed
from scripts._common import DATA_DIR, ScriptError

# Outside 555-0100..0199, so the scripts treat them as real, owned numbers (none is a real line).
DEV_PHONE = "+12065550006"
NUM_A = "+12065550007"
NUM_B = "+12065550008"
TABLE_ARGS = [a for name in TABLE_NAMES for a in ("--table", f"{name}=t-{name}")]


def load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def docs() -> tuple[Any, Any, Any]:
    return load("fixtures.json"), load("policy.json"), load("prompt_catalog.json")


@pytest.fixture
def dynamo() -> Iterator[DynamoStore]:
    import boto3
    from moto import mock_aws

    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        for name in TABLE_NAMES:
            client.create_table(**create_table_kwargs(name, f"t-{name}"))
        resource = boto3.resource("dynamodb", region_name="us-east-1")
        yield DynamoStore({n: f"t-{n}" for n in TABLE_NAMES}, resource=resource)


def run(argv: list[str], store: Any = None) -> int:
    return seed.main(argv, store=store)


# --------------------------------------------------------------------------- apply against moto


def test_apply_seeds_all_tables(dynamo: DynamoStore, capsys: pytest.CaptureFixture[str]) -> None:
    rc = run(["--stage", "dev", *TABLE_ARGS, "--developer-phone", DEV_PHONE,
              "--test-number", f"A={NUM_A}", "--test-number", f"B={NUM_B}", "--apply"], dynamo)
    assert rc == 0
    thomas = dynamo.get("customers", {"customer_id": "C-1001"})
    assert thomas and thomas["phone"] == "+12025550101"
    assert len(dynamo.query("customers", {"phone": "+12025550102"}, index="phone-index")) == 2
    # Real numbers are separate alias records pointing at the fixtures.
    a = dynamo.query("customers", {"phone": NUM_A}, index="phone-index")
    assert [(i["customer_id"], i["alias_of"], i["kind"]) for i in a] == [
        ("ani#A#C-1001", "C-1001", "ani_alias")]
    b = dynamo.query("customers", {"phone": NUM_B}, index="phone-index")
    assert sorted(i["alias_of"] for i in b) == ["C-1002", "C-1003"]
    dev = dynamo.query("customers", {"phone": DEV_PHONE}, index="phone-index")
    assert [i["alias_of"] for i in dev] == ["C-1001"]
    assert dynamo.get("policy", {"pk": "meta"})["max_level"] == 3
    assert dynamo.get("catalog", {"key": "_meta", "lang": "_"}) is not None
    assert {i["lang"] for i in dynamo.query("catalog", {"key": "agent.system.haiku-5-5"})} == {"en-US", "es-US"}
    out = capsys.readouterr().out
    assert DEV_PHONE not in out and NUM_A not in out


def test_seed_keeps_changed_fixtures_and_reseed_restores(dynamo: DynamoStore) -> None:
    assert run(["--stage", "dev", *TABLE_ARGS, "--apply"], dynamo) == 0
    dynamo.update("customers", {"customer_id": "C-1001"}, {"first_name": "Changed"})
    assert run(["--stage", "dev", *TABLE_ARGS, "--apply"], dynamo) == 0
    assert dynamo.get("customers", {"customer_id": "C-1001"})["first_name"] == "Changed"
    assert run(["--stage", "dev", *TABLE_ARGS, "--reseed", "--apply"], dynamo) == 0
    assert dynamo.get("customers", {"customer_id": "C-1001"})["first_name"] == "Thomas"


def test_reseed_reassigns_a_test_number(dynamo: DynamoStore) -> None:
    base = ["--stage", "test", *TABLE_ARGS, "--test-number", f"A={NUM_A}", "--apply"]
    assert run(base, dynamo) == 0
    assert run([*base, "--reseed", "--assign", "A=C-1004"], dynamo) == 0
    hits = dynamo.query("customers", {"phone": NUM_A}, index="phone-index")
    assert [h["alias_of"] for h in hits] == ["C-1004"]
    assert dynamo.get("customers", {"customer_id": "ani#A#C-1001"}) is None
    # ...and back to "no customer" for this number.
    assert run([*base, "--reseed", "--assign", "A=none"], dynamo) == 0
    assert dynamo.query("customers", {"phone": NUM_A}, index="phone-index") == []


def test_label_d_is_not_seeded(docs: tuple[Any, Any, Any]) -> None:
    plan = seed.build_plan(*docs, stage="test", test_numbers={"D": NUM_A})
    assert plan.aliases == []


def test_apply_requires_explicit_table_names(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--stage", "dev", "--apply"], MemoryStore()) == 2
    assert "no table name" in capsys.readouterr().err


def test_table_names_from_cdk_outputs(tmp_path: Path) -> None:
    out = tmp_path / "o.json"
    out.write_text(json.dumps({"OwlBankStack": {"TableCustomers": "phys-c", "PolicyTableName": "phys-p",
                                                "CatalogTableName": "phys-k"}}))
    names = seed.resolve_tables(seed.load_cdk_outputs(out), {}, require=seed.SEEDED_TABLES)
    assert (names["customers"], names["policy"], names["catalog"]) == ("phys-c", "phys-p", "phys-k")
    assert names["tokens"] == "owlbank-tokens"


# --------------------------------------------------------------------------- guards


def _data_dir_with(tmp_path: Path, mutate: Any) -> Path:
    d = tmp_path / "data"
    shutil.copytree(DATA_DIR, d)
    fx = json.loads((d / "fixtures.json").read_text(encoding="utf-8"))
    mutate(fx)
    (d / "fixtures.json").write_text(json.dumps(fx), encoding="utf-8")
    return d


@pytest.mark.parametrize("mutate", [
    lambda fx: fx["customers"][0].__setitem__("phone", "+12065550009"),
    lambda fx: fx["customers"][0].__setitem__("phone", "+12025550200"),
    lambda fx: fx["unseededNumbers"].append({"phone": "+442079460000"}),
    lambda fx: fx["_notes"].append("call me on +1 (206) 555-0042"),
    lambda fx: fx["realNumberPlaceholders"]["testNumbers"]["A"].__setitem__("phone", "+12065550009"),
])
def test_real_number_in_data_files_is_refused(tmp_path: Path, mutate: Any,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore()
    d = _data_dir_with(tmp_path, mutate)
    assert run(["--stage", "dev", "--data-dir", str(d), *TABLE_ARGS, "--apply"], store) == 2
    err = capsys.readouterr().err
    assert "fictional" in err and "0009" not in err and "0042" not in err
    assert store.get("policy", {"pk": "meta"}) is None   # nothing written


@pytest.mark.parametrize("stage", ["prod", "staging", "demo", "DEV"])
def test_real_numbers_refused_outside_dev_and_test(stage: str, capsys: pytest.CaptureFixture[str]) -> None:
    store = MemoryStore()
    for extra in (["--developer-phone", DEV_PHONE], ["--test-number", f"A={NUM_A}"]):
        assert run(["--stage", stage, *TABLE_ARGS, *extra, "--apply"], store) == 2
        assert "only in" in capsys.readouterr().err
    assert store.get("customers", {"customer_id": "C-1001"}) is None


def test_prod_without_real_numbers_is_allowed() -> None:
    store = MemoryStore()
    assert run(["--stage", "prod", *TABLE_ARGS, "--apply"], store) == 0
    assert store.query("customers", {"phone": "+12025550101"}, index="phone-index")


@pytest.mark.parametrize("kwargs, msg", [
    ({"stage": "dev", "test_numbers": {"A": NUM_A}, "assignments": {"A": "C-1004"}, "reseed": True},
     "test"),
    ({"stage": "test", "test_numbers": {"A": NUM_A}, "assignments": {"A": "C-1004"}}, "--reseed"),
    ({"stage": "test", "test_numbers": {"A": NUM_A}, "assignments": {"A": "C-9999"}, "reseed": True},
     "unknown fixture"),
    ({"stage": "test", "assignments": {"B": "C-1004"}, "reseed": True}, "no number"),
    ({"stage": "dev", "test_numbers": {"Z": NUM_A}}, "unknown test-number label"),
    ({"stage": "dev", "test_numbers": {"A": "4155238887"}}, "E.164"),
    ({"stage": "dev", "test_numbers": {"A": NUM_A, "B": NUM_A}}, "share one number"),
])
def test_argument_guards(docs: tuple[Any, Any, Any], kwargs: dict[str, Any], msg: str) -> None:
    with pytest.raises(ScriptError, match=msg):
        seed.build_plan(*docs, **kwargs)


# --------------------------------------------------------------------------- dry run and masking


def test_dry_run_is_default_writes_nothing_and_masks(capsys: pytest.CaptureFixture[str]) -> None:
    class NoStore:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"store.{name} called in dry run")

    rc = run(["--stage", "dev", "--developer-phone", DEV_PHONE, "--test-number", f"A={NUM_A}"],
             NoStore())
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN")
    fixtures = load("fixtures.json")
    for c in fixtures["customers"]:
        for h in c["factors"].values():
            salt, _, digest = h.partition("$")
            assert salt not in out and digest not in out
    assert '"factors": {"ssn4": "***", "zip": "***"}' in out
    assert DEV_PHONE not in out and NUM_A not in out
    assert "+1******0007" in out
    assert "+12025550101" in out   # fictional numbers are fine to show


phones = st.from_regex(r"\+1[2-9]\d{2}[2-9]\d{6}", fullmatch=True)


@settings(max_examples=200, deadline=None)
@given(phones)
def test_property_only_fictional_numbers_pass_the_data_check(phone: str) -> None:
    doc = {"customers": [{"customer_id": "X", "phone": phone}]}
    fictional = phone[5:10] == "55501"
    if fictional:
        seed.check_fictional("f.json", doc)
    else:
        with pytest.raises(ScriptError):
            seed.check_fictional("f.json", doc)


@pytest.mark.parametrize("email", ["thomas.doe@owlbank-demo.example", "a@x.invalid", "b@example.com",
                                   "c@mail.example.org", "d@corp.test"])
def test_reserved_email_domains_pass_the_data_check(email: str) -> None:
    seed.check_fictional("f.json", {"customers": [{"customer_id": "X", "memory_email": email}]})


@pytest.mark.parametrize("email", ["thomas.doe@owlbank-demo.com", "a@twilio.com", "b@example.com.evil.io",
                                   "not-an-address"])
def test_real_looking_emails_fail_the_data_check(email: str) -> None:
    with pytest.raises(ScriptError, match="e-mail"):
        seed.check_fictional("f.json", {"customers": [{"customer_id": "X", "memory_email": email}]})
    if "@" in email:  # also caught inside free text, not only in the memory_email field
        with pytest.raises(ScriptError, match="e-mail"):
            seed.check_fictional("f.json", {"notes": [f"contact {email}"]})


def test_the_seed_plan_carries_each_customers_memory_email() -> None:
    fixtures = json.loads((DATA_DIR / "fixtures.json").read_text(encoding="utf-8"))
    plan = seed.build_plan(fixtures, json.loads((DATA_DIR / "policy.json").read_text(encoding="utf-8")),
                           json.loads((DATA_DIR / "prompt_catalog.json").read_text(encoding="utf-8")),
                           stage="dev")
    emails = [c["memory_email"] for c in plan.customers]
    assert len(emails) == 4 and len(set(emails)) == 4


@settings(max_examples=100, deadline=None)
@given(st.dictionaries(st.sampled_from(["zip", "ssn4", "otp"]),
                       st.text(alphabet="0123456789abcdef$", min_size=8, max_size=40), min_size=1),
       phones)
def test_property_masked_never_shows_factors_or_real_numbers(factors: dict[str, str], phone: str) -> None:
    item = {"customer_id": "C-1", "phone": phone, "factors": factors}
    shown = json.dumps(seed.masked(item))
    for v in factors.values():
        assert v not in shown
    if phone[5:10] != "55501":
        assert phone not in shown
    assert item["factors"] == factors   # the original is untouched
