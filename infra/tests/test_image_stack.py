"""AgentCore hosting: the ``OwlBankBridgeImage`` stack.

The ECR repository has immutable tags, scan on push, the project CMK and ``RETAIN``, no
repository policy (so none with a ``*`` principal), and cdk-nag is clean. The stack is in the app on
both hostings; on Fargate nothing depends on it (the Fargate snapshot test holds its template). No
PBT: IaC, checked by assertions.
"""

from __future__ import annotations

import json

import pytest
from conftest import _synth
from test_synth_and_nag import _nag_report

from owlbank_infra.image_stack import DATA_KEY_ALIAS

IMAGE = "OwlBankBridgeImage"


@pytest.fixture(scope="module")
def synth_image_prod(tmp_path_factory):
    return _synth(tmp_path_factory, "image-prod", {
        "stage": "prod", "domainName": "owlbank.example.com",
        "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/11111111-2222-3333-4444-555555555555",
        "bridgeDesiredCount": 2})


@pytest.fixture(scope="module", params=["synth_default", "synth_image_prod"])
def image_any(request):
    return request.getfixturevalue(request.param)


def _repo(synth) -> dict:
    found = [r for s, _, r in synth.resources("AWS::ECR::Repository") if s == IMAGE]
    assert len(found) == 1, found
    return found[0]


def test_image_stack_on_fargate_and_nothing_depends_on_it(synth_default):
    """The repository is in the app on both hostings; on Fargate no other stack depends on it."""
    assert synth_default.stacks["image"].stack_name == IMAGE and IMAGE in synth_default.templates
    for stack in synth_default.stacks.values():
        assert IMAGE not in {d.stack_name for d in stack.dependencies}
    assert "BridgeImage" not in json.dumps({n: t for n, t in synth_default.templates.items() if n != IMAGE})


def test_image_stack_in_the_app(image_any):
    assert IMAGE in image_any.templates
    assert image_any.stacks["image"].stack_name == IMAGE


def test_repository_name_from_config(image_any):
    stage = image_any.stacks["data"].node.try_get_context("stage") or "dev"
    assert _repo(image_any)["Properties"]["RepositoryName"] == f"owlbank-{stage}-bridge"


def test_tags_are_immutable(image_any):
    assert _repo(image_any)["Properties"]["ImageTagMutability"] == "IMMUTABLE"


def test_scan_on_push(image_any):
    assert _repo(image_any)["Properties"]["ImageScanningConfiguration"] == {"ScanOnPush": True}


def test_encrypted_with_the_project_cmk(image_any):
    enc = _repo(image_any)["Properties"]["EncryptionConfiguration"]
    assert enc["EncryptionType"] == "KMS"
    # The project CMK through its fixed alias, in this stack's (fake test) account and Region.
    assert enc["KmsKey"] == {"Fn::Join": ["", [
        "arn:", {"Ref": "AWS::Partition"}, f":kms:us-east-1:111111111111:{DATA_KEY_ALIAS}"]]}


def test_alias_matches_the_data_stack_key(image_any):
    aliases = [r["Properties"] for s, _, r in image_any.resources("AWS::KMS::Alias") if s == "OwlBankData"]
    assert DATA_KEY_ALIAS in {a["AliasName"] for a in aliases}


def test_retained(image_any):
    repo = _repo(image_any)
    assert repo["DeletionPolicy"] == "Retain" and repo["UpdateReplacePolicy"] == "Retain"
    assert "EmptyOnDelete" not in repo["Properties"]


def test_no_repository_policy_and_no_star_principal(image_any):
    tpl = image_any.templates[IMAGE]
    assert "RepositoryPolicyText" not in _repo(image_any)["Properties"]
    assert "*" not in set(_walk(tpl))
    # The repository is the stack's only resource: no policy, role or custom resource beside it.
    assert {r["Type"] for r in tpl["Resources"].values()} == {"AWS::ECR::Repository"}


def test_no_cross_stack_import_and_follows_data(image_any):
    tpl = image_any.templates[IMAGE]
    assert "Fn::ImportValue" not in str(tpl)
    deps = {d.stack_name for d in image_any.stacks["image"].dependencies}
    assert deps == {"OwlBankData"}


def test_outputs(image_any):
    outs = image_any.templates[IMAGE]["Outputs"]
    assert {"RepositoryUri", "RepositoryArn", "RepositoryName"} <= set(outs)


def test_stack_is_tagged(image_any):
    tags = {t["Key"]: t["Value"] for t in _repo(image_any)["Properties"].get("Tags", [])}
    assert tags.get("project") == "owlbank-blueprint"


def test_cdk_nag_clean(image_any):
    report = _nag_report(image_any)
    if report is not None:
        errors = [v for v in report["violations"] if v["severity"] == "error"]
        assert not errors, [v["ruleName"] for v in errors]
        assert report["conclusion"] == "success"


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj
