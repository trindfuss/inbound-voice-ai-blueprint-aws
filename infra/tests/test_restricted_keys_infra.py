"""Restricted Twilio key for the Context API.

``contextApiRestrictedKey`` (default ``on``) gives the Context API its own Restricted-key secret and
never the shared Standard ``twilio_api_key``. The Flex authorizer always reads the shared key (Twilio
has no Restricted permission for ``Tokens/validate``). Each handler reads
``OWLBANK_SECRET_TWILIO_API_KEY``, so the variable must hold the ARN of the one secret the role may read.

Checked two ways: :func:`secrets_of` over both flag values (no synth), and template assertions over the
default synth and a synth that passes the removed Flex authorizer key flag.

PBT: skipped (configuration and IAM glue over a two-row flag matrix).
"""

from __future__ import annotations

import json

import pytest
from conftest import _synth

from owlbank_infra.compute_stack import ACCESS, RESTRICTED_KEY_OF, SECRET_ENV, secrets_of
from owlbank_infra.config import ConfigError, from_context
from owlbank_infra.data_stack import CONTEXT_API_KEY_SECRET, OPTIONAL_SECRETS, secrets_for

COMPUTE, DATA = "OwlBankCompute", "OwlBankData"
SECRET_ACTIONS = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
HANDLER_ENV = "OWLBANK_SECRET_TWILIO_API_KEY"
REMOVED_FLAG = "flexAuthorizer" + "RestrictedKey"  # no longer a deploy flag


def _cfg(tmp_path, ctx: str | None = None):
    context = {}
    if ctx is not None:
        context["contextApiRestrictedKey"] = ctx
    return from_context(context, base_dir=tmp_path)


# --------------------------------------------------------------------------- grant map, no synth


def test_defaults_context_api_on_flex_authorizer_shared(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.context_api_restricted_key_on
    assert secrets_of("context_api", cfg) == (CONTEXT_API_KEY_SECRET,)
    assert secrets_of("flex_authorizer", cfg) == ("twilio_api_key",)


@pytest.mark.parametrize("ctx", ["on", "off"])
def test_grant_map_follows_the_flag(tmp_path, ctx: str) -> None:
    cfg = _cfg(tmp_path, ctx)
    assert secrets_of("context_api", cfg) == ((CONTEXT_API_KEY_SECRET,) if ctx == "on" else ("twilio_api_key",))
    assert secrets_of("flex_authorizer", cfg) == ("twilio_api_key",)
    # Every other function keeps exactly what ACCESS lists.
    for name in set(ACCESS) - set(RESTRICTED_KEY_OF):
        assert secrets_of(name, cfg) == ACCESS[name][2], name
    # The Data stack creates a restricted secret exactly when its function reads it.
    created = set(secrets_for(cfg))
    assert (CONTEXT_API_KEY_SECRET in created) == (ctx == "on")


def test_each_restricted_secret_lands_in_the_variable_the_handlers_read() -> None:
    for name, (secret, _) in RESTRICTED_KEY_OF.items():
        assert SECRET_ENV[secret] == SECRET_ENV["twilio_api_key"] == HANDLER_ENV, name
    assert OPTIONAL_SECRETS[CONTEXT_API_KEY_SECRET][0] == "twilio/context-api-key"


# --------------------------------------------------------------------------- templates


def _res(synth, stack: str) -> dict:
    return synth.templates[stack]["Resources"]


def _lid(synth, stack: str, prefix: str, rtype: str) -> str:
    found = [lid for lid, r in _res(synth, stack).items() if r["Type"] == rtype and lid.startswith(prefix)]
    assert len(found) == 1, (prefix, found)
    return found[0]


def _secret_statements(synth, role_prefix: str) -> list[dict]:
    role = _lid(synth, COMPUTE, role_prefix, "AWS::IAM::Role")
    out = []
    for r in _res(synth, COMPUTE).values():
        if r["Type"] == "AWS::IAM::Policy" and {"Ref": role} in r["Properties"].get("Roles", []):
            out += [st for st in r["Properties"]["PolicyDocument"]["Statement"]
                    if any(str(a).startswith("secretsmanager:") for a in
                           (st["Action"] if isinstance(st["Action"], list) else [st["Action"]]))]
    return out


def _import_of(synth, logical: str) -> dict:
    """``Fn::ImportValue`` of the Data stack's export of ``logical``'s secret ARN."""
    lid = _lid(synth, DATA, "".join(p.capitalize() for p in logical.split("_")) + "Secret",
               "AWS::SecretsManager::Secret")
    names = [o["Export"]["Name"] for o in synth.templates[DATA].get("Outputs", {}).values()
             if o.get("Value") == {"Ref": lid} and "Export" in o]
    assert len(names) == 1, (logical, names)
    return {"Fn::ImportValue": names[0]}


def _env(synth, fn_prefix: str) -> dict:
    lid = _lid(synth, COMPUTE, fn_prefix, "AWS::Lambda::Function")
    return _res(synth, COMPUTE)[lid]["Properties"]["Environment"]["Variables"]


def _reads_only(synth, fn: str, logical: str) -> None:
    sts = _secret_statements(synth, f"{fn}Role")
    assert len(sts) == 1 and sts[0]["Action"] == SECRET_ACTIONS, sts
    own = _import_of(synth, logical)
    assert sts[0]["Resource"] == own
    assert _env(synth, f"{fn}Fn")[HANDLER_ENV] == own


def test_default_template(synth_default) -> None:
    _reads_only(synth_default, "ContextApi", CONTEXT_API_KEY_SECRET)
    _reads_only(synth_default, "FlexAuthorizer", "twilio_api_key")
    standard = json.dumps(_import_of(synth_default, "twilio_api_key"))
    assert standard not in json.dumps(_secret_statements(synth_default, "ContextApiRole"))


def test_flex_authorizer_grants_only_the_shared_api_key(synth_default) -> None:
    shared = _import_of(synth_default, "twilio_api_key")
    sts = _secret_statements(synth_default, "FlexAuthorizerRole")
    assert [st["Resource"] for st in sts] == [shared], sts
    assert _env(synth_default, "FlexAuthorizerFn")[HANDLER_ENV] == shared
    secrets = {r["Properties"]["Name"] for r in _res(synth_default, DATA).values()
               if r["Type"] == "AWS::SecretsManager::Secret"}
    assert not any("authorizer" in n for n in secrets), secrets


def test_removed_flag_is_refused_as_an_unknown_key(tmp_path_factory) -> None:
    # Not a deploy flag any more, so no secret can appear for it: the app refuses the key by name.
    with pytest.raises(ConfigError, match=REMOVED_FLAG):
        _synth(tmp_path_factory, "rk-removed-flag", {REMOVED_FLAG: "on"})
