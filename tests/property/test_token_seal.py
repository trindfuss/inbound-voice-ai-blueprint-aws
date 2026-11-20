"""Token sealing: round-trip, call binding, no plaintext in the sealed value."""

from __future__ import annotations

import base64

import boto3
import pytest
from hypothesis import given
from hypothesis import strategies as st
from moto import mock_aws

from owlbank.token_seal import FakeSealer, KmsSealer, UnsealError

tokens = st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=16, max_size=64)
sids = st.from_regex(r"CA[0-9a-f]{32}", fullmatch=True)


@given(tokens, sids)
def test_fake_round_trip(token, sid):
    s = FakeSealer()
    sealed = s.seal(token, call_sid=sid)
    assert token not in sealed
    assert s.unseal(sealed, call_sid=sid) == token


@given(tokens, sids, sids)
def test_fake_wrong_call_fails(token, sid, other):
    if sid == other:
        return
    s = FakeSealer()
    with pytest.raises(UnsealError):
        s.unseal(s.seal(token, call_sid=sid), call_sid=other)


def test_fake_garbage_fails():
    with pytest.raises(UnsealError):
        FakeSealer().unseal("not-base64!!", call_sid="CA" + "0" * 32)


@mock_aws
def test_kms_round_trip_and_context_binding():
    kms = boto3.client("kms", region_name="us-east-1")
    key = kms.create_key()["KeyMetadata"]["KeyId"]
    s = KmsSealer(kms, key)
    sid, other = "CA" + "1" * 32, "CA" + "2" * 32
    sealed = s.seal("tok-abc-123-xyz", call_sid=sid)
    assert "tok-abc-123-xyz" not in sealed
    assert b"tok-abc-123-xyz" not in base64.b64decode(sealed)
    assert s.unseal(sealed, call_sid=sid) == "tok-abc-123-xyz"
    with pytest.raises(UnsealError):
        s.unseal(sealed, call_sid=other)
