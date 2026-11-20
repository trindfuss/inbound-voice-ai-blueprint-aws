"""AgentCore hosting config: every hosting-key parser
accepts exactly its domain; ``stage=prod`` + ``public`` without the acknowledgement always fails;
``agentcore`` + ``reuseConversation=false`` always fails, and ``fargate`` accepts either value.

The template part is in ``test_bridge_agentcore.py``. Example-based checks are in ``test_stage2_config.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank_infra.config import (
    AGENTCORE_IDLE_TIMEOUT_MIN_S,
    AGENTCORE_MAX_LIFETIME_S,
    AGENTCORE_NETWORK_MODES,
    AGENTCORE_PUBLIC_AT_PROD_ACK,
    AGENTCORE_SESSION_QUOTA_MAX,
    BRIDGE_HOSTINGS,
    ConfigError,
    from_context,
)

BASE = Path(__file__).resolve().parents[1]
DIGEST = "sha256:" + "ab" * 32
PROD = {"domainName": "owlbank.example.com",
        "certificateArn": "arn:aws:acm:us-east-1:111111111111:certificate/x"}


def _cfg(ctx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return from_context(ctx, base_dir=BASE)


def _outcome(ctx):
    """The parsed config, or the :class:`ConfigError` message."""
    try:
        return _cfg(ctx), None
    except ConfigError as err:
        return None, str(err)


def _spellings(word: str):
    """``word`` with random case and surrounding whitespace (all inside the domain)."""
    return st.tuples(st.sampled_from(["", " ", "\t"]),
                     st.lists(st.booleans(), min_size=len(word), max_size=len(word)),
                     st.sampled_from(["", " ", "\n"])).map(
        lambda t: t[0] + "".join(c.upper() if up else c for c, up in zip(word, t[1], strict=True)) + t[2])


def _texts(words):
    """Arbitrary text, near misses of ``words`` and in-domain spellings of them."""
    near = st.sampled_from(words).flatmap(lambda w: st.sampled_from(
        [w[:-1], w + "s", w.replace("-", "_"), "-" + w, w + "," + w, w[::-1], w.replace("a", "@")]))
    return st.one_of(st.text(max_size=20), near, st.sampled_from(words).flatmap(_spellings))


non_strings = st.one_of(st.booleans(), st.integers(), st.floats(allow_nan=False),
                        st.lists(st.sampled_from(["fargate", "vpc", AGENTCORE_PUBLIC_AT_PROD_ACK]), max_size=2))


def _choice_domain(raw, allowed) -> bool:
    return isinstance(raw, str) and (raw.strip() == "" or raw.strip().lower() in allowed)


# --- bridgeHosting -----------------------------------------------------------------------------------

@settings(max_examples=300)
@given(st.one_of(_texts(BRIDGE_HOSTINGS), non_strings))
def test_bridge_hosting_parser_accepts_exactly_its_domain(raw):
    cfg, err = _outcome({"bridgeHosting": raw, "bridgeImageDigest": DIGEST})
    if _choice_domain(raw, BRIDGE_HOSTINGS):
        assert cfg is not None, err
        assert cfg.bridge_hosting == (raw.strip().lower() or "fargate")
        assert cfg.is_agentcore is (cfg.bridge_hosting == "agentcore")
    else:
        assert cfg is None and "bridgeHosting" in err


# --- agentcoreNetworkMode (checked in both hostings) --------------------------------------------------

@settings(max_examples=300)
@given(st.one_of(_texts(AGENTCORE_NETWORK_MODES), non_strings), st.sampled_from(BRIDGE_HOSTINGS))
def test_network_mode_parser_accepts_exactly_its_domain(raw, hosting):
    cfg, err = _outcome({"bridgeHosting": hosting, "bridgeImageDigest": DIGEST, "agentcoreNetworkMode": raw})
    if _choice_domain(raw, AGENTCORE_NETWORK_MODES):
        assert cfg is not None, err
        mode = raw.strip().lower() or "vpc"
        assert cfg.agentcore_network_mode == mode
        assert cfg.agentcore_in_vpc is (hosting == "agentcore" and mode == "vpc")
    else:
        assert cfg is None and "agentcoreNetworkMode" in err


# --- agentcoreNetworkModeAck --------------------------------------------------------------------------

@settings(max_examples=300)
@given(st.one_of(_texts((AGENTCORE_PUBLIC_AT_PROD_ACK,)), non_strings), st.sampled_from(BRIDGE_HOSTINGS))
def test_ack_parser_accepts_exactly_its_domain(raw, hosting):
    cfg, err = _outcome({"bridgeHosting": hosting, "bridgeImageDigest": DIGEST,
                         "agentcoreNetworkModeAck": raw})
    if _choice_domain(raw, (AGENTCORE_PUBLIC_AT_PROD_ACK,)):
        assert cfg is not None, err
    else:
        assert cfg is None and "agentcoreNetworkModeAck" in err


@settings(max_examples=100)
@given(st.sampled_from(["", " ", "\t", "  \n"]))
def test_blank_ack_is_no_ack(raw):
    """A blank value never counts as the acknowledgement."""
    cfg, err = _outcome({**PROD, "stage": "prod", "bridgeHosting": "agentcore", "bridgeImageDigest": DIGEST,
                         "agentcoreNetworkMode": "public", "agentcoreNetworkModeAck": raw})
    assert cfg is None and "agentcoreNetworkModeAck" in err


# --- integer keys ------------------------------------------------------------------------------------

def _int_domain(raw, lo: int, hi: int):
    """The integer ``_int`` reads from ``raw``, or None when it must reject it."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() and lo <= raw <= hi else None
    if isinstance(raw, int):
        return raw if lo <= raw <= hi else None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if lo <= value <= hi else None


def _int_values(lo: int, hi: int):
    edges = st.sampled_from([lo - 1, lo, lo + 1, hi - 1, hi, hi + 1, 0, -1])
    ints = st.one_of(edges, st.integers(min_value=-10 * hi, max_value=10 * hi))
    return st.one_of(ints, ints.map(str), ints.map(float), st.floats(allow_nan=False, allow_infinity=False),
                     st.text(alphabet="0123456789 .-+_e", min_size=1, max_size=8), st.booleans())


@settings(max_examples=300)
@given(_int_values(AGENTCORE_IDLE_TIMEOUT_MIN_S, AGENTCORE_MAX_LIFETIME_S))
def test_idle_timeout_accepts_exactly_60_to_below_max_lifetime(raw):
    cfg, err = _outcome({"bridgeHosting": "agentcore", "bridgeImageDigest": DIGEST,
                         "agentcoreIdleTimeoutSeconds": raw})
    expected = _int_domain(raw, AGENTCORE_IDLE_TIMEOUT_MIN_S, AGENTCORE_MAX_LIFETIME_S - 1)
    if expected is not None:
        assert cfg is not None, err
        assert cfg.agentcore_idle_timeout_s == expected < AGENTCORE_MAX_LIFETIME_S
    else:
        assert cfg is None and "agentcoreIdleTimeoutSeconds" in err


@settings(max_examples=300)
@given(_int_values(1, AGENTCORE_SESSION_QUOTA_MAX))
def test_session_quota_accepts_exactly_1_to_max(raw):
    cfg, err = _outcome({"bridgeHosting": "agentcore", "bridgeImageDigest": DIGEST, "agentcoreSessionQuota": raw})
    expected = _int_domain(raw, 1, AGENTCORE_SESSION_QUOTA_MAX)
    if expected is not None:
        assert cfg is not None, err
        assert cfg.agentcore_session_quota == expected
    else:
        assert cfg is None and "agentcoreSessionQuota" in err


# --- combinations ------------------------------------------------------------------------------------

reuse_values = st.sampled_from([True, False, "true", "false", "True", "FALSE", None, ""])


def _reuse(raw) -> bool:
    return raw in (None, "") or raw is True or (isinstance(raw, str) and raw.lower() == "true")


@st.composite
def valid_contexts(draw, hosting: str):
    """Contexts whose every key is inside its domain (the combination rules decide the outcome)."""
    stage = draw(st.sampled_from(["dev", "test", "prod"]))
    ctx: dict = {"stage": stage, "bridgeHosting": draw(_spellings(hosting))}
    if stage == "prod":
        ctx.update(PROD)
    if hosting == "agentcore" or draw(st.booleans()):
        ctx["bridgeImageDigest"] = DIGEST
    if draw(st.booleans()):
        ctx["agentcoreNetworkMode"] = draw(st.sampled_from(AGENTCORE_NETWORK_MODES).flatmap(_spellings))
    if draw(st.booleans()):
        ctx["agentcoreNetworkModeAck"] = draw(_spellings(AGENTCORE_PUBLIC_AT_PROD_ACK))
    if draw(st.booleans()):
        ctx["agentcoreIdleTimeoutSeconds"] = draw(st.integers(AGENTCORE_IDLE_TIMEOUT_MIN_S,
                                                              AGENTCORE_MAX_LIFETIME_S - 1))
    if draw(st.booleans()):
        ctx["agentcoreSessionQuota"] = draw(st.integers(1, AGENTCORE_SESSION_QUOTA_MAX))
    ctx["reuseConversation"] = draw(reuse_values)
    return ctx


@settings(max_examples=300)
@given(valid_contexts("agentcore"))
def test_agentcore_combination_rules(ctx):
    public = ctx.get("agentcoreNetworkMode", "vpc").strip().lower() == "public"
    reuse_off = not _reuse(ctx["reuseConversation"])
    unacked_prod_public = ctx["stage"] == "prod" and public and "agentcoreNetworkModeAck" not in ctx
    cfg, err = _outcome(ctx)
    if reuse_off:
        assert cfg is None and "bridgeHosting=agentcore" in err and "reuseConversation" in err
    elif unacked_prod_public:
        assert cfg is None and "agentcoreNetworkModeAck" in err
    else:
        assert cfg is not None, err
        assert cfg.is_agentcore and cfg.agentcore_in_vpc is not public


@settings(max_examples=200)
@given(valid_contexts("fargate"))
def test_fargate_accepts_any_valid_combination(ctx):
    """``fargate`` takes either ``reuseConversation``, needs no digest or ack, and never runs in the
    Runtime's VPC mode."""
    cfg, err = _outcome(ctx)
    assert cfg is not None, err
    assert not cfg.is_agentcore and not cfg.agentcore_in_vpc
    assert cfg.reuse_conversation is _reuse(ctx["reuseConversation"])


@settings(max_examples=100)
@given(valid_contexts("agentcore"))
def test_agentcore_without_digest_is_judged_like_with_one(ctx):
    """The digest never decides config (the image is pushed before the deploy): removing it changes no outcome."""
    without = {k: v for k, v in ctx.items() if k != "bridgeImageDigest"}
    (c1, e1), (c2, e2) = _outcome(ctx), _outcome(without)
    assert (c1 is None) == (c2 is None) and e1 == e2
    if c2 is not None:
        assert c2.bridge_image_digest is None


@pytest.mark.parametrize("mode", ["public", " PUBLIC "])
def test_public_warning_is_emitted_only_in_agentcore(mode):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from_context({"bridgeHosting": "agentcore", "bridgeImageDigest": DIGEST,
                      "agentcoreNetworkMode": mode}, base_dir=BASE)
    assert any("egress control" in str(w.message) for w in record)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        from_context({"agentcoreNetworkMode": mode}, base_dir=BASE)
    assert not any("egress control" in str(w.message) for w in record)
    assert any("ignored" in str(w.message) for w in record)
