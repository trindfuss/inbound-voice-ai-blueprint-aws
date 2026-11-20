"""Admin prompt editing by exclusion (keyed-in phone login):

* For any key and any texts, ``PUT /prompts/{key}`` succeeds only if the key isn't
  protected and every text is non-empty, within :data:`MAX_PROMPT_CHARS`, free of markup, has
  no stray brace and keeps the current text's placeholder set; a protected key is never
  written; only an accepted write that changes something is audited, with exactly one entry,
  and a refused write changes nothing. Model instructions (``agent.*``,
  never spoken) have their own limit, :data:`MAX_AGENT_PROMPT_CHARS`, and may contain ``<`` and
  ``>``; the other rules hold for them too.
* For an existing key outside the ``mask.``/``arg.``/``option.`` checks, model instructions
  included, the rules are also sufficient: a write that satisfies them is accepted.
* Parity: for any text, the API's check and the console's ``checkPromptTexts`` agree on accept
  or refuse, for ``agent.*`` and for spoken keys (run through node; skipped without it).
* ``mask.*`` phrases are exempt from the same-placeholder rule only: a
  write is accepted exactly when the other generic rules hold and every placeholder is one the
  owning tool allows, with a catalog word for each of its values in that language.
* The API and console half: for any ``agent.*``
  key (shipped, new or made up) and any text, the API's check and the console's
  ``checkPromptTexts`` accept exactly when the text passes the length and blank rules and
  :func:`~owlbank.agent_prompt.placeholder_problems` is ``None``, whatever the current text's
  placeholders are. This replaces the same-set rule for ``agent.*`` only.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lambdas.admin_api import config_routes
from lambdas.admin_api.config_routes import (
    AGENT_PREFIX,
    MAX_AGENT_PROMPT_CHARS,
    MAX_PROMPT_CHARS,
    NEW_KEY_PREFIXES,
    protected_reason,
)
from lambdas.admin_api.http import ApiError
from owlbank import masking_config as mc
from owlbank.agent_prompt import placeholder_problems
from owlbank.catalog import PLACEHOLDER
from owlbank.policy import DEFAULT_CONFIRM, confirm_prompt_keys
from owlbank.store import MemoryStore
from tests.admin_api_fakes import call, claims, make_api, seeded_store

_SEED = seeded_store()
EDITOR = claims(["admin-editor"])
_CMETA = _SEED.get("catalog", {"key": "_meta", "lang": "_"})
LANGS: tuple[str, ...] = tuple(_CMETA["languages"])
KEYS: list[str] = list(_CMETA["keys"])
TOOLS: list[str] = list(_SEED.get("policy", {"pk": "meta"})["tools"])
SINGLE: dict[str, str] = dict(_CMETA.get("singleLanguageKeys") or {})
PROTECTED = {k for t in [*TOOLS, *DEFAULT_CONFIRM] for k in confirm_prompt_keys(t)}


def fresh_store() -> MemoryStore:
    store = MemoryStore()
    store._tables = copy.deepcopy(_SEED._tables)
    return store


def current_text(key: str, lang: str) -> str | None:
    item = _SEED.get("catalog", {"key": key, "lang": lang})
    return item.get("text") if item else None


def limit_for(key: str) -> int:
    return MAX_AGENT_PROMPT_CHARS if key.startswith(AGENT_PREFIX) else MAX_PROMPT_CHARS


def placeholders(text: str | None) -> frozenset[str]:
    return frozenset(mc.placeholder_names(text)) if isinstance(text, str) else frozenset()


_META = _SEED.get("policy", {"pk": "meta"})
REASONS: tuple[str, ...] = tuple(_META.get("reasons") or ())
_PHRASE_KEYS: dict[str, str] = {
    t: (_SEED.get("policy", {"pk": f"tool#{t}"}) or {}).get("mask", {}).get("phrase_key") or f"mask.{t}"
    for t in TOOLS}


def mask_owner(key: str) -> str | None:
    """The tool whose phrase pool ``key`` is in (``<phrase_key>`` or ``<phrase_key>.<n>``)."""
    for t, base in _PHRASE_KEYS.items():
        if key == base or (key.startswith(base + ".") and key[len(base) + 1:].isdigit()):
            return t
    return None


def mask_placeholders_ok(key: str, lang: str, text: str) -> bool:
    """validate_phrase's rule, stated from the seed: every placeholder is an enumerated
    argument of the owning tool and each of its values has a brace-free catalog word."""
    owner = mask_owner(key)
    allowed = mc.placeholders_for(owner, reasons=REASONS) if owner else {}
    for name in placeholders(text):
        if name not in allowed:
            return False
        for value in allowed[name]:
            word = current_text(mc.word_key(name, value), lang)
            if word is None or not word.strip() or "{" in word or "}" in word:
                return False
    return True


# --------------------------------------------------------------------------- strategies

_names = st.sampled_from(["firstName", "amount", "account", "options", "label", "x", "fromAccount",
                          "from_account", "to_account", "to", "reason", "profile", "memory"])
_fragments = st.one_of(
    st.sampled_from(["Hello", " ", "thanks.", "¿Qué?", "{", "}", "<", ">", "<speak>", "&amp;",
                     "{ firstName}", "{{x}}", "\n", "x" * 300]),
    _names.map(lambda n: "{" + n + "}"),
    st.text(max_size=12),
)
_composed = st.lists(_fragments, max_size=6).map("".join)

_generated_keys = st.one_of(
    st.sampled_from(sorted(PROTECTED)),
    st.sampled_from(TOOLS).map(lambda t: f"confirm.{t}"),
    st.from_regex(r"(confirm|agent|greeting|mask|arg|option|tool|clarify)\.[a-z0-9_.-]{1,20}",
                  fullmatch=True),
    st.sampled_from(["_meta", "Greeting.generic", "mask.{x}", "confirm", "agent", "x"]),
)
keys = st.one_of(st.sampled_from(KEYS), st.sampled_from(KEYS), _generated_keys)


@st.composite
def text_for(draw: st.DrawFn, key: str, lang: str) -> str:
    """A text near the current one (kept, extended, placeholders dropped or added) or anything."""
    cur = current_text(key, lang) or current_text(key, LANGS[0]) or ""
    kind = draw(st.sampled_from(["same", "suffix", "prefix", "strip", "add", "long", "any", "any"]))
    if kind == "same":
        return cur
    if kind == "suffix":
        return cur + draw(_composed)
    if kind == "prefix":
        return draw(_composed) + cur
    if kind == "strip":
        return PLACEHOLDER.sub(draw(st.sampled_from(["", "you", "{x}"])), cur)
    if kind == "add":
        return cur + " {" + draw(_names) + "}"
    if kind == "long":  # around the key's limit
        limit = limit_for(key)
        return cur + "a" * draw(st.integers(max(0, limit - 5 - len(cur)), limit + 5 - len(cur)))
    return draw(_composed)


@st.composite
def writes(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    key = draw(keys)
    langs = draw(st.sampled_from([LANGS, LANGS, (SINGLE.get(key, LANGS[0]),), LANGS[:1]]))
    texts = {lang: draw(text_for(key, lang)) for lang in langs}
    return key, texts




def _catalog_snapshot(store: MemoryStore) -> list[tuple[Any, ...]]:
    return sorted((it["key"], it["lang"], repr(sorted(it.items()))) for it in store.scan("catalog"))


def _audits(store: MemoryStore) -> list[dict[str, Any]]:
    return store.scan("admin_audit")


def _key_langs(key: str) -> tuple[str, ...]:
    return (SINGLE[key],) if key in SINGLE else LANGS


def _text_ok(key: str, text: str) -> bool:
    """Non-empty, within the key's limit, and markup-free unless the key is a model instruction
    (``agent.*``: up to MAX_AGENT_PROMPT_CHARS, ``<`` and ``>`` allowed)."""
    agent = key.startswith(AGENT_PREFIX)
    if not text.strip() or len(text) > limit_for(key):
        return False
    return agent or ("<" not in text and ">" not in text)


def _generic_ok(key: str, texts: dict[str, str]) -> bool:
    """The generic rules for an existing key, as the requirement states them: an
    ``agent.*`` text has its key's allowed set, any other the current text's."""
    if set(texts) != set(_key_langs(key)):
        return False
    for lang, text in texts.items():
        cur = current_text(key, lang)
        if cur is None:
            cur = current_text(key, LANGS[0])
        if not _text_ok(key, text):
            return False
        if key.startswith(AGENT_PREFIX):
            if placeholder_problems(key, text) is not None:
                return False
        elif mc.has_stray_brace(text) or placeholders(text) != placeholders(cur):
            return False
    return True


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(writes())
def test_prompt_edit_outcomes(write: tuple[str, dict[str, Any]]) -> None:
    key, texts = write
    api, store, _, _ = make_api(fresh_store())
    before, audits_before = _catalog_snapshot(store), len(_audits(store))
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    protected = protected_reason(key, TOOLS)

    # The exclusion rule matches the design's protected set (and nothing protected is missed).
    if key in PROTECTED or key.startswith("confirm."):
        assert protected is not None
    else:  # nothing else is protected: agent.* is editable
        assert protected is None, (key, protected)

    if protected is not None:
        assert status == 403 and body["error"] == "protected", (key, status, body)
    if status != 200:
        assert _catalog_snapshot(store) == before, (key, status, body)
        assert len(_audits(store)) == audits_before
        return

    # Accepted: never protected; every text non-empty, within the limit, markup-free; an
    # existing key keeps its placeholder set exactly, except a mask phrase, whose placeholders
    # are the owning tool's (with catalog words) and never another.
    assert protected is None
    exists = key in KEYS
    for lang, text in texts.items():
        assert _text_ok(key, text), (key, lang, len(text))
        if key.startswith(mc.MASK_PREFIX):
            assert not mc.has_stray_brace(text)
            assert mask_placeholders_ok(key, lang, text), (key, lang, text)
        elif key.startswith(AGENT_PREFIX):
            assert exists and placeholder_problems(key, text) is None, (key, lang, text)
        elif exists:
            cur = current_text(key, lang) or current_text(key, LANGS[0])
            assert placeholders(text) == placeholders(cur), (key, lang, text, cur)
            assert not mc.has_stray_brace(text)
        else:
            assert key.startswith(NEW_KEY_PREFIXES)
        assert store.get("catalog", {"key": key, "lang": lang})["text"] == text
    # Only accepted writes that change something are audited, with exactly one entry.
    if body["changed"]:
        new = [e for e in _audits(store) if e["pk"] == f"catalog#{key}"]
        assert len(new) == 1 and len(_audits(store)) == audits_before + 1
    else:
        assert _catalog_snapshot(store) == before and len(_audits(store)) == audits_before


_plain_keys = [k for k in KEYS if not k.startswith(NEW_KEY_PREFIXES) and protected_reason(k, TOOLS) is None]


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(_plain_keys).flatmap(
    lambda k: st.fixed_dictionaries({lang: text_for(k, lang) for lang in _key_langs(k)}).map(lambda t: (k, t))))
def test_generic_rules_decide_plain_keys(write: tuple[str, dict[str, str]]) -> None:
    """For a caller-facing key with no prefix-specific check, accepted exactly when the generic
    rules hold (such prompts are editable and the rules refuse)."""
    key, texts = write
    api, store, _, _ = make_api(fresh_store())
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert (status == 200) == _generic_ok(key, texts), (key, texts, status, body)
    if status == 400 and body["error"] == "placeholders":
        assert body["field"].startswith("texts[")


_agent_keys = [k for k in KEYS if k.startswith("agent.")]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(_agent_keys).flatmap(
    lambda k: st.fixed_dictionaries({lang: text_for(k, lang) for lang in _key_langs(k)}).map(lambda t: (k, t))))
def test_agent_keys_follow_their_own_rules(write: tuple[str, dict[str, str]]) -> None:
    """Model instructions are editable under their own rules (up to
    MAX_AGENT_PROMPT_CHARS, markup allowed, the key's allowed placeholder set), and every
    accepted write that changes something is audited exactly once, as a security change."""
    key, texts = write
    api, store, _, _ = make_api(fresh_store())
    before, audits_before = _catalog_snapshot(store), len(_audits(store))
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert status != 403, body
    assert (status == 200) == _generic_ok(key, texts), (key, texts, status, body)
    if status == 200 and body["changed"]:
        [entry] = [e for e in _audits(store) if e["pk"] == f"catalog#{key}"]
        assert entry["security_change"] is True
        assert len(_audits(store)) == audits_before + 1
    else:
        assert _catalog_snapshot(store) == before and len(_audits(store)) == audits_before


_mask_keys = [k for k in KEYS if k.startswith(mc.MASK_PREFIX)]


@st.composite
def mask_text(draw: st.DrawFn, key: str, lang: str) -> str:
    """A mask phrase near the current one, with placeholders gained, lost or swapped."""
    cur = current_text(key, lang) or current_text(key, LANGS[0]) or ""
    base = draw(st.sampled_from([cur, PLACEHOLDER.sub("it", cur), "One moment."]))
    extra = draw(st.lists(_names.map(lambda n: "{" + n + "}"), max_size=2))
    return draw(st.one_of(st.just(base + " " + " ".join(extra)), text_for(key, lang)))


def _mask_ok(key: str, texts: dict[str, str]) -> bool:
    if set(texts) != set(_key_langs(key)):
        return False
    return all(text.strip() and len(text) <= MAX_PROMPT_CHARS and "<" not in text and ">" not in text
               and not mc.has_stray_brace(text) and mask_placeholders_ok(key, lang, text)
               for lang, text in texts.items())


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(_mask_keys).flatmap(
    lambda k: st.fixed_dictionaries({lang: mask_text(k, lang) for lang in _key_langs(k)}).map(lambda t: (k, t))))
def test_mask_phrases_follow_their_own_placeholder_rule(write: tuple[str, dict[str, str]]) -> None:
    """A mask phrase may gain or lose a placeholder its tool allows, never
    take one the tool doesn't have; the other generic rules still apply; accepted writes are
    audited exactly once and refused ones change nothing."""
    key, texts = write
    api, store, _, _ = make_api(fresh_store())
    before, audits_before = _catalog_snapshot(store), len(_audits(store))
    status, body = call(api, "PUT", f"/prompts/{key}", {"rev": 0, "texts": texts}, who=EDITOR)
    assert (status == 200) == _mask_ok(key, texts), (key, texts, status, body)
    if status == 200 and body["changed"]:
        assert len([e for e in _audits(store) if e["pk"] == f"catalog#{key}"]) == 1
        assert len(_audits(store)) == audits_before + 1
    else:
        assert _catalog_snapshot(store) == before and len(_audits(store)) == audits_before


def test_mask_phrase_gains_an_allowed_placeholder_and_refuses_another() -> None:
    """Concrete anchors for the mask rule: get_balance has {account}, not {amount}."""
    assert "account" in mc.placeholders_for("get_balance", reasons=REASONS)
    api, store, _, _ = make_api(fresh_store())
    ok = dict.fromkeys(_key_langs("mask.get_balance.1"), "Checking your {account} balance.")
    assert call(api, "PUT", "/prompts/mask.get_balance.1", {"rev": 0, "texts": ok}, who=EDITOR)[0] == 200
    bad = dict.fromkeys(_key_langs("mask.get_balance.1"), "Checking your {amount} balance.")
    status, body = call(api, "PUT", "/prompts/mask.get_balance.1", {"rev": 1, "texts": bad}, who=EDITOR)
    assert status == 400 and body["field"].startswith("texts["), body
    assert len([e for e in _audits(store) if e["pk"] == "catalog#mask.get_balance.1"]) == 1


def test_protected_set_covers_the_design() -> None:
    """The read-backs and outcomes of every deterministic-confirm tool and clarify.yes_no."""
    for key in PROTECTED:
        assert protected_reason(key, TOOLS) == "protected read-back"
    assert "clarify.yes_no" in PROTECTED
    assert protected_reason("clarify.digits", TOOLS) is None
    assert protected_reason("login.phone_number", TOOLS) is None
    assert all(protected_reason(k, TOOLS) is None for k in KEYS if k.startswith(("agent.", "mask.")))


# --------------------------------------------------------------------------- API/console parity

_NODE = shutil.which("node")
_JS_RULES = Path(__file__).with_name("js_prompt_rules.mjs")
_ODD = [chr(c) for c in (0x85, 0xA0, 0x1C, 0x2028, 0x3000, 0xFEFF, 0x1F600, 0xE9)]
_CALLER_LINE = ("Caller: {firstName}. Verified profile: {profile}. "
                "Recent interactions from other channels: {memory}.")
_parity_fragments = st.one_of(
    _fragments,
    st.sampled_from(_ODD),
    st.sampled_from(["<rules>", "</rules>", "a > b", "{firstName}", "{a}", "{b}", "{ a}", "}{",
                     "{profile}", "{memory}", "{firstName} {profile} {memory}", _CALLER_LINE]),
    st.integers(590, 610).map(lambda n: "a" * n),
    st.integers(15_990, 16_010).map(lambda n: "a" * n),
    st.integers(590, 610).map(lambda n: chr(0x1F600) * n),
)
#: Every shipped agent.* key, both caller-context frames, a new model key, and made-up agent.* keys.
_p12_keys = st.one_of(
    st.sampled_from(sorted({*_agent_keys, "agent.caller_context", "agent.caller_context_unverified",
                            "agent.system.opus-5-5", "agent.system.x"})),
    st.from_regex(r"agent\.[a-z0-9_.-]{1,24}", fullmatch=True),
)
_currents = st.sampled_from([None, "Hi {firstName}.", "Hello.", "{a} and {b}",
                             "{firstName} {profile} {memory}", "Rules.\n" + _CALLER_LINE])
_parity_case = st.fixed_dictionaries({
    "key": st.one_of(_p12_keys, st.sampled_from(["greeting.personal", "greeting.generic", "option.plan.gold"])),
    "current": _currents,
    "text": st.lists(_parity_fragments, max_size=5).map("".join),
})


def _api_accepts(key: str, text: str, current: str | None) -> bool:
    expected = None if current is None else frozenset(mc.placeholder_names(current))
    try:
        config_routes._check_prompt_text(key, text, "texts[en-US]", expected)
    except ApiError:
        return False
    return True


def _p12_accepts(key: str, text: str) -> bool:
    """The oracle for an agent.* text: the unchanged length and blank rules."""
    assert key.startswith(AGENT_PREFIX)
    return bool(text.strip()) and len(text) <= MAX_AGENT_PROMPT_CHARS and placeholder_problems(key, text) is None


@settings(max_examples=400, deadline=None)
@given(_p12_keys, st.lists(_parity_fragments, max_size=6).map("".join), _currents)
def test_the_api_accepts_an_agent_text_exactly_when_holds(key: str, text: str, current: str | None) -> None:
    """Whatever the current text has (a pre-feature caller line, another
    set, or no current text at all for a new key or a restore), the API accepts an agent.* text
    exactly when placeholder_problems is None (and the 16,000 and blank rules hold); a
    placeholder refusal is a 400 placeholders whose message is placeholder_problems'."""
    assert _api_accepts(key, text, current) == _p12_accepts(key, text), (key, current, text[:80])
    problem = placeholder_problems(key, text)
    if problem is not None and text.strip() and len(text) <= MAX_AGENT_PROMPT_CHARS:
        expected = None if current is None else frozenset(mc.placeholder_names(current))
        with pytest.raises(ApiError) as err:
            config_routes._check_prompt_text(key, text, "texts[en-US]", expected)
        assert (err.value.status, err.value.error, err.value.message) == (400, "placeholders", problem)


@pytest.mark.skipif(_NODE is None, reason="node is not installed")
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(_parity_case, min_size=1, max_size=40))
def test_api_and_console_agree_on_prompt_texts(cases: list[dict[str, Any]]) -> None:
    """For any text, the API and the console agree on accept or refuse, for agent.* and for
    spoken keys (the shared table tests/fixtures/prompt_text_rules.json pins the boundaries).
    For an agent.* key both accept exactly when the allow-list holds (placeholder_problems is None)."""
    assert _NODE is not None
    run = subprocess.run([_NODE, str(_JS_RULES)], input=json.dumps(cases), capture_output=True,  # noqa: S603
                         text=True, encoding="utf-8", check=True, timeout=30)
    js = json.loads(run.stdout)
    for case, js_ok in zip(cases, js, strict=True):
        api_ok = _api_accepts(case["key"], case["text"], case["current"])
        assert api_ok == js_ok, (case["key"], case["current"], case["text"][:80], len(case["text"]))
        if case["key"].startswith(AGENT_PREFIX):
            assert js_ok == _p12_accepts(case["key"], case["text"]), (case["key"], case["text"][:80])
