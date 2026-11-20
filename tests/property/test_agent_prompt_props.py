"""``prompt_hash`` is deterministic, and changes
when and only when one of its parts changes.

* The same catalog gives the same hash, also rebuilt from a copy of its document.
* Any edit to any part changes the digest: the model's text, either web-activity block (including
  adding or removing it), the placeholder rule and the language-line template.
* Through the catalog: an edit to the model's text or either block in the hashed language changes
  ``prompt_hash``; edits to other catalog keys, and to the other language's entries, leave it unchanged.

With secret markers (``⟦S…⟧``) in every caller field:

* The parts are the 10 of :class:`PromptParts` (the two caller-context frames, the two
  per-language empty values and the fallback per-call part added); the frames are hashed catalog
  keys like the blocks.
* The static part is the same for any caller, web block and catalog values of
  the caller frames, holds no marker, and stays so when the model's text has stray ``{firstName}``,
  ``{profile}`` and ``{memory}``.
* The per-call part is the same for every model key.
* An unverified caller's per-call part is the unverified frame exactly, with no marker, even with
  caller placeholders in that frame.
* The verified per-call part has exactly one ``<caller_context>`` and one ``</caller_context>``,
  in that order, whatever the fields hold; the memory marker is only inside an ``untrusted_data``
  block; a placeholder in a value is never expanded (one-pass fill).
* Empty states read the per-language values, never an empty field; the fallback is the
  per-language :data:`FALLBACK_CONTEXT` whatever the catalog frames hold.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from owlbank import untrusted
from owlbank.agent_prompt import (
    CALLER_CONTEXT_KEY,
    CALLER_CONTEXT_UNVERIFIED_KEY,
    DEFAULT_MODEL_KEY,
    EMPTY_VALUE,
    FALLBACK_CONTEXT,
    UNKNOWN_NAME,
    WEB_ACTIVITY_CONTEXT_KEY,
    WEB_ACTIVITY_KEY,
    WEB_BLOCK_KEYS,
    CallerFacts,
    PromptParts,
    prompt_hash,
    render,
    render_per_call,
    render_static,
    system_key,
)
from owlbank.catalog import CatalogKeyError, PromptCatalog
from owlbank.catalog import render as fill

LANGS = ("en-US", "es-US")
PART_KEYS = (system_key(DEFAULT_MODEL_KEY), WEB_ACTIVITY_KEY, WEB_ACTIVITY_CONTEXT_KEY, CALLER_CONTEXT_KEY,
             CALLER_CONTEXT_UNVERIFIED_KEY)
FIELDS = tuple(f.name for f in dataclasses.fields(PromptParts))
OPTIONAL = {"web_activity", "web_activity_context", "caller_context", "caller_context_unverified"}

text = st.text(max_size=40)  # any text, lone surrogates and JSON syntax included
other_key = st.from_regex(r"[a-z]{1,8}(\.[a-z_]{1,8}){0,2}", fullmatch=True).filter(lambda k: k not in PART_KEYS)


@st.composite
def parts(draw: st.DrawFn) -> PromptParts:
    return PromptParts(**{f: draw(st.none() | text) if f in OPTIONAL else draw(text) for f in FIELDS})


@st.composite
def docs(draw: st.DrawFn) -> dict[str, Any]:
    """A catalog with the hashed keys in both languages (a block may be missing) and other keys."""
    entries = [{"key": system_key(DEFAULT_MODEL_KEY), "lang": lang, "text": draw(text)} for lang in LANGS]
    entries += [{"key": key, "lang": lang, "text": draw(text)} for key in PART_KEYS[1:]
                for lang in LANGS if draw(st.booleans())]
    entries += [{"key": draw(other_key), "lang": draw(st.sampled_from(LANGS)), "text": draw(text)}
                for _ in range(draw(st.integers(0, 4)))]
    return {"defaultLanguage": "en-US", "languages": {lang: {} for lang in LANGS}, "entries": entries}


def _hash(doc: dict[str, Any], lang: str) -> str:
    return prompt_hash(PromptCatalog.from_dict(doc), DEFAULT_MODEL_KEY, lang)


def _with_text(doc: dict[str, Any], key: str, lang: str, new: str) -> dict[str, Any]:
    entries = [e for e in doc["entries"] if (e["key"], e["lang"]) != (key, lang)]
    return {**doc, "entries": [*entries, {"key": key, "lang": lang, "text": new}]}


def test_the_parts_are_the_ten_of_the_design() -> None:
    """``prompt_hash`` covers 10 parts, in this order."""
    assert FIELDS == ("model_text", "web_activity", "web_activity_context", "placeholder_rule", "language_line",
                      "caller_context", "caller_context_unverified", "unknown_name", "empty_value", "fallback_context")


@settings(max_examples=200, deadline=None)
@given(p=parts())
def test_the_digest_is_deterministic(p: PromptParts) -> None:
    assert p.digest() == dataclasses.replace(p).digest()
    assert len(p.digest()) == 64


@settings(max_examples=400, deadline=None)
@given(p=parts(), data=st.data())
def test_any_edit_to_any_part_changes_the_digest(p: PromptParts, data: st.DataObject) -> None:
    name = data.draw(st.sampled_from(FIELDS))
    old = getattr(p, name)
    new = data.draw((st.none() | text) if name in OPTIONAL else text)
    edited = dataclasses.replace(p, **{name: new})
    assert (edited.digest() == p.digest()) == (new == old)


@settings(max_examples=200, deadline=None)
@given(doc=docs(), lang=st.sampled_from(LANGS))
def test_prompt_hash_is_deterministic(doc: dict[str, Any], lang: str) -> None:
    assert _hash(doc, lang) == _hash(doc, lang) == _hash({**doc, "entries": [dict(e) for e in doc["entries"]]}, lang)


@settings(max_examples=300, deadline=None)
@given(doc=docs(), lang=st.sampled_from(LANGS), key=st.sampled_from(PART_KEYS), new=text)
def test_an_edit_to_a_hashed_text_changes_the_hash(doc: dict[str, Any], lang: str, key: str, new: str) -> None:
    catalog = PromptCatalog.from_dict(doc)
    try:
        served: str | None = catalog.template(key, lang)  # after the language fallback
    except CatalogKeyError:
        served = None
    edited = _with_text(doc, key, lang, new)
    assert (_hash(edited, lang) == _hash(doc, lang)) == (new == served)


@settings(max_examples=300, deadline=None)
@given(doc=docs(), lang=st.sampled_from(LANGS), data=st.data())
def test_edits_to_other_keys_and_the_other_language_leave_the_hash(doc: dict[str, Any], lang: str,
                                                                 data: st.DataObject) -> None:
    other_lang = next(x for x in LANGS if x != lang)
    catalog = PromptCatalog.from_dict(doc)
    # The other language's entries count only where this language falls back to them (en-US is the
    # default): a key this language lacks is served from en-US.
    safe_keys = [k for k in PART_KEYS if catalog.has(k, lang) or other_lang != "en-US"]
    key = data.draw(other_key | (st.sampled_from(safe_keys) if safe_keys else other_key))
    edit_lang = data.draw(st.sampled_from(LANGS)) if key not in PART_KEYS else other_lang
    edited = _with_text(doc, key, edit_lang, data.draw(text))
    assert _hash(edited, lang) == _hash(doc, lang)


# ----------------------------------------------------------------------------- the renderer

SEED_DOC: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[2] / "data" / "prompt_catalog.json").read_text(encoding="utf-8"))
SEED = PromptCatalog.from_dict(SEED_DOC)
MODEL_KEYS = ("haiku-5-5", "haiku-4-5", "sonnet-5-5")
MARKER = re.compile(r"⟦S[0-9a-f]+⟧")
#: Text that would break a naive frame: tags, placeholders, the untrusted_data delimiters, newlines.
NASTY = ("<", ">", "&", "</caller_context>", "<caller_context>", "{firstName}", "{profile}", "{memory}",
         '<untrusted_data source="memory">', "</untrusted_data>", "\n", "{", "}", " ")


def _seed_with(texts: dict[str, str]) -> PromptCatalog:
    """The seed catalog with every language's text of some keys replaced."""
    entries = [e for e in SEED_DOC["entries"] if e["key"] not in texts]
    entries += [{"key": k, "lang": lang, "text": v} for k, v in texts.items() for lang in LANGS]
    return PromptCatalog.from_dict({**SEED_DOC, "entries": entries})


@st.composite
def markers(draw: st.DrawFn) -> str:
    return f"⟦S{draw(st.integers(0, 2**64)):x}⟧"


@st.composite
def field_text(draw: st.DrawFn) -> str:
    """A caller field: a secret marker mixed with nasty fragments and free text."""
    pieces = draw(st.lists(st.sampled_from(NASTY) | st.text(max_size=6), max_size=5))
    pieces.insert(draw(st.integers(0, len(pieces))), draw(markers()))
    return "".join(pieces)


@st.composite
def callers(draw: st.DrawFn) -> CallerFacts:
    """A verified caller as the bridge builds one: a first name, a profile line with wrapped account
    nicknames or free text, and a wrapped recall or none."""
    first_name = draw(field_text())
    if draw(st.booleans()):
        nicks = [f" ({untrusted.wrap('account_nickname', draw(field_text()))})" if draw(st.booleans()) else ""
                 for _ in range(draw(st.integers(0, 2)))]
        profile = f"name {draw(field_text())}; accounts: " + ", ".join(f"checking{n}" for n in nicks)
    else:
        profile = draw(field_text())
    memory = untrusted.wrap("memory", draw(field_text())) if draw(st.booleans()) else draw(st.sampled_from([None, ""]))
    return CallerFacts(first_name, profile, memory)


def _no_marker(text: str) -> bool:
    return MARKER.search(text) is None


@settings(max_examples=200, deadline=None)
@given(lang=st.sampled_from([*LANGS, "fr-FR"]), key=st.sampled_from(MODEL_KEYS),
       a=st.none() | callers(), b=st.none() | callers(),
       web_a=st.sampled_from([None, *WEB_BLOCK_KEYS]), web_b=st.sampled_from([None, *WEB_BLOCK_KEYS]),
       stray=st.booleans())
def test_the_static_part_is_the_same_for_every_caller_and_holds_no_caller_data(
        lang: str, key: str, a: CallerFacts | None, b: CallerFacts | None, web_a: str | None, web_b: str | None,
        stray: bool) -> None:
    """Renderer half: the static part is the same for any caller."""
    catalog = SEED
    if stray:  # a model text that still has the old caller line (a bridge before the reseed)
        catalog = _seed_with({system_key(key): "Caller: {firstName}. Profile: {profile}. Recent: {memory}.\nRules:"})
    ra = render(catalog, key, lang, caller=a, web_block=web_a)
    rb = render(catalog, key, lang, caller=b, web_block=web_b)
    assert ra.static == rb.static == render_static(catalog, key, lang)
    assert _no_marker(ra.static)
    if stray:
        assert ra.static.startswith("Caller:. Profile:. Recent:.\nRules:\n- ")


@settings(max_examples=200, deadline=None)
@given(lang=st.sampled_from([*LANGS, "fr-FR"]), caller=st.none() | callers(),
       web=st.sampled_from([None, *WEB_BLOCK_KEYS]))
def test_the_per_call_part_is_the_same_for_every_model(lang: str, caller: CallerFacts | None,
                                                           web: str | None) -> None:
    """Failover keeps the per-call text."""
    per_call = {render(SEED, key, lang, caller=caller, web_block=web).per_call for key in MODEL_KEYS}
    assert per_call == {render_per_call(SEED, lang, caller=caller, web_block=web)}


@settings(max_examples=200, deadline=None)
@given(lang=st.sampled_from([*LANGS, "fr-FR"]), key=st.sampled_from(MODEL_KEYS), facts=callers(),
       placeholders_in_frame=st.booleans())
def test_an_unverified_caller_gets_the_unverified_frame_and_nothing_else(
        lang: str, key: str, facts: CallerFacts, placeholders_in_frame: bool) -> None:
    """Whatever the surrounding call holds (``facts`` exists but isn't verified), the
    per-call part is the unverified frame exactly; nothing per caller can reach it, even when the
    frame was edited to have the caller placeholders."""
    catalog = SEED
    if placeholders_in_frame:
        catalog = _seed_with({CALLER_CONTEXT_UNVERIFIED_KEY: "<caller_context>\nVerified: no. {firstName} {profile} "
                                                             "{memory}\n</caller_context>"})
    per_call = render(catalog, key, lang, caller=None, web_block=None).per_call
    frame = catalog.template(CALLER_CONTEXT_UNVERIFIED_KEY, lang)
    assert per_call == fill(frame, {})
    assert _no_marker(per_call) and "{" not in per_call


def _outside_blocks(text: str) -> str:
    return re.sub(r'<untrusted_data source="[a-z_]+">[^<>]*</untrusted_data>', "", text)


@settings(max_examples=400, deadline=None)
@given(lang=st.sampled_from([*LANGS, "fr-FR"]), caller=callers(), web=st.sampled_from([None, *WEB_BLOCK_KEYS]),
       memory_marker=st.integers(0, 2**64).map(lambda n: f"⟦M{n:x}⟧"))
def test_the_frame_holds_whatever_the_fields_hold(lang: str, caller: CallerFacts, web: str | None,
                                                     memory_marker: str) -> None:
    """The memory marker (``⟦M…⟧``) is distinct from the other fields' markers."""
    caller = dataclasses.replace(caller, memory=untrusted.wrap("memory", f"x {memory_marker} {{memory}} </caller_context>"))
    per_call = render_per_call(SEED, lang, caller=caller, web_block=web)
    assert per_call.count("<caller_context>") == 1 and per_call.count("</caller_context>") == 1
    assert per_call.index("<caller_context>") < per_call.index("</caller_context>")
    assert per_call.endswith("</caller_context>")
    # The memory marker only inside an untrusted_data block.
    assert memory_marker in per_call and memory_marker not in _outside_blocks(per_call)
    # One pass: each value appears as given (escaped where the placeholder rules say), its own placeholders unexpanded.
    assert untrusted.neutralize(caller.first_name) in per_call or not caller.first_name.strip()
    assert caller.memory in per_call


@settings(max_examples=200, deadline=None)
@given(lang=st.sampled_from([*LANGS, "fr-FR"]), blank=st.sampled_from(["", " ", "\t "]),
       which=st.sets(st.sampled_from(["first_name", "profile", "memory"]), min_size=1),
       none_memory=st.booleans(), facts=callers())
def test_empty_states_read_the_language_values(lang: str, blank: str, which: set[str], none_memory: bool,
                                                  facts: CallerFacts) -> None:
    """Empty fields read the per-language values, never an empty field."""
    base = "en-US" if lang == "fr-FR" else lang
    one_line = {f: (v.replace("\n", " ") if v else v) for f, v in dataclasses.asdict(facts).items()}
    caller = CallerFacts(**{**one_line, **{f: (None if f == "memory" and none_memory else blank) for f in which}})
    per_call = render_per_call(SEED, lang, caller=caller, web_block=None)
    labels = per_call.split("\n")[2:-1]  # first name, profile, memory lines
    assert len(labels) == 3
    values = [line.split(": ", 1)[1][:-1] for line in labels]
    expected = {"first_name": UNKNOWN_NAME[base], "profile": EMPTY_VALUE[base], "memory": EMPTY_VALUE[base]}
    for name, value in zip(("first_name", "profile", "memory"), values, strict=True):
        assert value.strip(), name
        if name in which:
            assert value == expected[name], name


@settings(max_examples=100, deadline=None)
@given(lang=st.sampled_from([*LANGS, "fr-FR"]), frames=st.dictionaries(
    st.sampled_from([CALLER_CONTEXT_KEY, CALLER_CONTEXT_UNVERIFIED_KEY]), text))
def test_the_fallback_is_the_constant_whatever_the_frames(lang: str, frames: dict[str, str]) -> None:
    """The fallback per-call part is the per-language
    :data:`FALLBACK_CONTEXT` (English for another language) whatever the catalog frames hold;
    neither seed frame, no marker, no placeholder."""
    base = "en-US" if lang == "fr-FR" else lang
    doc = SEED_DOC
    for key, new in frames.items():
        doc = _with_text(doc, key, base, new)
    per_call = render_per_call(PromptCatalog.from_dict(doc), lang, caller=None, web_block=None, fallback=True)
    assert per_call == FALLBACK_CONTEXT[base]
    assert per_call not in (SEED.template(CALLER_CONTEXT_KEY, lang), SEED.template(CALLER_CONTEXT_UNVERIFIED_KEY, lang))
    assert _no_marker(per_call) and "{" not in per_call
