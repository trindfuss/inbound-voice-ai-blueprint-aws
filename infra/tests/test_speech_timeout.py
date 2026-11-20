"""The front door's Gather speechTimeout setting (``speechTimeout`` context)."""

from __future__ import annotations

import pytest

from pathlib import Path

from owlbank_infra.config import ConfigError, InfraConfig, from_context

BASE = Path(__file__).resolve().parent.parent


def _cfg(**ctx: object) -> InfraConfig:
    return from_context(dict(ctx), base_dir=BASE)


@pytest.mark.parametrize("raw, expected", [(None, "3"), ("auto", "auto"), ("AUTO", "auto"),
                                           ("3", "3"), (2, "2"), ("10", "10")])
def test_speech_timeout_values(raw: object, expected: str) -> None:
    ctx = {} if raw is None else {"speechTimeout": raw}
    assert _cfg(**ctx).speech_timeout == expected


@pytest.mark.parametrize("raw", ["0", "11", "1.5", "fast", True])
def test_bad_speech_timeout_is_refused(raw: object) -> None:
    with pytest.raises(ConfigError):
        _cfg(speechTimeout=raw)
