"""``flex_mode`` is the one switch.

A pure property over ``config.py`` (no synth): for any ``twilioOutputs``, exactly one of
``workspaceSid`` and ``workflowSid`` present is a synth error naming ``--with-flex``; otherwise the
human leg, the TaskRouter route, the human-leg env value and the briefing flag all read the same
``flex_mode``. Validates R 1.3, 2.4, 3.2.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from owlbank_infra.config import ConfigError, InfraConfig

SIDS = st.one_of(st.none(), st.just(""), st.text(alphabet="WSQ0123456789", min_size=1, max_size=34))
EXTRA = st.dictionaries(st.sampled_from(["memoryStoreId", "accountSid", "queueSids"]), st.text(min_size=1, max_size=8))
FLAGS = st.sampled_from(["on", "off"])
CEILINGS = st.sampled_from(["deterministic", "assisted", "generative"])
OBS = st.sampled_from(["on", "off"])


@given(ws=SIDS, wf=SIDS, extra=EXTRA, human=FLAGS, ceiling=CEILINGS, obs=OBS)
def test_p15_one_flex_mode_drives_everything(ws, wf, extra, human, ceiling, obs) -> None:
    outputs = dict(extra)
    for key, value in (("workspaceSid", ws), ("workflowSid", wf)):
        if value is not None:
            outputs[key] = value
    cfg = InfraConfig(twilio_outputs=outputs, human_leg_conversation=human, slot_mode_ceiling=ceiling,
                      call_observability=obs)
    present = [bool(ws), bool(wf)]
    if present[0] != present[1]:
        with pytest.raises(ConfigError, match="--with-flex"):
            _ = cfg.flex_mode
        return
    flex = cfg.flex_mode
    assert flex == all(present)
    assert cfg.human_leg_conversation_on == (flex and human == "on")
    assert cfg.taskrouter_route_on == (flex and (obs == "on" or human == "on"))
    assert cfg.briefing_on == ((not flex) and ceiling != "deterministic")
    # Without Flex nothing human-leg is on; the briefing and the human leg are never both on.
    assert not (cfg.briefing_on and cfg.human_leg_conversation_on)
    if not flex:
        assert not cfg.human_leg_conversation_on and not cfg.taskrouter_route_on
