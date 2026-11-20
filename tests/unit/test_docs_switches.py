"""Every switch the production guide lists for the human leg exists in the source tree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = (ROOT / "docs" / "production-guide.md").read_text()
START = GUIDE.index("### The human leg: transcript, Memory and retention")
SECTION = GUIDE[START:GUIDE.index("### Web activity in Memory", START)]

#: Name the guide's human-leg section uses -> a file that must contain it.
NAMES = {
    "humanLegConversation": "infra/owlbank_infra/config.py",
    "contextApiRestrictedKey": "infra/owlbank_infra/config.py",
    "twilio/context-api-key": "infra/owlbank_infra/data_stack.py",
    "human_leg": "src/owlbank/policy.py",
    "agent_legs": "src/owlbank/policy.py",
    "`memory`": "src/owlbank/policy.py",
    "memoryExtractionEnabled": "scripts/twilio_setup.py",
    "intelligenceConfigurationIds": "scripts/twilio_setup.py",
    "statusTimeouts": "scripts/twilio_setup.py",
    "/voice/taskrouter": "src/lambdas/frontdoor/app.py",
    "/voice/transcription": "infra/owlbank_infra/api_stack.py",
    "scripts/populate_secrets.py": "scripts/populate_secrets.py",
}


def test_each_named_switch_is_in_the_guide_section_and_in_source():
    for name, rel in NAMES.items():
        assert name in SECTION, f"{name} is not in the guide section"
        target = ROOT / rel
        assert target.exists(), rel
        if name.endswith(".py"):
            continue
        assert name in target.read_text(), f"{name} not found in {rel}"


def test_guide_section_has_no_price_claim():
    assert "$" not in SECTION
