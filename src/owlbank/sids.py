"""Twilio SID patterns shared by the Lambdas and the bridge."""

from __future__ import annotations

import re

#: A Twilio CallSid: ``CA`` + 32 lowercase hex. Also the AgentCore Runtime session ID (a CallSid is
#: 34 characters, inside AgentCore's 33..256), so the front door's signer and the
#: Runtime warmer accept exactly the same values.
CALL_SID_RE = re.compile(r"^CA[0-9a-f]{32}\Z")
