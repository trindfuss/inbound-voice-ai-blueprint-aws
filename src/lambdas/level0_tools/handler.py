"""AgentCore Gateway Lambda target for the level-0 tools (thin; logic in ``service``).

Gateway contract (from the AWS
"Gateway Lambda targets" docs): ``event`` is the tool's arguments object, and the
tool name is ``context.client_context.custom["bedrockAgentCoreToolName"]`` as
``"<target>___<tool>"``; everything up to and including ``___`` is stripped.
[unverified until first deploy: exact casing and nesting of ``client_context``.]

Tools: ``get_branch_hours({"branch"?: str, "lang"?: "en-US"|"es-US"})`` and
``faq_lookup({"question": str, "lang"?: ...})``. Content comes from
``data/faq.json`` (override the path with ``OWLBANK_FAQ_PATH``). No customer data,
no token, no AWS calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from lambdas.level0_tools import service
from owlbank import obs

log = logging.getLogger(__name__)

TOOL_NAME_KEY = "bedrockAgentCoreToolName"
DELIMITER = "___"


def tool_name(context: Any) -> str | None:
    """The tool name from the Gateway's client context, without the target prefix."""
    client_context = getattr(context, "client_context", None)
    custom = getattr(client_context, "custom", None)
    if custom is None and isinstance(client_context, dict):
        custom = client_context.get("custom")
    if not isinstance(custom, dict):
        return None
    name = custom.get(TOOL_NAME_KEY)
    if not isinstance(name, str) or not name:
        return None
    return name.split(DELIMITER, 1)[1] if DELIMITER in name else name


@obs.instrumented("level0_tools", op=lambda _event, ctx: tool_name(ctx))
def handler(event: Any, context: Any = None) -> dict[str, Any]:
    if isinstance(event, (str, bytes)):
        try:
            event = json.loads(event)
        except ValueError:
            log.warning("level0_tools: event is not JSON", extra={
                "operation": "parse_event", "decision": "return_invalid_argument"})
            return {"error": "INVALID_ARGUMENT"}
    name = tool_name(context)
    if name is None:
        log.warning("level0_tools: no tool name in client context", extra={"decision": "return_unknown_tool"})
        return {"error": "UNKNOWN_TOOL"}
    try:
        content = service.load_content(os.environ.get("OWLBANK_FAQ_PATH") or None)
        return service.dispatch(content, name, event if event is not None else {})
    except Exception as exc:
        obs.log_caught(log, "level0_tools tool failed", exc, level=logging.ERROR, tool=name[:64],
                       decision="return_unavailable")
        obs.degraded("tool_failed")
        return {"error": "UNAVAILABLE"}
