"""The Flex agent skills the OwlBank queues select on, and the pure merge that gives them to a worker.

Shared by ``twilio_setup.py`` (defines the names in Flex) and ``assign_agent_skills.py`` (gives them to
named agents). A unit test keeps :data:`AGENT_SKILLS` in step with the queue targets in ``twilio_setup.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

#: One skill per queue target (``routing.skills HAS "<name>"``).
AGENT_SKILLS: tuple[str, ...] = ("en", "es", "fraud", "disputes")


def merge_worker_skills(attributes_json: str, wanted: Iterable[str]) -> str | None:
    """A worker's attributes JSON with the missing ``wanted`` skills appended to ``routing.skills``.

    Every other attribute (and ``routing.levels``) is kept as it is, existing skills keep their order.
    Returns None when nothing is missing, and when the attributes aren't a JSON object (never overwritten)."""
    try:
        attrs = json.loads(attributes_json or "{}")
    except (ValueError, TypeError):
        return None
    if not isinstance(attrs, dict):
        return None
    routing = attrs.get("routing")
    if routing is None:
        routing = {}
    if not isinstance(routing, dict):
        return None
    skills = routing.get("skills")
    if skills is None:
        skills = []
    if not isinstance(skills, list):
        return None
    missing = [n for n in dict.fromkeys(wanted) if n not in skills]
    if not missing:
        return None
    attrs["routing"] = {**routing, "skills": [*skills, *missing]}
    return json.dumps(attrs)
