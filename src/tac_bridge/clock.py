"""The bank's local date and time for the agent's system prompt.

The agent otherwise has no idea what day or time it is, so it can't answer "are you open now?" or
"what are Saturday's hours?" from the branch-hours tool sensibly. The bank's time zone is a
deployment setting (``timezone``, IANA name, default ``America/New_York``). The text is built here,
not by the model, in the call's language. Pure: ``now`` is passed in.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TIMEZONE = "America/New_York"

_WEEKDAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
             "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]}
_MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                  "October", "November", "December"],
           "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                  "octubre", "noviembre", "diciembre"]}


def zone(name: str | None) -> ZoneInfo:
    """The IANA zone, or the default if ``name`` is empty or unknown."""
    try:
        return ZoneInfo(name or DEFAULT_TIMEZONE)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo(DEFAULT_TIMEZONE)


def local_now_line(now_utc: datetime, tz: ZoneInfo, lang: str) -> str:
    """One system-prompt line with the bank's local date and time, in the call's language."""
    t = now_utc.astimezone(tz)
    fam = "es" if lang.lower().startswith("es") else "en"
    day, month = _WEEKDAYS[fam][t.weekday()], _MONTHS[fam][t.month - 1]
    if fam == "es":
        return (f"Fecha y hora actuales del banco ({tz.key}): {day} {t.day} de {month} de {t.year}, "
                f"{t:%H:%M}.")
    hour = t.hour % 12 or 12
    return (f"Current date and time at the bank ({tz.key}): {day}, {month} {t.day}, {t.year}, "
            f"{hour}:{t:%M} {'AM' if t.hour < 12 else 'PM'}.")
