"""Front-door session sidecar: per-call facts that ``owlbank.callstate.CallState`` has no field for.

Stored in the ``calls`` table under ``call_sid = "<CallSid>#session"`` (the ``#``
keeps it apart from the ``CallState`` item and its ``#v<n>`` claims). Same TTL as the
call. It holds the front door's own flow state (``lookup``, the factor ``flow`` in
progress, step-up and reconnect counters, a known Orchestrator ``conversation_id``, whether the
auth service was asked to prepare the welcome offer).
The bridge contract (``pending_result``, ``interrupted_utterance``, ``sealed_token``,
``pending_tool``, ``language``, ``level``) lives in ``CallState``.

* ``flow["held"]``: during disambiguation only, a factor already collected on this
  call, *sealed* with the call's ``Sealer`` (KMS, ``CallSid`` context), so it can be
  re-sent with the disambiguating factor (the auth service is stateless). Deleted as
  soon as that verify call returns.
* Model-assisted slots: ``flow["modes"]`` (each slot's mode decision, frozen at
  flow start), ``flow["tries"]`` (``{slot: n}``, this flow), ``flow["pending"]`` (the
  follow-up question being answered, :class:`owlbank.slots.Pending` as JSON with
  ``sealed_value`` sealed, and ``listed``, the order it read options in),
  ``flow["listed"]`` (``{slot, values}``: the order a question with nothing pending, such as
  ``prompt.card``, read a choice slot's options in; read once by the next answer),
  ``flow["earlier_strikes"]``, ``flow["seq"]`` and ``flow["api_calls"]`` (the API
  calls the front door has made in this flow, counted by :meth:`SlotFlow.api_call`).
  :attr:`Session.collected` lists the slots collected on the call, across flows: name, slot
  type, the flow's ``seq``, whether it was sent to the auth service, and ``at_api_call``, the
  flow's ``api_calls`` when the slot was first collected in that flow (a slot is correctable only
  while no API call has happened since). Names only, never a value.

Never stored here: plaintext factor values or digits. :func:`save` runs the
redaction guard (registered values) over the whole item.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from owlbank.segments import assert_clean
from owlbank.store import Store

TABLE = "calls"
SUFFIX = "#session"


@dataclass(slots=True)
class Session:
    call_sid: str
    expires_at: int
    lookup: str = "pending"  # pending | ok | error | withheld
    flow: dict[str, Any] | None = None
    stepups: int = 0
    relay_failures: int = 0
    ended: bool = False
    ended_at: int | None = None
    conversation_id: str | None = None  # Orchestrator conversation, if a handoff named it
    stream_reconnects: int = 0  # agentcore: silent reconnects after the 60-minute cap (<= 4)
    flow_seq: int = 0  # flows started on this call; the current flow's ``flow["seq"]``
    #: The auth service's ``prepare_welcome`` was started on this call (once
    #: per call, whether or not the invoke was accepted).
    welcome_requested: bool = False
    #: Slots collected on this call:
    #: ``{name, slot_type, flow_seq, verified, at_api_call}``.
    collected: list[dict[str, Any]] = field(default_factory=list)
    #: Keyed-in phone login: keyed-number lookups tried on this call (at most 2). The
    #: number itself is sealed in the call state's sensitive store, never kept here.
    keyed_tries: int = 0

    def key(self) -> str:
        return f"{self.call_sid}{SUFFIX}"


def load(store: Store, call_sid: str) -> Session | None:
    item = store.get(TABLE, {"call_sid": f"{call_sid}{SUFFIX}"})
    if item is None:
        return None
    names = {f.name for f in fields(Session)}
    data = {k: v for k, v in item.items() if k in names}
    data["call_sid"] = call_sid
    return Session(**data)


def save(store: Store, session: Session) -> None:
    item = copy.deepcopy(asdict(session))
    assert_clean(session.call_sid, item, deny_keys=False)
    item["call_sid"] = session.key()
    store.put(TABLE, item)
