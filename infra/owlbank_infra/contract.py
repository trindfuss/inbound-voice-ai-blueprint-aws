"""What the stacks share with the application code in ``../src``.

* :data:`TABLES` duplicates ``owlbank.store.SCHEMAS`` (and ``NUMERIC_KEYS``), and
  :data:`FAILOVER_TABLES` ``owlbank.store.FAILOVER_SCHEMAS``, on purpose, so the CDK app doesn't depend on the application package for its data
  model. ``tests/test_table_schemas.py`` imports ``owlbank.store`` and fails if
  the two ever differ.
* :func:`gateway_level0_tool_names` and :func:`gateway_target_name` read
  ``owlbank.tools_contract`` at synth time, so the Gateway target and the bridge
  agree on the level-0 tool names with no copy to drift.
* :func:`level0_tool_definitions` describes the Gateway Lambda target's tools,
  with branch IDs taken from ``data/faq.json``.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class TableDef:
    pk: str
    sk: str | None = None
    #: index name -> (pk, sk)
    indexes: tuple[tuple[str, str, str | None], ...] = ()
    ttl: bool = False
    #: Indexes that project only the keys (``KEYS_ONLY``); every other index projects ``ALL``.
    keys_only: tuple[str, ...] = ()
    #: Indexes no role gets through ``grants.grant_tables``: each is granted by its own statement to
    #: the one function that reads it (``sweep-due``: the conversation sweep's ``dynamodb:Query``).
    own_grant: tuple[str, ...] = ()

    def projection(self, index: str) -> str:
        return "KEYS_ONLY" if index in self.keys_only else "ALL"


#: The ``calls`` table's index the conversation sweep queries (``owlbank.callstate.SWEEP_INDEX``).
SWEEP_DUE_INDEX = "sweep-due"
#: Logical tables. Keep identical to owlbank.store.SCHEMAS.
TABLES: dict[str, TableDef] = {
    "customers": TableDef(pk="customer_id", indexes=(("phone-index", "phone", None),)),
    "tokens": TableDef(pk="token_hash", indexes=(("call_sid-index", "call_sid", None),), ttl=True),
    # Human-leg: the sparse ``sweep-due`` index of escalated calls whose conversation is not
    # closed (keys derived by ``owlbank.callstate.CallState.to_item``), read by the sweep only.
    "calls": TableDef(pk="call_sid", indexes=((SWEEP_DUE_INDEX, "sweep_open", "sweep_due_at"),), ttl=True,
                      keys_only=(SWEEP_DUE_INDEX,), own_grant=(SWEEP_DUE_INDEX,)),
    "segments": TableDef(pk="call_sid", sk="seq", ttl=True),
    "policy": TableDef(pk="pk"),
    "catalog": TableDef(pk="key", sk="lang"),
    # Untrusted-ANI failure slots; auth_service only.
    "risk": TableDef(pk="risk_key", ttl=True),
}
#: The model-failover table. Keep identical to owlbank.store.FAILOVER_SCHEMAS.
#: Not in TABLES: it is always built (retained, with TTL, no point-in-time recovery, no stream and no
#: index), and only the bridge and the poller get its name, the latter two only with ``modelFailover=on``.
FAILOVER_TABLES: dict[str, TableDef] = {
    "modelstats": TableDef(pk="s", sk="sk", ttl=True),
}
#: Key attributes stored as DynamoDB numbers; every other key is a string.
NUMERIC_KEYS: frozenset[str] = frozenset({"seq", "sweep_due_at"})
TTL_ATTRIBUTE = "expires_at"


def _import_from_src(module: str) -> Any:
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(module)


def gateway_level0_tool_names() -> tuple[str, ...]:
    return tuple(_import_from_src("owlbank.tools_contract").GATEWAY_LEVEL0_TOOLS)


def gateway_target_name() -> str:
    """The Gateway target name (``owlbank.tools_contract``)."""
    return str(_import_from_src("owlbank.tools_contract").GATEWAY_TARGET_NAME)


def admin_api_routes() -> list[tuple[str, str]]:
    """``lambdas.admin_api.app.ROUTES`` as ``(method, path template)``: the admin API's routes,
    so the REST API and its web ACLs are built from the handler's own table."""
    return [(r.method, r.template) for r in _import_from_src("lambdas.admin_api.app").ROUTES]


def level0_tool_definitions() -> list[dict[str, Any]]:
    """Gateway tool schemas for the ``level0_tools`` Lambda target (no customer data)."""
    faq = json.loads((DATA_DIR / "faq.json").read_text())
    branch_ids = [b["id"] for b in faq.get("branches", [])]
    lang = {"type": "string", "description": "Caller language: en-US or es-US."}
    defs = {
        "get_branch_hours": {
            "description": "Opening hours and address of an OwlBank branch, or the phone "
                           "support hours when no branch is given. Contains no customer data.",
            "input_schema": {"type": "object",
                             "properties": {"branch": {"type": "string",
                                                       "description": "Branch id: " + ", ".join(branch_ids)},
                                            "lang": lang},
                             "required": []},
        },
        "faq_lookup": {
            "description": "Answer a general banking question from OwlBank's published FAQ "
                           "(lost card, opening hours, fees). Contains no customer data.",
            "input_schema": {"type": "object",
                             "properties": {"question": {"type": "string",
                                                         "description": "The caller's question in their words."},
                                            "lang": lang},
                             "required": ["question"]},
        },
    }
    names = gateway_level0_tool_names()
    missing = set(names) - set(defs)
    if missing:
        raise RuntimeError(f"no Gateway schema for level-0 tools {sorted(missing)}")
    return [{"name": n, **defs[n]} for n in names]
