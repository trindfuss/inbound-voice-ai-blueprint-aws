"""Lambda entry point for the mock bank API (thin: wiring only; logic in ``service``).

Invoked by Lambda direct invoke (IAM) from the front door (``ani_lookup``,
``get_balance`` for the deterministic flow) and the TAC bridge task role (inline
tools, with the call token attached by the bridge). No API, URL or other trigger.

Event: ``{"op": "<operation>", "token": "<call token>", "call_sid": "CA…", "args": {...}}``

* ``ani_lookup``: ``args = {"ani": "+1…"}``; no token. → ``{"tool": "ani_lookup",
  "data": {"candidates": ["C-1002", "C-1003"], "withheld": false}}``
* ``get_balance``: ``args = {"account": "checking"|"savings"}`` → ``data =
  {accountType, last4, available, current, pending, currency, asOf}``
* ``list_recent_transactions``: ``{"account", "limit"?}`` → ``{accountType, last4, transactions}``
* ``change_address``: ``{"street", "city", "state", "postal_code"}`` → ``{status, address}``
* ``make_payment``: ``{"from_account", "to": "red"|"purple"|"rewards" (the policy's ``card`` options), "amount": "250.00"}``
* ``transfer_funds``: ``{"from_account", "to_account", "amount"}``
  → ``{confirmation, amount, fromAccount, toAccount}``

Errors: ``{"error": "AUTH_LEVEL_INSUFFICIENT"|"NOT_FOUND"|"UNAVAILABLE"|"INVALID_ARGUMENT"|
"INSUFFICIENT_FUNDS"|"UNKNOWN_OPERATION", "tool": op}``. ``customer_id``/``account_id``
(or similar) anywhere in the event is ``INVALID_ARGUMENT``.

Cold start: outside Lambda, importing this module makes no AWS call. Inside Lambda, the
dependencies (including the first policy read) are built once during the INIT phase
(``coldstart.build_at_init``), which a provisioned-concurrency environment completes before any
call arrives.
"""

from __future__ import annotations

import os
from typing import Any

from lambdas.bank_tools import service
from owlbank import coldstart, obs
from owlbank.config import Settings
from owlbank.policy import Policy
from owlbank.store import DynamoStore

_deps: service.BankDeps | None = None


def build_deps(env: dict[str, str] | None = None) -> service.BankDeps:
    settings = Settings.from_env(dict(os.environ) if env is None else env)
    store = DynamoStore.from_settings(settings)
    return service.BankDeps(store=store, policy=Policy.from_store(store, ttl_s=settings.cache_ttl_s))


@obs.instrumented("bank_tools")
def handler(event: Any, context: Any = None) -> dict[str, Any]:
    global _deps
    if _deps is None:
        _deps = build_deps()
    return service.handle_event(_deps, event)


# Inside Lambda only (a no-op in tests and scripts): wire up now, in the INIT phase.
_deps = coldstart.build_at_init(build_deps)
