"""The tools as the agent sees them.

Customer-data, handoff and escalation tools are **inline tools**: the bridge's Strands
loop (``tac_bridge.agent_client_strands``) pauses on the call and the TAC bridge runs
them. Level-0 tools (:data:`GATEWAY_LEVEL0_TOOLS`) are AgentCore Gateway Lambda
targets and are defined on the Gateway, not here.

Rules every schema here follows:

* **No customer or account ID inputs.** The customer comes from the token. Accounts
  are chosen by type (``checking``/``savings``), never by number; a card by its product
  (``make_payment.to``, one of the ``card`` slot's options), never by number.
* **No token, factor or level inputs.** ``authenticate`` takes only the tool name.
* **Amounts are decimal strings**, never floats.

``validate_args`` implements just the JSON Schema subset used below (object,
properties, required, additionalProperties, enum, string min/max length and
pattern, integer min/max), plus a few cross-field checks.

Two views of each schema: :func:`tool_schemas` is the FULL schema and the
source of truth for ``validate_args``. :func:`inline_tool_specs` is the model-facing view
the Strands loop offers the model. It carries no booleans or numbers (see
:func:`sanitize_schema`): the schemas were first bound into a CloudFormation Json property,
which turns every scalar into a string, and the model-facing view has kept that shape
since. The limits are stated in each property's ``description`` instead, and enforced
server-side.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Collection, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, Protocol

ACCOUNT_TYPES: tuple[str, ...] = ("checking", "savings")

#: Escalation reasons (handover "Escalation to a human"); the policy table may override.
DEFAULT_ESCALATION_REASONS: tuple[str, ...] = (
    "caller_requested_human", "unable_to_resolve", "step_up_failed",
    "fraud_suspected", "card_dispute", "complaint", "vulnerable_customer",
)
#: The seed's reason descriptions (``data/policy.json``, ``escalation.reasons.<name>.description``), like
#: :data:`DEFAULT_CARDS`; a unit test keeps them in step. They go into the
#: ``escalate`` tool's ``reason`` description, so the model tells the reasons apart from data. A
#: caller may pass the live policy's (:attr:`owlbank.policy.Policy.reason_descriptions`).
DEFAULT_REASON_DESCRIPTIONS: Mapping[str, str] = {
    "fraud_suspected": "The caller thinks a card or account is compromised or being misused by someone else.",
    "card_dispute": "The caller disputes a charge or doesn't recognise one, or follows up on a dispute "
                    "they already raised.",
}
FALLBACK_REASON = "unable_to_resolve"

#: The ``card`` choice slot: ``make_payment.to`` is one of its known options.
CARD_SLOT = "card"
#: The ``phone`` digits slot: the account phone number a caller
#: whose caller ID found no customer keys in or says. Its accepted value is a claim of identity
#: that the front door looks up (``FrontDoor._claim_phone``), never a factor for the auth service.
PHONE_SLOT = "phone"
#: The seed's ``card`` options (``data/policy.json``) with their English spoken names (catalog
#: ``option.card.<value>``), like :data:`DEFAULT_ESCALATION_REASONS`. The bridge's agent offers
#: these; the bridge and the bank tools check ``to`` against the live policy's options
#: (:func:`card_values`).
DEFAULT_CARDS: Mapping[str, str] = {
    "red": "OwlBank Red Card",
    "purple": "OwlBank Purple Card",
    "rewards": "OwlBank Rewards Card",
}
SUMMARY_MAX_LEN = 1000

#: Level-0 tools served by AgentCore Gateway (Lambda target ``level0_tools``).
GATEWAY_LEVEL0_TOOLS: tuple[str, ...] = ("get_branch_hours", "faq_lookup")
#: Name of the Gateway target. The Gateway names each tool ``<target>___<tool>``, and the
#: bridge's approved-tool filter uses that full name (:func:`gateway_tool_full_name`).
GATEWAY_TARGET_NAME = "level0"

#: Customer-data tools: the bridge checks the level, then executes with the token.
BANK_TOOLS: tuple[str, ...] = (
    "list_recent_transactions", "change_address", "make_payment", "transfer_funds",
)
#: Deterministic tools: level check, then handoff to a scripted flow.
DETERMINISTIC_TOOLS: tuple[str, ...] = ("get_balance",)
#: Tools that need a verified level (valid ``authenticate`` targets).
PROTECTED_TOOLS: tuple[str, ...] = DETERMINISTIC_TOOLS + BANK_TOOLS
#: The tool the model calls when the caller is done: a hand-off with reason
#: :data:`CALLER_DONE`; the front door says the goodbye and hangs up. Not a bank tool.
END_CALL = "end_call"
#: The hand-off reason of an ``end_call`` (HandoffData ``reason``).
CALLER_DONE = "caller_done"
#: Tools that end the ConversationRelay session (handoff, escalation or the call's end).
HANDOFF_TOOLS: frozenset[str] = frozenset({"get_balance", "authenticate", "escalate", END_CALL})

#: Money-moving bank operations: each call carries an ``idempotency_key`` (the model's
#: ``tool_use_id``) as a top-level event field, and ``bank_tools`` applies it once.
IDEMPOTENT_OPS: frozenset[str] = frozenset({"make_payment", "transfer_funds"})

AUTH_LEVEL_INSUFFICIENT = "AUTH_LEVEL_INSUFFICIENT"
#: ``end_call`` refused while something is pending; the result names ``pending``.
CALL_IN_PROGRESS = "CALL_IN_PROGRESS"
#: A tool the policy switched off was called anyway (``end_call.enabled`` false).
TOOL_DISABLED = "TOOL_DISABLED"
#: ``authenticate`` refused after the call's login ended without verification (keyed-phone).
VERIFICATION_CLOSED = "VERIFICATION_CLOSED"

_AMOUNT = {"type": "string", "pattern": r"^(0|[1-9][0-9]{0,8})(\.[0-9]{1,2})?$",
           "description": "Amount in US dollars as a decimal string, for example \"250.00\"."}
_ACCOUNT = {"type": "string", "enum": list(ACCOUNT_TYPES)}


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


class _HasSlots(Protocol):
    def slot(self, name: str) -> Any: ...


def card_values(policy: _HasSlots) -> tuple[str, ...]:
    """The policy's ``card`` known options, in policy order (``()`` without a ``card`` slot, so
    every ``to`` is refused: fail closed)."""
    try:
        slot = policy.slot(CARD_SLOT)
    except KeyError:
        return ()
    return tuple(o.value for o in slot.options)


def _card_schema(cards: Collection[str]) -> dict[str, Any]:
    """``make_payment.to``: one of ``cards``. A mapping gives each value's spoken name."""
    names = cards if isinstance(cards, Mapping) else {}
    listed = ", ".join(f"{v} = {names[v]}" if v in names else v for v in cards)
    return {"type": "string", "enum": list(cards),
            "description": f"The caller's card to pay, if they named it: {listed}. Leave it out "
                           "if the caller hasn't said which card; a separate step asks for and "
                           "confirms the card with the caller."}


def _reason_schema(reasons: Collection[str], descriptions: Mapping[str, str]) -> dict[str, Any]:
    """``escalate.reason``: one of ``reasons``; the description lists each listed reason that has
    a description. No description at all when none has one."""
    schema: dict[str, Any] = {"type": "string", "enum": list(reasons)}
    described = [f"{r}: {descriptions[r]}" for r in reasons if descriptions.get(r)]
    if described:
        schema["description"] = ("Why the caller needs a human; choose the value whose description "
                                 "fits. " + " ".join(described))
    return schema


def _schemas(reasons: Collection[str],
             cards: Collection[str] = DEFAULT_CARDS,
             reason_descriptions: Mapping[str, str] = DEFAULT_REASON_DESCRIPTIONS,
             ) -> dict[str, tuple[str, dict[str, Any]]]:
    return {
        "get_balance": (
            "Read the caller's balance for one account aloud. If the caller hasn't said "
            "which account, ask them first. Never guess.",
            _obj({"account": _ACCOUNT}, ["account"]),
        ),
        "list_recent_transactions": (
            "List the caller's most recent transactions on one account. Ask which account "
            "if the caller hasn't said.",
            _obj({"account": _ACCOUNT,
                  "limit": {"type": "integer", "minimum": 1, "maximum": 10,
                           "description": "Number of transactions, 1 to 10."}}, ["account"]),
        ),
        "change_address": (
            "Change the caller's mailing address. Confirm the full new address with the "
            "caller before calling.",
            _obj({
                "street": {"type": "string", "minLength": 1, "maxLength": 100,
                           "description": "Street address, 1 to 100 characters."},
                "city": {"type": "string", "minLength": 1, "maxLength": 60,
                         "description": "City, 1 to 60 characters."},
                "state": {"type": "string", "pattern": r"^[A-Z]{2}$",
                          "description": "Two-letter uppercase US state code, for example \"CA\"."},
                "postal_code": {"type": "string", "pattern": r"^[0-9]{5}(-[0-9]{4})?$",
                                "description": "ZIP code, 5 digits or ZIP+4, for example \"94105\"."},
            }, ["street", "city", "state", "postal_code"]),
        ),
        "make_payment": (
            "Pay toward one of the caller's cards from one of their accounts. Call it as soon "
            "as you have the account and the amount; a separate step confirms the card and the "
            "payment with the caller.",
            _obj({"from_account": _ACCOUNT, "to": _card_schema(cards), "amount": _AMOUNT},
                 ["from_account", "amount"]),
        ),
        "transfer_funds": (
            "Move money between the caller's checking and savings accounts. Call it as soon "
            "as you have the accounts and the amount; a separate step confirms with the caller.",
            _obj({"from_account": _ACCOUNT, "to_account": _ACCOUNT, "amount": _AMOUNT},
                 ["from_account", "to_account", "amount"]),
        ),
        "authenticate": (
            "Verify the caller's identity for a tool that returned AUTH_LEVEL_INSUFFICIENT. "
            "Call it straight away, without speaking first, with the name of that tool. A "
            "scripted flow asks the caller the questions and the result comes back here.",
            _obj({"tool": {"type": "string", "enum": list(PROTECTED_TOOLS)}}, ["tool"]),
        ),
        "escalate": (
            "Transfer the caller to a human agent. Call it without speaking first. The "
            "summary is for the human agent: what the caller wants, what was done, what is "
            "unresolved. Never include account numbers or verification answers.",
            _obj({"reason": _reason_schema(reasons, reason_descriptions),
                  "summary": {"type": "string", "minLength": 1, "maxLength": SUMMARY_MAX_LEN,
                              "description": f"Summary for the human agent, 1 to {SUMMARY_MAX_LEN} characters."}},
                 ["reason", "summary"]),
        ),
        END_CALL: (
            "End the call after the caller has said they are done (for example \"that's all\", "
            "\"no, thanks, goodbye\") and nothing is in progress. Call it without speaking first: "
            "the caller hears the bank's goodbye. Don't call it for a \"thanks\" in the middle of "
            "a task.",
            _obj({}, []),
        ),
    }


def tool_schemas(reasons: Collection[str] = DEFAULT_ESCALATION_REASONS,
                 cards: Collection[str] = DEFAULT_CARDS,
                 reason_descriptions: Mapping[str, str] = DEFAULT_REASON_DESCRIPTIONS,
                 ) -> dict[str, dict[str, Any]]:
    """``{tool name: input JSON Schema}`` for every inline tool. ``cards``: the ``card`` slot's
    known options (``make_payment.to``); a mapping also gives their spoken names.
    ``reason_descriptions``: ``{reason: one line}`` for ``escalate.reason``."""
    return {name: schema for name, (_, schema) in _schemas(reasons, cards, reason_descriptions).items()}


def _keep(value: Any) -> bool:
    """Only strings, objects and arrays survive (a bool is an int; ``None`` is dropped too)."""
    return isinstance(value, str | Mapping | list | tuple)


def sanitize_schema(node: Any) -> Any:
    """Copy of a schema with every boolean, number and null removed, recursively.

    CloudFormation turns scalars in a Json property into strings (``false`` becomes
    ``"false"``), and the model API then rejects the schema. Only strings, objects and
    arrays of strings survive: ``type``, ``properties``, ``enum``, ``pattern``,
    ``required``, ``description``. Dropped keywords (``additionalProperties``,
    ``minimum``, ``maximum``, ``minLength``, ``maxLength``) are still enforced by
    :func:`validate_args` against the full schema.
    """
    if isinstance(node, Mapping):
        return {k: sanitize_schema(v) for k, v in node.items() if _keep(v)}
    if isinstance(node, list | tuple):
        return [sanitize_schema(v) for v in node if _keep(v)]
    return node


def inline_tool_specs(reasons: Collection[str] = DEFAULT_ESCALATION_REASONS,
                      cards: Collection[str] = DEFAULT_CARDS,
                      reason_descriptions: Mapping[str, str] = DEFAULT_REASON_DESCRIPTIONS,
                      exclude: Collection[str] = (),
                      ) -> list[dict[str, Any]]:
    """The model-facing spec of each inline tool, ``{type: inline_function, name, config:
    {inlineFunction: {description, inputSchema}}}``, which the Strands loop turns into tools.

    The ``inputSchema`` is the sanitized, model-facing copy (no booleans or numbers).
    ``exclude``: tool names left out of the list (``{END_CALL}`` when the policy switch is off); unknown names
    are ignored.
    """
    return [
        {"type": "inline_function", "name": name,
         "config": {"inlineFunction": {"description": desc,
                                       "inputSchema": sanitize_schema(schema)}}}
        for name, (desc, schema) in _schemas(reasons, cards, reason_descriptions).items()
        if name not in exclude
    ]


INLINE_TOOLS: tuple[str, ...] = tuple(_schemas(DEFAULT_ESCALATION_REASONS))


def auth_level_insufficient(tool: str) -> dict[str, str]:
    """The only thing a tool returns when the level is too low: no levels, factors or data."""
    return {"error": AUTH_LEVEL_INSUFFICIENT, "tool": tool}


def _validate(schema: Mapping[str, Any], value: Any, path: str, errors: list[str]) -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, Mapping):
            errors.append(f"{path}: must be an object")
            return
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}.{req}: required")
        for key, sub in value.items():
            if key in props:
                _validate(props[key], sub, f"{path}.{key}", errors)
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key}: unexpected property")
        return
    if kind == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: must be a string")
            return
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: too long")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            errors.append(f"{path}: invalid format")
    elif kind == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: must be an integer")
            return
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: not one of the allowed values")


def validate_args(tool: str, args: Any, *,
                  reasons: Collection[str] = DEFAULT_ESCALATION_REASONS,
                  cards: Collection[str] = DEFAULT_CARDS) -> list[str]:
    """Errors in ``args`` for ``tool`` (empty list = valid). Messages never echo values.

    Note for ``escalate``: the bridge should not refuse an escalation over these
    errors; :func:`owlbank.task_attributes.build_task_attributes` maps unknown reasons
    and trims the summary server-side.
    """
    schemas = tool_schemas(reasons, cards)
    if tool not in schemas:
        return [f"unknown tool {tool!r}"]
    errors: list[str] = []
    _validate(schemas[tool], args, "$", errors)
    if errors:
        return errors
    if "amount" in args:
        try:
            if Decimal(args["amount"]) <= 0:
                errors.append("$.amount: must be greater than zero")
        except InvalidOperation:  # pragma: no cover - pattern already guarantees a decimal
            errors.append("$.amount: invalid format")
    if tool == "transfer_funds" and args["from_account"] == args["to_account"]:
        errors.append("$.to_account: must differ from from_account")
    if tool == "escalate" and not args["summary"].strip():
        errors.append("$.summary: empty")
    return errors


# --------------------------------------------------------------------------- Gateway tool names


def gateway_tool_full_name(tool: str) -> str:
    """The name the Gateway gives a target's tool: ``<target>___<tool>``."""
    return f"{GATEWAY_TARGET_NAME}___{tool}"


# --------------------------------------------------------------------------- tool outcomes

ToolKind = Literal["inline", "gateway"]
ToolOutcome = Literal["success", "error", "refused", "rejected", "cancelled"]
TOOL_OUTCOMES: tuple[ToolOutcome, ...] = ("success", "error", "refused", "rejected", "cancelled")

#: A business answer from the bank: the call worked, the answer is no.
BUSINESS_REFUSAL_CODES: frozenset[str] = frozenset({"NOT_FOUND", "INSUFFICIENT_FUNDS", AUTH_LEVEL_INSUFFICIENT})
#: The bridge refused the model's call before running it.
BRIDGE_REJECTION_CODES: frozenset[str] = frozenset({
    "INVALID_ARGUMENTS", "UNKNOWN_TOOL", "CALL_ALONE", "ONE_HANDOFF_AT_A_TIME"})
#: Codes that are failures: a dependency failed, or our request was malformed.
FAILURE_CODES: frozenset[str] = frozenset({"UNAVAILABLE", "INVALID_ARGUMENT", "UNKNOWN_OPERATION"})
#: The ``error`` of the result synthesized for an inline call the bridge never answered (the caller
#: hung up or barged in): ``agent_client_strands.CANCELLED``.
CANCELLED_CODE = "cancelled"
#: ``error_code`` values a ``tool_result`` line may carry; anything else is logged as ``other``.
KNOWN_ERROR_CODES: frozenset[str] = (BUSINESS_REFUSAL_CODES | BRIDGE_REJECTION_CODES | FAILURE_CODES
                                     | {CANCELLED_CODE})
OTHER_ERROR_CODE = "other"
_NO_CODE = object()  # the result carries no ``error`` at all


def _is_cancellation(exc: BaseException) -> bool:
    return isinstance(exc, (asyncio.CancelledError, GeneratorExit))


def _answer_code(answer: Any) -> Any:
    """The ``error`` of one Gateway answer (the Level-0 Lambda's returned object): directly, or under
    a ``body`` (a dict or a JSON string, the Lambda-proxy shape). ``_NO_CODE`` when it has none."""
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except ValueError:
            return _NO_CODE
    if not isinstance(answer, Mapping):
        return _NO_CODE
    if "error" in answer:
        return answer["error"]
    if "body" in answer:
        return _answer_code(answer["body"])
    return _NO_CODE


def _gateway_code(result: Mapping[str, Any]) -> Any:
    """The error code a Gateway tool's MCP result carries in its answer: ``structuredContent``, else
    the first JSON text item that has one. The Level-0 Lambda never raises; it answers
    ``{"error": "UNAVAILABLE"|"UNKNOWN_TOOL"|"INVALID_ARGUMENT"}`` inside a successful result.
    Non-JSON text has no code."""
    code = _answer_code(result.get("structuredContent"))
    if code is not _NO_CODE:
        return code
    content = result.get("content")
    for item in content if isinstance(content, list) else ():
        if isinstance(item, Mapping) and isinstance(item.get("text"), str):
            code = _answer_code(item["text"])
            if code is not _NO_CODE:
                return code
    return _NO_CODE


def _result_code(kind: str, result: Mapping[str, Any]) -> Any:
    if kind == "gateway":
        return _gateway_code(result)
    return result.get("error", _NO_CODE)


def tool_outcome(kind: str, result: Any = None, *, exception: BaseException | None = None,
                 cancel_message: str | None = None) -> ToolOutcome:
    """How one inline or Gateway tool call ended, for the ``tool_result`` line. Pure; no tool names, so a new
    tool needs no change here.

    * ``cancelled``: no result because the caller hung up or barged in: ``exception`` is an
      ``asyncio.CancelledError`` (or ``GeneratorExit``), the inline result is the synthesized
      ``{"error": "cancelled"}``, or the Gateway call was cancelled by Strands (``cancel_message``
      on the hook event, or an MCP result with ``cancelled: true``). Never ``error``.
    * ``error``: an actual failure. Any other exception; a result of no known shape; a Gateway result
      with status other than ``success``; an ``error`` code in :data:`FAILURE_CODES` or not listed
      here (fails towards counting). For a Gateway answer every code but a business refusal is an
      error: the Level-0 Lambda's ``UNKNOWN_TOOL`` means our Gateway wiring is broken, not that the
      bridge refused the model.
    * ``refused``: a business answer (:data:`BUSINESS_REFUSAL_CODES`).
    * ``rejected``: the bridge refused an inline call before running it (:data:`BRIDGE_REJECTION_CODES`).
    * ``success``: an inline result with no ``error`` key; a Gateway result with status ``success``
      whose answer carries no error code (non-JSON text included).

    ``result`` is the inline result's content (the dict the model gets) or Strands' Gateway
    ``ToolResult`` (``{"status", "content", "structuredContent"?, ...}``).
    """
    if exception is not None:
        return "cancelled" if _is_cancellation(exception) else "error"
    if cancel_message is not None:
        return "cancelled"
    if not isinstance(result, Mapping):
        return "error"
    if result.get("cancelled") is True:
        return "cancelled"
    if kind == "gateway" and result.get("status") != "success":
        return "error"
    code = _result_code(kind, result)
    if code is _NO_CODE:
        return "success"
    if not isinstance(code, str):
        return "error"
    if code in BUSINESS_REFUSAL_CODES:
        return "refused"
    if kind == "gateway":
        return "error"
    if code == CANCELLED_CODE:
        return "cancelled"
    if code in BRIDGE_REJECTION_CODES:
        return "rejected"
    return "error"


def tool_error_code(kind: str, result: Any) -> str | None:
    """The ``error_code`` of a ``tool_result`` line: the result's error code when it is one of
    :data:`KNOWN_ERROR_CODES`, ``other`` for any other value, ``None`` when there is none. Never
    the result text."""
    if not isinstance(result, Mapping):
        return None
    code = _result_code(kind, result)
    if code is _NO_CODE:
        return None
    return code if isinstance(code, str) and code in KNOWN_ERROR_CODES else OTHER_ERROR_CODE
