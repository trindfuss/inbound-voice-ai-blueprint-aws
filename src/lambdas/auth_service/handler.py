"""Lambda entry point for the auth service (thin: wiring only; logic in ``service``).

Invoked by Lambda direct invoke (``lambda:InvokeFunction``), from the front door
only (IAM; the function has no API, URL or other trigger). The payload is JSON:

``{"op": "verify", "call_sid": "CA…", "ani": "+1…", "factors": {"zip": "94105"}, "target_level": 1,
"stir_verstat": "TN-Validation-Passed-A"}`` (``stir_verstat`` optional: absent or not in the
policy's ``ani_trust.accepted_verstat`` means an untrusted caller ID). ``verify``,
``precheck`` and ``send_otp`` also take an optional ``"identity_source"``: ``"ani"`` (the default)
or ``"keyed_phone"`` (the ``ani`` is a number the caller keyed in: untrusted, no ``ani_match``,
whatever ``on_untrusted`` says); any other value is ``bad_request``
    → ``{"result": "verified", "level": 1, "customer_id": "C-1001", "token": "…",
    "factors_verified": ["ani_match", "zip"], "step_up": false}``, or
    ``{"result": "not_verified", "level": n, "reason": "mismatch"|"ambiguous"|"no_candidates",
    "attempts_remaining": k}``, ``{"result": "rate_limited"|"invalid_level"|"unavailable", "level": n}``,
    ``{"result": "bad_request", "reason": "…", "missing": [...]?}``.

``{"result": "escalate", "level": n, "reason": "verification_risk"}`` (untrusted caller ID:
untrusted-attempt cap reached, or a one-time code needed before login).

``{"op": "precheck", "call_sid": "CA…", "ani": "+1…", "target_level": 1, "stir_verstat": …}``
    → ``{"result": "ok"}`` or ``{"result": "escalate", "reason": "verification_risk"}``
    (or ``bad_request`` / ``unavailable``): the same gate, before any factor is asked.

``{"op": "send_otp", "call_sid": "CA…", "ani": "+1…", "language": "en-US", "stir_verstat": …,
"record": true}`` (``record`` optional)
    → ``{"result": "sent"|"not_verified"|"rate_limited"|"unavailable"|"bad_request"}``, or
    ``{"result": "escalate", "reason": "otp_unavailable"}`` when no code can go to the
    calling number (withheld, malformed or not on file for the customer) or Verify can't
    deliver. ``language`` is the Verify ``Locale`` source (``en-US``→``en``,
    ``es-US``→``es``). A code is sent for an untrusted caller ID too; that case logs
    the ``otp_sent_untrusted_ani`` security line (``owlbank.security``), which is alarmed on.
    With ``"record": true`` (the front door's asynchronous ``Event`` invoke) the outcome is
    also written to the call state as ``otp_send = {"status": "sent"|"escalate"|"error", "at"}``.

``{"op": "prepare_welcome", "call_sid": "CA…", "customer_id": "C-1001", "language": "en-US"}``
    (the front door's asynchronous ``Event`` invoke when the last factor is
    asked) → ``{"result": "ok"|"unavailable", "reason": "matched"|…}`` or ``bad_request``. Writes
    ``welcome_prepared`` (a sealed offer ID or none, the customer, the language, the reason) to the
    call state; nobody reads the answer.

Cold start: outside Lambda, importing this module reads no secret and makes no AWS or Twilio
call. Inside Lambda, the dependencies (including the first policy read) are built once during the
INIT phase (``coldstart.build_at_init``; no secret is read),
and in a provisioned-concurrency environment the Twilio SDK is imported there too, so the first
``send_otp`` or ``verify`` with a code doesn't pay for it. The Twilio client itself (and the API
key it needs) is still created on first use.

Environment: the ``owlbank.config`` variables; ``OWLBANK_VERIFY_SERVICE_SID``
(``settings.verify_service_sid``, a ``VA…`` Verify service) enables the OTP factor.
The OTP goes only to the calling ANI, and only when it matches the verified customer
(directly or via an ``alias_of`` record), or on a keyed-in number's call to that keyed number
on the same terms, after ZIP and SSN-4 (the same compliance review applies).
Never to any other number. Pass the same ``ani`` to ``send_otp`` and to the ``verify`` call that
carries the code. Without it, OTP verification
returns ``unavailable``.

``prepare_welcome`` reads ``OWLBANK_MEMORY_STORE_ID`` (unset: every
welcome is ``no_profile`` and Memory is never called), ``OWLBANK_MEMORY_TRAIT_GROUPS`` and
``OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS`` (unset: the bridge's defaults; invalid: logged, and the
welcome is off or the default window is used, so a bad value never stops verification), and
``OWLBANK_TOKEN_KMS_KEY`` (``settings.token_kms_key``) to seal the record. Memory reads use the same
Twilio client as Verify (the ``twilio_api_key`` secret), built once on first use.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any

from lambdas.auth_service import service
from lambdas.auth_service.verify_client import TwilioVerifyClient, preload_twilio_sdk, twilio_client_factory
from owlbank import coldstart, obs, web_activity
from owlbank.config import Settings
from owlbank.policy import Policy
from owlbank.store import DynamoStore
from owlbank.token_seal import KmsSealer

log = logging.getLogger(__name__)

_deps: service.AuthDeps | None = None


def _shared(factory: Callable[[], Any]) -> Callable[[], Any]:
    """``factory`` called once, on first use, then its result reused (Verify and Memory share one
    Twilio client, so the API-key secret is read once)."""
    lock = threading.Lock()
    box: list[Any] = []

    def get() -> Any:
        if not box:
            with lock:
                if not box:
                    box.append(factory())
        return box[0]

    return get


def welcome_settings_from_env(env: Mapping[str, str]) -> dict[str, Any]:
    """``AuthDeps`` keyword arguments for ``prepare_welcome``: the Memory store ID, the allowed trait
    groups and the web-activity window, as the bridge reads them (``tac_bridge.app``). Never raises:
    an invalid group list allows no group (the welcome is off, ``no_web_activity``) and an invalid
    window is the default, each with one ERROR line naming the variable, never its value."""
    out: dict[str, Any] = {"memory_store_id": (env.get("OWLBANK_MEMORY_STORE_ID") or "").strip() or None}
    groups_raw = (env.get("OWLBANK_MEMORY_TRAIT_GROUPS") or "").strip()
    if groups_raw:
        names = tuple(part.strip() for part in groups_raw.split(","))
        if (1 <= len(names) <= web_activity.MAX_TRAIT_GROUPS and len(set(names)) == len(names)
                and all(web_activity.TRAIT_GROUP_NAME_RE.fullmatch(n) for n in names)):
            out["memory_trait_groups"] = names
        else:
            obs.emit(log, logging.ERROR, "setting_invalid", setting="OWLBANK_MEMORY_TRAIT_GROUPS",
                     decision="welcome_off")
            out["memory_trait_groups"] = ()
    hours_raw = (env.get("OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS") or "").strip()
    if hours_raw:
        low, high = web_activity.MAX_AGE_HOURS_RANGE
        if hours_raw.isascii() and hours_raw.isdigit() and low <= int(hours_raw) <= high:
            out["web_activity_max_age_hours"] = int(hours_raw)
        else:
            obs.emit(log, logging.ERROR, "setting_invalid", setting="OWLBANK_WEB_ACTIVITY_MAX_AGE_HOURS",
                     decision="use_default")
    return out


def build_deps(env: dict[str, str] | None = None) -> service.AuthDeps:
    """Production dependencies (no AWS or Twilio call until first use, except the policy load)."""
    env = dict(os.environ) if env is None else env
    settings = Settings.from_env(env)
    store = DynamoStore.from_settings(settings)
    verify_client = None
    twilio = _shared(twilio_client_factory(settings.secrets.twilio_api_key)) \
        if settings.secrets.twilio_api_key else None
    sid = settings.verify_service_sid
    if sid and twilio is not None:
        verify_client = TwilioVerifyClient(sid, twilio)
    sealer = None
    if settings.token_kms_key:
        import boto3  # noqa: PLC0415 - only when the key is configured
        from botocore.config import Config  # noqa: PLC0415

        quick = Config(connect_timeout=1, read_timeout=2, retries={"total_max_attempts": 2})
        sealer = KmsSealer(coldstart.LazyClient(lambda: boto3.client("kms", config=quick)),
                           settings.token_kms_key)
    return service.AuthDeps(
        store=store,
        policy=Policy.from_store(store, ttl_s=settings.cache_ttl_s),
        verify_client=verify_client,
        token_max_ttl_s=settings.token_max_ttl_s,
        record_ttl_s=settings.record_ttl_s,
        sealer=sealer,
        memory_client=twilio,
        **welcome_settings_from_env(env),
    )


@obs.instrumented("auth_service")
def handler(event: Any, context: Any = None) -> dict[str, Any]:
    global _deps
    if _deps is None:
        _deps = build_deps()
    return service.handle_event(_deps, event)


def _build_at_init() -> service.AuthDeps:
    deps = build_deps()
    if coldstart.provisioned():
        preload_twilio_sdk()  # imports only: no secret read, no network
    return deps


# Inside Lambda only (a no-op in tests and scripts): wire up now, in the INIT phase.
_deps = coldstart.build_at_init(_build_at_init)
