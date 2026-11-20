"""Cold-start helpers shared by the Lambda handlers.

Two things, both without changing what a handler does:

* :func:`build_at_init` runs a handler's dependency wiring while the Lambda runtime is still in
  its INIT phase (a provisioned-concurrency environment does that work before any call arrives,
  and an on-demand INIT gets burst CPU), instead of inside the first request. It runs only inside
  Lambda (``AWS_LAMBDA_INITIALIZATION_TYPE`` is set), never in tests or scripts, and a failure
  is swallowed: the first request builds again and reports the real error. The wiring it runs
  creates clients, imports libraries and does the first policy and catalog table read (as the
  first request did before); it reads no secret.
* :class:`LazyClient` defers creating a boto3 client until the first call on it, so a route
  that never needs a client (the KMS, Bedrock and Comprehend clients are not used by
  ``/voice/incoming``) doesn't pay for building it.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Mapping
from typing import Any

from owlbank.obs import log_caught

log = logging.getLogger(__name__)

INIT_TYPE_ENV = "AWS_LAMBDA_INITIALIZATION_TYPE"
PROVISIONED = "provisioned-concurrency"


def init_type(env: Mapping[str, str] | None = None) -> str | None:
    """Lambda's initialization type (``on-demand``, ``provisioned-concurrency``, ...) or ``None``
    outside Lambda."""
    value = (os.environ if env is None else env).get(INIT_TYPE_ENV)
    return value or None


def provisioned(env: Mapping[str, str] | None = None) -> bool:
    """True in an environment Lambda initialised ahead of any request (provisioned concurrency)."""
    return init_type(env) == PROVISIONED


def build_at_init[T](build: Callable[[], T], env: Mapping[str, str] | None = None) -> T | None:
    """``build()`` during Lambda's INIT phase; ``None`` outside Lambda or if it fails."""
    if init_type(env) is None:
        return None
    try:
        return build()
    except Exception as exc:  # noqa: BLE001 - the first request retries and surfaces the real error
        log_caught(log, "init_build_failed", exc, decision="build_on_first_request")
        return None


class LazyClient:
    """A boto3 client created by ``factory`` on first attribute access, then reused."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._client: Any | None = None
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # not a client operation (and keeps copy/pickle probes quiet)
            raise AttributeError(name)
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._factory()
        return getattr(self._client, name)
