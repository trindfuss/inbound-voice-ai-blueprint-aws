"""Seal the call token for hand-over between the front door and the bridge.

The auth service returns the raw token to the front door, which must pass it to
the bridge (a different process) through the call state. The raw token is never
stored: the front door seals it with KMS, using the ``CallSid`` as encryption
context, and the bridge unseals it in memory just before invoking a tool. A
sealed token copied to another call fails to unseal (wrong context).

Only the front door (encrypt) and the bridge and front door (decrypt) are
granted the key. Tokens and sealed values are never logged.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from typing import Any, Protocol

from owlbank.obs import error_code_of, log_caught, timed_call

__all__ = ["Sealer", "KmsSealer", "FakeSealer", "UnsealError"]

_CONTEXT_KEY = "call_sid"
log = logging.getLogger(__name__)


class UnsealError(RuntimeError):
    """The sealed value could not be unsealed for this call.

    ``transient`` is true when a retry may succeed (KMS throttled, timed out or unreachable) and
    false when it can't (bad ciphertext or encoding, the wrong call's context, a disabled or wrong
    key): :func:`owlbank.sensitive.restore` retries only a transient failure."""

    def __init__(self, message: str = "token could not be unsealed for this call", *,
                 transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


#: KMS error codes a retry may clear (throttling, KMS or dependency outages). Any other KMS error
#: code (``InvalidCiphertextException`` for a bad blob or the wrong call's context,
#: ``IncorrectKeyException``, ``DisabledException``, ``AccessDeniedException``...) is not transient.
TRANSIENT_KMS_CODES: frozenset[str] = frozenset({
    "ThrottlingException", "KMSInternalException", "DependencyTimeoutException",
    "KeyUnavailableException", "LimitExceededException", "InternalFailure",
    "ServiceUnavailableException", "RequestTimeout", "RequestTimeoutException"})


def _transient(exc: BaseException) -> bool:
    """Whether a KMS ``Decrypt`` failure may clear on a retry: an AWS error with a code in
    :data:`TRANSIENT_KMS_CODES`, or a client-side failure that is not about the input (botocore's
    connect and read timeouts, an unreachable endpoint). Bad base64 (``binascii.Error``, a
    ``ValueError``) and every other AWS error code are not."""
    code = error_code_of(exc)
    if code is not None:
        return code in TRANSIENT_KMS_CODES
    return not isinstance(exc, (ValueError, TypeError))


class Sealer(Protocol):
    def seal(self, token: str, *, call_sid: str) -> str: ...
    def unseal(self, sealed: str, *, call_sid: str) -> str: ...


class KmsSealer:
    """KMS ``Encrypt``/``Decrypt`` with ``{"call_sid": CallSid}`` as encryption context."""

    def __init__(self, kms_client: Any, key_id: str) -> None:
        self._kms = kms_client
        self._key_id = key_id

    def seal(self, token: str, *, call_sid: str) -> str:
        with timed_call("kms", "encrypt"):
            resp = self._kms.encrypt(KeyId=self._key_id, Plaintext=token.encode("utf-8"),
                                     EncryptionContext={_CONTEXT_KEY: call_sid})
        return base64.b64encode(resp["CiphertextBlob"]).decode("ascii")

    def unseal(self, sealed: str, *, call_sid: str) -> str:
        try:
            with timed_call("kms", "decrypt"):
                resp = self._kms.decrypt(CiphertextBlob=base64.b64decode(sealed),
                                         EncryptionContext={_CONTEXT_KEY: call_sid},
                                         KeyId=self._key_id)
        except Exception as exc:  # InvalidCiphertextException, bad base64, throttling, ...
            transient = _transient(exc)
            log_caught(log, "unseal_failed", exc, operation="unseal", decision="raise_unseal_error",
                       transient=transient)
            raise UnsealError(transient=transient) from None
        return resp["Plaintext"].decode("utf-8")


class FakeSealer:
    """Offline sealer for tests: XOR with a per-call keystream plus an HMAC tag.

    Not for production. It keeps the same contract as :class:`KmsSealer`: the
    sealed value never contains the token, and unsealing under another
    ``call_sid`` fails.
    """

    def __init__(self, key: bytes = b"owlbank-test-seal-key") -> None:
        self._key = key

    def _stream(self, call_sid: str, n: int) -> bytes:
        out, counter = b"", 0
        while len(out) < n:
            out += hashlib.sha256(self._key + call_sid.encode() + counter.to_bytes(4, "big")).digest()
            counter += 1
        return out[:n]

    def _tag(self, call_sid: str, body: bytes) -> bytes:
        return hmac.new(self._key, call_sid.encode() + b"|" + body, hashlib.sha256).digest()

    def seal(self, token: str, *, call_sid: str) -> str:
        raw = token.encode("utf-8")
        body = bytes(a ^ b for a, b in zip(raw, self._stream(call_sid, len(raw)), strict=True))
        return base64.urlsafe_b64encode(self._tag(call_sid, body) + body).decode("ascii")

    def unseal(self, sealed: str, *, call_sid: str) -> str:
        try:
            blob = base64.urlsafe_b64decode(sealed.encode("ascii"))
        except Exception as exc:
            log_caught(log, "unseal_failed", exc, level=logging.DEBUG, operation="fake_unseal",
                       decision="raise_unseal_error")
            raise UnsealError("token could not be unsealed for this call") from None
        tag, body = blob[:32], blob[32:]
        if len(tag) < 32 or not hmac.compare_digest(tag, self._tag(call_sid, body)):
            raise UnsealError("token could not be unsealed for this call")
        return bytes(a ^ b for a, b in zip(body, self._stream(call_sid, len(body)), strict=True)).decode("utf-8")
