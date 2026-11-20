"""One-time nonces for the ConversationRelay WebSocket.

The relay URL is static and Twilio signs only the URL, so one captured ``/ws``
signature plus a live ``CallSid`` would let anyone send ``setup`` for that call.
The front door therefore puts a fresh nonce in the relay URL (``?n=...``, part of
the signed URL) on every ``<Connect><ConversationRelay>`` and stores only its
SHA-256 in call state (``CallState.ws_nonce_hash``). The bridge checks it on
``setup`` in constant time and consumes it, so each nonce opens one socket.

Hashes are base64url (no padding): server-generated, never caller input.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

QUERY_PARAM = "n"
NONCE_BYTES = 24
MAX_NONCE_LEN = 128


def nonce_hash(nonce: str) -> str:
    digest = hashlib.sha256(nonce.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def new_nonce() -> tuple[str, str]:
    """``(nonce, hash)``: the nonce goes in the URL, only the hash is stored."""
    nonce = secrets.token_urlsafe(NONCE_BYTES)
    return nonce, nonce_hash(nonce)


def matches(stored_hash: str | None, nonce: str | None) -> bool:
    """Constant-time check of ``nonce`` against the stored hash. Missing either: ``False``."""
    if not stored_hash or not isinstance(nonce, str) or not nonce or len(nonce) > MAX_NONCE_LEN:
        return False
    return hmac.compare_digest(stored_hash.encode("ascii"), nonce_hash(nonce).encode("ascii"))


def with_nonce(url: str, nonce: str) -> str:
    """``url`` with ``n=<nonce>`` set in its query (replacing any existing ``n``)."""
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != QUERY_PARAM]
    query.append((QUERY_PARAM, nonce))
    return urlunsplit(parts._replace(query=urlencode(query)))
