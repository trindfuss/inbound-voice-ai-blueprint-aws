"""A reference verifier for SigV4 query-string signatures, written from the SigV4 spec.

It doesn't use botocore, so the signer's tests check botocore's output against an independent
reading of the algorithm ("Create a signed AWS API request" and "Authenticating requests: using
query parameters" in the AWS docs):

1. Canonical request: ``GET``, the canonical URI (each segment of the path as sent URI-encoded
   once more, which is the spec's "encode twice" for every service but S3), the canonical query
   (every parameter but ``X-Amz-Signature``, names and values URI-encoded with only
   ``A-Za-z0-9-_.~`` left as is, sorted by name then value), the canonical headers (the signed
   ones, lower-case names, trimmed values), the signed header list, and the hex SHA-256 of the
   (empty) payload.
2. String to sign: ``AWS4-HMAC-SHA256``, ``X-Amz-Date``, the credential scope, and the hex SHA-256
   of the canonical request, one per line.
3. Signing key: HMAC-SHA256 chained over ``"AWS4" + secret``, date, region, service,
   ``aws4_request``; the signature is the lower-case hex HMAC of the string to sign.

The verifier is deliberately strict: only ``host`` may be signed (a URL alone carries no other
header value), duplicate parameters, dot segments and malformed escapes are rejected.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import unquote, urlsplit

ALGORITHM = "AWS4-HMAC-SHA256"
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~")
_ESCAPE_OK = re.compile(r"%(?![0-9A-Fa-f]{2})")
_AMZ_DATE = re.compile(r"^\d{8}T\d{6}Z\Z")
_HEX64 = re.compile(r"^[0-9a-f]{64}\Z")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def uri_encode(value: str) -> str:
    """Percent-encode every byte of ``value`` (UTF-8) except the unreserved characters."""
    out = []
    for byte in value.encode("utf-8"):
        ch = chr(byte)
        out.append(ch if ch in _UNRESERVED else f"%{byte:02X}")
    return "".join(out)


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def signing_key(secret_key: str, date: str, region: str, service: str) -> bytes:
    k = _hmac(("AWS4" + secret_key).encode("utf-8"), date)
    k = _hmac(k, region)
    k = _hmac(k, service)
    return _hmac(k, "aws4_request")


def parse_query(raw: str) -> list[tuple[str, str]] | None:
    """Decoded ``(name, value)`` pairs, or ``None`` for a malformed query."""
    if not raw:
        return []
    pairs = []
    for part in raw.split("&"):
        if not part or "=" not in part or _ESCAPE_OK.search(part):
            return None
        name, _, value = part.partition("=")
        pairs.append((unquote(name, errors="strict"), unquote(value, errors="strict")))
    return pairs


def canonical_uri(raw_path: str) -> str | None:
    if not raw_path:
        return "/"
    segments = raw_path.split("/")
    if any(s in (".", "..") for s in segments) or _ESCAPE_OK.search(raw_path):
        return None
    return "/".join(uri_encode(s) for s in segments)


def verify(url: str, *, secret_key: str, region: str, service: str,
           now: datetime | None = None) -> bool:
    """True iff ``url`` carries a valid SigV4 query signature for ``service`` in ``region``.

    With ``now``, the URL must also be inside its validity window."""
    try:
        return _verify(url, secret_key=secret_key, region=region, service=service, now=now)
    except (UnicodeError, ValueError):
        return False


def _verify(url: str, *, secret_key: str, region: str, service: str, now: datetime | None) -> bool:
    parts = urlsplit(url)
    if parts.fragment or not parts.netloc:
        return False
    pairs = parse_query(parts.query)
    if pairs is None:
        return False
    names = [n for n, _ in pairs]
    if len(names) != len(set(names)):
        return False
    params = dict(pairs)
    signature = params.pop("X-Amz-Signature", None)
    if params.get("X-Amz-Algorithm") != ALGORITHM or signature is None or not _HEX64.match(signature):
        return False
    amz_date = params.get("X-Amz-Date", "")
    if not _AMZ_DATE.match(amz_date):
        return False
    credential = params.get("X-Amz-Credential", "").split("/")
    if len(credential) != 5 or not credential[0]:
        return False
    _akid, date, cred_region, cred_service, terminator = credential
    if (date != amz_date[:8] or cred_region != region or cred_service != service
            or terminator != "aws4_request"):
        return False
    expires = params.get("X-Amz-Expires", "")
    if not expires.isdigit() or not 1 <= int(expires) <= 604800 or str(int(expires)) != expires:
        return False
    if params.get("X-Amz-SignedHeaders") != "host":
        return False
    path = canonical_uri(parts.path)
    if path is None:
        return False
    host = parts.netloc.strip()
    query = "&".join(f"{n}={v}" for n, v in sorted((uri_encode(n), uri_encode(v))
                                                    for n, v in params.items()))
    canonical_request = "\n".join(["GET", path, query, f"host:{host}\n", "host", EMPTY_SHA256])
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [ALGORITHM, amz_date, scope, hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()])
    expected = hmac.new(signing_key(secret_key, date, region, service),
                        string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return False
    if now is not None:
        signed_at = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        if not signed_at - timedelta(minutes=5) <= now <= signed_at + timedelta(seconds=int(expires)):
            return False
    return True
