"""``X-Twilio-Signature`` validation.

Built on ``twilio.request_validator.RequestValidator`` (its ``compute_signature``
and ``compute_hash`` primitives), with a stricter wrapper:

* **Form requests** (``application/x-www-form-urlencoded``): Twilio signs the
  full URL followed by every POST parameter, sorted by name, as ``name + value``.
  Pass the decoded POST parameters as ``params``.
* **JSON requests:** Twilio adds ``bodySHA256=<hex sha256 of body>`` to the
  query string and signs the URL alone. Pass the raw body text as ``body``;
  ``params`` must then be empty. A URL with ``bodySHA256`` but no ``body``, or a
  ``body`` without ``bodySHA256``, fails.

Pitfalls (read before wiring a handler):

* **Validate the public URL Twilio requested,** not the internal one. Behind API
  Gateway or an ALB, the Lambda/container sees a different scheme, host or
  path. Use :func:`reconstruct_url`.
* **Port handling.** Twilio is inconsistent about whether the signed URL
  carries the port. Like the SDK, :func:`validate` accepts a signature over the
  URL either without a port or with the explicit port (the scheme default,
  443/80, when the URL has none). Don't add a port yourself: pass the URL as
  Twilio would have seen it.
* **Keep the query string, byte for byte.** It is part of the signed URL. Use
  the raw query string as received (API Gateway HTTP API ``rawQueryString``).
  Re-encoding it from a parsed dict (API Gateway REST ``queryStringParameters``)
  can change order or percent-encoding and break the signature.
* **Don't strip the path prefix.** If a custom-domain base path or stage is in
  the URL Twilio was given, it must be in the URL you validate.
* **User info.** Basic credentials in the webhook URL arrive in the
  ``Authorization`` header (after a 401 challenge), not in the request line, so
  the URL passed to :func:`validate` never has ``user:pass@``; URLs with ``@``
  in the authority are rejected. Twilio may nevertheless have signed the URL
  *with* the credentials: pass the credentials the caller already proved
  (``credentials=(user, password)``) and :func:`signed_url_form` also tries the
  URL with ``user:password@`` inserted. Only those two forms are accepted, each
  with and without the port; both are HMAC-SHA1 over the Auth Token, so neither
  is weaker. The returned form name (never the URL) says which one matched.
* **No normalization.** Unlike ``RequestValidator.validate`` (which round-trips
  the URL through ``urlparse``), this wrapper compares the URL string exactly:
  only lower-case ``http``/``https`` schemes are accepted, and a trailing ``?``
  or ``#`` is significant. Any single-character change fails.
* **Auth Token only.** The signing key is the account Auth Token; never use it
  for REST calls. Fetch it with ``owlbank.secrets.get_secret``.
"""

from __future__ import annotations

import hmac
import logging
import re
from collections.abc import Mapping
from urllib.parse import parse_qs, urlsplit

from twilio.request_validator import RequestValidator

from owlbank.obs import emit

log = logging.getLogger(__name__)

__all__ = ["validate", "signed_url_form", "with_credentials", "reconstruct_url", "DEFAULT_PORTS",
           "FORM_PLAIN", "FORM_CREDENTIALS"]

#: Names of the signed-URL forms :func:`signed_url_form` reports.
FORM_PLAIN = "plain"  # the URL without user info
FORM_CREDENTIALS = "credentials"  # the URL with ``user:password@`` inserted

#: Default port per scheme, used for the "with port" signature variant.
DEFAULT_PORTS: Mapping[str, int] = {"https": 443, "http": 80}

_URL_RE = re.compile(r"^(https?)://([^/?#]*)(.*)\Z", re.DOTALL)


def _port_variants(url: str) -> tuple[str, str] | None:
    """Return (url without port, url with explicit port), or None if malformed."""
    m = _URL_RE.match(url)
    if m is None:
        return None
    scheme, netloc, rest = m.groups()
    if not netloc or "@" in netloc:
        return None
    host, port = netloc, ""
    if netloc.startswith("["):  # IPv6 literal: [::1] or [::1]:8443
        close = netloc.find("]")
        if close == -1:
            return None
        host, tail = netloc[: close + 1], netloc[close + 1 :]
        if tail:
            if not tail.startswith(":"):
                return None
            port = tail[1:]
            if not port:
                return None
    elif ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        if not host:
            return None
    if port and not (port.isascii() and port.isdigit()):
        return None
    if ":" in host and not host.startswith("["):
        return None
    if port == "" and netloc.endswith(":"):
        return None
    explicit = port or str(DEFAULT_PORTS[scheme])
    return f"{scheme}://{host}{rest}", f"{scheme}://{host}:{explicit}{rest}"


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def with_credentials(url: str, user: str, password: str) -> str | None:
    """``url`` with ``user:password@`` inserted before the host, or ``None`` if it can't be.

    ``url`` must be a plain ``http(s)`` URL without user info. Credentials that are empty or hold
    a character that would change how the URL parses (``/?#@``, whitespace, or ``:`` in the user)
    give ``None``. Twilio warns against such characters in webhook credentials.
    """
    if not user or not password:
        return None
    if any(c in "/?#@ \t\r\n" for c in user + password) or ":" in user:
        return None
    m = _URL_RE.match(url)
    if m is None or "@" in m.group(2):
        return None
    scheme, netloc, rest = m.groups()
    return f"{scheme}://{user}:{password}@{netloc}{rest}"


def signed_url_form(
    url: str,
    params: Mapping[str, str] | None,
    signature: str,
    auth_token: str,
    *,
    body: str | None = None,
    credentials: tuple[str, str] | None = None,
) -> str | None:
    """Which signed-URL form ``signature`` matches: :data:`FORM_PLAIN`, :data:`FORM_CREDENTIALS` or ``None``.

    ``url`` is the public URL Twilio requested, without user info (see :func:`reconstruct_url`).
    ``credentials`` are the Basic credentials the caller has already proved; when given, the same
    URL with ``user:password@`` inserted is also tried. Every candidate (each form, with and
    without the port) is evaluated: no early exit on the timing path. ``params``/``body`` as in
    :func:`validate`. Never raises: malformed input returns ``None``.
    """
    try:
        if not (isinstance(url, str) and isinstance(signature, str) and isinstance(auth_token, str)):
            return None
        if not url or not signature or not auth_token:
            return None
        variants = _port_variants(url)
        if variants is None:
            return None
        validator = RequestValidator(auth_token)
        query = parse_qs(urlsplit(url).query)
        if body is not None:
            if params or "bodySHA256" not in query:
                return None
            if not _eq(validator.compute_hash(body), query["bodySHA256"][0]):
                return None
            signed_params: Mapping[str, str] = {}
        else:
            if "bodySHA256" in query:
                return None
            signed_params = params if params is not None else {}
        candidates: list[tuple[str, str]] = [(FORM_PLAIN, v) for v in variants]
        if credentials is not None:
            for v in variants:
                with_creds = with_credentials(v, *credentials)
                if with_creds is not None:
                    candidates.append((FORM_CREDENTIALS, with_creds))
        matched: str | None = None
        for form, candidate in candidates:  # evaluate all: no early exit on the timing path
            if _eq(validator.compute_signature(candidate, signed_params), signature) and matched is None:
                matched = form
        return matched
    except (TypeError, ValueError, UnicodeError, KeyError, AttributeError) as exc:
        emit(log, logging.DEBUG, "signature_check_error", operation="validate",
             error_class=type(exc).__name__, decision="reject")
        return None


def validate(
    url: str,
    params: Mapping[str, str] | None,
    signature: str,
    auth_token: str,
    *,
    body: str | None = None,
    credentials: tuple[str, str] | None = None,
) -> bool:
    """True only if ``signature`` is Twilio's signature for this request.

    ``url`` is the public URL Twilio requested (see :func:`reconstruct_url`).
    ``params`` are the decoded form parameters (``None`` or empty for JSON).
    ``body`` is the raw JSON body, for URLs carrying ``bodySHA256``.
    ``credentials``: see :func:`signed_url_form` (the URL with ``user:password@`` also counts).
    Never raises: malformed input returns ``False``.
    """
    return signed_url_form(url, params, signature, auth_token, body=body,
                           credentials=credentials) is not None


def reconstruct_url(
    scheme: str,
    host: str,
    path: str,
    query: str | None,
    *,
    forwarded_proto: str | None = None,
    forwarded_host: str | None = None,
    forwarded_port: str | int | None = None,
) -> str:
    """Rebuild the public URL Twilio signed, from what the server received.

    * ``scheme``/``host``/``path``/``query``: as the server saw them. ``host`` is
      the ``Host`` header and may include a port. ``query`` is the *raw* query
      string, without ``?``, or ``None``/empty for none.
    * ``forwarded_proto``/``forwarded_host``: ``X-Forwarded-Proto`` /
      ``X-Forwarded-Host`` set by the proxy (ALB, API Gateway); they win over
      ``scheme``/``host``. A comma-separated list uses its first (client-side)
      entry. Only trust these headers when a proxy you control sets them.
    * ``forwarded_port``: ``X-Forwarded-Port``. Added to the host only if the host
      has no port and it is not the scheme's default (443 for https, 80 for http).

    Raises ``ValueError`` for an unsupported scheme or an invalid host.
    """
    proto = _first(forwarded_proto) or scheme
    proto = proto.strip().lower()
    if proto not in DEFAULT_PORTS:
        raise ValueError(f"unsupported scheme {proto!r}")
    authority = (_first(forwarded_host) or host or "").strip()
    if not authority or any(c in authority for c in "/?#@ \t\r\n"):
        raise ValueError("invalid host")
    has_port = authority.rsplit("]", 1)[-1].count(":") > 0
    if forwarded_port is not None and not has_port:
        port = str(forwarded_port).strip()
        if port:
            if not (port.isascii() and port.isdigit()):
                raise ValueError("invalid forwarded port")
            if int(port) != DEFAULT_PORTS[proto]:
                authority = f"{authority}:{port}"
    path = path or "/"
    if not path.startswith("/"):
        path = "/" + path
    url = f"{proto}://{authority}{path}"
    if query:
        url += "?" + (query[1:] if query.startswith("?") else query)
    return url


def _first(header: str | None) -> str | None:
    if header is None:
        return None
    first = header.split(",", 1)[0].strip()
    return first or None
