"""Salted SHA-256 hashes for verification factors.

Stored form: ``<salt>$<hex digest>`` where ``salt`` is 16 random bytes as hex
(``secrets.token_hex``) and the digest is ``sha256(salt + "$" + value)``.

Usage::

    uv run python scripts/hash_tool.py 94105          # prints salt$hex
    uv run python scripts/hash_tool.py --verify 94105 'salt$hex'

This is the mock back end's pattern, so the blueprint stores no plaintext
factors. It is *not* a production recommendation: a 4- or 5-digit factor is
cheap to brute-force from a leaked salted hash. A real bank would verify
against its core system, or use a keyed HMAC with a secret pepper held in KMS
or Secrets Manager. See ``docs/production-guide.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import secrets
import sys

SEPARATOR = "$"
SALT_BYTES = 16


def _digest(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}{SEPARATOR}{value}".encode()).hexdigest()


def hash_factor(value: str, *, salt: str | None = None) -> str:
    """Return ``salt$hex`` for ``value``. A fresh random salt is used unless given."""
    if not value:
        raise ValueError("factor value must be non-empty")
    salt = secrets.token_hex(SALT_BYTES) if salt is None else salt
    if not salt or SEPARATOR in salt:
        raise ValueError("salt must be non-empty and must not contain '$'")
    return f"{salt}{SEPARATOR}{_digest(salt, value)}"


def verify(value: str, stored: str) -> bool:
    """True if ``value`` matches ``stored`` (``salt$hex``). Constant-time comparison."""
    salt, sep, expected = stored.partition(SEPARATOR)
    if not sep or not salt or not expected:
        return False
    return hmac.compare_digest(_digest(salt, value), expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("value", help="factor value to hash (fake test data only)")
    parser.add_argument("--verify", metavar="STORED", help="check VALUE against a stored salt$hex")
    args = parser.parse_args(argv)
    if args.verify is not None:
        ok = verify(args.value, args.verify)
        print("match" if ok else "no match")
        return 0 if ok else 1
    print(hash_factor(args.value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
