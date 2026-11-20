"""The bridge container's entry point: uvicorn, with TLS on the container port on Fargate.

* **Fargate** (``OWLBANK_HOSTING`` absent, empty or ``fargate``): the server speaks TLS on its port,
  so the hop from the ALB to the task is encrypted. The certificate is self-signed (an EC P-256 key
  made at start-up); the ALB doesn't validate target certificates, so no CA or rotation is needed.
  The key and certificate exist only in memory: on Linux they go into an anonymous ``memfd`` (no
  path on any filesystem) that the SSL context loads through ``/proc/self/fd/N``; elsewhere (local
  runs on macOS) into a ``0600`` temp file. Either way the descriptor or file is gone once the
  context is built (uvicorn's ``ssl_context_factory``), before the app is built and the server listens.
* **AgentCore** (``OWLBANK_HOSTING=agentcore``): plain HTTP. The Runtime contract is HTTP on port
  8080, and the Runtime terminates TLS in front of the container.

Run: ``python -m tac_bridge.serve`` (the Dockerfile's CMD). ``python -m tac_bridge.serve --healthcheck``
probes ``/health`` the same way (HTTPS without verification on Fargate, HTTP on AgentCore) and exits
0 when it answers 200.
"""

from __future__ import annotations

import datetime as dt
import os
import ssl
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager

import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from tac_bridge.hosting import hosting_from_env

APP = "tac_bridge.app:build_app_from_env"
HOST = "0.0.0.0"  # noqa: S104  # nosec B104 (the container port; only the ALB security group reaches it)
PORT = 8080
HEALTH_PATH = "/health"
CERT_COMMON_NAME = "owlbank-bridge"
CERT_DAYS = 365


def tls_enabled(env: Mapping[str, str]) -> bool:
    """TLS on the container port everywhere except AgentCore (whose contract is HTTP)."""
    return hosting_from_env(env) == "fargate"


def self_signed_pem(*, now: dt.datetime | None = None) -> bytes:
    """A new EC P-256 private key and a self-signed certificate for it, as one PEM blob (key first)."""
    now = now or dt.datetime.now(dt.UTC)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CERT_COMMON_NAME)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=CERT_DAYS))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(CERT_COMMON_NAME)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption())
    return key_pem + cert.public_bytes(serialization.Encoding.PEM)


@contextmanager
def pem_path(pem: bytes) -> Iterator[str]:
    """A path that reads as ``pem`` for the duration of the block, then no longer exists."""
    if sys.platform == "linux":
        fd = os.memfd_create("bridge-tls", os.MFD_CLOEXEC)
        try:
            os.write(fd, pem)
            yield f"/proc/self/fd/{fd}"
        finally:
            os.close(fd)
        return
    fd, path = tempfile.mkstemp(prefix="bridge-tls-", suffix=".pem")  # created 0600
    try:
        os.write(fd, pem)
        os.close(fd)
        yield path
    finally:
        os.unlink(path)


def make_server(env: Mapping[str, str] | None = None, *, app: str = APP, port: int = PORT,
                host: str = HOST, factory: bool = True) -> uvicorn.Server:
    """A loaded ``uvicorn.Server``, ready to ``run()``. Same options as the former uvicorn command line.

    WebSocket keepalive is explicit: a ping every 20 s, and a connection with no pong within 20 s is
    closed. This is also uvicorn's default; it is spelled out so an upgrade can't change it silently,
    and because the pings keep the ConversationRelay socket active across CloudFront's and the ALB's
    idle timeouts.
    """
    env = os.environ if env is None else env
    cfg = uvicorn.Config(app, factory=factory, host=host, port=port, server_header=False,
                         access_log=False, ws_ping_interval=20.0, ws_ping_timeout=20.0,
                         ssl_context_factory=_self_signed_context if tls_enabled(env) else None)
    # load() builds the SSL context first, then imports and builds the app.
    cfg.load()
    return uvicorn.Server(cfg)


def server_ssl_context() -> ssl.SSLContext:
    """A TLS 1.2+ server context holding a fresh self-signed key and certificate."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    with pem_path(self_signed_pem()) as path:
        ctx.load_cert_chain(path)
    return ctx


def _self_signed_context(_config: uvicorn.Config, _default: Callable[[], ssl.SSLContext]) -> ssl.SSLContext:
    return server_ssl_context()


def healthcheck(env: Mapping[str, str] | None = None, *, port: int = PORT,
                opener: Callable[..., object] = urllib.request.urlopen) -> int:
    """0 when ``/health`` answers 200 on this container, else 1."""
    env = os.environ if env is None else env
    try:
        if tls_enabled(env):
            # The certificate is our own self-signed one; this probe only checks liveness.
            ctx = ssl._create_unverified_context()  # noqa: S323  # nosec B323
            resp = opener(f"https://127.0.0.1:{port}{HEALTH_PATH}", timeout=2, context=ctx)
        else:
            resp = opener(f"http://127.0.0.1:{port}{HEALTH_PATH}", timeout=2)
        return 0 if getattr(resp, "status", None) == 200 else 1
    except Exception:  # noqa: BLE001 (any failure is "unhealthy")
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--healthcheck"]:
        return healthcheck()
    if argv:
        print("usage: python -m tac_bridge.serve [--healthcheck]", file=sys.stderr)
        return 2
    server = make_server()
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
