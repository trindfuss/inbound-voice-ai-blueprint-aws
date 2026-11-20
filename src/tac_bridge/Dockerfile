# OwlBank TAC bridge.
# Build from the repository root:
#   docker build -f src/tac_bridge/Dockerfile -t owlbank-tac-bridge .
# requirements.txt is `uv export --frozen --no-default-groups --group bridge --no-emit-project`
# (pinned, with hashes). Regenerate it after changing the `bridge` group or uv.lock.
# Pinned to the multi-arch index digest. To update: read Docker-Content-Digest from a HEAD of registry-1.docker.io/v2/library/python/manifests/3.12-slim (or `docker buildx imagetools inspect python:3.12-slim`).
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    TAC_ANALYTICS_DISABLED=true

WORKDIR /app

COPY src/tac_bridge/requirements.txt /app/requirements.txt
RUN pip install --require-hashes -r /app/requirements.txt

COPY src/owlbank /app/src/owlbank
COPY src/tac_bridge /app/src/tac_bridge

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-m", "tac_bridge.serve", "--healthcheck"]

# Secrets (TWILIO_AUTH_TOKEN, TWILIO_API_KEY, TWILIO_API_SECRET) arrive as ECS
# task-definition `secrets` (valueFrom Secrets Manager), never baked in here.
# uvicorn with the same options as before (WebSocket ping 20 s / timeout 20 s, no server header, no
# access log). On Fargate it serves TLS on 8080 with a self-signed certificate made in memory at
# start-up; on AgentCore, plain HTTP (see tac_bridge/serve.py).
CMD ["python", "-m", "tac_bridge.serve"]
