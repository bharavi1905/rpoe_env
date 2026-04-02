# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled venv from builder (no build tools needed at runtime)
COPY --from=builder /app/.venv /app/.venv

# Copy only application source files
COPY --from=builder /app/server      /app/server
COPY --from=builder /app/tasks       /app/tasks
COPY --from=builder /app/models.py   /app/models.py
COPY --from=builder /app/__init__.py /app/__init__.py
COPY --from=builder /app/openenv.yaml /app/openenv.yaml
COPY --from=builder /app/README.md   /app/README.md
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

ENV ENABLE_WEB_INTERFACE=true
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
