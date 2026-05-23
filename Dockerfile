# ============================================================================
# Nairacle Research Agent — Railway Docker Image
# ============================================================================
# Multi-stage build to keep the final image lean.
# Stage 1: Install Python deps + pre-download the reranker model.
# Stage 2: Copy only what's needed into a slim runtime image.
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for psycopg2-binary and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps into a virtual env for clean copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY research_agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Reranker model caching removed to save memory on Railway (OOM prevention).

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Runtime deps only (no gcc needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 && \
    rm -rf /var/lib/apt/lists/*

# Copy the virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the cached model from builder
COPY --from=builder /root/.cache /root/.cache

# Copy application code
COPY . .

# Railway injects PORT; shell form lets the shell expand $PORT
# Default to 8000 for local testing
ENV PORT=8000
EXPOSE ${PORT}

# Health check for Railway's built-in monitoring
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Start the server — JSON form for proper signal handling,
# sh -c for $PORT expansion
CMD ["sh", "-c", "uvicorn research_agent.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
