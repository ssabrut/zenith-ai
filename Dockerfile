# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# 1. Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. FIX: Copy 'uv' directly from the official image (No pip install needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Create venv and install dependencies
# We use --compile-bytecode to speed up startup time
COPY requirements.txt .
RUN uv venv /opt/venv && \
    uv pip install --no-cache -r requirements.txt --python /opt/venv --compile-bytecode

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.10-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install ONLY runtime libs (libpq5 for Postgres)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]