# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy dependency files first (for Docker cache)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY ai_proxy/ ./ai_proxy/

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "ai_proxy.main:app", "--host", "0.0.0.0", "--port", "8080"]
