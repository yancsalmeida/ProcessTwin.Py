# Production Dockerfile for ProcessTwin.Py
FROM python:3.12-slim

# Copy uv binary from official distribution for ultra-fast packaging
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Runtime environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy project specification files for layer caching
COPY pyproject.toml README.md uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev || uv sync --no-dev

# Copy source code and tests
COPY src/ src/
COPY tests/ tests/

# Install package
RUN uv sync --no-dev

# Create data directory for persistent SQLite time-series storage
RUN mkdir -p /app/data

# Set persistent database path and ensure virtual environment PATH
ENV PROCESS_TWIN_DB=/app/data/processtwin.db
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Launch Uvicorn ASGI server
CMD ["uvicorn", "process_twin.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
