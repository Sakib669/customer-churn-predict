# Build and serve Customer Churn Prediction API
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Optimize python & UV behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application and model artifact
COPY main.py churn_model.pkl ./

# Expose FastAPI port
EXPOSE 8000

# Start Uvicorn server
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
