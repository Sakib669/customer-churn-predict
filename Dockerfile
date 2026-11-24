# ==============================================================================
# Production Dockerfile for Customer Churn Prediction Service
# Optimized with UV for sub-second builds and deterministic caching.
# ==============================================================================

# Use official lightweight Python 3.14 Bookworm image with UV pre-installed
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# Set standard container working directory
WORKDIR /app

# ------------------------------------------------------------------------------
# Environment Configurations
# - PYTHONUNBUFFERED: Stream logs immediately to container stdout/stderr
# - PYTHONDONTWRITEBYTECODE: Avoid writing transient .pyc files
# - UV_COMPILE_BYTECODE: Compile clean bytecode during sync for faster startup
# - UV_LINK_MODE: Use file copies across filesystems
# ------------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ------------------------------------------------------------------------------
# Layer 1: Dependency Installation & Layer Caching
# Copy only lockfile and project descriptor to leverage Docker build cache
# ------------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./

# Install production dependencies without development packages
RUN uv sync --frozen --no-dev --no-install-project

# ------------------------------------------------------------------------------
# Layer 2: Application Source & Model Artifacts
# ------------------------------------------------------------------------------
COPY main.py churn_model.pkl ./

# Expose internal FastAPI Uvicorn listener port
EXPOSE 8000

# Container entrypoint executing production Uvicorn server bound to all interfaces
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
