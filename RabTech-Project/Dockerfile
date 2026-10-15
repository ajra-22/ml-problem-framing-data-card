# ==============================================================================
# Production Dockerfile for Customer Churn ML REST API Microservice
# ==============================================================================

FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr for real-time logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    PORT=8000

# Install system dependencies (curl required for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

# Create an unprivileged user for security compliance (non-root)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy and install pinned python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source, model artifacts, and supporting configuration
COPY app/ /workspace/app/
COPY artifacts/ /workspace/artifacts/
COPY pytest.ini /workspace/pytest.ini

# Ensure correct permissions for the non-root user
RUN chown -R appuser:appgroup /workspace

# Switch to unprivileged runtime user
USER appuser

# Expose microservice HTTP port
EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
