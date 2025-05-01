## Dockerfile for FastAPI App with GitPython Support and Best Practices

# Use the official Python slim image
FROM python:3.12.3-slim AS base

# Set environment variables for reproducibility and GitPython
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git \
    GIT_PYTHON_REFRESH=quiet

# Install system dependencies (git) and clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy and install Python dependencies early for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/

# Create a non-root user and adjust ownership
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Optional healthcheck (adjust path if you have a /health endpoint)
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
