FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from root-level requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create the package directory structure the relative imports expect:
#   /app/
#   ├── meeting_env/
#   │   ├── __init__.py
#   │   ├── models.py
#   │   ├── client.py
#   │   └── server/
#   │       ├── __init__.py
#   │       ├── app.py
#   │       ├── meeting_environment.py
#   │       ├── tasks.py
#   │       └── grader.py
#   ├── inference.py
#   └── openenv.yaml
RUN mkdir -p /app/meeting_env/server

COPY server/ /app/server/

# Copy root-level files
COPY inference.py  /app/inference.py
COPY openenv.yaml  /app/openenv.yaml

# PYTHONPATH must include /app so that "import server" resolves correctly
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check — validates /health endpoint required by OpenEnv
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
