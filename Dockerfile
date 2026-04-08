FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY server/grader.py             ./server/grader.py
COPY server/tasks.py              ./server/tasks.py
COPY server/models.py             ./server/models.py
COPY server/client.py             ./server/client.py
COPY server/meeting_environment.py ./server/meeting_environment.py
COPY server/app.py                ./server/app.py
COPY server/__init__.py           ./server/__init__.py
COPY inference.py   .
COPY openenv.yaml   .

RUN pip install --no-cache-dir -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
