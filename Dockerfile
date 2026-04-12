FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

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

ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
