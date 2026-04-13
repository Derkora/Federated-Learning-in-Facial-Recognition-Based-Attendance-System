# Tahap 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --user --no-cache-dir -r requirements-train.txt

# Tahap 2: Runner
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY app/ app/

ENV PATH=/root/.local/bin:$PATH

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
