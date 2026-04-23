# Tahap 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .
RUN pip install --user --no-cache-dir -r requirements-inference.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN find /root/.local -name "*.so" -exec patchelf --clear-execstack {} +

# Tahap 2: Runner
FROM python:3.10-slim

WORKDIR /app

# Library dasar untuk OpenCV & GLIB (ARM64 Stability)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY app/ app/

# Environment Variables
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
