FROM python:3.10-slim-bullseye

WORKDIR /app

RUN echo "Acquire::Retries \"10\";" > /etc/apt/apt.conf.d/80-retries && \
    apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    libglib2.0-0 \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV OPENBLAS_CORETYPE=ARMV8
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY app/ app/

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
