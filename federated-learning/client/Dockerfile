# Tahap 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Dependensi sistem untuk build numerik di ARM64
# libopenblas-dev & liblapack-dev: backend BLAS yang native ARM64
# (menggantikan MKL yang hanya ada di x86)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-inference.txt .

# PENTING: Tidak menggunakan --extra-index-url https://download.pytorch.org/whl/cpu
# karena indeks itu berisi wheel x86 (AVX/AVX2) yang crash di ARM64.
# PyPI default sudah menyediakan wheel ARM64 native untuk torch>=2.0.
RUN pip install --user --no-cache-dir -r requirements-inference.txt

# Tahap 2: Runner (image bersih tanpa build tools)
FROM python:3.10-slim

WORKDIR /app

# Runtime dependencies untuk OpenCV & BLAS di ARM64
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY app/ app/

# Environment Variables
ENV PATH="/root/.local/bin:${PATH}"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
# Paksa numpy/OpenBLAS menggunakan thread tunggal (stabilitas Jetson)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
