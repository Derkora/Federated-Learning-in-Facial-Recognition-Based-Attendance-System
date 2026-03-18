FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model pre-trained ke dalam container
COPY global_model_v0.pth /app/global_model_v0.pth

# Copy source code
COPY app/ app/

# Buat folder untuk menyimpan model classifier lokal
RUN mkdir -p models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]