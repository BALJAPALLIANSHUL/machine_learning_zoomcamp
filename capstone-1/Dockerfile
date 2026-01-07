FROM python:3.10-slim

WORKDIR /app

# Install system deps (important for sklearn, numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST (smallest possible)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Install remaining deps
RUN pip install --no-cache-dir -r requirements.txt

COPY src src
COPY models models

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port $PORT"]
