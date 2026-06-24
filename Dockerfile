FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FIFA_DATA_DIR=/data \
    FIFA_DB_PATH=/data/fifa_forecasts.db

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code (the .dockerignore keeps the local DB/artifacts out of the image —
# those live on the persistent volume mounted at /data)
COPY . .

# Persistent volume: SQLite DB + JSON archive (raw_requests/responses/traces/metadata)
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
