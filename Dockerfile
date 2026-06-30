FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Persist the dataset on the mounted volume (compose overrides these too).
    FIFA_DATA_DIR=/data \
    FIFA_DB_PATH=/data/fifa_forecasts.db \
    FIFA_MATCHES_CSV=/data/matches.csv

WORKDIR /app

# Dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code.
COPY . .
RUN chmod +x /app/docker-entrypoint.sh

# Dataset (DB, raw archive, exports, matches CSV) lives here on a volume.
VOLUME ["/data"]

# Web dashboard port (Coolify maps your domain to this).
EXPOSE 8000

# Entrypoint seeds /data on first boot, then runs the dashboard. Coolify
# scheduled tasks / the terminal still exec the CLI inside this container.
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "fifa_forecast.web:app", "--host", "0.0.0.0", "--port", "8000"]
