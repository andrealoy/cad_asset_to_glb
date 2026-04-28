FROM python:3.13-slim

# OpenCascade system libs needed by cascadio
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libocct-modeling-algorithms-7.8 \
      libocct-modeling-data-7.8 \
      libocct-data-exchange-7.8 \
      libocct-foundation-7.8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Remove local .env — secrets come from Fly.io env vars
RUN rm -f .env

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--timeout", "600", \
     "--worker-class", "gthread", "--workers", "1", "--threads", "8"]
