FROM python:3.10-slim

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mlflow psycopg2 boto3 minio gunicorn
