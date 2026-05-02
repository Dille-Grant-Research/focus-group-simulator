FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --system --uid 1000 --no-create-home appuser

RUN mkdir /data && chown appuser:appuser /data

WORKDIR /app

COPY backend/ ./
RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
