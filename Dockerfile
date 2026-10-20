# syntax=docker/dockerfile:1
# ---------- Stage 1: build dependencies + train the model ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model/train_model.py model/train_model.py
# Train inside the image so the pickle is produced by the exact library versions that will load it.
RUN python model/train_model.py --out /build/artifacts

# ---------- Stage 2: slim runtime image ----------
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="iris-classifier-service" \
      org.opencontainers.image.description="Flask model-serving PoC with logging and drift monitoring"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/model/artifacts \
    LOG_DIR=/app/logs \
    LOG_LEVEL=INFO \
    PORT=8000
# Non-root user: a compromised process cannot modify the image filesystem
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/artifacts /app/model/artifacts
COPY app /app/app
COPY wsgi.py /app/wsgi.py
RUN mkdir -p /app/logs && chown -R app:app /app/logs
USER app
EXPOSE 8000
VOLUME ["/app/logs"]

# Container-level health check: uses only the standard library (no curl in slim images).
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).status==200 else 1)"

# ONE worker process (threads for concurrency) keeps the in-process monitoring state coherent.
# Scale out with more container replicas and let Prometheus aggregate /metrics.
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--worker-class", "gthread", \
     "--threads", "8", "--timeout", "30", "--graceful-timeout", "20", "--access-logfile", "-", \
     "--max-requests", "5000", "--max-requests-jitter", "500"]
