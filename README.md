# Iris Classifier – Model Deployment & Monitoring (Proof of Concept)

A self-contained project that turns a pre-trained scikit-learn model into a containerised
HTTP service with structured logging, Prometheus-style metrics, health checks, input
validation, and data/performance-drift monitoring. The complete write-up is in
`Model_Deployment_Documentation.pdf` (also delivered as a separate file; a copy is in `docs/`).

## What is inside

| Path | Purpose |
|---|---|
| `model/train_model.py` | Trains the Random Forest, writes `model.joblib`, `metadata.json`, `baseline.json` |
| `app/main.py` | Flask app factory – endpoints, request hooks, error handlers |
| `app/schemas.py` | Strict request validation |
| `app/model_service.py` | Artifact loading, SHA-256 integrity check, warm-up, inference |
| `app/monitoring.py` | Metrics registry, usage stats, rolling accuracy, PSI drift monitor |
| `app/logging_config.py` | JSON logging → stdout + rotating files + prediction audit log |
| `Dockerfile`, `docker-compose.yml` | Multi-stage, non-root, health-checked container |
| `monitoring/` | Prometheus scrape config and alert rules |
| `scripts/demo_client.py` | End-to-end demo with dummy data and a simulated drift event |
| `scripts/load_test.py` | Small concurrent load generator |
| `tests/test_api.py` | 31 automated tests |
| `docs/evidence/` | Captured output of the demo, tests, load test, logs and metrics |

## Quick start (local, no Docker)

```bash
pip install -r requirements.txt
python model/train_model.py          # (artifacts are already included; this regenerates them)
python -m unittest discover -s tests -v
python wsgi.py                       # serves on http://localhost:8000
python scripts/demo_client.py        # in a second terminal
```

## Quick start (Docker)

```bash
docker build -t iris-classifier-service:1.0.0 .
docker run --rm -p 8000:8000 -v model-logs:/app/logs iris-classifier-service:1.0.0
# or, with Prometheus on :9090
docker compose --profile monitoring up --build
```

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}}'
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | One prediction. `features` is an object (named) or a 4-number array |
| POST | `/predict/batch` | Up to `MAX_BATCH_SIZE` (100) instances in `instances` |
| POST | `/feedback` | Submit ground truth for a `prediction_id` → rolling accuracy |
| GET | `/health/live` | Liveness – process is running |
| GET | `/health/ready` | Readiness – model loaded and verified (503 otherwise); `/health` is an alias |
| GET | `/model/info` | Model metadata and training metrics |
| GET | `/metrics` | Prometheus text exposition |
| GET | `/monitoring/summary` | Usage, performance and drift summary with human-readable alerts |
| GET | `/monitoring/drift` | Per-feature PSI, mean shift and prediction-distribution drift |

## Configuration (environment variables)

`MODEL_DIR`, `LOG_DIR`, `LOG_LEVEL`, `MAX_BATCH_SIZE`, `MAX_CONTENT_LENGTH`, `FEATURE_HARD_MIN`,
`FEATURE_HARD_MAX`, `DRIFT_WINDOW`, `DRIFT_MIN_SAMPLES`, `PSI_WARN`, `PSI_ALERT`,
`PERFORMANCE_WINDOW`, `PERFORMANCE_MIN_SAMPLES`, `ACCURACY_ALERT`, `LATENCY_SLO_MS`,
`MAX_PENDING_PREDICTIONS`, `PORT`. Defaults are in `app/config.py`.

## Known limitations

* Monitoring state lives in process memory; run one worker per container and scale by replicas.
* The Dockerfile / compose files were written and reviewed but **could not be built in the
  authoring environment (no Docker daemon, no network)**. Everything else was executed and tested.
* The prediction audit log stores raw inputs – review for PII before reusing on real data.
