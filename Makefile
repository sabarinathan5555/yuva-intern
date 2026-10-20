.PHONY: install train test run demo load docker-build docker-run docker-monitor clean
install:        ; pip install -r requirements.txt
train:          ; python model/train_model.py
test:           ; python -m unittest discover -s tests -v
run:            ; python wsgi.py
demo:           ; python scripts/demo_client.py
load:           ; python scripts/load_test.py
docker-build:   ; docker build -t iris-classifier-service:1.0.0 .
docker-run:     ; docker run --rm -p 8000:8000 -v model-logs:/app/logs iris-classifier-service:1.0.0
docker-monitor: ; docker compose --profile monitoring up --build
clean:          ; rm -rf logs model/artifacts __pycache__ */__pycache__
