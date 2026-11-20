.PHONY: install test lint eda train serve dashboard stream compose k8s loadtest

PY ?= python
export PYTHONPATH := src

install:
	$(PY) -m pip install -e ".[dev,graph,imbalance]"

test:
	$(PY) -m pytest --cov=fraud --cov-report=term-missing

lint:
	ruff check src tests

eda:
	$(PY) scripts/eda.py --dataset synthetic

train:
	$(PY) -m fraud.pipeline train --dataset synthetic --out artifacts

serve:
	uvicorn fraud.serving.api:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

loadtest:
	$(PY) scripts/benchmark.py --n 2000

compose:
	docker compose up --build

k8s:  ## Minikube: build inside the cluster's docker daemon, then apply all tiers
	eval $$(minikube docker-env) && docker build -t fraud-platform:latest .
	kubectl apply -f deploy/k8s/
	kubectl -n fraud wait --for=condition=complete job/train-initial --timeout=600s
	minikube service -n fraud fraud-dashboard --url
