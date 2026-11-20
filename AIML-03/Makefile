.PHONY: install test lint train spark-features serve ui ab bench triton compose k8s

PY ?= python
export PYTHONPATH := src
export TF_USE_LEGACY_KERAS := 1

install:
	$(PY) -m pip install -e ".[dev,spark,ui]"

test:
	$(PY) -m pytest --cov=recsys --cov-report=term-missing

lint:
	ruff check src tests dags ui scripts

train:
	$(PY) -m recsys.pipeline train --dataset synthetic --out artifacts

train-ml25m:  ## after downloading https://files.grouplens.org/datasets/movielens/ml-25m.zip to data/
	$(PY) -m recsys.pipeline train --dataset movielens --path data/ml-25m --out artifacts

spark-features:
	spark-submit src/recsys/features/spark_job.py --ratings data/ml-25m/ratings.csv \
	  --movies data/ml-25m/movies.csv --as-of 2019-01-01 --out data/features

serve:
	uvicorn recsys.serving.api:app --reload --port 8100

ui:
	streamlit run ui/app.py

ab:
	$(PY) -m recsys.experiments.simulate --artifacts artifacts --sessions 4000

bench:
	$(PY) scripts/benchmark.py --artifacts artifacts --n 500

triton:
	cp -rn triton/model_repository/. artifacts/models/
	docker run --rm -p 8000:8000 -v $$PWD/artifacts/models:/models nvcr.io/nvidia/tritonserver:24.05-py3 \
	  tritonserver --model-repository=/models --model-control-mode=explicit \
	  --load-model=user_tower --load-model=ranker_dcn_v2 --load-model=ranker_wide_deep

compose:
	docker compose up --build

k8s:
	eval $$(minikube docker-env) && docker build -t recsys-engine:latest .
	kubectl apply -f deploy/k8s/
	kubectl -n recsys wait --for=condition=complete job/train-initial --timeout=900s
	minikube service -n recsys recsys-ui --url
