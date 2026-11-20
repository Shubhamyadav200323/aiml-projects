# AIML-01 · Real-Time Financial Fraud Detection Platform

Streams live transactions, scores each one for fraud risk in about **1 ms**, links
suspicious accounts into **fraud rings** with a graph neural network, and gives
analysts a console to review cases. Their feedback labels close the retraining loop.

| | |
|---|---|
| **Stack** | Kafka · XGBoost · PyTorch Geometric (GraphSAGE) · Feast · MLflow · FastAPI · Evidently AI · Redis · PostgreSQL · Streamlit · Kubernetes |
| **Test PR-AUC** | 0.959 to 0.966 (baseline 0.011) - see [docs/RESULTS.md](docs/RESULTS.md) |
| **Latency** | p50 0.94 ms · p99 1.38 ms · ~1k TPS per core |
| **Tests** | 23 passing (`make test`) |

## Architecture (3-tier)

| Tier | Components |
|---|---|
| **Presentation** | `dashboard/app.py`: Streamlit console (live alerts, case review with SHAP reasons, fraud-ring graph, manual scoring) |
| **Application** | `serving/api.py` FastAPI · `streaming/consumer.py` Kafka scorer · `pipeline.py` training · `monitoring/` drift + retraining |
| **Data** | Kafka topics · Redis online feature state · PostgreSQL cases/labels · Feast feature repo · MLflow tracking · model artifact volume |

Full diagrams and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). API and
data contracts: [docs/API.md](docs/API.md).

```
AIML-01/
├── src/fraud/
│   ├── data/          dataset adapters (PaySim, IEEE-CIS) + synthetic generator with fraud rings
│   ├── features/      point-in-time batch features, streaming feature engine, Feast/Redis store
│   ├── models/        XGBoost, account graph, GraphSAGE, blender, metrics, MLflow tracker
│   ├── serving/       FastAPI app, schemas, scorer domain service, Prometheus metrics
│   ├── streaming/     Kafka producer (replay) / consumer (scorer) + in-memory broker
│   ├── monitoring/    PSI + Evidently drift, retrain trigger, champion/challenger
│   ├── repository/    SQLAlchemy case store (sqlite / PostgreSQL)
│   └── pipeline.py    end-to-end training CLI
├── feature_repo/      Feast definitions (feature_store.yaml, definitions.py)
├── dashboard/         Streamlit analyst console
├── deploy/k8s/        namespace/config, data tier, application tier (+HPA, CronJob), presentation tier
├── scripts/           eda.py, benchmark.py
├── tests/             23 tests
└── docs/              ARCHITECTURE, API, EDA, RESULTS
```

## Quick start (local, no infrastructure)

```bash
cd AIML-01
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"          # add ".[graph,mlops,streaming,imbalance]" for the full stack
make eda          # docs/EDA.md: class imbalance and graph signal
make train        # artifacts/ (xgb.json, gnn.pt, graph_scores.parquet, policy.json, rings.json, metrics.json)
make test         # 23 tests
make serve        # http://localhost:8000/docs
make dashboard    # http://localhost:8501
```

Score a transaction:

```bash
curl -s localhost:8000/v1/score -H 'content-type: application/json' -d '{
  "transaction_id":"T1","timestamp":"2026-01-15T10:32:00Z","user_id":"U42","card_id":"C42",
  "device_id":"D9000001","merchant_id":"M7","location":"L3","amount":950,"channel":"TRANSFER"}'
```

## Full stack

```bash
docker compose up --build       # Kafka, Redis, Postgres, MLflow, trainer, API, consumer, producer, dashboard
# MLflow http://localhost:5000 · API http://localhost:8000 · Dashboard http://localhost:8501
```

Kubernetes (Minikube):

```bash
minikube start --cpus 4 --memory 8g
make k8s                        # builds image in-cluster, applies deploy/k8s/, waits for training job
```

Use a real dataset:

```bash
# Kaggle: ealaxi/paysim1  or  ieee-fraud-detection
python -m fraud.pipeline train --dataset paysim --path data/PS_20174392719_1491204439457_log.csv --imbalance smote
cd feature_repo && feast apply      # after: python -m fraud.pipeline train ... --export-feast
```

## Feature breakdown

| Milestone | Implementation |
|---|---|
| 1. Dataset + EDA | `data/loaders.py` maps PaySim / IEEE-CIS onto one schema; `scripts/eda.py` writes `docs/EDA.md` (0.52% fraud, 1:192) |
| 2. Features + Feast | 19 features: velocity (1h/24h count and sum), amount vs user average, time since last txn, new device/location/card, device/card sharing. Vectorised batch path plus O(window) streaming path with **tested parity**; Feast views in `feature_repo/` |
| 3. XGBoost + MLflow | `scale_pos_weight` or SMOTE, early stopping on val **aucpr**, TreeSHAP reasons, every run tracked in MLflow |
| 4. Graph + GNN | Account graph via shared devices/cards, inductive **GraphSAGE** (PyG `SAGEConv`, pure-torch fallback), blend fitted on validation, ring detection |
| 5. Kafka + FastAPI | Event-time replay producer keyed by user; consumer with micro-batch commits, DLQ, idempotent cases; FastAPI with validation, batch scoring, OpenAPI |
| 6. Evidently + retraining | PSI per feature on a sliding window, Evidently HTML report, trigger on drift / decay / label volume, champion/challenger promotion |
| 7. Dashboard + K8s | Streamlit console with feedback labels; manifests per tier with probes, HPA, PDB, CronJob |

## Next steps

Streaming graph store for real-time ring updates, a cost-sensitive threshold tuned to
chargeback cost versus analyst cost, and shadow deployment of challengers before promotion.
