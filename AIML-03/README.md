# AIML-03 · Large-Scale Personalised Recommendation Engine

A two-stage recommender in the style used by Netflix and Amazon. **Stage 1** retrieves a
few hundred candidates in milliseconds with a two-tower model and FAISS. **Stage 2**
re-ranks them precisely with a deep ranking model (DCN-v2 or Wide & Deep, chosen per user
by an online A/B test). Cold-start users get content-based and popularity fallbacks,
results are cached in Redis, models are served by Triton, and Airflow retrains nightly.

| | |
|---|---|
| **Stack** | TensorFlow Recommenders · FAISS · Redis · Apache Spark · Airflow · Triton Inference Server · FastAPI · PostgreSQL · Streamlit · Kubernetes |
| **Offline** | NDCG@10 **0.103** vs 0.074 popularity (+40%) · Recall@10 0.154 · see [docs/RESULTS.md](docs/RESULTS.md) |
| **Online A/B** | DCN-v2 vs Wide & Deep: p = 0.39, no significant winner (power analysis included) |
| **Latency** | p50 4.3 ms · p99 6.7 ms (300 candidates) · cache hit 0.03 ms |
| **Tests** | 20 passing, including Spark/pandas feature parity |

## Architecture (3-tier)

| Tier | Components |
|---|---|
| **Presentation** | `ui/app.py`: Streamlit console (personalised feed with click logging, similar items, live A/B report) |
| **Application** | `serving/api.py` FastAPI gateway · `serving/service.py` orchestration · Triton (`triton/`) · Airflow DAG (`dags/`) |
| **Data** | Redis cache · FAISS index · feature tables (Spark/pandas parquet) · PostgreSQL event store |

Diagrams and design decisions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). API and
data contracts: [docs/API.md](docs/API.md).

```
AIML-03/
├── src/recsys/
│   ├── data/          MovieLens 25M loader, synthetic generator, global temporal split
│   ├── features/      pandas feature tables + Spark job (same columns, parity-tested)
│   ├── retrieval/     point-in-time inputs, TFRS two-tower model, FAISS index (flat/IVF/HNSW)
│   ├── ranking/       ranking features, DCN-v2 & Wide&Deep (TFRS), serving-distribution dataset
│   ├── serving/       service, FastAPI app, Redis/memory cache, cold start, local/Triton backends
│   ├── experiments/   A/B assignment, event store, z-test, power analysis, traffic simulator
│   ├── evaluation.py  Recall@K, NDCG@K, HitRate@K, MRR, coverage
│   └── pipeline.py    end-to-end training + evaluation CLI
├── dags/              Airflow: spark -> train -> quality gate -> Triton publish -> reload -> smoke test
├── triton/            model repository configs (dynamic batching, versioning)
├── ui/                Streamlit console
├── deploy/k8s/        data / application / presentation tiers (+HPA)
├── scripts/           benchmark.py
└── tests/             20 tests
```

## Quick start

```bash
cd AIML-03
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,spark,ui]"
export TF_USE_LEGACY_KERAS=1          # TFRS needs Keras 2 (tf-keras); the Makefile sets it
make train        # ~40 s on CPU: artifacts/{models,index,features,metrics.json}
make test         # 20 tests
make ab           # simulated online A/B test -> report
make bench        # latency
make serve        # http://localhost:8100/docs
make ui           # http://localhost:8501
```

```bash
curl -s 'localhost:8100/v1/recommendations/u42?k=5' | jq
curl -s 'localhost:8100/v1/recommendations/new_user?k=5&genres=Horror&genres=Sci-Fi' | jq   # cold start
curl -s localhost:8100/v1/experiments/report | jq
```

Full stack with Triton, Redis and Postgres: `docker compose up --build`. Kubernetes
(Minikube): `make k8s`. MovieLens 25M: download `ml-25m.zip` from GroupLens into `data/`,
then run `make spark-features` and `make train-ml25m`.

## Feature breakdown

| Milestone | Implementation |
|---|---|
| 1. Data + temporal split | `data/movielens.py` (MovieLens 25M schema), `data/split.py` global time cut, implicit positives = rating >= 4 |
| 2. Spark features | `features/spark_job.py`: interaction table, user history, genre affinity, recency-weighted popularity; pandas twin with a parity test |
| 3. Two-tower (TFRS) | Shared item embeddings, mean-pooled history, ID dropout, L2-normalised towers, in-batch softmax with log-Q correction and accidental-hit removal |
| 4. FAISS | `retrieval/index.py`: flat / IVF / HNSW, seen-item exclusion, persistence |
| 5. Deep ranking | DCN-v2 (`tfrs.layers.dcn.Cross`) and Wide & Deep over 44 features, trained on retrieval's own candidates |
| 6. Cold start + Redis | Routing policy; content (genres) + popularity blend; Redis cache with TTL, LRU fallback, per-user invalidation |
| 7. Triton, Airflow, A/B | SavedModels in Triton layout with dynamic batching; Airflow DAG with quality gate and atomic promotion; hash-assigned A/B with z-test, CI, click-NDCG and power analysis; NDCG, Recall@K and CTR reported |

## Next steps

Diversity-aware re-ranking (MMR) to recover coverage, sequence-aware user towers
(transformer over history), and replacing the click simulator with interleaving on real traffic.
