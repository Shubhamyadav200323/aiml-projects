# AIML-03 Architecture

## Three-tier view

```mermaid
flowchart LR
  subgraph P[Presentation tier]
    UI[Streamlit console<br/>For you / Similar items / A/B report]
  end
  subgraph A[Application tier]
    GW[FastAPI gateway<br/>/v1/recommendations /v1/events /v1/experiments]
    SVC[(RecommendationService<br/>routing, retrieval, ranking, A/B)]
    TR[Triton Inference Server<br/>user_tower · ranker_dcn_v2 · ranker_wide_deep]
    AF[Airflow DAG<br/>features -> train -> gate -> publish -> reload]
  end
  subgraph D[Data tier]
    RD[(Redis<br/>rec cache, TTL)]
    FX[(FAISS index<br/>item embeddings)]
    FT[(Feature tables<br/>user / item / seen)]
    PG[(PostgreSQL<br/>impression + click events)]
    SP[(Spark parquet<br/>interactions, features)]
  end
  UI -- HTTP --> GW --> SVC
  SVC --> RD
  SVC --> FX
  SVC --> FT
  SVC -- gRPC/HTTP tensors --> TR
  SVC --> PG
  AF --> SP --> AF
  AF --> TR
  AF --> FX
```

## Online request path (p50 4.3 ms, p99 6.7 ms in-process; 0.03 ms cache hit)

```mermaid
sequenceDiagram
  participant C as Client
  participant S as RecommendationService
  participant R as Redis
  participant T as user_tower (Triton)
  participant F as FAISS
  participant K as ranker (A/B arm)
  participant E as Event store
  C->>S: GET /v1/recommendations/u42?k=10
  S->>R: cache lookup rec:u42:...
  alt hit
    R-->>S: cached list
  else miss
    S->>S: route: >=3 positives -> two-stage, 1-2 or onboarding -> content, else popularity
    S->>T: user features -> 64-d embedding
    S->>F: top-300 by inner product, minus seen items
    S->>K: 300 x 44 ranking features (hash-assigned DCN-v2 or Wide&Deep)
    K-->>S: scores -> top-10
    S->>R: set (TTL 300 s)
  end
  S->>E: log impressions (variant, position, request_id)
  S-->>C: items + strategy + variant + latency
```

## Offline pipeline and timeline

```
|---------------- train ----------------|----- val -----|----- test -----|
                                        t1              t2
two-tower: fit on train positives, features as of t1 (point-in-time histories)
rankers  : fit on val-window labels over two-tower top-200 candidates
serving  : features rebuilt as of t2 with the *same id vocabulary*; evaluated on test
```

## Key design decisions

| Decision | Why |
|---|---|
| Global temporal split | Random splits let models train on a user's future; temporal split also produces real cold-start users. |
| Point-in-time training histories | `test_retrieval_examples_never_see_their_target` guarantees the target never appears in its own history. |
| In-batch softmax with log-Q correction | Popular items appear more often as in-batch negatives; log-Q correction removes that bias. Accidental hits are masked. |
| ID dropout in the user tower | Forces the tower to use history and taste features, so new users (id 0) still get good embeddings. |
| Fixed id vocabulary across refreshes | Embedding rows are tied to ids; new users map to the unknown row instead of shifting everyone else's index (a real bug caught in development). |
| Ranker trained on retrieval's output | Ranks exactly the candidate distribution it will see in production, not random negatives. |
| Ranker without user-id embeddings | User-id embeddings overfit the short labelling window; taste features generalise. |
| Spark and pandas engines with a parity test | Spark for 25M rows, pandas for CI. The test caught a session-time-zone bug that shifted the Spark cutoff by 5.5 hours. |
| Hash-based sticky assignment + z-test + power analysis | Stateless, reproducible experiments; no "winner" is declared without significance. |
| Quality-gated Airflow promotion with atomic swap | A bad nightly retrain cannot ship; `.prev` gives one-step rollback. |
| Same tensors for local and Triton backends | `LocalModels` and `TritonModels` share one interface; switching is an env var. |
