# Results

Reproduce: `make train && make ab && make bench && make test`. The dataset is MovieLens-shaped
synthetic data (2,000 users, 1,500 items, 88,716 interactions, 18 genres, Zipf popularity,
taste drift, 12% late-joining users) with a global temporal 80/10/10 split. Run on MovieLens
25M with `make train-ml25m` (Spark features via `make spark-features`).

## Offline: warm users (1,297 test users with 3 or more prior positives)

Candidates: two-tower top-200 from FAISS (exact inner product), already-seen items excluded.

| Model | Recall@10 | NDCG@10 | HitRate@10 | MRR | Recall@100 | Coverage@10 |
|---|---|---|---|---|---|---|
| Popularity (30-day decay) | 0.113 | 0.074 | 0.227 | 0.104 | 0.385 | 1.8% |
| Two-tower retrieval only | 0.131 | 0.086 | 0.256 | 0.122 | 0.423 | 27.9% |
| **Two-tower + DCN-v2** | 0.152 | **0.103** | 0.292 | **0.142** | 0.438 | 12.3% |
| **Two-tower + Wide & Deep** | **0.154** | 0.102 | **0.295** | 0.140 | **0.442** | 11.1% |

* The two-stage system improves NDCG@10 by **+40% over popularity** and **+20% over
  retrieval alone**; recall@10 rises by +34% and +16%.
* Rankers trade some catalogue coverage for precision, a known effect (they learn
  popularity as a prior). Diversity re-ranking is listed under next steps.

## Offline: cold-start users (108 new users, one onboarding signal)

| Fallback | Recall@10 | NDCG@10 | HitRate@10 | MRR |
|---|---|---|---|---|
| Popularity | 0.187 | 0.114 | 0.315 | 0.136 |
| **Content (genres) + popularity, w=0.8** | **0.204** | **0.124** | **0.333** | **0.146** |

The blend weight was tuned on a separate synthetic seed, not on this test set.

## Online A/B test (simulated traffic through the live service)

4,000 sessions, hash-assigned 50/50, position-biased cascade click model, events logged to the
event store and read back exactly as production traffic would be.

| Arm | Impressions | Clicks | CTR | Click-NDCG@10 |
|---|---|---|---|---|
| A: DCN-v2 (control) | 19,420 | 459 | 2.36% | 0.125 |
| B: Wide & Deep | 20,580 | 460 | 2.24% | 0.111 |

Relative lift -5.4%, z = -0.86, **p = 0.39**, 95% CI on the CTR difference [-0.42, +0.17] pp:
**no significant difference**, which agrees with the near-tie offline. Power analysis
(`required_sample_size`) shows about **273k impressions per arm** are needed to detect a 5%
lift at 2.3% CTR, so the decision is to keep the control and keep collecting traffic rather
than ship on noise.

## Serving latency (in-process TF SavedModels, 300 candidates, cache disabled, 1 CPU)

| p50 | p95 | p99 | Redis/memory cache hit |
|---|---|---|---|
| 4.29 ms | 5.70 ms | 6.70 ms | 0.03 ms |

## Tests

`pytest`: **20 tests passed** covering temporal split integrity, point-in-time features,
stable vocabularies, target-leak guard, table round-trip, MovieLens loader,
**Spark vs pandas feature parity**, metric correctness, FAISS exact/IVF search with
exclusion and persistence, exported Triton signatures, two-stage invariants, seen-item
exclusion, cold-start routing, cache hit and TTL, sticky and balanced assignment, z-test,
power analysis, the simulated A/B test, and every API endpoint.
