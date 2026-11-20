# Recommendation API reference

OpenAPI: `GET /docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness, index backend, catalogue size, live p50/p99 |
| GET | `/metrics` | Prometheus: requests by strategy, cache hits, latency quantiles |
| GET | `/v1/recommendations/{user_id}?k=10&genres=Horror&genres=Comedy` | Top-k for a user (`genres` = onboarding picks for new users) |
| GET | `/v1/items/{item_id}/similar?k=10` | Nearest items in embedding space |
| POST | `/v1/events/click` | Log a click for the A/B experiment |
| GET | `/v1/experiments/report` | CTR per arm, lift, z-test p-value, 95% CI, click-NDCG |
| POST | `/v1/cache/invalidate?user_id=u42` | Drop cached lists for one user (or all) |

### `GET /v1/recommendations/u42?k=3`

```json
{
  "user_id": "u42",
  "request_id": "5d0c3f8e9b0a4c1e8d7f6a5b4c3d2e1f",
  "strategy": "two_stage",
  "variant": "dcn_v2",
  "items": [
    {"item_id": "i318", "score": 0.4123, "title": "Title 318", "genres": ["Drama", "War"]},
    {"item_id": "i77",  "score": 0.3981, "title": "Title 77",  "genres": ["Drama"]},
    {"item_id": "i905", "score": 0.3550, "title": "Title 905", "genres": ["Crime", "Drama"]}
  ],
  "cached": false,
  "latency_ms": 4.31
}
```

`strategy` is `two_stage` (3 or more positive interactions), `content` (1-2 positives, or
onboarding genres supplied) or `popularity` (no signal). `variant` is set only for `two_stage`.

### `POST /v1/events/click`

```json
{"user_id": "u42", "item_id": "i318", "position": 0, "request_id": "5d0c3f8e..."}
```

### `GET /v1/experiments/report`

```json
{
  "experiment": "ranker_dcn_vs_widedeep",
  "variants": {
    "dcn_v2":    {"impression": 19420, "click": 459, "ctr": 0.0236, "click_ndcg@10": 0.125},
    "wide_deep": {"impression": 20580, "click": 460, "ctr": 0.0224, "click_ndcg@10": 0.111}
  },
  "lift_abs": -0.0013, "lift_rel": -0.054, "z": -0.86, "p_value": 0.39,
  "ci95_low": -0.0042, "ci95_high": 0.0017, "significant_at_0.05": false, "winner": null
}
```

## Data contracts

| Artifact | Schema |
|---|---|
| `features/user_features.parquet` | `user_id` (index), `n_interactions`, `n_positives`, `avg_rating`, `aff_<genre>` x18, `history` (last 20 positive item ids) |
| `features/item_features.parquet` | `item_id` (index), `g_<genre>` x18, `pop_log`, `pop_recent` (30-day half-life), `item_avg_rating` |
| `features/seen.parquet` | `user_id`, `items` (all interacted ids, used to exclude them from recommendations) |
| `features/vocab.json` | `user_index`, `item_index` (embedding rows), `as_of` |
| `index/` | `items.faiss`, `item_vectors.npy`, `index_meta.json` |
| `models/<name>/1/model.savedmodel` | Triton layout; signatures in `triton/model_repository/*/config.pbtxt` |
| `events` table | `experiment, variant, user_id, item_id, position, event (impression/click), request_id, ts` |
