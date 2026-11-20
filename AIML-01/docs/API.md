# Scoring API reference

Interactive OpenAPI docs are served at `GET /docs` (Swagger) and `GET /redoc`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness, model version, live p50/p99 latency |
| GET | `/metrics` | Prometheus metrics (decision counters, latency histogram) |
| POST | `/v1/score` | Score one transaction |
| POST | `/v1/score/batch` | Score up to 1000 transactions (processed in event-time order) |
| GET | `/v1/cases?status=open&min_score=0.5&limit=50&offset=0` | List alerts, highest risk first |
| GET | `/v1/cases/stats` | Case counts by status |
| GET | `/v1/cases/{id}` | Case detail incl. SHAP reasons |
| PATCH | `/v1/cases/{id}` | Analyst verdict -> feedback label |
| GET | `/v1/rings?min_size=3` | Detected fraud rings |
| GET | `/v1/users/{user_id}/risk` | Graph risk + ring membership for an account |

## `POST /v1/score`

```json
{
  "transaction_id": "T000000123",
  "timestamp": "2026-01-15T10:32:00Z",
  "user_id": "U42", "card_id": "C42", "device_id": "D42",
  "merchant_id": "M7", "location": "L3",
  "amount": 129.99, "channel": "ONLINE"
}
```

Response:

```json
{
  "transaction_id": "T000000123",
  "fraud_score": 0.912,
  "xgb_score": 0.97, "graph_score": 0.85,
  "decision": "decline",
  "reasons": [
    {"feature": "amount_to_avg_ratio", "contribution": 2.31},
    {"feature": "is_new_device", "contribution": 1.12},
    {"feature": "device_user_count", "contribution": 0.74}
  ],
  "case_id": 17,
  "model_version": "3f9a1c0b2d4e",
  "latency_ms": 1.12
}
```

Validation errors return `422` with field-level detail (for example `amount` must be > 0).

## `PATCH /v1/cases/{id}`

```json
{"status": "confirmed_fraud", "analyst": "asha", "note": "matches ring #3"}
```

`status` is one of `open | in_review | confirmed_fraud | false_positive`; the last two
set `label` to 1 / 0, and those labels feed the retraining trigger.

## Kafka topics

| Topic | Key | Value |
|---|---|---|
| `transactions` | `user_id` | transaction JSON (schema above, no label) |
| `fraud-scores` | `transaction_id` | score response JSON |
| `transactions.dlq` | `transaction_id` | messages that failed to score |

## Data model (`cases` table)

`id, transaction_id (unique), user_id, amount, fraud_score, xgb_score, graph_score,
decision, status, label, reasons (json), features (json), analyst, note, created_at, updated_at`
