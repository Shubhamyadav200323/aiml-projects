"""Feast definitions.

Two views share the *same* feature names as `fraud.features.definitions`:

* `transaction_features` - point-in-time features per transaction, used with
  `get_historical_features` to build leakage-free training sets.
* `user_profile` - latest per-user aggregates, materialised to Redis and
  refreshed by the streaming consumer through the `user_profile_push` source,
  so the online path reads exactly what the offline path trained on.

Apply with:  cd feature_repo && feast apply && feast materialize-incremental $(date -u +%FT%T)
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, PushSource
from feast.types import Float64, Int64

transaction = Entity(name="transaction", join_keys=["transaction_id"])
user = Entity(name="user", join_keys=["user_id"])

transaction_source = FileSource(
    name="transaction_features_source",
    path="data/transaction_features.parquet",
    timestamp_field="timestamp",
)

user_profile_source = FileSource(
    name="user_profile_source",
    path="data/user_profile.parquet",
    timestamp_field="timestamp",
)

user_profile_push = PushSource(name="user_profile_push", batch_source=user_profile_source)

_TXN_FEATURES = [
    "amount_log", "txn_count_1h", "txn_count_24h", "amount_sum_1h", "amount_sum_24h",
    "user_txn_count", "user_avg_amount", "amount_to_avg_ratio", "seconds_since_last_txn",
    "is_new_device", "is_new_location", "is_new_card", "device_changed", "location_changed",
    "device_user_count", "card_user_count", "hour_of_day", "channel_code",
]

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=90),
    schema=[Field(name=f, dtype=Float64) for f in _TXN_FEATURES] + [Field(name="is_fraud", dtype=Int64)],
    source=transaction_source,
    online=False,
)

user_profile = FeatureView(
    name="user_profile",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="user_txn_count", dtype=Float64),
        Field(name="user_avg_amount", dtype=Float64),
        Field(name="graph_score", dtype=Float64),
    ],
    source=user_profile_push,
    online=True,
)

fraud_model_v1 = FeatureService(name="fraud_model_v1", features=[transaction_features])
