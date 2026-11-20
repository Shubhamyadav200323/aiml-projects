"""Feature-store integration (data tier).

* `export_offline` writes the parquet tables that the Feast repository in
  `feature_repo/` registers as offline sources.
* `build_online_engine` wires the streaming feature engine to Redis when
  `REDIS_URL` is set (multi-replica deployments) or to process memory otherwise.
* `FeastClient` is a thin, optional wrapper used for historical retrieval and
  pushing refreshed user profiles; it degrades to a no-op when Feast is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fraud.config import Settings
from fraud.features.definitions import FEATURE_COLUMNS
from fraud.features.online import InMemoryBackend, OnlineFeatureEngine, RedisBackend

log = logging.getLogger(__name__)


def export_offline(features: pd.DataFrame, repo_dir: str | Path = "feature_repo") -> dict[str, Path]:
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    txn_path = data_dir / "transaction_features.parquet"
    cols = ["transaction_id", "user_id", "timestamp", *FEATURE_COLUMNS, "is_fraud"]
    features[cols].to_parquet(txn_path, index=False)

    last = features.sort_values("timestamp").groupby("user_id").tail(1)
    profile = pd.DataFrame(
        {
            "user_id": last["user_id"],
            "timestamp": last["timestamp"],
            "user_txn_count": last["user_txn_count"] + 1,
            "user_avg_amount": (last["user_avg_amount"] * last["user_txn_count"] + last["amount"])
            / (last["user_txn_count"] + 1),
            "graph_score": 0.0,
        }
    )
    profile_path = data_dir / "user_profile.parquet"
    profile.to_parquet(profile_path, index=False)
    return {"transaction_features": txn_path, "user_profile": profile_path}


def build_online_engine(settings: Settings) -> OnlineFeatureEngine:
    if settings.redis_url:
        try:
            import redis

            client = redis.Redis.from_url(settings.redis_url)
            client.ping()
            log.info("online feature state -> redis %s", settings.redis_url)
            return OnlineFeatureEngine(RedisBackend(client))
        except Exception as exc:  # pragma: no cover - depends on infra
            log.warning("redis unavailable (%s); falling back to in-memory state", exc)
    return OnlineFeatureEngine(InMemoryBackend())


class FeastClient:
    def __init__(self, repo_path: str | Path = "feature_repo") -> None:
        try:
            from feast import FeatureStore

            self.store = FeatureStore(repo_path=str(repo_path))
        except Exception as exc:  # pragma: no cover - optional dependency
            log.info("feast disabled: %s", exc)
            self.store = None

    @property
    def enabled(self) -> bool:
        return self.store is not None

    def historical(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            raise RuntimeError("feast is not installed/configured")
        return self.store.get_historical_features(
            entity_df=entity_df, features=self.store.get_feature_service("fraud_model_v1")
        ).to_df()

    def push_profiles(self, profiles: pd.DataFrame) -> None:
        if self.enabled:
            self.store.push("user_profile_push", profiles)
