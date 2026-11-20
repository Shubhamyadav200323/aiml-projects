"""Central, environment-driven configuration.

Every tier reads its settings from here so that the same container image can be
promoted from a laptop (sqlite + in-memory state) to Kubernetes (Postgres +
Redis + Kafka) purely through environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- data tier -------------------------------------------------------
    artifacts_dir: Path = field(default_factory=lambda: Path(_env("FRAUD_ARTIFACTS", "artifacts")))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", "sqlite:///cases.db"))
    redis_url: str | None = field(default_factory=lambda: os.environ.get("REDIS_URL"))

    # --- streaming -------------------------------------------------------
    kafka_bootstrap: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:9092"))
    topic_transactions: str = field(default_factory=lambda: _env("TOPIC_TRANSACTIONS", "transactions"))
    topic_scores: str = field(default_factory=lambda: _env("TOPIC_SCORES", "fraud-scores"))

    # --- model / decisioning --------------------------------------------
    review_threshold: float = field(
        default_factory=lambda: float(_env("REVIEW_THRESHOLD", "0.5"))
    )
    decline_threshold: float = field(
        default_factory=lambda: float(_env("DECLINE_THRESHOLD", "0.9"))
    )
    mlflow_tracking_uri: str | None = field(
        default_factory=lambda: os.environ.get("MLFLOW_TRACKING_URI")
    )
    mlflow_experiment: str = field(default_factory=lambda: _env("MLFLOW_EXPERIMENT", "fraud"))

    # --- monitoring ------------------------------------------------------
    drift_share_trigger: float = field(
        default_factory=lambda: float(_env("DRIFT_SHARE_TRIGGER", "0.3"))
    )
    psi_threshold: float = field(default_factory=lambda: float(_env("PSI_THRESHOLD", "0.2")))


def get_settings() -> Settings:
    return Settings()
