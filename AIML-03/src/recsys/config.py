"""Environment-driven settings shared by every tier."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(k: str, d: str) -> str:
    return os.environ.get(k, d)


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path = field(default_factory=lambda: Path(_env("RECSYS_ARTIFACTS", "artifacts")))
    redis_url: str | None = field(default_factory=lambda: os.environ.get("REDIS_URL"))
    cache_ttl_s: int = field(default_factory=lambda: int(_env("CACHE_TTL_S", "300")))
    triton_url: str | None = field(default_factory=lambda: os.environ.get("TRITON_URL"))
    n_candidates: int = field(default_factory=lambda: int(_env("N_CANDIDATES", "300")))
    cold_start_min_interactions: int = field(default_factory=lambda: int(_env("COLD_START_MIN", "3")))
    experiment_name: str = field(default_factory=lambda: _env("EXPERIMENT", "ranker_dcn_vs_widedeep"))
    experiment_split: float = field(default_factory=lambda: float(_env("EXPERIMENT_SPLIT", "0.5")))
    events_db: str = field(default_factory=lambda: _env("EVENTS_DB", "sqlite:///events.db"))


def get_settings() -> Settings:
    return Settings()
