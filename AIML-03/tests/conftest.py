from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest  # noqa: E402

from recsys.data import generate_synthetic, temporal_split  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    return generate_synthetic(n_users=500, n_items=300, interactions_per_user=40, seed=21)


@pytest.fixture(scope="session")
def split(dataset):
    return temporal_split(dataset.interactions)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory, dataset):
    """Train the full pipeline once (small) and share the artifacts across tests."""
    from recsys.pipeline import train
    from recsys.ranking.models import RankerConfig
    from recsys.retrieval.two_tower import TrainConfig

    out = tmp_path_factory.mktemp("artifacts")
    report = train(dataset, out, TrainConfig(epochs=4, batch_size=256), RankerConfig(epochs=2), n_candidates=100)
    return out, report


@pytest.fixture(scope="session")
def service(artifacts, tmp_path_factory):
    from recsys.experiments.ab import EventStore, Experiment
    from recsys.serving.models import LocalModels
    from recsys.serving.service import RecommendationService

    out, _ = artifacts
    db = tmp_path_factory.mktemp("db") / "events.db"
    return RecommendationService.from_artifacts(out, LocalModels(out / "models"), Experiment("test_exp"),
                                                events=EventStore(f"sqlite:///{db}"), n_candidates=100)
