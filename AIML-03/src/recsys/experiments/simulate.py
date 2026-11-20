"""Simulated online A/B test: ranker DCN-v2 (control) vs Wide & Deep (treatment).

    python -m recsys.experiments.simulate --artifacts artifacts --sessions 4000

Live traffic is not available offline, so we replay held-out test users through
the *real* online service (hash assignment, caching off, impression logging)
and generate clicks from a position-biased cascade model:

    P(click | item at position p) = relevance(item) * 0.95 ** p * 0.7  +  0.002 noise

where relevance = 1 if the user positively interacted with the item in the
test period. CTR, lift, z-test and click-NDCG are read from the event store,
exactly as they would be for real traffic.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from recsys.experiments.ab import EventStore
from recsys.serving.service import RecommendationService


def simulate(service: RecommendationService, truth: dict[str, set[str]], sessions: int = 2000,
             k: int = 10, seed: int = 0) -> dict:
    assert service.events is not None, "service needs an EventStore"
    rng = np.random.default_rng(seed)
    users = [u for u in truth if service.policy.route(service.tables, u) == "two_stage"]
    for _ in range(sessions):
        u = users[rng.integers(len(users))]
        resp = service.recommend(u, k=k, use_cache=False)
        for p, it in enumerate(resp.items):
            rel = it.item_id in truth[u]
            if rng.random() < (0.7 * 0.95 ** p if rel else 0.002):
                service.log_click(u, it.item_id, p, resp.request_id)
    return service.events.report(service.experiment)


def main() -> None:  # pragma: no cover - CLI
    from recsys.config import get_settings
    from recsys.data import load_dataset, temporal_split
    from recsys.experiments.ab import Experiment
    from recsys.serving.models import LocalModels

    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--path")
    p.add_argument("--sessions", type=int, default=4000)
    p.add_argument("--events-db", default="sqlite:///ab_events.db")
    a = p.parse_args()
    s = get_settings()
    svc = RecommendationService.from_artifacts(
        a.artifacts, LocalModels(f"{a.artifacts}/models"), Experiment(s.experiment_name, split=s.experiment_split),
        events=EventStore(a.events_db))
    sp = temporal_split(load_dataset(a.dataset, a.path).interactions)
    seen = pd.concat([sp.train, sp.val]).groupby("user_id")["item_id"].apply(set)
    truth = {u: set(g) - seen.get(u, set()) for u, g in sp.test[sp.test.label == 1].groupby("user_id")["item_id"]}
    report = simulate(svc, {u: t for u, t in truth.items() if t}, a.sessions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
