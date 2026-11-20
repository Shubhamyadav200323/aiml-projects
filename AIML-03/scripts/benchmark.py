"""End-to-end online latency of the recommendation service (cache disabled).

    python scripts/benchmark.py --artifacts artifacts --n 500
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np  # noqa: E402

from recsys.experiments.ab import Experiment  # noqa: E402
from recsys.serving.models import LocalModels  # noqa: E402
from recsys.serving.service import RecommendationService  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--n", type=int, default=500)
    a = p.parse_args()
    svc = RecommendationService.from_artifacts(a.artifacts, LocalModels(f"{a.artifacts}/models"), Experiment("bench"))
    users = [u for u in svc.tables.user_features.index if svc.policy.route(svc.tables, u) == "two_stage"]
    rng = np.random.default_rng(0)
    for u in users[:20]:  # warm-up (graph tracing)
        svc.recommend(u, use_cache=False)
    lat = []
    for u in rng.choice(users, a.n):
        t = time.perf_counter()
        svc.recommend(str(u), k=10, use_cache=False)
        lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    for u in users[:200]:
        svc.recommend(u, k=10)
    cached = [(time.perf_counter(), svc.recommend(u, k=10))[0] for u in users[:200]]
    hit_ms = (time.perf_counter() - cached[0]) / 200 * 1000
    print(json.dumps({"requests": a.n, "candidates": svc.n_candidates,
                      "p50_ms": round(float(np.percentile(lat, 50)), 2),
                      "p95_ms": round(float(np.percentile(lat, 95)), 2),
                      "p99_ms": round(float(np.percentile(lat, 99)), 2),
                      "cache_hit_ms": round(hit_ms, 3)}, indent=2))


if __name__ == "__main__":
    main()
