"""In-process latency benchmark of the scoring path (features + XGB + blend + decision).

    python scripts/benchmark.py --artifacts artifacts --n 2000
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from fraud.data import generate_transactions
from fraud.features.engineering import replay_order
from fraud.features.online import OnlineFeatureEngine
from fraud.serving.scorer import FraudScorer, ModelBundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--n", type=int, default=2000)
    a = p.parse_args()
    scorer = FraudScorer(ModelBundle.load(a.artifacts), OnlineFeatureEngine(), cases=None)
    events = replay_order(generate_transactions()).head(a.n).to_dict("records")
    lat = []
    t0 = time.perf_counter()
    for e in events:
        s = time.perf_counter()
        scorer.score(e)
        lat.append((time.perf_counter() - s) * 1000)
    wall = time.perf_counter() - t0
    out = {"n": a.n, "throughput_tps_single_core": round(a.n / wall, 1),
           "p50_ms": round(float(np.percentile(lat, 50)), 3),
           "p95_ms": round(float(np.percentile(lat, 95)), 3),
           "p99_ms": round(float(np.percentile(lat, 99)), 3)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
