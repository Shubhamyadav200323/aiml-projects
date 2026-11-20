"""Replay a historical dataset onto Kafka in event-time order.

    python -m fraud.streaming.producer --dataset synthetic --speedup 3600 --limit 50000

`--speedup 3600` replays one hour of history per second; `--speedup 0` sends as
fast as possible (load testing). Messages are keyed by `user_id` so all events
of a user land on one partition, preserving per-user ordering - which the
stateful feature engine relies on.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from fraud.features.engineering import replay_order
from fraud.streaming.kafka_io import Broker

log = logging.getLogger(__name__)


def replay(df: pd.DataFrame, broker: Broker, topic: str, speedup: float = 0.0,
           limit: int | None = None, include_label: bool = False,
           sleep=time.sleep) -> int:
    events = replay_order(df)
    if limit:
        events = events.head(limit)
    if not include_label:
        events = events.drop(columns=["is_fraud"])
    t_prev = None
    sent = 0
    for rec in events.to_dict("records"):
        ts = rec["timestamp"]
        if speedup > 0 and t_prev is not None:
            gap = (ts - t_prev).total_seconds() / speedup
            if gap > 0:
                sleep(min(gap, 5.0))
        t_prev = ts
        rec["timestamp"] = pd.Timestamp(ts).isoformat()
        broker.produce(topic, str(rec["user_id"]), rec)
        sent += 1
        if sent % 10_000 == 0:
            log.info("replayed %d events", sent)
    broker.flush()
    return sent


def main() -> None:  # pragma: no cover - CLI
    from fraud.config import get_settings
    from fraud.data import load_dataset
    from fraud.streaming.kafka_io import KafkaBroker

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--path")
    p.add_argument("--speedup", type=float, default=3600)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    s = get_settings()
    n = replay(load_dataset(a.dataset, a.path), KafkaBroker(s.kafka_bootstrap),
               s.topic_transactions, a.speedup, a.limit)
    log.info("done: %d events -> %s", n, s.topic_transactions)


if __name__ == "__main__":  # pragma: no cover
    main()
