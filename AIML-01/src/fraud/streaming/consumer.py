"""Streaming scorer: consume `transactions`, score, publish to `fraud-scores`.

    python -m fraud.streaming.consumer

Scores in-process (no HTTP hop) using the same `FraudScorer` as the API, and
commits offsets in micro-batches *after* results are produced, giving
at-least-once delivery. Duplicate deliveries are harmless because case
creation is idempotent on `transaction_id`.
"""

from __future__ import annotations

import logging
import signal
from collections.abc import Callable

from fraud.serving.scorer import FraudScorer
from fraud.streaming.kafka_io import Broker

log = logging.getLogger(__name__)


def run(scorer: FraudScorer, broker: Broker, in_topic: str, out_topic: str,
        commit_every: int = 100, max_messages: int | None = None,
        should_stop: Callable[[], bool] = lambda: False) -> int:
    n = 0
    for msg in broker.consume(in_topic):
        try:
            result = scorer.score(msg)
            broker.produce(out_topic, result.transaction_id, result.as_dict())
        except Exception:  # poison message -> dead-letter, keep the stream flowing
            log.exception("failed to score %s", msg.get("transaction_id"))
            broker.produce(f"{in_topic}.dlq", str(msg.get("transaction_id")), msg)
        n += 1
        if n % commit_every == 0:
            broker.flush()
            broker.commit()
        if should_stop() or (max_messages and n >= max_messages):
            break
    broker.flush()
    broker.commit()
    return n


def main() -> None:  # pragma: no cover - CLI
    from fraud.config import get_settings
    from fraud.features.store import build_online_engine
    from fraud.repository.cases import CaseRepository
    from fraud.serving.scorer import ModelBundle
    from fraud.streaming.kafka_io import KafkaBroker

    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    scorer = FraudScorer(ModelBundle.load(s.artifacts_dir), build_online_engine(s),
                         CaseRepository(s.database_url))
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    n = run(scorer, KafkaBroker(s.kafka_bootstrap), s.topic_transactions, s.topic_scores,
            should_stop=lambda: stop["flag"])
    log.info("consumer stopped after %d messages", n)


if __name__ == "__main__":  # pragma: no cover
    main()
