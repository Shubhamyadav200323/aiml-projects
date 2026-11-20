"""Thin broker abstraction so producer/consumer logic is testable without Kafka.

`KafkaBroker` wraps confluent-kafka; `MemoryBroker` is an in-process queue
with the same interface used by unit tests and `--dry-run`.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any, Protocol


class Broker(Protocol):
    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def consume(self, topic: str, timeout: float = 1.0) -> Iterator[dict[str, Any]]: ...
    def commit(self) -> None: ...


class MemoryBroker:
    def __init__(self) -> None:
        self.topics: dict[str, deque] = defaultdict(deque)
        self.committed = 0

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None:
        self.topics[topic].append((key, json.loads(json.dumps(value, default=str))))

    def flush(self) -> None:
        pass

    def consume(self, topic: str, timeout: float = 1.0) -> Iterator[dict[str, Any]]:
        q = self.topics[topic]
        while q:
            yield q.popleft()[1]

    def commit(self) -> None:
        self.committed += 1


class KafkaBroker:  # pragma: no cover - requires a running broker
    def __init__(self, bootstrap: str, group_id: str = "fraud-scorer") -> None:
        from confluent_kafka import Consumer, Producer

        self._producer = Producer({
            "bootstrap.servers": bootstrap, "linger.ms": 5, "enable.idempotence": True,
            "compression.type": "lz4", "acks": "all"})
        self._consumer_conf = {
            "bootstrap.servers": bootstrap, "group.id": group_id, "auto.offset.reset": "earliest",
            "enable.auto.commit": False}
        self._consumer = None
        self._Consumer = Consumer

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None:
        self._producer.produce(topic, key=key.encode(), value=json.dumps(value, default=str).encode())
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush()

    def consume(self, topic: str, timeout: float = 1.0) -> Iterator[dict[str, Any]]:
        if self._consumer is None:
            self._consumer = self._Consumer(self._consumer_conf)
            self._consumer.subscribe([topic])
        while True:
            msg = self._consumer.poll(timeout)
            if msg is None:
                continue
            if msg.error():
                raise RuntimeError(msg.error())
            yield json.loads(msg.value())

    def commit(self) -> None:
        if self._consumer is not None:
            self._consumer.commit(asynchronous=False)
