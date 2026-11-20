"""Dependency-free Prometheus exposition for scoring latency and decisions."""

from __future__ import annotations

import threading
from collections import Counter

_BUCKETS_MS = (1, 2, 5, 10, 20, 50, 100, 250, 500, 1000)


class ScoringMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.decisions: Counter[str] = Counter()
        self.buckets = [0] * len(_BUCKETS_MS)
        self.count = 0
        self.total_ms = 0.0
        self._recent: list[float] = []

    def observe(self, decision: str, latency_ms: float) -> None:
        with self._lock:
            self.decisions[decision] += 1
            self.count += 1
            self.total_ms += latency_ms
            for i, b in enumerate(_BUCKETS_MS):
                if latency_ms <= b:
                    self.buckets[i] += 1
            self._recent.append(latency_ms)
            if len(self._recent) > 10_000:
                self._recent = self._recent[-5_000:]

    def percentile(self, q: float) -> float:
        with self._lock:
            data = sorted(self._recent)
        return data[min(len(data) - 1, int(q * len(data)))] if data else 0.0

    def render(self) -> str:
        lines = ["# HELP fraud_decisions_total Scored transactions by decision",
                 "# TYPE fraud_decisions_total counter"]
        lines += [f'fraud_decisions_total{{decision="{d}"}} {n}' for d, n in sorted(self.decisions.items())]
        lines += ["# HELP fraud_score_latency_ms End-to-end scoring latency",
                  "# TYPE fraud_score_latency_ms histogram"]
        lines += [f'fraud_score_latency_ms_bucket{{le="{b}"}} {c}' for b, c in zip(_BUCKETS_MS, self.buckets)]
        lines += [f'fraud_score_latency_ms_bucket{{le="+Inf"}} {self.count}',
                  f"fraud_score_latency_ms_sum {self.total_ms:.3f}",
                  f"fraud_score_latency_ms_count {self.count}"]
        return "\n".join(lines) + "\n"


METRICS = ScoringMetrics()
