"""MLflow tracking facade.

Training code calls `tracker.run(...)` unconditionally; when MLflow is not
installed (unit tests, minimal images) it becomes a structured no-op that still
records params/metrics in memory so callers can assert on them.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, tracking_uri: str | None = None, experiment: str = "fraud") -> None:
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        try:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
        except Exception as exc:
            log.info("mlflow disabled: %s", exc)
            self._mlflow = None

    @contextlib.contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[Tracker]:
        if self._mlflow is None:
            yield self
            return
        with self._mlflow.start_run(run_name=name, tags=tags or {}):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        if self._mlflow:
            self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        scoped = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
        self.metrics.update(scoped)
        if self._mlflow:
            self._mlflow.log_metrics(scoped)

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(str(path))
