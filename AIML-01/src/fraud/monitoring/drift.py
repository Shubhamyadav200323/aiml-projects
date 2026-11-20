"""Data-drift detection.

* `psi_report` - Population Stability Index per feature, always available and
  cheap enough to run every few minutes on a sliding window.
* `evidently_report` - full Evidently AI `DataDriftPreset` HTML report for
  analysts (optional dependency, run by the nightly CronJob).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins from the reference. <0.1 stable, 0.1-0.2 watch, >0.2 drift."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # constant feature: drift iff current differs from it
        return 0.0 if np.allclose(current, reference[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / max(len(current), 1)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


@dataclass
class DriftReport:
    per_feature: dict[str, float]
    drifted: list[str]
    share_drifted: float
    threshold: float

    @property
    def dataset_drift(self) -> bool:
        return bool(self.drifted)

    def to_dict(self) -> dict:
        return {"per_feature_psi": self.per_feature, "drifted": self.drifted,
                "share_drifted": self.share_drifted, "psi_threshold": self.threshold}


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str],
               threshold: float = 0.2) -> DriftReport:
    scores = {c: round(psi(reference[c].to_numpy(), current[c].to_numpy()), 4) for c in columns}
    drifted = [c for c, v in scores.items() if v > threshold]
    return DriftReport(scores, drifted, len(drifted) / max(len(columns), 1), threshold)


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame,
                     out_html: str | Path) -> dict | None:  # pragma: no cover - optional dep
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except Exception as exc:
        log.info("evidently not installed (%s); skipping HTML report", exc)
        return None
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(out_html))
    return report.as_dict()
