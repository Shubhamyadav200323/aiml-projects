"""Automatic retraining trigger with champion/challenger promotion.

    python -m fraud.monitoring.retrain --current recent_features.parquet --dataset synthetic

Policy (all thresholds env-configurable):
  1. data drift: share of features with PSI > PSI_THRESHOLD >= DRIFT_SHARE_TRIGGER
  2. performance decay: PR-AUC on analyst-labelled cases below `min_pr_auc`
  3. label volume: enough new analyst labels accumulated since last training

A challenger trained on fresh data is promoted only if its test PR-AUC is not
worse than the champion's by more than `tolerance`, so a noisy retrain can
never silently degrade production.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fraud.config import get_settings
from fraud.features.definitions import FEATURE_COLUMNS
from fraud.models.metrics import evaluate
from fraud.monitoring.drift import DriftReport, psi_report

log = logging.getLogger(__name__)


@dataclass
class RetrainDecision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)
    drift: DriftReport | None = None


@dataclass
class RetrainPolicy:
    drift_share_trigger: float = 0.3
    psi_threshold: float = 0.2
    min_pr_auc: float = 0.5
    min_new_labels: int = 500
    tolerance: float = 0.01

    def evaluate(self, reference: pd.DataFrame, current: pd.DataFrame,
                 feedback: pd.DataFrame | None = None, new_labels: int = 0) -> RetrainDecision:
        drift = psi_report(reference, current, FEATURE_COLUMNS, self.psi_threshold)
        reasons = []
        if drift.share_drifted >= self.drift_share_trigger:
            reasons.append(f"drift: {drift.share_drifted:.0%} of features drifted {drift.drifted}")
        if feedback is not None and len(feedback) >= 50 and feedback["label"].nunique() == 2:
            pr = evaluate(feedback["label"].to_numpy(), feedback["fraud_score"].to_numpy())["pr_auc"]
            if pr < self.min_pr_auc:
                reasons.append(f"performance: feedback PR-AUC {pr:.3f} < {self.min_pr_auc}")
        if new_labels >= self.min_new_labels:
            reasons.append(f"labels: {new_labels} new analyst labels")
        return RetrainDecision(bool(reasons), reasons, drift)

    def should_promote(self, champion: dict, challenger: dict) -> bool:
        return challenger["test"]["blend"]["pr_auc"] >= champion["test"]["blend"]["pr_auc"] - self.tolerance


def retrain_and_maybe_promote(transactions: pd.DataFrame, champion_dir: Path,
                              policy: RetrainPolicy, **train_kwargs) -> dict:
    from fraud.pipeline import train

    challenger_dir = champion_dir.parent / f"{champion_dir.name}-challenger"
    challenger = train(transactions, challenger_dir, **train_kwargs)
    champ_metrics = champion_dir / "metrics.json"
    champion = json.loads(champ_metrics.read_text()) if champ_metrics.exists() else None
    promote = champion is None or policy.should_promote(champion, challenger)
    if promote:
        backup = champion_dir.parent / f"{champion_dir.name}-previous"
        if champion_dir.exists():
            shutil.rmtree(backup, ignore_errors=True)
            champion_dir.rename(backup)
        challenger_dir.rename(champion_dir)
    return {"promoted": promote, "challenger_pr_auc": challenger["test"]["blend"]["pr_auc"],
            "champion_pr_auc": champion["test"]["blend"]["pr_auc"] if champion else None}


def main() -> None:  # pragma: no cover - CLI / CronJob entrypoint
    from fraud.data import load_dataset

    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--current", required=True, help="parquet of recent production features")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--path")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    s = get_settings()
    policy = RetrainPolicy(s.drift_share_trigger, s.psi_threshold)
    ref = pd.read_parquet(s.artifacts_dir / "reference_features.parquet")
    decision = policy.evaluate(ref, pd.read_parquet(a.current))
    print(json.dumps({"retrain": decision.retrain, "reasons": decision.reasons,
                      "drift": decision.drift.to_dict() if decision.drift else None}, indent=2))
    if decision.retrain or a.force:
        print(json.dumps(retrain_and_maybe_promote(load_dataset(a.dataset, a.path),
                                                   Path(s.artifacts_dir), policy), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
