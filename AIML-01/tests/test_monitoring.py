import json

import numpy as np
import pandas as pd

from fraud.features.definitions import FEATURE_COLUMNS
from fraud.monitoring.drift import psi, psi_report
from fraud.monitoring.retrain import RetrainPolicy


def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.02
    assert psi(ref, rng.normal(1.0, 1, 20_000)) > 0.2


def test_retrain_triggered_by_drift_not_by_noise(artifacts):
    out, _ = artifacts
    ref = pd.read_parquet(out / "reference_features.parquet")
    same = ref.sample(frac=0.5, random_state=1)
    policy = RetrainPolicy()
    assert not policy.evaluate(ref, same).retrain

    shifted = same.copy()
    for c in FEATURE_COLUMNS[: len(FEATURE_COLUMNS) // 2]:
        shifted[c] = shifted[c] * 3 + 5
    decision = policy.evaluate(ref, shifted)
    assert decision.retrain and decision.reasons[0].startswith("drift")
    assert psi_report(ref, shifted, FEATURE_COLUMNS).share_drifted >= 0.3


def test_label_volume_and_promotion_rules(artifacts):
    out, _ = artifacts
    policy = RetrainPolicy(min_new_labels=10)
    ref = pd.read_parquet(out / "reference_features.parquet")
    assert policy.evaluate(ref, ref, new_labels=10).retrain
    champ = json.loads((out / "metrics.json").read_text())
    worse = json.loads(json.dumps(champ))
    worse["test"]["blend"]["pr_auc"] -= 0.2
    assert policy.should_promote(champ, champ)
    assert not policy.should_promote(champ, worse)
