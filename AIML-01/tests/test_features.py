"""Training/serving parity and leakage guards for the feature pipeline."""

import numpy as np
import pandas as pd
import pytest

from fraud.features import FEATURE_COLUMNS, OnlineFeatureEngine, build_features
from fraud.features.engineering import replay_order


@pytest.fixture(scope="module")
def offline(transactions):
    return build_features(transactions)


def test_all_features_present_and_finite(offline):
    assert set(FEATURE_COLUMNS) <= set(offline.columns)
    assert np.isfinite(offline[FEATURE_COLUMNS].to_numpy()).all()


def test_online_matches_offline_exactly(transactions, offline):
    """The streaming engine must reproduce the batch features (no training/serving skew)."""
    engine = OnlineFeatureEngine()
    stream = replay_order(transactions).head(4000)
    online = pd.DataFrame([engine.compute(r) for r in stream.to_dict("records")])[FEATURE_COLUMNS]
    batch = offline.head(4000)[FEATURE_COLUMNS].reset_index(drop=True)
    np.testing.assert_allclose(online.to_numpy(), batch.to_numpy(), rtol=1e-6, atol=1e-4)


def test_no_future_leakage(transactions):
    """Appending future events must not change features of past events."""
    cut = len(transactions) // 2
    early = build_features(transactions.iloc[:cut])
    full = build_features(transactions).set_index("transaction_id").loc[early["transaction_id"]]
    np.testing.assert_allclose(early[FEATURE_COLUMNS].to_numpy(),
                               full[FEATURE_COLUMNS].to_numpy(), rtol=1e-9, atol=1e-6)


def test_velocity_window_semantics():
    base = pd.Timestamp("2026-01-01")
    tx = pd.DataFrame({
        "transaction_id": ["a", "b", "c", "d"],
        "timestamp": [base, base + pd.Timedelta("10min"), base + pd.Timedelta("50min"),
                      base + pd.Timedelta("2h")],
        "user_id": "U1", "card_id": "C1", "device_id": ["D1", "D1", "D2", "D2"],
        "merchant_id": "M", "location": "L", "amount": [10.0, 20.0, 30.0, 40.0],
        "channel": "ONLINE", "is_fraud": 0,
    })
    f = build_features(tx).set_index("transaction_id")
    assert f.loc["c", "txn_count_1h"] == 2 and f.loc["c", "amount_sum_1h"] == 30
    assert f.loc["d", "txn_count_1h"] == 0 and f.loc["d", "txn_count_24h"] == 3
    assert f.loc["c", "is_new_device"] == 1 and f.loc["d", "is_new_device"] == 0
    assert f.loc["d", "user_avg_amount"] == pytest.approx(20.0)
    assert f.loc["d", "amount_to_avg_ratio"] == pytest.approx(2.0)
