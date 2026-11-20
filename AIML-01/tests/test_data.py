import pandas as pd

from fraud.data.loaders import TRANSACTION_COLUMNS, load_paysim


def test_synthetic_schema_and_imbalance(transactions):
    assert list(transactions.columns) == TRANSACTION_COLUMNS
    assert transactions["transaction_id"].is_unique
    rate = transactions["is_fraud"].mean()
    assert 0.001 < rate < 0.02, "fraud must be a rare class (<2%)"
    assert transactions["timestamp"].is_monotonic_increasing


def test_rings_share_devices(transactions):
    fraud = transactions[transactions.is_fraud == 1]
    users_per_device = fraud.groupby("device_id")["user_id"].nunique()
    assert (users_per_device >= 3).any(), "planted rings should share devices"


def test_paysim_adapter(tmp_path):
    raw = pd.DataFrame({
        "step": [1, 2], "type": ["TRANSFER", "PAYMENT"], "amount": [100.0, 5.0],
        "nameOrig": ["C1", "C2"], "nameDest": ["C9", "M3"], "isFraud": [1, 0],
    })
    p = tmp_path / "paysim.csv"
    raw.to_csv(p, index=False)
    df = load_paysim(p)
    assert list(df.columns) == TRANSACTION_COLUMNS
    assert df["is_fraud"].tolist() == [1, 0]
