"""Dataset adapters that map public fraud datasets onto one canonical schema.

Canonical schema (one row per transaction)::

    transaction_id  str        unique id
    timestamp       datetime   event time
    user_id         str        paying account
    card_id         str        payment instrument
    device_id       str        device fingerprint
    merchant_id     str        counter-party
    location        str        coarse geo bucket
    amount          float
    channel         str
    is_fraud        int {0,1}
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TRANSACTION_COLUMNS = [
    "transaction_id",
    "timestamp",
    "user_id",
    "card_id",
    "device_id",
    "merchant_id",
    "location",
    "amount",
    "channel",
    "is_fraud",
]

_EPOCH = pd.Timestamp("2026-01-01")


def load_paysim(path: str | Path) -> pd.DataFrame:
    """PaySim (Kaggle: ealaxi/paysim1). `step` is one simulated hour."""
    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "transaction_id": "P" + raw.index.astype(str),
            "timestamp": _EPOCH + pd.to_timedelta(raw["step"], unit="h"),
            "user_id": raw["nameOrig"],
            # PaySim has no card/device: the origin account is the instrument and
            # the destination prefix (C/M) distinguishes customer vs merchant.
            "card_id": raw["nameOrig"],
            "device_id": "D" + raw["nameOrig"].str[1:],
            "merchant_id": raw["nameDest"],
            "location": "L0",
            "amount": raw["amount"].astype(float),
            "channel": raw["type"],
            "is_fraud": raw["isFraud"].astype(int),
        }
    )
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_ieee_cis(transaction_csv: str | Path, identity_csv: str | Path | None = None) -> pd.DataFrame:
    """IEEE-CIS (Kaggle: ieee-fraud-detection). Users are approximated by card1+addr1,
    the standard community heuristic for this anonymised dataset."""
    raw = pd.read_csv(transaction_csv)
    if identity_csv is not None:
        raw = raw.merge(pd.read_csv(identity_csv), on="TransactionID", how="left")
    device = raw["DeviceInfo"] if "DeviceInfo" in raw else pd.Series("unknown", index=raw.index)
    df = pd.DataFrame(
        {
            "transaction_id": raw["TransactionID"].astype(str),
            "timestamp": _EPOCH + pd.to_timedelta(raw["TransactionDT"], unit="s"),
            "user_id": raw["card1"].astype(str) + "_" + raw["addr1"].fillna(-1).astype(int).astype(str),
            "card_id": raw["card1"].astype(str) + "_" + raw["card2"].fillna(-1).astype(int).astype(str),
            "device_id": device.fillna("unknown").astype(str),
            "merchant_id": raw["ProductCD"].astype(str),
            "location": raw["addr2"].fillna(-1).astype(int).astype(str),
            "amount": raw["TransactionAmt"].astype(float),
            "channel": raw["ProductCD"].astype(str),
            "is_fraud": raw["isFraud"].astype(int),
        }
    )
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_dataset(name: str, path: str | Path | None = None, **kwargs) -> pd.DataFrame:
    """Single entry point used by the pipeline CLI."""
    name = name.lower()
    if name == "synthetic":
        from fraud.data.synthetic import SyntheticConfig, generate_transactions

        return generate_transactions(SyntheticConfig(**kwargs))
    if path is None:
        raise ValueError(f"dataset '{name}' requires a path")
    if name == "paysim":
        return load_paysim(path)
    if name in {"ieee", "ieee-cis"}:
        return load_ieee_cis(path, kwargs.get("identity_csv"))
    raise ValueError(f"unknown dataset: {name}")
