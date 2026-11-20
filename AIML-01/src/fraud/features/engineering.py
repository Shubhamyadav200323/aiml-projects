"""Offline (batch) feature engineering.

All features are **point-in-time correct**: a transaction only sees events that
happened strictly before it in replay order `(timestamp, transaction_id)`, so
training never leaks future information. The implementation is fully
vectorised (O(n log n)) using a composite (user, time) key and `searchsorted`
for the sliding windows, so it scales to the 6M rows of PaySim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.features.definitions import CHANNEL_CODES, FEATURE_COLUMNS, NO_HISTORY_GAP_S, WINDOWS_S


def replay_order(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical event order shared by batch and streaming code."""
    return df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def _window_aggregates(user_codes: np.ndarray, t: np.ndarray, amount: np.ndarray,
                       window: float) -> tuple[np.ndarray, np.ndarray]:
    """Count / sum of a user's previous transactions in [t - window, t).

    Rows must be sorted by (user, t). Encoding each row as ``user*span + t`` keeps
    users in disjoint key ranges, so a single global searchsorted finds the
    left edge of every user's window at once.
    """
    span = (t.max() - t.min()) + 2 * window + 1.0
    key = user_codes * span + (t - t.min() + window)
    left = np.searchsorted(key, key - window, side="left")
    idx = np.arange(len(key))
    csum = np.concatenate([[0.0], np.cumsum(amount)])
    return (idx - left).astype(np.float64), csum[idx] - csum[left]


def _first_seen(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return (~df.duplicated(cols, keep="first")).to_numpy()


def build_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return the transactions (in replay order) with all FEATURE_COLUMNS added."""
    df = replay_order(transactions)
    df["_order"] = np.arange(len(df))
    t_sec = (df["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    df["_t"] = t_sec

    # ---- per-user, time-ordered view --------------------------------------
    by_user = df.sort_values(["user_id", "_order"], kind="stable")
    u_codes = pd.factorize(by_user["user_id"])[0].astype(np.float64)
    t = by_user["_t"].to_numpy()
    amt = by_user["amount"].to_numpy(dtype=np.float64)
    feats = pd.DataFrame(index=by_user.index)
    for name, w in WINDOWS_S.items():
        cnt, s = _window_aggregates(u_codes, t, amt, float(w))
        feats[f"txn_count_{name}"] = cnt
        feats[f"amount_sum_{name}"] = s

    g = by_user.groupby("user_id", sort=False)
    prev_count = g.cumcount().to_numpy().astype(np.float64)
    prev_sum = g["amount"].cumsum().to_numpy() - amt
    feats["user_txn_count"] = prev_count
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(prev_count > 0, prev_sum / np.maximum(prev_count, 1), 0.0)
    feats["user_avg_amount"] = avg
    feats["amount_to_avg_ratio"] = np.where(avg > 0, amt / np.where(avg > 0, avg, 1.0), 1.0)
    prev_t = g["_t"].shift(1).to_numpy()
    feats["seconds_since_last_txn"] = np.where(np.isnan(prev_t), NO_HISTORY_GAP_S, t - prev_t)

    prev_device = g["device_id"].shift(1)
    prev_loc = g["location"].shift(1)
    feats["device_changed"] = (prev_device.notna() & (prev_device != by_user["device_id"])).astype(int)
    feats["location_changed"] = (prev_loc.notna() & (prev_loc != by_user["location"])).astype(int)
    has_history = prev_count > 0
    for col, name in [("device_id", "is_new_device"), ("location", "is_new_location"),
                      ("card_id", "is_new_card")]:
        feats[name] = (_first_seen(by_user, ["user_id", col]) & has_history).astype(int)

    df = df.join(feats)

    # ---- entity sharing (global time order; includes current txn) ----------
    for col, name in [("device_id", "device_user_count"), ("card_id", "card_user_count")]:
        first_pair = pd.Series(_first_seen(df, [col, "user_id"]).astype(int), index=df.index)
        df[name] = first_pair.groupby(df[col], sort=False).cumsum().astype(np.float64)

    df["amount_log"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["timestamp"].dt.hour.astype(np.float64)
    df["channel_code"] = df["channel"].map(CHANNEL_CODES).fillna(-1).astype(np.float64)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(np.float64)
    return df.drop(columns=["_order", "_t"])
