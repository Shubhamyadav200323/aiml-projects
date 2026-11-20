"""Global temporal split.

Random splits let a model train on a user's future to predict their past,
which inflates offline metrics and hides cold-start. We cut the timeline at
two global timestamps instead: everything before t1 is train, [t1, t2) is
validation, after t2 is test. Positives are interactions rated >= `min_rating`
(implicit-feedback framing, as in production click/watch logs).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    t1: pd.Timestamp
    t2: pd.Timestamp

    def summary(self) -> dict:
        tr_users = set(self.train["user_id"])
        te_users = set(self.test["user_id"])
        return {
            "train": len(self.train), "val": len(self.val), "test": len(self.test),
            "t1": str(self.t1), "t2": str(self.t2),
            "test_users": len(te_users), "test_cold_users": len(te_users - tr_users),
        }


def temporal_split(interactions: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1,
                   min_rating: float = 4.0) -> TemporalSplit:
    df = interactions.sort_values("timestamp", kind="stable")
    t1 = df["timestamp"].quantile(1 - val_frac - test_frac)
    t2 = df["timestamp"].quantile(1 - test_frac)
    df = df.assign(label=(df["rating"] >= min_rating).astype(int))
    return TemporalSplit(df[df["timestamp"] < t1].reset_index(drop=True),
                         df[(df["timestamp"] >= t1) & (df["timestamp"] < t2)].reset_index(drop=True),
                         df[df["timestamp"] >= t2].reset_index(drop=True), t1, t2)
