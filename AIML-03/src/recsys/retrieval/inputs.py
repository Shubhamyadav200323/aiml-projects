"""Model inputs, built identically for training and serving.

Training examples are **point-in-time**: the user features attached to an
interaction (genre affinity, positive count, history) are computed only from
that user's earlier positives, so the model never sees the target item in its
own history. Serving uses the same encoders on the `FeatureTables` snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.data.movielens import GENRES
from recsys.features.tables import HISTORY_LEN, FeatureTables

USER_DENSE_DIM = len(GENRES) + 1   # affinity + log1p(n_positives)
ITEM_DENSE_DIM = len(GENRES) + 3   # genres + pop_log + pop_recent + avg_rating


def item_inputs(tables: FeatureTables) -> dict[str, np.ndarray]:
    f = tables.item_features
    pop_max = max(float(f["pop_log"].max()), 1e-6)
    rec_max = max(float(f["pop_recent"].max()), 1e-6)
    dense = np.hstack([
        tables.genre_matrix(),
        (f["pop_log"].to_numpy() / pop_max)[:, None],
        (f["pop_recent"].to_numpy() / rec_max)[:, None],
        (f["item_avg_rating"].to_numpy() / 5.0)[:, None],
    ]).astype(np.float32)
    idx = np.array([tables.item_index.get(i, 0) for i in f.index], dtype=np.int64)
    return {"item_idx": idx, "item_dense": dense}


def _pad(hist: list[int]) -> list[int]:
    hist = hist[-HISTORY_LEN:]
    return [0] * (HISTORY_LEN - len(hist)) + hist


def user_inputs(tables: FeatureTables, user_ids: list[str]) -> dict[str, np.ndarray]:
    uf = tables.user_features
    aff_cols = [f"aff_{g}" for g in GENRES]
    known = [u in uf.index for u in user_ids]
    aff = np.zeros((len(user_ids), len(GENRES)), np.float32)
    npos = np.zeros(len(user_ids), np.float32)
    hist = np.zeros((len(user_ids), HISTORY_LEN), np.int64)
    sub = uf.reindex([u for u, k in zip(user_ids, known) if k])
    rows = np.where(known)[0]
    if len(rows):
        aff[rows] = sub[aff_cols].to_numpy(np.float32)
        npos[rows] = sub["n_positives"].to_numpy(np.float32)
        for r, h in zip(rows, sub["history"]):
            hist[r] = _pad([tables.item_index.get(i, 0) for i in h])
    return {
        "user_idx": np.array([tables.user_index.get(u, 0) for u in user_ids], dtype=np.int64),
        "user_dense": np.hstack([aff, np.log1p(npos)[:, None]]).astype(np.float32),
        "history": hist,
    }


def retrieval_examples(train: pd.DataFrame, tables: FeatureTables) -> dict[str, np.ndarray]:
    """One example per positive interaction with point-in-time user features."""
    pos = train[train["label"] == 1].sort_values(["user_id", "timestamp"], kind="stable")
    pos = pos[pos["item_id"].isin(tables.item_index)]
    genres = tables.item_features.loc[pos["item_id"], [f"g_{g}" for g in GENRES]].to_numpy(np.float32)
    g = pd.DataFrame(genres, index=pos.index).groupby(pos["user_id"].to_numpy())
    prior = g.cumsum().to_numpy() - genres                 # genres of earlier positives only
    aff = prior / np.maximum(prior.sum(1, keepdims=True), 1.0)
    npos = pos.groupby("user_id").cumcount().to_numpy().astype(np.float32)

    item_idx = pos["item_id"].map(tables.item_index).to_numpy(np.int64)
    hist = np.zeros((len(pos), HISTORY_LEN), np.int64)
    users = pos["user_id"].to_numpy()
    start = 0
    for end in np.r_[np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]:
        seq = item_idx[start:end]
        for j in range(end - start):
            h = seq[max(0, j - HISTORY_LEN):j]
            hist[start + j, HISTORY_LEN - len(h):] = h
        start = end

    counts = pos["item_id"].value_counts()
    prob = (pos["item_id"].map(counts) / len(pos)).to_numpy(np.float32)  # logQ correction
    return {
        "user_idx": pos["user_id"].map(tables.user_index).fillna(0).to_numpy(np.int64),
        "user_dense": np.hstack([aff, np.log1p(npos)[:, None]]).astype(np.float32),
        "history": hist,
        "item_idx": item_idx,
        "item_dense": item_inputs_by_idx(tables, item_idx),
        "item_prob": prob,
    }


def item_inputs_by_idx(tables: FeatureTables, item_idx: np.ndarray) -> np.ndarray:
    inp = item_inputs(tables)
    row_of = np.zeros(int(inp["item_idx"].max()) + 1, np.int64)
    row_of[inp["item_idx"]] = np.arange(len(inp["item_idx"]))
    return inp["item_dense"][row_of[item_idx]]
