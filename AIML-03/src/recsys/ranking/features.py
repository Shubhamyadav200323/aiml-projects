"""Ranking features for (user, candidate) pairs - shared by training and serving."""

from __future__ import annotations

import numpy as np

from recsys.data.movielens import GENRES
from recsys.features.tables import FeatureTables

RANK_DENSE_NAMES = (
    ["retrieval_score", "retrieval_rank_log", "genre_match", "pop_log", "pop_recent", "item_avg_rating",
     "user_npos_log", "user_avg_rating"]
    + [f"aff_{g}" for g in GENRES] + [f"g_{g}" for g in GENRES]
)
RANK_DENSE_DIM = len(RANK_DENSE_NAMES)


class RankingFeaturizer:
    """Precomputes per-item and per-user blocks once; pairs are gathered by index."""

    def __init__(self, tables: FeatureTables) -> None:
        self.tables = tables
        f = tables.item_features
        self.item_pos = {it: i for i, it in enumerate(f.index)}
        self.item_genres = tables.genre_matrix()
        self.item_block = np.stack([
            f["pop_log"].to_numpy() / max(float(f["pop_log"].max()), 1e-6),
            f["pop_recent"].to_numpy() / max(float(f["pop_recent"].max()), 1e-6),
            f["item_avg_rating"].to_numpy() / 5.0,
        ], 1).astype(np.float32)

    def user_block(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        uf = self.tables.user_features
        aff = self.tables.user_genre_affinity(user_id)
        if user_id in uf.index:
            stats = np.array([np.log1p(uf.at[user_id, "n_positives"]), uf.at[user_id, "avg_rating"] / 5.0],
                             np.float32)
        else:
            stats = np.zeros(2, np.float32)
        return aff, stats

    def build(self, user_id: str, items: list[str], retrieval_scores: np.ndarray) -> dict[str, np.ndarray]:
        n = len(items)
        aff, stats = self.user_block(user_id)
        rows = np.array([self.item_pos[i] for i in items], dtype=np.int64)
        g = self.item_genres[rows]
        match = (g @ aff) / np.maximum(g.sum(1), 1.0)
        dense = np.hstack([
            np.asarray(retrieval_scores, np.float32)[:, None],
            np.log1p(np.arange(n, dtype=np.float32))[:, None] / np.log1p(max(n, 1)),
            match[:, None].astype(np.float32),
            self.item_block[rows],
            np.tile(stats, (n, 1)),
            np.tile(aff, (n, 1)),
            g,
        ]).astype(np.float32)
        return {
            "user_idx": np.full(n, self.tables.user_index.get(user_id, 0), np.int64),
            "item_idx": np.array([self.tables.item_index.get(i, 0) for i in items], np.int64),
            "dense": dense,
        }
