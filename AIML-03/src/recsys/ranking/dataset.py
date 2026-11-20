"""Ranking training data drawn from the *serving distribution*.

For each warm user in a labelling window we retrieve the top-N candidates with
the trained two-tower model (features as of the window start), and label a
candidate 1 if the user interacted positively with it during the window.
Training the ranker on exactly what retrieval will hand it at serving time
avoids the classic mismatch of ranking random negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.features.tables import FeatureTables
from recsys.ranking.features import RankingFeaturizer
from recsys.retrieval.index import ItemIndex


def seen_items(history: pd.DataFrame) -> dict[str, set[str]]:
    return history.groupby("user_id")["item_id"].apply(set).to_dict()


def build_ranking_set(window: pd.DataFrame, tables: FeatureTables, user_vectors: dict[str, np.ndarray],
                      index: ItemIndex, seen: dict[str, set[str]], n_candidates: int = 200,
                      neg_per_pos: int = 15, seed: int = 0, force_positives: bool = True) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feat = RankingFeaturizer(tables)
    truth = window[window["label"] == 1].groupby("user_id")["item_id"].apply(set)
    users = [u for u in truth.index if u in user_vectors]
    if not users:
        raise ValueError("no warm users in window")
    q = np.stack([user_vectors[u] for u in users])
    cands, scores = index.search(q, n_candidates, [seen.get(u, set()) for u in users])
    parts = []
    for u, items, sc in zip(users, cands, scores):
        rel = truth[u]
        labels = np.array([it in rel for it in items], np.float32)
        pos = np.flatnonzero(labels == 1)
        neg = np.flatnonzero(labels == 0)
        if len(pos) == 0 and not force_positives:
            continue
        keep_neg = rng.choice(neg, min(len(neg), max(1, len(pos)) * neg_per_pos), replace=False)
        keep = np.sort(np.concatenate([pos, keep_neg]))
        block = feat.build(u, items, sc[: len(items)])
        parts.append({k: v[keep] for k, v in block.items()} | {"label": labels[keep]})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
