"""Offline ranking metrics (binary relevance)."""

from __future__ import annotations

import numpy as np


def recall_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recs: list[str], relevant: set[str], k: int) -> float:
    gains = [1.0 / np.log2(i + 2) for i, it in enumerate(recs[:k]) if it in relevant]
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(sum(gains) / ideal) if ideal else 0.0


def mrr(recs: list[str], relevant: set[str]) -> float:
    for i, it in enumerate(recs):
        if it in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_lists(all_recs: dict[str, list[str]], truth: dict[str, set[str]],
                   ks: tuple[int, ...] = (10, 50, 100), catalogue_size: int | None = None) -> dict[str, float]:
    users = [u for u in truth if u in all_recs and truth[u]]
    out: dict[str, float] = {"users": float(len(users))}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([recall_at_k(all_recs[u], truth[u], k) for u in users])) if users else 0.0
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(all_recs[u], truth[u], k) for u in users])) if users else 0.0
        out[f"hit@{k}"] = float(np.mean([bool(set(all_recs[u][:k]) & truth[u]) for u in users])) if users else 0.0
    out["mrr"] = float(np.mean([mrr(all_recs[u], truth[u]) for u in users])) if users else 0.0
    if catalogue_size:
        shown = {it for u in users for it in all_recs[u][:10]}
        out["coverage@10"] = len(shown) / catalogue_size
    return out
