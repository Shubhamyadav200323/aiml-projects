"""Cold-start fallbacks.

* **Popularity** - recency-weighted (30-day half-life) so trending items surface
  rather than all-time classics.
* **Content-based** - genre affinity from a user's first few positives (or from
  onboarding genre picks) matched against item genres, blended with
  popularity so niche-but-matching items still need some social proof.

`ColdStartPolicy.route` decides which path a request takes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recsys.data.movielens import GENRES
from recsys.features.tables import FeatureTables


class PopularityRecommender:
    def __init__(self, tables: FeatureTables) -> None:
        f = tables.item_features
        self.ranked = f.sort_values(["pop_recent", "pop_log"], ascending=False).index.tolist()
        self.score = dict(zip(f.index, f["pop_recent"] / max(float(f["pop_recent"].max()), 1e-9)))

    def recommend(self, k: int, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        ex = exclude or set()
        out = []
        for it in self.ranked:
            if it not in ex:
                out.append((it, float(self.score[it])))
                if len(out) == k:
                    break
        return out


class ContentRecommender:
    def __init__(self, tables: FeatureTables, pop_weight: float = 0.8) -> None:  # tuned on a held-out seed
        self.tables = tables
        self.items = np.array(tables.item_ids)
        g = tables.genre_matrix()
        self.g = g / np.maximum(g.sum(1, keepdims=True), 1.0)
        pop = tables.item_features["pop_recent"].to_numpy(np.float32)
        self.pop = pop / max(float(pop.max()), 1e-9)
        self.pop_weight = pop_weight

    def affinity_from_genres(self, genres: list[str]) -> np.ndarray:
        a = np.array([float(g in genres) for g in GENRES], np.float32)
        return a / max(a.sum(), 1.0)

    def recommend(self, affinity: np.ndarray, k: int, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        score = (1 - self.pop_weight) * (self.g @ affinity) + self.pop_weight * self.pop
        order = np.argsort(-score)
        ex = exclude or set()
        out = []
        for j in order:
            if self.items[j] not in ex:
                out.append((str(self.items[j]), float(score[j])))
                if len(out) == k:
                    break
        return out


@dataclass(frozen=True)
class ColdStartPolicy:
    min_interactions: int = 3

    def route(self, tables: FeatureTables, user_id: str, onboarding_genres: list[str] | None = None) -> str:
        uf = tables.user_features
        n_pos = float(uf.at[user_id, "n_positives"]) if user_id in uf.index else 0.0
        if n_pos >= self.min_interactions:
            return "two_stage"
        if n_pos > 0 or onboarding_genres:
            return "content"
        return "popularity"
