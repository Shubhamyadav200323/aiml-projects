"""Recommendation domain service - orchestrates the online two-stage path.

    request -> cache? -> route (two-stage | content | popularity)
            -> user tower -> FAISS top-N (minus seen) -> A/B-assigned ranker -> top-K
            -> cache (TTL) -> response

Framework-free so the FastAPI layer, the A/B simulator and tests share it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.experiments.ab import EventStore, Experiment
from recsys.features.tables import FeatureTables, load_tables
from recsys.ranking.features import RankingFeaturizer
from recsys.retrieval.index import ItemIndex
from recsys.retrieval.inputs import user_inputs
from recsys.serving.cache import Cache, MemoryCache
from recsys.serving.cold_start import ColdStartPolicy, ContentRecommender, PopularityRecommender
from recsys.serving.models import ModelBackend


@dataclass
class Recommendation:
    item_id: str
    score: float
    title: str
    genres: list[str]


@dataclass
class RecResponse:
    user_id: str
    request_id: str
    strategy: str
    variant: str | None
    items: list[Recommendation]
    cached: bool
    latency_ms: float

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["items"] = [i.__dict__ for i in self.items]
        return d


@dataclass
class RecommendationService:
    tables: FeatureTables
    index: ItemIndex
    models: ModelBackend
    items_meta: pd.DataFrame
    experiment: Experiment
    cache: Cache = field(default_factory=MemoryCache)
    events: EventStore | None = None
    n_candidates: int = 300
    cache_ttl_s: int = 300
    policy: ColdStartPolicy = field(default_factory=ColdStartPolicy)

    def __post_init__(self) -> None:
        self.popularity = PopularityRecommender(self.tables)
        self.content = ContentRecommender(self.tables)
        self.featurizer = RankingFeaturizer(self.tables)
        meta = self.items_meta.set_index("item_id")
        self._titles = meta["title"].to_dict()
        self._genres = meta["genres"].map(list).to_dict()
        self.model_version = str(self.tables.as_of)

    @classmethod
    def from_artifacts(cls, directory: str | Path, models: ModelBackend, experiment: Experiment, **kw):
        d = Path(directory)
        return cls(load_tables(d / "features"), ItemIndex.load(d / "index"), models,
                   pd.read_parquet(d / "items.parquet"), experiment, **kw)

    # ------------------------------------------------------------------
    def _seen(self, user_id: str) -> set[str]:
        return self.tables.seen_items(user_id)

    def _two_stage(self, user_id: str, k: int, variant: str, exclude: set[str]) -> list[tuple[str, float]]:
        q = self.models.user_embedding(user_inputs(self.tables, [user_id]))
        cands, scores = self.index.search(q, self.n_candidates, [exclude])
        cands = cands[0]
        if not cands:
            return self.popularity.recommend(k, exclude)
        feats = self.featurizer.build(user_id, cands, scores[0][: len(cands)])
        s = self.models.rank(variant, feats)
        order = np.argsort(-s)[:k]
        return [(cands[j], float(s[j])) for j in order]

    def recommend(self, user_id: str, k: int = 10, onboarding_genres: list[str] | None = None,
                  exclude: list[str] | None = None, use_cache: bool = True) -> RecResponse:
        t0 = time.perf_counter()
        strategy = self.policy.route(self.tables, user_id, onboarding_genres)
        variant = self.experiment.assign(user_id) if strategy == "two_stage" else None
        key = f"rec:{user_id}:{self.model_version}:{variant}:{k}:{','.join(sorted(onboarding_genres or []))}"
        rid = uuid.uuid4().hex
        hit = self.cache.get(key) if use_cache and not exclude else None
        if hit is not None:
            ranked, cached = [tuple(x) for x in hit], True
        else:
            ex = self._seen(user_id) | set(exclude or [])
            if strategy == "two_stage":
                ranked = self._two_stage(user_id, k, variant, ex)
            elif strategy == "content":
                aff = self.tables.user_genre_affinity(user_id)
                if onboarding_genres:
                    aff = 0.5 * aff + 0.5 * self.content.affinity_from_genres(onboarding_genres)
                ranked = self.content.recommend(aff, k, ex)
            else:
                ranked = self.popularity.recommend(k, ex)
            cached = False
            if use_cache and not exclude:
                self.cache.set(key, [list(r) for r in ranked], self.cache_ttl_s)
        items = [Recommendation(i, round(s, 5), self._titles.get(i, i), self._genres.get(i, [])) for i, s in ranked]
        resp = RecResponse(user_id, rid, strategy, variant, items, cached,
                           round((time.perf_counter() - t0) * 1000, 3))
        if self.events is not None and variant is not None:
            self.events.log([{"experiment": self.experiment.name, "variant": variant, "user_id": user_id,
                              "item_id": it.item_id, "position": p, "event": "impression", "request_id": rid}
                             for p, it in enumerate(items)])
        return resp

    def log_click(self, user_id: str, item_id: str, position: int, request_id: str) -> str | None:
        variant = self.experiment.assign(user_id)
        if self.events is not None:
            self.events.log([{"experiment": self.experiment.name, "variant": variant, "user_id": user_id,
                              "item_id": item_id, "position": position, "event": "click",
                              "request_id": request_id}])
        return variant

    def similar_items(self, item_id: str, k: int = 10) -> list[Recommendation]:
        ids, scores = self.index.search(self.index.vector(item_id)[None, :], k, [{item_id}])
        return [Recommendation(i, round(float(s), 5), self._titles.get(i, i), self._genres.get(i, []))
                for i, s in zip(ids[0], scores[0])]
