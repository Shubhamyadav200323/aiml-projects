"""FastAPI gateway (application tier).

    uvicorn recsys.serving.api:app --port 8100       # docs at /docs
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from recsys.config import Settings, get_settings
from recsys.experiments.ab import EventStore, Experiment
from recsys.serving.cache import build_cache
from recsys.serving.service import RecommendationService

log = logging.getLogger(__name__)


class Item(BaseModel):
    item_id: str
    score: float
    title: str
    genres: list[str]


class RecommendationsOut(BaseModel):
    user_id: str
    request_id: str
    strategy: str = Field(description="two_stage | content | popularity")
    variant: str | None = Field(description="A/B ranker variant (two_stage only)")
    items: list[Item]
    cached: bool
    latency_ms: float


class ClickIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    item_id: str = Field(min_length=1, max_length=64)
    position: int = Field(ge=0, le=500)
    request_id: str = Field(min_length=1, max_length=64)


class _Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.by_strategy: Counter[str] = Counter()
        self.latencies: list[float] = []
        self.cache_hits = 0

    def observe(self, strategy: str, ms: float, cached: bool) -> None:
        with self.lock:
            self.by_strategy[strategy] += 1
            self.cache_hits += int(cached)
            self.latencies = (self.latencies + [ms])[-5000:]

    def pct(self, q: float) -> float:
        with self.lock:
            d = sorted(self.latencies)
        return d[min(len(d) - 1, int(q * len(d)))] if d else 0.0


def create_app(service: RecommendationService | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    metrics = _Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.svc = service
        else:
            if settings.triton_url:
                from recsys.serving.models import TritonModels

                models = TritonModels(settings.triton_url)
            else:
                from recsys.serving.models import LocalModels

                models = LocalModels(settings.artifacts_dir / "models")
            app.state.svc = RecommendationService.from_artifacts(
                settings.artifacts_dir, models, Experiment(settings.experiment_name, split=settings.experiment_split),
                cache=build_cache(settings.redis_url), events=EventStore(settings.events_db),
                n_candidates=settings.n_candidates, cache_ttl_s=settings.cache_ttl_s)
        yield

    app = FastAPI(title="Recommendation API", version="1.0.0", lifespan=lifespan,
                  description="Two-stage recommender: two-tower + FAISS retrieval, DCN-v2 / Wide&Deep ranking.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def svc(request: Request) -> RecommendationService:
        return request.app.state.svc

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(s: RecommendationService = Depends(svc)) -> dict:
        return {"status": "ready", "features_as_of": s.model_version, "index": s.index.backend,
                "items": len(s.index.item_ids), "p50_ms": metrics.pct(0.5), "p99_ms": metrics.pct(0.99)}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def prom() -> str:
        lines = ["# TYPE recsys_requests_total counter"]
        lines += [f'recsys_requests_total{{strategy="{k}"}} {v}' for k, v in sorted(metrics.by_strategy.items())]
        lines += ["# TYPE recsys_cache_hits_total counter", f"recsys_cache_hits_total {metrics.cache_hits}",
                  "# TYPE recsys_latency_ms summary",
                  f'recsys_latency_ms{{quantile="0.5"}} {metrics.pct(0.5):.3f}',
                  f'recsys_latency_ms{{quantile="0.99"}} {metrics.pct(0.99):.3f}']
        return "\n".join(lines) + "\n"

    @app.get("/v1/recommendations/{user_id}", response_model=RecommendationsOut, tags=["recommendations"])
    def recommend(user_id: str, k: int = Query(10, ge=1, le=100),
                  genres: list[str] | None = Query(None, description="onboarding genres for new users"),
                  s: RecommendationService = Depends(svc)) -> dict:
        t = time.perf_counter()
        resp = s.recommend(user_id, k=k, onboarding_genres=genres)
        metrics.observe(resp.strategy, (time.perf_counter() - t) * 1000, resp.cached)
        return resp.as_dict()

    @app.get("/v1/items/{item_id}/similar", response_model=list[Item], tags=["recommendations"])
    def similar(item_id: str, k: int = Query(10, ge=1, le=100), s: RecommendationService = Depends(svc)) -> list:
        if item_id not in s.index.pos:
            raise HTTPException(404, "unknown item")
        return [r.__dict__ for r in s.similar_items(item_id, k)]

    @app.post("/v1/events/click", tags=["experiments"])
    def click(body: ClickIn, s: RecommendationService = Depends(svc)) -> dict:
        return {"logged": True, "variant": s.log_click(body.user_id, body.item_id, body.position, body.request_id)}

    @app.get("/v1/experiments/report", tags=["experiments"])
    def report(s: RecommendationService = Depends(svc)) -> dict:
        if s.events is None:
            raise HTTPException(503, "event store not configured")
        return s.events.report(s.experiment)

    @app.post("/v1/cache/invalidate", tags=["ops"])
    def invalidate(user_id: str | None = None, s: RecommendationService = Depends(svc)) -> dict:
        return {"deleted": s.cache.delete_prefix("rec:" if user_id is None else f"rec:{user_id}:")}

    return app


app = create_app()
