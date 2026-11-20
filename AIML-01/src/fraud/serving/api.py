"""FastAPI application tier.

Run:  uvicorn fraud.serving.api:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs (OpenAPI generated from `schemas.py`)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from fraud.config import Settings, get_settings
from fraud.features.store import build_online_engine
from fraud.repository.cases import CaseRepository
from fraud.serving.metrics import METRICS
from fraud.serving.schemas import (
    BatchRequest,
    CaseOut,
    CaseUpdate,
    Ring,
    ScoreResponse,
    Transaction,
)
from fraud.serving.scorer import FraudScorer, ModelBundle

log = logging.getLogger(__name__)


def create_app(scorer: FraudScorer | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if scorer is not None:
            app.state.scorer = scorer
        else:
            bundle = ModelBundle.load(settings.artifacts_dir)
            app.state.scorer = FraudScorer(bundle, build_online_engine(settings),
                                           CaseRepository(settings.database_url))
            log.info("loaded model %s from %s", bundle.version, Path(settings.artifacts_dir))
        yield

    app = FastAPI(title="Fraud Scoring API", version="1.0.0", lifespan=lifespan,
                  description="Real-time transaction risk scoring, case management and ring intel.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def get_scorer(request: Request) -> FraudScorer:
        return request.app.state.scorer

    def get_cases(s: FraudScorer = Depends(get_scorer)) -> CaseRepository:
        if s.cases is None:
            raise HTTPException(503, "case repository not configured")
        return s.cases

    # ---- health ---------------------------------------------------------
    @app.get("/health", tags=["ops"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready(s: FraudScorer = Depends(get_scorer)) -> dict:
        return {"status": "ready", "model_version": s.bundle.version,
                "p50_ms": METRICS.percentile(0.5), "p99_ms": METRICS.percentile(0.99)}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def metrics() -> str:
        return METRICS.render()

    # ---- scoring --------------------------------------------------------
    @app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
    def score(txn: Transaction, s: FraudScorer = Depends(get_scorer)) -> dict:
        return s.score(txn.model_dump()).as_dict()

    @app.post("/v1/score/batch", response_model=list[ScoreResponse], tags=["scoring"])
    def score_batch(req: BatchRequest, s: FraudScorer = Depends(get_scorer)) -> list[dict]:
        ordered = sorted(req.transactions, key=lambda t: (t.timestamp, t.transaction_id))
        return [s.score(t.model_dump()).as_dict() for t in ordered]

    # ---- case management -----------------------------------------------
    @app.get("/v1/cases", response_model=list[CaseOut], tags=["cases"])
    def list_cases(status: str | None = None, limit: int = Query(50, ge=1, le=500),
                   offset: int = Query(0, ge=0), min_score: float = Query(0.0, ge=0, le=1),
                   repo: CaseRepository = Depends(get_cases)) -> list:
        return [c.__dict__ for c in repo.list(status, limit, offset, min_score)]

    @app.get("/v1/cases/stats", tags=["cases"])
    def case_stats(repo: CaseRepository = Depends(get_cases)) -> dict:
        return repo.stats()

    @app.get("/v1/cases/{case_id}", response_model=CaseOut, tags=["cases"])
    def get_case(case_id: int, repo: CaseRepository = Depends(get_cases)) -> dict:
        case = repo.get(case_id)
        if case is None:
            raise HTTPException(404, "case not found")
        return case.__dict__

    @app.patch("/v1/cases/{case_id}", response_model=CaseOut, tags=["cases"])
    def update_case(case_id: int, body: CaseUpdate, repo: CaseRepository = Depends(get_cases)) -> dict:
        case = repo.update_status(case_id, body.status, body.analyst, body.note)
        if case is None:
            raise HTTPException(404, "case not found")
        return case.__dict__

    # ---- graph intelligence --------------------------------------------
    @app.get("/v1/rings", response_model=list[Ring], tags=["graph"])
    def rings(min_size: int = Query(3, ge=2), s: FraudScorer = Depends(get_scorer)) -> list:
        return [r for r in s.bundle.rings if r["size"] >= min_size]

    @app.get("/v1/users/{user_id}/risk", tags=["graph"])
    def user_risk(user_id: str, s: FraudScorer = Depends(get_scorer)) -> dict:
        ring = next((r["ring_id"] for r in s.bundle.rings if user_id in r["members"]), None)
        return {"user_id": user_id,
                "graph_score": s.bundle.graph_scores.get(user_id, s.bundle.policy["graph_prior"]),
                "known": user_id in s.bundle.graph_scores, "ring_id": ring}

    return app


app = create_app()
