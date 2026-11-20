"""Domain service: transaction -> features -> blended risk -> decision -> case.

Pure Python, no web framework: the FastAPI layer and the Kafka consumer both
call `FraudScorer.score`, so HTTP and streaming paths share one code path.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fraud.features.definitions import FEATURE_COLUMNS
from fraud.features.online import OnlineFeatureEngine
from fraud.models.xgb import XGBFraudModel
from fraud.repository.cases import CaseRepository
from fraud.serving.metrics import METRICS


@dataclass
class ModelBundle:
    xgb: XGBFraudModel
    graph_scores: dict[str, float]
    policy: dict[str, float]
    rings: list[dict]
    version: str

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        d = Path(directory)
        gs = pd.read_parquet(d / "graph_scores.parquet")
        version = hashlib.sha256((d / "xgb.json").read_bytes()).hexdigest()[:12]
        return cls(
            xgb=XGBFraudModel.load(d),
            graph_scores=dict(zip(gs["user_id"], gs["graph_score"].astype(float))),
            policy=json.loads((d / "policy.json").read_text()),
            rings=json.loads((d / "rings.json").read_text()) if (d / "rings.json").exists() else [],
            version=version,
        )


@dataclass
class ScoreResult:
    transaction_id: str
    fraud_score: float
    xgb_score: float
    graph_score: float
    decision: str
    reasons: list[dict[str, float]]
    features: dict[str, float]
    case_id: int | None
    model_version: str
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("features")
        return d


@dataclass
class FraudScorer:
    bundle: ModelBundle
    features: OnlineFeatureEngine
    cases: CaseRepository | None = None
    explain: bool = True
    _cols: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def decide(self, score: float) -> str:
        p = self.bundle.policy
        if score >= p["decline_threshold"]:
            return "decline"
        if score >= p["review_threshold"]:
            return "review"
        return "approve"

    def score(self, txn: dict[str, Any]) -> ScoreResult:
        t0 = time.perf_counter()
        feats = self.features.compute(txn)
        X = pd.DataFrame([feats], columns=self._cols)
        p_x = float(self.bundle.xgb.predict_proba(X)[0])
        p_g = float(self.bundle.graph_scores.get(str(txn["user_id"]), self.bundle.policy["graph_prior"]))
        w = self.bundle.policy["weight_xgb"]
        score = w * p_x + (1 - w) * p_g
        decision = self.decide(score)
        reasons = self.bundle.xgb.explain(X)[0] if (self.explain and decision != "approve") else []
        case_id = None
        if decision != "approve" and self.cases is not None:
            case_id = self.cases.create(
                transaction_id=str(txn["transaction_id"]), user_id=str(txn["user_id"]),
                amount=float(txn["amount"]), fraud_score=score, xgb_score=p_x, graph_score=p_g,
                decision=decision, reasons=reasons, features=feats)
        latency = (time.perf_counter() - t0) * 1000
        METRICS.observe(decision, latency)
        return ScoreResult(str(txn["transaction_id"]), score, p_x, p_g, decision, reasons, feats,
                           case_id, self.bundle.version, round(latency, 3))
