"""API contracts (pydantic v2). These are the documented request/response shapes."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from fraud.repository.cases import CaseStatus


class Transaction(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "transaction_id": "T000000123", "timestamp": "2026-01-15T10:32:00Z", "user_id": "U42",
        "card_id": "C42", "device_id": "D42", "merchant_id": "M7", "location": "L3",
        "amount": 129.99, "channel": "ONLINE"}})

    transaction_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    user_id: str = Field(min_length=1, max_length=64)
    card_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    merchant_id: str = Field(min_length=1, max_length=64)
    location: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, lt=1e9)
    channel: str = "ONLINE"


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    DECLINE = "decline"


class Reason(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float = Field(ge=0, le=1)
    xgb_score: float
    graph_score: float
    decision: Decision
    reasons: list[Reason]
    case_id: int | None = None
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class CaseOut(BaseModel):
    id: int
    transaction_id: str
    user_id: str
    amount: float
    fraud_score: float
    xgb_score: float
    graph_score: float
    decision: str
    status: CaseStatus
    label: int | None
    reasons: list[Reason]
    analyst: str | None
    note: str | None
    created_at: datetime
    updated_at: datetime


class CaseUpdate(BaseModel):
    status: CaseStatus
    analyst: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=2000)


class Ring(BaseModel):
    ring_id: int
    members: list[str]
    mean_score: float
    max_score: float
    size: int
