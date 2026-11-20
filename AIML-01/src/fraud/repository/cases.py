"""Case-management repository (data tier).

SQLAlchemy Core keeps the persistence layer portable: sqlite for local runs
and tests, PostgreSQL in Kubernetes (`DATABASE_URL=postgresql+psycopg2://...`).
Analyst feedback labels written here are the ground truth for retraining.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine


class CaseStatus(str, enum.Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    CONFIRMED_FRAUD = "confirmed_fraud"
    FALSE_POSITIVE = "false_positive"

    @property
    def label(self) -> int | None:
        return {CaseStatus.CONFIRMED_FRAUD: 1, CaseStatus.FALSE_POSITIVE: 0}.get(self)


metadata = MetaData()
cases_table = Table(
    "cases", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("transaction_id", String(64), nullable=False, unique=True, index=True),
    Column("user_id", String(64), nullable=False, index=True),
    Column("amount", Float, nullable=False),
    Column("fraud_score", Float, nullable=False),
    Column("xgb_score", Float, nullable=False),
    Column("graph_score", Float, nullable=False),
    Column("decision", String(16), nullable=False),
    Column("status", String(32), nullable=False, default=CaseStatus.OPEN.value, index=True),
    Column("label", Integer, nullable=True),
    Column("reasons", Text, nullable=False, default="[]"),
    Column("features", Text, nullable=False, default="{}"),
    Column("analyst", String(64), nullable=True),
    Column("note", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


@dataclass
class Case:
    id: int
    transaction_id: str
    user_id: str
    amount: float
    fraud_score: float
    xgb_score: float
    graph_score: float
    decision: str
    status: str
    label: int | None
    reasons: list
    features: dict
    analyst: str | None
    note: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row) -> Case:
        d = dict(row._mapping)
        d["reasons"] = json.loads(d["reasons"] or "[]")
        d["features"] = json.loads(d["features"] or "{}")
        return cls(**d)


class CaseRepository:
    def __init__(self, url: str = "sqlite:///cases.db", engine: Engine | None = None) -> None:
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {
            "pool_size": 10, "max_overflow": 20, "pool_pre_ping": True}
        self.engine = engine or create_engine(url, future=True, **kwargs)
        metadata.create_all(self.engine)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def create(self, *, transaction_id: str, user_id: str, amount: float, fraud_score: float,
               xgb_score: float, graph_score: float, decision: str, reasons: list,
               features: dict) -> int:
        now = self._now()
        with self.engine.begin() as conn:
            existing = conn.execute(select(cases_table.c.id).where(
                cases_table.c.transaction_id == transaction_id)).scalar()
            if existing is not None:  # idempotent: Kafka is at-least-once
                return int(existing)
            res = conn.execute(insert(cases_table).values(
                transaction_id=transaction_id, user_id=user_id, amount=amount,
                fraud_score=fraud_score, xgb_score=xgb_score, graph_score=graph_score,
                decision=decision, status=CaseStatus.OPEN.value, reasons=json.dumps(reasons),
                features=json.dumps(features), created_at=now, updated_at=now))
            return int(res.inserted_primary_key[0])

    def get(self, case_id: int) -> Case | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(cases_table).where(cases_table.c.id == case_id)).first()
        return Case.from_row(row) if row else None

    def list(self, status: str | None = None, limit: int = 100, offset: int = 0,
             min_score: float = 0.0) -> list[Case]:
        q = select(cases_table).where(cases_table.c.fraud_score >= min_score)
        if status:
            q = q.where(cases_table.c.status == status)
        q = q.order_by(cases_table.c.fraud_score.desc(), cases_table.c.id).limit(limit).offset(offset)
        with self.engine.connect() as conn:
            return [Case.from_row(r) for r in conn.execute(q)]

    def update_status(self, case_id: int, status: CaseStatus, analyst: str | None = None,
                      note: str | None = None) -> Case | None:
        values = {"status": status.value, "label": status.label, "updated_at": self._now()}
        if analyst is not None:
            values["analyst"] = analyst
        if note is not None:
            values["note"] = note
        with self.engine.begin() as conn:
            res = conn.execute(update(cases_table).where(cases_table.c.id == case_id).values(**values))
        return self.get(case_id) if res.rowcount else None

    def stats(self) -> dict[str, int]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(cases_table.c.status, func.count()).group_by(cases_table.c.status))
            return {status: int(n) for status, n in rows}

    def labelled(self) -> list[Case]:
        """Analyst-confirmed cases: the feedback set for monitoring and retraining."""
        with self.engine.connect() as conn:
            rows = conn.execute(select(cases_table).where(cases_table.c.label.is_not(None)))
            return [Case.from_row(r) for r in rows]
