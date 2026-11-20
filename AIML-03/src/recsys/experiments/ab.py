"""Online A/B testing: assignment, event logging and statistical readout.

* Assignment is a salted SHA-256 hash of (experiment, user) -> sticky, uniform,
  stateless, and independent across experiments.
* Impressions and clicks are logged to an `events` table (sqlite / Postgres).
* `report()` computes CTR per variant, absolute and relative lift, a
  two-proportion z-test p-value and a 95% CI on the difference, plus
  NDCG@10 of click positions as a ranking-quality signal.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine, func, insert, select

metadata = MetaData()
events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("experiment", String(64), index=True, nullable=False),
    Column("variant", String(32), index=True, nullable=False),
    Column("user_id", String(64), index=True, nullable=False),
    Column("item_id", String(64), nullable=False),
    Column("position", Integer, nullable=False),
    Column("event", String(16), nullable=False),  # impression | click
    Column("request_id", String(64), nullable=False),
    Column("ts", DateTime(timezone=True), nullable=False),
)


@dataclass(frozen=True)
class Experiment:
    name: str
    variants: tuple[str, ...] = ("dcn_v2", "wide_deep")
    split: float = 0.5  # share of traffic for variants[0]

    def assign(self, user_id: str) -> str:
        h = int(hashlib.sha256(f"{self.name}:{user_id}".encode()).hexdigest()[:15], 16) / 16**15
        return self.variants[0] if h < self.split else self.variants[1]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_proportion_test(c1: int, n1: int, c2: int, n2: int) -> dict[str, float]:
    p1, p2 = c1 / max(n1, 1), c2 / max(n2, 1)
    pooled = (c1 + c2) / max(n1 + n2, 1)
    se0 = math.sqrt(max(pooled * (1 - pooled) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-12))
    z = (p2 - p1) / se0
    se = math.sqrt(max(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1), 1e-12))
    return {"z": z, "p_value": 2 * (1 - _norm_cdf(abs(z))), "diff": p2 - p1,
            "ci95_low": p2 - p1 - 1.96 * se, "ci95_high": p2 - p1 + 1.96 * se}


class EventStore:
    def __init__(self, url: str = "sqlite:///events.db") -> None:
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine = create_engine(url, future=True, **kw)
        metadata.create_all(self.engine)

    def log(self, rows: list[dict]) -> None:
        if not rows:
            return
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            conn.execute(insert(events), [{**r, "ts": r.get("ts", now)} for r in rows])

    def counts(self, experiment: str) -> dict[str, dict[str, int]]:
        q = (select(events.c.variant, events.c.event, func.count())
             .where(events.c.experiment == experiment).group_by(events.c.variant, events.c.event))
        out: dict[str, dict[str, int]] = {}
        with self.engine.connect() as conn:
            for variant, ev, n in conn.execute(q):
                out.setdefault(variant, {"impression": 0, "click": 0})[ev] = int(n)
        return out

    def click_ndcg(self, experiment: str, k: int = 10) -> dict[str, float]:
        """Mean per-request DCG of clicked positions / ideal DCG for that many clicks."""
        q = select(events.c.variant, events.c.request_id, events.c.position).where(
            (events.c.experiment == experiment) & (events.c.event == "click") & (events.c.position < k))
        per: dict[tuple[str, str], list[int]] = {}
        with self.engine.connect() as conn:
            for v, rid, pos in conn.execute(q):
                per.setdefault((v, rid), []).append(pos)
        req = select(events.c.variant, func.count(func.distinct(events.c.request_id))).where(
            (events.c.experiment == experiment) & (events.c.event == "impression")).group_by(events.c.variant)
        with self.engine.connect() as conn:
            n_req = {v: int(n) for v, n in conn.execute(req)}
        sums: dict[str, float] = {}
        for (v, _), positions in per.items():
            dcg = sum(1 / math.log2(p + 2) for p in positions)
            idcg = sum(1 / math.log2(i + 2) for i in range(len(positions)))
            sums[v] = sums.get(v, 0.0) + dcg / idcg
        return {v: sums.get(v, 0.0) / max(n, 1) for v, n in n_req.items()}

    def report(self, exp: Experiment) -> dict:
        c = self.counts(exp.name)
        a, b = exp.variants
        ca, cb = c.get(a, {"impression": 0, "click": 0}), c.get(b, {"impression": 0, "click": 0})
        ctr_a = ca["click"] / max(ca["impression"], 1)
        ctr_b = cb["click"] / max(cb["impression"], 1)
        test = two_proportion_test(ca["click"], ca["impression"], cb["click"], cb["impression"])
        ndcg = self.click_ndcg(exp.name)
        return {
            "experiment": exp.name,
            "variants": {a: {**ca, "ctr": ctr_a, "click_ndcg@10": ndcg.get(a, 0.0)},
                         b: {**cb, "ctr": ctr_b, "click_ndcg@10": ndcg.get(b, 0.0)}},
            "lift_abs": ctr_b - ctr_a,
            "lift_rel": (ctr_b - ctr_a) / ctr_a if ctr_a else 0.0,
            **{k: v for k, v in test.items() if k != "diff"},
            "significant_at_0.05": test["p_value"] < 0.05,
            "winner": (b if ctr_b > ctr_a else a) if test["p_value"] < 0.05 else None,
        }


def required_sample_size(baseline_ctr: float, min_detectable_lift: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """Impressions per variant to detect a relative CTR lift (two-sided z-test)."""
    from statistics import NormalDist

    p1, p2 = baseline_ctr, baseline_ctr * (1 + min_detectable_lift)
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return int(math.ceil(n))
