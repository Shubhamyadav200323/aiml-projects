"""Online (streaming) feature engine.

Maintains incremental per-user and per-entity state so each incoming
transaction is featurised in O(window) time. The arithmetic deliberately
mirrors :mod:`fraud.features.engineering` term by term; parity is enforced by
tests. State lives behind a :class:`StateBackend` so the same engine runs on
an in-process dict (tests, single pod) or Redis (horizontally scaled pods).
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import pandas as pd

from fraud.features.definitions import CHANNEL_CODES, NO_HISTORY_GAP_S, WINDOWS_S

_MAX_WINDOW = max(WINDOWS_S.values())


@dataclass
class UserState:
    count: int = 0
    total: float = 0.0
    last_t: float | None = None
    last_device: str | None = None
    last_location: str | None = None
    devices: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    cards: list[str] = field(default_factory=list)
    recent: list[tuple[float, float]] = field(default_factory=list)  # (t, amount) within max window


class StateBackend(Protocol):
    def get_user(self, user_id: str) -> UserState: ...
    def put_user(self, user_id: str, state: UserState) -> None: ...
    def add_entity_user(self, kind: str, entity_id: str, user_id: str) -> int: ...


class InMemoryBackend:
    def __init__(self) -> None:
        self._users: dict[str, UserState] = {}
        self._entities: dict[tuple[str, str], set[str]] = {}

    def get_user(self, user_id: str) -> UserState:
        return self._users.get(user_id) or UserState()

    def put_user(self, user_id: str, state: UserState) -> None:
        self._users[user_id] = state

    def add_entity_user(self, kind: str, entity_id: str, user_id: str) -> int:
        users = self._entities.setdefault((kind, entity_id), set())
        users.add(user_id)
        return len(users)


class RedisBackend:
    """Redis-backed state: one JSON blob per user, one SET per device/card."""

    def __init__(self, client: Any, prefix: str = "fraud") -> None:
        self.r = client
        self.prefix = prefix

    def get_user(self, user_id: str) -> UserState:
        raw = self.r.get(f"{self.prefix}:user:{user_id}")
        if not raw:
            return UserState()
        data = json.loads(raw)
        data["recent"] = [tuple(x) for x in data["recent"]]
        return UserState(**data)

    def put_user(self, user_id: str, state: UserState) -> None:
        self.r.set(f"{self.prefix}:user:{user_id}", json.dumps(asdict(state)))

    def add_entity_user(self, kind: str, entity_id: str, user_id: str) -> int:
        key = f"{self.prefix}:{kind}:{entity_id}"
        pipe = self.r.pipeline()
        pipe.sadd(key, user_id)
        pipe.scard(key)
        return int(pipe.execute()[1])


def _to_epoch_seconds(ts: Any) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    return (stamp - pd.Timestamp("1970-01-01")).total_seconds()


class OnlineFeatureEngine:
    """`compute(txn)` returns features *and* folds the txn into state (read-then-write)."""

    def __init__(self, backend: StateBackend | None = None) -> None:
        self.backend = backend or InMemoryBackend()
        self._lock = threading.Lock()  # per-process; Redis deployments shard by user_id key

    def compute(self, txn: Mapping[str, Any], update: bool = True) -> dict[str, float]:
        with self._lock:
            return self._compute(txn, update)

    def _compute(self, txn: Mapping[str, Any], update: bool) -> dict[str, float]:
        user = str(txn["user_id"])
        device, card, loc = str(txn["device_id"]), str(txn["card_id"]), str(txn["location"])
        amount = float(txn["amount"])
        t = _to_epoch_seconds(txn["timestamp"])
        st = self.backend.get_user(user)
        st.recent = [(rt, ra) for rt, ra in st.recent if rt >= t - _MAX_WINDOW]

        f: dict[str, float] = {"amount": amount, "amount_log": math.log1p(amount)}
        for name, w in WINDOWS_S.items():
            win = [ra for rt, ra in st.recent if rt >= t - w]
            f[f"txn_count_{name}"] = float(len(win))
            f[f"amount_sum_{name}"] = float(sum(win))
        avg = st.total / st.count if st.count else 0.0
        f["user_txn_count"] = float(st.count)
        f["user_avg_amount"] = avg
        f["amount_to_avg_ratio"] = amount / avg if avg > 0 else 1.0
        f["seconds_since_last_txn"] = t - st.last_t if st.last_t is not None else NO_HISTORY_GAP_S
        has_hist = st.count > 0
        f["is_new_device"] = float(has_hist and device not in st.devices)
        f["is_new_location"] = float(has_hist and loc not in st.locations)
        f["is_new_card"] = float(has_hist and card not in st.cards)
        f["device_changed"] = float(st.last_device is not None and st.last_device != device)
        f["location_changed"] = float(st.last_location is not None and st.last_location != loc)

        if update:
            f["device_user_count"] = float(self.backend.add_entity_user("device", device, user))
            f["card_user_count"] = float(self.backend.add_entity_user("card", card, user))
        else:  # read-only scoring (what-if analysis)
            f["device_user_count"] = f["card_user_count"] = 1.0
        ts = pd.Timestamp(txn["timestamp"])
        f["hour_of_day"] = float(ts.hour)
        f["channel_code"] = float(CHANNEL_CODES.get(str(txn.get("channel", "")), -1))

        if update:
            st.count += 1
            st.total += amount
            st.last_t, st.last_device, st.last_location = t, device, loc
            for lst, v in ((st.devices, device), (st.locations, loc), (st.cards, card)):
                if v not in lst:
                    lst.append(v)
            st.recent.append((t, amount))
            self.backend.put_user(user, st)
        return f
