"""PaySim-style synthetic transaction generator with planted fraud rings.

Kaggle datasets require an authenticated download, so CI and the demo use this
generator. It reproduces the properties that matter for the modelling problem:

* heavy class imbalance (~0.5-1% fraud),
* per-user spending habits (log-normal amounts around a personal mean),
* fraud *rings*: groups of mule accounts that share devices and cards,
  transact in bursts, from new locations, with amounts far above their norm.

The output uses the canonical schema from :mod:`fraud.data.loaders`, so the
rest of the pipeline cannot tell synthetic data from PaySim / IEEE-CIS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraud.data.loaders import TRANSACTION_COLUMNS

CHANNELS = np.array(["CARD_PRESENT", "ONLINE", "TRANSFER", "CASH_OUT", "PAYMENT"])


@dataclass(frozen=True)
class SyntheticConfig:
    n_users: int = 3_000
    n_merchants: int = 400
    n_locations: int = 60
    days: int = 30
    txns_per_user_per_day: float = 1.2
    household_share: float = 0.15      # legit accounts sharing a device with family
    splurge_rate: float = 0.015        # legit large purchases (hard negatives)
    n_takeovers: int = 60              # individual account-takeover victims
    n_rings: int = 12
    ring_size: tuple[int, int] = (4, 9)
    ring_txns_per_member: tuple[int, int] = (3, 8)
    seed: int = 7
    start: str = "2026-01-01"


def generate_transactions(cfg: SyntheticConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start)
    horizon_s = cfg.days * 86_400

    # ---- legitimate behaviour ------------------------------------------
    user_mean = rng.lognormal(mean=3.6, sigma=0.8, size=cfg.n_users)  # ~ $35 median
    user_home = rng.integers(0, cfg.n_locations, size=cfg.n_users)
    n_tx = rng.poisson(cfg.txns_per_user_per_day * cfg.days, size=cfg.n_users)

    users = np.repeat(np.arange(cfg.n_users), n_tx)
    total = users.size
    ts = rng.uniform(0, horizon_s, size=total)
    amount = rng.lognormal(np.log(user_mean[users]), 0.5)
    # 95% of the time users stay home and on their own device/card
    travelling = rng.random(total) < 0.08
    location = np.where(travelling, rng.integers(0, cfg.n_locations, total), user_home[users])
    # households: some legit users share a family tablet -> legit device sharing
    household = np.arange(cfg.n_users)
    fam = rng.random(cfg.n_users) < cfg.household_share
    household[fam] = rng.integers(0, max(1, int(cfg.n_users * cfg.household_share / 3)), fam.sum()) + 5_000_000
    second_device = rng.random(total) < 0.08
    device = np.where(second_device, users + 1_000_000, users)
    family_device = (rng.random(total) < 0.3) & fam[users]
    device = np.where(family_device, household[users], device)
    splurge = rng.random(total) < cfg.splurge_rate
    amount = np.where(splurge, amount * rng.uniform(3, 10, total), amount)
    legit = pd.DataFrame(
        {
            "user_id": users,
            # card re-issues / second card -> legit "new card" events
            "card_id": np.where(rng.random(total) < 0.03, users + 2_000_000, users),
            "device_id": device,
            "merchant_id": rng.integers(0, cfg.n_merchants, total),
            "location": location,
            "amount": amount,
            "channel": rng.choice(CHANNELS, total, p=[0.35, 0.3, 0.15, 0.1, 0.1]),
            "ts": ts,
            "is_fraud": 0,
        }
    )

    # ---- account takeover: victim's card used from attacker's device ----
    frames = []
    victims = rng.choice(cfg.n_users, size=cfg.n_takeovers, replace=False)
    for i, v in enumerate(victims):
        k = int(rng.integers(1, 4, endpoint=True))
        frames.append(pd.DataFrame({
            "user_id": v, "card_id": v,
            # attackers occasionally reuse a handful of devices -> weak graph signal
            "device_id": 7_000_000 + int(rng.integers(0, cfg.n_takeovers // 3)) if rng.random() < 0.4
            else 7_500_000 + i,
            "merchant_id": rng.integers(0, cfg.n_merchants, k),
            "location": rng.integers(0, cfg.n_locations, k),
            "amount": user_mean[v] * rng.uniform(1.2, 6, k),
            "channel": rng.choice(CHANNELS, k),
            "ts": rng.uniform(0.1 * horizon_s, horizon_s) + rng.exponential(3_600, k).cumsum(),
            "is_fraud": 1,
        }))

    # ---- fraud rings ---------------------------------------------------
    for ring in range(cfg.n_rings):
        size = rng.integers(*cfg.ring_size, endpoint=True)
        members = rng.choice(cfg.n_users, size=size, replace=False)
        shared_devices = 9_000_000 + ring * 10 + np.arange(rng.integers(1, 3, endpoint=True))
        shared_cards = 8_000_000 + ring * 10 + np.arange(rng.integers(1, 3, endpoint=True))
        burst_start = rng.uniform(0.3 * horizon_s, 0.95 * horizon_s)
        for m in members:
            k = rng.integers(*cfg.ring_txns_per_member, endpoint=True)
            frames.append(
                pd.DataFrame(
                    {
                        "user_id": m,
                        # mules mix their own instruments with the ring's shared ones,
                        # so per-transaction signals are weaker than the ring structure
                        "card_id": np.where(rng.random(k) < 0.5, m, rng.choice(shared_cards, k)),
                        "device_id": np.where(rng.random(k) < 0.35, m, rng.choice(shared_devices, k)),
                        "merchant_id": rng.integers(0, cfg.n_merchants, k),
                        "location": rng.integers(0, cfg.n_locations, k),
                        "amount": user_mean[m] * rng.uniform(1.5, 8, k),
                        "channel": rng.choice(np.array(["TRANSFER", "CASH_OUT", "ONLINE", "PAYMENT"]), k),
                        "ts": np.clip(burst_start + rng.exponential(4 * 3_600, k).cumsum(), 0, horizon_s),
                        "is_fraud": 1,
                    }
                )
            )
    df = pd.concat([legit, *frames], ignore_index=True)

    # ---- canonical schema ---------------------------------------------
    df["ts"] = np.clip(df["ts"], 0, horizon_s)
    df["timestamp"] = start + pd.to_timedelta(df.pop("ts"), unit="s")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["transaction_id"] = [f"T{i:09d}" for i in range(len(df))]
    for col, prefix in [("user_id", "U"), ("card_id", "C"), ("device_id", "D"),
                        ("merchant_id", "M"), ("location", "L")]:
        df[col] = prefix + df[col].astype(np.int64).astype(str)
    df["amount"] = df["amount"].round(2)
    return df[TRANSACTION_COLUMNS]
