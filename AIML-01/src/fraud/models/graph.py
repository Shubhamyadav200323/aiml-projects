"""Account-device-card entity graph and fraud-ring extraction.

The heterogeneous graph (accounts <-> devices, accounts <-> cards) is projected
onto accounts: two accounts are linked when they share a device or a card.
Hub entities (e.g. a shared public kiosk) are capped so the projection stays
linear in practice instead of quadratic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class AccountGraph:
    users: np.ndarray                 # node index -> user_id
    edge_index: np.ndarray            # shape (2, E), undirected (both directions present)
    edge_type: np.ndarray             # 0 = shared device, 1 = shared card
    x: np.ndarray                     # node features (N, F)
    y: np.ndarray                     # node labels: account involved in fraud
    feature_names: list[str]
    norm: tuple[np.ndarray, np.ndarray] | None = None  # (mean, std) used to standardise x

    @property
    def num_nodes(self) -> int:
        return len(self.users)

    def index_of(self) -> dict[str, int]:
        return {u: i for i, u in enumerate(self.users)}


NODE_FEATURES = [
    "txn_count", "amount_mean_log", "amount_max_log", "n_devices", "n_cards", "n_locations",
    "max_device_sharing", "max_card_sharing", "night_share", "transfer_share",
]


def _node_features(tx: pd.DataFrame) -> pd.DataFrame:
    dev_users = tx.groupby("device_id")["user_id"].nunique()
    card_users = tx.groupby("card_id")["user_id"].nunique()
    tx = tx.assign(
        _dev_share=tx["device_id"].map(dev_users),
        _card_share=tx["card_id"].map(card_users),
        _night=tx["timestamp"].dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(float),
        _transfer=tx["channel"].isin(["TRANSFER", "CASH_OUT"]).astype(float),
        _alog=np.log1p(tx["amount"]),
    )
    g = tx.groupby("user_id", sort=True)
    return pd.DataFrame(
        {
            "txn_count": np.log1p(g.size()),
            "amount_mean_log": g["_alog"].mean(),
            "amount_max_log": g["_alog"].max(),
            "n_devices": g["device_id"].nunique(),
            "n_cards": g["card_id"].nunique(),
            "n_locations": g["location"].nunique(),
            "max_device_sharing": np.log1p(g["_dev_share"].max()),
            "max_card_sharing": np.log1p(g["_card_share"].max()),
            "night_share": g["_night"].mean(),
            "transfer_share": g["_transfer"].mean(),
        }
    )


def build_account_graph(tx: pd.DataFrame, max_entity_degree: int = 50,
                        norm: tuple[np.ndarray, np.ndarray] | None = None) -> AccountGraph:
    """Build the projected account graph.

    Pass the `norm` of the training graph when building an inference graph so
    node features are standardised with training statistics (inductive use).
    """
    feats = _node_features(tx)
    users = feats.index.to_numpy()
    uidx = pd.Series(np.arange(len(users)), index=users)
    src, dst, etype = [], [], []
    for t, col in enumerate(["device_id", "card_id"]):
        pairs = tx[["user_id", col]].drop_duplicates()
        for _, members in pairs.groupby(col)["user_id"]:
            m = uidx[members.to_numpy()].to_numpy()
            if len(m) < 2 or len(m) > max_entity_degree:
                continue
            a, b = np.meshgrid(m, m)
            mask = a != b
            src.append(a[mask])
            dst.append(b[mask])
            etype.append(np.full(mask.sum(), t))
    if src:
        ei = np.vstack([np.concatenate(src), np.concatenate(dst)])
        et = np.concatenate(etype)
        # de-duplicate parallel edges of the same type
        key = pd.DataFrame({"s": ei[0], "d": ei[1], "t": et}).drop_duplicates()
        ei, et = key[["s", "d"]].to_numpy().T, key["t"].to_numpy()
    else:
        ei, et = np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)
    x = feats.to_numpy(dtype=np.float32)
    if norm is None:
        norm = (x.mean(0), x.std(0) + 1e-6)
    x = (x - norm[0]) / norm[1]
    y = tx.groupby("user_id", sort=True)["is_fraud"].max().reindex(users).to_numpy()
    return AccountGraph(users, ei.astype(np.int64), et.astype(np.int64), x.astype(np.float32),
                        y.astype(np.float32), list(NODE_FEATURES), norm)


@dataclass
class FraudRing:
    ring_id: int
    members: list[str]
    mean_score: float
    max_score: float
    size: int


def detect_rings(graph: AccountGraph, scores: np.ndarray, threshold: float = 0.5,
                 min_size: int = 3) -> list[FraudRing]:
    """Connected components of the subgraph induced by high-risk accounts."""
    risky = scores >= threshold
    s, d = graph.edge_index
    keep = risky[s] & risky[d]
    n = graph.num_nodes
    adj = coo_matrix((np.ones(keep.sum()), (s[keep], d[keep])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    rings = []
    for comp in np.unique(labels[risky]):
        members = np.where((labels == comp) & risky)[0]
        if len(members) >= min_size:
            rings.append(FraudRing(
                ring_id=len(rings), members=graph.users[members].tolist(),
                mean_score=float(scores[members].mean()), max_score=float(scores[members].max()),
                size=len(members)))
    return sorted(rings, key=lambda r: (-r.size, -r.mean_score))
