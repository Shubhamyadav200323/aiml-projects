"""Feature tables (pandas engine).

Produces the same three tables as the Spark job in `spark_job.py` - the
pandas path serves tests and laptop-scale runs; Spark serves MovieLens 25M.
`tests/test_features.py` asserts both engines agree.

* interactions: positives with integer ids (model input)
* user_features: n_interactions, n_positives, avg_rating, genre affinity (18-d), recent history (last H items)
* item_features: popularity (log count), recency-weighted popularity, avg_rating, genre multi-hot
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recsys.data.movielens import GENRES

HISTORY_LEN = 20
POP_HALF_LIFE_DAYS = 30.0


@dataclass
class FeatureTables:
    user_index: dict[str, int]          # user_id -> row (0 reserved for unknown)
    item_index: dict[str, int]          # item_id -> row (0 reserved for padding/unknown)
    user_features: pd.DataFrame         # indexed by user_id
    item_features: pd.DataFrame         # indexed by item_id, includes all catalogue items
    as_of: pd.Timestamp
    seen: dict[str, set[str]] = field(default_factory=dict)  # every item a user interacted with

    def seen_items(self, user_id: str) -> set[str]:
        return self.seen.get(user_id, set())

    @property
    def item_ids(self) -> list[str]:
        return list(self.item_features.index)

    def genre_matrix(self) -> np.ndarray:
        return self.item_features[[f"g_{g}" for g in GENRES]].to_numpy(np.float32)

    def user_genre_affinity(self, user_id: str) -> np.ndarray:
        if user_id in self.user_features.index:
            return self.user_features.loc[user_id, [f"aff_{g}" for g in GENRES]].to_numpy(np.float32)
        return np.zeros(len(GENRES), np.float32)

    def history(self, user_id: str) -> list[str]:
        if user_id in self.user_features.index:
            return list(self.user_features.at[user_id, "history"])
        return []


def _genre_onehot(items: pd.DataFrame) -> pd.DataFrame:
    genres = items["genres"].tolist()
    data = {f"g_{g}": np.array([float(g in gs) for gs in genres]) for g in GENRES}
    return pd.DataFrame(data, index=pd.Index(items["item_id"].to_numpy(), name="item_id"))


def build_feature_tables(train: pd.DataFrame, items: pd.DataFrame, as_of: pd.Timestamp | None = None,
                         vocab: tuple[dict[str, int], dict[str, int]] | None = None) -> FeatureTables:
    """`train` must contain only events before `as_of` (point-in-time).

    Pass `vocab=(user_index, item_index)` of the trained models when refreshing
    features: embedding rows are tied to ids, so users first seen after training
    map to the shared "unknown" row 0 instead of shifting everyone's index.
    """
    as_of = as_of or train["timestamp"].max()
    onehot = _genre_onehot(items)

    # ---- items ----------------------------------------------------------
    age_days = (as_of - train["timestamp"]).dt.total_seconds() / 86_400
    decay = np.power(0.5, age_days / POP_HALF_LIFE_DAYS)
    g = train.assign(_decay=decay).groupby("item_id")
    item_stats = pd.DataFrame({
        "pop_log": np.log1p(g.size()),
        "pop_recent": np.log1p(g["_decay"].sum()),
        "item_avg_rating": g["rating"].mean(),
    })
    item_feats = onehot.join(item_stats, how="left").fillna(
        {"pop_log": 0.0, "pop_recent": 0.0, "item_avg_rating": train["rating"].mean()})

    # ---- users ----------------------------------------------------------
    pos = train[train["label"] == 1].sort_values("timestamp", kind="stable")
    gu = train.groupby("user_id")
    user_feats = pd.DataFrame({"n_interactions": gu.size().astype(float),
                               "n_positives": gu["label"].sum().astype(float),
                               "avg_rating": gu["rating"].mean()})
    pos_genres = onehot.reindex(pos["item_id"]).to_numpy()
    aff = pd.DataFrame(pos_genres, columns=[f"aff_{x}" for x in GENRES], index=pos.index)
    aff["user_id"] = pos["user_id"].to_numpy()
    aff = aff.groupby("user_id").sum()
    aff = aff.div(aff.sum(axis=1).replace(0, 1), axis=0)
    user_feats = user_feats.join(aff, how="left").fillna(0.0)
    hist = pos.groupby("user_id")["item_id"].apply(lambda s: list(s.iloc[-HISTORY_LEN:]))
    user_feats["history"] = hist.reindex(user_feats.index).map(lambda h: h if isinstance(h, list) else [])
    user_feats = user_feats.sort_index()

    if vocab is not None:
        user_index, item_index = vocab
    else:
        user_index = {u: i + 1 for i, u in enumerate(user_feats.index)}
        item_index = {it: i + 1 for i, it in enumerate(item_feats.index)}
    seen = {u: set(g) for u, g in train.groupby("user_id")["item_id"]}
    return FeatureTables(user_index, item_index, user_feats, item_feats, as_of, seen)


def save_tables(tables: FeatureTables, directory) -> None:
    import json
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tables.user_features.to_parquet(d / "user_features.parquet")
    tables.item_features.to_parquet(d / "item_features.parquet")
    pd.DataFrame({"user_id": list(tables.seen), "items": [sorted(v) for v in tables.seen.values()]}).to_parquet(
        d / "seen.parquet", index=False)
    (d / "vocab.json").write_text(json.dumps({"user_index": tables.user_index, "item_index": tables.item_index,
                                              "as_of": str(tables.as_of)}))


def load_tables(directory) -> FeatureTables:
    import json
    from pathlib import Path

    d = Path(directory)
    v = json.loads((d / "vocab.json").read_text())
    uf = pd.read_parquet(d / "user_features.parquet")
    uf["history"] = uf["history"].map(list)
    seen_df = pd.read_parquet(d / "seen.parquet")
    seen = {u: set(items) for u, items in zip(seen_df["user_id"], seen_df["items"])}
    return FeatureTables(v["user_index"], v["item_index"], uf, pd.read_parquet(d / "item_features.parquet"),
                         pd.Timestamp(v["as_of"]), seen)
