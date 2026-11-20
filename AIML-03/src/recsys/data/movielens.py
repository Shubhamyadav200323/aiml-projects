"""MovieLens loaders and a MovieLens-shaped synthetic generator.

Canonical tables
----------------
interactions: user_id (str), item_id (str), rating (float), timestamp (datetime)
items:        item_id (str), title (str), genres (list[str])

MovieLens 25M (GroupLens) ships `ratings.csv` and `movies.csv` with exactly
these concepts, so `load_movielens` is a rename. The synthetic generator is
used by CI and the demo; it plants learnable structure (user genre tastes,
item quality, Zipf popularity, taste drift over time) so that model
comparisons are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]


@dataclass
class Dataset:
    interactions: pd.DataFrame
    items: pd.DataFrame

    @property
    def n_users(self) -> int:
        return self.interactions["user_id"].nunique()

    @property
    def n_items(self) -> int:
        return len(self.items)


def load_movielens(directory: str | Path) -> Dataset:
    d = Path(directory)
    ratings = pd.read_csv(d / "ratings.csv")
    movies = pd.read_csv(d / "movies.csv")
    inter = pd.DataFrame({
        "user_id": "u" + ratings["userId"].astype(str),
        "item_id": "i" + ratings["movieId"].astype(str),
        "rating": ratings["rating"].astype(float),
        "timestamp": pd.to_datetime(ratings["timestamp"], unit="s"),
    })
    items = pd.DataFrame({
        "item_id": "i" + movies["movieId"].astype(str),
        "title": movies["title"],
        "genres": movies["genres"].str.split("|").map(lambda g: [x for x in g if x in GENRES]),
    })
    return Dataset(inter.sort_values("timestamp", kind="stable").reset_index(drop=True), items)


def generate_synthetic(n_users: int = 2_000, n_items: int = 1_500, interactions_per_user: int = 60,
                       days: int = 365, seed: int = 3) -> Dataset:
    rng = np.random.default_rng(seed)
    g = len(GENRES)
    # items: 1-3 genres, latent quality, Zipf popularity
    item_genres = np.zeros((n_items, g))
    for i in range(n_items):
        item_genres[i, rng.choice(g, rng.integers(1, 4), replace=False)] = 1
    quality = rng.normal(0, 1, n_items)
    pop = 1.0 / np.arange(1, n_items + 1) ** 0.8
    pop = pop[rng.permutation(n_items)]
    release = rng.uniform(-0.5, 1.0, n_items)  # some items appear mid-way (new-item cold start)

    # users: sparse genre tastes that slowly drift
    taste0 = rng.dirichlet(np.full(g, 0.3), n_users)
    taste1 = 0.7 * taste0 + 0.3 * rng.dirichlet(np.full(g, 0.3), n_users)
    # 12% of users sign up near the end of the timeline -> genuine cold-start users at test time
    late = rng.random(n_users) < 0.12
    join = np.where(late, rng.uniform(0.95, 0.995, n_users), rng.uniform(0, 0.85, n_users))
    n_per_user = rng.poisson(interactions_per_user, n_users).clip(5)
    n_per_user[late] = rng.integers(1, 8, late.sum())  # newcomers have only a handful of events

    rows_u, rows_i, rows_r, rows_t = [], [], [], []
    for u in range(n_users):
        k = n_per_user[u]
        t = np.sort(rng.uniform(join[u], 1.0, k))
        for tt in t:
            taste = (1 - tt) * taste0[u] + tt * taste1[u]
            aff = item_genres @ taste / item_genres.sum(1)
            avail = release <= tt
            logits = 6 * aff + np.log(pop) + 0.3 * quality
            logits[~avail] = -np.inf
            p = np.exp(logits - logits[avail].max())
            p /= p.sum()
            i = rng.choice(n_items, p=p)
            r = np.clip(np.round(2.6 + 6 * aff[i] + 0.6 * quality[i] + rng.normal(0, 0.7)), 1, 5)
            rows_u.append(u)
            rows_i.append(i)
            rows_r.append(r)
            rows_t.append(tt)
    start = pd.Timestamp("2025-01-01")
    inter = pd.DataFrame({
        "user_id": [f"u{u}" for u in rows_u],
        "item_id": [f"i{i}" for i in rows_i],
        "rating": np.asarray(rows_r, dtype=float),
        "timestamp": start + pd.to_timedelta(np.asarray(rows_t) * days * 86_400, unit="s"),
    }).drop_duplicates(["user_id", "item_id"], keep="first")
    items = pd.DataFrame({
        "item_id": [f"i{i}" for i in range(n_items)],
        "title": [f"Title {i}" for i in range(n_items)],
        "genres": [[GENRES[j] for j in np.where(item_genres[i])[0]] for i in range(n_items)],
    })
    return Dataset(inter.sort_values("timestamp", kind="stable").reset_index(drop=True), items)


def load_dataset(name: str = "synthetic", path: str | Path | None = None, **kwargs) -> Dataset:
    if name == "synthetic":
        return generate_synthetic(**kwargs)
    if name in {"movielens", "ml-25m", "ml"}:
        if path is None:
            raise ValueError("movielens requires --path to the extracted ml-25m directory")
        return load_movielens(path)
    raise ValueError(f"unknown dataset {name}")
