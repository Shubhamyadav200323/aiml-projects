"""End-to-end training + offline evaluation.

    python -m recsys.pipeline train --dataset synthetic --out artifacts
    python -m recsys.pipeline train --dataset movielens --path data/ml-25m --out artifacts

Timeline (global temporal split)::

    |------------- train -------------|---- val ----|---- test ----|
                                      t1            t2
    two-tower fit on train positives (features as of t1)
    rankers fit on val-window labels over two-tower candidates (serving distribution)
    serving snapshot = features as of t2 (train + val); evaluate on test

Artifacts follow Triton's model-repository layout so `triton/` can mount them directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data import load_dataset, temporal_split
from recsys.evaluation import evaluate_lists
from recsys.features.tables import build_feature_tables, save_tables
from recsys.ranking.dataset import build_ranking_set, seen_items
from recsys.ranking.features import RankingFeaturizer
from recsys.ranking.models import RankerConfig, export_ranker, predict, train_ranker
from recsys.retrieval.index import ItemIndex
from recsys.retrieval.inputs import item_inputs, retrieval_examples, user_inputs
from recsys.retrieval.two_tower import (
    TrainConfig,
    encode,
    export_item_tower,
    export_user_tower,
    train_two_tower,
)
from recsys.serving.cold_start import ColdStartPolicy, ContentRecommender, PopularityRecommender

log = logging.getLogger("recsys.pipeline")
RANKER_KINDS = ("dcn_v2", "wide_deep")


def _round(d: dict) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}


def train(dataset, out: str | Path, retrieval_cfg: TrainConfig | None = None,
          ranker_cfg: RankerConfig | None = None, n_candidates: int = 200, export: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sp = temporal_split(dataset.interactions)
    log.info("split %s", sp.summary())

    # ---- stage 1: retrieval ---------------------------------------------
    tab1 = build_feature_tables(sp.train, dataset.items, sp.t1)
    vocab = (tab1.user_index, tab1.item_index)
    n_users, n_items = len(tab1.user_index), len(tab1.item_index)
    tt, tt_loss = train_two_tower(retrieval_examples(sp.train, tab1), n_users, n_items, retrieval_cfg)
    idx1 = ItemIndex(tab1.item_ids).build(encode(tt.item_tower, item_inputs(tab1)))
    users1 = list(tab1.user_features.index)
    uvec1 = dict(zip(users1, encode(tt.user_tower, user_inputs(tab1, users1))))

    # ---- stage 2: rankers on val-window labels ---------------------------
    rank_data = build_ranking_set(sp.val, tab1, uvec1, idx1, seen_items(sp.train), n_candidates)
    rankers, ranker_hist = {}, {}
    for kind in RANKER_KINDS:
        rankers[kind], ranker_hist[kind] = train_ranker(kind, rank_data, n_users, n_items, cfg=ranker_cfg)

    # ---- serving snapshot (as of t2) + offline evaluation on test --------
    hist2 = pd.concat([sp.train, sp.val], ignore_index=True)
    tab2 = build_feature_tables(hist2, dataset.items, sp.t2, vocab=vocab)
    idx2 = ItemIndex(tab2.item_ids).build(encode(tt.item_tower, item_inputs(tab2)))
    seen2 = seen_items(hist2)
    test_pos = sp.test[sp.test["label"] == 1]
    truth = {u: set(g) - seen2.get(u, set()) for u, g in test_pos.groupby("user_id")["item_id"]}
    truth = {u: s for u, s in truth.items() if s}
    policy = ColdStartPolicy()
    routes = {u: policy.route(tab2, u) for u in truth}
    warm = [u for u, r in routes.items() if r == "two_stage"]
    cold = [u for u, r in routes.items() if r != "two_stage"]

    pop = PopularityRecommender(tab2)
    content = ContentRecommender(tab2)
    K = 100
    results: dict[str, dict] = {}
    results["popularity"] = evaluate_lists(
        {u: [i for i, _ in pop.recommend(K, seen2.get(u))] for u in warm}, truth, catalogue_size=len(tab2.item_ids))
    uvec2 = encode(tt.user_tower, user_inputs(tab2, warm))
    cands, scores = idx2.search(uvec2, n_candidates, [seen2.get(u, set()) for u in warm])
    results["two_tower"] = evaluate_lists({u: c[:K] for u, c in zip(warm, cands)}, truth,
                                          catalogue_size=len(tab2.item_ids))
    feat2 = RankingFeaturizer(tab2)
    latency = {}
    for kind, model in rankers.items():
        recs, t = {}, time.perf_counter()
        for u, c, s in zip(warm, cands, scores):
            p = predict(model, feat2.build(u, c, s[: len(c)]))
            recs[u] = [c[j] for j in np.argsort(-p)][:K]
        latency[kind] = (time.perf_counter() - t) / max(len(warm), 1) * 1000
        results[f"two_stage_{kind}"] = evaluate_lists(recs, truth, catalogue_size=len(tab2.item_ids))
    # cold users: simulate a one-shot onboarding signal = genres of their first test positive,
    # which is then removed from the ground truth so neither method sees it
    first = test_pos.sort_values("timestamp").groupby("user_id").head(1).set_index("user_id")["item_id"]
    genres_of = dataset.items.set_index("item_id")["genres"]
    cold_truth = {u: truth[u] - {first[u]} for u in cold if u in first.index}
    cold_truth = {u: t for u, t in cold_truth.items() if t}
    cold_eval = {
        "popularity": evaluate_lists({u: [i for i, _ in pop.recommend(K, seen2.get(u, set()) | {first[u]})]
                                      for u in cold_truth}, cold_truth),
        "content+popularity": evaluate_lists(
            {u: [i for i, _ in content.recommend(content.affinity_from_genres(list(genres_of[first[u]])), K,
                                                 seen2.get(u, set()) | {first[u]})] for u in cold_truth},
            cold_truth),
    }

    # ---- artifacts --------------------------------------------------------
    if export:
        models = out / "models"
        export_user_tower(tt.user_tower, models / "user_tower" / "1" / "model.savedmodel")
        export_item_tower(tt.item_tower, models / "item_tower" / "1" / "model.savedmodel")
        for kind, model in rankers.items():
            export_ranker(model, models / f"ranker_{kind}" / "1" / "model.savedmodel")
        idx2.save(out / "index")
        save_tables(tab2, out / "features")
        dataset.items.to_parquet(out / "items.parquet", index=False)

    report = {
        "split": sp.summary(),
        "ranking_set": {"rows": int(len(rank_data["label"])), "positive_rate": float(rank_data["label"].mean())},
        "warm_test_users": len(warm), "cold_test_users": len(cold_truth),
        "warm": {k: _round(v) for k, v in results.items()},
        "cold": {k: _round(v) for k, v in cold_eval.items()},
        "ranker_latency_ms_per_user": _round(latency),
        "two_tower_loss": [round(x, 2) for x in tt_loss],
        "ranker_history": ranker_hist,
        "index_backend": idx2.backend,
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="recsys.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", default="synthetic")
    t.add_argument("--path")
    t.add_argument("--out", default="artifacts")
    t.add_argument("--epochs", type=int, default=8)
    a = p.parse_args()
    rep = train(load_dataset(a.dataset, a.path), a.out, TrainConfig(epochs=a.epochs))
    print(json.dumps({"warm": rep["warm"], "cold": rep["cold"]}, indent=2))


if __name__ == "__main__":
    main()
