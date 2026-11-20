"""End-to-end training pipeline.

    python -m fraud.pipeline train --dataset synthetic --out artifacts
    python -m fraud.pipeline train --dataset paysim --path data/paysim.csv --imbalance smote

Stages: load -> point-in-time features -> temporal split -> XGBoost ->
account graph + GraphSAGE -> blend fit on validation -> ring detection ->
artifacts (+ MLflow run, + Feast offline tables).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fraud.config import get_settings
from fraud.data import load_dataset
from fraud.features import FEATURE_COLUMNS, build_features
from fraud.features.store import export_offline
from fraud.models.ensemble import Blender
from fraud.models.gnn import save_scores, score_graph, train_gnn
from fraud.models.graph import build_account_graph, detect_rings
from fraud.models.metrics import best_f1_threshold, evaluate
from fraud.models.tracking import Tracker
from fraud.models.xgb import XGBFraudModel

log = logging.getLogger("fraud.pipeline")


def temporal_split(df: pd.DataFrame, train: float = 0.7, val: float = 0.15
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by event time - random splits leak future behaviour into training."""
    t = df["timestamp"]
    t1, t2 = t.quantile(train), t.quantile(train + val)
    return df[t < t1], df[(t >= t1) & (t < t2)], df[t >= t2]


def _graph_score_lookup(users: pd.Series, table: pd.DataFrame, prior: float) -> np.ndarray:
    return users.map(table.set_index("user_id")["graph_score"]).fillna(prior).to_numpy()


def _high_precision_threshold(y: np.ndarray, p: np.ndarray, floor: float,
                              target_precision: float = 0.95) -> float:
    """Lowest score at which validation precision >= target: safe to auto-decline."""
    from sklearn.metrics import precision_recall_curve

    prec, _, thr = precision_recall_curve(y, p)
    ok = thr[prec[:-1] >= target_precision]
    return float(max(floor, ok.min() if ok.size else 0.99))


def train(transactions: pd.DataFrame, out: str | Path, imbalance: str = "weight",
          gnn_epochs: int = 200, seed: int = 42, export_feast: bool = False) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    tracker = Tracker(s.mlflow_tracking_uri, s.mlflow_experiment)
    t0 = time.perf_counter()

    with tracker.run("train", tags={"imbalance": imbalance}):
        feats = build_features(transactions)
        tr, va, te = temporal_split(feats)
        tracker.log_params({"n_train": len(tr), "n_val": len(va), "n_test": len(te),
                            "fraud_rate_train": round(float(tr["is_fraud"].mean()), 5),
                            "imbalance": imbalance, "features": len(FEATURE_COLUMNS)})

        # ---- tabular model --------------------------------------------------
        xgbm = XGBFraudModel(imbalance=imbalance).fit(tr, tr["is_fraud"], va, va["is_fraud"], seed)
        p_val_x, p_te_x = xgbm.predict_proba(va), xgbm.predict_proba(te)
        m_xgb = evaluate(te["is_fraud"], p_te_x)
        tracker.log_metrics(m_xgb, prefix="test_xgb_")

        # ---- graph model: fit on train-period graph, infer inductively ------
        g_train = build_account_graph(tr)
        gnn = train_gnn(g_train, epochs=gnn_epochs, seed=seed)
        g_full = build_account_graph(feats, norm=g_train.norm)
        node_scores = score_graph(gnn.model, g_full)
        score_table = pd.DataFrame({"user_id": g_full.users, "graph_score": node_scores})
        prior = float(np.median(node_scores))
        p_val_g = _graph_score_lookup(va["user_id"], score_table, prior)
        p_te_g = _graph_score_lookup(te["user_id"], score_table, prior)
        tracker.log_metrics(gnn.metrics, prefix="val_gnn_node_")
        tracker.log_metrics(evaluate(te["is_fraud"], p_te_g), prefix="test_gnn_")

        # ---- blend ---------------------------------------------------------
        blender, grid = Blender.fit(va["is_fraud"].to_numpy(), p_val_x, p_val_g)
        p_val_b = blender(p_val_x, p_val_g)
        p_te_b = blender(p_te_x, p_te_g)
        review_thr = best_f1_threshold(va["is_fraud"].to_numpy(), p_val_b)
        m_blend = evaluate(te["is_fraud"], p_te_b, threshold=review_thr)
        tracker.log_metrics(m_blend, prefix="test_blend_")
        tracker.log_params({"blend_weight_xgb": blender.weight_xgb})

        rings = detect_rings(g_full, node_scores, threshold=max(0.5, float(np.quantile(node_scores, 0.98))))

        # ---- artifacts -----------------------------------------------------
        xgbm.save(out)
        torch.save({"state_dict": gnn.model.state_dict(), "in_dim": g_train.x.shape[1],
                    "norm_mean": g_train.norm[0].tolist(), "norm_std": g_train.norm[1].tolist()},
                   out / "gnn.pt")
        save_scores(g_full, node_scores, out / "graph_scores.parquet")
        tr.sample(min(len(tr), 20_000), random_state=seed)[FEATURE_COLUMNS].to_parquet(
            out / "reference_features.parquet", index=False)
        policy = {"weight_xgb": blender.weight_xgb, "graph_prior": prior,
                  "review_threshold": review_thr,
                  "decline_threshold": _high_precision_threshold(va["is_fraud"].to_numpy(), p_val_b,
                                                                 review_thr)}
        (out / "policy.json").write_text(json.dumps(policy, indent=2))
        (out / "rings.json").write_text(json.dumps([asdict(r) for r in rings], indent=2))
        report = {
            "dataset": {"transactions": len(feats), "fraud_rate": float(feats["is_fraud"].mean()),
                        "train": len(tr), "val": len(va), "test": len(te)},
            "test": {"xgboost": m_xgb, "gnn": evaluate(te["is_fraud"], p_te_g), "blend": m_blend},
            "gnn_node_validation": gnn.metrics,
            "blend_grid_val_pr_auc": grid,
            "policy": policy,
            "rings_detected": len(rings),
            "feature_importance": xgbm.feature_importance(),
            "train_seconds": round(time.perf_counter() - t0, 2),
        }
        (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
        for f in ["xgb.json", "metrics.json", "policy.json", "rings.json"]:
            tracker.log_artifact(out / f)
        if export_feast:
            export_offline(feats)
    log.info("PR-AUC test: xgb=%.4f blend=%.4f (baseline %.4f)",
             m_xgb["pr_auc"], m_blend["pr_auc"], m_blend["baseline_pr_auc"])
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(prog="fraud.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", default="synthetic")
    t.add_argument("--path")
    t.add_argument("--out", default="artifacts")
    t.add_argument("--imbalance", choices=["weight", "smote", "none"], default="weight")
    t.add_argument("--gnn-epochs", type=int, default=200)
    t.add_argument("--export-feast", action="store_true")
    a = p.parse_args()
    if a.cmd == "train":
        rep = train(load_dataset(a.dataset, a.path), a.out, a.imbalance, a.gnn_epochs,
                    export_feast=a.export_feast)
        print(json.dumps(rep["test"], indent=2))


if __name__ == "__main__":
    main()
