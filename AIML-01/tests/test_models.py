import json

import numpy as np

from fraud.models.ensemble import Blender
from fraud.models.gnn import MeanSAGEConv, train_gnn
from fraud.models.graph import build_account_graph, detect_rings
from fraud.models.metrics import evaluate
from fraud.models.xgb import XGBFraudModel


def test_metrics_reject_accuracy_trap():
    y = np.array([0] * 990 + [1] * 10)
    m = evaluate(y, np.zeros(1000), threshold=0.5)
    assert m["recall"] == 0 and m["pr_auc"] <= 0.02  # all-negative model is useless


def test_pipeline_beats_baseline_by_wide_margin(artifacts):
    _, report = artifacts
    xgb_m = report["test"]["xgboost"]
    assert xgb_m["pr_auc"] > 10 * xgb_m["baseline_pr_auc"]
    assert xgb_m["roc_auc"] > 0.9


def test_artifacts_roundtrip(artifacts, transactions):
    out, _ = artifacts
    for f in ["xgb.json", "gnn.pt", "graph_scores.parquet", "policy.json", "rings.json",
              "metrics.json", "reference_features.parquet"]:
        assert (out / f).exists(), f
    policy = json.loads((out / "policy.json").read_text())
    assert 0 <= policy["weight_xgb"] <= 1
    assert policy["decline_threshold"] >= policy["review_threshold"]
    model = XGBFraudModel.load(out)
    from fraud.features import build_features

    feats = build_features(transactions.head(200))
    p = model.predict_proba(feats)
    assert p.shape == (200,) and ((p >= 0) & (p <= 1)).all()


def test_mean_sage_aggregation():
    import torch

    conv = MeanSAGEConv(2, 2)
    with torch.no_grad():
        conv.lin_self.weight.zero_()
        conv.lin_self.bias.zero_()
        conv.lin_neigh.weight.copy_(torch.eye(2))
    x = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]])
    ei = torch.tensor([[1, 2], [0, 0]])  # node 0 receives from 1 and 2
    out = conv(x, ei)
    assert torch.allclose(out[0], torch.tensor([1.5, 2.5]))
    assert torch.allclose(out[1], torch.zeros(2))


def test_graph_and_gnn_find_rings(transactions):
    g = build_account_graph(transactions)
    assert g.edge_index.shape[0] == 2 and g.edge_index.shape[1] > 0
    res = train_gnn(g, epochs=60)
    assert res.metrics["roc_auc"] > 0.8
    rings = detect_rings(g, res.scores, threshold=0.5)
    fraud_users = set(transactions.loc[transactions.is_fraud == 1, "user_id"])
    flagged = {u for r in rings for u in r.members}
    assert flagged, "at least one ring should be detected"
    assert len(flagged & fraud_users) / len(flagged) > 0.5


def test_blender_picks_best_weight():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    good = y * 0.8 + rng.random(2000) * 0.3
    noise = rng.random(2000)
    blender, _ = Blender.fit(y, good, noise)
    assert blender.weight_xgb >= 0.8
