import numpy as np

from recsys.evaluation import evaluate_lists, ndcg_at_k, recall_at_k
from recsys.retrieval.index import ItemIndex


def test_metrics_known_values():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert ndcg_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert abs(ndcg_at_k(["x", "a"], {"a"}, 10) - 1 / np.log2(3)) < 1e-9
    m = evaluate_lists({"u": ["a", "b"]}, {"u": {"b"}}, ks=(1, 2))
    assert m["hit@1"] == 0 and m["hit@2"] == 1 and m["mrr"] == 0.5


def test_index_exact_search_and_exclusion(tmp_path):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = [f"i{j}" for j in range(500)]
    idx = ItemIndex(ids).build(v)
    q = v[:3]
    res, scores = idx.search(q, 5)
    assert [r[0] for r in res] == ["i0", "i1", "i2"]  # self is nearest under cosine
    res_ex, _ = idx.search(q, 5, exclude=[{"i0"}, set(), set()])
    assert "i0" not in res_ex[0] and len(res_ex[0]) == 5
    idx.save(tmp_path)
    back, _ = ItemIndex.load(tmp_path).search(q, 5)
    assert back == res


def test_ivf_index_close_to_exact():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(2000, 32)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = [str(j) for j in range(2000)]
    exact, _ = ItemIndex(ids, "flat").build(v).search(v[:50], 10)
    approx, _ = ItemIndex(ids, "ivf", nlist=32, nprobe=8).build(v).search(v[:50], 10)
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(exact, approx)])
    assert overlap > 0.6


def test_pipeline_report_invariants(artifacts):
    """Quality is measured on the full-size run (docs/RESULTS.md); here we check invariants
    that must hold at any scale."""
    _, rep = artifacts
    warm = rep["warm"]
    # rankers re-order the 100 retrieved candidates, so recall@100 cannot change
    for kind in ("dcn_v2", "wide_deep"):
        assert warm[f"two_stage_{kind}"]["recall@100"] == warm["two_tower"]["recall@100"]
    # far above a random ranking (expected recall@10 of random ~ 10 / 300 items)
    assert warm["two_tower"]["recall@10"] > 2 * 10 / 300
    assert max(warm[f"two_stage_{k}"]["ndcg@10"] for k in ("dcn_v2", "wide_deep")) > 0.03
    assert rep["cold_test_users"] > 0 and rep["index_backend"].startswith(("faiss", "numpy"))


def test_exported_models_have_serving_signatures(artifacts):
    from recsys.tf_compat import tf

    out, _ = artifacts
    for name in ["user_tower", "item_tower", "ranker_dcn_v2", "ranker_wide_deep"]:
        sm = tf.saved_model.load(str(out / "models" / name / "1" / "model.savedmodel"))
        assert "serving_default" in sm.signatures, name
