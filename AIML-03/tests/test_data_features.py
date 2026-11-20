import numpy as np
import pandas as pd
import pytest

from recsys.data.movielens import GENRES, load_movielens
from recsys.features.tables import build_feature_tables, load_tables, save_tables
from recsys.retrieval.inputs import retrieval_examples


def test_temporal_split_has_no_overlap(split):
    assert split.train["timestamp"].max() < split.t1 <= split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.t2 <= split.test["timestamp"].min()
    assert split.summary()["test_cold_users"] > 0, "late joiners must create real cold-start users"


def test_feature_tables_point_in_time(split, dataset):
    tab = build_feature_tables(split.train, dataset.items, split.t1)
    aff = tab.user_features[[f"aff_{g}" for g in GENRES]].to_numpy()
    assert np.allclose(aff.sum(1)[aff.sum(1) > 0], 1.0)
    assert tab.item_features.notna().all().all()
    assert len(tab.item_features) == len(dataset.items), "whole catalogue, incl. never-seen items"
    for u, h in tab.user_features["history"].items():
        assert set(h) <= tab.seen_items(u)


def test_vocab_is_stable_across_refreshes(split, dataset):
    t1 = build_feature_tables(split.train, dataset.items, split.t1)
    t2 = build_feature_tables(pd.concat([split.train, split.val]), dataset.items, split.t2,
                              vocab=(t1.user_index, t1.item_index))
    assert t2.user_index is t1.user_index
    new_users = set(t2.user_features.index) - set(t1.user_index)
    assert all(t2.user_index.get(u, 0) == 0 for u in new_users)


def test_retrieval_examples_never_see_their_target(split, dataset):
    tab = build_feature_tables(split.train, dataset.items, split.t1)
    ex = retrieval_examples(split.train, tab)
    assert not np.any(ex["history"] == ex["item_idx"][:, None]), "target leaked into its own history"
    assert np.isfinite(ex["user_dense"]).all() and np.isfinite(ex["item_dense"]).all()
    assert (ex["item_prob"] > 0).all()


def test_tables_roundtrip(tmp_path, split, dataset):
    tab = build_feature_tables(split.train, dataset.items, split.t1)
    save_tables(tab, tmp_path)
    back = load_tables(tmp_path)
    assert back.user_index == tab.user_index and back.seen == tab.seen
    pd.testing.assert_frame_equal(back.item_features, tab.item_features)


def test_movielens_loader(tmp_path):
    pd.DataFrame({"userId": [1, 1], "movieId": [10, 20], "rating": [4.0, 2.5],
                  "timestamp": [1_000_000, 1_000_100]}).to_csv(tmp_path / "ratings.csv", index=False)
    pd.DataFrame({"movieId": [10, 20], "title": ["A", "B"],
                  "genres": ["Action|Comedy", "(no genres listed)"]}).to_csv(tmp_path / "movies.csv", index=False)
    ds = load_movielens(tmp_path)
    assert ds.interactions["user_id"].tolist() == ["u1", "u1"]
    assert ds.items["genres"].tolist() == [["Action", "Comedy"], []]


def test_spark_job_matches_pandas(split, dataset):
    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    from recsys.features.spark_job import build

    try:
        spark = SparkSession.builder.master("local[1]").appName("t").config("spark.ui.enabled", "false").getOrCreate()
    except Exception as exc:  # no JVM available
        pytest.skip(f"spark unavailable: {exc}")
    raw = dataset.interactions.copy()
    raw["timestamp"] = raw["timestamp"].astype("int64") // 10**9
    movies = dataset.items.assign(genres=dataset.items["genres"].map("|".join))
    _, users, items = build(spark, spark.createDataFrame(raw), spark.createDataFrame(movies),
                            str(split.t1))
    pd_tab = build_feature_tables(split.train, dataset.items, split.t1)
    su = users.toPandas().set_index("user_id").sort_index()
    pu = pd_tab.user_features.loc[su.index]
    np.testing.assert_allclose(su["n_interactions"], pu["n_interactions"])
    np.testing.assert_allclose(su["n_positives"], pu["n_positives"])
    np.testing.assert_allclose(su[[f"aff_{g}" for g in GENRES]], pu[[f"aff_{g}" for g in GENRES]], atol=1e-9)
    si = items.toPandas().set_index("item_id").loc[pd_tab.item_features.index]
    np.testing.assert_allclose(si["pop_log"], pd_tab.item_features["pop_log"], atol=1e-9)
    spark.stop()
    assert pyspark
