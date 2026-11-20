import time

import pytest
from fastapi.testclient import TestClient

from recsys.experiments.ab import Experiment, two_proportion_test
from recsys.serving.api import create_app
from recsys.serving.cache import MemoryCache


def _warm_user(service):
    return next(u for u in service.tables.user_features.index
                if service.policy.route(service.tables, u) == "two_stage")


def test_two_stage_excludes_seen_and_is_sorted(service):
    u = _warm_user(service)
    r = service.recommend(u, k=10, use_cache=False)
    assert r.strategy == "two_stage" and r.variant in {"dcn_v2", "wide_deep"}
    ids = [i.item_id for i in r.items]
    assert len(ids) == 10 and len(set(ids)) == 10
    assert not set(ids) & service.tables.seen_items(u)
    scores = [i.score for i in r.items]
    assert scores == sorted(scores, reverse=True)


def test_cold_start_routes(service):
    assert service.recommend("brand_new_user", k=5).strategy == "popularity"
    r = service.recommend("brand_new_user_2", k=5, onboarding_genres=["Horror"])
    assert r.strategy == "content"
    top_genres = [g for it in r.items for g in it.genres]
    assert "Horror" in top_genres


def test_cache_hit_and_ttl(service):
    u = _warm_user(service)
    first = service.recommend(u, k=7)
    second = service.recommend(u, k=7)
    assert second.cached and [i.item_id for i in second.items] == [i.item_id for i in first.items]
    c = MemoryCache()
    c.set("a", 1, ttl_s=0)
    time.sleep(0.01)
    assert c.get("a") is None


def test_assignment_is_sticky_and_balanced():
    exp = Experiment("e1")
    a = [exp.assign(f"u{i}") for i in range(10_000)]
    assert a[:50] == [exp.assign(f"u{i}") for i in range(50)]
    share = a.count("dcn_v2") / len(a)
    assert 0.47 < share < 0.53
    assert Experiment("e2").assign("u1") in exp.variants


def test_z_test():
    sig = two_proportion_test(100, 10_000, 160, 10_000)
    assert sig["p_value"] < 0.001 and sig["ci95_low"] > 0
    flat = two_proportion_test(100, 10_000, 102, 10_000)
    assert flat["p_value"] > 0.5


def test_simulated_online_ab(service, split):
    from recsys.experiments.simulate import simulate

    seen = service.tables.seen
    truth = {u: set(g) - seen.get(u, set()) for u, g in split.test[split.test.label == 1].groupby("user_id")["item_id"]}
    rep = simulate(service, {u: t for u, t in truth.items() if t}, sessions=300)
    v = rep["variants"]
    assert set(v) == {"dcn_v2", "wide_deep"}
    assert all(x["impression"] > 0 for x in v.values())
    assert 0 <= rep["p_value"] <= 1


@pytest.fixture()
def client(service):
    with TestClient(create_app(service)) as c:
        yield c


def test_api_endpoints(client, service):
    u = _warm_user(service)
    assert client.get("/health").json()["status"] == "ok"
    r = client.get(f"/v1/recommendations/{u}", params={"k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 5 and body["strategy"] == "two_stage"
    click = client.post("/v1/events/click", json={"user_id": u, "item_id": body["items"][0]["item_id"],
                                                 "position": 0, "request_id": body["request_id"]})
    assert click.json()["logged"]
    assert "variants" in client.get("/v1/experiments/report").json()
    item = body["items"][0]["item_id"]
    sim = client.get(f"/v1/items/{item}/similar", params={"k": 3}).json()
    assert len(sim) == 3 and item not in [s["item_id"] for s in sim]
    assert client.get("/v1/items/nope/similar").status_code == 404
    assert client.get(f"/v1/recommendations/{u}", params={"k": 0}).status_code == 422
    assert "recsys_requests_total" in client.get("/metrics").text
    assert client.post("/v1/cache/invalidate", params={"user_id": u}).json()["deleted"] >= 1


def test_power_analysis():
    from recsys.experiments.ab import required_sample_size

    n = required_sample_size(0.023, 0.05)
    assert 250_000 < n < 350_000  # ~5% lift on a 2.3% CTR needs ~300k impressions per arm
    assert required_sample_size(0.023, 0.20) < n
