import pytest
from fastapi.testclient import TestClient

from fraud.features.online import OnlineFeatureEngine
from fraud.repository.cases import CaseRepository
from fraud.serving.api import create_app
from fraud.serving.scorer import FraudScorer, ModelBundle


@pytest.fixture()
def client(artifacts, tmp_path):
    out, _ = artifacts
    scorer = FraudScorer(ModelBundle.load(out), OnlineFeatureEngine(),
                         CaseRepository(f"sqlite:///{tmp_path / 'cases.db'}"))
    with TestClient(create_app(scorer)) as c:
        yield c


def _txn(i: int, amount: float = 25.0, device: str = "D1", user: str = "U1") -> dict:
    return {"transaction_id": f"X{i}", "timestamp": f"2026-02-01T10:{i:02d}:00Z", "user_id": user,
            "card_id": "C1", "device_id": device, "merchant_id": "M1", "location": "L1",
            "amount": amount, "channel": "CARD_PRESENT"}


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "model_version" in client.get("/ready").json()


def test_score_contract(client):
    r = client.post("/v1/score", json=_txn(1))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["latency_ms"] < 250


def test_validation_rejects_bad_input(client):
    bad = _txn(2)
    bad["amount"] = -5
    assert client.post("/v1/score", json=bad).status_code == 422


def test_suspicious_burst_creates_case_and_feedback_loop(client, artifacts):
    out, _ = artifacts
    bundle = ModelBundle.load(out)
    ring_user = bundle.rings[0]["members"][0] if bundle.rings else "U1"
    for i in range(20):  # establish a normal history
        client.post("/v1/score", json=_txn(i, 20.0, user=ring_user))
    hits = [client.post("/v1/score", json={**_txn(30 + i, 900.0, device=f"DX{i}", user=ring_user),
                                           "channel": "TRANSFER"}).json() for i in range(5)]
    flagged = [h for h in hits if h["decision"] != "approve"]
    assert flagged, f"burst should be flagged: {[h['fraud_score'] for h in hits]}"
    case_id = flagged[0]["case_id"]
    assert flagged[0]["reasons"], "flagged decisions must carry explanations"

    cases = client.get("/v1/cases", params={"status": "open"}).json()
    assert any(c["id"] == case_id for c in cases)
    upd = client.patch(f"/v1/cases/{case_id}", json={"status": "confirmed_fraud", "analyst": "qa"})
    assert upd.status_code == 200 and upd.json()["label"] == 1
    assert client.get("/v1/cases/stats").json().get("confirmed_fraud") == 1
    assert client.get("/v1/cases/99999").status_code == 404


def test_metrics_and_rings(client):
    client.post("/v1/score", json=_txn(3))
    assert "fraud_decisions_total" in client.get("/metrics").text
    assert isinstance(client.get("/v1/rings").json(), list)
