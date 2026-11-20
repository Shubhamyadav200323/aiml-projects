from fraud.features.online import OnlineFeatureEngine
from fraud.repository.cases import CaseRepository
from fraud.serving.scorer import FraudScorer, ModelBundle
from fraud.streaming.consumer import run
from fraud.streaming.kafka_io import MemoryBroker
from fraud.streaming.producer import replay


def test_replay_then_consume_end_to_end(artifacts, transactions, tmp_path):
    out, _ = artifacts
    broker = MemoryBroker()
    sent = replay(transactions, broker, "transactions", limit=500)
    assert sent == 500
    first = broker.topics["transactions"][0][1]
    assert "is_fraud" not in first, "labels must never be streamed to the scorer"

    repo = CaseRepository(f"sqlite:///{tmp_path / 'c.db'}")
    scorer = FraudScorer(ModelBundle.load(out), OnlineFeatureEngine(), repo)
    n = run(scorer, broker, "transactions", "fraud-scores", commit_every=100)
    assert n == 500
    assert len(broker.topics["fraud-scores"]) == 500
    assert broker.committed >= 5
    assert not broker.topics.get("transactions.dlq")


def test_poison_message_goes_to_dlq(artifacts):
    out, _ = artifacts
    broker = MemoryBroker()
    broker.produce("transactions", "k", {"transaction_id": "bad"})
    scorer = FraudScorer(ModelBundle.load(out), OnlineFeatureEngine())
    run(scorer, broker, "transactions", "fraud-scores")
    assert len(broker.topics["transactions.dlq"]) == 1
