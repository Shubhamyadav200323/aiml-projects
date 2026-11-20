from __future__ import annotations

import pytest

from fraud.data.synthetic import SyntheticConfig, generate_transactions


@pytest.fixture(scope="session")
def small_cfg() -> SyntheticConfig:
    return SyntheticConfig(n_users=800, days=20, n_rings=6, n_takeovers=25, seed=11)


@pytest.fixture(scope="session")
def transactions(small_cfg):
    return generate_transactions(small_cfg)


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory, transactions):
    """Train once per session; downstream tests reuse the bundle."""
    from fraud.pipeline import train

    out = tmp_path_factory.mktemp("artifacts")
    report = train(transactions, out, gnn_epochs=60)
    return out, report
