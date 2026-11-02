import numpy as np
import pandas as pd
import pytest

from fraud.costs import CostModel, alert_metrics, best_threshold
from fraud.features import FEATURES, build_features
from fraud.simulator import SimConfig, simulate


@pytest.fixture(scope="module")
def small_sim():
    cfg = SimConfig(n_customers=150, n_merchants=80, days=30, card_testing_rate=0.1, ato_rate=0.1, poc_merchants=2)
    return simulate(cfg)


def test_simulation_is_reproducible():
    cfg = SimConfig(n_customers=40, n_merchants=30, days=10)
    a, _, _ = simulate(cfg)
    b, _, _ = simulate(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_simulation_shape_and_labels(small_sim):
    tx, cust, merch = small_sim
    assert tx["timestamp"].is_monotonic_increasing
    assert set(tx["scenario"]) >= {"legit", "card_testing", "account_takeover"}
    assert ((tx["scenario"] != "legit") == (tx["is_fraud"] == 1)).all()
    assert tx["amount"].gt(0).all()


def test_features_complete_and_aligned(small_sim):
    tx, cust, _ = small_sim
    X = build_features(tx, cust)
    assert list(X.columns) == FEATURES
    assert X.index.equals(tx.index)
    assert X.drop(columns="dist_home_km").notna().all().all()


def test_features_do_not_look_ahead():
    """Velocity features must only count strictly earlier transactions of the same card."""
    tx = pd.DataFrame(
        {
            "transaction_id": [0, 1, 2],
            "timestamp": pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:10", "2026-01-01 10:20"]),
            "customer_id": [1, 1, 1],
            "merchant_id": [5, 5, 6],
            "category": ["retail"] * 3,
            "amount": [2.0, 3.0, 500.0],
            "online": [True] * 3,
            "device_id": ["d", "d", "x"],
            "m_lat": [19.4] * 3,
            "m_lon": [-99.1] * 3,
        }
    )
    cust = pd.DataFrame({"customer_id": [1], "home_lat": [19.4], "home_lon": [-99.1]})
    X = build_features(tx, cust)
    assert X["n_tx_1h"].tolist() == [0, 1, 2]
    assert X["n_small_1h"].tolist() == [0, 1, 2]
    assert X["new_device"].tolist() == [1, 0, 1]
    assert X["new_merchant"].tolist() == [1, 0, 1]
    assert X.loc[2, "amount_vs_card_mean"] == pytest.approx(500 / 2.5)


def test_cost_model_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    amount = np.array([10, 10, 10, 1000, 2])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.35])
    cost = CostModel(review_cost=5)
    assert cost.baseline_cost(y, amount) == 1002
    assert cost.total_cost(y, amount, scores >= 0.9) == 2 + 5  # miss the $2, review one alert
    # catching the $2 fraud is not worth a $5 review -> threshold should stay high
    assert best_threshold(y, amount, scores, cost) >= 0.35
    m = alert_metrics(y, amount, scores >= 0.9, days=1)
    assert m["recall"] == 0.5 and m["dollar_recall"] == pytest.approx(1000 / 1002)
