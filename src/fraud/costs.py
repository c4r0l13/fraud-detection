"""Business-aware evaluation: choose the alert threshold by money, not by F1.

Cost model (per transaction, all values configurable):
    missed fraud        -> lose the transaction amount (chargeback)
    any alert           -> ``review_cost`` (analyst time + customer contact)
    caught fraud        -> loss avoided

Missing a $1,200 laptop matters more than missing a $2 card test, which is
why the model is judged on *dollars* caught, not only on counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    review_cost: float = 5.0

    def total_cost(self, y: np.ndarray, amount: np.ndarray, alert: np.ndarray) -> float:
        y, amount, alert = np.asarray(y), np.asarray(amount), np.asarray(alert).astype(bool)
        missed = amount[(y == 1) & ~alert].sum()
        return float(missed + self.review_cost * alert.sum())

    def baseline_cost(self, y: np.ndarray, amount: np.ndarray) -> float:
        """Cost with no fraud screening at all: every fraud is lost."""
        return float(np.asarray(amount)[np.asarray(y) == 1].sum())


def cost_curve(y, amount, scores, cost: CostModel, n: int = 400) -> pd.DataFrame:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.90, 0.99995, n)))
    rows = [
        {"threshold": t, "cost": cost.total_cost(y, amount, scores >= t), "alerts": int((scores >= t).sum())}
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def best_threshold(y, amount, scores, cost: CostModel) -> float:
    curve = cost_curve(y, amount, scores, cost)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


def alert_metrics(y, amount, alert, days: float) -> dict[str, float]:
    y, amount, alert = np.asarray(y), np.asarray(amount), np.asarray(alert).astype(bool)
    tp = (alert & (y == 1)).sum()
    return {
        "alerts_per_day": float(alert.sum() / days),
        "precision": float(tp / max(alert.sum(), 1)),
        "recall": float(tp / max((y == 1).sum(), 1)),
        "dollar_recall": float(amount[alert & (y == 1)].sum() / max(amount[y == 1].sum(), 1e-9)),
    }
