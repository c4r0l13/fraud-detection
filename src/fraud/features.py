"""Leak-free behavioural features.

Every feature for a transaction uses only that card's history *strictly
before* the transaction (``closed="left"`` windows, shifted expanding
statistics). This mirrors what a real-time scoring system can know at
authorization time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulator import CATEGORIES

CATEGORY_CODES = {c: i for i, c in enumerate(CATEGORIES)}

FEATURES = [
    "amount",
    "log_amount",
    "amount_vs_card_mean",
    "hour",
    "is_night",
    "online",
    "category_code",
    "new_device",
    "new_merchant",
    "dist_home_km",
    "hours_since_last",
    "n_tx_1h",
    "amount_1h",
    "n_small_1h",
    "n_tx_24h",
    "amount_24h",
    "n_devices_24h",
]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _rolling(df: pd.DataFrame, col: str, window: str, how: str) -> np.ndarray:
    r = df.groupby("customer_id", sort=False).rolling(window, on="timestamp", closed="left")[col]
    out = getattr(r, how)()
    return out.fillna(0).to_numpy()


def build_features(tx: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Return features aligned with ``tx`` (same index)."""
    df = tx.merge(customers[["customer_id", "home_lat", "home_lon"]], on="customer_id", how="left")
    df.index = tx.index
    df = df.sort_values(["customer_id", "timestamp"], kind="stable")
    g = df.groupby("customer_id", sort=False)

    out = pd.DataFrame(index=df.index)
    out["amount"] = df["amount"]
    out["log_amount"] = np.log1p(df["amount"])
    past_sum = g["amount"].cumsum() - df["amount"]
    past_n = g.cumcount()
    out["amount_vs_card_mean"] = np.where(past_n > 0, df["amount"] / (past_sum / past_n.clip(lower=1)), 1.0)
    out["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    out["is_night"] = (df["timestamp"].dt.hour < 6).astype(int)
    out["online"] = df["online"].astype(int)
    out["category_code"] = df["category"].map(CATEGORY_CODES).astype(int)
    out["new_device"] = (df.groupby(["customer_id", "device_id"], sort=False).cumcount() == 0).astype(int)
    out["new_merchant"] = (df.groupby(["customer_id", "merchant_id"], sort=False).cumcount() == 0).astype(int)
    dist = haversine_km(df["home_lat"], df["home_lon"], df["m_lat"], df["m_lon"])
    out["dist_home_km"] = np.where(df["online"], np.nan, dist)  # location is meaningless online
    out["hours_since_last"] = (g["timestamp"].diff().dt.total_seconds() / 3600).fillna(999)

    df["_one"] = 1.0
    df["_small"] = (df["amount"] < 5).astype(float)
    df["_newdev"] = out["new_device"].astype(float)
    out["n_tx_1h"] = _rolling(df, "_one", "1h", "sum")
    out["amount_1h"] = _rolling(df, "amount", "1h", "sum")
    out["n_small_1h"] = _rolling(df, "_small", "1h", "sum")
    out["n_tx_24h"] = _rolling(df, "_one", "24h", "sum")
    out["amount_24h"] = _rolling(df, "amount", "24h", "sum")
    out["n_devices_24h"] = _rolling(df, "_newdev", "24h", "sum")
    return out.loc[tx.index, FEATURES]
