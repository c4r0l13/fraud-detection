"""Synthetic card-transaction generator with realistic fraud patterns.

Public fraud datasets are either anonymized PCA components (no feature
engineering possible) or require an account to download. A simulator gives
full control: every fraud carries the *scenario* that produced it, so we can
measure which attack patterns the model actually catches.

Legitimate behaviour
    Each customer has a home location, a spending level, preferred merchant
    categories, a usual device and active hours. Transactions follow a daily
    Poisson process.

Fraud scenarios
    card_testing        Stolen card number: a burst of tiny online charges
                        within minutes, then one large purchase.
    account_takeover    Credentials stolen: a new device far from home buys
                        electronics / gift cards at night.
    point_of_compromise A skimmed merchant leaks every card that used it;
                        days later the cards are used for mid-sized purchases
                        that look almost normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES = {
    # name: (share of merchants, mean amount multiplier, online share)
    "grocery": (0.22, 0.6, 0.05),
    "restaurants": (0.18, 0.5, 0.10),
    "fuel": (0.10, 0.7, 0.0),
    "retail": (0.15, 1.2, 0.35),
    "pharmacy": (0.06, 0.5, 0.10),
    "travel": (0.04, 4.0, 0.70),
    "entertainment": (0.08, 0.8, 0.60),
    "electronics": (0.06, 3.0, 0.50),
    "gift_cards": (0.03, 1.5, 0.80),
    "utilities": (0.08, 1.4, 0.90),
}


@dataclass
class SimConfig:
    n_customers: int = 3000
    n_merchants: int = 800
    days: int = 120
    start: str = "2026-01-01"
    tx_per_day: float = 1.6
    card_testing_rate: float = 0.05  # share of customers hit over the period
    ato_rate: float = 0.04
    poc_merchants: int = 10
    poc_conversion: float = 0.45  # share of exposed cards that are later abused
    seed: int = 7


def _customers(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_customers
    cats = list(CATEGORIES)
    weights = np.array([v[0] for v in CATEGORIES.values()])
    prefs = rng.dirichlet(weights * 20, size=n)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "home_lat": rng.normal(19.4, 0.6, n),  # a metro-sized region
            "home_lon": rng.normal(-99.1, 0.6, n),
            "spend_level": rng.lognormal(np.log(35), 0.6, n),
            "active_hour": rng.normal(15, 3, n).clip(8, 22),
            "device_id": [f"dev_{i}" for i in range(n)],
            "online_affinity": rng.beta(2, 5, n),
            "prefs": list(prefs),
            "cats": [cats] * n,
        }
    )


def _merchants(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    cats = list(CATEGORIES)
    shares = np.array([v[0] for v in CATEGORIES.values()])
    cat = rng.choice(cats, size=cfg.n_merchants, p=shares / shares.sum())
    cat[: len(cats)] = cats  # every category has at least one merchant
    return pd.DataFrame(
        {
            "merchant_id": np.arange(cfg.n_merchants),
            "category": cat,
            "m_lat": rng.normal(19.4, 0.7, cfg.n_merchants),
            "m_lon": rng.normal(-99.1, 0.7, cfg.n_merchants),
        }
    )


def _legit(cfg: SimConfig, cust: pd.DataFrame, merch: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    by_cat = {c: merch.loc[merch.category == c, "merchant_id"].to_numpy() for c in CATEGORIES}
    cats = list(CATEGORIES)
    rows = []
    for c in cust.itertuples(index=False):
        n = rng.poisson(cfg.tx_per_day * cfg.days)
        day = rng.integers(0, cfg.days, n)
        hour = np.clip(rng.normal(c.active_hour, 3, n), 0, 23.99)
        cat_idx = rng.choice(len(cats), size=n, p=c.prefs)
        # each customer keeps returning to a small set of favourite merchants
        favourites = {k: rng.choice(v, size=min(3, len(v)), replace=False) for k, v in by_cat.items()}
        for d, h, ci in zip(day, hour, cat_idx, strict=True):
            cat = cats[ci]
            mult, online_share = CATEGORIES[cat][1], CATEGORIES[cat][2]
            m = rng.choice(favourites[cat]) if rng.uniform() < 0.8 else rng.choice(by_cat[cat])
            rows.append(
                (
                    c.customer_id,
                    int(m),
                    d * 24 + h,
                    float(np.round(rng.lognormal(np.log(c.spend_level * mult), 0.5), 2)),
                    bool(rng.uniform() < online_share * (0.5 + c.online_affinity)),
                    c.device_id,
                    0,
                    "legit",
                )
            )
    legit = pd.DataFrame(
        rows, columns=["customer_id", "merchant_id", "t_hours", "amount", "online", "device_id", "is_fraud", "scenario"]
    )
    return _legit_noise(cfg, legit, rng)


def _legit_noise(cfg: SimConfig, legit: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Honest behaviour that looks suspicious, so no single signal is a giveaway."""
    legit = legit.copy()
    # 30% of customers get a new phone at some point and keep using it
    cid = legit.customer_id.to_numpy()
    switch_day = rng.uniform(0, cfg.days, cid.max() + 1) * 24
    switches = rng.uniform(size=len(switch_day)) < 0.30
    new_phone = switches[cid] & (legit.t_hours.to_numpy() > switch_day[cid])
    legit.loc[new_phone, "device_id"] = legit.loc[new_phone, "device_id"] + "_v2"
    # 4% of online purchases come from a one-off browser or a relative's device
    oneoff = legit.online & (rng.uniform(size=len(legit)) < 0.04)
    legit.loc[oneoff, "device_id"] = [f"browser_{i}" for i in rng.integers(0, 10**9, oneoff.sum())]
    # 0.5% of purchases are unusually large (furniture, flights, a new laptop)
    big = rng.uniform(size=len(legit)) < 0.005
    legit.loc[big, "amount"] = np.round(legit.loc[big, "amount"] * rng.uniform(5, 20, big.sum()), 2)
    # 1.5% are micro-charges: app stores, parking, card verification by real merchants
    micro = rng.uniform(size=len(legit)) < 0.015
    legit.loc[micro, "amount"] = np.round(rng.uniform(0.5, 9, micro.sum()), 2)
    legit.loc[micro, "online"] = True
    # 2% happen late at night (shift workers, night owls)
    night = rng.uniform(size=len(legit)) < 0.02
    legit.loc[night, "t_hours"] = np.floor(legit.loc[night, "t_hours"] / 24) * 24 + rng.uniform(0, 5, night.sum())
    # online shopping sessions: 4% of online purchases trigger 1-3 more within the hour
    seed_rows = legit[legit.online & (rng.uniform(size=len(legit)) < 0.04)]
    extra = seed_rows.loc[seed_rows.index.repeat(rng.integers(1, 4, len(seed_rows)))].copy()
    extra["t_hours"] = extra["t_hours"] + rng.uniform(2, 50, len(extra)) / 60
    extra["amount"] = np.round(extra["amount"] * rng.uniform(0.3, 1.5, len(extra)), 2)
    return pd.concat([legit, extra], ignore_index=True)


def _card_testing(cfg, cust, merch, rng) -> list[tuple]:
    online_m = merch.loc[merch.category.isin(["retail", "entertainment", "gift_cards"]), "merchant_id"].to_numpy()
    victims = rng.choice(cust.customer_id, int(cfg.card_testing_rate * len(cust)), replace=False)
    rows = []
    for v in victims:
        t0 = rng.uniform(5, cfg.days - 1) * 24
        dev = f"fraud_dev_{rng.integers(1e9)}"
        t = t0
        for _ in range(rng.integers(1, 5)):  # careful testers probe only once or twice
            rows.append(
                (v, int(rng.choice(online_m)), t, float(np.round(rng.uniform(0.5, 9), 2)), True, dev, 1, "card_testing")
            )
            t += rng.uniform(2, 90) / 60
        if rng.uniform() < 0.8:  # cash-out purchase
            rows.append(
                (
                    v,
                    int(rng.choice(online_m)),
                    t + rng.uniform(0.2, 12),
                    float(np.round(rng.uniform(60, 900), 2)),
                    True,
                    dev,
                    1,
                    "card_testing",
                )
            )
    return rows


def _account_takeover(cfg, cust, merch, rng) -> list[tuple]:
    target = merch.loc[merch.category.isin(["electronics", "gift_cards", "travel", "retail"]), "merchant_id"]
    target = target.to_numpy()
    victims = rng.choice(cust.customer_id, int(cfg.ato_rate * len(cust)), replace=False)
    rows = []
    for v in victims:
        hour = rng.uniform(0, 5) if rng.uniform() < 0.4 else rng.uniform(8, 23)
        t0 = rng.integers(5, cfg.days - 1) * 24 + hour
        # 25% of takeovers run through the victim's own device (remote-access malware)
        dev = cust.loc[v, "device_id"] if rng.uniform() < 0.25 else f"fraud_dev_{rng.integers(1e9)}"
        spend = cust.loc[v, "spend_level"]
        for k in range(rng.integers(1, 4)):
            rows.append(
                (
                    v,
                    int(rng.choice(target)),
                    t0 + k * rng.uniform(0.3, 6),
                    float(np.round(spend * rng.uniform(2, 12), 2)),
                    True,
                    dev,
                    1,
                    "account_takeover",
                )
            )
    return rows


def _point_of_compromise(cfg, legit, merch, rng) -> tuple[list[tuple], np.ndarray]:
    poc = rng.choice(
        merch.loc[merch.category.isin(["fuel", "restaurants", "retail"]), "merchant_id"],
        cfg.poc_merchants,
        replace=False,
    )
    rows = []
    any_m = merch.merchant_id.to_numpy()
    # breaches are spread over the whole period so every split contains some
    starts = np.linspace(5, cfg.days - 22, len(poc)) + rng.uniform(-2, 2, len(poc))
    for m, start_day in zip(poc, starts, strict=True):
        window_start = start_day * 24
        exposed = legit[(legit.merchant_id == m) & legit.t_hours.between(window_start, window_start + 10 * 24)]
        for cust_id, t_exp in exposed.groupby("customer_id")["t_hours"].min().items():
            if rng.uniform() > cfg.poc_conversion:
                continue
            t0 = t_exp + rng.uniform(2, 12) * 24
            if t0 > cfg.days * 24:
                continue
            for k in range(rng.integers(1, 4)):
                rows.append(
                    (
                        cust_id,
                        int(rng.choice(any_m)),
                        t0 + k * rng.uniform(1, 6),
                        float(np.round(rng.uniform(80, 400), 2)),
                        bool(rng.uniform() < 0.5),
                        f"clone_{cust_id}",
                        1,
                        "point_of_compromise",
                    )
                )
    return rows, poc


def simulate(cfg: SimConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (transactions, customers, merchants). Transactions are time-ordered."""
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    cust = _customers(cfg, rng)
    merch = _merchants(cfg, rng)
    legit = _legit(cfg, cust, merch, rng)
    poc_rows, _ = _point_of_compromise(cfg, legit, merch, rng)
    fraud = pd.DataFrame(
        _card_testing(cfg, cust, merch, rng) + _account_takeover(cfg, cust, merch, rng) + poc_rows,
        columns=legit.columns,
    )
    tx = pd.concat([legit, fraud], ignore_index=True)
    tx = tx[tx.t_hours < cfg.days * 24].sort_values("t_hours", kind="stable").reset_index(drop=True)
    tx["timestamp"] = pd.Timestamp(cfg.start) + pd.to_timedelta(tx["t_hours"], unit="h")
    tx = tx.merge(merch, on="merchant_id", how="left")
    tx["transaction_id"] = np.arange(len(tx))
    cols = [
        "transaction_id",
        "timestamp",
        "customer_id",
        "merchant_id",
        "category",
        "amount",
        "online",
        "device_id",
        "m_lat",
        "m_lon",
        "is_fraud",
        "scenario",
    ]
    return tx[cols], cust.drop(columns=["prefs", "cats"]), merch
