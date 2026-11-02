"""Simulate -> features -> time-based split -> models -> cost-optimal threshold -> report.

Run with ``python -m fraud.pipeline`` (add ``--mlflow`` to log the runs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import style
from .costs import CostModel, alert_metrics, best_threshold, cost_curve
from .features import FEATURES, build_features
from .simulator import SimConfig, simulate

TRAIN_DAYS, VALID_DAYS = 80, 20  # remaining days = test
LABELS = {
    "amount": "Amount",
    "log_amount": "Amount (log)",
    "amount_vs_card_mean": "Amount vs card's usual",
    "hour": "Hour of day",
    "is_night": "Night (00-06h)",
    "online": "Online",
    "category_code": "Merchant category",
    "new_device": "New device for card",
    "new_merchant": "New merchant for card",
    "dist_home_km": "Distance from home",
    "hours_since_last": "Hours since last purchase",
    "n_tx_1h": "Purchases, last hour",
    "amount_1h": "Amount, last hour",
    "n_small_1h": "Tiny (<$5) charges, last hour",
    "n_tx_24h": "Purchases, last 24h",
    "amount_24h": "Amount, last 24h",
    "n_devices_24h": "New devices, last 24h",
}


def rules_engine(X: pd.DataFrame) -> np.ndarray:
    """A typical legacy rule set, used as the baseline to beat."""
    return (
        (X["amount"] > 800)
        | ((X["new_device"] == 1) & (X["amount"] > 300))
        | (X["n_tx_1h"] >= 4)
        | ((X["is_night"] == 1) & (X["amount"] > 400))
    ).to_numpy()


def make_models(pos_weight: float, seed: int) -> dict:
    return {
        "Logistic regression": make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced"),
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=200,
            reg_lambda=5.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--review-cost", type=float, default=5.0)
    ap.add_argument("--mlflow", action="store_true", help="log params/metrics to ./mlflow.db")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out, figs = Path(args.out), Path(args.out) / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    style.apply_style()
    cost = CostModel(review_cost=args.review_cost)

    print("simulating ...")
    cfg = SimConfig(seed=args.seed)
    tx, customers, _ = simulate(cfg)
    print("features ...")
    X = build_features(tx, customers)
    day = (tx["timestamp"] - tx["timestamp"].min()).dt.days
    tr, va, te = day < TRAIN_DAYS, (day >= TRAIN_DAYS) & (day < TRAIN_DAYS + VALID_DAYS), day >= TRAIN_DAYS + VALID_DAYS
    y, amt = tx["is_fraud"].to_numpy(), tx["amount"].to_numpy()
    test_days = cfg.days - TRAIN_DAYS - VALID_DAYS
    data_summary = {
        "transactions": len(tx),
        "customers": cfg.n_customers,
        "days": cfg.days,
        "fraud_rate": float(y.mean()),
        "frauds_by_scenario": tx.loc[y == 1, "scenario"].value_counts().to_dict(),
        "split_sizes": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
        "test_frauds": int(y[te].sum()),
    }

    results, scores_te, thresholds = {}, {}, {}
    pos_weight = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    for name, model in make_models(pos_weight, args.seed).items():
        print(f"training {name} ...")
        if isinstance(model, LGBMClassifier):
            # early stopping on the validation period (PR-AUC), never on test
            model.fit(
                X[tr],
                y[tr],
                eval_set=[(X[va], y[va])],
                eval_metric="average_precision",
                callbacks=[early_stopping(100, verbose=False)],
            )
        else:
            model.fit(X[tr], y[tr])
        s_va, s_te = model.predict_proba(X[va])[:, 1], model.predict_proba(X[te])[:, 1]
        thr = best_threshold(y[va], amt[va], s_va, cost)  # chosen on validation only
        alert = s_te >= thr
        results[name] = {
            "pr_auc": float(average_precision_score(y[te], s_te)),
            "roc_auc": float(roc_auc_score(y[te], s_te)),
            "threshold": thr,
            "cost": cost.total_cost(y[te], amt[te], alert),
            **alert_metrics(y[te], amt[te], alert, test_days),
        }
        scores_te[name], thresholds[name] = s_te, thr
        if name == "LightGBM":
            lgbm = model

    rule_alert = rules_engine(X[te])
    results["Rules engine (baseline)"] = {
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "threshold": float("nan"),
        "cost": cost.total_cost(y[te], amt[te], rule_alert),
        **alert_metrics(y[te], amt[te], rule_alert, test_days),
    }
    no_model = cost.baseline_cost(y[te], amt[te])
    for r in results.values():
        r["savings_vs_no_screening"] = no_model - r["cost"]
    table = pd.DataFrame(results).T
    order = ["Rules engine (baseline)", "Logistic regression", "LightGBM"]
    table = table.loc[order]

    # recall by scenario for the champion and the rules
    champ_alert = scores_te["LightGBM"] >= thresholds["LightGBM"]
    scen = tx.loc[te, "scenario"].to_numpy()
    by_scenario = {}
    for s in ["card_testing", "account_takeover", "point_of_compromise"]:
        m = scen == s
        by_scenario[s] = {
            "frauds": int(m.sum()),
            "LightGBM": float(champ_alert[m].mean()),
            "Rules engine": float(rule_alert[m].mean()),
            "LightGBM dollar recall": float(amt[te][m & champ_alert].sum() / amt[te][m].sum()),
        }

    contrib = lgbm.booster_.predict(X[te], pred_contrib=True)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    # example explanations for the top alerts
    top = np.argsort(scores_te["LightGBM"])[::-1][:3]
    examples = []
    for i in top:
        c = pd.Series(contrib[i], index=FEATURES).sort_values(ascending=False).head(3)
        row = tx.loc[te].iloc[i]
        examples.append(
            {
                "amount": float(row.amount),
                "category": row.category,
                "scenario": row.scenario,
                "score": float(scores_te["LightGBM"][i]),
                "top_reasons": [LABELS[f] for f in c.index],
            }
        )

    # figures
    _fig_pr(y[te], scores_te, rule_alert, figs)
    _fig_cost(y[va], amt[va], lgbm.predict_proba(X[va])[:, 1], thresholds["LightGBM"], cost, figs)
    _fig_scenarios(by_scenario, figs)
    _fig_importance(importance, figs)

    metrics = {
        "data": data_summary,
        "cost_model": {"review_cost": cost.review_cost},
        "cost_no_screening_test": no_model,
        "models": table.to_dict(orient="index"),
        "recall_by_scenario": by_scenario,
        "feature_importance": importance.round(4).to_dict(),
        "example_alerts": examples,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    show = table[
        [
            "pr_auc",
            "roc_auc",
            "alerts_per_day",
            "precision",
            "recall",
            "dollar_recall",
            "cost",
            "savings_vs_no_screening",
        ]
    ]
    md = [
        f"# Results (test period: last {test_days} days)",
        "",
        show.round(3).to_markdown(),
        "",
        f"Cost with no screening: ${no_model:,.0f}",
        "",
        "## Recall by scenario",
        "",
        pd.DataFrame(by_scenario).T.round(3).to_markdown(),
        "",
        "## Example alerts",
        "",
    ]
    md += [
        f"- ${e['amount']:,.2f} {e['category']} ({e['scenario']}), score {e['score']:.2f}: "
        + "; ".join(e["top_reasons"])
        for e in examples
    ]
    (out / "results.md").write_text("\n".join(md) + "\n")

    if args.mlflow:
        _log_mlflow(results, data_summary, cost)

    print(json.dumps(data_summary, indent=1, default=int))
    print(show.round(3).to_string())
    print("no screening", round(no_model))
    print(pd.DataFrame(by_scenario).T.round(3).to_string())
    print(importance.round(3).head(8).to_string())
    print(examples)


def _log_mlflow(results: dict, data_summary: dict, cost: CostModel) -> None:
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")
    mlflow.set_experiment("fraud-detection")
    for name, r in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "review_cost": cost.review_cost, **data_summary["split_sizes"]})
            mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, float) and np.isfinite(v)})


# ---------------------------------------------------------------------- figures


def _fig_pr(y, scores: dict, rule_alert, figs: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for (name, s), color in zip(scores.items(), [style.SERIES[1], style.SERIES[0]], strict=True):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r * 100, p * 100, color=color, label=f"{name} (PR-AUC {average_precision_score(y, s):.2f})")
    tp = (rule_alert & (y == 1)).sum()
    ax.scatter(
        [tp / (y == 1).sum() * 100],
        [tp / rule_alert.sum() * 100],
        s=70,
        marker="D",
        color=style.SERIES[2],
        edgecolor=style.SURFACE,
        linewidth=1.5,
        zorder=5,
        label="Rules engine",
    )
    ax.set_xlabel("Recall: share of frauds caught (%)")
    ax.set_ylabel("Precision: share of alerts that are fraud (%)")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 101)
    ax.set_title("Precision-recall on the test period")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.03))
    fig.savefig(figs / "precision_recall.png")
    plt.close(fig)


def _fig_cost(y, amt, s, thr, cost: CostModel, figs: Path) -> None:
    curve = cost_curve(y, amt, s, cost)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(curve["alerts"], curve["cost"] / 1000, color=style.SERIES[0])
    best = curve.loc[curve["cost"].idxmin()]
    ax.scatter(
        [best["alerts"]],
        [best["cost"] / 1000],
        s=60,
        color=style.SERIES[0],
        edgecolor=style.SURFACE,
        linewidth=2,
        zorder=5,
    )
    ax.annotate(
        f"optimum: {int(best['alerts']):,} alerts",
        (best["alerts"], best["cost"] / 1000),
        xytext=(50, 70),
        textcoords="offset points",
        fontsize=9,
        color=style.INK,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": style.MUTED, "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Alerts in the validation period (log scale)")
    ax.set_ylabel("Total cost ($ thousands)")
    ax.set_title(f"Too few alerts lose fraud, too many cost ${cost.review_cost:.0f} each")
    fig.savefig(figs / "cost_curve.png")
    plt.close(fig)


def _fig_scenarios(by_scenario: dict, figs: Path) -> None:
    names = {
        "card_testing": "Card testing",
        "account_takeover": "Account takeover",
        "point_of_compromise": "Point of compromise",
    }
    keys = list(by_scenario)
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (col, color) in enumerate((("Rules engine", style.SERIES[2]), ("LightGBM", style.SERIES[0]))):
        vals = [by_scenario[k][col] * 100 for k in keys]
        ax.bar(x + (i - 0.5) * w, vals, w - 0.04, color=color, label=col)
        for xi, v in zip(x, vals, strict=True):
            ax.text(xi + (i - 0.5) * w, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=style.INK_2)
    ax.set_xticks(x, [f"{names[k]}\n(n={by_scenario[k]['frauds']})" for k in keys])
    ax.set_ylabel("Frauds caught (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Which attack patterns are caught")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(figs / "recall_by_scenario.png")
    plt.close(fig)


def _fig_importance(importance: pd.Series, figs: Path) -> None:
    imp = importance.sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh([LABELS[i] for i in imp.index], imp.values, color=style.SERIES[0], height=0.65)
    ax.set_xlabel("Mean |contribution| (log-odds, TreeSHAP)")
    ax.set_title("What the model looks at")
    ax.grid(axis="y", visible=False)
    fig.savefig(figs / "feature_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
