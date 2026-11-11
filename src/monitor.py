"""Rule-based transaction monitoring in SQL (SQLite) + model-based alert prioritization.

Step 1: six SQL scenarios generate customer-level alerts.
Step 2: evaluate rules against the planted answer key (precision, recall, per typology).
Step 3: score alerted customers with a gradient-boosting risk model (5-fold CV, out-of-fold
        scores) and measure how much review volume can be cut while keeping recall.
Step 4: tune the structuring threshold (alert volume vs. detection trade-off).
"""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

RULES = {
    "R1_structuring": """
        SELECT DISTINCT a.customer_id FROM tx a
        WHERE a.type = 'cash_deposit' AND a.amount BETWEEN 8000 AND 9999.99
          AND (SELECT COUNT(*) FROM tx b WHERE b.customer_id = a.customer_id AND b.type = 'cash_deposit'
               AND b.amount BETWEEN 8000 AND 9999.99 AND b.day BETWEEN a.day AND a.day + 6) >= {min_count}""",
    "R2_pass_through": """
        SELECT DISTINCT i.customer_id FROM tx i JOIN tx o
          ON o.customer_id = i.customer_id AND o.type = 'wire_out' AND o.day BETWEEN i.day AND i.day + 2
        WHERE i.type = 'wire_in' AND i.amount >= 20000 AND o.amount >= 0.8 * i.amount""",
    "R3_fan_in": """
        SELECT DISTINCT a.customer_id FROM tx a
        WHERE a.type IN ('ach_in', 'wire_in')
          AND (SELECT COUNT(DISTINCT b.counterparty) FROM tx b WHERE b.customer_id = a.customer_id
               AND b.type IN ('ach_in', 'wire_in') AND b.day BETWEEN a.day AND a.day + 6) >= 10""",
    "R4_fan_out": """
        SELECT DISTINCT a.customer_id FROM tx a
        WHERE a.type IN ('ach_out', 'wire_out')
          AND (SELECT COUNT(DISTINCT b.counterparty) FROM tx b WHERE b.customer_id = a.customer_id
               AND b.type IN ('ach_out', 'wire_out') AND b.day BETWEEN a.day AND a.day + 6) >= 10""",
    "R5_high_risk_geo": """
        SELECT DISTINCT customer_id FROM tx
        WHERE type IN ('wire_in', 'wire_out') AND amount >= 10000 AND country IN ('IR', 'KP', 'MM', 'SY', 'YE')""",
    "R6_dormant_reactivation": """
        WITH first_big AS (SELECT customer_id, MIN(day) AS d FROM tx WHERE amount >= 15000 GROUP BY customer_id)
        SELECT f.customer_id FROM first_big f
        WHERE (SELECT COUNT(*) FROM tx t WHERE t.customer_id = f.customer_id AND t.day BETWEEN f.d - 90 AND f.d - 1) = 0""",
}


def load(con):
    t = pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["ts"])
    t["day"] = (t["ts"] - t["ts"].min().normalize()).dt.days
    t.drop(columns=["ts"]).to_sql("tx", con, index=False)
    con.execute("CREATE INDEX ix ON tx(customer_id, type, day)")
    return t, pd.read_csv(ROOT / "data" / "customers.csv").fillna({"typology": "", "lookalike": ""})


def run_rules(con, min_count=3):
    hits = {}
    for name, sql in RULES.items():
        ids = pd.read_sql(sql.format(min_count=min_count), con)["customer_id"]
        hits[name] = set(ids)
    return hits


def features(t, c, hits):
    g = t.groupby("customer_id")
    f = pd.DataFrame({
        "n_tx": g.size(),
        "total_in": t[t.type.str.endswith("_in") | (t.type == "cash_deposit")].groupby("customer_id")["amount"].sum(),
        "total_out": t[t.type.str.endswith("_out") | (t.type == "cash_withdrawal")].groupby("customer_id")["amount"].sum(),
        "cash_dep_8_10k": t[(t.type == "cash_deposit") & t.amount.between(8000, 9999.99)].groupby("customer_id").size(),
        "n_counterparties": g["counterparty"].nunique(),
        "max_amount": g["amount"].max(),
        "active_days": g["day"].nunique(),
        "high_risk_ctry_amt": t[t.country.isin(["IR", "KP", "MM", "SY", "YE"])].groupby("customer_id")["amount"].sum(),
    }).reindex(c.customer_id).fillna(0)
    f["in_out_ratio"] = f.total_out / f.total_in.clip(lower=1)
    exp = c.set_index("customer_id")["expected_monthly"] * 6
    f["volume_vs_expected"] = (f.total_in + f.total_out) / exp.clip(lower=1)
    f["is_business"] = (c.set_index("customer_id")["segment"] == "small_business").astype(int)
    for r, ids in hits.items():
        f[r] = f.index.isin(ids).astype(int)
    f["n_rules_hit"] = f[list(hits)].sum(axis=1)
    return f


def main():
    con = sqlite3.connect(":memory:")
    t, c = load(con)
    y = c.set_index("customer_id")["suspicious"]
    hits = run_rules(con)
    alerted = set().union(*hits.values())

    per_rule = {r: {"alerts": len(ids), "true_positives": int(y.loc[list(ids)].sum()) if ids else 0} for r, ids in hits.items()}
    for v in per_rule.values():
        v["precision"] = round(v["true_positives"] / v["alerts"], 3) if v["alerts"] else None
    tp = int(y.loc[list(alerted)].sum())
    by_typ = {k: round(float(c[c.typology == k].customer_id.isin(alerted).mean()), 3) for k in c.typology.unique() if k}

    # Alert prioritization model, trained only on the alerted population (what investigators actually review).
    f = features(t, c, hits)
    fa, ya = f.loc[sorted(alerted)], y.loc[sorted(alerted)]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    model = GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3)
    score = pd.Series(cross_val_predict(model, fa, ya, cv=cv, method="predict_proba")[:, 1], index=fa.index)
    order = score.sort_values(ascending=False)
    cum_tp = ya.loc[order.index].cumsum().values
    k95 = int(np.argmax(cum_tp >= 0.95 * ya.sum()) + 1)
    k100 = int(np.argmax(cum_tp >= ya.sum()) + 1)

    # Threshold tuning for structuring (number of 8k-10k deposits in 7 days).
    tune = []
    for m in range(2, 7):
        ids = run_rules(con, min_count=m)["R1_structuring"]
        struct = set(c[c.typology == "structuring"].customer_id)
        tune.append({"min_deposits": m, "alerts": len(ids), "structuring_recall": round(len(ids & struct) / len(struct), 3),
                     "false_positives": len(ids - set(y[y == 1].index))})
    pd.DataFrame(tune).to_csv(OUT / "structuring_threshold_tuning.csv", index=False)

    metrics = {
        "customers": len(c), "transactions": len(t), "suspicious_customers": int(y.sum()),
        "rules": per_rule,
        "rules_combined": {"alerted_customers": len(alerted), "true_positives": tp,
                           "precision": round(tp / len(alerted), 3), "recall": round(tp / int(y.sum()), 3)},
        "recall_by_typology": by_typ,
        "prioritization": {
            "alerts_to_review_for_95pct_recall": k95, "alerts_to_review_for_100pct_recall": k100,
            "review_volume_cut_at_95pct_recall": round(1 - k95 / len(alerted), 3),
            "precision_top_decile": round(float(ya.loc[order.index[: max(1, len(order) // 10)]].mean()), 3),
        },
        "structuring_threshold_tuning": tune,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    order.rename("risk_score").to_frame().join(f[list(hits)]).join(c.set_index("customer_id")[["segment", "typology", "lookalike"]]) \
        .to_csv(OUT / "alert_queue_ranked.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.arange(1, len(order) + 1) / len(order), cum_tp / ya.sum(), label="Model-ranked queue")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="Unranked (FIFO)")
    ax[0].axhline(0.95, ls=":", color="red")
    ax[0].set(title="Alert review: share reviewed vs. suspicious found", xlabel="Share of alerts reviewed", ylabel="Share of true cases found")
    ax[0].legend()
    td = pd.DataFrame(tune)
    ax[1].plot(td.min_deposits, td.alerts, marker="o", label="Alerts")
    ax[1].plot(td.min_deposits, td.false_positives, marker="o", label="False positives")
    ax2 = ax[1].twinx()
    ax2.plot(td.min_deposits, td.structuring_recall, marker="s", color="green", label="Recall")
    ax2.set_ylim(0, 1.05)
    ax[1].set(title="Structuring rule: threshold tuning", xlabel="Min. 8k-10k cash deposits in 7 days", ylabel="Count")
    ax[1].legend(loc="center left")
    ax2.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "monitoring.png", dpi=150)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
