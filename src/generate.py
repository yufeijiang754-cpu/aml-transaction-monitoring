"""Synthetic retail/business banking transactions with planted money-laundering typologies.

Six typologies are planted in ~3% of customers (labels are the answer key):
structuring, pass-through (rapid movement), fan-in, fan-out, high-risk jurisdiction wires,
dormant-account reactivation. Legitimate look-alikes (cash-heavy shops, payroll payers,
importers) are added on purpose so rules produce realistic false positives.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)
N_CUST, DAYS = 5000, 180
HIGH_RISK = ["IR", "KP", "MM", "SY", "YE"]
NORMAL_CTRY = ["US", "US", "US", "US", "CA", "GB", "MX", "CN", "DE"]
START = pd.Timestamp("2026-01-01")


def customers():
    seg = rng.choice(["individual", "small_business"], N_CUST, p=[0.8, 0.2])
    c = pd.DataFrame({"customer_id": np.arange(1, N_CUST + 1), "segment": seg})
    c["expected_monthly"] = np.where(seg == "individual", rng.lognormal(8.0, 0.6, N_CUST), rng.lognormal(10.0, 0.7, N_CUST)).round(0)
    c["lookalike"] = ""
    biz = c.index[c.segment == "small_business"]
    for kind, n in [("cash_heavy_shop", 120), ("payroll_payer", 120), ("importer", 60)]:
        pick = rng.choice(biz.difference(c.index[c.lookalike != ""]), n, replace=False)
        c.loc[pick, "lookalike"] = kind
    c["typology"] = ""
    pool = c.index[c.lookalike == ""]
    for t, n in [("structuring", 30), ("pass_through", 25), ("fan_in", 25), ("fan_out", 20), ("high_risk_geo", 25), ("dormant_reactivation", 25)]:
        pick = rng.choice(pool.difference(c.index[c.typology != ""]), n, replace=False)
        c.loc[pick, "typology"] = t
    c["suspicious"] = (c.typology != "").astype(int)
    return c


def tx(cid, day, ttype, amount, cp, ctry="US"):
    return (cid, START + pd.Timedelta(days=int(day), hours=int(rng.integers(8, 20))), ttype, round(float(amount), 2), cp, ctry)


def generate():
    c = customers()
    rows = []
    for r in c.itertuples():
        monthly = r.expected_monthly
        n = rng.poisson(25 if r.segment == "individual" else 60)
        dormant = r.typology == "dormant_reactivation"
        for _ in range(n):
            day = rng.integers(0, DAYS)
            if dormant and day < 150:
                continue
            ttype = rng.choice(["card", "ach_in", "ach_out", "wire_in", "wire_out", "cash_deposit", "cash_withdrawal"], p=[.40, .15, .15, .05, .05, .12, .08])
            amt = rng.lognormal(np.log(monthly / 15 + 20), 0.9)
            rows.append(tx(r.customer_id, day, ttype, amt, f"CP{rng.integers(1, 3000)}", rng.choice(NORMAL_CTRY)))
        # legitimate look-alikes
        if r.lookalike == "cash_heavy_shop":
            for d in rng.choice(DAYS, 20, replace=False):
                rows.append(tx(r.customer_id, d, "cash_deposit", rng.uniform(6000, 9800), "BRANCH"))
        elif r.lookalike == "payroll_payer":
            for d in range(14, DAYS, 14):
                for k in range(rng.integers(10, 25)):
                    rows.append(tx(r.customer_id, d, "ach_out", rng.uniform(800, 3000), f"EMP{r.customer_id}_{k}"))
        elif r.lookalike == "importer":
            for d in rng.choice(DAYS, 6, replace=False):
                rows.append(tx(r.customer_id, d, "wire_in", rng.uniform(20000, 80000), f"BUYER{rng.integers(1, 50)}"))
                rows.append(tx(r.customer_id, d + rng.integers(1, 3), "wire_out", rng.uniform(15000, 60000), f"SUPPLIER{rng.integers(1, 50)}", rng.choice(["CN", "VN", "MX"])))
        # planted typologies
        t = r.typology
        if t == "structuring":
            for start in rng.choice(DAYS - 7, 3, replace=False):
                for k in range(rng.integers(3, 6)):
                    rows.append(tx(r.customer_id, start + rng.integers(0, 5), "cash_deposit", rng.uniform(8000, 9900), "BRANCH"))
        elif t == "pass_through":
            for d in rng.choice(DAYS - 3, 4, replace=False):
                a = rng.uniform(20000, 120000)
                rows.append(tx(r.customer_id, d, "wire_in", a, f"SHELL{rng.integers(1, 200)}"))
                rows.append(tx(r.customer_id, d + rng.integers(0, 2), "wire_out", a * rng.uniform(0.9, 0.98), f"SHELL{rng.integers(200, 400)}", rng.choice(NORMAL_CTRY)))
        elif t == "fan_in":
            for start in rng.choice(DAYS - 7, 2, replace=False):
                for k in range(rng.integers(12, 25)):
                    rows.append(tx(r.customer_id, start + rng.integers(0, 6), "ach_in", rng.uniform(300, 2500), f"MULE{r.customer_id}_{start}_{k}"))
                rows.append(tx(r.customer_id, start + 6, "wire_out", rng.uniform(10000, 30000), f"COLLECT{rng.integers(1, 50)}"))
        elif t == "fan_out":
            rows.append(tx(r.customer_id, 60, "wire_in", rng.uniform(30000, 90000), f"SRC{rng.integers(1, 50)}"))
            for k in range(rng.integers(12, 30)):
                rows.append(tx(r.customer_id, 61 + rng.integers(0, 5), "ach_out", rng.uniform(1000, 4000), f"NEW{r.customer_id}_{k}"))
        elif t == "high_risk_geo":
            for d in rng.choice(DAYS, 3, replace=False):
                rows.append(tx(r.customer_id, d, rng.choice(["wire_in", "wire_out"]), rng.uniform(10000, 60000), f"FX{rng.integers(1, 100)}", rng.choice(HIGH_RISK)))
        elif t == "dormant_reactivation":
            for d in rng.choice(np.arange(150, DAYS), 6, replace=False):
                rows.append(tx(r.customer_id, d, rng.choice(["wire_in", "wire_out", "cash_deposit"]), rng.uniform(15000, 70000), f"CP{rng.integers(1, 3000)}"))
    t = pd.DataFrame(rows, columns=["customer_id", "ts", "type", "amount", "counterparty", "country"])
    t = t[t.ts < START + pd.Timedelta(days=DAYS)].sort_values("ts").reset_index(drop=True)
    t.insert(0, "tx_id", np.arange(1, len(t) + 1))
    (ROOT / "data").mkdir(exist_ok=True)
    c.to_csv(ROOT / "data" / "customers.csv", index=False)
    t.to_csv(ROOT / "data" / "transactions.csv", index=False)
    print(f"customers={len(c)} suspicious={c.suspicious.sum()} lookalikes={(c.lookalike!='').sum()} transactions={len(t)}")


if __name__ == "__main__":
    generate()
