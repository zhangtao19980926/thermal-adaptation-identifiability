# -*- coding: utf-8 -*-
"""
Two-window incremental test on REAL slow bands. Features {x_now, mem_3d(short), mem_30d(long)}.
Does the LONG window add independent predictive information once the short window is known?
Strong slow band (In-Gauge): long window should add a sizeable, stable gain and carry a
significant (negative, adaptive) coefficient. Weak slow band (DORN): long-window gain ~ 0
(coefficient collapses) -> long memory is neither needed nor identifiable.
Averages over 3 DGP seeds.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(ROOT, "src", "p1mk"))
from ingauge_inject import ingauge_daily, dorn_daily, simulate_external

def mem_at(votes, meta, w):
    t, Tamb, Tn = meta["t"], meta["T_amb"], meta["Tn"]
    tv, subj = votes.t_day.values, votes.subject.values
    hi = np.clip(np.searchsorted(t, tv), 1, len(t) - 1)
    lo = np.clip(np.searchsorted(t, tv - w), 0, hi - 1)
    csum = np.concatenate([[0.0], np.cumsum(np.ascontiguousarray(Tamb))])
    run = (csum[hi] - csum[lo]) / (hi - lo)
    return run - Tn[subj]

def evaluate(votes, meta):
    xnow = (meta["T_amb"][np.clip(np.searchsorted(meta["t"], votes.t_day.values), 1, len(meta["t"]) - 1)]
            - meta["Tn"][votes.subject.values])
    Xs = {"x_now": xnow, "mem3": mem_at(votes, meta, 3), "mem30": mem_at(votes, meta, 30)}
    y = votes.TSV.values; g = votes.subject.values
    configs = {"M0_inst": ["x_now"], "M1_short": ["x_now", "mem3"],
               "M2_long": ["x_now", "mem30"], "M3_dual": ["x_now", "mem3", "mem30"]}
    out = {}
    for name, cols in configs.items():
        X = np.column_stack([Xs[c] for c in cols])
        ll, ms, n = [], [], 0
        for tr, te in GroupKFold(5).split(X, y, groups=g):
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            clf = LogisticRegression(max_iter=2000, C=10.0).fit(Xtr, y[tr])
            ll.append(log_loss(y[te], clf.predict_proba(Xte), labels=list(range(-3, 4)), normalize=False))
            rg = LinearRegression().fit(Xtr, y[tr])
            ms.append(mean_squared_error(y[te], rg.predict(Xte)) * len(te)); n += len(te)
        out[name + "_ll"] = sum(ll) / n; out[name + "_mse"] = sum(ms) / n
    # standardised long-window coefficient in the dual model (full data)
    Xd = np.column_stack([Xs["x_now"], Xs["mem3"], Xs["mem30"]])
    sc = StandardScaler().fit(Xd)
    rg = LinearRegression().fit(sc.transform(Xd), y)
    out["coef_mem30_std"] = rg.coef_[2]; out["coef_mem3_std"] = rg.coef_[1]
    out["dLong_ll"] = out["M1_short_ll"] - out["M3_dual_ll"]    # >0: long window helps
    out["dLong_mse"] = out["M1_short_mse"] - out["M3_dual_mse"]
    return out

def main():
    bands = {"InGauge_strong": pd.Series(ingauge_daily()).interpolate().dropna().values,
             "DORN_weak": pd.Series(dorn_daily()).interpolate().dropna().values}
    allr = []
    for nm, daily in bands.items():
        for seed in (1, 2, 3):
            votes, meta = simulate_external(daily, 100, 120, seed=seed)
            r = evaluate(votes, meta); r["band"] = nm; r["seed"] = seed; allr.append(r)
    df = pd.DataFrame(allr)
    agg = df.groupby("band").agg({c: "mean" for c in df.columns if c not in ("band", "seed")})
    cols = ["M0_inst_ll", "M1_short_ll", "M2_long_ll", "M3_dual_ll", "dLong_ll",
            "coef_mem3_std", "coef_mem30_std", "M1_short_mse", "M3_dual_mse", "dLong_mse"]
    print(agg[cols].round(4).to_string())
    outp = os.path.join(ROOT, "outputs", "standards", "realband_dual_window.csv")
    df.to_csv(outp, index=False)
    print("\nsaved", outp)

if __name__ == "__main__":
    main()
