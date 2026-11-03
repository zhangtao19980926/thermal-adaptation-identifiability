# -*- coding: utf-8 -*-
"""
Mechanistic window scan on REAL slow bands. Inject In-Gauge (strong) / DORN (weak) real
daily outdoor trajectories into the dual-tau DGP (true tau_s=21d), then scan the causal
rectangular memory window w and measure held-out-subject predictive logloss/MSE using only
{x_now, running-mean_w}. If the optimum shifts to LONG windows under the strong slow band and
is flat/short under the weak band, the standard's fixed window is shown to be excitation-
dependent (and cross-sectional data that lack slow excitation cannot identify it).
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(ROOT, "src", "p1mk"))
from ingauge_inject import ingauge_daily, dorn_daily, simulate_external

WINDOWS = [1, 2, 3, 5, 7, 10, 15, 21, 30, 45]

def build_features(votes, meta, w):
    t, Tamb, Tn = meta["t"], meta["T_amb"], meta["Tn"]
    tv = votes.t_day.values; subj = votes.subject.values
    idx = np.clip(np.searchsorted(t, tv), 1, len(t) - 1)
    idw = np.searchsorted(t, tv - w); idw = np.clip(idw, 0, idx - 1)
    xnow, mem = np.zeros(len(tv)), np.zeros(len(tv))
    Tamb_c = np.ascontiguousarray(Tamb)
    csum = np.concatenate([[0.0], np.cumsum(Tamb_c)])
    for k in range(len(tv)):
        lo, hi = idw[k], idx[k]
        run = (csum[hi] - csum[lo]) / (hi - lo)
        xnow[k] = Tamb_c[idx[k]] - Tn[subj[k]]
        mem[k] = run - Tn[subj[k]]
    return np.column_stack([xnow, mem])

def heldout(votes, X):
    y = votes.TSV.values; g = votes.subject.values
    ll, ms, n = [], [], 0
    for tr, te in GroupKFold(5).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(Xtr, y[tr])
        ll.append(log_loss(y[te], clf.predict_proba(Xte), labels=list(range(-3, 4)),
                           normalize=False))
        rg = LinearRegression().fit(Xtr, y[tr])
        ms.append(mean_squared_error(y[te], rg.predict(Xte)) * len(te)); n += len(te)
    return sum(ll) / n, sum(ms) / n

def main():
    bands = {"InGauge_strong": pd.Series(ingauge_daily()).interpolate().dropna().values,
             "DORN_weak": pd.Series(dorn_daily()).interpolate().dropna().values}
    rows = []
    for nm, daily in bands.items():
        votes, meta = simulate_external(daily, 100, 120, seed=1)
        t, Tamb, Tn = meta["t"], meta["T_amb"], meta["Tn"]
        tv, subj = votes.t_day.values, votes.subject.values
        idx = np.clip(np.searchsorted(t, tv), 1, len(t) - 1)
        xnow = Tamb[idx] - Tn[subj]
        y = votes.TSV.values; g = votes.subject.values
        # base: instantaneous only
        Xb = xnow[:, None]
        llb, msb = [], []
        ll0, ms0, n = [], [], 0
        for tr, te in GroupKFold(5).split(Xb, y, groups=g):
            sc = StandardScaler().fit(Xb[tr])
            clf = LogisticRegression(max_iter=2000, C=10.0).fit(sc.transform(Xb[tr]), y[tr])
            ll0.append(log_loss(y[te], clf.predict_proba(sc.transform(Xb[te])),
                                labels=list(range(-3, 4)), normalize=False))
            rg = LinearRegression().fit(sc.transform(Xb[tr]), y[tr])
            ms0.append(mean_squared_error(y[te], rg.predict(sc.transform(Xb[te]))) * len(te)); n += len(te)
        rows.append(dict(band=nm, w=0, logloss=sum(ll0)/n, mse=sum(ms0)/n))
        for w in WINDOWS:
            X = build_features(votes, meta, w)
            ll, ms = heldout(votes, X)
            rows.append(dict(band=nm, w=w, logloss=ll, mse=ms))
        r = pd.DataFrame(rows)
        rb = r[r.band == nm]
        print("\n%s" % nm)
        print(rb.round(4).to_string(index=False))
        print("  argmin logloss w=%s (base %.4f -> best %.4f)" %
              (rb.loc[rb.logloss.idxmin(), "w"], rb.logloss.iloc[0], rb.logloss.min()))
    out = pd.DataFrame(rows)
    outp = os.path.join(ROOT, "outputs", "standards", "realband_window_scan.csv")
    out.to_csv(outp, index=False)
    print("\nsaved", outp)

if __name__ == "__main__":
    main()
