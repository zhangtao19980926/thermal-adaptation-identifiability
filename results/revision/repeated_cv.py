# -*- coding: utf-8 -*-
"""Repeated city-blocked 5-fold CV for the CTCD window scan (answers R1.4: uncertainty).

Each repeat permutes the city labels before GroupKFold, giving an independent set of
city-blocked folds. 10 repeats x 5 folds = 50 paired fold contrasts per window.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, mean_squared_error

P1 = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(P1, "src", "p1mk"))
from ctcd_window import load, mode_group  # noqa: E402

OUT = r"D:\主文件夹\文章返修\_rev"


def main():
    df = load()
    df["mode_g"] = df["mode"].map(mode_group)
    scales = ["m1", "m7", "m15", "m30"]
    d = df.dropna(subset=["tsv", "top", "clo", "met"] + scales).copy()
    d = d[np.abs(d["tsv"]) <= 3]
    y = d["tsv"].values
    y_neu = (d["tsv"] == 0).astype(int).values
    ctrl = ["top", "clo", "met"]
    configs = [("base", [])] + [("+" + s, [s]) for s in scales]
    cities = pd.factorize(d["city"])[0]
    rows = []
    for rep in range(10):
        rng = np.random.default_rng(100 + rep)
        uc = np.unique(cities)
        rng.shuffle(uc)
        assign = {c: i % 5 for i, c in enumerate(uc)}
        fold_of_row = np.array([assign[c] for c in cities])
        folds = [(np.where(fold_of_row != k)[0], np.where(fold_of_row == k)[0]) for k in range(5)]
        for name, mem in configs:
            feats = ctrl + mem
            X = d[feats].values
            ll, ms, n = [], [], 0
            for tr, te in folds:
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y_neu[tr])
                p = clf.predict_proba(Xte)
                ll.append(log_loss(y_neu[te], p, labels=[0, 1]) * len(te))
                reg = LinearRegression().fit(Xtr, y[tr])
                ms.append(mean_squared_error(y[te], reg.predict(Xte)) * len(te))
                n += len(te)
            rows.append(dict(rep=rep, config=name, mse=sum(ms) / n, logloss=sum(ll) / n, n=n))
        print("rep", rep, "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "ctcd_repeated_cv.csv"), index=False)
    piv = r.pivot_table(index="rep", columns="config", values="mse")
    print("\nMSE by repeat:\n", piv.round(4).to_string())
    base = piv["base"]
    print("\npaired over 10 repeats (MSE):")
    for cfg in ["+m1", "+m7", "+m15", "+m30"]:
        diff = piv[cfg] - base
        print("  %-4s mean %+.4f SD %.4f  min %+.4f max %+.4f" %
              (cfg, diff.mean(), diff.std(ddof=1), diff.min(), diff.max()))
    print("\nwindow-to-window (MSE):")
    for a, b in [("+m15", "+m7"), ("+m15", "+m30"), ("+m7", "+m30")]:
        diff = piv[a] - piv[b]
        print("  %s-%s mean %+.5f SD %.5f" % (a, b, diff.mean(), diff.std(ddof=1)))


if __name__ == "__main__":
    main()
