# -*- coding: utf-8 -*-
"""Per-fold dispersion of the CTCD running-window comparison (answers R1.4).

Replicates ctcd_window.cv_metrics but keeps fold-level values so we can report the
city-blocked fold-to-fold SD / 95% CI of each window's MSE and log-loss.
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, mean_squared_error

P1 = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(P1, "src", "p1mk"))
from ctcd_window import load, mode_group  # noqa: E402

OUTDIR = r"D:\主文件夹\文章返修\_rev"
os.makedirs(OUTDIR, exist_ok=True)


def fold_metrics(d, scales, stratum):
    d = d.dropna(subset=["tsv", "top", "clo", "met"] + scales).copy()
    d = d[np.abs(d["tsv"]) <= 3]
    d["grp"] = pd.factorize(d["city"])[0]
    ng = d["grp"].nunique()
    if ng < 2:
        return []
    y_neu = (d["tsv"] == 0).astype(int).values
    y = d["tsv"].values
    ctrl = ["top", "clo", "met"]
    configs = [("base", [])] + [("+" + s, [s]) for s in scales]
    nfold = min(5, ng)
    gkf = GroupKFold(n_splits=nfold)
    rows = []
    folds = list(gkf.split(np.zeros(len(d)), y, groups=d["grp"].values))
    for name, mem in configs:
        feats = [c for c in ctrl + mem if c in d.columns]
        X = d[feats].values
        for k, (tr, te) in enumerate(folds):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, y_neu[tr])
            p = clf.predict_proba(Xte)
            ll = log_loss(y_neu[te], p, labels=[0, 1])
            reg = LinearRegression().fit(Xtr, y[tr])
            mse = mean_squared_error(y[te], reg.predict(Xte))
            rows.append(dict(stratum=stratum, config=name, fold=k, n=len(te),
                             logloss=ll, mse=mse))
    return rows


def main():
    df = load()
    df["zone"] = df["zone"].astype(str).str.strip().str.replace("Sever cold", "Severe cold", regex=False)
    df["mode_g"] = df["mode"].map(mode_group)
    scales = ["m1", "m7", "m15", "m30"]
    strata = [("ALL", df)]
    for mg in ["NV", "MM", "AC", "Heated"]:
        sub = df[df.mode_g == mg]
        if len(sub) > 400:
            strata.append((mg, sub))
    rows = []
    for name, sub in strata:
        rows += fold_metrics(sub, scales, name)
    out = pd.DataFrame(rows)
    p = os.path.join(OUTDIR, "ctcd_fold_metrics.csv")
    out.to_csv(p, index=False)

    # weighted mean and fold SD (within-fold spread is the relevant uncertainty)
    agg = []
    for (st, cfg), g in out.groupby(["stratum", "config"]):
        agg.append(dict(stratum=st, config=cfg, n=int(g.n.sum()),
                        mse_mean=np.average(g.mse, weights=g.n),
                        mse_fold_sd=g.mse.std(ddof=1),
                        ll_mean=np.average(g.logloss, weights=g.n),
                        ll_fold_sd=g.logloss.std(ddof=1)))
    a = pd.DataFrame(agg)
    a["mse_sem"] = a.mse_fold_sd / np.sqrt(out.groupby(["stratum", "config"]).size().values)
    a.to_csv(os.path.join(OUTDIR, "ctcd_fold_summary.csv"), index=False)
    print(a.round(4).to_string(index=False))
    print("\nsaved", p)


if __name__ == "__main__":
    main()
