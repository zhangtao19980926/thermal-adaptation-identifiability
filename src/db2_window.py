# -*- coding: utf-8 -*-
"""
Result C (external validation, ASHRAE DB II, 98 cities, n~104k): DB II has no per-day
dates, only instantaneous outdoor (prev_ta) and monthly mean outdoor. We test whether the
SLOW (monthly) outdoor climate explains thermal sensation beyond the instantaneous outdoor
and the indoor environment, grouped-CV by city, stratified by cooling mode and Koppen band;
and estimate the classic adaptive slope (neutral temperature vs monthly outdoor).
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, mean_squared_error

ROOT = r"D:\thermal-adaptation-series\p1"
OUT = os.path.join(ROOT, "outputs", "standards"); os.makedirs(OUT, exist_ok=True)

def load():
    d = pd.read_csv(r"D:\Research\thermal-comfort-pino\experiments\data\global_db2\db2_tsa_valid.csv",
                    low_memory=False)
    d = d.rename(columns={"TSA": "tsv", "ta": "tin", "top": "top", "clo": "clo", "met": "met",
                          "prev_ta": "out_now", "city": "city", "country": "country",
                          "koppen": "koppen", "cooling": "cooling", "season": "season"})
    d["out_mon"] = (pd.to_numeric(d["Outdoor.monthly.air.temperature...F."], errors="coerce") - 32) * 5 / 9
    for c in ["tsv", "tin", "top", "clo", "met", "out_now", "out_mon"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["tin"] = d["tin"].fillna(d["top"])
    d["kband"] = d["koppen"].astype(str).str[0].map({"A": "A:Tropical", "B": "B:Arid",
                                                      "C": "C:Temperate", "D": "D:Continental"})
    return d

def cv(d, mem):
    d = d.dropna(subset=["tsv", "tin", "clo", "met"] + mem).copy()
    d = d[np.abs(d.tsv) <= 3]
    d["grp"] = pd.factorize(d.city)[0]
    if d.grp.nunique() < 2: return None
    y = d.tsv.values; yneu = (d.tsv == 0).astype(int).values
    feats = ["tin", "clo", "met"] + mem
    X = d[feats].values
    ll, ms, n = [], [], 0
    for tr, te in GroupKFold(min(5, d.grp.nunique())).split(X, y, groups=d.grp.values):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=3000).fit(Xtr, yneu[tr])
        ll.append(log_loss(yneu[te], clf.predict_proba(Xte), labels=[0, 1]) * len(te))
        rg = LinearRegression().fit(Xtr, y[tr])
        ms.append(mean_squared_error(y[te], rg.predict(Xte)) * len(te)); n += len(te)
    return dict(n=n, logloss=sum(ll) / n, mse=sum(ms) / n)

def main():
    d = load()
    print("n=%d cities=%d" % (len(d), d.city.nunique()))
    for c in ["cooling", "kband", "season"]:
        print("\n%s values:\n%s" % (c, d[c].value_counts(dropna=False).head(12)))
    configs = [("base", []), ("+out_now(instant)", ["out_now"]),
               ("+out_mon(monthly)", ["out_mon"]), ("+both", ["out_now", "out_mon"])]
    rows = []
    strata = [("ALL", d)]
    for kb in ["A:Tropical", "B:Arid", "C:Temperate", "D:Continental"]:
        strata.append((kb, d[d.kband == kb]))
    # cooling mode: inspect common tokens
    cool = d.cooling.astype(str).str.lower()
    strata += [("cooled(AC)", d[cool.str.contains("air|ac|mech|centr")]),
               ("uncooled/NV", d[cool.str.contains("natural|none|fan only|naturally")])]
    for name, sub in strata:
        line = {"stratum": name}
        ok = True
        for cname, mem in configs:
            r = cv(sub, mem)
            if r is None: ok = False; break
            line[cname + "_mse"] = r["mse"]; line[cname + "_ll"] = r["logloss"]; line["n"] = r["n"]
        if ok:
            rows.append(line)
            print("\n[%s n=%d]" % (name, line["n"]))
            for cname, _ in configs:
                print("  %-20s mse=%.4f logloss=%.4f" % (cname, line[cname + "_mse"], line[cname + "_ll"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "db2_window.csv"), index=False)

    # classic adaptive slope: per city neutral temperature from TSA~tin, regress on monthly out
    dd = d.dropna(subset=["tsv", "tin", "out_mon", "city"]).copy()
    rec = []
    for city, g in dd.groupby("city"):
        if len(g) < 80 or g.tin.nunique() < 10: continue
        r = LinearRegression().fit(g[["tin"]].values, g.tsv.values)
        Tn = -r.intercept_ / r.coef_[0]          # TSA = a + b*tin -> neutral at TSA=0
        rec.append((city, Tn, g.out_mon.mean(), g.kband.dropna().iloc[0] if g.kband.notna().any() else "?"))
    ad = pd.DataFrame(rec, columns=["city", "Tneutral", "out_mon", "kband"]).dropna()
    sl = LinearRegression().fit(ad[["out_mon"]].values, ad.Tneutral.values)
    print("\nADAPTIVE SLOPE (Tneutral vs monthly outdoor), %d cities: slope=%.3f C/C, R2=%.3f" %
          (len(ad), sl.coef_[0], sl.score(ad[["out_mon"]].values, ad.Tneutral.values)))
    for kb, g in ad.groupby("kband"):
        if len(g) >= 5:
            s = LinearRegression().fit(g[["out_mon"]].values, g.Tneutral.values)
            print("  %-14s n=%2d slope=%.3f R2=%.3f" %
                  (kb, len(g), s.coef_[0], s.score(g[["out_mon"]].values, g.Tneutral.values)))
    ad.to_csv(os.path.join(OUT, "db2_adaptive_slope.csv"), index=False)
    print("\nsaved db2_window.csv / db2_adaptive_slope.csv")

if __name__ == "__main__":
    main()
