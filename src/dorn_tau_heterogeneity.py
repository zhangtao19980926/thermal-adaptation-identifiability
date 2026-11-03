# -*- coding: utf-8 -*-
"""Result D (part 1): between-subject heterogeneity of the fast skin-response tau.
Per-user event tau (DORN) vs BMI, age, sex, cold-hand (peripheral circulation), exercise.
n=20 -> exploratory Spearman associations only, reported with explicit small-sample caveat.
"""
import os, sys, types, zipfile, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DATA = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main\data"
OUT = r"D:\thermal-adaptation-series\p1\outputs\eda"
sh = types.ModuleType("pandas.core.indexes.numeric")
class _I(pd.Index): pass
for nm in ("Int64Index","Float64Index","UInt64Index","NumericIndex"):
    setattr(sh, nm, getattr(pd, nm, _I))
sys.modules["pandas.core.indexes.numeric"] = sh
def load(name):
    with zipfile.ZipFile(os.path.join(DATA, name)) as z:
        t = [n for n in z.namelist() if not n.endswith("/")][0]
        with z.open(t) as f: return pickle.load(f)

info = load("df_info_subjects.pkl.zip")
info["user"] = pd.to_numeric(info["Subject ID"], errors="coerce")
info["BMI"] = pd.to_numeric(info["Weight"], errors="coerce") / \
              (pd.to_numeric(info["Height"], errors="coerce") / 100.0) ** 2
info["Age_n"] = pd.to_numeric(info["Age"], errors="coerce")
info["female"] = (info["Sex"].astype(str).str.lower() == "female").astype(float)
info["coldhand"] = (~info["Cold hand"].astype(str).str.lower().str.contains("never")).astype(float)
ex_map = {"never": 0, "1-3 times a month": 1, "1-2 times a week": 2, "4-6 times a week": 3, "daily": 4}
info["exercise"] = info["Exercise"].astype(str).str.lower().str.strip().map(ex_map)

def yrsg(v):
    s = str(v).lower()
    if "more than 4" in s: return 5.0
    if "less than 1" in s or "<1" in s: return 0.5
    if "feb" in s: return 1.5      # Excel-mangled "1-2"
    if "mar" in s: return 2.5      # "2-3"
    try: return float(v)
    except Exception: return np.nan
info["years_SG"] = info["Years in SG"].map(yrsg)

ev = pd.read_csv(os.path.join(OUT, "dorn_event_tau.csv"))
agg = ev.groupby("user").agg(
    n_ev=("tau_min", "size"), tau=("tau_min", "median"),
    tau_warm=("tau_min", lambda s: s[ev.loc[s.index, "direction"] == "warm"].median()),
    tau_cool=("tau_min", lambda s: s[ev.loc[s.index, "direction"] == "cool"].median()),
    gain=("gain", "median")).reset_index()
m = agg.merge(info[["user", "BMI", "Age_n", "female", "coldhand", "exercise", "years_SG"]], on="user")
m = m.dropna(subset=["tau"])
print("users merged:", len(m))
print(m.round(2).to_string(index=False))

print("\nSpearman correlations with fast tau (n=%d, exploratory):" % len(m))
for cov, lab in [("BMI", "BMI"), ("Age_n", "Age"), ("female", "female"),
                 ("coldhand", "cold-hand(yes)"), ("exercise", "exercise"), ("years_SG", "years in SG")]:
    x = m[cov].values; y = m["tau"].values
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 8:
        rho, p = spearmanr(x[ok], y[ok])
        print("  tau_all  ~ %-14s rho=%+.2f p=%.3f (n=%d)" % (lab, rho, p, ok.sum()))
for tcol, lab in [("tau_warm", "warm"), ("tau_cool", "cool")]:
    for cov, cn in [("BMI", "BMI"), ("coldhand", "cold-hand"), ("female", "female")]:
        d2 = m[[tcol, cov]].dropna()
        if len(d2) >= 8:
            rho, p = spearmanr(d2[cov], d2[tcol])
            print("  tau_%-5s~ %-10s rho=%+.2f p=%.3f (n=%d)" % (lab, cn, rho, p, len(d2)))
m.to_csv(os.path.join(OUT, "dorn_tau_heterogeneity.csv"), index=False)
print("\nsaved dorn_tau_heterogeneity.csv")
