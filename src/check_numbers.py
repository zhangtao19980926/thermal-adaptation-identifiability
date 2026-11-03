# -*- coding: utf-8 -*-
"""Recompute every hard number used in the manuscript / figures from the result files."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = r"D:\thermal-adaptation-series\p1\outputs"
def P(*a): return os.path.join(ROOT, *a)

def line(t): print("\n=== %s ===" % t)

# ---- A. fast tau events ----
ev = pd.read_csv(P("eda", "dorn_event_tau.csv"))
line("A. FAST-tau EVENTS  dorn_event_tau.csv")
print("rows:", len(ev), "| direction values:", ev.direction.value_counts().to_dict())
print("ALL  tau median %.2f  IQR %.2f-%.2f" % (ev.tau_min.median(), ev.tau_min.quantile(.25), ev.tau_min.quantile(.75)))
print("ALL  gain median %.3f   r2 median %.3f" % (ev.gain.median(), ev.r2.median()))
for g in ev.direction.unique():
    d = ev[ev.direction == g]
    print("  %-8s n=%d  tau med %.2f (%.2f-%.2f)  gain med %.3f  r2 med %.3f  ceiling(tau>=299.9)=%d"
          % (g, len(d), d.tau_min.median(), d.tau_min.quantile(.25), d.tau_min.quantile(.75),
             d.gain.median(), d.r2.median(), int((d.tau_min >= 299.9).sum())))
print("total ceiling:", int((ev.tau_min >= 299.9).sum()))

# ---- B. heterogeneity ----
ht = pd.read_csv(P("eda", "dorn_tau_heterogeneity.csv"))
line("B. HETEROGENEITY n=%d" % len(ht))
r, p = spearmanr(ht.Age_n, ht.tau)
print("Age vs tau  Spearman rho=%.3f p=%.4f" % (r, p))
for c in ["female", "BMI", "gain", "coldhand", "exercise", "years_SG"]:
    if c in ht.columns:
        rr, pp = spearmanr(ht[c], ht.tau, nan_policy="omit")
        print("  %-10s rho=%.3f p=%.3f" % (c, rr, pp))

# ---- C. identifiability map ----
im = pd.read_csv(P("identifiability", "ident_map.csv"))
line("C. IDENT MAP  columns=%s" % list(im.columns))
def cell(regime, N, T, col):
    q = im[(im.regime == regime) & (im.N == N) & (im["T"] == T)]
    return q[col].iloc[0]
for regime in im.regime.unique():
    print(regime, "N100T365 dNLL=%.2f argmin=%s | N40T180 dNLL=%.2f argmin=%s | N20T90 dNLL=%.2f argmin=%s"
          % (cell(regime, 100, 365, "delta_nll_total"), cell(regime, 100, 365, "argmin_ts"),
             cell(regime, 40, 180, "delta_nll_total"), cell(regime, 40, 180, "argmin_ts"),
             cell(regime, 20, 90, "delta_nll_total"), cell(regime, 20, 90, "argmin_ts")))

# ---- D. CTCD windows ----
sc = pd.read_csv(P("standards", "ctcd_window_scales.csv"))
line("D. CTCD WINDOWS columns=%s" % list(sc.columns))
for st in sc.stratum.unique():
    d = sc[sc.stratum == st]
    print("  stratum=%s" % st)
    print(d.to_string(index=False))

# ---- E. living years ----
ly = pd.read_csv(P("standards", "ctcd_living_years.csv"))
line("E. LIVING YEARS")
print(ly.to_string(index=False))

# ---- F. DB II window gain ----
db = pd.read_csv(P("standards", "db2_window.csv"))
line("F. DB II WINDOW columns=%s" % list(db.columns))
print(db.to_string(index=False))

# ---- G. counterfactual ----
cf = pd.read_csv(P("standards", "ingauge_counterfactual.csv"))
cf["d7"] = cf.Tcomf7 - cf.Tcomf_tau21
cf["d30"] = cf.Tcomf30 - cf.Tcomf_tau21
tr = (cf.day >= 15) & (cf.day <= 90)
line("G. COUNTERFACTUAL (169 d)")
print("7d  transition(day15-90) mean bias %+.4f  integral %+.3f" % (cf.d7[tr].mean(), cf.d7[tr].sum()))
print("30d transition mean bias %+.4f  integral %+.3f" % (cf.d30[tr].mean(), cf.d30[tr].sum()))
print("cool_frac mean %.4f%%  peak %.4f%%" % (cf.cool_frac.mean()*100, cf.cool_frac.max()*100))
print("outdoor mean %.2f  min %.2f max %.2f" % (cf.Tout.mean(), cf.Tout.min(), cf.Tout.max()))

# ---- H. excitation bands ----
line("H. EXCITATION BANDS")
try:
    sys.path.insert(0, os.path.join(ROOT, "..", "src", "p1mk"))
    from ingauge_inject import ingauge_daily, dorn_daily
    def bands(v):
        s = pd.Series(v); ma7 = s.rolling(7, min_periods=1, center=True).mean()
        ma30 = s.rolling(30, min_periods=1, center=True).mean()
        syn = s-ma7; slow = ma7-ma30; seas = ma30-s.mean()
        return [np.nanstd(syn), np.nanstd(slow), np.nanstd(seas)], float(seas.max()-seas.min())
    dorn, ing = dorn_daily(), ingauge_daily()
    print("DORN days=%d range %.2f  bandSD syn/slow/seas=%s seas_range %.2f"
          % (len(dorn), dorn.max()-dorn.min(), np.round(bands(dorn)[0], 3), bands(dorn)[1]))
    print("InG  days=%d range %.2f  bandSD syn/slow/seas=%s seas_range %.2f"
          % (len(ing), ing.max()-ing.min(), np.round(bands(ing)[0], 3), bands(ing)[1]))
except Exception as e:
    print("bands recompute skipped:", repr(e))
