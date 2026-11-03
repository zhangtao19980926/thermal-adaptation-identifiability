# -*- coding: utf-8 -*-
"""Fast-process mechanism anchor: estimate the near-body air-temp -> skin-temp
transfer time constant from paired 3-min wearable iButton traces (DORN).
First-order thermal response:  dTskin/dt = (Tskin_eq(Tair) - Tskin)/tau
discretised:  dTskin = b1*Tskin_prev + b2*Tair + b0 ;  tau = -dt/ln(1+b1)."""
import os, sys, types, zipfile, pickle
import numpy as np, pandas as pd

DATA = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main\data"
OUT = r"D:\thermal-adaptation-series\p1\outputs\eda"
def shim():
    sh = types.ModuleType("pandas.core.indexes.numeric")
    class _I(pd.Index): pass
    for nm in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
        setattr(sh, nm, getattr(pd, nm, _I))
    sys.modules["pandas.core.indexes.numeric"] = sh
shim()
def load(name):
    with zipfile.ZipFile(os.path.join(DATA, name)) as z:
        t = [n for n in z.namelist() if not n.endswith("/")][0]
        with z.open(t) as f: return pickle.load(f)

skin = load("df_ib_skin.pkl.zip")[["Temperature", "userid"]].rename(columns={"Temperature": "Tskin"})
air = load("df_ib_air.pkl.zip")[["Temperature", "userid"]].rename(columns={"Temperature": "Tair"})
skin.index = pd.to_datetime(skin.index); air.index = pd.to_datetime(air.index)
def rs3(df, val):
    parts = []
    for u, g in df.groupby("userid"):
        r = g[val].resample("3min").mean().to_frame(val)
        r["userid"] = u
        parts.append(r)
    o = pd.concat(parts).reset_index()
    o = o.rename(columns={o.columns[0]: "time"})
    return o[["time", "userid", val]]
a = rs3(air, "Tair"); s = rs3(skin, "Tskin")
m = a.merge(s, on=["time", "userid"], how="inner").dropna().sort_values(["userid", "time"])
# validity: worn skin 27-38 C, plausible near-body air 15-40 C
m = m[(m.Tskin.between(27, 38)) & (m.Tair.between(15, 40))].copy()
# only genuine consecutive 3-min pairs
m["dtmin"] = m.groupby("userid")["time"].diff().dt.total_seconds() / 60.0
m["dTskin"] = m.groupby("userid")["Tskin"].diff()
m["Tskin_prev"] = m.groupby("userid")["Tskin"].shift(1)
m = m[(m.dtmin >= 2.5) & (m.dtmin <= 3.5)].dropna(subset=["dTskin", "Tskin_prev"])
print("paired 3-min samples:", len(m), "| users:", m.userid.nunique())

rows = []
for u, g in m.groupby("userid"):
    if len(g) < 2000: continue
    X = np.column_stack([np.ones(len(g)), g.Tskin_prev.values, g.Tair.values])
    beta, *_ = np.linalg.lstsq(X, g.dTskin.values, rcond=None)
    b0, b1, b2 = beta
    phi = 1 + b1
    if 0 < phi < 1:
        tau_min = -3.0 / np.log(phi)
        k_gain = -b2 / b1                       # steady-state skin/air gain
        pred = X @ beta
        r2 = 1 - np.sum((g.dTskin.values - pred) ** 2) / np.sum((g.dTskin.values - g.dTskin.mean()) ** 2)
        rows.append((u, len(g), tau_min, k_gain, r2))
r = pd.DataFrame(rows, columns=["user", "n", "tau_min", "ss_gain", "R2"])
print(r.round(3).to_string(index=False))
print("\nskin response tau (minutes): median=%.1f  mean=%.1f  IQR=[%.1f,%.1f]"
      % (r.tau_min.median(), r.tau_min.mean(), r.tau_min.quantile(.25), r.tau_min.quantile(.75)))
print("tau in hours: median=%.2f" % (r.tau_min.median() / 60))
print("steady-state gain dTskin/dTair: median=%.3f" % r.ss_gain.median())
print("fit R2 median=%.3f" % r.R2.median())
r.to_csv(os.path.join(OUT, "dorn_fast_transfer_tau.csv"), index=False)
print("saved dorn_fast_transfer_tau.csv")
