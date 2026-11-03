# -*- coding: utf-8 -*-
"""Multi-timescale excitation spectrum of DORN indoor/outdoor temperature.
Decompose variance into intra-day / day-week / week-month / seasonal bands to
locate DORN on the identifiability surface (which bands carry slow-memory info)."""
import os, sys, types, zipfile, pickle
import pandas as pd, numpy as np

DATA = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main\data"
OUT = r"D:\thermal-adaptation-series\p1\outputs\eda"

def shim():
    sh = types.ModuleType("pandas.core.indexes.numeric")
    class _I(pd.Index): pass
    for nm in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
        setattr(sh, nm, getattr(pd, nm, _I))
    sys.modules["pandas.core.indexes.numeric"] = sh
shim()

def load_pkl_zip(name):
    with zipfile.ZipFile(os.path.join(DATA, name)) as z:
        target = [n for n in z.namelist() if not n.endswith("/")][0]
        with z.open(target) as f:
            return pickle.load(f)

def band_table(s_hourly, label):
    """hourly series -> variance SD in intra-day / day-week / week-month / month+ bands."""
    x = s_hourly.dropna()
    ma1 = x.rolling(24, min_periods=1).mean()
    ma7 = x.rolling(24 * 7, min_periods=1).mean()
    ma30 = x.rolling(24 * 30, min_periods=1).mean()
    bands = {
        "intra-day (<1d)": x - ma1,
        "day-week (1-7d)": ma1 - ma7,
        "week-month (7-30d)": ma7 - ma30,
        "seasonal (>30d)": ma30 - ma30.mean(),
    }
    print(f"\n[{label}] n_hours={len(x)}, span={(x.index.max()-x.index.min()).days}d, total SD={x.std():.3f}")
    tot = float((x - x.mean()).var())
    for k, v in bands.items():
        print(f"  {k:20s} band_SD={v.std():6.3f}  var_share={100*v.var()/tot:5.1f}%  "
              f"range=[{v.min():+.2f},{v.max():+.2f}]")
    return bands

# outdoor hourly weather
w = load_pkl_zip("df_weather_sg.pkl.zip")
w.index = pd.to_datetime(w.index)
out_b = band_table(w["t-out"].resample("1h").mean(), "OUTDOOR t-out (Singapore)")

# indoor: votes carry t-env at irregular times; resample to hourly mean
env = pd.read_csv(os.path.join(DATA, "df_cozie_env.csv"))
env["dt"] = pd.to_datetime(env["time"], utc=True, format="mixed").dt.tz_convert("Asia/Singapore")
env["t-env"] = pd.to_numeric(env["t-env"], errors="coerce")
env["t-net"] = pd.to_numeric(env["t-net"], errors="coerce")
env["t-skin"] = pd.to_numeric(env["t-skin"], errors="coerce")
inb = env.set_index("dt")
for col, lab in [("t-net", "INDOOR t-net (Netatmo, dense)"),
                 ("t-env", "INDOOR t-env (matched at vote)"),
                 ("t-skin", "SKIN t-skin (wearable)")]:
    s = inb[col].resample("1h").mean()
    if s.notna().sum() > 100:
        band_table(s, lab)

# how often does a >2C / >3C slow (7-30d band) excursion occur? (events for slow tau)
x = w["t-out"].resample("1h").mean().dropna()
ma7 = x.rolling(24*7, min_periods=1).mean(); ma30 = x.rolling(24*30, min_periods=1).mean()
slow = ma7 - ma30
print(f"\nOUTDOOR slow(7-30d) band: |band|>1C hours={int((slow.abs()>1).sum())}, "
      f"|band|>2C hours={int((slow.abs()>2).sum())}, max|band|={slow.abs().max():.2f}")
print("saved band diagnostics to console")
