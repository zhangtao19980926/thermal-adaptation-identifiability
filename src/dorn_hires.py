# -*- coding: utf-8 -*-
"""Audit high-resolution continuous indoor sensor (Netatmo, ~10-min) for fast-tau."""
import os, sys, types, zipfile, pickle
import pandas as pd, numpy as np

DATA = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main\data"
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

for name in ("df_netatmo.pkl.zip", "df_ib_skin.pkl.zip"):
    print("=" * 70, "\n", name)
    df = load(name)
    print("shape:", df.shape, "| cols:", list(df.columns))
    df.index = pd.to_datetime(df.index)
    tcol = "Temperature"
    rows = []
    for u, g in df.groupby("userid"):
        g = g.sort_index()
        dt = g.index.to_series().diff().dt.total_seconds().median() / 60
        rows.append((u, len(g), round((g.index.max()-g.index.min()).total_seconds()/86400, 1),
                     round(dt, 1), round(g[tcol].mean(), 2), round(g[tcol].std(), 2)))
    per = pd.DataFrame(rows, columns=["user", "n", "span_d", "median_gap_min", "Tmean", "Tsd"])
    print(per.to_string(index=False))
    # intra-day vs day+ variance on the pooled continuous trace (one dense user)
    u = per.sort_values("n", ascending=False).iloc[0]["user"]
    g = df[df.userid == u].sort_index()[tcol].resample("10min").mean().dropna()
    ma1 = g.rolling(144).mean()  # 144*10min = 1 day
    intra = (g - ma1).std(); slow = ma1.std()
    print(f"dense user {u}: n={len(g)}, intra-day SD={intra:.2f}, >1day mean SD={slow:.2f}")
