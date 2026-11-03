# -*- coding: utf-8 -*-
"""Fit memory-kernel models to REAL DORN longitudinal data.
3-point thermal preference (Warmer=-1 cold / No Change=0 / Cooler=+1 hot),
per-subject dense indoor temperature on a shared 0.1-day grid.
Tests: (a) fast tau_fast is estimable; (b) dual tau_slow is NOT identified on DORN
(should match the synthetic critical-surface prediction)."""
import os, sys, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"D:\thermal-adaptation-series\p1\src")
from p1mk.fit import fit_model, _vote_indices

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = r"D:\thermal-adaptation-series\p1\data\raw\dorn\dorn-longitudinal-tc-study-main\data\df_cozie_env.csv"
OUT = r"D:\thermal-adaptation-series\p1\outputs\eda"
DT = 0.1  # 2.4 h grid, resolves intra-day + day-scale dynamics

df = pd.read_csv(P if False else DATA)
df["dt"] = pd.to_datetime(df["time"], utc=True, format="mixed")
# preference vote: want warmer (too cold) = -1, no change = 0, want cooler (too hot) = +1
vmap = {"Warmer": -1, "No Change": 0, "Cooler": 1}
df = df[df["thermal"].isin(vmap)].copy()
df["y"] = df["thermal"].map(vmap)
# exposure: matched env first, fall back across indoor sensors
tcols = ["t-env", "t-net", "t-ubi", "t-nb"]
for c in tcols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["tdrive"] = df[tcols].bfill(axis=1).iloc[:, 0]
df = df.dropna(subset=["tdrive"]).copy()

uids = sorted(df["userid"].unique())
uid2i = {u: i for i, u in enumerate(uids)}
df["s"] = df["userid"].map(uid2i)
t0 = df["dt"].min().floor("D")
df["tday"] = (df["dt"] - t0).dt.total_seconds() / 86400.0
Tmax = float(df["tday"].max()) + 1.0
t_fine = np.arange(0.0, Tmax, DT)
N, G = len(uids), len(t_fine)
Tamb = np.zeros((N, G), dtype=np.float32)
print(f"users={N} votes={len(df)} Tmax={Tmax:.1f}d grid G={G} dt={DT}")
print("vote dist:", df["y"].value_counts().sort_index().to_dict())
for uid in uids:
    i = uid2i[uid]; g = df[df["userid"] == uid].sort_values("tday")
    # np.interp holds edge values outside the observed span (no spurious excitation)
    Tamb[i] = np.interp(t_fine, g["tday"].values, g["tdrive"].values).astype(np.float32)

t_grid = torch.tensor(t_fine, dtype=torch.float32, device=DEV)
tamb = torch.tensor(Tamb, device=DEV)
subj = torch.tensor(df["s"].values, dtype=torch.long, device=DEV)
tv = torch.tensor(df["tday"].values, dtype=torch.float32, device=DEV)
s_, g_, w_ = _vote_indices(t_grid, subj, tv)
y = torch.tensor(df["y"].values, dtype=torch.long, device=DEV)
data = dict(t_grid=t_grid, tamb=tamb, subj=s_, g=g_, w=w_, y=y, dt=DT)
tn0 = float(np.nanmean(Tamb))

results = {}
for kind in ("static", "fixed7", "fixed30", "single", "dual"):
    r = fit_model(kind, data, N, epochs=1500, lr=0.03, ncat=3, tn_init=tn0,
                  init_tau_fast=3.0, init_tau_slow=30.0, verbose=False)
    line = f"{kind:8s} NLL/vote={r['nll']:.4f}"
    if kind == "single":
        line += f" | tau_fast={r['tau_pop']:.2f} d"
    if kind == "dual":
        line += (f" | tau_fast={r['tau_fast_pop']:.2f} d  tau_slow={r['tau_slow_pop']:.2f} d"
                 f" | gains s/m/f=({r['b_slow']:.2f},{r['b_mid']:.2f},{r['b_fast']:.2f})")
    print(line, flush=True)
    results[kind] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in r.items() if k not in ("model",)}

# null (intercept-only) NLL for reference
p = df["y"].value_counts(normalize=True).sort_index().values
null_nll = float(-np.log([p[int(yy) + 1] for yy in df["y"]]).mean())
print(f"null(intercept-only) NLL/vote={null_nll:.4f}")
results["null_nll"] = null_nll
with open(os.path.join(OUT, "dorn_fit_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=float)
print("saved dorn_fit_results.json")
