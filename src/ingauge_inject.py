# -*- coding: utf-8 -*-
"""
Result A: inject REAL slow-band outdoor temperature trajectories into the dual-tau DGP.

Controlled comparison: identical true taus (fast=2d, slow=21d), identical fast excitation
(OU + diurnal), identical observation model; ONLY the real slow band changes:
  * In-Gauge (Melbourne, 16 classrooms, 169 d, spring->summer, strong slow band ~8 C)
  * DORN    (Singapore, tropical, ~constant, weekly-monthly band < +-1.4 C)
If tau_slow is recovered (~21 d) and dual beats fixed-window baselines under In-Gauge but
not under DORN, the identifiability critical surface is closed on REAL excitation spectra.
"""
import os, sys, glob, types, zipfile, pickle
import numpy as np, pandas as pd, torch

ROOT = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(ROOT, "src", "p1mk"))
from fit import fit_model, prep_inputs

# ---------- real slow bands (daily mean outdoor temperature) ----------
def ingauge_daily():
    frames = []
    for p in glob.glob(os.path.join(ROOT, "data", "raw", "ingauge", "longitudinal", "*.csv")):
        d = pd.read_csv(p, usecols=["Day", "OutdoorTemperature"])
        frames.append(d.groupby("Day").OutdoorTemperature.mean())
    alld = pd.concat(frames, axis=1)
    med = alld.median(axis=1).sort_index()
    med.index = med.index.astype(float)
    return med.values.astype(float)          # index day 1..169

def _pandas_shim():
    sh = types.ModuleType("pandas.core.indexes.numeric")
    class _I(pd.Index): pass
    for nm in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
        setattr(sh, nm, getattr(pd, nm, _I))
    sys.modules["pandas.core.indexes.numeric"] = sh

def dorn_daily():
    _pandas_shim()
    p = os.path.join(ROOT, "data", "raw", "dorn", "dorn-longitudinal-tc-study-main",
                     "data", "df_weather_sg.pkl.zip")
    with zipfile.ZipFile(p) as z:
        t = [n for n in z.namelist() if not n.endswith("/")][0]
        with z.open(t) as f: df = pickle.load(f)
    tcol = "t-out" if "t-out" in df.columns else [c for c in df.columns if "out" in c.lower()][0]
    s = pd.to_numeric(df[tcol], errors="coerce").dropna()
    s.index = pd.to_datetime(df.index, errors="coerce")
    daily = s.resample("1D").mean().dropna()
    return daily.values.astype(float)

# ---------- DGP with external slow band ----------
def simulate_external(daily, n_subjects, n_votes, seed=0,
                      tau_f=2.0, tau_s=21.0, dt=0.06,
                      ou_sigma=1.0, ou_tau=4.0, diurnal_amp=0.8,
                      k0=1.0, kf=0.45, ks=0.35, sigma_b=1.0, bias_sd=0.25):
    rng = np.random.default_rng(seed)
    days = np.arange(1.0, len(daily) + 1.0)
    T = len(daily)
    t = np.arange(0.0, T + dt, dt)
    slow = np.interp(t, days - 1.0, daily)         # align day1 -> t=0
    # identical fast components for both arms
    ou = np.zeros_like(t)
    for k in range(len(t) - 1):
        d = dt
        e = rng.normal() * ou_sigma * np.sqrt(1.0 - np.exp(-2.0 * d / ou_tau))
        ou[k + 1] = ou[k] * np.exp(-d / ou_tau) + e
    diurnal = diurnal_amp * np.sin(2 * np.pi * t - np.pi / 2.0)
    T_amb = slow + ou + diurnal
    Tn_pop = float(np.mean(daily))
    Tn = rng.normal(Tn_pop, sigma_b, n_subjects)
    tau_fi = np.exp(np.log(tau_f) + rng.normal(0, 0.25, n_subjects))
    tau_si = np.exp(np.log(tau_s) + rng.normal(0, 0.35, n_subjects))
    bias = rng.normal(0.0, bias_sd, n_subjects)
    x = T_amb[None, :] - Tn[:, None]
    af = np.zeros_like(x); asl = np.zeros_like(x)
    for k in range(len(t) - 1):
        d = t[k + 1] - t[k]
        ef = 1.0 - np.exp(-d / tau_fi); es = 1.0 - np.exp(-d / tau_si)
        af[:, k + 1] = af[:, k] + (x[:, k] - af[:, k]) * ef
        asl[:, k + 1] = asl[:, k] + (x[:, k] - asl[:, k]) * es
    cuts = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    rows = []
    for i in range(n_subjects):
        tv = np.sort(rng.uniform(2.0, T, n_votes) + rng.uniform(8 / 24, 20 / 24, n_votes))
        idx = np.clip(np.searchsorted(t, tv), 1, len(t) - 1)
        def ip(a):
            t0, t1 = t[idx - 1], t[idx]; w = (tv - t0) / (t1 - t0)
            return a[i, idx - 1] * (1 - w) + a[i, idx] * w
        xv, fv, sv = ip(x), ip(af), ip(asl)
        z = k0 * xv - kf * fv - ks * sv + bias[i]
        tsv = np.digitize(z + rng.logistic(size=n_votes), cuts) - 3
        rows.append(pd.DataFrame({"subject": i, "t_day": tv, "x_true": xv,
                                  "z_latent": z, "TSV": tsv, "Tn": Tn[i]}))
    votes = pd.concat(rows, ignore_index=True)
    meta = dict(t=t, T_amb=T_amb, Tn=Tn)
    return votes, meta

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev)
    bands = {"InGauge_strong": ingauge_daily(), "DORN_weak": dorn_daily()}
    for nm, daily in bands.items():
        # remove any nan edges
        daily = pd.Series(daily).interpolate().dropna().values
        print("\n==== %s: %d days, daily mean %.2f, 28d-slow swing %.2f C, range %.1f-%.1f" %
              (nm, len(daily), daily.mean(),
               pd.Series(daily).rolling(28, min_periods=14, center=True).mean().max() -
               pd.Series(daily).rolling(28, min_periods=14, center=True).mean().min(),
               daily.min(), daily.max()))
    rows = []
    for nm, daily0 in bands.items():
        daily = pd.Series(daily0).interpolate().dropna().values
        for N, nv in [(40, 150), (100, 120)]:
            votes, meta = simulate_external(daily, N, nv, seed=1)
            data = prep_inputs(meta, votes, torch.device(dev), dt=0.2)
            res = {}
            for kind in ["static", "fixed7", "fixed30", "single", "dual"]:
                r = fit_model(kind, data, N, epochs=600, lr=0.03, seed=0,
                              init_tau_fast=4.0, init_tau_slow=40.0, ncat=7,
                              tn_init=float(np.mean(daily)))
                res[kind] = r
                extra = ""
                if kind == "dual":
                    extra = " tau_f=%.2f tau_s=%.1f bs=%.2f bf=%.2f bi=%.2f" % (
                        r["tau_fast_pop"], r["tau_slow_pop"], r["b_slow"], r["b_mid"], r["b_fast"])
                if kind == "single":
                    extra = " tau=%.2f" % r["tau_pop"]
                print("  %-7s N=%d nll=%.4f%s" % (kind, N, r["nll"], extra))
            base = res["static"]["nll"]
            rows.append(dict(band=nm, N=N,
                             nll_static=res["static"]["nll"], nll_fixed7=res["fixed7"]["nll"],
                             nll_fixed30=res["fixed30"]["nll"], nll_single=res["single"]["nll"],
                             nll_dual=res["dual"]["nll"],
                             dNLL_dual_vs_fixed30=res["fixed30"]["nll"] - res["dual"]["nll"],
                             dNLL_dual_vs_static=base - res["dual"]["nll"],
                             tau_fast=res["dual"]["tau_fast_pop"], tau_slow=res["dual"]["tau_slow_pop"]))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "outputs", "identifiability"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "outputs", "identifiability", "realband_injection.csv"), index=False)
    print("\n", out.round(4).to_string(index=False))
    print("\nsaved realband_injection.csv")

if __name__ == "__main__":
    main()
