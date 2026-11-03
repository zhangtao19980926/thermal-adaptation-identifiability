# -*- coding: utf-8 -*-
"""Sensitivity of the injection experiment to the assumed ground-truth slow constant,
run at the manuscript's full training schedule (1800 epochs, SI Table S2) rather than the
600 epochs used for the model-hierarchy comparison in Table 2."""
import os, sys, numpy as np, pandas as pd, torch

ROOT = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(ROOT, "src", "p1mk"))
from fit import fit_model, prep_inputs
from ingauge_inject import ingauge_daily, simulate_external

OUT = r"D:\主文件夹\文章返修\_rev"

daily = pd.Series(ingauge_daily()).interpolate().dropna().values
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)
rows = []
for true_ts in (7.0, 14.0, 21.0, 28.0, 35.0):
    votes, meta = simulate_external(daily, 100, 120, seed=1, tau_s=true_ts)
    data = prep_inputs(meta, votes, torch.device(dev), dt=0.2)
    r = fit_model("dual", data, 100, epochs=1800, lr=0.03, seed=0,
                  init_tau_fast=4.0, init_tau_slow=40.0, ncat=7,
                  tn_init=float(np.mean(daily)))
    rows.append(dict(true_ts=true_ts, epochs=1800, recovered_ts=r["tau_slow_pop"],
                     recovered_tf=r["tau_fast_pop"], nll=r["nll"]))
    print("true %.0f d -> recovered %.2f d (tau_f %.2f, nll %.4f)" %
          (true_ts, r["tau_slow_pop"], r["tau_fast_pop"], r["nll"]), flush=True)
df = pd.DataFrame(rows)
df["abs_err_d"] = (df.recovered_ts - df.true_ts).abs()
df.to_csv(os.path.join(OUT, "tau_sensitivity_1800ep.csv"), index=False)
print(df.round(2).to_string(index=False))
