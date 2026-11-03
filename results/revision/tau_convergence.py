# -*- coding: utf-8 -*-
"""Check whether the tau_s recovery bias at short/long truth is a convergence artefact."""
import os, sys, numpy as np, pandas as pd, torch

ROOT = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(ROOT, "src", "p1mk"))
from fit import fit_model, prep_inputs
from ingauge_inject import ingauge_daily, simulate_external

OUT = r"D:\主文件夹\文章返修\_rev"

daily = pd.Series(ingauge_daily()).interpolate().dropna().values
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for true_ts in (7.0, 35.0):
    for epochs in (1800, 3600):
        votes, meta = simulate_external(daily, 100, 120, seed=1, tau_s=true_ts)
        data = prep_inputs(meta, votes, torch.device(dev), dt=0.2)
        r = fit_model("dual", data, 100, epochs=epochs, lr=0.03, seed=0,
                      init_tau_fast=4.0, init_tau_slow=40.0, ncat=7,
                      tn_init=float(np.mean(daily)))
        rows.append(dict(true_ts=true_ts, epochs=epochs,
                         recovered_ts=r["tau_slow_pop"], recovered_tf=r["tau_fast_pop"],
                         nll=r["nll"]))
        print("true %.0f d, epochs %d -> recovered %.1f d (nll %.4f)" %
              (true_ts, epochs, r["tau_slow_pop"], r["nll"]), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tau_convergence_check.csv"), index=False)
print(pd.DataFrame(rows).round(2).to_string(index=False))
