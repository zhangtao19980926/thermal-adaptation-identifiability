# -*- coding: utf-8 -*-
"""Sensitivity of the injection experiment to the assumed ground-truth slow constant
(answers R1.2 and R4.C08: 'justify 21 d').

Same real In-Gauge (Melbourne) outdoor series as ingauge_inject.py, same DGP, same
observation model; only true tau_s changes: 7 / 14 / 21 / 28 / 35 d.
Reports the recovered tau_s and dNLL vs fixed-30 for each value and seed.
"""
import os, sys, numpy as np, pandas as pd, torch

ROOT = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(ROOT, "src", "p1mk"))
from fit import fit_model, prep_inputs          # noqa: E402
from ingauge_inject import ingauge_daily, simulate_external  # noqa: E402

OUT = r"D:\主文件夹\文章返修\_rev"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, flush=True)
    daily = pd.Series(ingauge_daily()).interpolate().dropna().values
    rows = []
    for true_ts in (7.0, 14.0, 21.0, 28.0, 35.0):
        for seed in (1, 2):
            votes, meta = simulate_external(daily, 100, 120, seed=seed, tau_s=true_ts)
            data = prep_inputs(meta, votes, torch.device(dev), dt=0.2)
            nll = {}
            for kind in ["fixed30", "dual"]:
                r = fit_model(kind, data, 100, epochs=600, lr=0.03, seed=0,
                              init_tau_fast=4.0, init_tau_slow=40.0, ncat=7,
                              tn_init=float(np.mean(daily)))
                nll[kind] = r["nll"]
                if kind == "dual":
                    rec_f, rec_s = r["tau_fast_pop"], r["tau_slow_pop"]
            rows.append(dict(true_ts=true_ts, seed=seed, recovered_ts=rec_s,
                             recovered_tf=rec_f, nll_dual=nll["dual"],
                             nll_fixed30=nll["fixed30"],
                             dNLL_dual_vs_fixed30=nll["fixed30"] - nll["dual"]))
            print("  true %.0f d seed %d -> recovered %.1f d (dNLL %.4f)" %
                  (true_ts, seed, rec_s, nll["fixed30"] - nll["dual"]), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "tau_sensitivity.csv"), index=False)
    print("\n", out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
