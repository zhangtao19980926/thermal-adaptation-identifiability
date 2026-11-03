# -*- coding: utf-8 -*-
"""Re-run the five-model injection hierarchy (Table 2) at the SI Table S2 schedule
(1800 epochs), reproducing ingauge_inject.main() exactly except for the epoch count,
and recording hardware + per-fit wall time (fills R4-C14)."""
import os, sys, time, json
import numpy as np, pandas as pd, torch

ROOT = r"D:\thermal-adaptation-series\p1"
sys.path.insert(0, os.path.join(ROOT, "src", "p1mk"))
from fit import fit_model, prep_inputs                      # noqa: E402
from ingauge_inject import ingauge_daily, dorn_daily, simulate_external  # noqa: E402

OUT = r"D:\主文件夹\文章返修\_rev"
EPOCHS = 1800


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print("device", dev, "|", gpu, flush=True)

    bands = {"InGauge_strong": ingauge_daily(), "DORN_weak": dorn_daily()}
    rows, timings = [], []
    for nm, daily0 in bands.items():
        daily = pd.Series(daily0).interpolate().dropna().values
        for N, nv in [(40, 150), (100, 120)]:
            votes, meta = simulate_external(daily, N, nv, seed=1)
            data = prep_inputs(meta, votes, torch.device(dev), dt=0.2)
            res = {}
            for kind in ["static", "fixed7", "fixed30", "single", "dual"]:
                t0 = time.time()
                r = fit_model(kind, data, N, epochs=EPOCHS, lr=0.03, seed=0,
                              init_tau_fast=4.0, init_tau_slow=40.0, ncat=7,
                              tn_init=float(np.mean(daily)))
                dt = time.time() - t0
                timings.append(dict(band=nm, N=N, kind=kind, sec=round(dt, 1)))
                res[kind] = r
                print("  %s N=%d %-6s nll=%.4f  (%.0fs)" % (nm, N, kind, r["nll"], dt),
                      flush=True)
            rows.append(dict(band=nm, N=N, epochs=EPOCHS,
                             nll_static=res["static"]["nll"],
                             nll_fixed7=res["fixed7"]["nll"],
                             nll_fixed30=res["fixed30"]["nll"],
                             nll_single=res["single"]["nll"],
                             nll_dual=res["dual"]["nll"],
                             dNLL_dual_vs_fixed30=res["fixed30"]["nll"] - res["dual"]["nll"],
                             dNLL_dual_vs_static=res["static"]["nll"] - res["dual"]["nll"],
                             tau_fast=res["dual"]["tau_fast_pop"],
                             tau_slow=res["dual"]["tau_slow_pop"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "table2_1800ep.csv"), index=False)
    td = pd.DataFrame(timings)
    td.to_csv(os.path.join(OUT, "table2_1800ep_timings.csv"), index=False)
    print("\n", df.round(4).to_string(index=False))
    print("\nper-fit seconds: mean %.0f  median %.0f  total %.0f  (20 fits)" %
          (td.sec.mean(), td.sec.median(), td.sec.sum()))
    print("GPU:", gpu)
    json.dump({"gpu": gpu, "per_fit_min": round(td.sec.median() / 60, 1),
               "total_hours": round(td.sec.sum() / 3600, 2)},
              open(os.path.join(OUT, "compute_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
