# -*- coding: utf-8 -*-
"""
Test whether repeated step events (heat-waves / cold snaps / travel / mode switches)
make tau_slow identifiable where smooth multi-sine excitation failed.

One-dimensional tau_s profile under the 'events' signal, known regime, with independent
test-set evaluation; plus a 2-D profile for the headline design cell.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p1mk.synth import simulate_dataset
from p1mk.fit import prep_inputs
from p1mk.identifiability_map import CellModel, DT, TS_GRID, profile_cell, TS_BOUNDARY

OUT = r"D:\thermal-adaptation-series\p1\outputs\identifiability"


def profile_with_test(N, T, nv, signal, dev, true_ts=21.0, true_tf=2.0, epochs=500,
                      seed=1, test_seed=77):
    r = profile_cell(N, T, nv, signal, "known", true_ts=true_ts, true_tf=true_tf,
                     epochs=epochs, seed=seed, dev=dev)
    # independent test cohort, re-evaluate fitted params per tau_s
    vte, mete = simulate_dataset(1000, nv, float(T), true_tf, true_ts, signal=signal,
                                 seed=test_seed, return_states=True, report_bias_sd=0.0)
    dtest = prep_inputs(mete, vte, dev, dt=DT)
    # refit quickly on train then transfer gains/cuts to test Tn
    vtr, metr = simulate_dataset(N, nv, float(T), true_tf, true_ts, signal=signal,
                                 seed=seed, return_states=True, report_bias_sd=0.0)
    dtr = prep_inputs(metr, vtr, dev, dt=DT)
    test_nll = []
    for ts in TS_GRID:
        torch.manual_seed(0)
        m = CellModel(metr["Tn"].shape[0], dev, free_Tn=False, Tn_true=metr["Tn"])
        opt = torch.optim.Adam(m.parameters(), lr=0.03)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            opt.zero_grad()
            lo = m.nll_at(dtr["tamb"], dtr["subj"], dtr["g"], dtr["w"], dtr["y"], true_tf, float(ts))
            lo.backward(); opt.step(); sch.step()
        mt = CellModel(mete["Tn"].shape[0], dev, free_Tn=False, Tn_true=mete["Tn"])
        mt.load_state_dict({k: v for k, v in m.state_dict().items() if k != "Tn"}, strict=False)
        with torch.no_grad():
            test_nll.append(mt.nll_at(dtest["tamb"], dtest["subj"], dtest["g"], dtest["w"],
                                      dtest["y"], true_tf, float(ts)).item())
    r["test_profile"] = test_nll
    r["test_argmin"] = float(TS_GRID[int(np.argmin(test_nll))])
    i_b = int(np.argmin(np.abs(TS_GRID - TS_BOUNDARY)))
    i_t = int(np.argmin(np.abs(TS_GRID - true_ts)))
    r["test_delta_nll"] = float((test_nll[i_b] - test_nll[i_t]) * len(vte))
    return r


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    rows = []
    for signal in ("events", "multisine"):
        for N in (20, 40):
            for T in (90, 180):
                r = profile_with_test(N, T, 150, signal, dev, epochs=450)
                rows.append(r)
                print(f"{signal:9s} N={N:3d} T={T:3d} | train argmin={r['argmin_ts']:6.1f} "
                      f"dNLL={r['delta_nll_total']:7.1f} | TEST argmin={r['test_argmin']:6.1f} "
                      f"test-dNLL={r['test_delta_nll']:7.1f}", flush=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("profile")} for r in rows]
                 ).to_csv(os.path.join(OUT, "event_check.csv"), index=False)
    with open(os.path.join(OUT, "event_check.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for col, signal in enumerate(("events", "multisine")):
        for row, N in enumerate((20, 40)):
            for k, T in enumerate((90, 180)):
                ax = axes[row, col * 2 + k]
                r = next(x for x in rows if x["signal"] == signal and x["N"] == N and x["T"] == T)
                ax.plot(TS_GRID, r["profile"], "o-", ms=3, label="train")
                ax.plot(TS_GRID, r["test_profile"], "s-", ms=3, label="test(N=1000)")
                ax.axvline(21, color="red", ls="--", lw=1)
                ax.set_xscale("log"); ax.set_title(f"{signal} N={N} T={T}", fontsize=9)
                ax.set_xlabel("tau_s (d)")
                if k == 0:
                    ax.set_ylabel("NLL/vote"); ax.legend(fontsize=7)
    fig.suptitle("Step events (left pair per signal) vs multi-sine: tau_s identifiability\n"
                 "red dashed = true tau_s=21 d")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "event_check.png"), dpi=145)
    print("saved event_check.png", flush=True)


if __name__ == "__main__":
    main()
