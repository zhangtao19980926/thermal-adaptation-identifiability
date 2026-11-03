# -*- coding: utf-8 -*-
"""Single persistent step (relocation/season switch): does the clean relaxation
separate tau_fast and tau_slow under ordinal EMA?"""
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
from p1mk.identifiability_map import CellModel, DT, TS_GRID
from p1mk.event_check import profile_with_test

OUT = r"D:\thermal-adaptation-series\p1\outputs\identifiability"


def profile2d(N, T, nv, signal, dev, epochs=450, seed=1):
    votes, meta = simulate_dataset(N, nv, float(T), 2.0, 21.0, signal=signal,
                                   seed=seed, return_states=True, report_bias_sd=0.0)
    data = prep_inputs(meta, votes, dev, dt=DT)
    tf_grid = np.round(np.logspace(np.log10(0.5), np.log10(8.0), 12), 3)
    ts_grid = np.round(np.logspace(np.log10(5.0), np.log10(120.0), 12), 2)
    nll = np.zeros((len(ts_grid), len(tf_grid)))
    for i, ts in enumerate(ts_grid):
        for j, tf in enumerate(tf_grid):
            torch.manual_seed(0)
            m = CellModel(meta["Tn"].shape[0], dev, free_Tn=False, Tn_true=meta["Tn"])
            opt = torch.optim.Adam(m.parameters(), lr=0.03)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            for _ in range(epochs):
                opt.zero_grad()
                lo = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"], data["y"],
                              float(tf), float(ts))
                lo.backward(); opt.step(); sch.step()
            with torch.no_grad():
                nll[i, j] = m.nll_at(data["tamb"], data["subj"], data["g"], data["w"],
                                     data["y"], float(tf), float(ts)).item()
        print(f"  2d row ts={ts:6.1f} best={nll[:i+1].min():.4f}", flush=True)
    return tf_grid, ts_grid, nll


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    rows = []
    for N in (10, 20, 40):
        for T in (60, 90, 180):
            r = profile_with_test(N, T, 150, "step", dev, epochs=450)
            rows.append(r)
            print(f"step N={N:3d} T={T:3d} | train argmin={r['argmin_ts']:6.1f} "
                  f"dNLL={r['delta_nll_total']:8.1f} | TEST argmin={r['test_argmin']:6.1f} "
                  f"test-dNLL={r['test_delta_nll']:9.1f}", flush=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("profile")} for r in rows]
                 ).to_csv(os.path.join(OUT, "step_check.csv"), index=False)

    print("2-D profile (N=20, T=180, step) ...", flush=True)
    tf_g, ts_g, nll = profile2d(20, 180, 150, "step", dev, epochs=450)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for r in rows:
        ax.plot(TS_GRID, r["test_profile"], "s-", ms=3,
                label=f"N={r['N']},T={r['T']} dNLL={r['test_delta_nll']:.0f}")
    ax.axvline(21, color="red", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("tau_s (d)"); ax.set_ylabel("independent-test NLL/vote")
    ax.set_title("Step relaxation: tau_s profile (test cohort N=1000)"); ax.legend(fontsize=7)
    ax = axes[1]
    im = ax.contourf(tf_g, ts_g, nll, levels=25, cmap="viridis_r")
    ax.contour(tf_g, ts_g, nll, levels=12, colors="white", linewidths=0.4, alpha=0.6)
    ax.scatter([2], [21], marker="*", s=260, c="red", edgecolors="white", zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("tau_f (d)"); ax.set_ylabel("tau_s (d)")
    ax.set_title("2-D profile, step N=20 T=180"); plt.colorbar(im, ax=ax, label="NLL/vote")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "step_check.png"), dpi=145)
    print("saved step_check.png", flush=True)


if __name__ == "__main__":
    main()
